#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Backward hook injection utilities for training pipelines.

This module provides utilities for injecting work functions into the backward
pass of EC (EmbeddingCollection) and EBC (EmbeddingBagCollection) modules.
Work functions are registered at specific injection sites and executed during
the backward all-to-all communication phase.

Example usage:
    from torchrec.distributed.train_pipeline.backward_injection import (
        OutputDistSite,
    )
    from torchrec.distributed.types import ShardingType

    # Register hooks on the pipeline
    pipeline.register_backward_hook(
        OutputDistSite(fqn="sparse_arch.ebc", sharding_type=ShardingType.TABLE_WISE),
        lambda p: p._optimizer.step(),
    )
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from torch import nn
from torchrec.distributed.comm_ops import Request
from torchrec.distributed.embedding import EmbeddingCollectionAwaitable
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from torchrec.distributed.types import NoWait, ShardingType


if TYPE_CHECKING:
    from torchrec.distributed.train_pipeline.train_pipelines import (  # @manual  # pyrefly: ignore[missing-import]
        TrainPipeline,
    )


logger: logging.Logger = logging.getLogger(__name__)


# Type alias for work function that receives pipeline reference
BackwardHookWork = Callable[["TrainPipeline"], None]


@dataclass(frozen=True)
class InjectionSite:
    """
    Base class for backward hook injection sites.

    Attributes:
        fqn: Fully qualified name of the target module (e.g., "sparse_arch.ebc")
    """

    fqn: str

    def find_target_module(self, model: nn.Module) -> Optional[nn.Module]:
        """
        Finds the module matching ``self.fqn`` in the model.

        Returns:
            The matching module, or ``None`` if not found.
        """
        try:
            return model.get_submodule(self.fqn)
        except AttributeError:
            return None

    def find_grad_tensor(self, output: Any) -> Optional[torch.Tensor]:
        """
        Finds the first tensor with ``requires_grad=True`` from a module output.

        Handles single tensors, tuples/lists, dicts, and nested combinations.

        Args:
            output: The module's forward output.

        Returns:
            The first grad-requiring tensor, or ``None`` if none found.
        """
        if isinstance(output, torch.Tensor):
            if output.requires_grad:
                return output
        elif isinstance(output, (tuple, list)):
            for item in output:
                t = self.find_grad_tensor(item)
                if t is not None:
                    return t
        elif isinstance(output, dict):
            for v in output.values():
                t = self.find_grad_tensor(v)
                if t is not None:
                    return t
        return None


def register_backward_hook(
    site: InjectionSite,
    model: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """
    Registers a backward hook at this injection site.

    Installs a forward hook on the target module. Each forward pass, the
    forward hook finds the first grad-requiring output tensor and registers
    ``hook_fn`` as a backward hook on it. The forward hook persists across
    iterations; call ``.remove()`` on the returned handle to unregister.

    Args:
        model: The model containing the target module.
        hook_fn: Backward hook function (receives gradient tensor).

    Returns:
        A removable handle for the forward hook.

    Raises:
        ValueError: If the target module is not found in the model.
        RuntimeError: If no grad-requiring tensor is found in the
            module's output during forward.
    """
    target = site.find_target_module(model)
    if target is None:
        raise ValueError(
            f"register_backward_hook: module '{site.fqn}' not found in model."
        )

    def _fwd_hook(
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        tensor = site.find_grad_tensor(output)
        if tensor is None:
            raise RuntimeError(
                f"register_hook: no grad-requiring tensor in "
                f"output of '{site.fqn}'."
            )
        tensor.register_hook(hook_fn)

    return target.register_forward_hook(_fwd_hook)


@dataclass(frozen=True)
class OutputDistSite(InjectionSite):
    """
    Injection site for hooking during backward all-to-all on output dist tensors.

    Targets the ``dummy_tensor`` of the EC/EBC output dist awaitable matching
    the given module FQN and sharding type.

    Attributes:
        fqn: Fully qualified name of the EC/EBC module (e.g., "sparse_arch.ebc")
        sharding_type: The sharding type to target (e.g., ShardingType.TABLE_WISE)
    """

    sharding_type: ShardingType = ShardingType.TABLE_WISE

    def find_grad_tensor(self, output: Any) -> Optional[torch.Tensor]:
        """
        Finds the dummy tensor from the output dist awaitable matching
        this site's sharding type.

        For pipelined modules, the forward output is an EC/EBC awaitable.
        This method extracts the per-sharding awaitable matching
        ``self.sharding_type`` and returns its ``dummy_tensor``.

        Args:
            output: The pipelined module's forward output (an EC/EBC awaitable,
                possibly MC tuple-wrapped).

        Returns:
            The dummy tensor for the matching sharding type, or ``None`` if
            not found.
        """

        # Handle MC EC/EBC tuple wrapping
        if isinstance(output, tuple):
            output = output[0]

        # NOTE: We avoid importing VariableBatchEmbeddingBagCollectionAwaitable
        # directly due to torch.package compatibility issues with repackaging.
        # Instead, we use hasattr to detect EBC-like awaitables (including VB-EBC).
        match output:
            case EmbeddingBagCollectionAwaitable():
                awaitables = output._awaitables
                sharding_types = output._sharding_types
            case EmbeddingCollectionAwaitable():
                awaitables = output._awaitables_per_sharding
                sharding_types = output._sharding_types
            case _ if hasattr(output, "_awaitables") and hasattr(
                output, "_sharding_types"
            ):
                awaitables = output._awaitables
                sharding_types = output._sharding_types
            case _:
                raise RuntimeError(
                    f"Unsupported awaitable type: {type(output).__name__}"
                )

        # Find the awaitable matching our sharding type, skipping DP (NoWait)
        for w, st in zip(  # pyrefly: ignore[no-matching-overload]
            awaitables, sharding_types
        ):
            if isinstance(w, NoWait):
                continue
            # pyrefly: ignore[unsupported-operation]
            if ShardingType(st) == self.sharding_type:
                tensor_awaitable = getattr(w, "_tensor_awaitable", None)
                if isinstance(tensor_awaitable, Request):
                    return tensor_awaitable.dummy_tensor
                return None

        raise RuntimeError(
            f"Could not find awaitable for module {self.fqn} "
            f"with sharding type: {self.sharding_type}"
        )
