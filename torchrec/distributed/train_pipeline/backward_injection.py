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

An ``InjectionSite`` pairs a module FQN with a ``GradTensorFinder`` strategy
that determines which tensor to attach the backward hook to. Built-in finders:
- ``FirstGradTensorFinder``: finds the first ``requires_grad`` tensor in output/input
- ``OutputDistTensorFinder``: extracts ``dummy_tensor`` from EBC/EC output dist awaitables

Example usage:
    from torchrec.distributed.train_pipeline.backward_injection import (
        InjectionSite,
        OutputDistTensorFinder,
    )
    from torchrec.distributed.types import ShardingType

    # Register hooks on the pipeline
    pipeline.register_backward_hook(
        InjectionSite(
            fqn="sparse_arch.ebc",
            tensor_finder=OutputDistTensorFinder(sharding_type=ShardingType.TABLE_WISE),
        ),
        lambda p: p._optimizer.step(),
    )
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING

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


@runtime_checkable
class GradTensorFinder(Protocol):
    """
    Strategy for locating the tensor to attach a backward hook to.

    Receives the module's forward input and output, and returns the tensor
    on which to register the backward hook. Return ``None`` if no suitable
    tensor is found.
    """

    def __call__(
        self, module_input: Any, module_output: Any
    ) -> Optional[torch.Tensor]: ...


@dataclass(frozen=True)
class FirstGradTensorFinder:
    """
    Finds the first tensor with ``requires_grad=True`` from a module's
    forward output (or input if ``use_input=True``).

    Handles single tensors, tuples/lists, dicts, and nested combinations.
    """

    use_input: bool = False

    def _search(self, data: Any) -> Optional[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            if data.requires_grad:
                return data
        elif isinstance(data, (tuple, list)):
            for item in data:
                t = self._search(item)
                if t is not None:
                    return t
        elif isinstance(data, dict):
            for v in data.values():
                t = self._search(v)
                if t is not None:
                    return t
        return None

    def __call__(self, module_input: Any, module_output: Any) -> Optional[torch.Tensor]:
        data = module_input if self.use_input else module_output
        return self._search(data)


@dataclass(frozen=True)
class InjectionSite:
    """
    Backward hook injection site = module FQN + tensor finding strategy.

    Attributes:
        fqn: Fully qualified name of the target module (e.g., "sparse_arch.ebc")
        tensor_finder: Strategy for locating the tensor to attach the backward
            hook to. Must conform to the ``GradTensorFinder`` protocol.
    """

    fqn: str
    tensor_finder: GradTensorFinder


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
    try:
        target = model.get_submodule(site.fqn)
    except AttributeError:
        raise ValueError(
            f"register_backward_hook: module '{site.fqn}' not found in model."
        )

    def _fwd_hook(
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        tensor = site.tensor_finder(input, output)
        if tensor is None:
            raise RuntimeError(
                f"register_hook: no grad-requiring tensor in "
                f"output of '{site.fqn}'."
            )
        tensor.register_hook(hook_fn)

    return target.register_forward_hook(_fwd_hook)


@dataclass(frozen=True)
class OutputDistTensorFinder:
    """
    Extracts the ``dummy_tensor`` from an EC/EBC output dist awaitable
    matching the given sharding type.

    For pipelined modules, the forward output is an EC/EBC awaitable.
    This finder extracts the per-sharding awaitable matching
    ``self.sharding_type`` and returns its ``dummy_tensor``.

    Attributes:
        sharding_type: The sharding type to target (e.g., ShardingType.TABLE_WISE)
    """

    sharding_type: ShardingType = ShardingType.TABLE_WISE

    def __call__(self, module_input: Any, module_output: Any) -> Optional[torch.Tensor]:
        output = module_output

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

            if ShardingType(st) == self.sharding_type:
                tensor_awaitable = getattr(w, "_tensor_awaitable", None)
                if isinstance(tensor_awaitable, Request):
                    return tensor_awaitable.dummy_tensor
                return None

        raise RuntimeError(
            f"Could not find awaitable for sharding type: {self.sharding_type}"
        )
