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

Two hooking mechanisms are supported, selected via ``InjectionTargetType``:

* **PARAM_GRAD** — uses ``torch.autograd.graph.register_multi_grad_hook`` on the
  target module's trainable parameters.  This is **compile-safe**: unlike
  forward hooks (which ``torch.compile`` can inline away), parameter
  ``AccumulateGrad`` nodes survive compilation, so the hook fires reliably
  under both eager and compiled (FMC) execution.

* **ACTIVATION** — uses a forward hook on the target module.  Each forward pass,
  the hook calls ``site.tensor_finder`` to locate an output tensor (e.g. the
  ``dummy_tensor`` inside an output-dist awaitable), then registers a
  per-tensor backward hook via ``tensor.register_hook``.  This is required
  for sparse/pipelined modules where the backward hook must fire at a
  specific point tied to the output-dist communication tensor.

An ``InjectionSite`` pairs a module FQN with a ``GradTensorFinder`` strategy
and a ``target_type`` that selects the hooking mechanism.

Example usage:
    from torchrec.distributed.train_pipeline.backward_injection import (
        InjectionSite,
        InjectionTargetType,
        FirstGradTensorFinder,
        OutputDistTensorFinder,
    )

    # Dense module — compile-safe parameter-gradient hook
    pipeline.register_backward_hook(
        InjectionSite(
            fqn="dense",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
        ),
        lambda p: ...,
    )

    # Sparse module — forward-hook + tensor_finder
    pipeline.register_backward_hook(
        InjectionSite(
            fqn="sparse_arch.ebc",
            tensor_finder=OutputDistTensorFinder(sharding_type=ShardingType.TABLE_WISE),
            target_type=InjectionTargetType.ACTIVATION,
        ),
        lambda p: p._optimizer.step(),
    )
"""

import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TYPE_CHECKING,
)

import torch
from torch import nn
from torchrec.distributed.comm_ops import Request
from torchrec.distributed.embedding import EmbeddingCollectionAwaitable
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from torchrec.distributed.types import NoWait, ShardingType
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


if TYPE_CHECKING:
    from torchrec.distributed.train_pipeline.train_pipelines import (  # @manual  # pyrefly: ignore[missing-import]
        TrainPipeline,
    )


logger: logging.Logger = logging.getLogger(__name__)


# Type alias for work function that receives pipeline reference
BackwardHookWork = Callable[["TrainPipeline"], None]


@unique
class InjectionTargetType(Enum):
    """Selects the hooking mechanism used by ``register_backward_hook``.

    Attributes:
        PARAM_GRAD: Compile-safe hook via ``register_multi_grad_hook`` on the
            module's trainable parameters.  Suitable for dense sub-modules
            whose parameters participate directly in the loss.
        ACTIVATION: Forward-hook + ``tensor_finder`` approach.  A forward hook
            calls ``site.tensor_finder`` each forward pass to locate the
            output tensor, then registers a per-tensor backward hook.
            Required for sparse / pipelined modules (EC/EBC) where the
            hook must fire at a specific output-dist communication point.
    """

    PARAM_GRAD = "param_grad"
    ACTIVATION = "activation"


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

    Handles single tensors, tuples/lists, dicts, nested combinations, and
    ``(Keyed)JaggedTensor`` whose ``weights`` field carries the gradient
    (e.g. ``PositionWeightedModuleCollection`` outputs a KJT whose weights
    are the trainable position parameters).
    """

    use_input: bool = False

    def _search(self, data: Any) -> Optional[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            if data.requires_grad:
                return data
        elif isinstance(data, (KeyedJaggedTensor, JaggedTensor)):
            # KJT/JT carry their grad-tracking tensor in the optional
            # `weights` field (e.g. PositionWeightedModuleCollection).
            weights = data.weights_or_none()
            if weights is not None and weights.requires_grad:
                return weights
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
    Backward hook injection site = module FQN + tensor finding strategy
    + target type selecting the hooking mechanism.

    Attributes:
        fqn: Fully qualified name of the target module (e.g., "sparse_arch.ebc").
        tensor_finder: Strategy for locating the tensor to attach the backward
            hook to.  Consulted only when ``target_type`` is ``ACTIVATION``;
            ignored for ``PARAM_GRAD``.
        target_type: Selects the hooking mechanism.  Use ``PARAM_GRAD`` for
            compile-safe parameter-gradient hooks, ``ACTIVATION`` for
            forward-hook + ``tensor_finder`` hooks.
    """

    fqn: str
    tensor_finder: GradTensorFinder
    target_type: InjectionTargetType = InjectionTargetType.ACTIVATION


def register_backward_hook(
    site: InjectionSite,
    model: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """
    Registers a backward hook at this injection site.

    The hooking mechanism is selected by ``site.target_type``:

    * **PARAM_GRAD** — ``torch.autograd.graph.register_multi_grad_hook`` on the
      module's trainable parameters.  Compile-safe (``AccumulateGrad``
      nodes survive ``torch.compile``).  ``tensor_finder`` is ignored.

    * **ACTIVATION** — a forward hook that calls ``site.tensor_finder`` each
      forward pass, then registers ``hook_fn`` on the discovered tensor
      via ``tensor.register_hook``.  Required for pipelined EC/EBC modules.

    Args:
        site: Injection site specification.
        model: The model containing the target module.
        hook_fn: Backward hook function (receives a gradient tensor).

    Returns:
        A removable handle; call ``.remove()`` to unregister.

    Raises:
        ValueError: If the target module is not found in the model, has
            no trainable parameters (PARAM_GRAD), or an unknown target type
            is provided.
        RuntimeError: If ``tensor_finder`` returns ``None`` during forward
            (ACTIVATION) or if all parameter gradients are ``None`` during
            backward (PARAM_GRAD).
    """
    try:
        target = model.get_submodule(site.fqn)
    except AttributeError:
        raise ValueError(
            f"register_backward_hook: module '{site.fqn}' not found in model."
        )

    match site.target_type:
        case InjectionTargetType.PARAM_GRAD:
            return _register_param_grad_hook(site, target, hook_fn)
        case InjectionTargetType.ACTIVATION:
            return _register_activation_hook(site, target, hook_fn)
        case _:
            raise ValueError(
                f"register_backward_hook: unknown target_type '{site.target_type}'."
            )


def _register_param_grad_hook(
    site: InjectionSite,
    target: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """Compile-safe hook via ``register_multi_grad_hook`` on parameters."""
    params = [p for p in target.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            f"register_backward_hook: no trainable parameters in module '{site.fqn}'."
        )

    def _multi_grad_callback(
        grads: Sequence[torch.Tensor | None],
    ) -> None:
        """Invoke ``hook_fn`` with the first non-None gradient.

        ``register_multi_grad_hook`` calls this once all tracked
        parameters have accumulated their gradients.  We forward the
        first available gradient tensor to the user-supplied
        ``hook_fn``.

        Raises:
            RuntimeError: If every gradient in *grads* is ``None``.
        """
        grad = next((g for g in grads if g is not None), None)
        if grad is None:
            raise RuntimeError(
                f"register_backward_hook: no non-None gradient found for module '{site.fqn}'."
            )
        hook_fn(grad)

    return torch.autograd.graph.register_multi_grad_hook(params, _multi_grad_callback)


def _register_activation_hook(
    site: InjectionSite,
    target: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """Forward-hook + ``tensor_finder`` approach for sparse/pipelined modules."""

    def _fwd_hook(
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        tensor = site.tensor_finder(input, output)
        if tensor is None:
            raise RuntimeError(
                f"register_backward_hook: no grad-requiring tensor in "
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
