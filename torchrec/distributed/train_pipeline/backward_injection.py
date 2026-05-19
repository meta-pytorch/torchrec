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

* **PARAM_GRAD** — uses ``Tensor.register_post_accumulate_grad_hook`` on a
  single trainable parameter under the target module. This works for both
  plain leaf params (autograd's ``AccumulateGrad`` honors the hook dict) and
  FSDP2 (``fully_shard``) DTensor params (FSDP2's ``foreach_reduce``
  callback honors the same dict after writing the reduce-scattered grad).
  Compile-safe: ``torch.compile`` does not strip these hooks.

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
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)

import torch
from pyre_extensions import none_throws
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
        PARAM_GRAD: Hook via ``Tensor.register_post_accumulate_grad_hook`` on
            a single trainable param under the target module. Works for plain
            leaf params and FSDP2 DTensor params (both honor the
            ``_post_accumulate_grad_hooks`` dict from their respective grad
            writers — AccumulateGrad and FSDP2 ``foreach_reduce``). Compile-safe.
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

    Receives the module's forward positional input, keyword input, and output,
    and returns the tensor on which to register the backward hook. Return
    ``None`` if no suitable tensor is found.
    """

    def __call__(
        self,
        module_input: Any,
        module_kwargs_input: Any,
        module_output: Any,
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

    def __call__(
        self, module_input: Any, module_kwargs_input: Any, module_output: Any
    ) -> Optional[torch.Tensor]:
        data = module_input if self.use_input else module_output
        tensor = self._search(data)
        if tensor is None and self.use_input:
            tensor = self._search(module_kwargs_input)
        return tensor


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
        hook_position: Float in [0.0, 1.0] selecting which parameter to hook
            within the target module (``PARAM_GRAD`` only).  0.0 picks the
            first parameter (in ``module.parameters()`` order), 1.0 picks
            the last.  Ignored for ``ACTIVATION``.
    """

    fqn: str
    tensor_finder: GradTensorFinder
    target_type: InjectionTargetType = InjectionTargetType.ACTIVATION
    hook_position: float = 1.0


def register_backward_hook(
    site: InjectionSite,
    model: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """
    Registers a backward hook at this injection site.

    The hooking mechanism is selected by ``site.target_type``:

    * **PARAM_GRAD** — ``Tensor.register_post_accumulate_grad_hook`` on a
      single parameter under the target module. Works for both plain leaf
      params and FSDP2 DTensor params (the dict is honored by AccumulateGrad
      and FSDP2 ``foreach_reduce`` respectively). Compile-safe.
      ``tensor_finder`` is ignored.

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


def will_hook_fire(p: torch.Tensor) -> bool:
    """A post-accumulate-grad hook fires only if the writer of ``param.grad``
    iterates ``_post_accumulate_grad_hooks`` on the param. For plain leaves
    that writer is autograd's ``AccumulateGrad``; for FSDP2 (``fully_shard``)
    DTensor params it's FSDP2's ``foreach_reduce`` callback. ShardedTensor
    params (sharded embeddings under FBGEMM TBE) are written from a fused
    C++ backward that does not honor the dict, so a hook on them is silently
    dropped — exclude them."""
    return p.is_leaf and p.requires_grad and type(p).__name__ != "ShardedTensor"


def _summarize_unhookable(
    named_params: list[tuple[str, nn.Parameter]],
) -> str:
    """Bucket every ``will_hook_fire``-failing param by the *first* reason it
    fails (priority: no_requires_grad → ShardedTensor → non_leaf).
    Used in the no-hookable-param assert message so callers can see why."""
    no_grad = sharded = non_leaf = 0
    for _, p in named_params:
        t = type(p).__name__
        if not p.requires_grad:
            no_grad += 1
        elif t == "ShardedTensor":
            sharded += 1
        elif not p.is_leaf:
            non_leaf += 1
    return (
        f"total={len(named_params)}, no_requires_grad={no_grad}, "
        f"ShardedTensor={sharded}, non_leaf={non_leaf}"
    )


def _register_param_grad_hook(
    site: InjectionSite,
    target: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """Register a post-accumulate-grad hook on a single parameter selected by
    ``hook_position``.

    Uses ``Tensor.register_post_accumulate_grad_hook`` (not
    ``Tensor.register_hook``) so the hook fires regardless of who writes
    ``param.grad``:

    * For plain leaf params, autograd's ``AccumulateGrad`` node iterates
      ``_post_accumulate_grad_hooks`` after final accumulation.
    * For FSDP2 (``fully_shard``) DTensor params, FSDP2's ``foreach_reduce``
      callback iterates the same dict after writing the reduce-scattered
      sharded grad via Python attribute assignment.

    ``register_hook`` would only fire via ``AccumulateGrad``, which is
    bypassed in the FSDP2 case (the DTensor param is never on the live
    autograd graph; FSDP2 routes grad through a custom backward function
    against a temporary all-gathered tensor) — so this used to silently
    drop, gated off via ``will_hook_fire``.

    The position picks an index across *all* named parameters (so the
    percentage is stable regardless of which params are hookable). If the
    param at that index cannot fire a hook (ShardedTensor / non-leaf /
    no grad), walk outward to the nearest hookable neighbor. Assert if
    no parameter under ``site.fqn`` is hookable at all.

    ``hook_fn`` keeps its ``(grad: Tensor) -> None`` contract: the adapter
    reads ``param.grad`` (post-accumulate fires only after grad is fully
    written, so it's guaranteed non-None) and forwards it.
    """
    named_params = list(target.named_parameters())
    n = len(named_params)

    target_idx = _position_to_index(site.hook_position, n) if n else 0

    chosen_idx = next(
        (i for i in _walk_outward(target_idx, n) if will_hook_fire(named_params[i][1])),
        None,
    )

    assert chosen_idx is not None, (
        f"register_backward_hook: no hookable parameter in module "
        f"'{site.fqn}' ({_summarize_unhookable(named_params)}); "
        f"need is_leaf + requires_grad + not ShardedTensor."
    )

    name, param = named_params[chosen_idx]
    print(
        f"[hook target] requested_idx={target_idx} chosen_idx={chosen_idx}/{n} "
        f"fqn={site.fqn}.{name} type={type(param).__name__} "
        f"is_leaf={param.is_leaf}"
    )
    logger.info(
        "register_backward_hook: hooking param %d/%d "
        "(requested=%d, position=%.2f) in '%s'",
        chosen_idx,
        n,
        target_idx,
        site.hook_position,
        site.fqn,
    )

    def _grad_adapter(p: torch.Tensor) -> None:
        hook_fn(none_throws(p.grad))

    return param.register_post_accumulate_grad_hook(_grad_adapter)


def _walk_outward(start: int, n: int) -> Iterator[int]:
    """Yield indices in ``[0, n)`` ordered by distance from ``start``, forward
    direction first on ties: ``start, start+1, start-1, start+2, start-2, …``.
    Out-of-range indices are skipped, so callers see at most ``n`` values."""
    if 0 <= start < n:
        yield start
    for offset in range(1, n):
        forward = start + offset
        if forward < n:
            yield forward
        backward = start - offset
        if 0 <= backward:
            yield backward


def _position_to_index(position: float, length: int) -> int:
    """Convert a [0.0, 1.0] position to an index in a list of ``length``."""
    clamped = max(0.0, min(1.0, position))
    return min(int(clamped * length), length - 1)


def _register_activation_hook(
    site: InjectionSite,
    target: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """Forward-hook + ``tensor_finder`` approach for sparse/pipelined modules."""

    def _fwd_hook(
        module: nn.Module,
        input: Any,
        kwargs_input: Any,
        output: Any,
    ) -> None:
        tensor = site.tensor_finder(input, kwargs_input, output)
        if tensor is None:
            raise RuntimeError(
                f"register_backward_hook: no grad-requiring tensor in "
                f"output of '{site.fqn}'."
            )
        tensor.register_hook(hook_fn)

    return target.register_forward_hook(_fwd_hook, with_kwargs=True)


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

    def __call__(
        self,
        module_input: Any,
        module_kwargs_input: Any,
        module_output: Any,
    ) -> Optional[torch.Tensor]:
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
            # pyrefly: ignore
            awaitables,
            sharding_types,  # pyrefly: ignore
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
