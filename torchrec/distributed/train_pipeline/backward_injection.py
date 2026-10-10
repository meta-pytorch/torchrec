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

Three hooking mechanisms are supported, selected via ``InjectionTargetType``:

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

* **FORWARD_MARKER** — splices a custom op into the forward
  input of the module owning the ``hook_position`` parameter; the op's
  backward runs the work function.

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

import itertools
import logging
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from typing import (
    Any,
    Callable,
    cast,
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)

import torch
from pyre_extensions import none_throws
from torch import nn
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
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
        FORWARD_MARKER: Splices a side-effecting custom op into the forward
            input of the module owning the parameter ``hook_position`` selects.
            The op's backward runs ``hook_fn``. Nothing is registered on the
            parameter, so this survives SimpleFSDP's
            ``swap_dtensor_with_tensor``, which installs fresh parameter
            objects every forward and drops hooks registered on the old ones.
    """

    PARAM_GRAD = "param_grad"
    ACTIVATION = "activation"
    FORWARD_MARKER = "forward_marker"


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
            ignored for ``PARAM_GRAD`` and ``FORWARD_MARKER``.
        target_type: Selects the hooking mechanism.  Use ``PARAM_GRAD`` for
            compile-safe parameter-gradient hooks, ``ACTIVATION`` for
            forward-hook + ``tensor_finder`` hooks, ``FORWARD_MARKER`` for a
            marker op spliced into the forward.
        hook_position: Float in [0.0, 1.0] selecting which parameter to hook
            within the target module (``PARAM_GRAD`` and ``FORWARD_MARKER``
            only).  0.0 picks the first parameter (in ``module.parameters()``
            order), 1.0 picks the last.  Ignored for ``ACTIVATION``.
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

    * **FORWARD_MARKER** — splices a custom op into the forward input of the
      module owning the ``hook_position`` parameter; the op's backward runs
      ``hook_fn``.  Use under SimpleFSDP ``swap_dtensor_with_tensor``, which
      drops parameter-registered hooks.  ``tensor_finder`` is ignored.

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
        case InjectionTargetType.FORWARD_MARKER:
            return _register_forward_marker_hook(site, target, hook_fn)
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
    chosen_idx, target_idx = _select_param_index(site, named_params)

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


def _select_param_index(
    site: InjectionSite, named_params: list[tuple[str, nn.Parameter]]
) -> tuple[int, int]:
    """Resolve ``site.hook_position`` to ``(chosen_idx, requested_idx)``.

    The position indexes across *all* named parameters, so the percentage is
    stable regardless of which are hookable; if the parameter at that index
    fails ``will_hook_fire``, walk outward to the nearest neighbour that passes.

    Shared by ``PARAM_GRAD`` and ``FORWARD_MARKER`` so a given position resolves
    to the same parameter in both. ``will_hook_fire`` is ``PARAM_GRAD``'s real
    precondition; ``FORWARD_MARKER`` registers nothing on the parameter and
    reuses the filter only to keep the two modes comparable. Its own
    precondition — the owning module receiving a grad-requiring input — is not
    knowable here, and is checked at the first forward instead.
    """
    n = len(named_params)
    requested_idx = _position_to_index(site.hook_position, n) if n else 0
    chosen_idx = next(
        (
            i
            for i in _walk_outward(requested_idx, n)
            if will_hook_fire(named_params[i][1])
        ),
        None,
    )
    assert chosen_idx is not None, (
        f"register_backward_hook: no parameter under '{site.fqn}' passes "
        f"will_hook_fire ({_summarize_unhookable(named_params)}); "
        "need is_leaf + requires_grad + not ShardedTensor."
    )
    return chosen_idx, requested_idx


class _MarkerOwner(Protocol):
    """Structural type for the marker state attached to a marker's owning module."""

    # Stable for the module's lifetime, so a compiled graph that baked it in
    # still resolves after detach/attach. Never reused, unlike ``id()``.
    _backward_marker_key: int
    # handle id -> callback, in registration order. ``RemovableHandle`` ids are
    # never reused and its ``remove()`` is idempotent, so a stale or repeated
    # ``remove()`` can only drop that handle's own callback.
    _backward_marker_callbacks: "OrderedDict[int, Callable[[torch.Tensor], None]]"


# Weak so an un-removed registration cannot pin the pipeline: the callbacks live
# on the owner, and owner -> callback -> pipeline -> model -> owner is a plain
# collectible cycle, as with ``PARAM_GRAD``'s parameter hooks.
_MARKER_OWNERS: "weakref.WeakValueDictionary[int, nn.Module]" = (
    weakref.WeakValueDictionary()
)
_next_marker_key: Iterator[int] = itertools.count()


@torch.library.custom_op("torchrec_sdd::marker", mutates_args=())
def _marker(x: torch.Tensor, key: int) -> torch.Tensor:
    # A fresh tensor, not ``x``: a custom op may not return an alias of an input
    return x.clone()


@_marker.register_fake
def _marker_fake(x: torch.Tensor, key: int) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("torchrec_sdd::marker_trigger", mutates_args=())
def _marker_trigger(grad: torch.Tensor, key: int) -> None:
    owner = _MARKER_OWNERS.get(key)
    if owner is None:
        return
    for callback in list(cast(_MarkerOwner, owner)._backward_marker_callbacks.values()):
        callback(grad)


@_marker_trigger.register_fake
def _marker_trigger_fake(grad: torch.Tensor, key: int) -> None:
    return None


# The effect token is what keeps it in the backward graph and
# stops it being reordered; consuming ``grad`` is what pins where it runs.
_register_effectful_op(
    torch.ops.torchrec_sdd.marker_trigger.default, _EffectType.ORDERED
)


def _marker_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    ctx.key = inputs[1]


def _marker_autograd(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
    # The trigger has to be an op call: AOTAutograd traces this body, so plain
    # Python here would run at trace time and be absent from the backward graph.
    torch.ops.torchrec_sdd.marker_trigger(grad, ctx.key)
    return grad, None


torch.library.register_autograd(
    "torchrec_sdd::marker", _marker_autograd, setup_context=_marker_setup_context
)


def _register_forward_marker_hook(
    site: InjectionSite,
    target: nn.Module,
    hook_fn: Callable[[torch.Tensor], None],
) -> torch.utils.hooks.RemovableHandle:
    """Run ``hook_fn`` from a marker op spliced into the forward input of the
    module owning the parameter ``hook_position`` selects.

    Parameter selection reuses ``will_hook_fire`` so the chosen index matches
    what ``PARAM_GRAD`` picks for the same position, but nothing is registered
    on the parameter — the marker rides the activation instead. The marker's
    backward therefore runs once the owning module's backward has produced the
    gradient for that input, i.e. about where ``PARAM_GRAD`` would have fired.

    ``hook_fn`` may run more than once per step if the owning module is invoked
    more than once per forward; callers that are not re-entrant must guard.

    Several registrations may resolve to the same owner; they share one marker
    and run in registration order. Each returned handle removes only its own
    callback. The pre-hook stays on the owner once installed and stops
    marking when no callbacks remain.
    """
    named_params = list(target.named_parameters())
    n = len(named_params)
    chosen_idx, target_idx = _select_param_index(site, named_params)

    owner_fqn = named_params[chosen_idx][0].rpartition(".")[0]
    owner = target.get_submodule(owner_fqn) if owner_fqn else target

    logger.info(
        "register_backward_hook: marking forward input of '%s' "
        "(param %d/%d, requested=%d, position=%.2f) in '%s'",
        owner_fqn,
        chosen_idx,
        n,
        target_idx,
        site.hook_position,
        site.fqn,
    )

    if "_backward_marker_key" not in owner.__dict__:
        _install_marker(owner, f"{site.fqn}.{owner_fqn}")
    callbacks = cast(_MarkerOwner, owner)._backward_marker_callbacks
    handle = torch.utils.hooks.RemovableHandle(callbacks)
    callbacks[handle.id] = hook_fn
    return handle


def _install_marker(owner: nn.Module, owner_name: str) -> None:
    """Give ``owner`` a marker key, an empty callback table and the forward
    pre-hook that splices the marker into its input. Runs once per module."""
    key = next(_next_marker_key)
    callbacks: "OrderedDict[int, Callable[[torch.Tensor], None]]" = OrderedDict()
    marker_owner = cast(_MarkerOwner, owner)
    marker_owner._backward_marker_key = key
    marker_owner._backward_marker_callbacks = callbacks
    _MARKER_OWNERS[key] = owner

    # This file lives under ``torchrec/distributed``, which Dynamo skips
    # (``FBCODE_SKIP_TORCHREC_DIRS``). Without this the hook is a skipped
    # callable and every invocation graph-breaks the owning module's forward.
    @torch._dynamo.dont_skip_tracing
    def _pre_hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # Eval runs under no_grad (e.g. apf checkpoint_eval), where nothing
        # requires grad and there is no backward to fire in. With no callbacks
        # left, skip the marker rather than clone for nothing.
        if not torch.is_grad_enabled() or not callbacks:
            return args, kwargs
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                marked = list(args)
                marked[i] = torch.ops.torchrec_sdd.marker(arg, key)
                return tuple(marked), kwargs
        for name, arg in kwargs.items():
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                return args, {**kwargs, name: torch.ops.torchrec_sdd.marker(arg, key)}
        raise RuntimeError(
            "register_backward_hook: no grad-requiring tensor input to "
            f"'{owner_name}'; the marker would never fire in backward."
        )

    owner.register_forward_pre_hook(_pre_hook, with_kwargs=True)


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
