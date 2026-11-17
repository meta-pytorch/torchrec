#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import math
import os
import threading
from collections import defaultdict
from enum import Enum, unique
from functools import reduce
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from torchrec.optim.keyed import (
    _opt_side_stream,
    _OPT_STREAM_BIG_NUMEL,
    _opt_stream_chain,
    _opt_stream_numel,
    _opt_stream_ops,
    _OPT_STREAM_PLAIN,
    _OPT_STREAMS_CHECK_CALLS,
    _opt_streams_check_mode,
    _opt_streams_checked,
    _opt_streams_log_once,
    _opt_streams_poll,
    _opt_streams_rank,
    _opt_streams_same_all,
    _opt_streams_same_bits,
    _OPT_STREAMS_TLS,
    KeyedOptimizer,
    OptimizerWrapper,
)

logger: logging.Logger = logging.getLogger()

log_grad_norm: bool = False
use_64bit_grad_norm: bool = False

# ONEFLOW_CLIP_FLAT_CHECK compares the first steps' total norms with the
# per-tensor reference within this relative tolerance.
_CLIP_FLAT_CHECK_STEPS = 3
_CLIP_FLAT_NORM_RTOL = 1e-5

# ONEFLOW_CLIP_FLAT_SEGMENTS: a run of grads that are adjacent in a grad
# buffer is normed and clipped as one segment if it has at least this many
# elements; shorter runs stay in the foreach calls.
_CLIP_FLAT_MIN_SEGMENT = 1 << 21

# (base or segment of a base, the grads that tile it) for each flat tensor.
_FlatBases = List[Tuple[torch.Tensor, List[torch.Tensor]]]

# ONEFLOW_OPT_ADAMW_STREAMS cuts a tensor only at multiples of the
# multi_tensor_apply chunk (kChunkSize) and only if its numel is a multiple of
# kILP, so every element takes the same kernel branch as in the unsplit call.
_ADAMW_SPLIT_ALIGN = 65536
_ADAMW_SPLIT_ILP = 4
# Smaller torch._fused_adamw_ calls are not split.
_ADAMW_SPLIT_MIN_NUMEL = 1 << 24
# ONEFLOW_OPT_STREAMS_CHECK also checks the smallest whole tensor of a range
# if it has at most this many elements.
_ADAMW_CHECK_WHOLE_MAX = 1 << 22
_ADAMW_MAX_K = 8
# Process state: first checks left, latched off, plan cache, split calls,
# coverage of the first checks by dtype.
_ADAMW_STREAMS: Dict[str, Any] = {
    "checks": _OPT_STREAMS_CHECK_CALLS,
    "off": False,
    "plans": {},
    "calls": 0,
    "coverage": {},
}
# ONEFLOW_CLIP_FLAT_HALF_COEF process state: checks left, latched off, and
# whether the clipping of the optimizer stepping now met a grad group that the
# flag applies to.
_HALF_COEF: Dict[str, Any] = {
    "checks": _OPT_STREAMS_CHECK_CALLS,
    "off": False,
    "seen": False,
}
# A range of the split: ("run", i, j) passes tensors i..j-1 whole and
# ("cut", i, start, stop) passes view(-1)[start:stop] of tensor i.
_AdamWSegment = Tuple[Any, ...]
_AdamWPlan = List[List[_AdamWSegment]]


@unique
class GradientClipping(Enum):
    NORM = "norm"
    VALUE = "value"
    NONE = "none"


class GradientClippingOptimizer(OptimizerWrapper):
    """
    Clips gradients before doing optimization step.

    Args:
        optimizer (KeyedOptimizer): optimizer to wrap
        clipping (GradientClipping): how to clip gradients
        max_gradient (float): max value for clipping
        norm_type (float or str): type of the used p-norm. Can be ``'inf'`` for infinity norm.
        enable_global_grad_clip (bool): whether to enable global gradient clipping.
        param_to_pgs (Dict[torch.nn.Parameter, List[dist.ProcessGroup]], optional): Mapping of parameters
            to process groups. Used for global gradient clipping in n-D model parallelism case.
            Defaults to None, local gradient clipping is used.
        track_grad_norm (bool): track_grad_norm (bool): If set to True, the optimizer will also track the
            total gradient norm from the most recent step, which can be retrieved
            using the `last_total_grad_norm` property.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        clipping: GradientClipping = GradientClipping.NONE,
        max_gradient: float = 0.1,
        norm_type: Union[float, str] = 2.0,
        enable_global_grad_clip: bool = False,
        param_to_pgs: Optional[
            Dict[torch.nn.Parameter, List[dist.ProcessGroup]]
        ] = None,
        track_grad_norm: bool = False,
    ) -> None:
        super().__init__(optimizer)
        self._clipping = clipping
        self._max_gradient = max_gradient
        self._norm_type = float(norm_type)
        self._check_meta: bool = True
        self._enable_global_grad_clip = enable_global_grad_clip
        self._step_num = 0
        self._track_grad_norm = track_grad_norm
        self._last_total_grad_norm: Optional[torch.Tensor] = None
        self._clip_flat_checks_left: int = _CLIP_FLAT_CHECK_STEPS
        self._clip_flat_logged: Optional[Tuple[Any, ...]] = None

        # Group parameters by model parallelism process group if global clipping is enabled.
        # Otherwise, all parameters are treated as replicated and will be clipped locally.
        sharded_param_cnt = 0
        self._replicate_params: List[torch.Tensor] = []
        self._sharded_params: Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]] = (
            defaultdict(list)
        )

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if not self._enable_global_grad_clip:
                    self._replicate_params.append(param)
                    continue
                if param_to_pgs is None or len(param_to_pgs) == 0:
                    self._replicate_params.append(param)
                    continue

                # Group parameters by model parallelism process group.
                if param in param_to_pgs and len(param_to_pgs[param]) != 0:
                    # pyrefly: ignore[bad-index]
                    self._sharded_params[tuple(param_to_pgs[param])].append(param)
                    sharded_param_cnt += 1
                else:
                    self._replicate_params.append(param)
        logger.info(
            f"Optimizer found {sharded_param_cnt} dist params and {len(self._replicate_params)} replicate params."
        )

        # Sanity check: this path is currently not used in any production.
        if self._clipping == GradientClipping.VALUE:
            if sharded_param_cnt > 0:
                raise NotImplementedError(
                    "clip_grad_value_ for sharded parameters is not supported yet"
                )

    def step(self, closure: Any = None) -> None:
        if self._check_meta:
            # skip gradient clipping and early return
            if any(t.device.type == "meta" for t in self._replicate_params):
                self._last_total_grad_norm = None
                super().step(closure)
                return
            if any(
                t.device.type == "meta"
                for params in self._sharded_params.values()
                for t in params
            ):
                self._last_total_grad_norm = None
                super().step(closure)
                return
            self._check_meta = False

        if self._clipping == GradientClipping.NORM:
            # No sharded parameters, local gradient clipping == global gradient clipping
            if len(self._sharded_params) == 0:
                replicate_params = [
                    p._local_tensor if isinstance(p, DTensor) else p
                    for p in self._replicate_params
                ]
                # Filter out parameters with empty gradients to avoid
                # torch._foreach_norm failing on empty tensors with infinity
                # norm. This matches the filtering in _get_grads().
                replicate_params = [
                    p
                    for p in replicate_params
                    if p.grad is not None and p.grad.numel() > 0
                ]
                if replicate_params:
                    flat_plan = (
                        self._clip_flat_plan(replicate_params)
                        if os.environ.get("ONEFLOW_CLIP_FLAT", "0") == "1"
                        else None
                    )
                    if flat_plan is not None:
                        total_grad_norm = self._clip_flat_(replicate_params, *flat_plan)
                    else:
                        total_grad_norm = _get_total_norm(
                            [p.grad for p in replicate_params if p.grad is not None],
                            norm_type=self._norm_type,
                        )
                        torch.nn.utils.clip_grads_with_norm_(
                            parameters=replicate_params,
                            max_norm=self._max_gradient,
                            total_norm=total_grad_norm,
                        )
                    self._last_total_grad_norm = total_grad_norm
                else:
                    self._last_total_grad_norm = None
            else:
                result = self.clip_grad_norm_()
                self._last_total_grad_norm = (
                    torch.tensor(result) if isinstance(result, float) else result
                )

        elif self._clipping == GradientClipping.VALUE:
            if self._track_grad_norm:
                self._last_total_grad_norm = self._compute_grad_norm_for_logging()
            torch.nn.utils.clip_grad_value_(
                parameters=self._replicate_params, clip_value=self._max_gradient
            )

        elif self._clipping == GradientClipping.NONE:
            if self._track_grad_norm:
                self._last_total_grad_norm = self._compute_grad_norm_for_logging()

        self._step_wrapped(closure)
        self._step_num += 1

    def _step_wrapped(self, closure: Any) -> None:
        """``super().step(closure)``. With ONEFLOW_OPT_ADAMW_STREAMS=1 (read
        per call) the fused AdamW kernels of an optimizer with at least
        ``_OPT_STREAM_BIG_NUMEL`` elements are split across streams
        (``_AdamWStreams``). Reports ONEFLOW_CLIP_FLAT_HALF_COEF no-ops of the
        clipping that just ran."""
        _half_coef_report()
        if os.environ.get("ONEFLOW_OPT_ADAMW_STREAMS", "0") != "1":
            super().step(closure)
            return
        with _AdamWStreams(self):
            super().step(closure)

    @property
    def last_total_grad_norm(self) -> Optional[torch.Tensor]:
        """Returns the total gradient norm from the most recent step, or None
        if no step has run yet."""
        return self._last_total_grad_norm

    def _clip_flat_plan(
        self, replicate_params: List[torch.Tensor]
    ) -> Optional[Tuple[_FlatBases, List[torch.Tensor]]]:
        """ONEFLOW_CLIP_FLAT: returns ``_clip_flat_split`` of the grads of
        ``replicate_params``, or None to use the per-tensor path. Declines are
        logged when the decision changes."""
        if self._norm_type != 2.0:
            reason = f"norm_type {self._norm_type} is not 2"
        else:
            grads = [p.grad for p in replicate_params if p.grad is not None]
            dtype = reduce(torch.promote_types, (g.dtype for g in grads))
            segments = os.environ.get("ONEFLOW_CLIP_FLAT_SEGMENTS", "0") == "1"
            bases, rest = _clip_flat_split(
                grads, _CLIP_FLAT_MIN_SEGMENT if segments else None
            )
            if dtype != torch.float32:
                reason = f"grads promote to {dtype}, not float32"
            elif not bases:
                reason = "no fp32 grad buffer is fully covered by the grads" + (
                    f" and no run of adjacent grads has {_CLIP_FLAT_MIN_SEGMENT} "
                    "elements"
                    if segments
                    else ""
                )
            else:
                reason = ""
            decision = (reason, len(bases), len(rest), segments)
            if decision != self._clip_flat_logged:
                self._clip_flat_logged = decision
                logger.info(
                    f"ONEFLOW_CLIP_FLAT: {len(bases)} flat buffers "
                    f"({sum(b.numel() for b, _ in bases)} elements, "
                    f"{sum(len(m) for _, m in bases)} grads), {len(rest)} other "
                    f"grads ({sum(g.numel() for g in rest)} elements), "
                    f"segments={segments}"
                    f"{'; declined: ' + reason if reason else ''}"
                )
            return None if reason else (bases, rest)
        if self._clip_flat_logged != (reason,):
            self._clip_flat_logged = (reason,)
            logger.info(f"ONEFLOW_CLIP_FLAT: declined: {reason}")
        return None

    def _clip_flat_(
        self,
        replicate_params: List[torch.Tensor],
        flat_bases: _FlatBases,
        rest: List[torch.Tensor],
    ) -> torch.Tensor:
        """Clips like the per-tensor path, with one ``vector_norm`` and one
        ``mul_`` per flat base. The clip coefficient uses the expression of
        ``torch.nn.utils.clip_grads_with_norm_``, so every grad element gets
        the same product as there given the same total norm; the total norm
        differs from the per-tensor one only by summation order.

        ONEFLOW_CLIP_FLAT_CHECK raises on a mismatch rather than falling back:
        the grads are the same on every rank after the DDP all-reduce, so every
        rank reaches the same verdict without a collective, while a rank-local
        fallback would give that rank a different clip coefficient. It clones
        one flat buffer or segment at a time, so its extra memory peaks at the
        largest one (about 1.3 GB), and only during the first
        ``_CLIP_FLAT_CHECK_STEPS`` steps."""
        bases = [b for b, _ in flat_bases]
        total_norm = _get_total_norm(rest, self._norm_type, flat_bases=bases)
        check = (
            self._clip_flat_checks_left > 0
            and os.environ.get("ONEFLOW_CLIP_FLAT_CHECK", "1") == "1"
        )
        if check:
            self._clip_flat_checks_left -= 1
            reference = _get_total_norm(
                [p.grad for p in replicate_params if p.grad is not None],
                norm_type=self._norm_type,
            )
            got, want = float(total_norm), float(reference)
            if not _clip_flat_norm_close(got, want):
                _clip_flat_fail(f"total norm {got!r} vs per-tensor {want!r}")
        clip_coef = float(self._max_gradient) / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for base, members in flat_bases:
            saved = base.clone() if check else None
            base.mul_(clip_coef_clamped.to(base.device))
            if saved is not None:
                offset = base.storage_offset()
                _clip_flat_mul_(
                    [
                        saved[g.storage_offset() - offset :][: g.numel()].view_as(g)
                        for g in members
                    ],
                    clip_coef_clamped,
                )
                if not torch.equal(base.view(torch.int32), saved.view(torch.int32)):
                    _clip_flat_fail(
                        f"the clipped buffer of {base.numel()} elements differs "
                        "from the per-tensor product"
                    )
            del saved
        if rest:
            _clip_flat_mul_(rest, clip_coef_clamped)
        if check and self._clip_flat_checks_left == 0:
            logger.info(
                "ONEFLOW_CLIP_FLAT_CHECK: total norm and clipped grads matched the "
                f"per-tensor path in {_CLIP_FLAT_CHECK_STEPS} steps"
            )
        return total_norm

    def _compute_grad_norm_for_logging(self) -> Optional[torch.Tensor]:
        """Compute the total gradient norm for monitoring only (no clipping).

        Uses ``_get_total_norm`` - the same on-device primitive the
        non-sharded NORM clip path uses - instead of ``compute_total_grad_norm``.
        The latter routes through ``_compute_total_norm``, which does a
        ``non_blocking=True`` GPU->CPU copy of the norm and then reads it
        immediately; that race intermittently yields a stale/near-zero value in
        the logged metric. Keeping the reduction on-device avoids it.

        Falls back to ``compute_total_grad_norm`` when sharded params require a
        cross-rank reduction (VALUE clipping already forbids sharded params).
        """
        if len(self._sharded_params) != 0:
            total_norm, _ = self.compute_total_grad_norm()
            return total_norm
        replicate_params = [
            p._local_tensor if isinstance(p, DTensor) else p
            for p in self._replicate_params
        ]
        grads = [
            p.grad
            for p in replicate_params
            if p.grad is not None and p.grad.numel() > 0
        ]
        if not grads:
            return None
        return _get_total_norm(grads, norm_type=self._norm_type)

    def compute_total_grad_norm(
        self,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Computes the total gradient norm across all parameters without clipping.

        Returns the total norm and the flat list of gradient tensors."""
        sharded_grads = {
            pgs: _get_grads(dist_params)
            for pgs, dist_params in self._sharded_params.items()
        }
        replicate_grads = _get_grads(self._replicate_params)

        all_grads = []
        for grads in sharded_grads.values():
            all_grads.extend(grads)
        all_grads.extend(replicate_grads)

        total_norm = _compute_total_norm(
            replicate_grads=replicate_grads,
            sharded_grads=sharded_grads,
            norm_type=self._norm_type,
            max_grad_norm=self._max_gradient,
        )
        return total_norm, all_grads

    def clip_grad_norm_(self) -> Optional[Union[float, torch.Tensor]]:
        """Clip the gradient norm of all parameters."""

        total_grad_norm, all_grads = self.compute_total_grad_norm()

        # pyrefly: ignore[redundant-cast]
        clip_coef = cast(torch.Tensor, self._max_gradient / (total_grad_norm + 1e-6))
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        torch._foreach_mul_(all_grads, clip_coef_clamped)
        return total_grad_norm


@torch.no_grad()
def _get_total_norm(
    tensors: List[torch.Tensor],
    norm_type: float = 2.0,
    flat_bases: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    """Same result as ``torch.nn.utils.get_total_norm``, but the per-tensor norms
    of each (device, dtype) group are stacked separately and cast to the promoted
    dtype before being concatenated. One ``torch.stack`` over 0-d norms of mixed
    dtypes makes ``cat`` launch one copy per tensor instead of a batched copy.

    ``flat_bases`` (ONEFLOW_CLIP_FLAT) are more tensors to include, each normed
    with one ``torch.linalg.vector_norm``; their norms are stacked first in
    their (device, dtype) group.
    """
    if len(tensors) == 0 and len(flat_bases) == 0:
        return torch.tensor(0.0)
    first_device = (tensors[0] if len(tensors) > 0 else flat_bases[0]).device
    base_norms: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = (
        defaultdict(list)
    )
    for base in flat_bases:
        base_norms[(base.device, base.dtype)].append(
            torch.linalg.vector_norm(base, norm_type)
        )
    grouped_tensors = (
        _group_tensors_by_device_and_dtype(
            [cast(List[Optional[torch.Tensor]], tensors)]
        )
        if len(tensors) > 0
        else {}
    )
    group_norms: List[torch.Tensor] = []
    for (device, dtype), ([device_tensors], _) in grouped_tensors.items():
        group = cast(List[torch.Tensor], device_tensors)
        if _has_foreach_support(group, device):
            norms = torch._foreach_norm(group, norm_type)
        else:
            norms = [torch.linalg.vector_norm(g, norm_type) for g in group]
        extra = base_norms.pop((device, dtype), None)
        if extra:
            norms = extra + list(norms)
        group_norms.append(torch.stack(norms).to(first_device))
    for extra in base_norms.values():
        group_norms.append(torch.stack(extra).to(first_device))
    if len(group_norms) == 1:
        all_norms = group_norms[0]
    else:
        dtype = reduce(torch.promote_types, (n.dtype for n in group_norms))
        all_norms = torch.cat([n.to(dtype) for n in group_norms])
    return torch.linalg.vector_norm(all_norms, norm_type)


def _clip_flat_split(
    grads: List[torch.Tensor],
    min_segment: Optional[int] = None,
) -> Tuple[_FlatBases, List[torch.Tensor]]:
    """ONEFLOW_CLIP_FLAT: returns the flat bases of ``grads`` and the other grads.

    A flat base is a 1-D contiguous fp32 tensor that some contiguous fp32 grads
    are views of (DDP ``gradient_as_bucket_view`` buckets) such that the grads
    tile it exactly: their storage ranges are disjoint and add up to the base.
    The base then holds exactly the elements of those grads. Non-view grads,
    other dtypes and the grads of bases that are not tiled are the other grads.
    With ``min_segment``, bases that are not tiled contribute segments instead
    (see ``_clip_flat_segments``).
    """
    by_base: Dict[int, Tuple[torch.Tensor, List[torch.Tensor]]] = {}
    rest: List[torch.Tensor] = []
    for g in grads:
        base = g._base
        if (
            base is None
            or g.dtype != torch.float32
            or base.dtype != torch.float32
            or base.dim() != 1
            or not base.is_contiguous()
            or not g.is_contiguous()
        ):
            rest.append(g)
        else:
            by_base.setdefault(id(base), (base, []))[1].append(g)
    bases: _FlatBases = []
    for base, members in by_base.values():
        if min_segment is None:
            end = base.storage_offset()
            for start, numel in sorted(
                (g.storage_offset(), g.numel()) for g in members
            ):
                if start != end:
                    rest.extend(members)
                    break
                end += numel
            else:
                if end == base.storage_offset() + base.numel():
                    bases.append((base, members))
                else:
                    rest.extend(members)
        else:
            _clip_flat_segments(base, members, min_segment, bases, rest)
    return bases, rest


def _clip_flat_segments(
    base: torch.Tensor,
    members: List[torch.Tensor],
    min_segment: int,
    bases: _FlatBases,
    rest: List[torch.Tensor],
) -> None:
    """ONEFLOW_CLIP_FLAT_SEGMENTS: appends ``base`` to ``bases`` if
    ``members`` tile it, else each maximal run of adjacent members with at
    least ``min_segment`` elements as a segment ``base.narrow(...)`` that holds
    exactly the run's elements; other members go to ``rest``. If any members
    overlap, all of them go to ``rest``."""
    ordered = sorted(members, key=lambda g: g.storage_offset())
    runs: List[List[torch.Tensor]] = []
    end = -1
    for g in ordered:
        start = g.storage_offset()
        if start < end:
            rest.extend(members)
            return
        if start == end:
            runs[-1].append(g)
        else:
            runs.append([g])
        end = start + g.numel()
    base_start = base.storage_offset()
    if (
        len(runs) == 1
        and ordered[0].storage_offset() == base_start
        and end == base_start + base.numel()
    ):
        bases.append((base, members))
        return
    for run in runs:
        start = run[0].storage_offset()
        numel = sum(g.numel() for g in run)
        if numel >= min_segment:
            bases.append((base.narrow(0, start - base_start, numel), run))
        else:
            rest.extend(run)


def _clip_flat_mul_(tensors: List[torch.Tensor], clip_coef: torch.Tensor) -> None:
    """The multiply of ``torch.nn.utils.clip_grads_with_norm_``."""
    half_coef = os.environ.get("ONEFLOW_CLIP_FLAT_HALF_COEF", "0") == "1"
    grouped = _group_tensors_by_device_and_dtype(
        [cast(List[Optional[torch.Tensor]], tensors)]
    )
    for (device, _), ([device_tensors], _) in grouped.items():
        group = cast(List[torch.Tensor], device_tensors)
        if _has_foreach_support(group, device):
            _clip_flat_foreach_mul_(group, clip_coef, device, half_coef)
        else:
            clip_coef_device = clip_coef.to(device)
            for g in group:
                g.mul_(clip_coef_device)


def _clip_flat_foreach_mul_(
    group: List[torch.Tensor],
    clip_coef: torch.Tensor,
    device: torch.device,
    half_coef: bool,
) -> None:
    """``torch._foreach_mul_(group, clip_coef.to(device))``.

    ONEFLOW_CLIP_FLAT_HALF_COEF=1: for a CUDA group of another floating dtype
    (fp16 grads) the coefficient is first cast to the group's dtype, so that
    ``_foreach_mul_`` takes its fast path instead of falling back to one
    ``mul_`` per grad on the dtype mismatch. Both compute
    ``dtype(float(g) * float(dtype(coef)))``: the slow path's ``mul_`` loads the
    fp32 0-dim CUDA coefficient cast to the grads' dtype (TensorIterator's
    common dtype; BinaryFunctor<Half, Half, Half, MulFunctor<float>>) and the
    fast path multiplies by ``1 * float(*scalar)`` of the cast coefficient.
    The cast and the load use the same float-to-half conversion, so the
    products are bitwise the same. Not on CPU, whose slow path multiplies by
    the fp32 scalar without rounding it."""
    dtype = group[0].dtype
    if dtype in (torch.float16, torch.bfloat16) and any(g.dim() == 0 for g in group):
        torch._foreach_mul_(group, clip_coef.to(device))
        return
    if not half_coef or dtype == clip_coef.dtype:
        torch._foreach_mul_(group, clip_coef.to(device))
        return
    _HALF_COEF["seen"] = True
    reason = _half_coef_decline(device, dtype)
    if reason:
        _opt_streams_log_once(
            f"half_coef:{dtype}:{reason}",
            logging.WARNING,
            lambda: f"ONEFLOW_CLIP_FLAT_HALF_COEF=1 declined for {dtype} grads "
            f"on {device}: {reason}; the clip multiply keeps the per-grad path",
        )
        torch._foreach_mul_(group, clip_coef.to(device))
        return
    _opt_streams_log_once(
        f"half_coef:on:{dtype}",
        logging.INFO,
        lambda: f"ONEFLOW_CLIP_FLAT_HALF_COEF: the clip multiply of {len(group)} "
        f"{dtype} grads on {device} takes the foreach fast path with the "
        f"coefficient cast to {dtype}",
    )
    check = _HALF_COEF["checks"] > 0 and (
        os.environ.get("ONEFLOW_OPT_STREAMS_CHECK", "1") == "1"
    )
    saved = [g.clone() for g in group] if check else []
    torch._foreach_mul_(group, clip_coef.to(device=device, dtype=dtype))
    if check:
        _half_coef_check(group, saved, clip_coef.to(device))


def _half_coef_decline(device: torch.device, dtype: torch.dtype) -> str:
    if _HALF_COEF["off"]:
        return "latched off after a ONEFLOW_OPT_STREAMS_CHECK mismatch"
    if device.type != "cuda":
        return "only the CUDA slow path rounds the coefficient to the grads' dtype"
    if not dtype.is_floating_point:
        return "not a floating dtype"
    return ""


def _half_coef_check(
    group: List[torch.Tensor], saved: List[torch.Tensor], clip_coef: torch.Tensor
) -> None:
    """ONEFLOW_OPT_STREAMS_CHECK: replays the replaced per-grad multiply on
    the pre-multiply clones and compares bitwise; on a mismatch the replay
    result replaces the grads (the grads and the coefficient are the same on
    every rank, so every rank ends with the per-grad result) and the flag is
    latched off. Rank-local, no collectives."""
    torch._foreach_mul_(saved, clip_coef)
    if all(_opt_streams_same_bits(g, s) for g, s in zip(group, saved)):
        _HALF_COEF["checks"] -= 1
        if _HALF_COEF["checks"] == 0:
            logger.info(
                "ONEFLOW_OPT_STREAMS_CHECK: ONEFLOW_CLIP_FLAT_HALF_COEF matched "
                "the per-grad clip multiply bitwise in "
                f"{_OPT_STREAMS_CHECK_CALLS} calls"
            )
        return
    torch._foreach_copy_(group, saved)
    _HALF_COEF["off"] = True
    logger.error(
        f"ONEFLOW_OPT_STREAMS_CHECK: rank {_opt_streams_rank()}: the fast clip "
        "multiply differs from the per-grad one; kept the per-grad result, "
        "ONEFLOW_CLIP_FLAT_HALF_COEF is off for this process"
    )


def _half_coef_report() -> None:
    """ONEFLOW_CLIP_FLAT_HALF_COEF=1 with no grad group to apply to in the
    clipping of the optimizer stepping now: once per process."""
    seen = _HALF_COEF["seen"]
    _HALF_COEF["seen"] = False
    if seen or os.environ.get("ONEFLOW_CLIP_FLAT_HALF_COEF", "0") != "1":
        return
    if os.environ.get("ONEFLOW_CLIP_FLAT", "0") != "1":
        _opt_streams_log_once(
            "half_coef:noflat",
            logging.WARNING,
            lambda: "ONEFLOW_CLIP_FLAT_HALF_COEF=1 has no effect: it applies to "
            "the multiply of ONEFLOW_CLIP_FLAT, which is off",
        )
        return
    _opt_streams_log_once(
        "half_coef:nothing",
        logging.INFO,
        lambda: "ONEFLOW_CLIP_FLAT_HALF_COEF: an optimizer had no non-float32 "
        "grads in the foreach multiply of ONEFLOW_CLIP_FLAT (declined or "
        "float32 only); nothing to cast there",
    )


def _clip_flat_fail(what: str) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    message = f"ONEFLOW_CLIP_FLAT_CHECK: rank {rank}: {what}"
    logger.error(message)
    raise RuntimeError(message)


def _clip_flat_norm_close(got: float, want: float) -> bool:
    if math.isnan(want) or math.isinf(want):
        return got == want or (math.isnan(got) and math.isnan(want))
    return abs(got - want) <= _CLIP_FLAT_NORM_RTOL * abs(want)


def _get_grads(
    param_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Get the gradients of a list of parameters. Converts DTensors to local tensors if needed."""
    grads = [
        p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
        for p in param_list
        if p.grad is not None and p.grad.numel() > 0
    ]
    return grads


def _compute_total_norm(
    replicate_grads: List[torch.Tensor],
    sharded_grads: Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]],
    norm_type: float = 2.0,  # can be a normal float, or torch.inf
    max_grad_norm: float = 1.0,
) -> torch.Tensor:
    """
    Given both replicate grads and sharded grads, compute the total norm of the gradients of the full replicate params and the
    full sharded param (parameters with a process group).

    Args:
        replicate_grads (List[torch.Tensor]): list of gradients for replicate params
        sharded_grads (Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]]): dict that maps each process group to a list of gradients for sharded params
        norm_type (float): type of the used p-norm. Can be torch.inf for infinity norm.
        max_grad_norm (float): max gradient norm.
    """

    ## compute the norm |W|^p corresponding to all sharded params W

    combine_norm_operator = torch.maximum if norm_type == torch.inf else torch.add
    initial_value = float("-inf") if norm_type == torch.inf else 0.0
    sharded_grad_norm: torch.Tensor = torch.tensor(initial_value)

    # We need to move sharded_grad_norm to the same device as the first shard so that we can do addition (or take max)
    # this is specifically for the case where sharded_grad_norm is 0, and replicate_grad_norm is not,
    # because by default torch.tensor(0.0) is on cpu, and replicate_grad_norm is on GPU. For MTIA
    # specifically, adding a tensor on cpu and a tensor on GPU will result in an error.
    for pgs, dist_params in sharded_grads.items():
        current_shard_norm = _batch_cal_norm(
            grad_list=dist_params,
            max_norm=max_grad_norm,
            norm_type=norm_type,
            process_groups=pgs,
        )
        sharded_grad_norm = combine_norm_operator(
            sharded_grad_norm.to(current_shard_norm.device, non_blocking=True),
            current_shard_norm,
        )
    # compute |W|^p corresponding to all replicate params W
    # Similar to the case above, we move replicate_grad_norm to the same device as sharded_grad_norm so that we can do addition.
    replicate_grad_norm: torch.Tensor = (
        _batch_cal_norm(
            grad_list=replicate_grads, max_norm=max_grad_norm, norm_type=norm_type
        )
        if replicate_grads
        else torch.tensor(initial_value)
    ).to(sharded_grad_norm.device, non_blocking=True)

    # In the p-norm case, we are given norms |W_sharded|^p and |W_replicate|^p. To compute the total norm, we need to
    # sum them and take the p-th root. In the inf-norm case, we are given max(|W_sharded|) and max(|W_replicate|).
    # To compute the total norm, we need to take max(max(|W_sharded|), max(|W_replicate|).
    combined_norm = combine_norm_operator(replicate_grad_norm, sharded_grad_norm)
    total_grad_norm = (
        combined_norm.pow(1.0 / norm_type) if norm_type != torch.inf else combined_norm
    )

    return total_grad_norm


def _batch_cal_norm(
    grad_list: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    process_groups: Optional[Tuple[dist.ProcessGroup]] = None,
) -> torch.Tensor:
    """Helper function that calculates the p-th power of the norm of a list of gradients in batches.
    If process_groups are passed in, the norm will be aggregated across all ranks in the process group.
    """
    # Filter out any empty tensors for infinity norm.
    # This is necessary because torch._foreach_norm cannot compute infinity norm on empty tensors.
    if norm_type == torch.inf:
        grad_list = [g for g in grad_list if g.numel() > 0]
        if len(grad_list) == 0:
            return torch.tensor(float("-inf"))

    global use_64bit_grad_norm
    if use_64bit_grad_norm:
        grad_norms = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grad_list, norm_type, dtype=torch.float64)),
            norm_type,
        )
    else:
        grad_norms = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grad_list, norm_type)),
            norm_type,
        )

    if norm_type == torch.inf:
        if process_groups is not None:
            for pg in process_groups:
                dist.all_reduce(grad_norms, op=dist.ReduceOp.MAX, group=pg)
    else:
        grad_norms = grad_norms**norm_type
        if process_groups is not None:
            for pg in process_groups:
                dist.all_reduce(grad_norms, group=pg)

    if use_64bit_grad_norm:
        grad_norms = grad_norms.to(torch.float32)

    return grad_norms


def _dedup_to_base_tensors(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    This is a performance optimization specific to FSDP2. Each gradient tensor
    of the same FSDP module share the same base tensor, so for the total norm
    computation, we can directly use the base tensor to reduce the number of
    tensors to compute norm over.
    """
    seen_base_tensors = set()
    base_tensors = []
    for tensor in tensors:
        base_tensor = tensor._base if tensor._base is not None else tensor
        if base_tensor not in seen_base_tensors:
            seen_base_tensors.add(base_tensor)
            base_tensors.append(base_tensor)
    return base_tensors


_ADAMW_SMALL = "small"
_ADAMW_FIRST = "first"


def _adamw_streams_k() -> Tuple[str, int]:
    raw = os.environ.get("ONEFLOW_OPT_ADAMW_STREAMS_K", "2")
    try:
        k = int(raw)
    except ValueError:
        k = 0
    if not 2 <= k <= _ADAMW_MAX_K:
        return (
            f"ONEFLOW_OPT_ADAMW_STREAMS_K={raw!r} is not an integer in 2.."
            f"{_ADAMW_MAX_K}",
            0,
        )
    return "", k


def _adamw_streams_decline(owner: GradientClippingOptimizer) -> Tuple[str, int]:
    """ONEFLOW_OPT_ADAMW_STREAMS: why ``owner`` keeps the unsplit kernels in
    this step ("" if it does not; ``_ADAMW_SMALL`` below the size,
    ``_ADAMW_FIRST`` in its first step), and K. The first step is the lazy
    state init (RecOptimizer.init_state at construction), so the side
    streams are made, and the checks spent, in training steps."""
    reason, k = _adamw_streams_k()
    if reason:
        return reason, 0
    if torch.compiler.is_compiling():
        return "torch.compile is tracing", 0
    if _ADAMW_STREAMS["off"]:
        return "latched off after a ONEFLOW_OPT_STREAMS_CHECK mismatch", 0
    if isinstance(torch._fused_adamw_, _AdamWStreams):
        return "another optimizer step is splitting it already", 0
    reason, chain = _opt_stream_chain(owner)
    if reason:
        return reason, 0
    if type(chain[-1]) is not torch.optim.AdamW:
        return f"{type(chain[-1]).__qualname__} is not torch.optim.AdamW", 0
    if _opt_stream_numel(chain[-1]) < _OPT_STREAM_BIG_NUMEL:
        return _ADAMW_SMALL, 0
    if owner._step_num == 0:
        return _ADAMW_FIRST, 0
    return "", k


def _adamw_note(reason: str) -> str:
    """The one-time INFO line of an expected ``_adamw_streams_decline``."""
    if reason == _ADAMW_SMALL:
        return (
            "ONEFLOW_OPT_ADAMW_STREAMS: optimizers with fewer than "
            f"{_OPT_STREAM_BIG_NUMEL} elements keep the unsplit fused AdamW"
        )
    return (
        "ONEFLOW_OPT_ADAMW_STREAMS: the first step of an optimizer (its lazy "
        "state init, e.g. RecOptimizer.init_state) keeps the unsplit fused "
        "AdamW; the split starts with its next step"
    )


def _adamw_call_decline(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[str, Optional[torch.device]]:
    """Why a torch._fused_adamw_ call is passed through unsplit (or ""), and
    its device."""
    if len(args) != 6 or not all(isinstance(a, (list, tuple)) for a in args):
        return "unexpected torch._fused_adamw_ arguments", None
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps = args
    if kwargs.get("amsgrad") or len(max_exp_avg_sqs) > 0:
        return "amsgrad", None
    if kwargs.get("grad_scale") is not None or kwargs.get("found_inf") is not None:
        return "AMP grad scaling", None
    n = len(params)
    if n == 0 or any(len(t) != n for t in (grads, exp_avgs, exp_avg_sqs, steps)):
        return "unexpected torch._fused_adamw_ list lengths", None
    if exp_avgs[0].dtype != params[0].dtype:
        return "mixed-precision optimizer states", None
    device = params[0].device
    if not _opt_stream_ops().supports(device):
        return f"{device} is not a supported device", None
    return "", device


def _adamw_cuttable(tensors: Sequence[torch.Tensor], numel: int) -> bool:
    return (
        numel > _ADAMW_SPLIT_ALIGN
        and numel % _ADAMW_SPLIT_ILP == 0
        and all(t.numel() == numel and t.is_contiguous() for t in tensors)
    )


def _adamw_plan_add(
    part: List[_AdamWSegment], i: int, start: int, stop: int, n: int
) -> None:
    if start != 0 or stop != n:
        part.append(("cut", i, start, stop))
    elif part and part[-1][0] == "run" and part[-1][2] == i:
        part[-1] = ("run", part[-1][1], i + 1)
    else:
        part.append(("run", i, i + 1))


def _adamw_plan(
    numels: List[int], cuttable: List[bool], k: int, align: int
) -> _AdamWPlan:
    """Up to ``k`` ranges of the concatenated tensors, balanced by elements.
    A cuttable tensor that does not fit the rest of a range is cut at a
    multiple of ``align`` from its start; other tensors go whole into the
    current range."""
    target = -(-sum(numels) // k)
    parts: _AdamWPlan = [[]]
    filled = 0
    for i, n in enumerate(numels):
        start = 0
        while start < n:
            room = n - start if len(parts) == k else target - filled
            take = n - start
            if take > room and cuttable[i]:
                take = room // align * align
                if take == 0:
                    parts.append([])
                    filled = 0
                    continue
            _adamw_plan_add(parts[-1], i, start, start + take, n)
            start += take
            filled += take
            if filled >= target and len(parts) < k:
                parts.append([])
                filled = 0
    return [part for part in parts if part]


def _adamw_plan_for(args: Tuple[Any, ...], k: int) -> Dict[str, Any]:
    """The cached plan of a torch._fused_adamw_ call, keyed by K and the ids
    of its params and states (stable across steps)."""
    params, _, exp_avgs, exp_avg_sqs, _, _ = args
    key = (
        k,
        tuple(map(id, params)),
        tuple(map(id, exp_avgs)),
        tuple(map(id, exp_avg_sqs)),
    )
    plans = _ADAMW_STREAMS["plans"]
    entry = plans.get(key)
    if entry is None:
        entry = _adamw_build_plan(args, k)
        if len(plans) >= 16:
            plans.clear()
        plans[key] = entry
    return entry


def _adamw_build_plan(args: Tuple[Any, ...], k: int) -> Dict[str, Any]:
    params, _, exp_avgs, exp_avg_sqs, _, _ = args
    if any(
        type(t) not in _OPT_STREAM_PLAIN
        for ts in (params, exp_avgs, exp_avg_sqs)
        for t in ts
    ):
        return {"reason": "a param or state is not a plain tensor"}
    numels = [p.numel() for p in params]
    total = sum(numels)
    if total < _ADAMW_SPLIT_MIN_NUMEL:
        return {"reason": "", "small": f"a call with {total} elements"}
    cuttable = [
        _adamw_cuttable((p, m, v), n)
        for p, m, v, n in zip(params, exp_avgs, exp_avg_sqs, numels)
    ]
    parts = _adamw_plan(numels, cuttable, k, _ADAMW_SPLIT_ALIGN)
    return {
        "reason": "" if len(parts) > 1 else "one range only",
        "parts": parts,
        "numels": numels,
        "total": total,
    }


def _adamw_part_args(
    part: List[_AdamWSegment], args: Tuple[Any, ...], numels: List[int]
) -> Optional[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]]:
    """The four tensor lists and the steps of one range of the plan; None if
    a cut tensor cannot be viewed as planned."""
    params, grads, exp_avgs, exp_avg_sqs, _, steps = args
    sources = (params, grads, exp_avgs, exp_avg_sqs)
    lists: List[List[torch.Tensor]] = [[], [], [], []]
    out_steps: List[torch.Tensor] = []
    for seg in part:
        if seg[0] == "run":
            for out, src in zip(lists, sources):
                out.extend(src[seg[1] : seg[2]])
            out_steps.extend(steps[seg[1] : seg[2]])
            continue
        tensors = [src[seg[1]] for src in sources]
        if not _adamw_cuttable(tensors, numels[seg[1]]):
            return None
        for out, t in zip(lists, tensors):
            out.append(t.view(-1)[seg[2] : seg[3]])
        out_steps.append(steps[seg[1]])
    return lists, out_steps


def _adamw_launch(
    orig: Any,
    device: torch.device,
    parts_args: List[Tuple[List[List[torch.Tensor]], List[torch.Tensor]]],
    kwargs: Dict[str, Any],
) -> List[Any]:
    """Range 0 on the current stream, range j on side stream j - 1; every
    side stream waits for the current stream first and the current stream
    waits for all of them at the end. Returns the streams used."""
    ops = _opt_stream_ops()
    cur = ops.current(device)
    sides = [_opt_side_stream(device, "adamw", j) for j in range(len(parts_args) - 1)]
    for side in sides:
        ops.wait(side, cur)
    try:
        for side, (lists, steps) in zip(sides, parts_args[1:]):
            if steps:
                with ops.use(side):
                    orig(*lists, [], steps, **kwargs)
        lists, steps = parts_args[0]
        if steps:
            orig(*lists, [], steps, **kwargs)
    finally:
        for side in sides:
            ops.wait(cur, side)
    return [cur] + sides


def _adamw_check_pieces(
    entry: Dict[str, Any],
) -> Tuple[List[List[Tuple[int, int, int]]], int, int]:
    """ONEFLOW_OPT_STREAMS_CHECK: the pieces (tensor, start, stop) to check
    in each range of the plan: the ``_ADAMW_SPLIT_ALIGN``-element chunks on
    both sides of every cut boundary, each the first or the last chunk of
    its range's slice (so its elements keep their chunk offsets and kernel
    branch as a tensor of their own), at any tensor size; and the smallest
    non-empty whole tensor of the range if it has at most
    ``_ADAMW_CHECK_WHOLE_MAX`` elements (stop -1). Also returns the numbers
    of cut boundaries and of whole tensors."""
    numels = entry["numels"]
    align = _ADAMW_SPLIT_ALIGN
    pieces: List[List[Tuple[int, int, int]]] = []
    boundaries = whole = 0
    for part in entry["parts"]:
        mine: List[Tuple[int, int, int]] = []
        for seg in part:
            if seg[0] != "cut":
                continue
            _, i, start, stop = seg
            if start > 0:
                boundaries += 1
                mine.append((i, start, min(start + align, stop)))
            if stop < numels[i] and (i, stop - align, stop) not in mine:
                mine.append((i, stop - align, stop))
        runs = [
            i
            for seg in part
            if seg[0] == "run"
            for i in range(seg[1], seg[2])
            if 0 < numels[i] <= _ADAMW_CHECK_WHOLE_MAX
        ]
        if runs:
            whole += 1
            mine.append((min(runs, key=lambda i: numels[i]), 0, -1))
        pieces.append(mine)
    return pieces, boundaries, whole


def _adamw_check(
    orig: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    entry: Dict[str, Any],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], int, int, int]:
    """ONEFLOW_OPT_STREAMS_CHECK: runs the unsplit call on the live tensors
    (so the live result is the stock one on every rank, whatever the
    verdict), then the pieces of ``_adamw_check_pieces`` as tensors of their
    own on clones of their pre-step values, each on the stream of its range
    as in the split, and compares them with the live result bitwise without
    a host sync. Returns the verdict (``_opt_streams_same_all``) and the
    numbers of pieces, cut boundaries and whole tensors checked."""
    params, grads, exp_avgs, exp_avg_sqs, _, steps = args
    pieces, boundaries, whole = _adamw_check_pieces(entry)

    def views(i: int, start: int, stop: int) -> List[torch.Tensor]:
        tensors = [params[i], grads[i], exp_avgs[i], exp_avg_sqs[i]]
        return tensors if stop < 0 else [t.view(-1)[start:stop] for t in tensors]

    with torch.no_grad():
        clones = [
            [(piece, [views(*piece)[j].clone() for j in (0, 2, 3)]) for piece in part]
            for part in pieces
        ]
        orig(*args, **kwargs)
        parts_args = []
        for part in clones:
            lists: List[List[torch.Tensor]] = [[], [], [], []]
            for piece, (p, m, v) in part:
                for out, t in zip(lists, (p, views(*piece)[1], m, v)):
                    out.append(t)
            parts_args.append((lists, [steps[piece[0]] for piece, _ in part]))
        _adamw_launch(orig, device, parts_args, kwargs)
        ok = _opt_streams_same_all(
            [
                (views(*piece)[j], clone)
                for part in clones
                for piece, saved in part
                for j, clone in zip((0, 2, 3), saved)
            ]
        )
    return ok, sum(map(len, pieces)), boundaries, whole


def _adamw_checked(
    orig: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    entry: Dict[str, Any],
    device: torch.device,
    mode: str,
) -> None:
    """A ONEFLOW_OPT_STREAMS_CHECK call (``_adamw_check``) and its verdict
    (``_opt_streams_checked``: a mismatch is fatal). Rank-local, no
    collectives; a "sync" check reads the verdict (host sync), a "deferred"
    one does not."""
    ok, pieces, boundaries, whole = _adamw_check(orig, args, kwargs, entry, device)
    cut = len({seg[1] for part in entry["parts"] for seg in part if seg[0] == "cut"})
    coverage = _ADAMW_STREAMS["coverage"]
    if mode == "sync":
        coverage[str(args[0][0].dtype)] = (
            f"{pieces} pieces: both sides of all {boundaries} cut boundaries in "
            f"{cut} cut tensors, and {whole} whole tensors"
        )
    _opt_streams_checked(
        _ADAMW_STREAMS,
        "ONEFLOW_OPT_ADAMW_STREAMS",
        mode,
        ok,
        f"the split fused AdamW differs from the unsplit call in {pieces} "
        "checked pieces (kept the unsplit result)",
        _opt_stream_ops().current(device),
        lambda: "ONEFLOW_OPT_ADAMW_STREAMS matched the unsplit torch._fused_adamw_ "
        "bitwise (params, exp_avg, exp_avg_sq of the "
        f"{_ADAMW_SPLIT_ALIGN}-element chunks on both sides of every cut "
        "boundary and of the smallest whole tensor of each range; "
        + "; ".join(f"{dtype} calls: {text}" for dtype, text in coverage.items())
        + f") in {_OPT_STREAMS_CHECK_CALLS} calls",
    )


def _adamw_log_first(
    entry: Dict[str, Any],
    k: int,
    device: torch.device,
    streams: List[Any],
    dtype: torch.dtype,
) -> None:
    def message() -> str:
        ops = _opt_stream_ops()
        numels = entry["numels"]
        sizes = [
            sum(
                sum(numels[seg[1] : seg[2]]) if seg[0] == "run" else seg[3] - seg[2]
                for seg in part
            )
            for part in entry["parts"]
        ]
        cut = {seg[1] for part in entry["parts"] for seg in part if seg[0] == "cut"}
        return (
            f"ONEFLOW_OPT_ADAMW_STREAMS: split torch._fused_adamw_ ({len(numels)} "
            f"{dtype} tensors, {entry['total']} elements) with K={k} into ranges "
            f"of {sizes} elements ({len(cut)} tensors cut at {_ADAMW_SPLIT_ALIGN}"
            f"-element boundaries) on {device}: range 0 on the current stream "
            f"{ops.name(streams[0])}, the others on side streams "
            f"{', '.join(ops.name(s) for s in streams[1:])}"
        )

    _opt_streams_log_once(f"adamw:on:{dtype}", logging.INFO, message)


class _AdamWStreams:
    """ONEFLOW_OPT_ADAMW_STREAMS: while ``owner`` steps, ``torch._fused_adamw_``
    is this object (restored on exit). On the training thread it splits each
    call of at least ``_ADAMW_SPLIT_MIN_NUMEL`` elements into K ranges
    (``_adamw_plan``) launched by ``_adamw_launch``: the side streams start
    after everything already on the current stream (clipped grads, the step
    increments of ``_fused_adam``) and the current stream continues only
    after all ranges. Every element gets the same kernel branch, step tensor
    and hyper-parameters as in the unsplit call, so the result is bitwise
    the same. The arguments are long-lived params, grads and states, and the
    join orders every later use and free on the current stream after the
    side streams. On entry it reads the deferred check verdicts that are
    ready and marks the thread as inside a GradientClippingOptimizer step
    (``_opt_streams_reach``)."""

    def __init__(self, owner: GradientClippingOptimizer) -> None:
        self._owner = owner
        self._orig: Any = None
        self._k = 0
        self._thread: int = threading.get_ident()
        self._calls = 0
        self._split = 0
        self._notes: List[str] = []

    def __enter__(self) -> "_AdamWStreams":
        _opt_streams_poll()
        reason, k = _adamw_streams_decline(self._owner)
        if reason in (_ADAMW_SMALL, _ADAMW_FIRST):
            _opt_streams_log_once(
                "adamw:" + reason, logging.INFO, lambda: _adamw_note(reason)
            )
        elif reason:
            _opt_streams_log_once(
                "adamw:" + reason,
                logging.WARNING,
                lambda: f"ONEFLOW_OPT_ADAMW_STREAMS=1 declined: {reason}; the fused "
                "AdamW runs unsplit",
            )
        else:
            self._k = k
            self._orig = torch._fused_adamw_
            torch._fused_adamw_ = self
        _OPT_STREAMS_TLS.clipping = getattr(_OPT_STREAMS_TLS, "clipping", 0) + 1
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _OPT_STREAMS_TLS.clipping -= 1
        if self._orig is None:
            return
        torch._fused_adamw_ = self._orig
        if exc_type is None and self._split == 0:
            notes = "; ".join(self._notes) or "no call"
            _opt_streams_log_once(
                "adamw:unsplit:" + notes,
                logging.WARNING,
                lambda: "ONEFLOW_OPT_ADAMW_STREAMS=1: no torch._fused_adamw_ call was "
                f"split in a step of an optimizer with at least "
                f"{_OPT_STREAM_BIG_NUMEL} elements ({self._calls} calls: "
                f"{notes}); the unsplit kernels ran",
            )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        orig = self._orig
        if threading.get_ident() != self._thread:
            return orig(*args, **kwargs)
        self._calls += 1
        reason, device = _adamw_call_decline(args, kwargs)
        entry = {} if reason else _adamw_plan_for(args, self._k)
        reason = reason or entry["reason"] or entry.get("small", "")
        if reason or device is None:
            self._notes.append(reason)
            return orig(*args, **kwargs)
        parts_args = [
            _adamw_part_args(part, args, entry["numels"]) for part in entry["parts"]
        ]
        if any(a is None for a in parts_args):
            self._notes.append("a cut tensor is no longer contiguous")
            return orig(*args, **kwargs)
        self._split += 1
        _ADAMW_STREAMS["calls"] += 1
        mode = _opt_streams_check_mode(
            _ADAMW_STREAMS["checks"], _ADAMW_STREAMS["calls"]
        )
        if mode:
            _adamw_checked(orig, args, kwargs, entry, device, mode)
            return None
        streams = _adamw_launch(orig, device, cast(List[Any], parts_args), kwargs)
        _adamw_log_first(entry, self._k, device, streams, args[0][0].dtype)
        return None
