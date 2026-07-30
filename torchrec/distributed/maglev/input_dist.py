#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Flatten structured data to a list of tensors, rebuild it, and all-to-all it.

Used to move structured cross-stage "carriers" across HSD boundaries. A carrier
(dataclass of tensors / ``KeyedJaggedTensor`` / nested containers) is flattened to
a flat list of tensors on the sender and rebuilt on the receiver from a known
*example* (template) instance. Only tensors cross the wire; every non-tensor part
-- dataclass field values that are scalars/strings, container shapes, dict keys,
and a ``KeyedJaggedTensor``'s feature keys -- is supplied by the example.

Supported structure: ``torch.Tensor``, ``KeyedJaggedTensor``, dataclasses, and
nested ``list`` / ``tuple`` / ``dict``; any other value is a non-tensor leaf
(carried by the example, not transferred).

On top of flatten/unflatten this module provides a two-phase all-to-all of a list
of carriers (:func:`input_dist`): :func:`input_size_dist` exchanges the small
tensor-size metadata over a cheap CPU/gloo group, then :func:`input_data_dist`
moves the bulk tensors over an nccl group with the sizes learned in phase one.
Both phases issue their collectives with ``async_op=True`` and return a
:class:`~torchrec.distributed.types.LazyAwaitable`, so the caller can overlap
other work and ``wait()`` only when the result is needed.
"""

import math
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterator, List, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function
from torchrec.distributed.types import LazyAwaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

T = TypeVar("T")


def _kjt_tensor_fields(kjt: KeyedJaggedTensor) -> List[Tuple[str, torch.Tensor]]:
    """The KJT's tensor members in a fixed order: values, [weights], lengths|offsets.

    A KJT always has values and exactly one length representation (lengths *or*
    offsets); weights are present only for weighted features. Flatten and rebuild
    use this same order so they line up.
    """
    out: List[Tuple[str, torch.Tensor]] = [("values", kjt.values())]
    weights = kjt.weights_or_none()
    if weights is not None:
        out.append(("weights", weights))
    lengths = kjt.lengths_or_none()
    if lengths is not None:
        out.append(("lengths", lengths))
    else:
        out.append(("offsets", kjt.offsets()))
    return out


def flatten_to_tensors(obj: Any) -> List[torch.Tensor]:
    """Collect every tensor in ``obj`` into a flat list (deterministic DFS order).

    Recurses through dataclasses, lists/tuples, and dicts (in field / iteration /
    key order), and expands each ``KeyedJaggedTensor`` into its tensor members
    (see :func:`_kjt_tensor_fields`). Non-tensor leaves are ignored.

    Args:
        obj: the structured value to flatten.

    Returns:
        List[torch.Tensor]: the tensors, in the order a matching ``example`` is
        walked by :func:`unflatten_from_tensors`.
    """
    out: List[torch.Tensor] = []
    _flatten(obj, out)
    return out


def _flatten(obj: Any, out: List[torch.Tensor]) -> None:
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, KeyedJaggedTensor):
        for _name, tensor in _kjt_tensor_fields(obj):
            out.append(tensor)
    elif is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            _flatten(getattr(obj, f.name), out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten(item, out)
    elif isinstance(obj, dict):
        for value in obj.values():
            _flatten(value, out)
    # else: non-tensor leaf -> nothing to collect.


def unflatten_from_tensors(tensors: List[torch.Tensor], example: T) -> T:
    """Rebuild a value shaped like ``example`` from a flat list of tensors.

    ``tensors`` must be in the order :func:`flatten_to_tensors` produces for a
    value with the same structure as ``example``. Tensor (and KJT) slots are
    filled from ``tensors`` in order; every non-tensor part is copied from
    ``example``. Assumes dataclasses are constructible from their fields as
    keyword arguments.

    Args:
        tensors: the flattened tensors (e.g. received over the wire).
        example: a template instance defining the structure and non-tensor
            values.

    Returns:
        A new value of the same type/structure as ``example``.

    Raises:
        ValueError: if ``tensors`` has too few or too many tensors for
            ``example``.
    """
    it = iter(tensors)
    result: T = _unflatten(example, it)
    remaining = sum(1 for _ in it)
    if remaining:
        raise ValueError(
            f"too many tensors for the example: {len(tensors)} given, "
            f"{len(tensors) - remaining} used"
        )
    return result


def _unflatten(example: Any, it: Iterator[torch.Tensor]) -> Any:
    if isinstance(example, torch.Tensor):
        return _next(it)
    if isinstance(example, KeyedJaggedTensor):
        return _rebuild_kjt(example, it)
    if is_dataclass(example) and not isinstance(example, type):
        kwargs = {
            f.name: _unflatten(getattr(example, f.name), it) for f in fields(example)
        }
        return type(example)(**kwargs)
    if isinstance(example, tuple):
        values = [_unflatten(item, it) for item in example]
        # Preserve namedtuple type (constructed positionally) vs plain tuple.
        if hasattr(example, "_fields"):
            return type(example)(*values)
        return type(example)(values)
    if isinstance(example, list):
        return [_unflatten(item, it) for item in example]
    if isinstance(example, dict):
        return {key: _unflatten(value, it) for key, value in example.items()}
    # Non-tensor leaf: reuse the example's value.
    return example


def _rebuild_kjt(
    example: KeyedJaggedTensor, it: Iterator[torch.Tensor]
) -> KeyedJaggedTensor:
    parts: Dict[str, torch.Tensor] = {
        name: _next(it) for name, _tensor in _kjt_tensor_fields(example)
    }
    return KeyedJaggedTensor(
        keys=example.keys(),
        values=parts["values"],
        weights=parts.get("weights"),
        lengths=parts.get("lengths"),
        offsets=parts.get("offsets"),
    )


def _next(it: Iterator[torch.Tensor]) -> torch.Tensor:
    try:
        return next(it)
    except StopIteration:
        raise ValueError(
            "not enough tensors to reconstruct the example structure"
        ) from None


class _InputSizeAwaitable(
    LazyAwaitable[Tuple[List[List[torch.Tensor]], List[List[int]]]]
):
    """Awaitable for the size-metadata all-to-all (:func:`input_size_dist`).

    ``wait()`` completes the async exchange and returns ``(flat_send, recv_sizes)``:
    the flattened send tensors (available immediately, threaded through so phase two
    need not re-flatten) and ``recv_sizes[i][k]`` -- the dim-0 size of slot ``k`` of
    the carrier rank ``i`` is sending to this rank.
    """

    def __init__(
        self,
        # pyre-ignore[2]: dist work handle has no public type
        work: Any,
        flat_send: List[List[torch.Tensor]],
        recv_sizes: torch.Tensor,
        batch_id: int,
    ) -> None:
        super().__init__()
        self._work = work
        self._flat_send = flat_send
        self._recv_sizes = recv_sizes
        self._batch_id = batch_id

    def _wait_impl(self) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
        with record_function(f"## input_size_dist wait batch{self._batch_id} ##"):
            if self._work is not None:
                self._work.wait()
            return self._flat_send, self._recv_sizes.tolist()


class _InputDataAwaitable(LazyAwaitable[List[T]]):
    """Awaitable for the bulk tensor all-to-all (:func:`input_data_dist`).

    ``wait()`` completes the per-dtype async exchanges, slices each fused receive
    buffer back into its tensor slots, then regroups them per source rank and
    reconstructs each carrier from ``example``.
    """

    def __init__(
        self,
        # pyre-ignore[2]: dist work handles have no public type
        works: List[Any],
        out_bufs: List[torch.Tensor],
        in_bufs: List[torch.Tensor],
        bucket_slots: List[List[int]],
        recv_sizes: List[List[int]],
        row_elems: List[int],
        trailing: List[Tuple[int, ...]],
        example: T,
        world_size: int,
        num_tensors: int,
        batch_id: int,
    ) -> None:
        super().__init__()
        self._works = works
        self._out_bufs = out_bufs
        # Held only to keep the send buffers alive until the collectives complete.
        self._in_bufs = in_bufs
        self._bucket_slots = bucket_slots
        self._recv_sizes = recv_sizes
        self._row_elems = row_elems
        self._trailing = trailing
        self._example = example
        self._world_size = world_size
        self._num_tensors = num_tensors
        self._batch_id = batch_id

    def _wait_impl(self) -> List[T]:
        with record_function(f"## input_data_dist wait batch{self._batch_id} ##"):
            for work in self._works:
                if work is not None:
                    work.wait()
            # recv_slot[(k, i)] = slot k of the carrier received from rank i. Each
            # fused buffer is laid out source-major then slot-major, matching the
            # destination-major / slot-major send layout in input_data_dist.
            recv_slot: Dict[Tuple[int, int], torch.Tensor] = {}
            for buf, slots in zip(self._out_bufs, self._bucket_slots):
                pos = 0
                for i in range(self._world_size):
                    for k in slots:
                        rows = self._recv_sizes[i][k]
                        n = rows * self._row_elems[k]
                        recv_slot[(k, i)] = buf[pos : pos + n].view(
                            rows, *self._trailing[k]
                        )
                        pos += n
            recv: List[T] = []
            for i in range(self._world_size):
                recv_tensors = [recv_slot[(k, i)] for k in range(self._num_tensors)]
                recv.append(unflatten_from_tensors(recv_tensors, self._example))
            return recv


def input_size_dist(
    send: List[T],
    pg_gloo: dist.ProcessGroup,
    batch_id: int = 0,
) -> LazyAwaitable[Tuple[List[List[torch.Tensor]], List[List[int]]]]:
    """Phase one: flatten each carrier and async all-to-all the tensor-size metadata.

    ``send`` must have one carrier per rank in ``pg_gloo`` (``send[j]`` is destined
    for rank ``j``). Each carrier is flattened to the same number ``K`` of tensors
    (the homogeneity assumption), and only each tensor's dim-0 size varies between
    carriers -- trailing dims and dtype are fixed per slot and recovered from the
    example in phase two. This issues an async all-to-all of a small ``[W, K]``
    integer matrix of dim-0 sizes over the (cheap, CPU) gloo group so every rank
    learns the sizes of the tensors it is about to receive.

    Args:
        send: one carrier per destination rank; ``send[j]`` goes to rank ``j``.
        pg_gloo: a gloo process group used only for the tiny size exchange.
        batch_id: identifier for this input batch, used only to tag the profiler
            ranges (``## input_size_dist batch{batch_id} ##``).

    Returns:
        A :class:`LazyAwaitable` whose ``wait()`` yields ``(flat_send, recv_sizes)``:
        ``flat_send[j]`` is the flattened tensors of ``send[j]`` (threaded through so
        phase two need not re-flatten), and ``recv_sizes[i][k]`` is the dim-0 size of
        slot ``k`` of the carrier rank ``i`` is sending to this rank.

    Raises:
        ValueError: if ``len(send)`` does not match the group size, or the carriers
            do not all flatten to the same number of tensors.
    """
    world_size = dist.get_world_size(pg_gloo)
    if len(send) != world_size:
        raise ValueError(
            f"expected one carrier per rank: got {len(send)} for world size "
            f"{world_size}"
        )
    flat_send: List[List[torch.Tensor]] = [flatten_to_tensors(item) for item in send]
    num_tensors = len(flat_send[0])
    for j, tensors in enumerate(flat_send):
        if len(tensors) != num_tensors:
            raise ValueError(
                f"carriers must flatten to the same tensor count: send[0] has "
                f"{num_tensors}, send[{j}] has {len(tensors)}"
            )

    with record_function(f"## input_size_dist batch{batch_id} ##"):
        send_sizes = torch.tensor(
            [[t.shape[0] for t in tensors] for tensors in flat_send],
            dtype=torch.int64,
        )
        recv_sizes = torch.empty_like(send_sizes)
        work = dist.all_to_all_single(
            recv_sizes, send_sizes, group=pg_gloo, async_op=True
        )
    return _InputSizeAwaitable(work, flat_send, recv_sizes, batch_id)


def input_data_dist(
    flat_send: List[List[torch.Tensor]],
    recv_sizes: List[List[int]],
    example: T,
    pg_nccl: dist.ProcessGroup,
    batch_id: int = 0,
) -> LazyAwaitable[List[T]]:
    """Phase two: async all-to-all the bulk tensors and rebuild the carriers.

    Issues one async :func:`torch.distributed.all_to_all_single` per *dtype*: the
    tensor slots are bucketed by dtype (read from ``example``), and within a bucket
    every slot is flattened to 1-D and concatenated into one fused buffer, so a
    single collective moves all same-dtype slots at once. This mirrors TorchRec's
    ``KJTAllToAll``, which keeps native dtypes (never byte-packs) but fuses what it
    can -- here a mixed-dtype carrier collapses to one collective per distinct dtype
    rather than one per slot. The returned awaitable's ``wait()`` completes the
    collectives, slices each fused buffer back into its slots, regroups per source
    rank, and reconstructs each carrier with :func:`unflatten_from_tensors`.

    Args:
        flat_send: per-destination flattened tensors, from :func:`input_size_dist`.
        recv_sizes: per-source dim-0 sizes, from :func:`input_size_dist`.
        example: template carrier defining structure, dtypes, and non-tensor parts.
        pg_nccl: an nccl process group carrying the (CUDA) tensor data.
        batch_id: identifier for this input batch, used only to tag the profiler
            ranges (``## input_data_dist batch{batch_id} ##``).

    Returns:
        A :class:`LazyAwaitable` whose ``wait()`` yields ``recv`` with ``recv[i]``
        the carrier received from rank ``i``.
    """
    world_size = len(flat_send)
    num_tensors = len(flat_send[0])
    example_flat = flatten_to_tensors(example)
    # Per-slot layout from the example: elements per dim-0 row and the trailing shape
    # (only dim-0 varies between carriers, so these are fixed per slot).
    row_elems = [math.prod(t.shape[1:]) for t in example_flat]
    trailing = [tuple(t.shape[1:]) for t in example_flat]
    # Bucket slots by dtype in first-seen order. The example is a shared template, so
    # every rank derives the same buckets and the per-dtype collectives line up.
    buckets: Dict[torch.dtype, List[int]] = {}
    for k in range(num_tensors):
        buckets.setdefault(example_flat[k].dtype, []).append(k)

    device = flat_send[0][0].device
    # pyre-ignore[33]: dist work handles have no public type
    works: List[Any] = []
    out_bufs: List[torch.Tensor] = []
    in_bufs: List[torch.Tensor] = []
    bucket_slots: List[List[int]] = []
    with record_function(f"## input_data_dist batch{batch_id} ##"):
        for dtype, slots in buckets.items():
            # Destination-major, then slot-major: chunk j (-> rank j) is this rank's
            # slots for j laid end to end; input_split_sizes marks the j boundaries.
            in_splits = [
                sum(flat_send[j][k].numel() for k in slots) for j in range(world_size)
            ]
            in_buf = torch.cat(
                [flat_send[j][k].reshape(-1) for j in range(world_size) for k in slots]
            )
            out_splits = [
                sum(recv_sizes[i][k] * row_elems[k] for k in slots)
                for i in range(world_size)
            ]
            out_buf = torch.empty(sum(out_splits), dtype=dtype, device=device)
            with record_function(f"## input_data_dist batch{batch_id} {dtype} ##"):
                work = dist.all_to_all_single(
                    out_buf, in_buf, out_splits, in_splits, group=pg_nccl, async_op=True
                )
            works.append(work)
            out_bufs.append(out_buf)
            in_bufs.append(in_buf)
            bucket_slots.append(slots)

    return _InputDataAwaitable(
        works,
        out_bufs,
        in_bufs,
        bucket_slots,
        recv_sizes,
        row_elems,
        trailing,
        example,
        world_size,
        num_tensors,
        batch_id,
    )


def input_dist(
    send: List[T],
    example: T,
    pg_gloo: dist.ProcessGroup,
    pg_nccl: dist.ProcessGroup,
    batch_id: int = 0,
) -> LazyAwaitable[List[T]]:
    """All-to-all a list of structured carriers: one per rank in, one per rank out.

    Chains :func:`input_size_dist` (size metadata over gloo) and
    :func:`input_data_dist` (bulk tensor data over nccl). ``send[j]`` is delivered
    to rank ``j``; the returned awaitable's ``wait()`` yields ``recv[i]`` -- the
    carrier from rank ``i``.

    The size exchange is awaited here (its result is needed to lay out the data
    collectives), then the data all-to-alls are launched async and returned as a
    :class:`LazyAwaitable` so the caller can overlap work before ``wait()``.

    Args:
        send: one carrier per destination rank; ``send[j]`` goes to rank ``j``.
        example: template carrier defining structure, dtypes, and non-tensor parts.
        pg_gloo: gloo group for the size-metadata exchange.
        pg_nccl: nccl group for the tensor-data exchange.
        batch_id: identifier for this input batch, threaded to both phases to tag
            the profiler ranges.

    Returns:
        A :class:`LazyAwaitable` whose ``wait()`` yields ``recv`` with ``recv[i]``
        the carrier received from rank ``i``.
    """
    flat_send, recv_sizes = input_size_dist(send, pg_gloo, batch_id).wait()
    return input_data_dist(flat_send, recv_sizes, example, pg_nccl, batch_id)
