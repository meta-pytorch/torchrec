#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Flatten structured data to a list of tensors, rebuild it, and all-to-all it.

A "carrier" (dataclass of tensors / ``KeyedJaggedTensor`` / nested containers) is
flattened on the sender and rebuilt on the receiver from a known *example*
instance: only tensors cross the wire, and every non-tensor part comes from the
example. :func:`input_dist` all-to-alls a list of carriers in two phases -- sizes
over CPU/gloo, then bulk tensors over nccl -- both async, returning a
:class:`~torchrec.distributed.types.LazyAwaitable`.
"""

import math
from collections import deque
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Deque, Dict, Generic, Iterator, List, Tuple, TypeVar

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
    the carrier rank ``i`` is sending to this rank. The sizes arrive flat and are
    reshaped to ``[world_size, recv_slots]`` here.
    """

    def __init__(
        self,
        # pyre-ignore[2]: dist work handle has no public type
        work: Any,
        flat_send: List[List[torch.Tensor]],
        recv_sizes: torch.Tensor,
        recv_slots: int,
        batch_id: int,
    ) -> None:
        super().__init__()
        self._work = work
        self._flat_send = flat_send
        self._recv_sizes = recv_sizes
        self._recv_slots = recv_slots
        self._batch_id = batch_id

    def _wait_impl(self) -> Tuple[List[List[torch.Tensor]], List[List[int]]]:
        with record_function(f"## input_size_dist wait batch{self._batch_id} ##"):
            if self._work is not None:
                self._work.wait()
            # Flat on the wire, [source rank][slot] to the caller.
            return self._flat_send, self._recv_sizes.view(-1, self._recv_slots).tolist()


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
    example: T,
    pg_gloo: dist.ProcessGroup,
    batch_id: int = 0,
) -> LazyAwaitable[Tuple[List[List[torch.Tensor]], List[List[int]]]]:
    """Phase one: flatten each carrier and async all-to-all the tensor-size metadata.

    ``send`` must have one carrier per rank in ``pg_gloo`` (``send[j]`` is destined
    for rank ``j``), and each must have the structure *that destination* expects --
    which is what ``example`` describes locally. Carriers may therefore differ in
    tensor count between destinations: a rank sending a 2-slot carrier to one peer
    and a 3-slot carrier to another is fine, as long as each peer's own example
    agrees. What is *received* is always homogeneous, since every incoming carrier
    is built for this rank.

    Only each tensor's dim-0 size varies; trailing dims and dtype are fixed per
    slot and recovered from the example in phase two. This issues an async
    all-to-all of the dim-0 sizes over the (cheap, CPU) gloo group -- ragged on the
    way out (``input_split_sizes`` is each carrier's tensor count) and rectangular
    on the way back (every peer sends this rank ``K`` sizes, ``K`` taken from
    ``example``) -- so every rank learns the sizes of the tensors it is about to
    receive.

    Args:
        send: one carrier per destination rank; ``send[j]`` goes to rank ``j``.
        example: template carrier for what this rank *receives*; its tensor count
            fixes how many sizes to expect from each peer.
        pg_gloo: a gloo process group used only for the tiny size exchange.
        batch_id: identifier for this input batch, used only to tag the profiler
            ranges (``## input_size_dist batch{batch_id} ##``).

    Returns:
        A :class:`LazyAwaitable` whose ``wait()`` yields ``(flat_send, recv_sizes)``:
        ``flat_send[j]`` is the flattened tensors of ``send[j]`` (threaded through so
        phase two need not re-flatten), and ``recv_sizes[i][k]`` is the dim-0 size of
        slot ``k`` of the carrier rank ``i`` is sending to this rank.

    Raises:
        ValueError: if ``len(send)`` does not match the group size.
    """
    world_size = dist.get_world_size(pg_gloo)
    if len(send) != world_size:
        raise ValueError(
            f"expected one carrier per rank: got {len(send)} for world size "
            f"{world_size}"
        )
    flat_send: List[List[torch.Tensor]] = [flatten_to_tensors(item) for item in send]
    # Every carrier this rank receives is built for this rank, so they all have the
    # example's slot count -- the receive side stays rectangular even when the send
    # side is ragged.
    recv_slots = len(flatten_to_tensors(example))

    with record_function(f"## input_size_dist batch{batch_id} ##"):
        send_sizes = torch.tensor(
            [t.shape[0] for tensors in flat_send for t in tensors],
            dtype=torch.int64,
        )
        in_splits = [len(tensors) for tensors in flat_send]
        out_splits = [recv_slots] * world_size
        recv_sizes = torch.empty(recv_slots * world_size, dtype=torch.int64)
        work = dist.all_to_all_single(
            recv_sizes,
            send_sizes,
            out_splits,
            in_splits,
            group=pg_gloo,
            async_op=True,
        )
    return _InputSizeAwaitable(work, flat_send, recv_sizes, recv_slots, batch_id)


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
    example_flat = flatten_to_tensors(example)
    num_tensors = len(example_flat)
    # Per-slot layout from the example: elements per dim-0 row and the trailing shape
    # (only dim-0 varies between carriers, so these are fixed per slot). This is the
    # *receive* side, which is homogeneous -- every incoming carrier is built for
    # this rank.
    row_elems = [math.prod(t.shape[1:]) for t in example_flat]
    trailing = [tuple(t.shape[1:]) for t in example_flat]
    # Bucket slots by dtype in first-seen order. The example is a shared template, so
    # every rank derives the same buckets and the per-dtype collectives line up.
    buckets: Dict[torch.dtype, List[int]] = {}
    for k in range(num_tensors):
        buckets.setdefault(example_flat[k].dtype, []).append(k)

    # The *send* side may be ragged: a carrier built for a destination with more
    # layers has more slots than this rank's own example. So bucket each
    # destination's slots by its own dtypes rather than reusing the example's slot
    # indices, which describe a different carrier.
    send_slots: List[Dict[torch.dtype, List[int]]] = []
    for j in range(world_size):
        by_dtype: Dict[torch.dtype, List[int]] = {}
        for k, tensor in enumerate(flat_send[j]):
            if tensor.dtype not in buckets:
                raise ValueError(
                    f"send[{j}] slot {k} has dtype {tensor.dtype}, which the example "
                    "does not have; a carrier may differ from the example in slot "
                    "count, not in the dtypes it carries"
                )
            by_dtype.setdefault(tensor.dtype, []).append(k)
        send_slots.append(by_dtype)

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
            # Each destination contributes its *own* slots of this dtype, in slot
            # order -- which lines up with how that peer's example slices them out,
            # since the carrier was built to its structure.
            in_splits = [
                sum(flat_send[j][k].numel() for k in send_slots[j].get(dtype, []))
                for j in range(world_size)
            ]
            parts = [
                flat_send[j][k].reshape(-1)
                for j in range(world_size)
                for k in send_slots[j].get(dtype, [])
            ]
            in_buf = (
                torch.cat(parts)
                if parts
                else torch.empty(0, dtype=dtype, device=device)
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
    flat_send, recv_sizes = input_size_dist(send, example, pg_gloo, batch_id).wait()
    return input_data_dist(flat_send, recv_sizes, example, pg_nccl, batch_id)


class InputDistDriver(Generic[T]):
    """Feeds a pipeline schedule with microbatches produced by :func:`input_dist`.

    One :func:`input_dist` round over a cascade yields one carrier per stage --
    i.e. as many microbatches as there are stages, all destined for the calling
    rank. A schedule asks for ``n`` microbatches per pass, which need not match.
    This buffers whole rounds in a FIFO and hands out ``n`` at a time,
    decoupling the two:

    * more microbatches than stages (8 wanted, 4 stages) -> 2 rounds per pass;
    * fewer (8 wanted, 16 stages) -> one round every other pass, drained ``n``
      at a time.

    .. warning::
        :func:`input_dist` is a collective over the cascade, so **every rank in
        that cascade must run the same number of rounds**. The refill count here
        is a deterministic function of ``(queue length, n)`` and the queue starts
        empty everywhere, which is what keeps them in lock-step. Never make a
        refill depend on rank-local data -- a stage that decides to fetch one
        extra round hangs the whole cascade.

    Args:
        pg_gloo: group for the size-metadata exchange.
        pg_nccl: group for the tensor-data exchange. Usually the same handle: a
            group created on a ``cpu:gloo,cuda:nccl`` job dispatches by tensor
            device, so one handle drives both phases.
        self_index: this rank's own position in the cascade -- the entry it sends
            to itself, used as the reconstruction template since everything it
            receives has that same structure.

    Example::

        driver = InputDistDriver(pg, pg, self_index=stage_index)
        microbatches = driver.take(lambda: send_set(next(dataloader_iter)), n=8)
    """

    def __init__(
        self,
        pg_gloo: dist.ProcessGroup,
        pg_nccl: dist.ProcessGroup,
        self_index: int,
    ) -> None:
        self._pg_gloo = pg_gloo
        self._pg_nccl = pg_nccl
        self._self_index = self_index
        self._queue: Deque[T] = deque()
        # Monotonic round counter, tagging the profiler ranges.
        self._batch_id = 0

    @property
    def pending(self) -> int:
        """Microbatches already received and not yet handed out."""
        return len(self._queue)

    def exchange(self, send: List[T]) -> LazyAwaitable[List[T]]:
        """Run one all-to-all round, unwaited.

        Args:
            send: one carrier per rank of the cascade; ``send[j]`` goes to rank
                ``j``.

        Returns:
            LazyAwaitable[List[T]]: ``wait()`` yields one carrier from each rank,
            all destined here. Returned unwaited so the caller can overlap work.
        """
        batch_id = self._batch_id
        self._batch_id += 1
        return input_dist(
            send,
            send[self._self_index],
            pg_gloo=self._pg_gloo,
            pg_nccl=self._pg_nccl,
            batch_id=batch_id,
        )

    def take(self, next_send: Callable[[], List[T]], n: int) -> List[T]:
        """Hand out ``n`` microbatches, running whole rounds as needed.

        Args:
            next_send: produces one round's send set (one carrier per rank of the
                cascade). Called once per round, and only when the queue runs
                dry, so a caller reading a dataloader consumes exactly one batch
                per round.
            n: how many microbatches to return.

        Returns:
            List[T]: ``n`` carriers for this rank; the remainder of the last
            round stays queued for the next call.
        """
        while len(self._queue) < n:
            self._queue.extend(self.exchange(next_send()).wait())
        # Oldest first: the remainder of the last round stays queued for the
        # next call.
        return [self._queue.popleft() for _ in range(n)]
