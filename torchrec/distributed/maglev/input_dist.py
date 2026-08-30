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
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

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


def inplace_copy_to_gpu(
    source_tensors: List[torch.Tensor],
    destination_tensors: List[torch.Tensor],
    memcpy_stream: torch.Stream,
) -> None:
    """Copy flattened CPU tensors into preallocated GPU tensors.

    The dedicated memcpy stream waits for the current stream, which allocated the
    destination storage, before issuing the nonblocking ``copy_`` operations.

    The caller is responsible for making the consuming stream wait for
    ``memcpy_stream`` before reading the destination tensors.
    """
    device = destination_tensors[0].device
    device_module = torch.get_device_module(device)
    current_stream = device_module.current_stream(device)
    with record_function("## inplace_copy_to_gpu ##"):
        with device_module.stream(memcpy_stream):
            memcpy_stream.wait_stream(current_stream)
            for destination, source in zip(destination_tensors, source_tensors):
                destination.copy_(source, non_blocking=True)


class _InputSizeAwaitable(LazyAwaitable[List[List[int]]]):
    """Awaitable for the size-metadata all-to-all (:func:`input_size_dist`).

    ``wait()`` completes the async size exchange and returns ``recv_sizes``, where
    ``recv_sizes[i][k]`` is the dim-0 size of slot ``k`` of the carrier rank ``i``
    is sending to this rank.
    """

    def __init__(
        self,
        flat_send: List[List[torch.Tensor]],
        recv_slots: int,
        world_size: int,
        pg_gloo: dist.ProcessGroup,
        batch_id: int,
    ) -> None:
        super().__init__()
        self._recv_slots = recv_slots
        self._batch_id = batch_id
        send_sizes = torch.tensor(
            [tensor.shape[0] for tensors in flat_send for tensor in tensors],
            dtype=torch.int64,
        )
        in_splits = [len(tensors) for tensors in flat_send]
        out_splits = [recv_slots] * world_size
        self._recv_sizes = torch.empty(recv_slots * world_size, dtype=torch.int64)
        with record_function(f"## input_size_dist batch{batch_id} ##"):
            self._work = dist.all_to_all_single(
                self._recv_sizes,
                send_sizes,
                out_splits,
                in_splits,
                group=pg_gloo,
                async_op=True,
            )

    def _wait_impl(self) -> List[List[int]]:
        with record_function(f"## input_size_dist wait batch{self._batch_id} ##"):
            if self._work is not None:
                self._work.wait()
            # Flat on the wire, [source rank][slot] to the caller.
            return self._recv_sizes.view(-1, self._recv_slots).tolist()


class _InputDataAwaitable(LazyAwaitable[List[T]]):
    """Awaitable for the bulk tensor all-to-all (:func:`input_data_dist`).

    ``wait()`` orders the caller's current stream after ``pp_data_dist_stream``,
    slices each fused receive buffer back into its tensor slots, then regroups
    them per source rank and reconstructs each carrier from ``example``.
    """

    def __init__(
        self,
        out_bufs: List[torch.Tensor],
        in_bufs: List[torch.Tensor],
        out_splits_by_bucket: List[List[int]],
        in_splits_by_bucket: List[List[int]],
        bucket_slots: List[List[int]],
        recv_sizes: List[List[int]],
        row_elems: List[int],
        trailing: List[Tuple[int, ...]],
        example: T,
        world_size: int,
        num_tensors: int,
        device: torch.device,
        pg_nccl: dist.ProcessGroup,
        pp_data_dist_stream: torch.Stream,
        batch_id: int,
    ) -> None:
        super().__init__()
        device_module = torch.get_device_module(device)
        data_done_event = device_module.Event()
        with device_module.stream(pp_data_dist_stream):
            for out_buf, in_buf, out_splits, in_splits in zip(
                out_bufs,
                in_bufs,
                out_splits_by_bucket,
                in_splits_by_bucket,
            ):
                with record_function(
                    f"## input_data_dist batch{batch_id} {out_buf.dtype} ##"
                ):
                    dist.all_to_all_single(
                        out_buf,
                        in_buf,
                        out_splits,
                        in_splits,
                        group=pg_nccl,
                        async_op=False,
                    )
            data_done_event.record(pp_data_dist_stream)

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
        self._device = device
        self._data_done_event = data_done_event

    def _wait_impl(self) -> List[T]:
        with record_function(f"## input_data_dist wait batch{self._batch_id} ##"):
            current_stream = torch.get_device_module(self._device).current_stream(
                self._device
            )
            current_stream.wait_event(self._data_done_event)
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
    device: torch.device,
    memcpy_stream: torch.Stream,
    batch_id: int = 0,
) -> Tuple[
    List[torch.Tensor],
    List[List[int]],
    _InputSizeAwaitable,
    List[torch.Tensor],
    Any,
]:
    """Flatten and copy the input, then exchange tensor-size metadata.

    ``send`` must have one carrier per rank in ``pg_gloo``. Each carrier is
    flattened and grouped by dtype. One fused GPU input buffer is allocated per
    dtype, and the CPU tensors are copied directly into views of those buffers on
    ``memcpy_stream``. The CPU dim-0 sizes are exchanged asynchronously over gloo
    while that copy is in flight.

    Only each tensor's dim-0 size varies; trailing dims and dtype are fixed per
    slot and recovered from ``example`` in :func:`input_data_dist`.

    Args:
        send: one carrier per destination rank; ``send[j]`` goes to rank ``j``.
        example: template carrier defining the received structure and tensor slots.
        pg_gloo: a gloo process group used only for the tiny size exchange.
        device: CUDA device for the copied input tensors.
        memcpy_stream: dedicated stream for the CPU-to-GPU copy.
        batch_id: identifier for this input batch, used only to tag the profiler
            ranges (``## input_size_dist batch{batch_id} ##``).

    Returns:
        The fused GPU input buffers, their per-destination split sizes, the size-
        exchange awaitable, flattened example tensors, and copy-completion event
        consumed by :func:`input_data_dist`.

    Raises:
        ValueError: if ``device`` is not CUDA or ``len(send)`` does not match the
            group size.
    """
    if device.type != "cuda":
        raise ValueError(f"input_size_dist requires a CUDA device, got {device}")

    world_size = dist.get_world_size(pg_gloo)
    if len(send) != world_size:
        raise ValueError(
            f"expected one carrier per rank: got {len(send)} for world size "
            f"{world_size}"
        )

    source_flat_send = [flatten_to_tensors(item) for item in send]
    example_flat = flatten_to_tensors(example)
    (
        in_bufs,
        in_splits_by_bucket,
        source_tensors,
        destination_tensors,
    ) = prepare_input_buffers(source_flat_send, example_flat, device)
    inplace_copy_to_gpu(
        source_tensors,
        destination_tensors,
        memcpy_stream,
    )
    device_module = torch.get_device_module(device)
    copy_done_event = device_module.Event()
    copy_done_event.record(memcpy_stream)

    return (
        in_bufs,
        in_splits_by_bucket,
        _InputSizeAwaitable(
            source_flat_send,
            len(example_flat),
            world_size,
            pg_gloo,
            batch_id,
        ),
        example_flat,
        copy_done_event,
    )


def prepare_input_buffers(
    source_flat_send: List[List[torch.Tensor]],
    example_flat: List[torch.Tensor],
    device: torch.device,
) -> Tuple[
    List[torch.Tensor],
    List[List[int]],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """Allocate fused GPU input buffers and views for direct CPU-to-GPU copy.

    Returns the fused buffers, their per-destination split sizes, CPU source
    tensors, and matching GPU destination views.
    """
    slots_by_dtype: Dict[torch.dtype, List[int]] = {}
    for k, tensor in enumerate(example_flat):
        slots_by_dtype.setdefault(tensor.dtype, []).append(k)

    send_slots: List[Dict[torch.dtype, List[int]]] = []
    for j, tensors in enumerate(source_flat_send):
        by_dtype: Dict[torch.dtype, List[int]] = {}
        for k, tensor in enumerate(tensors):
            if tensor.dtype not in slots_by_dtype:
                raise ValueError(
                    f"send[{j}] slot {k} has dtype {tensor.dtype}, which the "
                    "example does not have; a carrier may differ from the "
                    "example in slot count, not in the dtypes it carries"
                )
            by_dtype.setdefault(tensor.dtype, []).append(k)
        send_slots.append(by_dtype)

    in_bufs: List[torch.Tensor] = []
    in_splits_by_bucket: List[List[int]] = []
    source_tensors: List[torch.Tensor] = []
    destination_tensors: List[torch.Tensor] = []
    for dtype in slots_by_dtype:
        in_splits = [
            sum(source_flat_send[j][k].numel() for k in send_slots[j].get(dtype, []))
            for j in range(len(source_flat_send))
        ]
        in_buf = torch.empty(sum(in_splits), dtype=dtype, device=device)
        pos = 0
        for j, tensors in enumerate(source_flat_send):
            for k in send_slots[j].get(dtype, []):
                source = tensors[k]
                next_pos = pos + source.numel()
                source_tensors.append(source)
                destination_tensors.append(in_buf[pos:next_pos].view(source.shape))
                pos = next_pos
        in_bufs.append(in_buf)
        in_splits_by_bucket.append(in_splits)

    return in_bufs, in_splits_by_bucket, source_tensors, destination_tensors


def prepare_output_buffers(
    recv_sizes: List[List[int]],
    example_flat: List[torch.Tensor],
    device: torch.device,
) -> Tuple[
    Dict[torch.dtype, List[int]],
    List[torch.Tensor],
    List[List[int]],
    List[int],
]:
    """Allocate fused receive buffers and their per-source split sizes."""
    world_size = len(recv_sizes)
    row_elems = [math.prod(tensor.shape[1:]) for tensor in example_flat]
    slots_by_dtype: Dict[torch.dtype, List[int]] = {}
    for k, tensor in enumerate(example_flat):
        slots_by_dtype.setdefault(tensor.dtype, []).append(k)

    out_bufs: List[torch.Tensor] = []
    out_splits_by_bucket: List[List[int]] = []
    for dtype, slots in slots_by_dtype.items():
        out_splits = [
            sum(recv_sizes[i][k] * row_elems[k] for k in slots)
            for i in range(world_size)
        ]
        out_bufs.append(torch.empty(sum(out_splits), dtype=dtype, device=device))
        out_splits_by_bucket.append(out_splits)

    return (
        slots_by_dtype,
        out_bufs,
        out_splits_by_bucket,
        row_elems,
    )


def input_data_dist(
    in_bufs: List[torch.Tensor],
    in_splits_by_bucket: List[List[int]],
    recv_sizes: List[List[int]],
    example: T,
    example_flat: List[torch.Tensor],
    pg_nccl: dist.ProcessGroup,
    pp_data_dist_stream: torch.Stream,
    copy_done_event: Any,
    batch_id: int = 0,
) -> LazyAwaitable[List[T]]:
    """Launch CUDA input-data distribution and return its awaitable.

    Receive buffers are allocated from the exchanged sizes. The collectives use
    the fused input buffers prepared by :func:`input_size_dist` and are enqueued on
    ``pp_data_dist_stream`` after the CPU-to-GPU copy.
    """
    device = in_bufs[0].device
    device_module = torch.get_device_module(device)
    current_stream = device_module.current_stream(device)
    world_size = len(recv_sizes)
    num_tensors = len(example_flat)
    trailing = [tuple(tensor.shape[1:]) for tensor in example_flat]

    with record_function(f"## input_data_dist batch{batch_id} ##"):
        (
            buckets,
            out_bufs,
            out_splits_by_bucket,
            row_elems,
        ) = prepare_output_buffers(recv_sizes, example_flat, device)

    bucket_slots = list(buckets.values())

    pp_data_dist_stream.wait_event(copy_done_event)
    pp_data_dist_stream.wait_stream(current_stream)

    return _InputDataAwaitable(
        out_bufs,
        in_bufs,
        out_splits_by_bucket,
        in_splits_by_bucket,
        bucket_slots,
        recv_sizes,
        row_elems,
        trailing,
        example,
        world_size,
        num_tensors,
        device,
        pg_nccl,
        pp_data_dist_stream,
        batch_id,
    )


def input_dist(
    send: List[T],
    example: T,
    pg_gloo: dist.ProcessGroup,
    pg_nccl: dist.ProcessGroup,
    device: torch.device,
    memcpy_stream: torch.Stream,
    pp_data_dist_stream: torch.Stream,
    batch_id: int = 0,
) -> LazyAwaitable[List[T]]:
    """All-to-all a list of structured carriers: one per rank in, one per rank out.

    Chains :func:`input_size_dist` (size metadata over gloo) and
    :func:`input_data_dist` (bulk tensor data over nccl). ``send[j]`` is delivered
    to rank ``j``; the returned awaitable's ``wait()`` yields ``recv[i]`` -- the
    carrier from rank ``i``.

    The size exchange is awaited here because its result lays out the data
    collectives. The data all-to-alls are then enqueued on ``pp_data_dist_stream``
    and returned as a :class:`LazyAwaitable` so the caller can overlap work before
    ``wait()`` establishes the stream dependency.

    Args:
        send: one carrier per destination rank; ``send[j]`` goes to rank ``j``.
        example: template carrier defining structure, dtypes, and non-tensor parts.
        pg_gloo: gloo group for the size-metadata exchange.
        pg_nccl: nccl group for the tensor-data exchange.
        batch_id: identifier for this input batch, threaded to both phases to tag
            the profiler ranges.
        device: CUDA device for the input and data collectives.
        memcpy_stream: dedicated stream for the CPU-to-GPU copy.
        pp_data_dist_stream: dedicated stream for the tensor-data all-to-alls.

    Returns:
        A :class:`LazyAwaitable` whose ``wait()`` yields ``recv`` with ``recv[i]``
        the carrier received from rank ``i``.
    """
    (
        in_bufs,
        in_splits_by_bucket,
        size_awaitable,
        example_flat,
        copy_done_event,
    ) = input_size_dist(send, example, pg_gloo, device, memcpy_stream, batch_id)
    return input_data_dist(
        in_bufs,
        in_splits_by_bucket,
        size_awaitable.wait(),
        example,
        example_flat,
        pg_nccl,
        pp_data_dist_stream,
        copy_done_event,
        batch_id,
    )


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
        self._device: Optional[torch.device] = None
        self._memcpy_stream: Optional[torch.Stream] = None
        self._pp_data_dist_stream: Optional[torch.Stream] = None
        # Monotonic round counter, tagging the profiler ranges.
        self._batch_id = 0

    def set_device(self, device: torch.device) -> None:
        """Create the copy and data-distribution streams for future exchanges."""
        if device.type != "cuda":
            raise ValueError(f"InputDistDriver requires a CUDA device, got {device}")
        self._device = device
        self._memcpy_stream = torch.get_device_module(device).Stream()
        self._pp_data_dist_stream = torch.get_device_module(device).Stream()

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
        device = self._device
        if device is None:
            raise ValueError("call set_device() before input distribution")
        memcpy_stream = self._memcpy_stream
        pp_data_dist_stream = self._pp_data_dist_stream
        if memcpy_stream is None or pp_data_dist_stream is None:
            raise ValueError("CUDA input distribution streams are not initialized")
        return input_dist(
            send,
            send[self._self_index],
            pg_gloo=self._pg_gloo,
            pg_nccl=self._pg_nccl,
            batch_id=batch_id,
            device=device,
            memcpy_stream=memcpy_stream,
            pp_data_dist_stream=pp_data_dist_stream,
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
