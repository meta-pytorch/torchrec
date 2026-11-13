#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ZORM with the metric work moved out of the trainer process.

`ProcessIsolatedCPUOffloadedRecMetricModule` keeps the
`CPUOffloadedRecMetricModule` contract but runs update, compute and all-gather
in a spawned worker (cpu_offloaded_metric_worker.py), so that work no longer
competes for the trainer's GIL. Each step's outputs reach the worker through
one pinned shared-memory ring slot, with no per-step pickling.
"""

from __future__ import annotations

import atexit
import collections
import concurrent.futures
import contextlib
import itertools
import logging
import multiprocessing.process
import os
import sys
import threading
import time
from collections.abc import Collection, Iterator
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.distributed as dist
from torch.nn.modules.module import _IncompatibleKeys
from torch.profiler import record_function
from torchrec.distributed.logging_utils import EventType
from torchrec.metrics.cpu_offloaded_metric_ipc import (
    align,
    build_slot_layout,
    header_ints,
    header_timestamp,
    load_counter,
    recv,
    send,
    SharedUpdateRing,
    slot_views,
    SlotLayout,
    store_counter,
)
from torchrec.metrics.cpu_offloaded_metric_module import (
    EventLoggingHandler,
    TorchrecComponent,
)
from torchrec.metrics.cpu_offloaded_metric_worker import (
    Control,
    ControlMessage,
    RendezvousInfo,
    Reply,
    ReplyMessage,
    require_handlers,
    TraceUploader,
    WindowStat,
    worker_main,
    WorkerArgs,
    WorkerFactory,
    WorkerLauncher,
)
from torchrec.metrics.deferrable_metrics import (
    _get_metric_dtoh_stream,
    DeferrableMetrics,
)
from torchrec.metrics.metric_module import OffloadedRecMetricModule
from torchrec.metrics.metrics_namespace import compose_customized_metric_key
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecMetricException
from typing_extensions import override

logger: logging.Logger = logging.getLogger(__name__)

metric_listener_thread_name: str = "metric_worker_listener"

_RING_FULL_WARN_INTERVAL_SEC: float = 30.0
_WORKER_REQUEST_TIMEOUT_SEC: float = 600.0
_WORKER_SHUTDOWN_TIMEOUT_SEC: float = 45.0
_WORKER_JOIN_TIMEOUT_SEC: float = 30.0

_CUDA_HOST_REGISTER_PORTABLE: int = 1
_RING_FULL_TIMEOUT_SEC: float = 600.0
_MIN_RING_SLOTS: int = 10
_LAYOUT_CACHE_SIZE: int = 64
# Hard cap on a grown ring's pinned bytes: huge outputs fail loudly instead of OOMing.
_MAX_RING_BYTES: int = 1 << 30
_GA_UNSUPPORTED: str = (
    "ProcessIsolatedCPUOffloadedRecMetricModule does not support gradient "
    "accumulation (micro-batching) yet; use threaded ZORM"
)
_RESTART_ENV_VARS: tuple[str, ...] = (
    "MAST_HPC_JOB_ATTEMPT_INDEX",
    "TORCHELASTIC_RESTART_COUNT",
)

_instance_counter: itertools.count[int] = itertools.count()


def _profiler_enabled() -> bool:
    # The Python flag also sees global profile_all_threads sessions.
    return (
        torch.autograd.profiler._is_profiler_enabled
        or torch.autograd._profiler_enabled()
    )


def _annotate(name: str) -> contextlib.AbstractContextManager[Any]:
    return record_function(name) if _profiler_enabled() else contextlib.nullcontext()


def _with_headroom(nbytes: int) -> int:
    return align(nbytes + nbytes // 4)


def _default_store_rendezvous(
    group: Optional[dist.ProcessGroup], instance: int
) -> RendezvousInfo:
    if not dist.is_initialized():
        return RendezvousInfo(rank=0, world_size=1, prefix=f"zorm_worker/{instance}")
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    group_name = dist.distributed_c10d._get_process_group_name(
        group or dist.distributed_c10d._get_default_group()
    )
    run_id = "/".join(os.environ.get(name, "0") for name in _RESTART_ENV_VARS)
    prefix = f"zorm_worker/{run_id}/{group_name}/{instance}"
    store = dist.distributed_c10d._get_default_store()
    while isinstance(store, dist.PrefixStore):
        store = store.underlying_store
    if isinstance(store, dist.TCPStore):
        return RendezvousInfo(
            rank, world_size, prefix, tcp_host=store.host, tcp_port=store.port
        )
    if isinstance(store, dist.FileStore):
        path = store.path  # pyrefly: ignore [missing-attribute]
        return RendezvousInfo(rank, world_size, prefix, file_path=path)
    if world_size == 1:
        return RendezvousInfo(rank=0, world_size=1, prefix=prefix)
    raise RecMetricException(
        f"cannot rendezvous metric workers through a {type(store).__name__}"
    )


class _RingWriter:
    """Writes model outputs into `SharedUpdateRing` slots.

    CUDA outputs are packed per dtype into a device staging buffer laid out like
    the slot, then copied in one DtoH on a side stream; a publisher thread
    advances `published` once each copy completes. A layout that outgrows the
    slots switches to a larger ring at an update boundary.
    """

    def __init__(
        self,
        ring: SharedUpdateRing,
        device: torch.device,
        ring_bytes_budget: int,
        max_slots: int,
        on_error: Callable[[Exception], None],
    ) -> None:
        self.device = device
        self._on_error = on_error
        self._use_cuda: bool = device.type == "cuda"
        self._budget = ring_bytes_budget
        self._max_slots = max_slots
        self._published: torch.Tensor = ring.published
        self._consumed: torch.Tensor = ring.consumed
        self._layouts: collections.OrderedDict[tuple[Any, ...], SlotLayout] = (
            collections.OrderedDict()
        )
        self._next_layout_id = 0
        self._staging: Optional[torch.Tensor] = None
        self._staging_views: dict[int, list[list[torch.Tensor]]] = {}
        self._retired: list[tuple[int, list[torch.Tensor]]] = []
        self._ring_registered: list[torch.Tensor] = []
        self._in_flight: collections.deque[torch.cuda.Event] = collections.deque()
        self._free_events: list[torch.cuda.Event] = []
        self._publish_cv = threading.Condition()
        self._publisher: Optional[threading.Thread] = None
        self._publisher_stop = False
        if self._use_cuda:
            self._stream: torch.cuda.Stream = _get_metric_dtoh_stream(device)
            self._packed: torch.cuda.Event = torch.cuda.Event()
            self._publisher = threading.Thread(
                target=self._publish_loop, name="zorm_ring_publisher", daemon=True
            )
            self._publisher.start()
        self._attach(ring)

    def _attach(self, ring: SharedUpdateRing) -> None:
        self.ring = ring
        self._views: dict[tuple[int, int], list[torch.Tensor]] = {}
        self._ring_registered = []
        if self._use_cuda:
            self._host_register(ring.data)
            self._ring_registered.append(ring.data)

    def _host_register(self, tensor: torch.Tensor) -> None:
        with torch.cuda.device(self.device):
            result = torch.cuda.cudart().cudaHostRegister(
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                _CUDA_HOST_REGISTER_PORTABLE,
            )
        if int(result) != 0:
            raise RecMetricException(
                f"cudaHostRegister failed ({result}) for the ZORM worker ring; "
                "unpinned DtoH copies would block the training step"
            )
        logger.info(
            f"ZORM worker ring pinned slots={self.ring.num_slots} "
            f"slot_bytes={self.ring.slot_bytes}"
        )

    @staticmethod
    def _host_unregister(tensors: list[torch.Tensor]) -> None:
        for tensor in tensors:
            torch.cuda.cudart().cudaHostUnregister(tensor.data_ptr())

    def unregister(self) -> None:
        if not self._use_cuda:
            return
        with self._publish_cv:
            self._publisher_stop = True
            self._publish_cv.notify_all()
        if self._publisher is not None:
            self._publisher.join(timeout=10.0)
        self._stream.synchronize()
        self._host_unregister(self._ring_registered)
        self._ring_registered = []
        for _, tensors in self._retired:
            self._host_unregister(tensors)
        self._retired = []

    def layout_for(
        self, items: list[tuple[str, torch.Tensor]]
    ) -> tuple[SlotLayout, bool, list[int]]:
        """Returns (layout, is_new, evicted layout ids). The cache is bounded
        so variable-shape inputs cannot grow it without limit."""
        signature = tuple((k, t.dtype, tuple(t.shape)) for k, t in items)
        layout = self._layouts.get(signature)
        if layout is not None:
            self._layouts.move_to_end(signature)
            return layout, False, []
        layout = build_slot_layout(self._next_layout_id, items)
        self._next_layout_id += 1
        self._layouts[signature] = layout
        evicted = []
        while len(self._layouts) > _LAYOUT_CACHE_SIZE:
            _, old = self._layouts.popitem(last=False)
            evicted.append(old.layout_id)
            self._staging_views.pop(old.layout_id, None)
            self._views = {
                key: views
                for key, views in self._views.items()
                if key[0] != old.layout_id
            }
        return layout, True, evicted

    def maybe_grow(self, seq: int, layout: SlotLayout) -> Optional[SharedUpdateRing]:
        """Returns a new ring generation starting at `seq` when `layout` does
        not fit the current slots."""
        if layout.total_bytes <= self.ring.slot_bytes:
            return None
        slot_bytes = _with_headroom(layout.total_bytes)
        num_slots = min(
            self._max_slots, max(_MIN_RING_SLOTS, self._budget // slot_bytes)
        )
        if num_slots * slot_bytes > _MAX_RING_BYTES:
            raise RecMetricException(
                f"one update needs {layout.total_bytes} bytes of metric inputs; a "
                f"{num_slots}-slot ZORM ring would pin more than {_MAX_RING_BYTES} bytes"
            )
        if num_slots * slot_bytes > self._budget:
            logger.warning(
                f"ZORM worker ring needs {num_slots * slot_bytes} bytes for "
                f"{num_slots} slots of {slot_bytes}, above the "
                f"{self._budget}-byte budget"
            )
        ring = SharedUpdateRing(
            num_slots, slot_bytes, counters=(self.ring.published, self.ring.consumed)
        )
        self._retired.append((seq, self._ring_registered))
        self._attach(ring)
        return ring

    def release_retired(self) -> None:
        # The worker consumed every retired-ring slot before switching rings.
        if not self._retired:
            return
        consumed = load_counter(self._consumed)
        while self._retired and self._retired[0][0] <= consumed:
            _, tensors = self._retired.pop(0)
            if self._use_cuda:
                self._host_unregister(tensors)

    def consumed(self) -> int:
        return load_counter(self._consumed)

    def slot_available(self, seq: int) -> bool:
        return seq - load_counter(self._consumed) < self.ring.num_slots

    def _advance_published(self) -> None:
        store_counter(self._published, int(self._published[0]) + 1)

    def _publish_loop(self) -> None:
        # Only this thread retires events, so none is re-recorded mid-synchronize.
        while True:
            with self._publish_cv:
                while not self._in_flight and not self._publisher_stop:
                    self._publish_cv.wait()
                if self._publisher_stop:
                    return
                event = self._in_flight[0]
            try:
                event.synchronize()
            except Exception as error:
                logger.exception("ZORM ring publisher failed")
                failure = RecMetricException("ZORM ring publisher failed")
                failure.__cause__ = error
                self._on_error(failure)
                return
            with self._publish_cv:
                self._free_events.append(self._in_flight.popleft())
                self._advance_published()

    def published(self) -> int:
        return int(self._published[0])

    def _staging_for(
        self, layout: SlotLayout
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        staging = self._staging
        if staging is None or staging.numel() < layout.total_bytes:
            # Side-stream allocation: a replaced buffer outlives its queued copies.
            with torch.cuda.stream(self._stream):
                staging = torch.empty(
                    _with_headroom(layout.total_bytes),
                    dtype=torch.uint8,
                    device=self.device,
                )
            self._staging = staging
            self._staging_views.clear()
        views = self._staging_views.get(layout.layout_id)
        if views is None:
            slot = slot_views(staging, layout)
            views = [[slot[i] for i in group.spec_indices] for group in layout.groups]
            self._staging_views[layout.layout_id] = views
        return staging, views

    def write(
        self,
        seq: int,
        layout: SlotLayout,
        tensors: list[torch.Tensor],
        timestamp: float,
    ) -> None:
        slot = seq % self.ring.num_slots
        if not self._use_cuda:
            buffer = self.ring.data[slot]
            dsts = self._views.get((layout.layout_id, slot))
            if dsts is None:
                dsts = slot_views(buffer, layout)
                self._views[(layout.layout_id, slot)] = dsts
            torch._foreach_copy_(dsts, tensors)
            header_ints(buffer).copy_(
                torch.tensor([seq, layout.layout_id], device="cpu")
            )
            header_timestamp(buffer).fill_(timestamp)
            self._advance_published()
            return
        staging, group_dsts = self._staging_for(layout)
        header = header_ints(staging)
        with self._publish_cv:
            event = self._free_events.pop() if self._free_events else torch.cuda.Event()
        compute_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(compute_stream)
        with torch.cuda.stream(self._stream):
            for group, dsts in zip(layout.groups, group_dsts):
                torch._foreach_copy_(dsts, [tensors[i] for i in group.spec_indices])
            self._packed.record(self._stream)
            header[:1].fill_(seq)
            header[1:].fill_(layout.layout_id)
            header_timestamp(staging).fill_(timestamp)
            self.ring.data[slot, : layout.total_bytes].copy_(
                staging[: layout.total_bytes], non_blocking=True
            )
            event.record(self._stream)
        with self._publish_cv:
            self._in_flight.append(event)
            self._publish_cv.notify_all()
        # Sources must survive packing only; the DtoH keeps overlapping compute.
        compute_stream.wait_event(self._packed)


class _TrainerHealth:
    """Trainer-side samples reported, then reset, with each worker heartbeat."""

    def __init__(self) -> None:
        self.update_ms = WindowStat("trainer_update_ms")
        self.ring_in_flight = WindowStat("ring_in_flight")
        self.ring_backlog = WindowStat("ring_backlog")
        self.ring_full_wait_ms = WindowStat("ring_full_wait_ms")
        self.compute_latency_ms = WindowStat("compute_latency_ms")

    def pop_report(self) -> dict[str, float]:
        return {
            key: value
            for stat in vars(self).values()
            for key, value in stat.pop_report().items()
        }


class _WorkerFuture(concurrent.futures.Future):
    """Future resolved by the listener thread. Waiting from the listener thread
    itself (e.g. a `subscribe()` callback that calls `state_dict()`) pumps
    replies inline instead of deadlocking."""

    def __init__(self, owner: ProcessIsolatedCPUOffloadedRecMetricModule) -> None:
        super().__init__()
        self._owner = owner

    def result(self, timeout: Optional[float] = None) -> Any:
        self._owner._before_wait(self)
        return super().result(timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        self._owner._before_wait(self)
        return super().exception(timeout)


class ProcessIsolatedCPUOffloadedRecMetricModule(OffloadedRecMetricModule):
    """
    RecMetricModule whose metric update/compute runs in a separate process.

    Same contract as `CPUOffloadedRecMetricModule`: `async_compute()` and
    `compute()` return Future-backed DeferrableMetrics, and checkpoints use the
    in-process module's keys because the worker runs that module.

    Args:
        model_out_device: device of the model outputs passed to update().
        worker_factory: picklable callable building the worker's
            CPUOffloadedRecMetricModule from the worker's gloo process group.
        num_slots: initial and maximum shared-memory ring depth, in update()
            calls.
        slot_bytes: initial bytes reserved per update() call; the ring grows
            when a step's outputs do not fit.
        ring_bytes_budget: pinned host bytes a grown ring aims to stay within.
        non_metric_model_out_keys: keys dropped before transfer unless a
            configured metric consumes them.
        cpu_process_group: group whose rank/world size the worker group mirrors.
        profile_window: (start_update, num_updates) to CPU-profile the worker.
            When None, the worker profiles exactly the updates issued while
            the trainer's profiler is enabled.
        trace_uploader: "module:function" taking (local_trace_path, rank), run
            in the worker after profiling.
        worker_launcher: returns the (target, args) the worker process runs;
            defaults to `worker_main(worker_args)`.
        health_interval_sec: seconds between worker health heartbeats.
    """

    def __init__(
        self,
        *args: Any,
        model_out_device: torch.device,
        worker_factory: WorkerFactory,
        num_slots: int = 256,
        slot_bytes: int = 64 * 1024,
        non_metric_model_out_keys: Collection[str] | None = None,
        cpu_process_group: Optional[dist.ProcessGroup] = None,
        profile_window: Optional[tuple[int, int]] = None,
        trace_uploader: Optional[TraceUploader] = None,
        worker_launcher: Optional[WorkerLauncher] = None,
        ring_bytes_budget: int = 64 * 1024 * 1024,
        health_interval_sec: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Hide the unused trainer-side metrics so recursive loads reach
        # `_load_from_state_dict` instead of these copies.
        for name in list(self._modules):
            object.__setattr__(self, name, self._modules.pop(name))
        metric_input_keys = set(self.rec_metrics.get_required_inputs() or [])
        for task in self.rec_tasks:
            metric_input_keys.update(
                key
                for key in (
                    task.label_name,
                    task.prediction_name,
                    task.weight_name,
                    task.tensor_name,
                )
                if key
            )
        self._non_metric_model_out_keys: frozenset[str] = frozenset(
            non_metric_model_out_keys or ()
        ).difference(metric_input_keys)
        self._model_out_device = model_out_device
        self._seq = 0
        self._shutdown_complete = False
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        self._captured_exception: Optional[Exception] = None
        self._pending: dict[int, concurrent.futures.Future[Any]] = {}
        self._request_ids: itertools.count[int] = itertools.count()
        self._worker_ready = threading.Event()
        self._follow_trainer_profiler: bool = profile_window is None
        self._worker_profiling = False
        self._kwargs_sent: Optional[dict[str, Any]] = None
        self._health = _TrainerHealth()
        self._control_lock = threading.Lock()
        self._rank: int = 0
        self._state_initialized = False

        instance = next(_instance_counter)
        rendezvous = _default_store_rendezvous(cpu_process_group, instance)
        self._rank = rendezvous.rank
        ctx = torch.multiprocessing.get_context("spawn")
        ring = SharedUpdateRing(num_slots, slot_bytes)
        self._writer = _RingWriter(
            ring,
            model_out_device,
            ring_bytes_budget,
            max_slots=num_slots,
            on_error=self._fail,
        )
        control_recv, self._control_conn = ctx.Pipe(duplex=False)
        self._result_conn, result_send = ctx.Pipe(duplex=False)
        worker_args = WorkerArgs(
            ring=ring,
            control_conn=control_recv,
            result_conn=result_send,
            worker_factory=worker_factory,
            rendezvous=rendezvous,
            num_threads=torch.get_num_threads(),
            profile_window=profile_window,
            trace_uploader=trace_uploader,
            health_interval_sec=health_interval_sec,
        )
        target, target_args = (worker_launcher or _launch_worker_main)(worker_args)
        self._process: multiprocessing.process.BaseProcess = ctx.Process(
            target=target,
            args=target_args,
            name=f"zorm_metric_worker_{rendezvous.rank}",
            daemon=True,
        )
        spawn_start = time.monotonic()
        with _without_main_module_import():
            self._process.start()
        control_recv.close()
        result_send.close()
        self._reply_handlers: dict[Reply, Callable[[ReplyMessage], None]] = {
            Reply.READY: self._on_ready,
            Reply.COMPUTE_INTERVAL: self._on_compute_interval,
            Reply.FATAL: self._on_fatal,
            Reply.RESULT: self._on_response,
            Reply.ERROR: self._on_response,
            Reply.HEALTH: self._on_health,
            Reply.TRACE: self._on_trace,
        }
        require_handlers(self._reply_handlers, Reply)
        self._listener = threading.Thread(
            target=self._listen, name=metric_listener_thread_name, daemon=True
        )
        self._listener.start()
        atexit.register(self.shutdown)
        logger.info(
            f"ProcessIsolatedCPUOffloadedRecMetricModule spawned worker "
            f"pid={self._process.pid} in {time.monotonic() - spawn_start:.2f}s "
            f"({num_slots=}, slot_bytes={ring.slot_bytes}, "
            f"rank={rendezvous.rank}/{rendezvous.world_size}, "
            f"prefix={rendezvous.prefix})"
        )

    def _listen(self) -> None:
        torch.multiprocessing._set_thread_name(metric_listener_thread_name)
        while self._receive_one():
            pass

    def _before_wait(self, future: concurrent.futures.Future[Any]) -> None:
        if threading.current_thread() is self._listener:
            while not future.done() and self._receive_one():
                pass

    def _receive_one(self) -> bool:
        try:
            message: ReplyMessage = recv(self._result_conn)
        except (EOFError, OSError):
            if self._shutting_down:
                self._fail_pending(RecMetricException("metric worker is shut down."))
            else:
                self._fail(
                    RecMetricException(
                        f"metric worker exited unexpectedly "
                        f"(exitcode={self._process.exitcode})"
                    )
                )
            return False
        self._reply_handlers[message.kind](message)
        return True

    def _on_ready(self, message: ReplyMessage) -> None:
        self._worker_ready.set()
        logger.info(f"metric worker ready after {message.payload:.1f}s")
        self._log_event("ready", {"startup_s": f"{message.payload:.1f}"})

    def _on_compute_interval(self, message: ReplyMessage) -> None:
        self.compute_interval_steps = int(message.payload)

    def _on_fatal(self, message: ReplyMessage) -> None:
        self._fail(RecMetricException(f"metric worker crashed:\n{message.payload}"))

    def _on_health(self, message: ReplyMessage) -> None:
        health = {
            "trainer_seq": self._seq,
            **message.payload,
            **self._health.pop_report(),
        }
        logger.info(f"metric worker health: {health}")
        self._log_event("health", health)

    def _on_trace(self, message: ReplyMessage) -> None:
        logger.info(f"ZORM worker trace (rank {self._rank}): {message.payload}")

    def _on_response(self, message: ReplyMessage) -> None:
        future = self._pending.pop(message.req_id, None)
        if future is None:
            return
        if message.kind is Reply.RESULT:
            with _annotate("## ZORM-MP:resolve ##"):
                future.set_result(message.payload)
        else:
            future.set_exception(RecMetricException(message.payload))

    def _log_event(
        self,
        name: str,
        metadata: Mapping[str, Any],
        event_type: EventType = EventType.INFO,
    ) -> None:
        EventLoggingHandler.log_event(
            component=TorchrecComponent.REC_METRICS.value,
            event_name=f"ProcessIsolatedCPUOffloadedRecMetricModule.{name}",
            event_type=event_type,
            metadata={key: str(value) for key, value in metadata.items()},
        )

    def _fail(self, error: Exception) -> None:
        if self._captured_exception is None:
            self._captured_exception = error
        logger.error(str(error))
        self._fail_pending(error)

    def _fail_pending(self, error: Exception) -> None:
        for req_id in list(self._pending):
            future = self._pending.pop(req_id, None)
            if future is not None and not future.done():
                future.set_exception(error)

    def _raise_if_failed(self) -> None:
        if self._captured_exception is not None:
            raise self._captured_exception
        if self._shutdown_complete:
            raise RecMetricException("metric worker is shut down.")

    def _send(self, kind: Control, *payload: Any) -> concurrent.futures.Future[Any]:
        self._raise_if_failed()
        req_id = next(self._request_ids)
        future: concurrent.futures.Future[Any] = _WorkerFuture(self)
        self._pending[req_id] = future
        # The listener may have failed pending futures before this one was added.
        if self._captured_exception is not None:
            self._pending.pop(req_id, None)
            raise self._captured_exception
        try:
            self._send_control(kind, *payload, req_id=req_id)
        except OSError as error:
            self._pending.pop(req_id, None)
            raise RecMetricException("metric worker is unreachable") from error
        return future

    def _request(self, kind: Control, *payload: Any) -> Any:
        return self._send(kind, *payload).result(timeout=_WORKER_REQUEST_TIMEOUT_SEC)

    def _sync_worker_profiler(self) -> None:
        """Starts or stops the worker's profiler at this update when the trainer's
        profiler turns on or off."""
        profiling = _profiler_enabled()
        if self._follow_trainer_profiler and profiling != self._worker_profiling:
            self._worker_profiling = profiling
            self._send_control(Control.PROFILE, profiling)

    @override
    def update(self, model_out: Dict[str, torch.Tensor], **kwargs: Any) -> None:
        with _annotate(f"## ZORM-MP:update seq {self._seq} ##"), torch.no_grad():
            start = time.monotonic()
            self._raise_if_failed()
            self._sync_worker_profiler()
            if self._seq == 0:
                self._warn_if_restarted_without_state()
            self._writer.release_retired()
            items = [
                (key, value)
                for key, value in model_out.items()
                if key not in self._non_metric_model_out_keys
                and isinstance(value, torch.Tensor)
            ]
            if self._debug_mode and len(self.rec_metrics) > 0 and self.rec_tasks:
                labels, predictions, weights, _ = parse_task_model_outputs(
                    self.rec_tasks, self._prepare_model_out_for_metrics(dict(items))
                )
                self._validate_rec_metric_inputs(labels, predictions, weights)
            self._send_kwargs(kwargs)
            timestamp = time.monotonic()
            layout, is_new, evicted = self._writer.layout_for(items)
            if evicted:
                self._send_control(Control.DROP_LAYOUTS, evicted)
            if is_new:
                self._send_control(Control.LAYOUT, layout)
            ring = self._writer.maybe_grow(self._seq, layout)
            if ring is not None:
                self._send_control(Control.RING, ring)
                logger.info(
                    f"ZORM worker ring grown to slots={ring.num_slots} "
                    f"slot_bytes={ring.slot_bytes} at update {self._seq}"
                )
            self._wait_for_slot()
            self._writer.write(self._seq, layout, [t for _, t in items], timestamp)
            # Consumed first: it never passes published, so the backlog is never negative.
            consumed = self._writer.consumed()
            published = self._writer.published()
            self._health.update_ms.add_elapsed(start)
            self._health.ring_in_flight.add(self._seq + 1 - published)
            self._health.ring_backlog.add(published - consumed)
            self._seq += 1
            self.trained_batches += 1

    def _warn_if_restarted_without_state(self) -> None:
        # A restore that bypasses load_state_dict() restarts worker metrics from zero.
        if self._state_initialized or all(
            os.environ.get(name, "0") == "0" for name in _RESTART_ENV_VARS
        ):
            return
        logger.warning(
            "ZORM worker received its first update on a restarted job without "
            "load_state_dict() or reset(); metric state starts from zero"
        )
        self._log_event("restart_without_state", {})

    def _send_control(self, kind: Control, *payload: Any, req_id: int = -1) -> None:
        # Subscribe callbacks send from the listener thread too; keep framing intact.
        with self._control_lock:
            send(self._control_conn, ControlMessage(kind, self._seq, req_id, payload))

    def _send_kwargs(self, kwargs: dict[str, Any]) -> None:
        try:
            unchanged = kwargs == self._kwargs_sent
        except (RuntimeError, ValueError, TypeError):
            unchanged = False
        if unchanged:
            return
        self._send_control(Control.KWARGS, kwargs)
        self._kwargs_sent = dict(kwargs)

    def _wait_for_slot(self) -> None:
        if self._writer.slot_available(self._seq):
            return
        with _annotate("## ZORM-MP:ring_full ##"):
            start = last_warn = time.monotonic()
            while not self._writer.slot_available(self._seq):
                self._raise_if_failed()
                time.sleep(0.001)
                if time.monotonic() - start > _RING_FULL_TIMEOUT_SEC:
                    raise RecMetricException(
                        f"ZORM worker made no progress for {_RING_FULL_TIMEOUT_SEC}s"
                    )
                if time.monotonic() - last_warn > _RING_FULL_WARN_INTERVAL_SEC:
                    last_warn = time.monotonic()
                    logger.warning(
                        f"ZORM worker ring full at update {self._seq} "
                        f"(worker_ready={self._worker_ready.is_set()})"
                    )
            self._health.ring_full_wait_ms.add_elapsed(start)

    @override
    def async_compute(self, *, _throughput_only: bool = False) -> DeferrableMetrics:
        return self._async_compute(_throughput_only)

    def _async_compute(self, _throughput_only: bool) -> DeferrableMetrics:
        with _annotate("## ZORM-MP:async_compute ##"):
            if self._shutdown_complete:
                future: concurrent.futures.Future[Any] = concurrent.futures.Future()
                future.set_exception(RecMetricException("metric worker is shut down."))
                return DeferrableMetrics(future)
            sent = time.monotonic()
            future = self._send(Control.COMPUTE, _throughput_only)
            deferrable = DeferrableMetrics(future)
            if not _throughput_only:
                future.add_done_callback(
                    lambda done: self._record_compute_latency(done, sent)
                )
                self.compute_count += 1
                state_values = self._state_metric_values()
                if state_values:
                    deferrable.update(state_values)
            return deferrable

    def _record_compute_latency(
        self, future: concurrent.futures.Future[Any], sent: float
    ) -> None:
        if not future.cancelled() and future.exception() is None:
            self._health.compute_latency_ms.add_elapsed(sent)

    def _state_metric_values(self) -> Dict[str, Any]:
        return {
            compose_customized_metric_key(namespace, name): value
            for namespace, component in self.state_metrics.items()
            for name, value in component.get_metrics().items()
        }

    @override
    def compute(self) -> DeferrableMetrics:
        return self._async_compute(False)

    @override
    def compute_throughput(self) -> DeferrableMetrics:
        return self._async_compute(True)

    def set_under_micro_batching(self, value: bool) -> None:
        # Fail at construction rather than publish 1/K of the data.
        if value:
            self.shutdown()
            raise RecMetricException(_GA_UNSUPPORTED)

    def update_micro_batch(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        raise RecMetricException(_GA_UNSUPPORTED)

    @override
    def local_compute(self) -> DeferrableMetrics:
        raise RecMetricException(
            "ProcessIsolatedCPUOffloadedRecMetricModule does not support local_compute()"
        )

    @override
    def sync(self) -> None:
        self._async_compute(False).resolve()

    @override
    def unsync(self) -> None:
        pass

    @override
    def reset(self) -> None:
        self._request(Control.RESET)
        self._state_initialized = True

    @override
    # pyrefly: ignore[bad-override]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        state = self._request(Control.STATE_DICT, prefix, keep_vars)
        if destination is None:
            return state
        destination.update(state)
        return destination

    @override
    # pyrefly: ignore[bad-override]
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        missing, unexpected = self._request(
            Control.LOAD_STATE_DICT, state_dict, strict, assign
        )
        self._state_initialized = True
        return _IncompatibleKeys(missing, unexpected)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Recursive parent loads bypass load_state_dict().
        own = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        try:
            # The parent applies `strict`.
            result = self.load_state_dict(own, strict=False)
        except Exception as error:
            error_msgs.append(f"{type(self).__name__} worker load failed: {error}")
            return
        missing_keys.extend(prefix + key for key in result.missing_keys)
        unexpected_keys.extend(prefix + key for key in result.unexpected_keys)

    @override
    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        atexit.unregister(self.shutdown)
        counters: Any = None
        start = time.monotonic()
        logger.info(
            f"ProcessIsolatedCPUOffloadedRecMetricModule shutting down worker "
            f"(trainer_updates={self._seq})"
        )
        try:
            if self._captured_exception is None and self._process.is_alive():
                counters = self._send(Control.SHUTDOWN).result(
                    timeout=_WORKER_SHUTDOWN_TIMEOUT_SEC
                )
        except Exception as error:
            logger.warning(f"metric worker shutdown request failed: {error}")
        finally:
            self._shutdown_complete = True
            self._stop_process()
            self._writer.unregister()
        logger.info(
            f"ProcessIsolatedCPUOffloadedRecMetricModule shutdown complete in "
            f"{time.monotonic() - start:.1f}s: trainer_updates={self._seq}, "
            f"worker={counters}"
        )
        self._log_event(
            "shutdown",
            {"trainer_updates": self._seq, **(counters or {})},
            EventType.FAILURE if self._captured_exception else EventType.INFO,
        )
        # Best-effort like threaded ZORM: raising here would mask the training error.
        if self._captured_exception is not None:
            logger.error(
                f"metric worker had failed before shutdown: {self._captured_exception}"
            )

    def _stop_process(self) -> None:
        self._process.join(timeout=_WORKER_JOIN_TIMEOUT_SEC)
        if self._process.is_alive():
            logger.error("metric worker did not exit; terminating")
            self._process.terminate()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        self._control_conn.close()
        self._listener.join(timeout=5.0)
        self._result_conn.close()
        self._fail_pending(RecMetricException("metric worker is shut down."))


@contextlib.contextmanager
def _without_main_module_import() -> Iterator[None]:
    """A spawned child re-imports `__main__` when it has a `__spec__` or
    `__file__`; for a training launcher that starts the whole service stack in
    the worker. The worker needs nothing from `__main__`."""
    main = sys.modules["__main__"]
    spec = getattr(main, "__spec__", None)
    has_file = hasattr(main, "__file__")
    path = getattr(main, "__file__", None)
    main.__spec__ = None
    if has_file:
        del main.__file__
    try:
        yield
    finally:
        main.__spec__ = spec
        if has_file:
            main.__file__ = path


def _launch_worker_main(worker_args: WorkerArgs) -> tuple[Any, tuple[Any, ...]]:
    return worker_main, (worker_args,)
