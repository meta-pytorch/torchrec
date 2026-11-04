#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point of the ZORM metric worker process.

The worker feeds a threaded `CPUOffloadedRecMetricModule` from the trainer's
ring. Control messages carry the trainer's update seq and apply only after every
earlier update, which keeps the ordering of the in-process
`SynchronizationMarker`.
"""


import collections
import concurrent.futures
import ctypes
import dataclasses
import enum
import functools
import logging
import os
import queue
import signal
import threading
import time
import traceback
from datetime import timedelta
from multiprocessing.connection import Connection
from typing import Any, Callable, Mapping, NamedTuple, Optional, TypeVar

import torch
import torch.distributed as dist
from torchrec.metrics.cpu_offloaded_metric_ipc import (
    load_counter,
    read_header,
    recv,
    send,
    SharedUpdateRing,
    slot_views,
    SlotLayout,
    store_counter,
)
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.metric_job_types import MetricUpdateJob, SynchronizationMarker
from torchrec.metrics.metric_module import generate_metric_module, RecMetricModule
from torchrec.metrics.metrics_config import MetricsConfig

logger: logging.Logger = logging.getLogger(__name__)

_PR_SET_PDEATHSIG: int = 1
# Idle ring poll interval; control messages wake the worker immediately.
_RING_POLL_SEC: float = 0.005
_INNER_POLL_SEC: float = 1.0
_SHUTDOWN_BOUND_SEC: float = 30.0

WorkerFactory = Callable[[Optional[dist.ProcessGroup]], RecMetricModule]


@dataclasses.dataclass
class RendezvousInfo:
    rank: int
    world_size: int
    prefix: str
    tcp_host: Optional[str] = None
    tcp_port: Optional[int] = None
    file_path: Optional[str] = None


@dataclasses.dataclass
class WorkerArgs:
    ring: SharedUpdateRing
    control_conn: Connection
    result_conn: Connection
    worker_factory: WorkerFactory
    rendezvous: RendezvousInfo
    num_threads: int


# Returns the (target, args) the spawned worker process runs.
WorkerLauncher = Callable[[WorkerArgs], tuple[Callable[..., Any], tuple[Any, ...]]]


_Kind = TypeVar("_Kind", bound=enum.Enum)


class Control(enum.Enum):
    """Trainer-to-worker messages."""

    LAYOUT = "layout"
    DROP_LAYOUTS = "drop_layouts"
    RING = "ring"
    KWARGS = "kwargs"
    COMPUTE = "compute"
    STATE_DICT = "state_dict"
    LOAD_STATE_DICT = "load_state_dict"
    RESET = "reset"
    SHUTDOWN = "shutdown"


class Reply(enum.Enum):
    """Worker-to-trainer messages."""

    READY = "ready"
    RESULT = "result"
    ERROR = "error"
    FATAL = "fatal"
    COMPUTE_INTERVAL = "compute_interval"


class ControlMessage(NamedTuple):
    kind: Control
    # The worker applies the message after every update before `seq`.
    seq: int
    req_id: int = -1
    payload: tuple[Any, ...] = ()


class ReplyMessage(NamedTuple):
    kind: Reply
    req_id: int = -1
    payload: Any = None


def require_handlers(handlers: Mapping[_Kind, object], kinds: type[_Kind]) -> None:
    """Fails at startup, rather than at the first message, when a kind has no handler."""
    missing = [kind.name for kind in kinds if kind not in handlers]
    if missing:
        raise NotImplementedError(f"no handler for {kinds.__name__} {missing}")


def build_oss_worker_module(
    process_group: Optional[dist.ProcessGroup],
    *,
    metrics_config: MetricsConfig,
    batch_size: int,
    world_size: int,
    my_rank: int,
    module_kwargs: Optional[dict[str, Any]] = None,
    metric_class: type[CPUOffloadedRecMetricModule] = CPUOffloadedRecMetricModule,
) -> RecMetricModule:
    return generate_metric_module(
        metric_class=metric_class,
        metrics_config=metrics_config,
        batch_size=batch_size,
        world_size=world_size,
        my_rank=my_rank,
        state_metrics_mapping={},
        device=torch.device("cpu"),
        process_group=process_group,
        module_kwargs={
            "model_out_device": torch.device("cpu"),
            "cpu_process_group": process_group,
            **(module_kwargs or {}),
            "update_batch_size": 1,
        },
    )


def oss_worker_factory(
    metrics_config: MetricsConfig,
    batch_size: int,
    world_size: int,
    my_rank: int,
    module_kwargs: Optional[dict[str, Any]] = None,
    metric_class: type[CPUOffloadedRecMetricModule] = CPUOffloadedRecMetricModule,
) -> WorkerFactory:
    return functools.partial(
        build_oss_worker_module,
        metrics_config=metrics_config,
        batch_size=batch_size,
        world_size=world_size,
        my_rank=my_rank,
        module_kwargs=module_kwargs,
        metric_class=metric_class,
    )


def _die_with_parent() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except OSError:
        logger.warning("prctl(PR_SET_PDEATHSIG) unavailable in metric worker")


def _init_worker_process_group(info: RendezvousInfo) -> Optional[dist.ProcessGroup]:
    if info.world_size <= 1 and info.tcp_host is None and info.file_path is None:
        dist.init_process_group(
            backend="gloo", store=dist.HashStore(), rank=0, world_size=1
        )
        return dist.group.WORLD
    timeout = timedelta(minutes=30)
    if info.tcp_host is not None and info.tcp_port is not None:
        base_store: dist.Store = dist.TCPStore(
            info.tcp_host,
            info.tcp_port,
            is_master=False,
            timeout=timeout,
            wait_for_workers=False,
        )
    else:
        base_store = dist.FileStore(str(info.file_path), -1)
    store = dist.PrefixStore(info.prefix, base_store)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=info.rank,
        world_size=info.world_size,
        timeout=timeout,
    )
    return dist.group.WORLD


class _MetricWorker:
    def __init__(self, args: WorkerArgs, module: CPUOffloadedRecMetricModule) -> None:
        self._args = args
        self._module = module
        self._set_ring(args.ring)
        self._kwargs: dict[str, Any] = {}
        self._outstanding_computes: collections.deque[
            concurrent.futures.Future[Any]
        ] = collections.deque()
        self._max_outstanding_computes: int = max(1, module.compute_queue.maxsize // 2)
        self._layouts: dict[int, SlotLayout] = {}
        self._compute_interval_sent: int = module.compute_interval_steps
        self._pending: collections.deque[ControlMessage] = collections.deque()
        self._send_lock = threading.Lock()
        self._next_seq = 0
        self._running = True
        self._handlers: dict[Control, Callable[[ControlMessage], None]] = {
            Control.LAYOUT: self._on_layout,
            Control.DROP_LAYOUTS: self._on_drop_layouts,
            Control.RING: self._on_ring,
            Control.KWARGS: self._on_kwargs,
            Control.COMPUTE: self._on_compute,
            Control.STATE_DICT: self._on_state_dict,
            Control.LOAD_STATE_DICT: self._on_load_state_dict,
            Control.RESET: self._on_reset,
            Control.SHUTDOWN: self._on_shutdown,
        }
        require_handlers(self._handlers, Control)

    def _set_ring(self, ring: SharedUpdateRing) -> None:
        self._ring = ring
        self._published: torch.Tensor = ring.published
        self._consumed: torch.Tensor = ring.consumed

    def reply(self, kind: Reply, req_id: int = -1, payload: Any = None) -> None:
        with self._send_lock:
            send(self._args.result_conn, ReplyMessage(kind, req_id, payload))

    def run(self) -> None:
        control = self._args.control_conn
        while self._running:
            if self._step():
                continue
            # A dead inner update thread would strand queued computes.
            self._check_inner_alive()
            if control.poll(_RING_POLL_SEC):
                self._drain_control()

    def _step(self) -> bool:
        if self._pending and self._pending[0].seq <= self._next_seq:
            self._handle_control(self._pending.popleft())
            return True
        if self._next_seq >= load_counter(self._published):
            return False
        # Messages sent before this slot was published are already in the pipe.
        self._drain_control()
        if self._pending and self._pending[0].seq <= self._next_seq:
            return True
        self._consume(self._next_seq % self._ring.num_slots)
        return True

    def _drain_control(self) -> None:
        conn = self._args.control_conn
        try:
            while conn.poll():
                message: ControlMessage = recv(conn)
                # Layouts are needed before the slots that use them.
                if message.kind is Control.LAYOUT:
                    self._on_layout(message)
                else:
                    self._pending.append(message)
        except EOFError:
            logger.warning("metric worker lost its trainer; shutting down")
            self._pending.append(ControlMessage(Control.SHUTDOWN, self._next_seq))

    def _consume(self, slot: int) -> None:
        seq, layout_id, timestamp = read_header(self._ring.data[slot])
        if seq != self._next_seq:
            raise RuntimeError(
                f"ring slot {slot} holds update {seq}, expected {self._next_seq}"
            )
        layout = self._layouts[layout_id]
        buffer = self._ring.data[slot, : layout.total_bytes].clone()
        store_counter(self._consumed, self._next_seq + 1)
        model_out = {
            spec.key: view
            for spec, view in zip(layout.specs, slot_views(buffer, layout))
        }
        kwargs = dict(self._kwargs)
        self._next_seq += 1
        module = self._module
        self._put(
            MetricUpdateJob(
                model_out=model_out,
                kwargs=kwargs,
                update_timestamps=(
                    (timestamp,) if module.throughput_metric is not None else ()
                ),
            )
        )
        module._total_updates_enqueued += 1
        module.trained_batches += 1

    def _handle_control(self, message: ControlMessage) -> None:
        try:
            self._handlers[message.kind](message)
        except Exception as error:
            # Fire-and-forget changes have no reply path: fail rather than diverge.
            if message.req_id < 0:
                raise
            logger.exception(f"metric worker failed handling {message.kind.name}")
            self.reply(Reply.ERROR, message.req_id, _format_error(error))
            if message.kind is Control.SHUTDOWN:
                self._running = False

    def _on_layout(self, message: ControlMessage) -> None:
        layout = message.payload[0]
        self._layouts[layout.layout_id] = layout

    def _on_drop_layouts(self, message: ControlMessage) -> None:
        for layout_id in message.payload[0]:
            self._layouts.pop(layout_id, None)

    def _on_ring(self, message: ControlMessage) -> None:
        self._set_ring(message.payload[0])
        logger.info(
            f"metric worker switched to ring slots={self._ring.num_slots} "
            f"slot_bytes={self._ring.slot_bytes} at seq {message.seq}"
        )

    def _on_kwargs(self, message: ControlMessage) -> None:
        self._kwargs = message.payload[0]

    def _on_state_dict(self, message: ControlMessage) -> None:
        self._drain_updates()
        prefix, keep_vars = message.payload
        state = self._module.state_dict(prefix=prefix, keep_vars=keep_vars)
        self.reply(Reply.RESULT, message.req_id, state)

    def _on_load_state_dict(self, message: ControlMessage) -> None:
        self._drain_updates()
        state_dict, strict, assign = message.payload
        result = self._module.load_state_dict(state_dict, strict=strict, assign=assign)
        self.reply(
            Reply.RESULT,
            message.req_id,
            (list(result.missing_keys), list(result.unexpected_keys)),
        )

    def _on_reset(self, message: ControlMessage) -> None:
        self._drain_updates()
        self._module.reset()
        self.reply(Reply.RESULT, message.req_id)

    def _check_inner_alive(self) -> None:
        module = self._module
        if module._captured_exception_event.is_set():
            raise module._captured_exception or RuntimeError("metric update failed")
        if not module.update_thread.is_alive():
            raise RuntimeError("metric update thread exited")

    def _put(self, item: MetricUpdateJob | SynchronizationMarker) -> None:
        # The inner module's put_nowait would fail whenever the worker lags.
        while True:
            self._check_inner_alive()
            try:
                self._module.update_queue.put(item, timeout=_INNER_POLL_SEC)
                return
            except queue.Full:
                continue

    def _enqueue_marker(self, throughput_only: bool) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._put(SynchronizationMarker(future, throughput_only=throughput_only))
        self._module._total_computes_enqueued += 1
        return future

    def _on_compute(self, message: ControlMessage) -> None:
        req_id, (throughput_only,) = message.req_id, message.payload

        def _on_done(future: concurrent.futures.Future[Any]) -> None:
            try:
                payload = future.result()
            except Exception as error:
                self.reply(Reply.ERROR, req_id, _format_error(error))
                return
            self._report_compute_interval()
            self.reply(Reply.RESULT, req_id, payload)

        if not throughput_only:
            self._wait_for_compute_capacity()
        future = self._enqueue_marker(throughput_only)
        if not throughput_only:
            self._outstanding_computes.append(future)
        future.add_done_callback(_on_done)

    def _report_compute_interval(self) -> None:
        # The trainer's should_compute() must see inner auto-tuned intervals.
        interval = self._module.compute_interval_steps
        if interval != self._compute_interval_sent:
            self._compute_interval_sent = interval
            self.reply(Reply.COMPUTE_INTERVAL, payload=interval)

    def _wait_for_compute_capacity(self) -> None:
        # Cap outstanding computes so a backlog becomes ring backpressure, not a
        # crash in the inner module's put_nowait.
        outstanding = self._outstanding_computes
        while outstanding and outstanding[0].done():
            outstanding.popleft()
        if len(outstanding) < self._max_outstanding_computes:
            return
        logger.warning(
            f"metric worker has {len(outstanding)} outstanding computes; "
            "blocking until the oldest completes"
        )
        while len(outstanding) >= self._max_outstanding_computes:
            self._check_inner_alive()
            try:
                outstanding[0].exception(timeout=_INNER_POLL_SEC)
            except concurrent.futures.TimeoutError:
                continue
            outstanding.popleft()

    def _drain_updates(self) -> None:
        # A throughput-only marker completes after every earlier update, with no
        # collective.
        future = self._enqueue_marker(throughput_only=True)
        while True:
            self._check_inner_alive()
            try:
                future.result(timeout=_INNER_POLL_SEC)
                return
            except concurrent.futures.TimeoutError:
                continue

    def _on_shutdown(self, message: ControlMessage) -> None:
        self._running = False
        module = self._module
        errors: list[Exception] = []

        def stop() -> None:
            try:
                try:
                    self._drain_updates()
                finally:
                    module.shutdown()
            except Exception as error:
                errors.append(error)

        # Bounded so a compute stuck on dead peers cannot outlive teardown.
        stopper = threading.Thread(target=stop, name="metric_worker_stop", daemon=True)
        stopper.start()
        stopper.join(_SHUTDOWN_BOUND_SEC)
        if stopper.is_alive():
            raise RuntimeError(
                f"metric worker shutdown exceeded {_SHUTDOWN_BOUND_SEC:.0f}s"
            )
        if errors:
            raise errors[0]
        if message.req_id >= 0:
            self.reply(
                Reply.RESULT,
                message.req_id,
                {
                    "updates_enqueued": module._total_updates_enqueued,
                    "updates_processed": module._total_updates_processed,
                    "computes_enqueued": module._total_computes_enqueued,
                    "computes_processed": module._total_computes_processed,
                },
            )


def _shm_usage() -> str:
    try:
        stat = os.statvfs("/dev/shm")
    except OSError as error:
        return f"/dev/shm unavailable ({error})"
    gib = 1 << 30
    return (
        f"/dev/shm total={stat.f_blocks * stat.f_frsize / gib:.1f}GiB "
        f"free={stat.f_bavail * stat.f_frsize / gib:.1f}GiB"
    )


def _format_error(error: BaseException) -> str:
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))


def _configure_logging(rank: int) -> None:
    # A spawned child inherits no logging handlers from the trainer.
    logging.basicConfig(
        level=logging.INFO,
        format=f"[zorm_worker{rank}]:%(levelname).1s%(asctime)s %(process)d "
        "%(filename)s:%(lineno)d] %(message)s",
        force=True,
    )


def worker_main(args: WorkerArgs) -> None:
    _die_with_parent()
    _configure_logging(args.rendezvous.rank)
    torch.set_num_threads(max(1, args.num_threads))
    torch.multiprocessing._set_thread_name("metric_worker")
    worker: Optional[_MetricWorker] = None
    try:
        start = time.monotonic()
        process_group = _init_worker_process_group(args.rendezvous)
        module = args.worker_factory(process_group)
        if not isinstance(module, CPUOffloadedRecMetricModule):
            raise TypeError(
                "metric worker factory must build a CPUOffloadedRecMetricModule, "
                f"got {type(module).__name__}"
            )
        if module._update_batch_size != 1:
            raise ValueError(
                "metric worker requires update_batch_size=1, got "
                f"{module._update_batch_size}"
            )
        worker = _MetricWorker(args, module)
        worker.reply(Reply.READY, payload=time.monotonic() - start)
        logger.info(
            f"metric worker ready on rank {args.rendezvous.rank} in "
            f"{time.monotonic() - start:.1f}s; {_shm_usage()}"
        )
        worker.run()
    except Exception as error:
        logger.exception("metric worker crashed")
        try:
            # Compute callbacks may still be replying on other threads.
            if worker is not None:
                worker.reply(Reply.FATAL, payload=_format_error(error))
            else:
                send(
                    args.result_conn,
                    ReplyMessage(Reply.FATAL, -1, _format_error(error)),
                )
            args.result_conn.close()
        except (OSError, EOFError):
            pass
        os._exit(1)
    args.result_conn.close()
    os._exit(0)
