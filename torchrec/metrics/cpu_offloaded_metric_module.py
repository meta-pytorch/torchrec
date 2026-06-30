#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import atexit
import concurrent
import contextlib
import logging
import queue
import sys
import threading
import time
import traceback
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import distributed as dist
from torch.profiler import record_function

try:
    # This is a safety measure against torch package issues for when
    # Torchrec is included in the inference side model code. We should
    # remove this once we are sure all model side packages have the required
    # dependencies
    from torchrec.distributed.logging_handlers import (
        EventLoggingHandler,
        TorchrecComponent,
    )
except Exception:
    torch._C._log_api_usage_once(
        "torchrec.metrics.cpu_offloaded_metric_module.import_failure.logging_handlers"
    )

    from enum import Enum as _Enum
    from typing import Callable, TYPE_CHECKING

    if TYPE_CHECKING:
        from torchrec.distributed.logging_handlers import (
            EventLoggingHandler,
            TorchrecComponent,
        )
    else:

        class TorchrecComponent(_Enum):
            REC_METRICS = "rec_metrics"

        class EventLoggingHandler:
            @staticmethod
            def event_logger(*args: object, **kwargs: object) -> Callable:
                def decorator(func: Callable) -> Callable:
                    return func

                return decorator

            @staticmethod
            def log_event(*args: object, **kwargs: object) -> None:
                pass


from torchrec.distributed.logging_utils import EventType
from torchrec.metrics.cpu_comms_metric_module import CPUCommsRecMetricModule
from torchrec.metrics.deferrable_metrics import (
    DeferrableMetrics,
    device_supports_async,
    transfer_tensors_to_cpu,
)
from torchrec.metrics.metric_job_types import (
    MetricComputeJob,
    MetricUpdateJob,
    SynchronizationMarker,
)
from torchrec.metrics.metric_module import MetricsResult, RecMetricModule
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecMetricException
from torchrec.utils.percentile_logger import PercentileLogger
from typing_extensions import override

logger: logging.Logger = logging.getLogger(__name__)
metric_update_thread_name: str = "metric_update"
metric_compute_thread_name: str = "metric_compute"

_DRAIN_WAIT_WARN_INTERVAL_SEC: float = 60.0

# Bound the compute-thread join at teardown. If the compute thread is wedged in
# an orphaned GLOO all_gather (a peer rank skipped the collective), an unbounded
# join hangs the post_train_teardown lease and the whole job is StuckJob-killed.
_COMPUTE_SHUTDOWN_JOIN_TIMEOUT_SEC: float = 300.0


def _format_thread_stack(thread: threading.Thread) -> str:
    ident = thread.ident
    if ident is None:
        return "<no frame; thread not started>"
    frame = sys._current_frames().get(ident)
    if frame is None:
        return "<no frame; thread not found or already exited>"
    return "".join(traceback.format_stack(frame))


def _safe_merge_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate tensors along the batch dim. Falls back to torch.stack
    for 0-dim tensors since torch.cat cannot concatenate them."""
    if tensors[0].dim() == 0:
        return torch.stack(tensors, dim=0)
    return torch.cat(tensors, dim=0)


def _merge_update_jobs(jobs: list[MetricUpdateJob]) -> MetricUpdateJob:
    """
    Merge a list of MetricUpdateJobs into a single job by concatenating
    per-key tensors in `model_out` and any tensor entries in
    `kwargs['required_inputs']` along the batch dimension (dim=0).

    Worker-side input batching: K mini-batches' worth of model outputs
    are passed to rec_metrics.update as a single (concatenated) call,
    cutting PyBind crossings on the worker thread by ~K× for sum-style
    metrics. Semantically equivalent for sum-reduction metrics; for
    raw-retention metrics (AUC family) the window-eviction math advances
    by +1 instead of +K, which is negligible when window_size >> K.

    All jobs must share the same model_out key set and required_inputs
    key set; non-tensor kwargs are taken from the first job.
    """
    if len(jobs) == 1:
        return jobs[0]

    first = jobs[0]
    merged_model_out: Dict[str, torch.Tensor] = {
        key: _safe_merge_tensors([j.model_out[key] for j in jobs])
        for key in first.model_out.keys()
    }

    merged_kwargs: Dict[str, Any] = dict(first.kwargs)
    required_inputs_first = first.kwargs.get("required_inputs")
    if isinstance(required_inputs_first, dict) and required_inputs_first:
        merged_required: Dict[str, torch.Tensor] = {
            key: _safe_merge_tensors([j.kwargs["required_inputs"][key] for j in jobs])
            for key in required_inputs_first.keys()
        }
        merged_kwargs["required_inputs"] = merged_required

    return MetricUpdateJob(
        model_out=merged_model_out,
        kwargs=merged_kwargs,
        merged_count=sum(j.merged_count for j in jobs),
    )


def _foreach_clone_dict(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Batched clone of tensor values; one PyBind crossing for N tensors."""
    tensor_keys: list[str] = []
    tensor_values: list[torch.Tensor] = []
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            tensor_keys.append(k)
            tensor_values.append(v)
        else:
            out[k] = v
    if tensor_values:
        # no_grad bypasses the autograd dispatch on _foreach_clone, which
        # rejects integer dtypes (e.g., int labels/required_inputs).
        with torch.no_grad():
            cloned = torch._foreach_clone(tensor_values)
        for k, t in zip(tensor_keys, cloned):
            out[k] = t
    return out


def _foreach_clone_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, dict):
            out[k] = _foreach_clone_dict(v)
        else:
            out[k] = v
    top_keys = [k for k, v in out.items() if isinstance(v, torch.Tensor)]
    if top_keys:
        with torch.no_grad():
            cloned = torch._foreach_clone([out[k] for k in top_keys])
        for k, t in zip(top_keys, cloned):
            out[k] = t
    return out


class CPUOffloadedRecMetricModule(RecMetricModule):
    """
    RecMetricModule that offloads metric update() and compute() to CPU using background threads.

    At a high level, this metric module consists of two queues:
    - update queue: stores metric update jobs and synchronization markers. A worker thread
        processes the update queue.
        1. It updates state tensors with intermediate model outputs, and
        2. On async_compute(), generates a snapshot of the local state tensors and enqueues
            compute jobs to the compute queue.

    - compute queue: stores metric compute jobs. A worker thread processes the compute queue.
        1. It loads the state tensors into the CommsRecMetricModule
        2. Performs a GLOO all gather to gather the state tensors from all ranks
        3. Computes the metrics and sets the result in the future.

    Why we have two queues:
    - A MetricComputeJob is compute intensive and causes head of line blocking if processed in
        the same queue as the MetricUpdateJob. This can lead to large queue sizes.
    - The SynchronizationMarker acts as a synchronization marker for the compute queue to include
        all of the MetricUpdateJobs that were scheduled before it so that all ranks use a consistent
        metric states snapshot during all gather. All ranks will have processed up to 'N' update()
        jobs before the SynchronizationMarker is processed.
    """

    def __init__(
        self,
        model_out_device: torch.device,
        update_queue_size: int = 100,
        compute_queue_size: int = 100,
        update_batch_size: int = 10,
        clone_model_out: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            batch_size: batch size used by this trainer.
            world_size: the number of trainers.
            model_out_device: the device where the model_out is located (used to determine whether to perform GPU to CPU transfers).
            update_queue_size: Maximum size of the update queue. Default is 100.
            compute_queue_size: Maximum size of the compute queue. Default is 100.
            update_batch_size: Worker-side micro-batching K. The worker
                drains up to K MetricUpdateJobs per cycle, merges their
                tensors on GPU via torch.cat, and dispatches one
                rec_metrics.update call — cutting PyBind crossings on the
                worker by ~K× with no added trainer-thread work. Drain
                stops at any SynchronizationMarker, which is processed
                after the merged batch. Default is 10; set to 1 to disable.
            *args: Additional positional arguments passed to RecMetricModule.
            **kwargs: Additional keyword arguments passed to RecMetricModule.
        """
        super().__init__(*args, **kwargs)
        self._model_out_device = model_out_device
        self._update_batch_size: int = max(1, update_batch_size)
        # Defaults to False: the enqueued job holds references to model_out and
        # the source-stream guard on the async DtoH keeps the tensors valid, so
        # the per-step GPU clone is pure overhead. Set True only when the caller
        # reuses the model_out buffer across steps (e.g. Pyper memcpy_compute),
        # where the worker would otherwise read mutated data.
        self._clone_model_out: bool = clone_model_out
        self._shutdown_event: threading.Event = threading.Event()
        self._compute_shutdown_event: threading.Event = threading.Event()
        self._shutdown_complete: bool = False
        self._captured_exception_event: threading.Event = threading.Event()
        self._captured_exception: Optional[Exception] = None

        self.update_queue: queue.Queue[
            Union[MetricUpdateJob, SynchronizationMarker]
        ] = queue.Queue(update_queue_size)
        self.compute_queue: queue.Queue[MetricComputeJob] = queue.Queue(
            compute_queue_size
        )

        self.update_thread = threading.Thread(
            target=self._update_loop, name=metric_update_thread_name, daemon=True
        )
        self.compute_thread = threading.Thread(
            target=self._compute_loop, name=metric_compute_thread_name, daemon=True
        )

        # pyrefly: ignore
        self.cpu_process_group: dist.ProcessGroup = dist.new_group(backend="gloo")
        self.comms_module: CPUCommsRecMetricModule = CPUCommsRecMetricModule(
            *args,
            **kwargs,
        )

        self.update_job_time_logger: PercentileLogger = PercentileLogger(
            metric_name="update_job_time_ms", log_interval=1000
        )
        self.update_queue_size_logger: PercentileLogger = PercentileLogger(
            metric_name="update_queue_size", log_interval=1000
        )
        self.compute_queue_size_logger: PercentileLogger = PercentileLogger(
            metric_name="compute_queue_size", log_interval=10
        )
        self.compute_job_time_logger: PercentileLogger = PercentileLogger(
            "compute_job_time_ms", log_interval=10
        )
        self.compute_metrics_time_logger: PercentileLogger = PercentileLogger(
            "compute_metrics_time_ms", log_interval=10
        )
        self.all_gather_time_logger: PercentileLogger = PercentileLogger(
            "all_gather_time_ms", log_interval=10
        )

        # Correctness counters
        self._total_updates_enqueued: int = 0
        self._total_updates_processed: int = 0
        self._total_computes_enqueued: int = 0
        self._total_computes_processed: int = 0
        self._update_errors: int = 0
        self._compute_errors: int = 0

        self.update_thread.start()
        self.compute_thread.start()

        atexit.register(self.shutdown)

        self.register_load_state_dict_post_hook(
            CPUOffloadedRecMetricModule._move_state_to_cpu_after_load
        )

        logger.info(
            f"CPUOffloadedRecMetricModule initialization complete with {model_out_device.type=}, {update_queue_size=}, {compute_queue_size=}."
        )
        self._log_event(
            "init",
            EventType.INFO,
            {
                "model_out_device": str(model_out_device),
                "update_queue_size": str(update_queue_size),
                "compute_queue_size": str(compute_queue_size),
            },
        )

    def _log_event(
        self,
        event_name: str,
        event_type: EventType,
        metadata: Optional[Dict[str, str]] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        EventLoggingHandler.log_event(
            component=TorchrecComponent.REC_METRICS.value,
            event_name=f"CPUOffloadedRecMetricModule.{event_name}",
            event_type=event_type,
            metadata=metadata,
            error_message=error_message,
            stack_trace=stack_trace,
        )

    @override
    def update(self, model_out: Dict[str, torch.Tensor], **kwargs: Any) -> None:
        self._update_rec_metrics(model_out, **kwargs)
        self.trained_batches += 1

    @override
    def _update_rec_metrics(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        """
        Called during RecMetricModule.update(). Snapshot the model outputs
        via a single batched clone, then enqueue a MetricUpdateJob.

        Args:
            model_out: intermediate model outputs to be used for metric updates
            kwargs: additional arguments required when updating metrics
        """

        if self._shutdown_event.is_set():
            raise RecMetricException("metric processor thread is shut down.")

        if self._captured_exception_event.is_set():
            assert self._captured_exception is not None
            raise self._captured_exception

        if self._clone_model_out:
            snapshot_model_out = _foreach_clone_dict(model_out)
            snapshot_kwargs = _foreach_clone_kwargs(kwargs)
        else:
            # Shallow-copy the containers (cheap; no tensor copy) so dict-level
            # mutation by the caller is isolated; tensor refs are shared.
            snapshot_model_out = dict(model_out)
            snapshot_kwargs = dict(kwargs)

        try:
            self.update_queue.put_nowait(
                MetricUpdateJob(
                    model_out=snapshot_model_out,
                    kwargs=snapshot_kwargs,
                )
            )
            self._total_updates_enqueued += 1
            self.update_queue_size_logger.add(self.update_queue.qsize())
        except queue.Full:
            self._log_event(
                "enqueue_update",
                EventType.FAILURE,
                {
                    "update_queue_size": str(self.update_queue.qsize()),
                    "total_updates_enqueued": str(self._total_updates_enqueued),
                },
                error_message="update metric queue is full",
            )
            raise RecMetricException("update metric queue is full.")

    @EventLoggingHandler.event_logger(
        TorchrecComponent.REC_METRICS, n=1000, add_wait_counter=True
    )
    def _process_metric_update_job(self, metric_update_job: MetricUpdateJob) -> None:
        """
        Process a single metric update job by a worker thread. It first
        waits until the async transfer to CPU is completed, then updates
        all metrics.

        Args:
            metric_update_job: metric update job to be processed
        """

        with record_function("## CPUOffloadedRecMetricModule:update ##"):
            start_time = time.time()
            cpu_model_out, transfer_completed_event = (
                transfer_tensors_to_cpu(metric_update_job.model_out)
                if device_supports_async(self._model_out_device)
                else (metric_update_job.model_out, None)
            )
            if transfer_completed_event is not None:
                transfer_completed_event.synchronize()

            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                self.rec_tasks,
                cpu_model_out,
                self.get_required_inputs(),
            )

            if required_inputs:
                metric_update_job.kwargs["required_inputs"] = required_inputs

            self.rec_metrics.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                **metric_update_job.kwargs,
            )

            if self.throughput_metric:
                for _ in range(metric_update_job.merged_count):
                    self.throughput_metric.update()

            elapsed_ms = (time.time() - start_time) * 1000
            self.update_job_time_logger.add(elapsed_ms)
            self._total_updates_processed += metric_update_job.merged_count

    @override
    def shutdown(self) -> None:
        """Two-phase shutdown: stop the update thread, then the compute thread.
        Idempotent — safe under explicit call + atexit, including the raise paths
        below (thread-stuck and captured-exception)."""

        if self._shutdown_complete:
            return
        self._shutdown_complete = True

        logger.info(
            f"Gracefully shutting down CPUOffloadedRecMetricModule... "
            f"update_queue={self.update_queue.qsize()}, "
            f"compute_queue={self.compute_queue.qsize()}"
        )

        self._shutdown_event.set()
        try:
            # pyrefly: ignore[implicit-import]
            self.update_queue.put_nowait(
                SynchronizationMarker(concurrent.futures.Future())
            )
        except queue.Full:
            pass
        if self.update_thread.is_alive():
            self.update_thread.join()
        logger.info(
            f"Update thread: alive={self.update_thread.is_alive()}, "
            f"update_queue_remaining={self.update_queue.qsize()}"
        )

        self._compute_shutdown_event.set()
        if self.compute_thread.is_alive():
            self.compute_thread.join(timeout=_COMPUTE_SHUTDOWN_JOIN_TIMEOUT_SEC)
            if self.compute_thread.is_alive():
                logger.error(
                    "compute_thread did not exit within "
                    f"{_COMPUTE_SHUTDOWN_JOIN_TIMEOUT_SEC}s at shutdown; abandoning "
                    "the join to avoid a teardown deadlock "
                    f"(compute_queue={self.compute_queue.qsize()}, "
                    f"captured_exception={self._captured_exception_event.is_set()}, "
                    f"compute_errors={self._compute_errors}). compute_thread stack:\n"
                    f"{_format_thread_stack(self.compute_thread)}"
                )
        logger.info(
            f"Compute thread: alive={self.compute_thread.is_alive()}, "
            f"compute_queue_remaining={self.compute_queue.qsize()}"
        )

        self.update_job_time_logger.log_percentiles()
        self.update_queue_size_logger.log_percentiles()
        self.compute_queue_size_logger.log_percentiles()
        self.compute_job_time_logger.log_percentiles()
        self.compute_metrics_time_logger.log_percentiles()
        self.all_gather_time_logger.log_percentiles()

        correctness_metadata = {
            "total_updates_enqueued": str(self._total_updates_enqueued),
            "total_updates_processed": str(self._total_updates_processed),
            "total_computes_enqueued": str(self._total_computes_enqueued),
            "total_computes_processed": str(self._total_computes_processed),
            "update_errors": str(self._update_errors),
            "compute_errors": str(self._compute_errors),
            "update_queue_remaining": str(self.update_queue.qsize()),
            "compute_queue_remaining": str(self.compute_queue.qsize()),
            "update_thread_alive": str(self.update_thread.is_alive()),
            "compute_thread_alive": str(self.compute_thread.is_alive()),
        }

        updates_match = self._total_updates_enqueued == self._total_updates_processed
        computes_match = self._total_computes_enqueued == self._total_computes_processed
        has_errors = self._update_errors > 0 or self._compute_errors > 0

        if not updates_match or not computes_match or has_errors:
            logger.warning(
                f"Correctness issue: updates_match={updates_match}, "
                f"computes_match={computes_match}, "
                f"update_errors={self._update_errors}, "
                f"compute_errors={self._compute_errors}"
            )

        if self.update_thread.is_alive():
            self._log_event(
                "shutdown",
                EventType.FAILURE,
                correctness_metadata,
                error_message="update thread did not shut down gracefully",
            )
            raise RecMetricException(
                f"update thread did not shut down gracefully. remaining queue size: {self.update_queue.qsize()}"
            )
        if self.compute_thread.is_alive():
            self._log_event(
                "shutdown",
                EventType.FAILURE,
                correctness_metadata,
                error_message="compute thread did not shut down gracefully",
            )
            raise RecMetricException(
                f"compute thread did not shut down gracefully. remaining queue size: {self.compute_queue.qsize()}"
            )

        # Surface a worker thread crash that happened before shutdown() ran;
        # is_alive() checks above pass for already-dead threads.
        if self._captured_exception_event.is_set():
            assert self._captured_exception is not None
            raise self._captured_exception

        self._log_event("shutdown", EventType.SUCCESS, correctness_metadata)
        logger.info(
            f"CPUOffloadedRecMetricModule shutdown complete. "
            f"updates={self._total_updates_processed}/{self._total_updates_enqueued}, "
            f"computes={self._total_computes_processed}/{self._total_computes_enqueued}"
        )

    @override
    def compute(self) -> DeferrableMetrics:
        raise RecMetricException(
            "CPUOffloadedRecMetricModule does not support compute(). Use async_compute() instead."
        )

    @override
    def async_compute(self) -> DeferrableMetrics:
        """
        Entry point for asynchronous metric compute. It enqueues a synchronization marker
        to the update queue.

        Returns:
            future: Pre-created future where the computed metrics will be set.
        """
        # pyrefly: ignore[implicit-import]
        metrics_future = concurrent.futures.Future()
        if self._shutdown_event.is_set():
            metrics_future.set_exception(
                RecMetricException("metric processor thread is shut down.")
            )
            return DeferrableMetrics(metrics_future)

        if self._captured_exception_event.is_set():
            assert self._captured_exception is not None
            raise self._captured_exception

        try:
            self.update_queue.put_nowait(SynchronizationMarker(metrics_future))
            self._total_computes_enqueued += 1
            self.update_queue_size_logger.add(self.update_queue.qsize())
        except queue.Full:
            self._log_event(
                "enqueue_compute",
                EventType.FAILURE,
                {
                    "update_queue_size": str(self.update_queue.qsize()),
                    "total_computes_enqueued": str(self._total_computes_enqueued),
                },
                error_message="update queue is full when enqueueing compute marker",
            )
            raise RecMetricException(
                "update queue is full when enqueueing compute marker."
            )
        return DeferrableMetrics(metrics_future)

    def _process_synchronization_marker(
        self, synchronization_marker: SynchronizationMarker
    ) -> None:
        """
        Process a synchronization marker. It generates a MetricStateSnapshot which includes
        the local metric states and enqueues a MetricComputeJob to the compute queue.

        Args:
            synchronization_marker: synchronization marker to be processed
        """

        with record_function("## CPUOffloadedRecMetricModule:sync_marker ##"):
            if not self.rec_metrics:
                raise RecMetricException("No metrics to compute.")

            if self._captured_exception_event.is_set():
                synchronization_marker.future.set_exception(
                    self._captured_exception
                    or RecMetricException("compute thread is unavailable")
                )
                return

            metric_state_snapshot = MetricStateSnapshot.from_metrics(
                self.rec_metrics,
                self.throughput_metric,
            )

            self.compute_queue.put_nowait(
                MetricComputeJob(
                    future=synchronization_marker.future,
                    metric_state_snapshot=metric_state_snapshot,
                )
            )
            self.compute_queue_size_logger.add(self.compute_queue.qsize())

    @EventLoggingHandler.event_logger(
        TorchrecComponent.REC_METRICS, add_wait_counter=True, n=1000
    )
    def _process_metric_compute_job(
        self, metric_compute_job: MetricComputeJob
    ) -> MetricsResult:
        """
        Process a metric compute job:
        1. Comms module performs all gather
        2. Load aggregated metric states into comms module
        3. Compute metrics via comms module
        """

        with record_function("## CPUOffloadedRecMetricModule:compute ##"):
            start_ms = time.time()
            self.comms_module.load_local_metric_state_snapshot(
                metric_compute_job.metric_state_snapshot
            )

            with record_function("## cpu_all_gather ##"):
                # Manual distributed sync (replaces TorchMetrics.metric.Metric.sync())
                all_gather_start_ms = time.time()
                aggregated_states = self.comms_module.get_pre_compute_states(
                    self.cpu_process_group
                )
                self.all_gather_time_logger.add(
                    (time.time() - all_gather_start_ms) * 1000
                )

            with record_function("## cpu_load_states ##"):
                self.comms_module.load_pre_compute_states(aggregated_states)

            with record_function("## metric_compute ##"):
                compute_start_ms = time.time()
                computed_metrics = self.comms_module.compute().resolve()
                self.compute_job_time_logger.add((time.time() - start_ms) * 1000)
                self.compute_metrics_time_logger.add(
                    (time.time() - compute_start_ms) * 1000
                )
                self.compute_count += 1
                self._total_computes_processed += 1
                self._adjust_compute_interval()
                return computed_metrics

    def _update_loop(self) -> None:
        """
        Main worker loop that processes update jobs and synchronization markers.
        """

        torch.multiprocessing._set_thread_name(metric_update_thread_name)
        logger.info(f"Started thread {torch.multiprocessing._get_thread_name()}")

        while not self._shutdown_event.is_set():
            try:
                self._do_batched_update_work()
            except Exception as e:
                self._update_errors += 1
                logger.exception(f"Exception in update loop: {e}")
                self._log_event(
                    "update_loop",
                    EventType.FAILURE,
                    {
                        "update_queue_size": str(self.update_queue.qsize()),
                        "total_updates_processed": str(self._total_updates_processed),
                    },
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                )
                self._captured_exception = e
                self._captured_exception_event.set()
                return

        remaining = self._flush_remaining_work(self.update_queue)
        logger.info(f"Flushed {remaining} remaining update items during shutdown.")

    def _compute_loop(self) -> None:
        """
        Main compute loop that processes compute jobs.
        """
        torch.multiprocessing._set_thread_name(metric_compute_thread_name)
        logger.info(f"Started thread {torch.multiprocessing._get_thread_name()}")

        while not self._compute_shutdown_event.is_set():
            try:
                self._do_work(self.compute_queue)
            except Exception as e:
                self._compute_errors += 1
                logger.exception(f"Exception in compute loop: {e}")
                self._log_event(
                    "compute_loop",
                    EventType.FAILURE,
                    {
                        "compute_queue_size": str(self.compute_queue.qsize()),
                        "total_computes_processed": str(self._total_computes_processed),
                    },
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                )
                self._captured_exception = e
                self._captured_exception_event.set()
                # Subsequent compute jobs all hit the same collective and would
                # fail identically — fail their futures and exit.
                self._fail_remaining_compute_jobs(e)
                return

        remaining = self._flush_remaining_work(self.compute_queue)
        logger.info(
            f"Compute thread flushed {remaining} remaining items during shutdown."
        )

    def _fail_remaining_compute_jobs(self, error: Exception) -> None:
        while not self.compute_queue.empty():
            try:
                job = self.compute_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(job, MetricComputeJob) and not job.future.done():
                job.future.set_exception(error)
            self.compute_queue.task_done()

    def _drain_update_batch(
        self, first: MetricUpdateJob
    ) -> tuple[list[MetricUpdateJob], Optional[SynchronizationMarker], int]:
        """Block until pending batch reaches update_batch_size or a
        SynchronizationMarker arrives."""
        pending_jobs: list[MetricUpdateJob] = [first]
        pending_marker: Optional[SynchronizationMarker] = None
        items_pulled = 1
        while len(pending_jobs) < self._update_batch_size:
            try:
                next_item = self.update_queue.get(timeout=_DRAIN_WAIT_WARN_INTERVAL_SEC)
            except queue.Empty:
                logger.warning(
                    f"metric_update worker blocked > {_DRAIN_WAIT_WARN_INTERVAL_SEC:.0f}s "
                    f"waiting for batch (pending={len(pending_jobs)}/"
                    f"{self._update_batch_size}, queue_size={self.update_queue.qsize()}). "
                    f"Producer may be stalled or compute_interval_steps too large."
                )
                continue
            items_pulled += 1
            if isinstance(next_item, MetricUpdateJob):
                pending_jobs.append(next_item)
            else:
                pending_marker = next_item
                break
        return pending_jobs, pending_marker, items_pulled

    def _do_batched_update_work(self) -> None:
        """Drain up to K MetricUpdateJobs, merge, and dispatch one
        rec_metrics.update call. Held SynchronizationMarker (if any) is
        processed after the merged batch."""
        first = self.update_queue.get()
        pending_jobs: list[MetricUpdateJob] = []
        pending_marker: Optional[SynchronizationMarker] = None
        items_pulled = 1

        if isinstance(first, MetricUpdateJob):
            pending_jobs, pending_marker, items_pulled = self._drain_update_batch(first)
        elif isinstance(first, SynchronizationMarker):
            pending_marker = first

        try:
            try:
                if pending_jobs:
                    self._process_metric_update_job(_merge_update_jobs(pending_jobs))
            except Exception:
                if pending_marker is not None and not pending_marker.future.done():
                    # pyrefly: ignore[implicit-import]
                    with contextlib.suppress(concurrent.futures.InvalidStateError):
                        pending_marker.future.set_exception(
                            RecMetricException(
                                "MetricUpdateJob batch failed before SynchronizationMarker"
                            )
                        )
                raise
            if pending_marker is not None:
                try:
                    self._process_synchronization_marker(pending_marker)
                except Exception as e:
                    # pyrefly: ignore[implicit-import]
                    with contextlib.suppress(concurrent.futures.InvalidStateError):
                        pending_marker.future.set_exception(e)
                    raise
        finally:
            for _ in range(items_pulled):
                self.update_queue.task_done()

    def _do_work(
        self,
        metric_job_queue: Union[
            queue.Queue[Union[MetricUpdateJob, SynchronizationMarker]],
            queue.Queue[MetricComputeJob],
        ],
    ) -> None:
        """
        Process a single item from the queue.

        Args:
            metric_job_queue: Either the update queue or the compute queue.
        """

        try:
            job = metric_job_queue.get(timeout=5.0)
        except queue.Empty:
            return
        # try/finally guarantees task_done() so queue.join() can't deadlock.
        try:
            if isinstance(job, MetricUpdateJob):
                self._process_metric_update_job(job)
            elif isinstance(job, SynchronizationMarker):
                self._process_synchronization_marker(job)
            elif isinstance(job, MetricComputeJob):
                computed_metrics = self._process_metric_compute_job(job)
                job.future.set_result(computed_metrics)
        except Exception as e:
            # Fail the future so DeferrableMetrics.resolve() doesn't block forever.
            if isinstance(job, (SynchronizationMarker, MetricComputeJob)):
                job.future.set_exception(e)
            raise
        finally:
            metric_job_queue.task_done()

    def _flush_remaining_work(
        self,
        metric_job_queue: Union[
            queue.Queue[Union[MetricUpdateJob, SynchronizationMarker]],
            queue.Queue[MetricComputeJob],
        ],
    ) -> int:
        """
        Process all remaining items in the queue during shutdown.

        Args:
            metric_job_queue: queue to process.

        Returns:
            Number of items processed.
        """
        items_processed = 0
        while not metric_job_queue.empty():
            job = metric_job_queue.get_nowait()
            try:
                if isinstance(job, MetricUpdateJob):
                    self._process_metric_update_job(job)
                elif isinstance(job, SynchronizationMarker):
                    self._process_synchronization_marker(job)
                elif isinstance(job, MetricComputeJob):
                    computed_metrics = self._process_metric_compute_job(job)
                    job.future.set_result(computed_metrics)
            except Exception as e:
                logger.warning(f"Ignoring error during shutdown flush: {e}")
                if (
                    isinstance(job, (SynchronizationMarker, MetricComputeJob))
                    and not job.future.done()
                ):
                    job.future.set_exception(e)
            items_processed += 1
            metric_job_queue.task_done()
        return items_processed

    def wait_until_queue_is_empty(
        self,
        metric_job_queue: Union[
            queue.Queue[Union[MetricUpdateJob, SynchronizationMarker]],
            queue.Queue[MetricComputeJob],
        ],
    ) -> None:
        """
        Wait until all queued work is completed.

        Args:
            metric_job_queue: queue to wait for: either update or compute queue.
        """

        metric_job_queue.join()

    @override
    def sync(self) -> None:
        """Sync the metric states across ranks. Required before checkpointing.
        Reuses the compute path so the all-gather happens via the compute thread;
        the metric values are not used here."""
        self.async_compute().resolve()
        logger.info("CPUOffloadedRecMetricModule synced.")

    @override
    # pyrefly: ignore[bad-override]
    def state_dict(
        self,
        *args: Any,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        """
        Return the state tensors for all metrics. Returns the comms module's state dict
        because it stores the aggregated metric states across ranks.


        Args are identical to torch.nn.Module.state_dict().
        """
        # pyrefly: ignore
        return self.comms_module.state_dict(  # pyrefly: ignore
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    @override
    # pyrefly: ignore
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        """
        Load state dict into the offloaded module instead of the comms module.

        The offloaded module will generate a snapshot to load into the comms module
        during compute() path.

        We need to override this method because we saved the state of the comms module
        and not the offloaded module during state_dict().

        Args are identical to torch.nn.Module.load_state_dict().
        """
        # Temporarily remove comms_module from _modules. Otherwise, it will try to traverse
        # the submodule tree and load the state dict into the comms module.
        comms_module = self._modules.pop("comms_module", None)

        try:
            super().load_state_dict(state_dict, strict=strict, assign=assign)
        finally:
            # Restore comms_module
            if comms_module is not None:
                self._modules["comms_module"] = comms_module

    @override
    def unsync(self) -> None:
        """
        unsync is not required as the local state tensors in CPUOffloadedRecMetricModule
        remains untouched. The comms module contains the aggregated state tensors after all gather
        and is essentially "discarded" after compute().
        """
        pass

    @staticmethod
    def _move_state_to_cpu_after_load(
        module: torch.nn.Module, incompatible_keys: Any
    ) -> None:
        """`load_state_dict` post-hook: move any non-CPU metric state back to CPU.

        torchmetrics `add_state` stores state via `setattr` (not `register_buffer`),
        and `_load_from_state_dict` writes loaded tensors the same way — so checkpoint
        restores can deposit cuda state on a CPU-only metric, crashing the next
        `update()`. Walk `_reductions` on each `RecMetricComputation` and move via
        getattr/setattr.
        """
        if not isinstance(module, CPUOffloadedRecMetricModule):
            return
        moved = 0
        with torch.no_grad():
            for metric in module.rec_metrics.rec_metrics:
                for comp in metric._metrics_computations:  # pyre-ignore[16]
                    for attr_name in comp._reductions:
                        state = getattr(comp, attr_name, None)
                        if isinstance(state, torch.Tensor):
                            if state.device.type != "cpu":
                                setattr(comp, attr_name, state.cpu())
                                moved += 1
                        elif isinstance(state, list):
                            list_moved = False
                            new_list = []
                            for x in state:
                                if (
                                    isinstance(x, torch.Tensor)
                                    and x.device.type != "cpu"
                                ):
                                    new_list.append(x.cpu())
                                    list_moved = True
                                else:
                                    new_list.append(x)
                            if list_moved:
                                setattr(comp, attr_name, new_list)
                                moved += 1
        if moved:
            logger.info(
                f"CPUOffloadedRecMetricModule: moved {moved} state tensors to CPU after load_state_dict"
            )
