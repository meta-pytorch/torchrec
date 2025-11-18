#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import concurrent
import logging
import queue
import threading
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch.profiler import record_function
from torchrec.metrics.cpu_comms_metric_module import CPUCommsRecMetricModule
from torchrec.metrics.metric_job_types import (
    MetricComputeJob,
    MetricUpdateJob,
    SynchronizationMarker,
)
from torchrec.metrics.metric_module import MetricValue, RecMetricModule
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecMetricException

from torchrec.utils.percentile_logger import PercentileLogger
from typing_extensions import override

logger: logging.Logger = logging.getLogger(__name__)
metric_update_thread_name: str = "metric_update"
metric_compute_thread_name: str = "metric_compute"


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
        update_queue_size: int = 100,
        compute_queue_size: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            All arguments are the same as RecMetricModule except for
            - update_queue_size: Maximum size of the update queue. Default is 100.
            - compute_queue_size: Maximum size of the update queue. Default is 100.
        """
        super().__init__(*args, **kwargs)
        self._shutdown_event = threading.Event()

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

        self.cpu_process_group: dist.ProcessGroup = dist.new_group(backend="gloo")
        self.comms_module: CPUCommsRecMetricModule = CPUCommsRecMetricModule(
            *args,
            **kwargs,
        )
        self.update_thread.start()
        self.compute_thread.start()

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

        logger.info("CPUOffloadedRecMetricModule initialization complete.")

    @override
    def _update_rec_metrics(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        """
        Called during RecMetricModule.update(). Start a non-blocking transfer of
        the model outputs and append a MetricUpdateJob to the update queue.

        Args:
            model_out: intermediate model outputs to be used for metric updates
            kwargs: additional arguments required when updating metrics
        """

        if self._shutdown_event.is_set():
            raise RecMetricException("metric processor thread is shut down.")

        try:
            cpu_model_out, transfer_completed_event = self._transfer_to_cpu(model_out)
            self.update_queue.put_nowait(
                MetricUpdateJob(
                    model_out=cpu_model_out,
                    transfer_completed_event=transfer_completed_event,
                    kwargs=kwargs,
                )
            )
            self.update_queue_size_logger.add(self.update_queue.qsize())
        except queue.Full:
            raise RecMetricException("update metric queue is full.")

    def _transfer_to_cpu(
        self,
        model_out: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.cuda.Event]:
        """
        Create a copy of model_out on CPU and return the copy. A cuda event
        is created to track when the copy is completed.


        Args:
            model_out: intermediate model outputs to be used for metric updates
        """

        transfer_completed_event = torch.cuda.Event()
        cpu_model_out = self._move_output_to_cpu(model_out)
        transfer_completed_event.record()

        return (
            cpu_model_out,
            transfer_completed_event,
        )

    def _move_output_to_cpu(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Move all tensors in output to CPU and preserve dictionary structure.
        Args:
            output: tensors to be moved to CPU
        """
        return {
            k: tensor.to(device="cpu", non_blocking=True)
            for k, tensor in output.items()
        }

    def _process_metric_update_job(self, metric_update_job: MetricUpdateJob) -> None:
        """
        Process a single metric update job by a worker thread. It first
        waits until the async transfer to CPU is completed, then updates
        all metrics.

        Args:
            metric_update_job: metric update job to be processed
        """

        with record_function("## CPUOffloadedRecMetricModule:update ##"):
            try:
                metric_update_job.transfer_completed_event.synchronize()
                labels, predictions, weights, required_inputs = (
                    parse_task_model_outputs(
                        self.rec_tasks,
                        metric_update_job.model_out,
                        self.get_required_inputs(),
                    )
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
                    self.throughput_metric.update()

            except Exception as e:
                logger.exception("Error processing metric update: %s", e)
                raise e

    @override
    def shutdown(self) -> None:
        """
        Stop the worker thread gracefully, processing all remaining queue items.
        """

        logger.info("Gracefully shutting down CPUOffloadedRecMetricModule...")
        self._shutdown_event.set()

        if self.update_thread.is_alive():
            self.update_thread.join(timeout=30.0)
        if self.compute_thread.is_alive():
            self.compute_thread.join(timeout=30.0)

        self.update_queue_size_logger.log_percentiles()
        self.compute_queue_size_logger.log_percentiles()
        self.compute_job_time_logger.log_percentiles()
        self.compute_metrics_time_logger.log_percentiles()
        self.all_gather_time_logger.log_percentiles()

        if self.update_thread.is_alive():
            raise RecMetricException(
                f"update thread did not shut down gracefully. remaining queue size: {self.update_queue.qsize()}"
            )
        if self.compute_thread.is_alive():
            raise RecMetricException(
                f"compute thread did not shut down gracefully. remaining queue size: {self.compute_queue.qsize()}"
            )
        logger.info("CPUOffloadedRecMetricModule has been successfully shutdown.")

    @override
    def compute(self) -> Dict[str, MetricValue]:
        raise RecMetricException(
            "compute() is not supported in CPUOffloadedRecMetricModule. Use async_compute() instead."
        )

    @override
    def async_compute(
        self, future: concurrent.futures.Future[Dict[str, MetricValue]]
    ) -> None:
        """
        Entry point for asynchronous metric compute. It enqueues a synchronization marker
        to the update queue.

        Args:
            future: Pre-created future where the computed metrics will be set.
        """
        if self._shutdown_event.is_set():
            future.set_exception(
                RecMetricException("metric processor thread is shut down.")
            )
            return

        self.update_queue.put_nowait(SynchronizationMarker(future))
        self.update_queue_size_logger.add(self.update_queue.qsize())

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
            self.compute_count += 1
            if not self.rec_metrics:
                raise RecMetricException("No metrics to compute.")

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

    def _process_metric_compute_job(
        self, metric_compute_job: MetricComputeJob
    ) -> Dict[str, MetricValue]:
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
                computed_metrics = self.comms_module.compute()
                self.compute_job_time_logger.add((time.time() - start_ms) * 1000)
                self.compute_metrics_time_logger.add(
                    (time.time() - compute_start_ms) * 1000
                )
                self.compute_count += 1
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
                self._do_work(self.update_queue)
            except Exception as e:
                logger.exception(f"Exception in update loop: {e}")
                raise e

        remaining = self._flush_remaining_work(self.update_queue)
        logger.info(f"Flushed {remaining} remaining items during shutdown.")

    def _compute_loop(self) -> None:
        """
        Main compute loop that processes compute jobs.
        """
        torch.multiprocessing._set_thread_name(metric_compute_thread_name)
        logger.info(f"Started thread {torch.multiprocessing._get_thread_name()}")

        while not self._shutdown_event.is_set():
            try:
                self._do_work(self.compute_queue)
            except Exception as e:
                logger.exception(f"Exception in compute loop: {e}")
                raise e

        remaining = self._flush_remaining_work(self.compute_queue)
        logger.info(
            f"Compute thread flushed {remaining} remaining items during shutdown."
        )

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
            if isinstance(job, MetricUpdateJob):
                self._process_metric_update_job(job)
            elif isinstance(job, SynchronizationMarker):
                self._process_synchronization_marker(job)
            elif isinstance(job, MetricComputeJob):
                computed_metrics = self._process_metric_compute_job(job)
                job.future.set_result(computed_metrics)
            metric_job_queue.task_done()
        except queue.Empty:
            pass

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
            if isinstance(job, MetricUpdateJob):
                self._process_metric_update_job(job)
            elif isinstance(job, SynchronizationMarker):
                self._process_synchronization_marker(job)
            elif isinstance(job, MetricComputeJob):
                computed_metrics = self._process_metric_compute_job(job)
                job.future.set_result(computed_metrics)
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
        """
        Sync the metric states across ranks. This is required before checkpointing.
        """

        # Prepare for checkpointing by waiting for all queued work to complete.
        # This ensures that the timing of the snapshot is consistent across all ranks.
        self.wait_until_queue_is_empty(self.update_queue)
        self.wait_until_queue_is_empty(self.compute_queue)
        logger.info("Ready for checkpoint.")

        snapshot = MetricStateSnapshot.from_metrics(
            self.rec_metrics,
            self.throughput_metric,
        )
        self.comms_module.load_local_metric_state_snapshot(snapshot)
        aggregated_states = self.comms_module.get_pre_compute_states(
            self.cpu_process_group
        )
        self.comms_module.load_pre_compute_states(aggregated_states)

        logger.info("CPUOffloadedRecMetricModule synced.")

    @override
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
        return self.comms_module.state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    @override
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
