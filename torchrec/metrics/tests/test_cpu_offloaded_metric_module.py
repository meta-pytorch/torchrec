#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import queue
import random
import threading
import time
import unittest
from typing import Any, Callable, cast, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torchrec.distributed.logging_utils import EventType
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.metrics.cpu_offloaded_metric_module import (
    _COMPUTE_SHUTDOWN_JOIN_TIMEOUT_SEC,
    _foreach_clone_dict,
    _foreach_clone_kwargs,
    _format_thread_stack,
    _merge_update_jobs,
    CPUOffloadedRecMetricModule,
    MetricUpdateJob,
)
from torchrec.metrics.deferrable_metrics import transfer_tensors_to_cpu
from torchrec.metrics.metric_job_types import SynchronizationMarker
from torchrec.metrics.metric_module import generate_metric_module, RecMetricModule
from torchrec.metrics.metrics_config import (
    DefaultMetricsConfig,
    MetricsConfig,
    RecComputeMode,
    RecMetricDef,
    RecMetricEnum,
)
from torchrec.metrics.rec_metric import RecMetricException, RecMetricList
from torchrec.metrics.test_utils import gen_test_tasks
from torchrec.metrics.test_utils.mock_metrics import (
    assert_tensor_dict_equals,
    create_tensor_states,
    MockRecMetric,
)
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.test_utils import get_free_port, skip_if_asan_class


def wait_until_true(
    condition: Callable[[], bool], timeout: float = 15.0, interval: float = 0.1
) -> None:
    """Wait until a condition is true or timeout is reached."""
    start_time = time.time()
    while not condition():
        time.sleep(interval)
        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout reached while waiting for condition")


def _init_process_group_with_retry(backend: str = "gloo", max_retries: int = 5) -> None:
    """init_process_group on a fresh free port, retrying on "address already in use".

    get_free_port() closes its probe socket before returning, so the port can be taken
    before the TCPStore binds it. Re-rolling the port per attempt mirrors torch's status
    quo (torch/testing/_internal/common_utils -> retry_on_connect_failures).
    """
    last_error: Optional[RuntimeError] = None
    for _ in range(max_retries):
        os.environ["MASTER_PORT"] = str(get_free_port())
        try:
            dist.init_process_group(backend=backend)
            return
        except RuntimeError as e:
            msg = str(e).lower()
            if "address already in use" not in msg:
                raise
            last_error = e
            time.sleep(random.random())
    raise RuntimeError(
        f"init_process_group failed after {max_retries} retries"
    ) from last_error


class CPUOffloadedRecMetricModuleTest(unittest.TestCase):

    def setUp(self) -> None:
        self.world_size = 1
        self.batch_size = 1
        self.my_rank = 0
        self.tasks = gen_test_tasks(["task1"])
        self.initial_states = create_tensor_states(["cross_entropy_sum"])

        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"

        self.mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=self.initial_states,
        )
        self.rec_metrics = RecMetricList([self.mock_metric])

        _init_process_group_with_retry("gloo")
        self.cpu_module: CPUOffloadedRecMetricModule = self._make_module(
            throughput_metric=ThroughputMetric(
                world_size=self.world_size,
                batch_size=self.batch_size,
                window_seconds=1,
            ),
        )

    def _make_module(self, **overrides: object) -> CPUOffloadedRecMetricModule:
        """K=1 isolates per-call semantics; batching paths use dedicated tests."""
        defaults: dict[str, object] = {
            "model_out_device": torch.device("cpu"),
            "batch_size": self.batch_size,
            "world_size": self.world_size,
            "rec_tasks": self.tasks,
            "rec_metrics": self.rec_metrics,
            "update_batch_size": 1,
        }
        # pyrefly: ignore[bad-argument-type]
        return CPUOffloadedRecMetricModule(**{**defaults, **overrides})

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        if hasattr(self, "cpu_module"):
            try:
                self.cpu_module.shutdown()
            except Exception:
                pass

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_transfer_to_cpu(self) -> None:
        """Test non-blocking tensor output transfer from GPU to CPU."""

        output = {
            "task1-prediction": torch.tensor([1.0, 2.0, 3.0]).to("cuda:0"),
            "task1-label": torch.tensor([0.0, 1.0, 0.0]).to("cuda:0"),
            "task1-weight": torch.tensor([5.0, 1.0, 0.0]).to("cuda:0"),
        }

        cpu_output, transfer_event = transfer_tensors_to_cpu(output)
        self.assertIsNotNone(transfer_event)
        assert transfer_event is not None
        wait_until_true(transfer_event.query)

        self.assertEqual(len(cpu_output), 3)
        for key, tensor in cpu_output.items():
            self.assertEqual(tensor.device.type, "cpu")
            torch.testing.assert_close(tensor, output[key].cpu())

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_update_rec_metrics(self) -> None:
        """
        Test updating the mock metric with a single batch of data. This goes through
        the update queue, to the update thread which calls update() on rec_metrics.
        """
        model_out = {
            "task1-prediction": torch.tensor([0.5, 0.7]),
            "task1-label": torch.tensor([0.0, 1.0]),
            "task1-weight": torch.tensor([1.0, 1.0]),
        }

        self.cpu_module.update(model_out)

        wait_until_true(self.mock_metric.update_called)
        self.assertTrue(self.mock_metric.predictions_update_calls is not None)
        torch.testing.assert_close(
            model_out["task1-prediction"],
            # pyrefly: ignore[bad-index]
            self.mock_metric.predictions_update_calls[0]["task1"],
        )
        self.assertTrue(self.mock_metric.labels_update_calls is not None)
        torch.testing.assert_close(
            model_out["task1-label"],
            # pyrefly: ignore[bad-index]
            self.mock_metric.labels_update_calls[0]["task1"],
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_update_rec_metrics_queue_full(self) -> None:
        cpu_module = self._make_module(
            model_out_device=torch.device("cuda"),
            update_queue_size=1,  # Small queue size
        )

        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.5]),
            "task1-weight": torch.tensor([1.0]),
        }

        block_event: threading.Event = threading.Event()

        def controlled_process_job(_: MetricUpdateJob) -> None:
            # Simulate "busy" update thread
            block_event.wait()

        with patch.object(
            cpu_module, "_process_metric_update_job", side_effect=controlled_process_job
        ):
            # Fill the queue beyond capacity
            # First item is polled and blocked. Second item will stay in queue.
            cpu_module._update_rec_metrics(model_out)
            cpu_module._update_rec_metrics(model_out)

            self.assertRaisesRegex(
                RecMetricException,
                "update metric queue is full",
                cpu_module._update_rec_metrics,
                model_out,
            )
            block_event.set()

    def test_sync_compute_raises_exception(self) -> None:
        self.assertRaisesRegex(
            RecMetricException,
            "CPUOffloadedRecMetricModule does not support compute\\(\\). Use async_compute\\(\\) instead.",
            self.cpu_module.compute,
        )

    def test_clone_model_out_false_skips_clone(self) -> None:
        """clone_model_out=False must not call the defensive _foreach_clone."""
        module = self._make_module(clone_model_out=False)
        src = torch.tensor([0.5, 0.7])
        model_out = {"task1-prediction": src}
        captured: list[MetricUpdateJob] = []
        try:
            # Intercept at enqueue so the worker thread can't drain it first.
            with patch.object(
                module.update_queue, "put_nowait", side_effect=captured.append
            ), patch(
                "torchrec.metrics.cpu_offloaded_metric_module._foreach_clone_dict"
            ) as mock_clone_dict, patch(
                "torchrec.metrics.cpu_offloaded_metric_module._foreach_clone_kwargs"
            ) as mock_clone_kwargs:
                module._update_rec_metrics(model_out)
                mock_clone_dict.assert_not_called()
                mock_clone_kwargs.assert_not_called()
            # No clone -> the enqueued tensor is the same object.
            self.assertIs(captured[0].model_out["task1-prediction"], src)
        finally:
            module.shutdown()

    def test_clone_model_out_true_clones(self) -> None:
        """Default clone_model_out=True must defensively clone model_out."""
        module = self._make_module(clone_model_out=True)
        src = torch.tensor([0.5, 0.7])
        model_out = {"task1-prediction": src}
        captured: list[MetricUpdateJob] = []
        try:
            with patch.object(
                module.update_queue, "put_nowait", side_effect=captured.append
            ):
                module._update_rec_metrics(model_out)
            # Cloned -> distinct object, equal values.
            self.assertIsNot(captured[0].model_out["task1-prediction"], src)
            torch.testing.assert_close(captured[0].model_out["task1-prediction"], src)
        finally:
            module.shutdown()

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_compute_synchronization_marker(self) -> None:
        """
        Test that async_compute() appends a synchronization marker to the compute queue
        after processing all pending metric update jobs.

        Note that the comms module's metrics are actually the ones that are computed.
        """
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.7]),
            "task1-weight": torch.tensor([1.0]),
        }

        for _ in range(10):
            self.cpu_module.update(model_out)

        self.cpu_module.async_compute()

        comms_mock_metric = cast(
            MockRecMetric, self.cpu_module.comms_module.rec_metrics.rec_metrics[0]
        )
        wait_until_true(comms_mock_metric.compute_called)

        self.assertEqual(self.cpu_module.update_queue_size_logger.count, 11)
        self.assertEqual(self.cpu_module.compute_queue_size_logger.count, 1)
        self.assertEqual(self.mock_metric.update_called_count, 10)

    def test_async_compute_after_shutdown(self) -> None:
        self.cpu_module.shutdown()

        future = self.cpu_module.async_compute()

        self.assertRaisesRegex(
            RecMetricException, "metric processor thread is shut down.", future.resolve
        )

    def test_update_after_shutdown(self) -> None:
        self.cpu_module.shutdown()

        # Should raise exception
        self.assertRaisesRegex(
            RecMetricException,
            "metric processor thread is shut down.",
            self.cpu_module.update,
            {"predictions": torch.tensor([0.5])},
        )

    def test_graceful_shutdown(self) -> None:
        self.assertTrue(self.cpu_module.update_thread.is_alive())
        self.assertTrue(self.cpu_module.compute_thread.is_alive())

        self.cpu_module.shutdown()

        self.assertFalse(self.cpu_module.update_thread.is_alive())
        self.assertFalse(self.cpu_module.compute_thread.is_alive())

    def test_shutdown_is_idempotent(self) -> None:
        self.cpu_module.shutdown()
        self.assertTrue(self.cpu_module._shutdown_complete)

        # Second call must be a no-op (no exception, no re-logging).
        with patch.object(self.cpu_module, "_log_event") as mock_log_event:
            self.cpu_module.shutdown()
            mock_log_event.assert_not_called()

    def test_shutdown_logs_stack_when_compute_thread_will_not_exit(self) -> None:
        """A compute thread still alive past the bounded join must not hang
        teardown: shutdown() abandons the join, logs an error with the thread
        stack, then raises so the job fails loudly."""
        with patch.object(
            self.cpu_module.compute_thread, "join", return_value=None
        ), patch.object(
            self.cpu_module.compute_thread, "is_alive", return_value=True
        ), patch.object(
            self.cpu_module, "_log_event"
        ), patch(
            "torchrec.metrics.cpu_offloaded_metric_module.logger"
        ) as mock_logger:
            with self.assertRaises(RecMetricException):
                self.cpu_module.shutdown()

        mock_logger.error.assert_called_once()
        (msg,) = mock_logger.error.call_args[0]
        self.assertIn(f"did not exit within {_COMPUTE_SHUTDOWN_JOIN_TIMEOUT_SEC}", msg)
        self.assertIn("compute_thread stack", msg)

    def test_format_thread_stack(self) -> None:
        # Not-started thread: ident is None.
        self.assertIn(
            "not started", _format_thread_stack(threading.Thread(target=lambda: None))
        )

        # Live thread: a real formatted stack.
        started = threading.Event()
        release = threading.Event()

        def _run() -> None:
            started.set()
            release.wait(timeout=5.0)

        live = threading.Thread(target=_run)
        live.start()
        try:
            self.assertTrue(started.wait(timeout=5.0))
            self.assertIn("File", _format_thread_stack(live))
        finally:
            release.set()
            live.join(timeout=5.0)

        # Exited thread: ident is set but has no current frame.
        self.assertIn("already exited", _format_thread_stack(live))

    def test_shutdown_reraises_captured_exception(self) -> None:
        """If a worker thread crashed before shutdown(), shutdown() must
        re-raise the captured exception so the training job fails properly."""
        with patch.object(
            self.cpu_module,
            "_process_metric_update_job",
            side_effect=RuntimeError("worker died"),
        ):
            self.cpu_module.update(
                {
                    "task1-prediction": torch.tensor([0.5]),
                    "task1-label": torch.tensor([0.5]),
                    "task1-weight": torch.tensor([1.0]),
                }
            )
            self.assertTrue(
                self.cpu_module._captured_exception_event.wait(timeout=5.0),
                "update thread did not capture exception",
            )

        with self.assertRaisesRegex(RuntimeError, "worker died"):
            self.cpu_module.shutdown()

    def test_shutdown_idempotent_after_captured_exception(self) -> None:
        """First shutdown() re-raises the captured worker exception; subsequent
        calls (e.g. via atexit after tearDown swallowed the first) must be no-ops
        rather than re-raising and re-running shutdown work."""
        with patch.object(
            self.cpu_module,
            "_process_metric_update_job",
            side_effect=RuntimeError("worker died"),
        ):
            self.cpu_module.update(
                {
                    "task1-prediction": torch.tensor([0.5]),
                    "task1-label": torch.tensor([0.5]),
                    "task1-weight": torch.tensor([1.0]),
                }
            )
            self.assertTrue(
                self.cpu_module._captured_exception_event.wait(timeout=5.0),
                "update thread did not capture exception",
            )

        with self.assertRaisesRegex(RuntimeError, "worker died"):
            self.cpu_module.shutdown()

        self.assertTrue(self.cpu_module._shutdown_complete)
        with patch.object(self.cpu_module, "_log_event") as mock_log_event:
            self.cpu_module.shutdown()
            mock_log_event.assert_not_called()

    def test_shutdown_processes_final_sync_marker_in_compute_thread(self) -> None:
        """Two-phase shutdown: the SyncMarker enqueued during update-thread
        flush must be processed by the compute thread before it stops."""
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.5]),
            "task1-weight": torch.tensor([1.0]),
        }
        for _ in range(3):
            self.cpu_module.update(model_out)

        computes_before = self.cpu_module._total_computes_processed
        self.cpu_module.shutdown()

        self.assertGreater(
            self.cpu_module._total_computes_processed,
            computes_before,
            "compute thread did not process the final SyncMarker enqueued "
            "during update-thread flush",
        )

    def test_early_stop_resolves_outstanding_compute_on_shutdown(self) -> None:
        """APS metric-based early stop: the trainer issues async_compute() to
        read a metric for the threshold check, the threshold trips, and the job
        tears down while updates are still queued and that compute is still in
        flight. shutdown() must drain the pending updates and resolve the
        outstanding future so the trainer's blocking .resolve() cannot deadlock
        the teardown lease."""
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.7]),
            "task1-weight": torch.tensor([1.0]),
        }
        for _ in range(5):
            self.cpu_module.update(model_out)

        # The metric read the early-stop threshold check blocks on.
        deferred = self.cpu_module.async_compute()

        # Early-stop teardown while work is still in flight.
        self.cpu_module.shutdown()

        # shutdown() flushes synchronously, so the outstanding future is already
        # resolved by the time it returns -- .resolve() returns immediately and
        # cannot hang the trainer.
        result = deferred.resolve()
        self.assertIsNotNone(result)

        # No queued update was dropped on the early stop.
        self.assertEqual(
            self.cpu_module._total_updates_enqueued,
            self.cpu_module._total_updates_processed,
        )
        # The outstanding compute ran; processed exceeds enqueued by the extra
        # sync marker shutdown() itself injects before joining the threads.
        self.assertGreaterEqual(
            self.cpu_module._total_computes_processed,
            self.cpu_module._total_computes_enqueued,
        )
        self.assertGreaterEqual(self.cpu_module._total_computes_processed, 1)
        self.assertTrue(self.cpu_module.update_queue.empty())
        self.assertTrue(self.cpu_module.compute_queue.empty())
        self.assertFalse(self.cpu_module.update_thread.is_alive())
        self.assertFalse(self.cpu_module.compute_thread.is_alive())

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_wait_until_queue_is_empty(self) -> None:
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.7]),
            "task1-weight": torch.tensor([1.0]),
        }
        self.cpu_module.update(model_out)
        self.cpu_module.async_compute()

        self.cpu_module.wait_until_queue_is_empty(self.cpu_module.update_queue)
        self.cpu_module.wait_until_queue_is_empty(self.cpu_module.compute_queue)

        self.assertTrue(self.cpu_module.update_queue.empty())
        self.assertTrue(self.cpu_module.compute_queue.empty())

    def test_update_thread_exception_captured(self) -> None:
        """
        Test that exceptions in update thread are:
        1. Captured in _captured_exception
        2. Cause the update thread to terminate
        3. Main thread raises it on the next update() call
        """
        test_exception = RuntimeError("Test exception from update thread")

        with patch.object(
            self.cpu_module,
            "_process_metric_update_job",
            side_effect=test_exception,
        ):
            model_out = {
                "task1-prediction": torch.tensor([0.5]),
                "task1-label": torch.tensor([0.7]),
                "task1-weight": torch.tensor([1.0]),
            }

            self.cpu_module.update(model_out)

            # Wait for exception to be captured
            captured = self.cpu_module._captured_exception_event.wait(timeout=5.0)

            self.assertTrue(captured, "Exception event should be set")
            self.assertIsNotNone(self.cpu_module._captured_exception)
            self.assertIsInstance(self.cpu_module._captured_exception, RuntimeError)
            self.assertEqual(
                str(self.cpu_module._captured_exception),
                "Test exception from update thread",
            )

            self.cpu_module.update_thread.join(timeout=5.0)
            self.assertFalse(
                self.cpu_module.update_thread.is_alive(),
                "Update thread should have terminated after exception",
            )

            with self.assertRaises(RuntimeError):
                self.cpu_module.update(model_out)

    def test_compute_thread_exception_captured(self) -> None:
        """
        Test that exceptions in compute thread are:
        1. Captured in _captured_exception
        2. Cause the compute thread to terminate
        3. Main thread raises it on the next compute() call
        """
        test_exception = RuntimeError("Test exception from compute thread")

        with patch.object(
            self.cpu_module,
            "_process_metric_compute_job",
            side_effect=test_exception,
        ):
            self.cpu_module.async_compute()

            # Wait for exception to be captured
            captured = self.cpu_module._captured_exception_event.wait(timeout=5.0)

            self.assertTrue(captured, "Exception event should be set")
            self.assertIsNotNone(self.cpu_module._captured_exception)
            self.assertIsInstance(self.cpu_module._captured_exception, RuntimeError)
            self.assertEqual(
                str(self.cpu_module._captured_exception),
                "Test exception from compute thread",
            )

            self.cpu_module.compute_thread.join(timeout=5.0)
            self.assertFalse(
                self.cpu_module.compute_thread.is_alive(),
                "compute thread should have terminated after exception",
            )

            with self.assertRaises(RuntimeError):
                self.cpu_module.async_compute()

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_state_dict_save_load(self) -> None:
        """
        Test state_dict() method. Generated from comms module, loaded into offloaded module

        Offloaded module: update local state tensors | load state_dict
        Comms module: aggregate global state tensors | save state_dict

        We want the offloaded module to load globally reduced states when starting from a checkpoint.
        Hence, we save comms module's state dict and load it into offloaded module.
        """

        offloaded_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states={
                "state_1": torch.tensor([1.0]),
                "state_2": torch.tensor([2.0]),
                "state_3": torch.tensor([3.0]),
            },
        )
        offloaded_module = self._make_module(
            model_out_device=torch.device("cuda"),
            rec_metrics=RecMetricList([offloaded_metric]),
        )

        comms_metric = cast(
            MockRecMetric, offloaded_module.comms_module.rec_metrics.rec_metrics[0]
        )
        comms_metric.set_computation_states(
            {
                "state_1": torch.tensor([4.0]),
                "state_2": torch.tensor([5.0]),
                "state_3": torch.tensor([6.0]),
            }
        )
        state_dict = offloaded_module.state_dict()
        assert_tensor_dict_equals(
            actual_states=state_dict,
            expected_states={
                "rec_metrics.rec_metrics.0._metrics_computations.0.state_1": torch.tensor(
                    [4.0]
                ),
                "rec_metrics.rec_metrics.0._metrics_computations.0.state_2": torch.tensor(
                    [5.0]
                ),
                "rec_metrics.rec_metrics.0._metrics_computations.0.state_3": torch.tensor(
                    [6.0]
                ),
            },
        )

        # Load comms state dict into offloaded module. Confirm that offloaded module
        # now also contains the updated state tensors from comms module.
        offloaded_module.load_state_dict(state_dict)

        assert_tensor_dict_equals(
            actual_states=offloaded_metric.get_computation_states(),
            expected_states={
                "state_1": torch.tensor([4.0]),
                "state_2": torch.tensor([5.0]),
                "state_3": torch.tensor([6.0]),
            },
        )
        offloaded_module.shutdown()

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_sync(self) -> None:
        """Test sync() method waits for queues to empty and syncs metric states."""
        offloaded_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states={
                "state_1": torch.tensor([0.5]),
                "state_2": torch.tensor([0.7]),
                "state_3": torch.tensor([1.0]),
            },
        )
        offloaded_module = self._make_module(
            model_out_device=torch.device("cuda"),
            rec_metrics=RecMetricList([offloaded_metric]),
        )

        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.7]),
            "task1-weight": torch.tensor([1.0]),
        }
        offloaded_module.update(model_out)
        offloaded_module.sync()

        self.assertTrue(offloaded_module.update_queue.empty())
        self.assertTrue(offloaded_module.compute_queue.empty())
        # pyrefly: ignore[not-callable]
        synced_state = offloaded_module.rec_metrics.rec_metrics[
            0
        ].get_computation_states()
        assert_tensor_dict_equals(
            actual_states=synced_state,
            expected_states={
                "state_1": torch.tensor([0.5]),
                "state_2": torch.tensor([0.7]),
                "state_3": torch.tensor([1.0]),
            },
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_transfer_to_cpu_with_indexed_cuda_device(self) -> None:
        """
        Test that _transfer_to_cpu is called correctly when model_out_device
        is a specific CUDA device index (e.g., cuda:1) rather than just 'cuda'.

        This tests that the device comparison handles indexed devices correctly,
        since torch.device("cuda:1") != torch.device("cuda").
        """
        # Create module with indexed cuda device (cuda:1)
        cpu_module_indexed = self._make_module(
            model_out_device=torch.device("cuda:1"),
        )

        # Create tensors on cuda:1
        output = {
            "task1-prediction": torch.tensor([1.0, 2.0, 3.0]).to("cuda:1"),
            "task1-label": torch.tensor([0.0, 1.0, 0.0]).to("cuda:1"),
            "task1-weight": torch.tensor([5.0, 1.0, 0.0]).to("cuda:1"),
        }

        # Update with the tensors - this should trigger transfer to CPU
        cpu_module_indexed.update(output)

        # Wait for update to be processed
        wait_until_true(self.mock_metric.update_called)

        # Verify that the tensors received by the metric are on CPU
        # (they should have been transferred from cuda:1 to cpu)
        for predictions in self.mock_metric.predictions_update_calls:
            # pyrefly: ignore[missing-attribute]
            for _, tensor in predictions.items():
                self.assertEqual(
                    tensor.device.type,
                    "cpu",
                    f"Expected tensor on CPU, but got {tensor.device}",
                )

        cpu_module_indexed.shutdown()

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_transfer_tensors_to_cpu_event_on_source_device_stream(self) -> None:
        """
        Regression test for ZORM metric corruption on non-rank-0 processes.

        Bug: transfer_tensors_to_cpu recorded the CUDA event on cuda:0 (default)
        instead of the source tensor's device (cuda:N), making event.synchronize()
        a no-op on non-rank-0 processes.

        Verifies the event tracks cuda:1's stream via event.query(): with the bug,
        the event on idle cuda:0 completes immediately; with the fix, it stays
        pending behind cuda:1's queued work.
        """
        with torch.cuda.device(0):
            source_tensors = {
                "predictions": torch.ones(1024, device="cuda:1") * 3.14,
                "labels": torch.ones(1024, device="cuda:1"),
                "weights": torch.ones(1024, device="cuda:1") * 2.0,
            }

            # Warm the pinned-memory cache and the dedicated DtoH stream first.
            # The first pinned allocation triggers a synchronizing cudaHostAlloc;
            # if that ran during the measured transfer it would drain the busy
            # matmuls and complete the event immediately, masking the cuda:0 bug
            # this test guards against. Warming up lets the measured transfer
            # reuse cached pinned blocks with no host-alloc sync.
            warmup_cpu, warmup_event = transfer_tensors_to_cpu(
                {k: torch.ones_like(v) for k, v in source_tensors.items()}
            )
            if warmup_event is not None:
                warmup_event.synchronize()
            del warmup_cpu, warmup_event
            torch.cuda.synchronize("cuda:1")

            busy = torch.randn(8192, 8192, device="cuda:1")
            for _ in range(50):
                busy = busy @ busy

            cpu_tensors, event = transfer_tensors_to_cpu(source_tensors)
            self.assertIsNotNone(event)
            assert event is not None

            self.assertFalse(
                event.query(),
                "CUDA event completed immediately — it was recorded on the "
                "wrong stream (cuda:0 instead of cuda:1).",
            )

            event.synchronize()
            torch.testing.assert_close(
                cpu_tensors["predictions"],
                torch.ones(1024) * 3.14,
            )
            torch.testing.assert_close(
                cpu_tensors["labels"],
                torch.ones(1024),
            )
            torch.testing.assert_close(
                cpu_tensors["weights"],
                torch.ones(1024) * 2.0,
            )

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_transfer_tensors_to_cpu_source_not_recycled_during_copy(self) -> None:
        """
        Regression test for a cross-stream use-after-free on the SOURCE tensor.

        transfer_tensors_to_cpu copies on a dedicated DtoH stream while the
        source was produced on the default stream. Without record_stream, the
        caching allocator may recycle the source's GPU block as soon as the
        default stream frees it — racing the still-in-flight async copy and
        corrupting the CPU result. record_stream must hold the block until the
        DtoH stream is done.

        We force the race: queue slow producer work so the copy is delayed,
        drop the source, then immediately reallocate the same byte size on the
        default stream and poison it. With the bug, the recycled block's poison
        is read by the copy and lands in the CPU tensor; with the fix, the
        block stays reserved and the CPU tensor holds the original value.
        """
        device = torch.device("cuda:0")
        numel = 4 * 1024 * 1024  # 16 MB block, large enough to get its own slab
        good_value = 3.14
        poison_value = -1.0

        for _ in range(8):
            source_tensors = {
                "predictions": torch.full(
                    (numel,), good_value, device=device, dtype=torch.float32
                ),
            }

            # Delay the DtoH so it is still queued (not yet reading) when we
            # recycle the source block below.
            busy = torch.randn(8192, 8192, device=device)
            for _ in range(30):
                busy = busy @ busy

            cpu_tensors, event = transfer_tensors_to_cpu(source_tensors)
            assert event is not None

            # Drop the source ref and force the allocator to hand the freed
            # block to a default-stream allocation, then overwrite with poison.
            del source_tensors
            recycler = torch.empty(numel, device=device, dtype=torch.float32)
            recycler.fill_(poison_value)

            event.synchronize()
            torch.testing.assert_close(
                cpu_tensors["predictions"],
                torch.full((numel,), good_value),
                msg=(
                    "Source GPU block was recycled and overwritten while the "
                    "DtoH copy was in flight — missing record_stream on the "
                    "source tensor."
                ),
            )
            del recycler, cpu_tensors, event
            torch.cuda.synchronize(device)

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_flush_remaining_work(self) -> None:
        test_queue = queue.Queue()
        metric_update_job = MetricUpdateJob(
            model_out={
                "task1-prediction": torch.tensor([0.5]),
                "task1-label": torch.tensor([0.7]),
                "task1-weight": torch.tensor([1.0]),
            },
            kwargs={},
        )

        test_queue.put(metric_update_job)
        test_queue.put(metric_update_job)

        # pyrefly: ignore[bad-argument-type]
        items_processed = self.cpu_module._flush_remaining_work(test_queue)

        self.assertEqual(items_processed, 2)
        self.assertTrue(test_queue.empty())

    def _run_dtoh_transfer_test(self, use_cuda: bool) -> None:
        offloaded_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=self.initial_states,
        )

        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        offloaded_module = self._make_module(
            model_out_device=device,
            rec_metrics=RecMetricList([offloaded_metric]),
        )

        transfer_call_info: list = []
        original_transfer = transfer_tensors_to_cpu

        def tracking_transfer(
            tensors: dict,
        ) -> tuple:
            transfer_call_info.append(threading.current_thread().name)
            return original_transfer(tensors)

        model_out = {
            "task1-prediction": torch.tensor([0.5, 0.7]),
            "task1-label": torch.tensor([0.0, 1.0]),
            "task1-weight": torch.tensor([1.0, 1.0]),
        }
        if use_cuda:
            model_out = {k: v.to("cuda:0") for k, v in model_out.items()}
            for tensor in model_out.values():
                self.assertEqual(tensor.device.type, "cuda")

        with patch(
            "torchrec.metrics.cpu_offloaded_metric_module.transfer_tensors_to_cpu",
            side_effect=tracking_transfer,
        ):
            offloaded_module.update(model_out)
            wait_until_true(offloaded_metric.update_called)

        if use_cuda:
            self.assertEqual(
                len(transfer_call_info),
                1,
                "transfer_tensors_to_cpu should be called exactly once for CUDA device",
            )
            self.assertEqual(
                transfer_call_info[0],
                "metric_update",
                f"DtoH transfer should happen in 'metric_update' thread, "
                f"but was called from '{transfer_call_info[0]}'",
            )
        else:
            self.assertEqual(
                len(transfer_call_info),
                0,
                "transfer_tensors_to_cpu should NOT be called when device is CPU",
            )

        self.assertTrue(offloaded_metric.predictions_update_calls is not None)
        # pyrefly: ignore[bad-argument-type]
        torch.testing.assert_close(
            offloaded_metric.predictions_update_calls[0],
            {"task1": torch.tensor([0.5, 0.7])},
        )
        # pyrefly: ignore[bad-argument-type]
        torch.testing.assert_close(
            offloaded_metric.labels_update_calls[0],
            {"task1": torch.tensor([0.0, 1.0])},
        )

        offloaded_module.shutdown()

    # pyrefly: ignore
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_dtoh_transfer_in_update_thread_for_cuda_device(self) -> None:
        self._run_dtoh_transfer_test(use_cuda=True)

    def test_no_dtoh_transfer_for_cpu_device(self) -> None:
        self._run_dtoh_transfer_test(use_cuda=False)

    def test_compute_count_increments_once_per_async_compute(self) -> None:
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.7]),
            "task1-weight": torch.tensor([1.0]),
        }

        self.assertEqual(self.cpu_module.compute_count, 0)

        for expected_count in range(1, 4):
            for _ in range(3):
                self.cpu_module.update(model_out)

            deferrable = self.cpu_module.async_compute()
            result_event = threading.Event()
            deferrable.subscribe(callback=lambda _, e=result_event: e.set())
            self.assertTrue(
                result_event.wait(timeout=15.0),
                f"async_compute #{expected_count} did not complete",
            )
            self.assertEqual(self.cpu_module.compute_count, expected_count)

    def test_generate_metric_module_creates_cpu_offloaded_module(self) -> None:
        module_kwargs = {
            "model_out_device": torch.device("cpu"),
            "update_queue_size": 50,
            "compute_queue_size": 75,
        }

        module = generate_metric_module(
            metric_class=CPUOffloadedRecMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            module_kwargs=module_kwargs,
        )

        self.assertIsInstance(module, CPUOffloadedRecMetricModule)
        self.assertEqual(module.update_queue.maxsize, 50)
        self.assertEqual(module.compute_queue.maxsize, 75)

        module.shutdown()

    def test_correctness_counters_after_update_and_compute(self) -> None:
        """Verify correctness counters are incremented through update+compute cycle."""
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.7]),
            "task1-weight": torch.tensor([1.0]),
        }

        num_updates = 3
        for _ in range(num_updates):
            self.cpu_module.update(model_out)

        self.assertEqual(self.cpu_module._total_updates_enqueued, num_updates)

        future = self.cpu_module.async_compute()
        self.assertEqual(self.cpu_module._total_computes_enqueued, 1)

        future.resolve()

        wait_until_true(lambda: self.cpu_module._total_updates_processed == num_updates)
        self.assertEqual(self.cpu_module._total_updates_processed, num_updates)
        self.assertEqual(self.cpu_module._total_computes_processed, 1)
        self.assertEqual(self.cpu_module._update_errors, 0)
        self.assertEqual(self.cpu_module._compute_errors, 0)

    def test_log_event_called_on_init(self) -> None:
        """Verify _log_event is called with INFO during __init__."""
        with patch.object(CPUOffloadedRecMetricModule, "_log_event") as mock_log_event:
            module = self._make_module()

            mock_log_event.assert_called_once_with(
                "init",
                EventType.INFO,
                {
                    "model_out_device": "cpu",
                    "update_queue_size": "100",
                    "compute_queue_size": "100",
                    "requested_update_batch_size": "1",
                    "update_batch_size": "1",
                },
            )
            module.shutdown()

    def test_shutdown_logs_success_event(self) -> None:
        """Verify shutdown logs SUCCESS event with correctness metadata."""
        with patch.object(self.cpu_module, "_log_event") as mock_log_event:
            self.cpu_module.shutdown()

            mock_log_event.assert_called_once()
            args = mock_log_event.call_args
            self.assertEqual(args[0][0], "shutdown")
            self.assertEqual(args[0][1], EventType.SUCCESS)
            metadata = args[0][2]
            self.assertIn("total_updates_enqueued", metadata)
            self.assertIn("total_updates_processed", metadata)
            self.assertIn("update_thread_alive", metadata)

    def test_update_error_counter_incremented_on_thread_exception(self) -> None:
        """Verify _update_errors is incremented when update thread hits an exception."""
        with patch.object(
            self.cpu_module,
            "_process_metric_update_job",
            side_effect=RuntimeError("test error"),
        ):
            self.cpu_module.update(
                {
                    "task1-prediction": torch.tensor([0.5]),
                    "task1-label": torch.tensor([0.7]),
                    "task1-weight": torch.tensor([1.0]),
                }
            )

            self.cpu_module._captured_exception_event.wait(timeout=5.0)
            self.assertEqual(self.cpu_module._update_errors, 1)

    def test_compute_error_counter_incremented_on_thread_exception(self) -> None:
        """Verify _compute_errors is incremented when compute thread hits an exception."""
        with patch.object(
            self.cpu_module,
            "_process_metric_compute_job",
            side_effect=RuntimeError("test error"),
        ):
            self.cpu_module.async_compute()

            self.cpu_module._captured_exception_event.wait(timeout=5.0)
            self.assertEqual(self.cpu_module._compute_errors, 1)

    def test_queue_join_does_not_deadlock_after_processing_failure(self) -> None:
        """Regression: if _do_work skips task_done() on processing failure,
        queue.join() in sync()/wait_until_queue_is_empty() would deadlock.
        The fix wraps processing in try/finally so task_done() always runs.
        """
        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.5]),
            "task1-weight": torch.tensor([1.0]),
        }

        with patch.object(
            self.cpu_module,
            "_process_metric_update_job",
            side_effect=RuntimeError("processing failed"),
        ):
            self.cpu_module._update_rec_metrics(model_out)
            # Wait for the update thread to die
            self.assertTrue(
                self.cpu_module._captured_exception_event.wait(timeout=5.0),
                "update thread did not capture exception",
            )

        # queue.join() has no timeout — run in a thread and detect deadlock via Event
        join_completed = threading.Event()

        def join_queue() -> None:
            self.cpu_module.update_queue.join()
            join_completed.set()

        threading.Thread(target=join_queue, daemon=True).start()
        self.assertTrue(
            join_completed.wait(timeout=5.0),
            "queue.join() deadlocked — task_done() was not called after processing failure",
        )

    def test_compute_job_failure_propagates_to_future(self) -> None:
        """Regression: _process_metric_compute_job failure must set the future's
        exception so DeferrableMetrics.resolve() doesn't block forever."""
        with patch.object(
            self.cpu_module,
            "_process_metric_compute_job",
            side_effect=RuntimeError("compute failed"),
        ):
            deferrable = self.cpu_module.async_compute()

            self.assertTrue(
                self.cpu_module._captured_exception_event.wait(timeout=5.0),
                "compute thread did not capture exception",
            )

            with self.assertRaisesRegex(RuntimeError, "compute failed"):
                deferrable.resolve()

    def test_sync_marker_failure_propagates_to_future(self) -> None:
        """SynchronizationMarker.future is the same future returned via
        DeferrableMetrics from async_compute(). A failure in
        _process_synchronization_marker must surface through resolve()."""
        with patch.object(
            self.cpu_module,
            "_process_synchronization_marker",
            side_effect=RuntimeError("sync marker failed"),
        ):
            deferrable = self.cpu_module.async_compute()

            self.assertTrue(
                self.cpu_module._captured_exception_event.wait(timeout=5.0),
                "update thread did not capture exception",
            )

            with self.assertRaisesRegex(RuntimeError, "sync marker failed"):
                deferrable.resolve()

    def test_enqueue_update_logs_failure_on_queue_full(self) -> None:
        """Verify FAILURE event is logged when update queue is full."""
        cpu_module = self._make_module(update_queue_size=1)

        block_event = threading.Event()

        def controlled_process_job(_: MetricUpdateJob) -> None:
            block_event.wait()

        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.5]),
            "task1-weight": torch.tensor([1.0]),
        }

        with patch.object(
            cpu_module, "_process_metric_update_job", side_effect=controlled_process_job
        ), patch.object(cpu_module, "_log_event") as mock_log_event:
            cpu_module._update_rec_metrics(model_out)
            cpu_module._update_rec_metrics(model_out)

            with self.assertRaises(RecMetricException):
                cpu_module._update_rec_metrics(model_out)

            mock_log_event.assert_called_once()
            args = mock_log_event.call_args
            self.assertEqual(args[0][0], "enqueue_update")
            self.assertEqual(args[0][1], EventType.FAILURE)

            block_event.set()

    def test_enqueue_compute_logs_failure_on_queue_full(self) -> None:
        """Verify FAILURE event is logged when update queue is full during async_compute."""
        cpu_module = self._make_module(update_queue_size=1)

        block_event = threading.Event()
        processing_started = threading.Event()

        def controlled_process_job(_: MetricUpdateJob) -> None:
            processing_started.set()
            block_event.wait()

        model_out = {
            "task1-prediction": torch.tensor([0.5]),
            "task1-label": torch.tensor([0.5]),
            "task1-weight": torch.tensor([1.0]),
        }

        with patch.object(
            cpu_module, "_process_metric_update_job", side_effect=controlled_process_job
        ), patch.object(cpu_module, "_log_event") as mock_log_event:
            # Fill the queue: 1 item being processed + 1 in queue = full
            cpu_module._update_rec_metrics(model_out)
            # Wait for the update thread to pick up the first item before
            # enqueueing the second, otherwise put_nowait races with get.
            processing_started.wait(timeout=5.0)
            cpu_module._update_rec_metrics(model_out)

            with self.assertRaises(RecMetricException):
                cpu_module.async_compute()

            mock_log_event.assert_called_once()
            args = mock_log_event.call_args
            self.assertEqual(args[0][0], "enqueue_compute")
            self.assertEqual(args[0][1], EventType.FAILURE)

            block_event.set()


@skip_if_asan_class
class CPUOffloadedMetricModuleDistributedTest(MultiProcessTestBase):
    """
    Distributed tests comparing CPUOffloadedRecMetricModule with standard RecMetricModule.
    Compare both the state_dict for checkpointing path, and the computed metrics for
    metric_module.update()/compute() path.
    """

    def setUp(self) -> None:
        super().setUp()

        if torch.cuda.device_count() < 2:
            self.skipTest("This test requires at least 2 GPUs")

    def test_cpu_offloaded_vs_standard_metric_module_results(self) -> None:
        """Test that CPUOffloadedRecMetricModule produces identical results to standard RecMetricModule."""
        world_size = 2
        batch_size = 8
        num_batches = 5

        self._run_multi_process_test(
            callable=_compare_metric_results_worker,
            world_size=world_size,
            batch_size=batch_size,
            num_batches=num_batches,
            compare_sync=False,
        )

    def test_cpu_offloaded_vs_standard_sync_workflow(self) -> None:
        """Test that CPUOffloadedRecMetricModule sync workflow produces identical state dicts."""
        world_size = 2
        batch_size = 2
        num_batches = 2

        self._run_multi_process_test(
            callable=_compare_metric_results_worker,
            world_size=world_size,
            batch_size=batch_size,
            num_batches=num_batches,
            compare_sync=True,
        )

    def test_cpu_offloaded_scalability_with_multiple_batches(self) -> None:
        world_size = 2
        batch_size = 16
        num_batches = 20

        self._run_multi_process_test(
            callable=_compare_metric_results_worker,
            world_size=world_size,
            batch_size=batch_size,
            num_batches=num_batches,
            compare_sync=False,
        )

    def test_cpu_offloaded_vs_standard_at_default_k_compute_only(self) -> None:
        world_size = 2
        batch_size = 8
        num_batches = 20

        self._run_multi_process_test(
            callable=_compare_metric_results_worker,
            world_size=world_size,
            batch_size=batch_size,
            num_batches=num_batches,
            compare_sync=False,
            update_batch_size=10,
            compare_per_call=False,
        )


def _compare_metric_results_worker(
    rank: int,
    world_size: int,
    batch_size: int,
    num_batches: int,
    compare_sync: bool,
    update_batch_size: int = 1,
    compare_per_call: bool = True,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    tasks = gen_test_tasks(["task1", "task2"])

    initial_states = {
        "state_1": torch.tensor([1.0]),
        "state_2": torch.tensor([2.0]),
        "state_3": torch.tensor([3.0]),
    }

    standard_metric = MockRecMetric(
        world_size=world_size,
        my_rank=rank,
        batch_size=batch_size,
        tasks=tasks,
        initial_states=initial_states.copy(),
    )

    offloaded_metric = MockRecMetric(
        world_size=world_size,
        my_rank=rank,
        batch_size=batch_size,
        tasks=tasks,
        initial_states=initial_states.copy(),
    )

    standard_module = RecMetricModule(
        batch_size=batch_size,
        world_size=world_size,
        rec_tasks=tasks,
        rec_metrics=RecMetricList([standard_metric]),
    ).to(device)

    cpu_offloaded_module = CPUOffloadedRecMetricModule(
        model_out_device=torch.device("cuda"),
        batch_size=batch_size,
        world_size=world_size,
        rec_tasks=tasks,
        rec_metrics=RecMetricList([offloaded_metric]),
        update_batch_size=update_batch_size,
    ).to(device)

    # Generate same training data for both modules
    torch.manual_seed(42 + rank)  # Ensure deterministic but rank-specific data

    model_outputs = []
    for _ in range(num_batches):
        model_out = {
            "task1-prediction": torch.rand(batch_size).to(device),
            "task1-label": torch.randint(0, 2, (batch_size,)).float().to(device),
            "task1-weight": torch.ones(batch_size).to(device),
            "task2-prediction": torch.rand(batch_size).to(device),
            "task2-label": torch.randint(0, 2, (batch_size,)).float().to(device),
            "task2-weight": torch.ones(batch_size).to(device),
        }
        model_outputs.append(model_out)

    for model_out in model_outputs:
        standard_module.update(model_out)
        cpu_offloaded_module.update(model_out)

    # Checkpointing
    if compare_sync:
        # Sync both modules
        standard_module.sync()
        cpu_offloaded_module.sync()

        standard_state_dict = standard_module.state_dict()
        offloaded_state_dict = cpu_offloaded_module.state_dict()

        assert_tensor_dict_equals(
            actual_states=offloaded_state_dict,
            expected_states=standard_state_dict,
        )

    standard_results = standard_module.compute().resolve()

    future = cpu_offloaded_module.async_compute()

    # Wait for async compute to finish. Compare the input to each update()
    offloaded_results = future.resolve()

    if compare_per_call:
        for (
            offloaded_predictions,
            offloaded_labels,
            offloaded_weights,
            standard_predictions,
            standard_labels,
            standard_weights,
        ) in zip(
            offloaded_metric.predictions_update_calls,
            offloaded_metric.labels_update_calls,
            offloaded_metric.weights_update_calls,
            standard_metric.predictions_update_calls,
            standard_metric.labels_update_calls,
            standard_metric.weights_update_calls,
        ):
            assert_tensor_dict_equals(
                actual_states=offloaded_predictions,
                expected_states=standard_predictions,
            )
            assert_tensor_dict_equals(
                actual_states=offloaded_labels,
                expected_states=standard_labels,
            )
            assert_tensor_dict_equals(
                actual_states=offloaded_weights,
                expected_states=standard_weights,
            )

    # Compare the computed metric results from both modules
    assert_tensor_dict_equals(
        actual_states=offloaded_results,
        expected_states=standard_results,
    )

    cpu_offloaded_module.shutdown()
    dist.destroy_process_group()


@skip_if_asan_class
class WorkerSideBatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world_size = 1
        self.batch_size = 1
        self.tasks = gen_test_tasks(["task1"])
        self.initial_states = create_tensor_states(["cross_entropy_sum"])

        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"

        self.mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=0,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=self.initial_states,
        )
        self.rec_metrics = RecMetricList([self.mock_metric])

        _init_process_group_with_retry("gloo")

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        if hasattr(self, "cpu_module"):
            try:
                self.cpu_module.shutdown()
            except Exception:
                pass

    def _make_module(self, update_batch_size: int) -> CPUOffloadedRecMetricModule:
        self.cpu_module: CPUOffloadedRecMetricModule = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cpu"),
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=self.rec_metrics,
            update_batch_size=update_batch_size,
        )
        return self.cpu_module

    def test_worker_drains_and_merges_into_one_call(self) -> None:
        cpu_module = self._make_module(update_batch_size=4)

        captured_jobs: list[MetricUpdateJob] = []
        original_process = cpu_module._process_metric_update_job

        def gated_process(job: MetricUpdateJob) -> None:
            captured_jobs.append(job)
            original_process(job)

        with patch.object(
            cpu_module,
            "_process_metric_update_job",
            side_effect=gated_process,
        ):
            for _ in range(4):
                cpu_module._update_rec_metrics(
                    {
                        "task1-prediction": torch.tensor([0.5]),
                        "task1-label": torch.tensor([0.5]),
                        "task1-weight": torch.tensor([1.0]),
                    }
                )
            wait_until_true(lambda: cpu_module._total_updates_processed == 4)

        self.assertEqual(cpu_module._total_updates_processed, 4)
        total_merged = sum(j.merged_count for j in captured_jobs)
        self.assertEqual(total_merged, 4)
        self.assertLessEqual(len(captured_jobs), 4)

    def test_marker_mid_batch_flushes_pending_then_processes_marker(self) -> None:
        cpu_module = self._make_module(update_batch_size=10)

        gate = threading.Event()
        original_get = cpu_module.update_queue.get

        def blocked_get(block: bool = True, timeout: Optional[float] = None) -> object:
            gate.wait()
            return original_get(block, timeout)

        process_order: list[str] = []
        original_process_update = cpu_module._process_metric_update_job
        original_process_marker = cpu_module._process_synchronization_marker

        def tracking_update(job: MetricUpdateJob) -> None:
            process_order.append(f"update(merged={job.merged_count})")
            original_process_update(job)

        def tracking_marker(marker: SynchronizationMarker) -> None:
            process_order.append("marker")
            original_process_marker(marker)

        with patch.object(
            cpu_module.update_queue, "get", side_effect=blocked_get
        ), patch.object(
            cpu_module,
            "_process_metric_update_job",
            side_effect=tracking_update,
        ), patch.object(
            cpu_module,
            "_process_synchronization_marker",
            side_effect=tracking_marker,
        ):
            for _ in range(3):
                cpu_module._update_rec_metrics(
                    {
                        "task1-prediction": torch.tensor([0.5]),
                        "task1-label": torch.tensor([0.5]),
                        "task1-weight": torch.tensor([1.0]),
                    }
                )
            cpu_module.async_compute()

            gate.set()

            wait_until_true(
                lambda: cpu_module._total_updates_processed == 3
                and cpu_module._total_computes_enqueued == 1
            )

        self.assertEqual(process_order[0], "update(merged=3)")
        self.assertEqual(process_order[1], "marker")

    def test_bare_marker_first_processed_alone(self) -> None:
        cpu_module = self._make_module(update_batch_size=4)

        future = cpu_module.async_compute()
        future.resolve()

        self.assertEqual(cpu_module._total_updates_processed, 0)
        self.assertEqual(cpu_module._total_computes_enqueued, 1)

    def test_batch_failure_fails_held_marker_future(self) -> None:
        cpu_module = self._make_module(update_batch_size=10)

        gate = threading.Event()
        original_get = cpu_module.update_queue.get

        def blocked_get(block: bool = True, timeout: Optional[float] = None) -> object:
            gate.wait()
            return original_get(block, timeout)

        with patch.object(
            cpu_module.update_queue, "get", side_effect=blocked_get
        ), patch.object(
            cpu_module,
            "_process_metric_update_job",
            side_effect=RuntimeError("merged batch failed"),
        ):
            cpu_module._update_rec_metrics(
                {
                    "task1-prediction": torch.tensor([0.5]),
                    "task1-label": torch.tensor([0.5]),
                    "task1-weight": torch.tensor([1.0]),
                }
            )
            deferrable = cpu_module.async_compute()
            gate.set()

            with self.assertRaisesRegex(
                RecMetricException,
                "MetricUpdateJob batch failed before SynchronizationMarker",
            ):
                deferrable.resolve()

    def test_sync_does_not_deadlock_with_partial_batch(self) -> None:
        cpu_module = self._make_module(update_batch_size=10)

        for _ in range(3):
            cpu_module._update_rec_metrics(
                {
                    "task1-prediction": torch.tensor([0.5]),
                    "task1-label": torch.tensor([0.5]),
                    "task1-weight": torch.tensor([1.0]),
                }
            )

        sync_completed = threading.Event()

        def call_sync() -> None:
            cpu_module.sync()
            sync_completed.set()

        threading.Thread(target=call_sync, daemon=True).start()
        self.assertTrue(
            sync_completed.wait(timeout=5.0),
            "sync() deadlocked when called with a partial K-batch in flight",
        )
        self.assertEqual(cpu_module._total_updates_processed, 3)

    def test_worker_waits_until_batch_size_or_sync_marker(self) -> None:
        cpu_module = self._make_module(update_batch_size=4)

        captured_jobs: list[MetricUpdateJob] = []
        original_process = cpu_module._process_metric_update_job

        def capture(job: MetricUpdateJob) -> None:
            captured_jobs.append(job)
            original_process(job)

        with patch.object(
            cpu_module,
            "_process_metric_update_job",
            side_effect=capture,
        ):
            for _ in range(2):
                cpu_module._update_rec_metrics(
                    {
                        "task1-prediction": torch.tensor([0.5]),
                        "task1-label": torch.tensor([0.5]),
                        "task1-weight": torch.tensor([1.0]),
                    }
                )

            time.sleep(0.5)
            self.assertEqual(len(captured_jobs), 0)

            for _ in range(2):
                cpu_module._update_rec_metrics(
                    {
                        "task1-prediction": torch.tensor([0.5]),
                        "task1-label": torch.tensor([0.5]),
                        "task1-weight": torch.tensor([1.0]),
                    }
                )

            wait_until_true(lambda: cpu_module._total_updates_processed == 4)

            self.assertEqual(len(captured_jobs), 1)
            self.assertEqual(captured_jobs[0].merged_count, 4)

    def test_worker_processes_partial_batch_when_marker_arrives(self) -> None:
        cpu_module = self._make_module(update_batch_size=10)

        captured_jobs: list[MetricUpdateJob] = []
        original_process = cpu_module._process_metric_update_job

        def capture(job: MetricUpdateJob) -> None:
            captured_jobs.append(job)
            original_process(job)

        with patch.object(
            cpu_module,
            "_process_metric_update_job",
            side_effect=capture,
        ):
            for _ in range(3):
                cpu_module._update_rec_metrics(
                    {
                        "task1-prediction": torch.tensor([0.5]),
                        "task1-label": torch.tensor([0.5]),
                        "task1-weight": torch.tensor([1.0]),
                    }
                )
            time.sleep(0.5)
            self.assertEqual(len(captured_jobs), 0)

            cpu_module.async_compute()

            wait_until_true(
                lambda: cpu_module._total_updates_processed == 3
                and cpu_module._total_computes_enqueued == 1
            )

            self.assertEqual(len(captured_jobs), 1)
            self.assertEqual(captured_jobs[0].merged_count, 3)

    def test_shutdown_unblocks_worker_waiting_for_batch(self) -> None:
        cpu_module = self._make_module(update_batch_size=10)

        cpu_module._update_rec_metrics(
            {
                "task1-prediction": torch.tensor([0.5]),
                "task1-label": torch.tensor([0.5]),
                "task1-weight": torch.tensor([1.0]),
            }
        )

        time.sleep(0.2)
        self.assertTrue(cpu_module.update_thread.is_alive())

        cpu_module.shutdown()

        self.assertFalse(cpu_module.update_thread.is_alive())


class MergeUpdateJobsTest(unittest.TestCase):
    def test_single_job_returned_unchanged(self) -> None:
        job = MetricUpdateJob(
            model_out={"label": torch.tensor([[1.0, 2.0]])},
            kwargs={},
        )
        merged = _merge_update_jobs([job])
        self.assertIs(merged, job)

    def test_concat_model_out_along_batch_dim(self) -> None:
        jobs = [
            MetricUpdateJob(
                model_out={
                    "label": torch.tensor([[1.0, 2.0]]),
                    "prediction": torch.tensor([[0.1, 0.2]]),
                },
                kwargs={},
            ),
            MetricUpdateJob(
                model_out={
                    "label": torch.tensor([[3.0, 4.0]]),
                    "prediction": torch.tensor([[0.3, 0.4]]),
                },
                kwargs={},
            ),
        ]
        merged = _merge_update_jobs(jobs)
        self.assertEqual(merged.model_out["label"].shape, (2, 2))
        self.assertEqual(merged.model_out["prediction"].shape, (2, 2))
        self.assertTrue(
            torch.equal(
                merged.model_out["label"],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                merged.model_out["prediction"],
                torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            )
        )

    def test_concat_required_inputs(self) -> None:
        jobs = [
            MetricUpdateJob(
                model_out={"label": torch.tensor([1.0])},
                kwargs={
                    "required_inputs": {"target_tensor": torch.tensor([5.0])},
                    "scalar_kwarg": "preserve_me",
                },
            ),
            MetricUpdateJob(
                model_out={"label": torch.tensor([2.0])},
                kwargs={
                    "required_inputs": {"target_tensor": torch.tensor([6.0])},
                    "scalar_kwarg": "preserve_me",
                },
            ),
        ]
        merged = _merge_update_jobs(jobs)
        self.assertEqual(merged.kwargs["scalar_kwarg"], "preserve_me")
        self.assertTrue(
            torch.equal(
                merged.kwargs["required_inputs"]["target_tensor"],
                torch.tensor([5.0, 6.0]),
            )
        )

    def test_safe_merge_zero_dim_falls_back_to_stack(self) -> None:
        jobs = [
            MetricUpdateJob(
                model_out={"label": torch.tensor(1.0)},
                kwargs={},
            ),
            MetricUpdateJob(
                model_out={"label": torch.tensor(2.0)},
                kwargs={},
            ),
        ]
        merged = _merge_update_jobs(jobs)
        self.assertEqual(merged.model_out["label"].shape, (2,))
        self.assertTrue(
            torch.equal(merged.model_out["label"], torch.tensor([1.0, 2.0]))
        )

    def test_merged_count_sums_input_counts(self) -> None:
        jobs = [
            MetricUpdateJob(
                model_out={"label": torch.tensor([1.0])},
                kwargs={},
                merged_count=3,
            ),
            MetricUpdateJob(
                model_out={"label": torch.tensor([2.0])},
                kwargs={},
                merged_count=5,
            ),
        ]
        merged = _merge_update_jobs(jobs)
        self.assertEqual(merged.merged_count, 8)

    def test_empty_required_inputs_passes_through(self) -> None:
        jobs = [
            MetricUpdateJob(
                model_out={"label": torch.tensor([[1.0]])},
                kwargs={"required_inputs": {}},
            ),
            MetricUpdateJob(
                model_out={"label": torch.tensor([[2.0]])},
                kwargs={"required_inputs": {}},
            ),
        ]
        merged = _merge_update_jobs(jobs)
        self.assertEqual(merged.kwargs.get("required_inputs"), {})


class ForeachCloneTest(unittest.TestCase):
    def test_int_tensors_clone_without_autograd_error(self) -> None:
        d = {
            "labels": torch.tensor([0, 1, 0], dtype=torch.int64),
            "weights": torch.tensor([1.0, 1.0, 1.0], requires_grad=True),
        }
        out = _foreach_clone_dict(d)
        self.assertEqual(out["labels"].dtype, torch.int64)
        self.assertEqual(out["weights"].dtype, torch.float32)
        torch.testing.assert_close(out["labels"], d["labels"])
        torch.testing.assert_close(out["weights"], d["weights"])
        self.assertFalse(out["weights"].requires_grad)

    def test_kwargs_with_int_required_inputs(self) -> None:
        kwargs = {
            "required_inputs": {
                "feature_id": torch.tensor([1, 2, 3], dtype=torch.int32),
            },
            "scale": torch.tensor([0.5], requires_grad=True),
        }
        out = _foreach_clone_kwargs(kwargs)
        self.assertEqual(out["required_inputs"]["feature_id"].dtype, torch.int32)
        torch.testing.assert_close(
            out["required_inputs"]["feature_id"],
            kwargs["required_inputs"]["feature_id"],
        )
        self.assertFalse(out["scale"].requires_grad)

    def test_dict_clone_independent_of_caller_mutation(self) -> None:
        """Reproduces the Pyper metric_update_reorder staleness bug at the helper
        level: caller mutates the source tensor in place after we clone, and the
        clone must still hold the pre-mutation value. Without _foreach_clone the
        snapshot would be a reference and silently observe the new value."""
        src = {
            "predictions": torch.tensor([1.0, 2.0, 3.0]),
            "labels": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
        snapshot = _foreach_clone_dict(src)
        # Caller overwrites the underlying storage (like Pyper's pre-allocated
        # model_out buffer being reused for the next iteration).
        src["predictions"].zero_()
        src["labels"].fill_(7)
        torch.testing.assert_close(
            snapshot["predictions"], torch.tensor([1.0, 2.0, 3.0])
        )
        torch.testing.assert_close(
            snapshot["labels"], torch.tensor([0, 1, 0], dtype=torch.int64)
        )


class LoadStateDictDevicePinTest(unittest.TestCase):
    """`load_state_dict` post-hook re-pins metric state to CPU after restore."""

    def setUp(self) -> None:
        self.tasks = gen_test_tasks(["task1"])
        self.initial_states = create_tensor_states(["cross_entropy_sum"])
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        self.mock_metric = MockRecMetric(
            world_size=1,
            my_rank=0,
            batch_size=1,
            tasks=self.tasks,
            initial_states=self.initial_states,
        )
        self.rec_metrics = RecMetricList([self.mock_metric])
        _init_process_group_with_retry("gloo")

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        if hasattr(self, "cpu_module"):
            try:
                self.cpu_module.shutdown()
            except RecMetricException as e:
                # shutdown() raises RecMetricException on thread-stuck / captured-exception paths;
                # tearDown must not propagate (would mask the real test failure), but log so it's visible.
                logging.warning(
                    "CPUOffloadedRecMetricModule.shutdown() failed in tearDown: %s", e
                )

    def _make_module(self) -> CPUOffloadedRecMetricModule:
        self.cpu_module = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cpu"),
            batch_size=1,
            world_size=1,
            rec_tasks=self.tasks,
            rec_metrics=self.rec_metrics,
        )
        return self.cpu_module

    def _get_first_state(self, module: CPUOffloadedRecMetricModule) -> torch.Tensor:
        # pyrefly: ignore[bad-index]
        first_comp = module.rec_metrics.rec_metrics[0]._metrics_computations[
            0
        ]  # pyre-ignore[16]
        for attr_name in first_comp._reductions:  # pyre-ignore[16]
            state = getattr(first_comp, attr_name, None)
            if isinstance(state, torch.Tensor):
                return state
        self.fail("No tensor state found")

    @unittest.skipIf(torch.cuda.device_count() < 1, "needs GPU")
    def test_post_hook_moves_cuda_state_in_reductions_to_cpu(self) -> None:
        cpu_module = self._make_module()
        # pyrefly: ignore[bad-index]
        first_comp = cpu_module.rec_metrics.rec_metrics[0]._metrics_computations[
            0
        ]  # pyre-ignore[16]
        attr_name = next(iter(first_comp._reductions))  # pyre-ignore[16]
        with torch.no_grad():
            # Citrine C7: prefer .to("cuda") over deprecated .cuda()
            setattr(first_comp, attr_name, getattr(first_comp, attr_name).to("cuda"))
        self.assertEqual(getattr(first_comp, attr_name).device.type, "cuda")
        CPUOffloadedRecMetricModule._move_state_to_cpu_after_load(cpu_module, None)
        self.assertEqual(getattr(first_comp, attr_name).device.type, "cpu")

    def test_post_hook_no_op_when_already_cpu(self) -> None:
        cpu_module = self._make_module()
        before = self._get_first_state(cpu_module).clone()
        CPUOffloadedRecMetricModule._move_state_to_cpu_after_load(cpu_module, None)
        after = self._get_first_state(cpu_module)
        self.assertEqual(after.device.type, "cpu")
        torch.testing.assert_close(before, after)

    def _deposit_cuda_state_out_of_band(
        self, cpu_module: CPUOffloadedRecMetricModule
    ) -> tuple[Any, str]:
        """Write cuda state via setattr, bypassing load_state_dict."""
        # pyrefly: ignore[bad-index]
        first_comp = cpu_module.rec_metrics.rec_metrics[0]._metrics_computations[
            0
        ]  # pyre-ignore[16]
        attr_name = next(iter(first_comp._reductions))  # pyre-ignore[16]
        with torch.no_grad():
            setattr(first_comp, attr_name, getattr(first_comp, attr_name).to("cuda"))
        self.assertEqual(getattr(first_comp, attr_name).device.type, "cuda")
        return first_comp, attr_name

    @unittest.skipIf(torch.cuda.device_count() < 1, "needs GPU")
    def test_out_of_band_cuda_state_mismatches_without_repin(self) -> None:
        """Out-of-band cuda state device-mismatches on a CPU-side add."""
        cpu_module = self._make_module()
        first_comp, attr_name = self._deposit_cuda_state_out_of_band(cpu_module)
        state = getattr(first_comp, attr_name)
        with self.assertRaisesRegex(RuntimeError, "two devices|same device"):
            state.add_(torch.zeros(state.shape, dtype=state.dtype))

    @unittest.skipIf(torch.cuda.device_count() < 1, "needs GPU")
    def test_repin_helper_recovers_out_of_band_cuda_state(self) -> None:
        """The re-pin helper recovers cuda state the post-hook can't see."""
        cpu_module = self._make_module()
        first_comp, attr_name = self._deposit_cuda_state_out_of_band(cpu_module)
        moved = cpu_module._repin_metric_state_to_cpu()
        self.assertGreaterEqual(moved, 1)
        self.assertEqual(getattr(first_comp, attr_name).device.type, "cpu")

    @unittest.skipIf(torch.cuda.device_count() < 1, "needs GPU")
    def test_first_update_repins_out_of_band_cuda_state(self) -> None:
        """First update re-pins out-of-band cuda state to CPU."""
        cpu_module = self._make_module()
        first_comp, attr_name = self._deposit_cuda_state_out_of_band(cpu_module)
        self.assertFalse(cpu_module._metric_state_repinned)

        model_out = {
            "task1-prediction": torch.tensor([0.5, 0.7]),
            "task1-label": torch.tensor([0.0, 1.0]),
            "task1-weight": torch.tensor([1.0, 1.0]),
        }
        cpu_module._process_metric_update_job(MetricUpdateJob(model_out, {}, 1))

        self.assertTrue(cpu_module._metric_state_repinned)
        self.assertEqual(getattr(first_comp, attr_name).device.type, "cpu")
        self.assertTrue(self.mock_metric.update_called())


_MERGE_TEST_METRICS: list[tuple[str, RecMetricEnum]] = [
    ("NE", RecMetricEnum.NE),
    ("CTR", RecMetricEnum.CTR),
    ("CALIBRATION", RecMetricEnum.CALIBRATION),
    ("MSE", RecMetricEnum.MSE),
    ("MAE", RecMetricEnum.MAE),
    ("AUC", RecMetricEnum.AUC),
    ("TOWER_QPS", RecMetricEnum.TOWER_QPS),
]


class WindowOverEvictionMergeTest(unittest.TestCase):
    """Repro + generalization + fix-verification for the ZORM window-metric NaN.

    Root cause: CPUOffloadedRecMetricModule micro-batches K MetricUpdateJobs into
    ONE rec_metrics.update() call (``_merge_update_jobs`` -> ``_safe_merge_tensors``,
    cat dim=0). Each windowed sum-reduction metric feeds its ``WindowBuffer`` ONE
    entry per update() call with ``size = num_samples`` (the batch length). The
    buffer evicts by cumulative size: ``while _window_used_size > _max_size:
    remove()`` where ``_max_size = ceil(window_size / world_size)`` (per-rank).

    When ``K * per_rank_batch > _max_size`` the single merged entry is evicted the
    instant it is appended -> the window state empties -> the windowed metric
    divides 0/0 -> NaN (NE/CTR/CALIBRATION) or masked-to-0 WRONG (MSE/MAE).
    Lifetime metrics never evict, so they stay valid. AUC self-truncates its raw
    window and TOWER_QPS has no WindowBuffer, so both stay valid.

    The init guard at rec_metric.py:472 (``_window_size < _batch_size``) checks the
    UNMERGED batch, so it passes; the MERGED entry (K*batch) still over-evicts.

    Fix (cap-K): CPUOffloadedRecMetricModule caps update_batch_size at
    construction to floor(min_per_rank_window / batch_size) so a merged entry
    can never exceed the smallest per-rank window (see
    _capped_update_batch_size). Here per_rank_window=40, batch=8 -> K is capped
    10->5, so a merged entry (5*8=40) fits the window exactly and never
    over-evicts. On prod configs the window dwarfs K*batch, so K stays 10.

    Tests here drive the REAL CPUOffloadedRecMetricModule on CPU (model_out_device
    = cpu, no GPU) with update_batch_size=K so the fix is exercised end-to-end
    through the worker thread. The reference is the same module run with K=1;
    the capped-K window must match the K=1 window within tolerance.
    """

    K: int = 10
    BATCH_SIZE: int = 8
    # Per-rank window (world_size=1 => _max_size == WINDOW_SIZE) chosen so that
    # BATCH_SIZE <= WINDOW_SIZE < K * BATCH_SIZE: the guard passes (40 >= 8),
    # K=1 keeps a valid window, but one merged entry of K*BATCH_SIZE=80 over-evicts.
    WINDOW_SIZE: int = 40

    def setUp(self) -> None:
        self.tasks = gen_test_tasks(["task1", "task2"])
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        _init_process_group_with_retry("gloo")
        self._modules: list[CPUOffloadedRecMetricModule] = []

    def tearDown(self) -> None:
        for module in self._modules:
            try:
                module.shutdown()
            except RecMetricException as e:
                # Must not propagate from tearDown (would mask the real failure); log so it stays visible.
                logging.warning("shutdown() failed in tearDown: %s", e)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _build_config(self, metric_enum: RecMetricEnum) -> MetricsConfig:
        return MetricsConfig(
            rec_tasks=self.tasks,
            rec_metrics={
                metric_enum: RecMetricDef(
                    rec_tasks=self.tasks, window_size=self.WINDOW_SIZE
                )
            },
            rec_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            compute_interval_steps=1,
        )

    def _build_offloaded_module(
        self, metric_enum: RecMetricEnum, update_batch_size: int
    ) -> CPUOffloadedRecMetricModule:
        module = cast(
            CPUOffloadedRecMetricModule,
            generate_metric_module(
                metric_class=CPUOffloadedRecMetricModule,
                metrics_config=self._build_config(metric_enum),
                batch_size=self.BATCH_SIZE,
                world_size=1,
                my_rank=0,
                state_metrics_mapping={},
                device=torch.device("cpu"),
                process_group=dist.group.WORLD,
                module_kwargs={
                    "model_out_device": torch.device("cpu"),
                    "update_batch_size": update_batch_size,
                },
            ),
        )
        self._modules.append(module)
        return module

    def _gen_batches(self) -> list[dict[str, torch.Tensor]]:
        torch.manual_seed(1234)
        batches: list[dict[str, torch.Tensor]] = []
        for _ in range(self.K):
            model_out: dict[str, torch.Tensor] = {}
            for task in self.tasks:
                model_out[task.prediction_name] = torch.rand(self.BATCH_SIZE)
                model_out[task.label_name] = torch.randint(
                    0, 2, (self.BATCH_SIZE,)
                ).float()
                model_out[task.weight_name] = torch.rand(self.BATCH_SIZE) + 0.5
            batches.append(model_out)
        return batches

    def _run_offloaded(
        self, metric_enum: RecMetricEnum, update_batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Feed K batches through the real worker thread and resolve one compute.
        Returns the windowed metric values."""
        module = self._build_offloaded_module(metric_enum, update_batch_size)
        for b in self._gen_batches():
            module.update(b)
        result = module.async_compute().resolve()
        return {
            k: cast(torch.Tensor, v)
            for k, v in result.items()
            if "window" in k.lower() and isinstance(v, torch.Tensor)
        }

    @staticmethod
    def _classify(
        reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
    ) -> str:
        """NAN | WRONG | VALID for the K-merge candidate vs the K=1 reference."""
        keys = set(reference) & set(candidate)
        if not keys:
            return "VALID"  # metric has no windowed sum-reduction state (TOWER_QPS)
        if any(torch.isnan(candidate[k]).any().item() for k in keys):
            return "NAN"
        all_close = all(
            torch.allclose(
                reference[k].double(), candidate[k].double(), rtol=1e-4, atol=1e-6
            )
            for k in keys
        )
        return "VALID" if all_close else "WRONG"

    # ---- STEP 1: focused NE repro through the real ZORM worker ----
    def test_step1_ne_window_nan_repro(self) -> None:
        # Guard is evaluated against the UNMERGED batch; it must not trip.
        self.assertGreaterEqual(self.WINDOW_SIZE, self.BATCH_SIZE)
        reference = self._run_offloaded(RecMetricEnum.NE, update_batch_size=1)
        merged = self._run_offloaded(RecMetricEnum.NE, update_batch_size=self.K)
        print("\n===== STEP 1 NE OVER-EVICTION REPRO (real ZORM worker) =====")
        for k in sorted(set(reference) & set(merged)):
            print(
                f"  {k}: K=1={reference[k].tolist()} "
                f"K={self.K}={merged[k].tolist()}"
            )
        classification = self._classify(reference, merged)
        print(f"  classification: {classification}")
        print("============================================================")
        self.assertEqual(
            classification,
            "VALID",
            "NE window under K-merge must equal the K=1 reference. On unfixed "
            "code the merged entry over-evicts the WindowBuffer -> window_ne=NaN.",
        )

    # ---- STEP 2 + 3: generalized over ALL RecMetric types ----
    def test_all_metrics_window_valid_under_kmerge(self) -> None:
        print("\n===== STEP 2/3 OVER-EVICTION TABLE (real ZORM worker) =====")
        print(
            f"  K={self.K} BATCH_SIZE={self.BATCH_SIZE} "
            f"WINDOW_SIZE(per-rank)={self.WINDOW_SIZE} "
            f"merged_entry_size={self.K * self.BATCH_SIZE}"
        )
        results: dict[str, str] = {}
        for name, metric_enum in _MERGE_TEST_METRICS:
            reference = self._run_offloaded(metric_enum, update_batch_size=1)
            merged = self._run_offloaded(metric_enum, update_batch_size=self.K)
            results[name] = self._classify(reference, merged)
            print(f"  {name:12s}: {results[name]}")
        print("===========================================================")
        bad = {n: c for n, c in results.items() if c != "VALID"}
        self.assertEqual(
            bad,
            {},
            f"Metrics with broken window under K-merge over-eviction: {bad}. "
            "cap-K must keep every metric's window within tolerance of K=1.",
        )


class CapUpdateBatchSizeTest(unittest.TestCase):
    """Directly asserts the cap-K policy: update_batch_size stays at the
    requested value on prod-shaped configs (window >> K*batch) and is lowered
    on dlrm-shaped configs (small NE window relative to K*batch)."""

    def setUp(self) -> None:
        self.tasks = gen_test_tasks(["task1"])
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        _init_process_group_with_retry("gloo")
        self._modules: list[CPUOffloadedRecMetricModule] = []

    def tearDown(self) -> None:
        for module in self._modules:
            try:
                module.shutdown()
            except RecMetricException as e:
                # Must not propagate from tearDown (would mask the real failure); log so it stays visible.
                logging.warning("shutdown() failed in tearDown: %s", e)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _effective_k(
        self, window_size: int, batch_size: int, world_size: int, requested_k: int
    ) -> int:
        config = MetricsConfig(
            rec_tasks=self.tasks,
            rec_metrics={
                RecMetricEnum.NE: RecMetricDef(
                    rec_tasks=self.tasks, window_size=window_size
                )
            },
            rec_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            compute_interval_steps=1,
        )
        module = cast(
            CPUOffloadedRecMetricModule,
            generate_metric_module(
                metric_class=CPUOffloadedRecMetricModule,
                metrics_config=config,
                batch_size=batch_size,
                world_size=world_size,
                my_rank=0,
                state_metrics_mapping={},
                device=torch.device("cpu"),
                process_group=dist.group.WORLD,
                module_kwargs={
                    "model_out_device": torch.device("cpu"),
                    "update_batch_size": requested_k,
                },
            ),
        )
        self._modules.append(module)
        return module._update_batch_size

    def test_prod_settings_k_uncapped(self) -> None:
        # NE window 1,000,000 / world 16 => per-rank 62,500; batch 2,048 =>
        # floor(62500/2048)=30 >= 10 => K stays 10.
        self.assertEqual(
            self._effective_k(
                window_size=1_000_000, batch_size=2048, world_size=16, requested_k=10
            ),
            10,
        )

    def test_dlrm_settings_k_capped(self) -> None:
        # NE window 100,000 / world 16 => per-rank 6,250; batch 1,280 =>
        # floor(6250/1280)=4 < 10 => K capped to 4.
        self.assertEqual(
            self._effective_k(
                window_size=100_000, batch_size=1280, world_size=16, requested_k=10
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
