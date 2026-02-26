#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import queue
import threading
import time
import unittest
from typing import Callable, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.metrics.cpu_offloaded_metric_module import (
    CPUOffloadedRecMetricModule,
    MetricUpdateJob,
)
from torchrec.metrics.metric_module import RecMetricModule
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
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"

        self.mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=self.initial_states,
        )
        self.rec_metrics = RecMetricList([self.mock_metric])

        dist.init_process_group("gloo")
        self.cpu_module: CPUOffloadedRecMetricModule = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cpu"),
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=self.rec_metrics,
            throughput_metric=ThroughputMetric(
                world_size=self.world_size,
                batch_size=self.batch_size,
                window_seconds=1,
            ),
        )

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

        cpu_output, transfer_event = self.cpu_module._transfer_to_cpu(output)
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
        cpu_module = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cuda"),
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=self.rec_metrics,
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
            RecMetricException, "metric processor thread is shut down.", future.result
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
        offloaded_module = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cuda"),
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
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
        offloaded_module = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cuda"),
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
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
        cpu_module_indexed = CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cuda:1"),
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=self.rec_metrics,
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
        offloaded_module = CPUOffloadedRecMetricModule(
            model_out_device=device,
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=RecMetricList([offloaded_metric]),
        )

        transfer_call_info: list = []
        original_transfer_to_cpu = offloaded_module._transfer_to_cpu

        def tracking_transfer_to_cpu(model_out: dict) -> tuple:
            transfer_call_info.append(threading.current_thread().name)
            return original_transfer_to_cpu(model_out)

        model_out = {
            "task1-prediction": torch.tensor([0.5, 0.7]),
            "task1-label": torch.tensor([0.0, 1.0]),
            "task1-weight": torch.tensor([1.0, 1.0]),
        }
        if use_cuda:
            model_out = {k: v.to("cuda:0") for k, v in model_out.items()}
            for tensor in model_out.values():
                self.assertEqual(tensor.device.type, "cuda")

        with patch.object(
            offloaded_module,
            "_transfer_to_cpu",
            side_effect=tracking_transfer_to_cpu,
        ):
            offloaded_module.update(model_out)
            wait_until_true(offloaded_metric.update_called)

        if use_cuda:
            self.assertEqual(
                len(transfer_call_info),
                1,
                "_transfer_to_cpu should be called exactly once for CUDA device",
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
                "_transfer_to_cpu should NOT be called when device is CPU",
            )

        self.assertTrue(offloaded_metric.predictions_update_calls is not None)
        # pyre-ignore[6]: predictions_update_calls contains Dict[str, Tensor]
        torch.testing.assert_close(
            offloaded_metric.predictions_update_calls[0],
            {"task1": torch.tensor([0.5, 0.7])},
        )
        # pyre-ignore[6]: labels_update_calls contains Dict[str, Tensor]
        torch.testing.assert_close(
            offloaded_metric.labels_update_calls[0],
            {"task1": torch.tensor([0.0, 1.0])},
        )

        offloaded_module.shutdown()

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_dtoh_transfer_in_update_thread_for_cuda_device(self) -> None:
        self._run_dtoh_transfer_test(use_cuda=True)

    def test_no_dtoh_transfer_for_cpu_device(self) -> None:
        self._run_dtoh_transfer_test(use_cuda=False)


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


def _compare_metric_results_worker(
    rank: int,
    world_size: int,
    batch_size: int,
    num_batches: int,
    compare_sync: bool,
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

    standard_results = standard_module.compute()

    future = cpu_offloaded_module.async_compute()

    # Wait for async compute to finish. Compare the input to each update()
    offloaded_results = future.result(timeout=10.0)
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


if __name__ == "__main__":
    unittest.main()
