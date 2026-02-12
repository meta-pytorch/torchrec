#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from collections import OrderedDict
from typing import Any, Dict
from unittest.mock import Mock, patch

import torch
from torchrec.metrics.metrics_config import BatchSizeStage
from torchrec.metrics.throughput import ThroughputMetric


THROUGHPUT_PATH = "torchrec.metrics.throughput"


class ThroughputMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world_size = 64
        self.batch_size = 256

    @patch(THROUGHPUT_PATH + ".time.monotonic")
    def test_no_batches(self, time_mock: Mock) -> None:
        time_mock.return_value = 1
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size, world_size=self.world_size, window_seconds=100
        )
        self.assertEqual(
            throughput_metric.compute(),
            {
                "throughput-throughput|total_examples": 0,
                "throughput-throughput|attempt_examples": 0,
            },
        )

    @patch(THROUGHPUT_PATH + ".time.monotonic")
    def test_one_batch(self, time_mock: Mock) -> None:
        time_mock.return_value = 1
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size, world_size=self.world_size, window_seconds=100
        )
        throughput_metric.update()
        self.assertEqual(
            throughput_metric.compute(),
            {
                "throughput-throughput|total_examples": self.batch_size
                * self.world_size,
                "throughput-throughput|attempt_examples": self.batch_size
                * self.world_size,
            },
        )

    @patch(THROUGHPUT_PATH + ".time.monotonic")
    def _test_throughput(self, time_mock: Mock, warmup_steps: int) -> None:
        update_timestamps = [10, 11, 12, 14, 15, 17, 18, 20, 21, 22, 25, 29, 30]
        update_timestamps.sort()
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=10,
            warmup_steps=warmup_steps,
        )

        window_time_lapse_buffer = []
        window_time_lapse = 0
        for i in range(len(update_timestamps)):
            time_mock.return_value = update_timestamps[i]
            throughput_metric.update()
            if i >= warmup_steps:
                window_time_lapse_buffer.append(
                    update_timestamps[i] - update_timestamps[i - 1]
                )
                window_time_lapse += update_timestamps[i] - update_timestamps[i - 1]
            ret = throughput_metric.compute()

            total_examples = self.world_size * self.batch_size * (i + 1)
            if i < warmup_steps:
                self.assertEqual(
                    ret,
                    {
                        "throughput-throughput|total_examples": total_examples,
                        "throughput-throughput|attempt_examples": total_examples,
                    },
                )
                continue

            lifetime_examples = total_examples - (
                self.world_size * self.batch_size * warmup_steps
            )
            lifetime_throughput = lifetime_examples / (
                update_timestamps[i] - update_timestamps[warmup_steps - 1]
            )

            while window_time_lapse > 10:
                window_time_lapse -= window_time_lapse_buffer.pop(0)
            window_throughput = (
                len(window_time_lapse_buffer)
                * self.world_size
                * self.batch_size
                / window_time_lapse
            )

            self.assertEqual(
                ret["throughput-throughput|lifetime_throughput"], lifetime_throughput
            )
            self.assertEqual(
                ret["throughput-throughput|window_throughput"], window_throughput
            )
            self.assertEqual(
                ret["throughput-throughput|total_examples"], total_examples
            )
            # only one attempt so attempt examples and throughput are the same as total/lifetime
            self.assertEqual(
                ret["throughput-throughput|attempt_examples"], total_examples
            )
            self.assertEqual(
                ret["throughput-throughput|attempt_throughput"], lifetime_throughput
            )

    def test_throughput_warmup_steps_0(self) -> None:
        with self.assertRaises(ValueError):
            self._test_throughput(warmup_steps=0)

    def test_throughput_warmup_steps_1(self) -> None:
        self._test_throughput(warmup_steps=1)

    def test_throughput_warmup_steps_2(self) -> None:
        self._test_throughput(warmup_steps=2)

    def test_throughput_warmup_steps_10(self) -> None:
        self._test_throughput(warmup_steps=10)

    def test_warmup_checkpointing(self) -> None:
        warmup_steps = 5
        extra_steps = 2
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=10,
            warmup_steps=warmup_steps,
        )
        for i in range(5):
            for _ in range(warmup_steps + extra_steps):
                throughput_metric.update()
            self.assertEqual(
                throughput_metric.warmup_examples.item(),
                warmup_steps * (i + 1) * self.batch_size * self.world_size,
            )
            self.assertEqual(
                throughput_metric.total_examples.item(),
                (warmup_steps + extra_steps)
                * (i + 1)
                * self.batch_size
                * self.world_size,
            )

            self.assertEqual(
                throughput_metric.attempt_warmup_examples.item(),
                warmup_steps * self.batch_size * self.world_size,
            )
            self.assertEqual(
                throughput_metric.attempt_examples.item(),
                (warmup_steps + extra_steps) * self.batch_size * self.world_size,
            )
            # Mimic trainer crashing and loading a checkpoint
            throughput_metric._steps = 0
            throughput_metric.attempt_examples = torch.tensor(0, dtype=torch.long)
            throughput_metric.attempt_warmup_examples = torch.tensor(
                0, dtype=torch.long
            )
            throughput_metric.attempt_time_lapse_after_warmup = torch.tensor(
                0, dtype=torch.double
            )

    @patch(THROUGHPUT_PATH + ".time.monotonic")
    def test_batch_size_schedule(self, time_mock: Mock) -> None:
        batch_size_stages = [BatchSizeStage(256, 1), BatchSizeStage(512, None)]
        time_mock.return_value = 1
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=100,
            batch_size_stages=batch_size_stages,
        )

        total_examples = 0
        throughput_metric.update()
        total_examples += batch_size_stages[0].batch_size * self.world_size
        self.assertEqual(
            throughput_metric.compute(),
            {
                "throughput-throughput|total_examples": total_examples,
                "throughput-throughput|attempt_examples": total_examples,
                "throughput-throughput|batch_size": 256,
            },
        )

        throughput_metric.update()
        total_examples += batch_size_stages[1].batch_size * self.world_size
        self.assertEqual(
            throughput_metric.compute(),
            {
                "throughput-throughput|total_examples": total_examples,
                "throughput-throughput|attempt_examples": total_examples,
                "throughput-throughput|batch_size": 512,
            },
        )

    def test_num_batch_without_batch_size_stages(self) -> None:
        # Create the module without the batch_size_stages
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=100,
            batch_size_stages=None,
        )

        # Make sure num_batch is not present as an argument of the class
        self.assertFalse(hasattr(throughput_metric, "num_batch"))

        throughput_metric.update()
        state_dict: Dict[str, Any] = throughput_metric.state_dict()
        # Ensure num_batch is not included in the state_dict for the module without batch_size_stages
        self.assertNotIn("num_batch", state_dict)

    def test_state_dict_load_module_lifecycle(self) -> None:
        """
        A test to ensure that the load_state_dict and state_dict hooks correctly handle the num_batch attribute
        through the module lifecycle.
        """

        throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )

        self.assertTrue(hasattr(throughput_metric, "_num_batch"))

        # Stage 1: create metric and update the state_dict before persisting it
        # Update metric, expecting num_batch to be incremented to 1
        throughput_metric.update()
        # Ensure num_batch is 1
        self.assertEqual(throughput_metric._num_batch, 1)
        # Ensure num_batch is included in the state_dict and has the correct value
        state_dict: Dict[str, Any] = throughput_metric.state_dict()
        self.assertIn("num_batch", state_dict)
        # Ensure num_batch was saved to state_dict with the correct value
        self.assertEqual(state_dict["num_batch"].item(), throughput_metric._num_batch)

        # Stage 2: load the state_dict and ensure num_batch is loaded correctly

        # Create a new metric instance
        new_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )
        # Ensure num_batch is 0
        self.assertEqual(new_throughput_metric._num_batch, 0)
        # Load the state_dict
        new_throughput_metric.load_state_dict(state_dict)
        # Ensure num_batch is loaded from the state_dict with the correct value
        self.assertEqual(new_throughput_metric._num_batch, 1)

        # Stage 3: update the metric after loading the state and resave the state_dict

        # Save the state_dict
        state_dict = new_throughput_metric.state_dict()
        # Ensure num_batch is included in the state_dict
        self.assertIn("num_batch", state_dict)
        # Ensure num_batch was saved to state_dict with the correct value
        self.assertEqual(
            state_dict["num_batch"].item(), new_throughput_metric._num_batch
        )

    def test_state_dict_hook_adds_key(self) -> None:
        """
        Ensures that the state_dict_hook adds the 'num_batch' key to the state_dict
        when batch_size_stages is True.
        """
        throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )
        for _ in range(5):
            throughput_metric.update()
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        prefix: str = "test_prefix_"
        ThroughputMetric.state_dict_hook(throughput_metric, state_dict, prefix, {})
        self.assertIn(f"{prefix}num_batch", state_dict)
        self.assertEqual(state_dict[f"{prefix}num_batch"].item(), 5)

    def test_state_dict_hook_no_batch_size_stages(self) -> None:
        """
        Verifies that the state_dict_hook does not add the 'num_batch' key when
        batch_size_stages is None.
        """
        # Hook-only test
        throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=None,
        )
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        prefix: str = "test_prefix_"
        ThroughputMetric.state_dict_hook(throughput_metric, state_dict, prefix, {})
        self.assertNotIn(f"{prefix}num_batch", state_dict)

        # Lifecycle test

        num_updates = 10
        prev_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=None,
        )
        for _ in range(num_updates):
            prev_job_throughput_metric.update()
        prev_state_dict = prev_job_throughput_metric.state_dict()

        curr_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=None,
        )

        curr_job_throughput_metric.load_state_dict(prev_state_dict)
        # Make sure _num_batch is not present as an argument of the class
        self.assertFalse(hasattr(curr_job_throughput_metric, "_num_batch"))

    def test_load_state_dict_hook_resumes_from_checkpoint_with_bss_from_bss(
        self,
    ) -> None:
        """
        Checks that the load_state_dict_hook correctly restores the 'num_batch' value
        from the state_dict.
        """
        num_updates = 10
        prev_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )
        for _ in range(num_updates):
            prev_job_throughput_metric.update()
        prev_state_dict = prev_job_throughput_metric.state_dict()

        curr_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(1024, 1), BatchSizeStage(2048, None)],
        )

        curr_job_throughput_metric.load_state_dict(prev_state_dict)
        self.assertEqual(curr_job_throughput_metric._num_batch, num_updates)

    def test_load_state_dict_hook_resumes_from_checkpoint_without_bss(self) -> None:
        """
        Verifies that the load_state_dict_hook correctly handles the case where a
        previously checkpointed job used the batch_size_stages, but a subsequent job,
        restored from a checkpoint, isn't using them.
        """

        prev_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )

        prev_state_dict = prev_job_throughput_metric.state_dict()

        curr_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=None,  # No batch_size_stages
        )

        curr_job_throughput_metric.load_state_dict(prev_state_dict)

        self.assertFalse(hasattr(curr_job_throughput_metric, "_num_batch"))

    def test_load_state_dict_hook_resumes_from_checkpoint_with_bss_without_key(
        self,
    ) -> None:
        """
        Verifies that the load_state_dict_hook correctly handles the case where a
        previously checkpointed job didn't use batch_size_stages, but a subsequent job,
        restored from a checkpoint, is using them.
        """
        prev_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=None,  # No batch_size_stages
        )
        prev_state_dict = prev_job_throughput_metric.state_dict()

        curr_job_throughput_metric = ThroughputMetric(
            batch_size=32,
            world_size=4,
            window_seconds=100,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )

        curr_job_throughput_metric.load_state_dict(prev_state_dict)

        # Expecting 0
        self.assertEqual(curr_job_throughput_metric._num_batch, 0)
