#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any

import torch
from torchrec.metrics.metrics_config import BatchSizeStage, DefaultTaskInfo, RecTaskInfo
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import gen_test_batch, gen_test_tasks


_CUDA_UNAVAILABLE: bool = not torch.cuda.is_available()


class RecMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        # Create testing labels, predictions and weights
        model_output = gen_test_batch(128)
        self.labels, self.predictions, self.weights, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )

    def test_optional_weights(self) -> None:
        ne1 = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        ne2 = NEMetric(
            world_size=1,
            my_rank=1,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )

        default_weights = {
            k: torch.ones_like(self.labels[k]) for k in self.weights.keys()
        }
        ne1.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=default_weights,
        )
        ne2.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=None,
        )
        ne1 = ne1._metrics_computations[0]
        ne2 = ne2._metrics_computations[0]
        self.assertTrue(ne1.cross_entropy_sum == ne2.cross_entropy_sum)
        self.assertTrue(ne1.weighted_num_samples == ne2.weighted_num_samples)
        self.assertTrue(ne1.pos_labels == ne2.pos_labels)
        self.assertTrue(ne1.neg_labels == ne2.neg_labels)

    def test_zero_weights(self) -> None:
        # Test if weights = 0 for an update
        mse = MSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
            should_validate_update=True,
        )
        mse_computation = mse._metrics_computations[0]

        zero_weights = {
            k: torch.zeros_like(self.weights[k]) for k in self.weights.keys()
        }
        mse.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=zero_weights,
        )
        self.assertEqual(mse_computation.error_sum, torch.tensor(0.0))
        self.assertEqual(mse_computation.weighted_num_samples, torch.tensor(0.0))

        res = mse.compute()
        self.assertEqual(res["mse-DefaultTask|lifetime_mse"], torch.tensor(0.0))
        self.assertEqual(res["mse-DefaultTask|lifetime_rmse"], torch.tensor(0.0))

        mse.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        #  got `Tensor`.
        # pyrefly: ignore[no-matching-overload]
        self.assertGreater(mse_computation.error_sum, torch.tensor(0.0))
        #  got `Tensor`.
        # pyrefly: ignore[no-matching-overload]
        self.assertGreater(mse_computation.weighted_num_samples, torch.tensor(0.0))

        res = mse.compute()
        self.assertGreater(res["mse-DefaultTask|lifetime_mse"], torch.tensor(0.0))
        self.assertGreater(res["mse-DefaultTask|lifetime_rmse"], torch.tensor(0.0))

        # Test if weights = 0 for one task of an update
        task_names = ["t1", "t2"]
        tasks = gen_test_tasks(task_names)
        _model_output = [
            gen_test_batch(
                label_name=task.label_name,
                prediction_name=task.prediction_name,
                weight_name=task.weight_name,
                batch_size=64,
            )
            for task in tasks
        ]
        model_output = {k: v for d in _model_output for k, v in d.items()}
        labels, predictions, weights, _ = parse_task_model_outputs(tasks, model_output)
        partial_zero_weights = {
            "t1": torch.zeros_like(weights["t1"]),
            "t2": weights["t2"],
        }

        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
            should_validate_update=True,
        )
        ne_computation = ne._metrics_computations

        ne.update(
            predictions=predictions,
            labels=labels,
            weights=partial_zero_weights,
        )
        self.assertEqual(ne_computation[0].cross_entropy_sum, torch.tensor(0.0))
        self.assertEqual(ne_computation[0].weighted_num_samples, torch.tensor(0.0))
        #  got `Tensor`.
        # pyrefly: ignore[no-matching-overload]
        self.assertGreater(ne_computation[1].cross_entropy_sum, torch.tensor(0.0))
        #  got `Tensor`.
        # pyrefly: ignore[no-matching-overload]
        self.assertGreater(ne_computation[1].weighted_num_samples, torch.tensor(0.0))

        res = ne.compute()
        self.assertEqual(res["ne-t1|lifetime_ne"], torch.tensor(0.0))
        self.assertGreater(res["ne-t2|lifetime_ne"], torch.tensor(0.0))

        ne.update(
            predictions=predictions,
            labels=labels,
            weights=weights,
        )
        #  got `Tensor`.
        # pyrefly: ignore[no-matching-overload]
        self.assertGreater(ne_computation[0].cross_entropy_sum, torch.tensor(0.0))
        #  got `Tensor`.
        # pyrefly: ignore[no-matching-overload]
        self.assertGreater(ne_computation[0].weighted_num_samples, torch.tensor(0.0))

        res = ne.compute()
        self.assertGreater(res["ne-t1|lifetime_ne"], torch.tensor(0.0))

    def test_compute(self) -> None:
        # Rank 0 does computation.
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertIn("ne-DefaultTask|lifetime_ne", res)
        self.assertIn("ne-DefaultTask|window_ne", res)

        # Rank non-zero skip computation.
        ne = NEMetric(
            world_size=1,
            my_rank=1,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertEqual({}, res)

        # Rank non-zero does computation if `compute_on_all_ranks` enabled.
        ne = NEMetric(
            world_size=1,
            my_rank=1,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
            compute_on_all_ranks=True,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertIn("ne-DefaultTask|lifetime_ne", res)
        self.assertIn("ne-DefaultTask|window_ne", res)

    def test_invalid_window_size(self) -> None:
        with self.assertRaises(ValueError):
            RecMetric(
                world_size=8,
                my_rank=0,
                window_size=50,
                batch_size=10,
                tasks=[DefaultTaskInfo],
            )

    def test_reset(self) -> None:
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=1000,
            fused_update_limit=0,
        )
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        ne = ne._metrics_computations[0]
        # pyrefly: ignore[bad-index, missing-attribute]
        window_buffer = ne._batch_window_buffers["window_cross_entropy_sum"].buffers
        self.assertTrue(len(window_buffer) > 0)
        # pyrefly: ignore[not-callable]
        ne.reset()
        # pyrefly: ignore[bad-index, missing-attribute]
        window_buffer = ne._batch_window_buffers["window_cross_entropy_sum"].buffers
        self.assertEqual(len(window_buffer), 0)

    @unittest.skipIf(_CUDA_UNAVAILABLE, "Test needs to run on GPU")
    def test_parse_task_model_outputs_ndcg(self) -> None:
        _, _, _, required_inputs = parse_task_model_outputs(
            tasks=[
                RecTaskInfo(
                    name="ndcg_example",
                ),
            ],
            # got Dict[str, Union[List[str], Tensor]]
            # pyrefly: ignore[bad-argument-type]
            model_out={
                "label": torch.tensor(
                    [0.0, 1.0, 0.0, 1.0], device=torch.device("cuda:0")
                ),
                "weight": torch.tensor(
                    [1.0, 1.0, 1.0, 1.0], device=torch.device("cuda:0")
                ),
                "prediction": torch.tensor(
                    [0.0, 1.0, 0.0, 1.0], device=torch.device("cuda:0")
                ),
                "session_id": ["1", "1", "2", "2"],
            },
            required_inputs_list=["session_id"],
        )
        self.assertEqual(required_inputs["session_id"].device, torch.device("cuda:0"))

    def test_batch_size_stages_kwargs_stripped(self) -> None:
        """Verify RecMetric.__init__ strips batch_size_stages from kwargs
        so non-TowerQPS metrics can be constructed with it in kwargs."""
        batch_size_stages = [
            BatchSizeStage(256, 100),
            BatchSizeStage(512, None),
        ]
        # NEMetric does NOT declare batch_size_stages as an explicit param.
        # This should succeed because RecMetric.__init__ strips it from kwargs.
        extra_kwargs: dict[str, Any] = {"batch_size_stages": batch_size_stages}
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
            **extra_kwargs,
        )
        # Verify the metric initialized and can compute
        ne.update(
            predictions=self.predictions,
            labels=self.labels,
            weights=self.weights,
        )
        res = ne.compute()
        self.assertIn("ne-DefaultTask|lifetime_ne", res)
