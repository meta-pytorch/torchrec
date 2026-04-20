#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict
from unittest.mock import patch

import torch
from torchrec.metrics.metrics_config import BatchSizeStage, DefaultTaskInfo, RecTaskInfo
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecMetricList
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
        self.assertEqual(ne1.cross_entropy_sum, ne2.cross_entropy_sum)
        self.assertEqual(ne1.weighted_num_samples, ne2.weighted_num_samples)
        self.assertEqual(ne1.pos_labels, ne2.pos_labels)
        self.assertEqual(ne1.neg_labels, ne2.neg_labels)

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


class RecMetricTensorSizeLoggingTest(unittest.TestCase):
    def test_get_tensor_size_metadata_tensor(self) -> None:
        tensor = torch.randn(4, 8)
        metadata = RecMetric._get_tensor_size_metadata("predictions", tensor)
        self.assertEqual(metadata["predictions_numel"], "32")
        self.assertEqual(metadata["predictions_shape"], "[4, 8]")

    def test_get_tensor_size_metadata_dict(self) -> None:
        tensor_dict: Dict[str, torch.Tensor] = {
            "task_a": torch.randn(4, 8),
            "task_b": torch.randn(2, 3),
        }
        metadata = RecMetric._get_tensor_size_metadata("labels", tensor_dict)
        self.assertIn("labels_numel", metadata)
        self.assertNotIn("labels_shape", metadata)
        numel_str = metadata["labels_numel"]
        self.assertIn("'task_a': 32", numel_str)
        self.assertIn("'task_b': 6", numel_str)

    @patch(
        "torchrec.metrics.rec_metric.EventLoggingHandler.n_batch_log_event",
    )
    def test_update_calls_log_event_with_tensor_metadata(
        self, mock_log_event: Any
    ) -> None:
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        model_output = gen_test_batch(128)
        labels, predictions, weights, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        ne.update(predictions=predictions, labels=labels, weights=weights)

        mock_log_event.assert_called_once()
        call_kwargs = mock_log_event.call_args[1]
        self.assertEqual(call_kwargs["component"], "rec_metrics")
        self.assertEqual(call_kwargs["event_name"], "update")
        metadata = call_kwargs["metadata"]
        self.assertIn("predictions_numel", metadata)
        self.assertIn("labels_numel", metadata)
        self.assertIn("weights_numel", metadata)

    @patch(
        "torchrec.metrics.rec_metric.EventLoggingHandler.n_batch_log_event",
    )
    def test_update_without_weights_omits_weights_metadata(
        self, mock_log_event: Any
    ) -> None:
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        model_output = gen_test_batch(128)
        labels, predictions, _, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        ne.update(predictions=predictions, labels=labels, weights=None)

        mock_log_event.assert_called_once()
        metadata = mock_log_event.call_args[1]["metadata"]
        self.assertIn("predictions_numel", metadata)
        self.assertIn("labels_numel", metadata)
        self.assertNotIn("weights_numel", metadata)

    def test_unfused_required_inputs_not_mutated_across_tasks(self) -> None:
        """Verify that scalar required_inputs are reshaped from originals on each
        task iteration, not from already-reshaped values of a prior iteration.

        Uses different batch sizes per task (4 vs 6) so that task_labels.size()
        differs across iterations. Without the fix, expand(1, 4) on iteration 1
        produces a tensor with numel=4; on iteration 2, numel > 1 routes to
        view(1, 6) which fails because 4 elements cannot be viewed as (1, 6).
        """
        task_names = ["t1", "t2"]
        tasks = gen_test_tasks(task_names)
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        # Different batch sizes per task to trigger the bug.
        predictions = {
            "t1": torch.rand(4, dtype=torch.double),
            "t2": torch.rand(6, dtype=torch.double),
        }
        labels = {
            "t1": torch.rand(4, dtype=torch.double),
            "t2": torch.rand(6, dtype=torch.double),
        }
        weights = {
            "t1": torch.ones(4, dtype=torch.double),
            "t2": torch.ones(6, dtype=torch.double),
        }
        # Scalar required_input: numel=1, triggers expand() path.
        required_inputs = {"scale": torch.tensor([5.0])}

        # Should not raise — without the fix, iteration 2 would crash with
        # RuntimeError because view(1, 6) cannot reshape 4 elements.
        ne.update(
            predictions=predictions,
            labels=labels,
            weights=weights,
            required_inputs=required_inputs,
        )

    @patch(
        "torchrec.metrics.rec_metric.EventLoggingHandler.n_batch_log_event",
    )
    def test_update_deduplicates_across_unfused_metrics(
        self, mock_log_event: Any
    ) -> None:
        """Verify logging fires once when multiple UNFUSED metrics process the same batch."""
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        mse = MSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        metric_list = RecMetricList([ne, mse])
        model_output = gen_test_batch(128)
        labels, predictions, weights, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        metric_list.update(predictions=predictions, labels=labels, weights=weights)

        mock_log_event.assert_called_once()

    @patch(
        "torchrec.metrics.rec_metric.EventLoggingHandler.n_batch_log_event",
    )
    def test_update_deduplicates_across_fused_metrics(
        self, mock_log_event: Any
    ) -> None:
        """Verify logging fires once when multiple FUSED metrics process the same batch."""
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        mse = MSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        metric_list = RecMetricList([ne, mse])
        model_output = gen_test_batch(128)
        labels, predictions, weights, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        metric_list.update(predictions=predictions, labels=labels, weights=weights)

        mock_log_event.assert_called_once()

    @patch(
        "torchrec.metrics.rec_metric.EventLoggingHandler.n_batch_log_event",
    )
    def test_fused_update_logs_post_transformation_shapes(
        self, mock_log_event: Any
    ) -> None:
        """Verify that in FUSED mode, logged shapes reflect the stacked
        (n_tasks, batch_size) tensors, not the original dict."""
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=0,
        )
        model_output = gen_test_batch(128)
        labels, predictions, weights, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        ne.update(predictions=predictions, labels=labels, weights=weights)

        mock_log_event.assert_called_once()
        metadata = mock_log_event.call_args[1]["metadata"]
        # FUSED mode stacks per-task tensors into a single tensor,
        # so the shape should be reported (not a dict of numels).
        self.assertIn("predictions_shape", metadata)
        self.assertIn("labels_shape", metadata)

    @patch(
        "torchrec.metrics.rec_metric.EventLoggingHandler.n_batch_log_event",
    )
    def test_fused_update_limit_deduplicates_logging(self, mock_log_event: Any) -> None:
        """Verify that when fused_update_limit > 0, the buffered flush
        still respects the _log_tensors flag from RecMetricList."""
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=5,
        )
        mse = MSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=64,
            tasks=[DefaultTaskInfo],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            fused_update_limit=5,
        )
        metric_list = RecMetricList([ne, mse])
        model_output = gen_test_batch(128)
        labels, predictions, weights, _ = parse_task_model_outputs(
            [DefaultTaskInfo], model_output
        )
        # Send 5 batches to trigger the fused update flush.
        for _ in range(5):
            metric_list.update(predictions=predictions, labels=labels, weights=weights)

        # Only the first metric (ne) should have logged, not both.
        mock_log_event.assert_called_once()
