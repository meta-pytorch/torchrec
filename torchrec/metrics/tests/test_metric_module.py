#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import dataclasses
import faulthandler
import logging
import multiprocessing
import os
import tempfile
import unittest
from collections.abc import Mapping
from typing import Any, Callable, cast, Dict, List, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.metrics.auc import _state_reduction, AUCMetric
from torchrec.metrics.deferrable_metrics import DeferrableMetrics
from torchrec.metrics.metric_module import (
    generate_metric_module,
    LossAggregationScope,
    MetricsResult,
    RecMetricModule,
    StateMetric,
    StateMetricEnum,
)
from torchrec.metrics.metrics_config import (
    _DEFAULT_WINDOW_SIZE,
    BatchSizeStage,
    DefaultMetricsConfig,
    DefaultTaskInfo,
    LOSS_DENOM_SUFFIX,
    LossAggregation,
    MetricsConfig,
    RecMetricDef,
    RecMetricEnum,
    ThroughputDef,
    validate_batch_size_stages,
)
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.noop_metric_module import NoOpMetricModule
from torchrec.metrics.rec_metric import (
    RecMetricException,
    RecMetricList,
    RecMetricValidationError,
    RecTaskInfo,
)
from torchrec.metrics.test_utils import gen_test_batch, gen_test_tasks
from torchrec.metrics.test_utils.mock_metrics import MockRecMetric
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.metrics.tower_qps import TowerQPSMetric
from torchrec.test_utils import get_free_port, seed_and_log, skip_if_asan_class

METRIC_MODULE_PATH = "torchrec.metrics.metric_module"


class MockOptimizer(StateMetric):
    def __init__(self) -> None:
        self.get_metrics_call = 0

    def get_metrics(self) -> MetricsResult:
        self.get_metrics_call += 1
        return {"learning_rate": torch.tensor(1.0)}


class TestMetricModule(RecMetricModule):
    r"""Implementation of RecMetricModule."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        rec_tasks: Optional[List[RecTaskInfo]] = None,
        rec_metrics: Optional[RecMetricList] = None,
        throughput_metric: Optional[ThroughputMetric] = None,
        state_metrics: Optional[Dict[str, StateMetric]] = None,
        compute_interval_steps: int = 100,
        min_compute_interval: float = 0.0,
        max_compute_interval: float = float("inf"),
    ) -> None:
        super().__init__(
            batch_size,
            world_size,
            rec_tasks=rec_tasks,
            rec_metrics=rec_metrics,
            throughput_metric=throughput_metric,
            state_metrics=state_metrics,
            compute_interval_steps=compute_interval_steps,
            min_compute_interval=min_compute_interval,
            max_compute_interval=max_compute_interval,
        )

    def _update_rec_metrics(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        if isinstance(model_out, MagicMock):
            return
        labels, predictions, weights, _ = parse_task_model_outputs(
            self.rec_tasks, model_out
        )
        self.rec_metrics.update(predictions=predictions, labels=labels, weights=weights)


class PreparingMetricModule(RecMetricModule):
    def _prepare_model_out_for_metrics(
        self, model_out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        prepared_model_out = dict(model_out)
        prepared_model_out["task-prediction"] = prepared_model_out["raw-prediction"]
        return prepared_model_out


class MetricModuleTest(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.WORLD_SIZE = 2

    def tearDown(self) -> None:
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _run_multi_process_test(
        self,
        world_size: int,
        backend: str,
        callable: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        processes = []
        ctx = multiprocessing.get_context("spawn")
        for rank in range(world_size):
            p = ctx.Process(
                target=callable,
                args=(
                    rank,
                    world_size,
                    backend,
                    *args,
                ),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    def test_metric_module(self) -> None:
        rec_metric_list_patch = patch(
            METRIC_MODULE_PATH + ".RecMetricList",
        )

        with tempfile.NamedTemporaryFile(delete=True) as backend:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{backend.name}",
                world_size=1,
                rank=0,
            )
            for pg in [None, dist.new_group([0])]:
                rec_metric_list_mock = rec_metric_list_patch.start()
                mock_optimizer = MockOptimizer()
                config = dataclasses.replace(
                    DefaultMetricsConfig, state_metrics=[StateMetricEnum.OPTIMIZERS]
                )
                metric_module = generate_metric_module(
                    TestMetricModule,
                    metrics_config=config,
                    batch_size=128,
                    world_size=64,
                    my_rank=0,
                    state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
                    device=torch.device("cpu"),
                    # pyrefly: ignore
                    process_group=pg,
                )
                metric_module.rec_metrics.compute = MagicMock(
                    return_value={"ne-ne|lifetime_ne": torch.tensor(0.75)}
                )
                self.assertEqual(
                    len(rec_metric_list_mock.call_args[0][0]),
                    len(DefaultMetricsConfig.rec_metrics),
                )
                self.assertEqual(len(metric_module.state_metrics), 1)
                metric_module.update(MagicMock())
                ret = metric_module.compute()
                rec_metric_list_patch.stop()
                metric_module.rec_metrics.compute.assert_called_once()
                self.assertTrue("ne-ne|lifetime_ne" in ret)
                self.assertTrue("throughput-throughput|total_examples" in ret)
                self.assertTrue("optimizers-optimizers|learning_rate" in ret)
            dist.destroy_process_group()

    def test_prepares_model_output_before_parsing(self) -> None:
        tasks = gen_test_tasks(["task"])
        rec_metric = MockRecMetric(
            world_size=1,
            my_rank=0,
            batch_size=2,
            tasks=tasks,
        )
        metric_module = PreparingMetricModule(
            batch_size=2,
            world_size=1,
            rec_tasks=tasks,
            rec_metrics=RecMetricList([rec_metric]),
        )
        raw_predictions = torch.tensor([0.25, 0.75])
        model_out = gen_test_batch(
            batch_size=2,
            label_name="task-label",
            prediction_name="unused-prediction",
            weight_name="task-weight",
        )
        model_out["raw-prediction"] = raw_predictions

        metric_module.update(model_out)

        self.assertEqual(1, rec_metric.update_called_count)
        actual_predictions = rec_metric.predictions_update_calls[0]
        self.assertIsInstance(actual_predictions, dict)
        self.assertTrue(
            torch.equal(
                raw_predictions,
                cast(Dict[str, torch.Tensor], actual_predictions)["task"],
            )
        )

    def test_compute_throughput_excludes_other_metrics(self) -> None:
        config = dataclasses.replace(
            DefaultMetricsConfig,
            throughput_metric=ThroughputDef(warmup_steps=1),
            state_metrics=[StateMetricEnum.OPTIMIZERS],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            backend_path = os.path.join(temp_dir, "store")
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{backend_path}",
                world_size=1,
                rank=0,
            )
            try:
                metric_module = generate_metric_module(
                    TestMetricModule,
                    metrics_config=config,
                    batch_size=128,
                    world_size=64,
                    my_rank=0,
                    state_metrics_mapping={StateMetricEnum.OPTIMIZERS: MockOptimizer()},
                    device=torch.device("cpu"),
                )
                for _ in range(3):
                    metric_module.update(gen_test_batch(128))

                throughput_only = metric_module.compute_throughput().resolve()
                full = metric_module.compute().resolve()

                self.assertIn(
                    "throughput-throughput|lifetime_throughput", throughput_only
                )
                self.assertIn(
                    "throughput-throughput|window_throughput", throughput_only
                )
                self.assertEqual(
                    set(throughput_only),
                    {key for key in full if key.startswith("throughput-")},
                )
                ne_keys = {key for key in full if key.startswith("ne-")}
                self.assertTrue(ne_keys)
                self.assertIn("optimizers-optimizers|learning_rate", full)
                self.assertTrue(ne_keys.isdisjoint(throughput_only))
                self.assertNotIn("optimizers-optimizers|learning_rate", throughput_only)
            finally:
                dist.destroy_process_group()

    @staticmethod
    def _run_compute_throughput_collective_free(
        rank: int, world_size: int, backend: str
    ) -> None:
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
        try:
            metric_module = generate_metric_module(
                TestMetricModule,
                metrics_config=DefaultMetricsConfig,
                batch_size=128,
                world_size=world_size,
                my_rank=rank,
                state_metrics_mapping={},
                device=torch.device("cpu"),
            )
            metric_module.update(gen_test_batch(128))

            if rank == 0:
                faulthandler.dump_traceback_later(30, exit=True)
                try:
                    ret = metric_module.compute_throughput().resolve()
                finally:
                    faulthandler.cancel_dump_traceback_later()
                assert "throughput-throughput|total_examples" in ret

            dist.barrier()
        finally:
            dist.destroy_process_group()

    def test_compute_throughput_is_collective_free(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            callable=self._run_compute_throughput_collective_free,
        )

    def test_rectask_info(self) -> None:
        mock_optimizer = MockOptimizer()
        config = DefaultMetricsConfig
        metric_module_seperate_task_info = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        config = dataclasses.replace(
            DefaultMetricsConfig,
            rec_metrics={
                RecMetricEnum.NE: RecMetricDef(
                    rec_tasks=[], rec_task_indices=[0], window_size=_DEFAULT_WINDOW_SIZE
                )
            },
        )
        metric_module_unified_task_info = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        self.assertEqual(
            metric_module_seperate_task_info.rec_metrics[0]._namespace,
            metric_module_unified_task_info.rec_metrics[0]._namespace,
        )
        self.assertEqual(
            metric_module_seperate_task_info.rec_metrics[0]._tasks,
            metric_module_unified_task_info.rec_metrics[0]._tasks,
        )

    def test_compatibility_with_older_metric_module(self) -> None:
        """
        This test checks if latest RecMetricModule can load up
        metric module from an older checkpoint
        """

        def _create_comprehensive_metrics_config() -> MetricsConfig:
            """
            Similar to DefaultMetricsConfig, but with comprehensive metrics and tasks.
            """
            from torchrec.metrics.metric_module import REC_METRICS_MAPPING
            from torchrec.metrics.metrics_config import SessionMetricDef

            metric_arguments = {
                RecMetricEnum.MULTICLASS_RECALL: {"number_of_classes": 2},
                RecMetricEnum.TOWER_QPS: {"warmup_steps": 100},
            }

            # Session-level metrics require special task configuration with session_metric_def
            session_task_info = RecTaskInfo(
                name="SessionTask",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
                session_metric_def=SessionMetricDef(
                    session_var_name="session",
                    top_threshold=10,
                    run_ranking_of_labels=False,
                ),
            )

            # Tensor weighted average metric requires tensor_name
            tensor_task_info = RecTaskInfo(
                name="TensorTask",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
                tensor_name="target_tensor",
            )

            session_metrics = {
                RecMetricEnum.RECALL_SESSION_LEVEL,
                RecMetricEnum.PRECISION_SESSION_LEVEL,
            }

            tensor_metrics = {
                RecMetricEnum.TENSOR_WEIGHTED_AVG,
            }

            all_metric_defs: Dict[RecMetricEnum, RecMetricDef] = {}
            for metric_enum in REC_METRICS_MAPPING.keys():
                if isinstance(metric_enum, RecMetricEnum):
                    # Session-level metrics require special task configuration
                    if metric_enum in session_metrics:
                        all_metric_defs[metric_enum] = RecMetricDef(
                            rec_tasks=[session_task_info],
                            window_size=_DEFAULT_WINDOW_SIZE,
                            arguments=metric_arguments.get(metric_enum),
                        )
                    # Tensor metrics require tensor_name
                    elif metric_enum in tensor_metrics:
                        all_metric_defs[metric_enum] = RecMetricDef(
                            rec_tasks=[tensor_task_info],
                            window_size=_DEFAULT_WINDOW_SIZE,
                            arguments=metric_arguments.get(metric_enum),
                        )
                    else:
                        arguments = metric_arguments.get(metric_enum)
                        all_metric_defs[metric_enum] = RecMetricDef(
                            rec_tasks=[DefaultTaskInfo],
                            window_size=_DEFAULT_WINDOW_SIZE,
                            arguments=arguments,
                        )
            comprehensive_config = MetricsConfig(
                rec_tasks=[DefaultTaskInfo],
                rec_metrics=all_metric_defs,
                throughput_metric=ThroughputDef(),
                state_metrics=[],
            )
            return comprehensive_config

        ComprehensiveMetricsConfig: MetricsConfig = (
            _create_comprehensive_metrics_config()
        )
        # This simulates what an older checkpoint may have
        predefined_state_dict_keys = [
            "rec_metrics.rec_metrics.0._metrics_computations.0.cross_entropy_sum",
            "rec_metrics.rec_metrics.0._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.0._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.0._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.1._metrics_computations.0.cross_entropy_positive_sum",
            "rec_metrics.rec_metrics.1._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.1._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.1._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.2._metrics_computations.0.cross_entropy_sum",
            "rec_metrics.rec_metrics.2._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.2._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.2._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.3._metrics_computations.0.cross_entropy_sum",
            "rec_metrics.rec_metrics.3._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.3._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.3._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.4._metrics_computations.0.calibration_num",
            "rec_metrics.rec_metrics.4._metrics_computations.0.calibration_denom",
            "rec_metrics.rec_metrics.5._metrics_computations.0.ctr_num",
            "rec_metrics.rec_metrics.5._metrics_computations.0.ctr_denom",
            "rec_metrics.rec_metrics.6._metrics_computations.0.calibration_num",
            "rec_metrics.rec_metrics.6._metrics_computations.0.calibration_denom",
            "rec_metrics.rec_metrics.10._metrics_computations.0.error_sum",
            "rec_metrics.rec_metrics.10._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.11._metrics_computations.0.error_sum",
            "rec_metrics.rec_metrics.11._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.12._metrics_computations.0.tp_at_k",
            "rec_metrics.rec_metrics.12._metrics_computations.0.total_weights",
            "rec_metrics.rec_metrics.13._metrics_computations.0.weighted_sum",
            "rec_metrics.rec_metrics.13._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.14._metrics_computations.0.num_examples",
            "rec_metrics.rec_metrics.14._metrics_computations.0.warmup_examples",
            "rec_metrics.rec_metrics.14._metrics_computations.0.time_lapse",
            "rec_metrics.rec_metrics.15._metrics_computations.0.num_true_pos",
            "rec_metrics.rec_metrics.15._metrics_computations.0.num_false_neg",
            "rec_metrics.rec_metrics.16._metrics_computations.0.num_true_pos",
            "rec_metrics.rec_metrics.16._metrics_computations.0.num_false_pos",
            "rec_metrics.rec_metrics.17._metrics_computations.0.accuracy_sum",
            "rec_metrics.rec_metrics.17._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.18._metrics_computations.0.sum_ndcg",
            "rec_metrics.rec_metrics.18._metrics_computations.0.num_sessions",
            "rec_metrics.rec_metrics.19._metrics_computations.0.error_sum",
            "rec_metrics.rec_metrics.19._metrics_computations.0.weighted_num_pairs",
            "rec_metrics.rec_metrics.21._metrics_computations.0.true_pos_sum",
            "rec_metrics.rec_metrics.21._metrics_computations.0.false_pos_sum",
            "rec_metrics.rec_metrics.22._metrics_computations.0.true_pos_sum",
            "rec_metrics.rec_metrics.22._metrics_computations.0.false_neg_sum",
            "rec_metrics.rec_metrics.23._metrics_computations.0.cross_entropy_sum",
            "rec_metrics.rec_metrics.23._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.23._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.23._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.23._metrics_computations.0.num_examples",
            "rec_metrics.rec_metrics.24._metrics_computations.0.calibration_num",
            "rec_metrics.rec_metrics.24._metrics_computations.0.calibration_denom",
            "rec_metrics.rec_metrics.24._metrics_computations.0.num_examples",
            "rec_metrics.rec_metrics.26._metrics_computations.0.weighted_sum",
            "rec_metrics.rec_metrics.26._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.27._metrics_computations.0.cross_entropy_sum",
            "rec_metrics.rec_metrics.27._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.27._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.27._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.27._metrics_computations.0.weighted_sum_predictions",
            "rec_metrics.rec_metrics.28._metrics_computations.0.cross_entropy_sum",
            "rec_metrics.rec_metrics.28._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.28._metrics_computations.0.pos_labels",
            "rec_metrics.rec_metrics.28._metrics_computations.0.neg_labels",
            "rec_metrics.rec_metrics.29._metrics_computations.0.true_pos_sum",
            "rec_metrics.rec_metrics.29._metrics_computations.0.false_pos_sum",
            "rec_metrics.rec_metrics.29._metrics_computations.0.false_neg_sum",
            "rec_metrics.rec_metrics.30._metrics_computations.0.error_sum",
            "rec_metrics.rec_metrics.30._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.30._metrics_computations.0.const_pred_error_sum",
            "rec_metrics.rec_metrics.31._metrics_computations.0.prediction_sum",
            "rec_metrics.rec_metrics.31._metrics_computations.0.label_sum",
            "rec_metrics.rec_metrics.31._metrics_computations.0.weighted_num_samples",
            "rec_metrics.rec_metrics.32._metrics_computations.0.false_pos_sum_label_0",
            "rec_metrics.rec_metrics.32._metrics_computations.0.true_pos_sum_label_0",
            "rec_metrics.rec_metrics.33._metrics_computations.0.missing_label_sum",
            "rec_metrics.rec_metrics.34._metrics_computations.0.weighted_predictions_sum",
            "rec_metrics.rec_metrics.35._metrics_computations.0.weighted_pos_sum",
            "rec_metrics.rec_metrics.36._metrics_computations.0.weighted_sum",
            "throughput_metric.total_examples",
            "throughput_metric.warmup_examples",
            "throughput_metric.time_lapse_after_warmup",
        ]

        # This is the latest RecMetricModule
        mock_optimizer = MockOptimizer()

        latest_metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=ComprehensiveMetricsConfig,
            batch_size=128,
            world_size=64,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        tc = unittest.TestCase()
        tc.assertSetEqual(
            set(predefined_state_dict_keys),
            set(latest_metric_module.state_dict().keys()),
            "RecMetricModule state_dict keys have changed - ensure backward compatibility with older checkpoints",
        )

    @staticmethod
    def _run_trainer_checkpointing(rank: int, world_size: int, backend: str) -> None:
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )

        mock_optimizer = MockOptimizer()
        config = dataclasses.replace(
            DefaultMetricsConfig, state_metrics=[StateMetricEnum.OPTIMIZERS]
        )
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=world_size,
            my_rank=rank,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        value = 12345
        state_dict = metric_module.state_dict()
        keys = list(state_dict.keys())
        for k in state_dict.keys():
            state_dict[k] = torch.tensor(value, dtype=torch.long).detach()
        logging.info(f"Metrics state keys = {keys}")
        metric_module.load_state_dict(state_dict)
        tc = unittest.TestCase()
        tc.assertTrue("throughput_metric.warmup_examples" in keys)
        tc.assertTrue("throughput_metric.total_examples" in keys)
        tc.assertTrue(
            "rec_metrics.rec_metrics.0._metrics_computations.0.cross_entropy_sum"
            in keys
        )

        # 1. Test sync()
        metric_module.sync()
        state_dict = metric_module.state_dict()
        for k, v in state_dict.items():
            if k.startswith("rec_metrics."):
                if k.endswith("has_valid_update"):
                    tc.assertEqual(v.item(), 1)
                else:
                    tc.assertEqual(v.item(), value * world_size)

        # 2. Test unsync()
        metric_module.unsync()
        state_dict = metric_module.state_dict()
        for v in state_dict.values():
            tc.assertEqual(v.item(), value)

        # 3. Test reset()
        metric_module.reset()
        state_dict = metric_module.state_dict()
        for k, v in state_dict.items():
            if k.startswith("rec_metrics."):
                tc.assertEqual(v.item(), 0)

    def test_rank0_checkpointing(self) -> None:
        # Call the tested methods to make code coverage visible to the testing system
        # Begin of dummy codes
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        state_dict = metric_module.state_dict()
        metric_module.load_state_dict(state_dict)
        metric_module.sync()
        metric_module.unsync()
        metric_module.reset()
        # End of dummy codes

        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            callable=self._run_trainer_checkpointing,
        )

    @staticmethod
    def _run_trainer_initial_states_checkpointing(
        rank: int, world_size: int, backend: str
    ) -> None:
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )

        mock_optimizer = MockOptimizer()
        config = dataclasses.replace(
            DefaultMetricsConfig,
            rec_metrics={
                RecMetricEnum.AUC: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
                )
            },
            state_metrics=[StateMetricEnum.OPTIMIZERS],
        )

        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
            device=torch.device("cpu"),
        )
        tc = unittest.TestCase()
        tc.assertTrue(isinstance(metric_module.rec_metrics.rec_metrics[0], AUCMetric))
        tc.assertEqual(
            len(
                # pyrefly: ignore[bad-index, missing-attribute]
                metric_module.rec_metrics.rec_metrics[0]
                ._metrics_computations[0]
                .predictions
            ),
            1,  # The predictions state is a list containing 1 tensor value
        )

        # 1. After the metric module is created
        tc.assertEqual(
            # pyrefly: ignore[bad-index, missing-attribute]
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
            # The 1st 1 is the number of tasks; the 2nd 1 is the default value length
        )

        metric_module.sync()
        tc.assertEqual(
            # pyrefly: ignore[bad-index, missing-attribute]
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 2),
        )

        metric_module.unsync()
        tc.assertEqual(
            # pyrefly: ignore[bad-index, missing-attribute]
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
        )

        # 2. After the metric module gets reset
        metric_module.update(gen_test_batch(128))
        metric_module.reset()
        metric_module.sync()
        metric_module.unsync()
        tc.assertEqual(
            # pyrefly: ignore[bad-index, missing-attribute]
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
        )

    def test_initial_states_rank0_checkpointing(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            callable=self._run_trainer_initial_states_checkpointing,
        )

    def test_should_compute(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        metric_module.trained_batches = 1
        self.assertFalse(metric_module.should_compute())
        metric_module.trained_batches = metric_module.compute_interval_steps - 1
        self.assertFalse(metric_module.should_compute())
        metric_module.trained_batches = metric_module.compute_interval_steps
        self.assertTrue(metric_module.should_compute())

    @staticmethod
    @patch("torchrec.metrics.metric_module.RecMetricList")
    @patch("torchrec.metrics.metric_module.time")
    def _test_adjust_compute_interval(
        rank: int,
        world_size: int,
        backend: str,
        batch_time: float,
        min_interval: float,
        max_interval: float,
        mock_time: MagicMock,
        mock_recmetric_list: MagicMock,
    ) -> None:
        init_by_me = False
        if not dist.is_initialized():
            init_by_me = True
            dist.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
            )
        mock_time.time = MagicMock(return_value=0.0)

        def _train(metric_module: RecMetricModule) -> float:
            for _ in range(metric_module.compute_interval_steps):
                metric_module.update(batch)
            elapsed_time = metric_module.compute_interval_steps * batch_time
            mock_time.time.return_value += elapsed_time
            return elapsed_time

        config = copy.deepcopy(DefaultMetricsConfig)
        config.min_compute_interval = min_interval
        config.max_compute_interval = max_interval
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        batch = MagicMock()

        tc = unittest.TestCase()
        compute_interval_steps = metric_module.compute_interval_steps
        # First compute
        elapsed_time = _train(metric_module)
        tc.assertTrue(metric_module.should_compute())
        metric_module.compute()
        # Second compute
        tc.assertEqual(compute_interval_steps, metric_module.compute_interval_steps)
        elapsed_time = _train(metric_module)
        tc.assertTrue(metric_module.should_compute())
        metric_module.compute()

        tc.assertEqual(
            (-1.0, -1.0),
            (metric_module.min_compute_interval, metric_module.max_compute_interval),
        )

        max_interval = (
            float("inf") if min_interval > 0 and max_interval <= 0 else max_interval
        )
        if min_interval <= 0 and max_interval <= 0:
            tc.assertEqual(compute_interval_steps, metric_module.compute_interval_steps)
        elif max_interval >= elapsed_time >= min_interval:
            tc.assertEqual(compute_interval_steps, metric_module.compute_interval_steps)
        else:
            tc.assertNotEqual(
                compute_interval_steps, metric_module.compute_interval_steps
            )
            elapsed_time = _train(metric_module)
            tc.assertTrue(elapsed_time >= min_interval)
            tc.assertTrue(elapsed_time <= max_interval)
        if init_by_me:
            dist.destroy_process_group()

    def _test_adjust_compute_interval_launcher(
        self,
        batch_time: float,
        min_interval: float = 0.0,
        max_interval: float = float("inf"),
    ) -> None:
        self._run_multi_process_test(
            self.WORLD_SIZE,
            "gloo",
            self._test_adjust_compute_interval,
            batch_time,
            min_interval,
            max_interval,
        )

    def test_adjust_compute_interval_not_set(self) -> None:
        self._test_adjust_compute_interval_launcher(
            batch_time=0.1,
        )

    def test_adjust_compute_interval_0_30(self) -> None:
        self._test_adjust_compute_interval_launcher(
            batch_time=1,
            min_interval=0.0,
            max_interval=30.0,
        )

        # This is to ensure the test coverage is correct.
        with tempfile.NamedTemporaryFile(delete=True) as backend_file:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{backend_file.name}",
                world_size=1,
                rank=0,
            )

            self._test_adjust_compute_interval(0, 1, "gloo", 1, 0.0, 30.0)
        # Needed to destroy the process group as _test_adjust_compute_interval
        # won't since we initialize the process group for it.
        dist.destroy_process_group()

    def test_adjust_compute_interval_15_inf(self) -> None:
        self._test_adjust_compute_interval_launcher(
            batch_time=0.1,
            min_interval=15.0,
            max_interval=float("inf"),
        )

        # This is to ensure the test coverage is correct.
        with tempfile.NamedTemporaryFile(delete=True) as backend_file:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{backend_file.name}",
                world_size=1,
                rank=0,
            )

            self._test_adjust_compute_interval(0, 1, "gloo", 0.1, 15.0, float("inf"))
        # Needed to destroy the process group as _test_adjust_compute_interval
        # won't since we initialize the process group for it.
        dist.destroy_process_group()

    def test_adjust_compute_interval_15_30(self) -> None:
        self._test_adjust_compute_interval_launcher(
            batch_time=1,
            min_interval=15.0,
            max_interval=30.0,
        )

        # This is to ensure the test coverage is correct.
        with tempfile.NamedTemporaryFile(delete=True) as backend_file:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{backend_file.name}",
                world_size=1,
                rank=0,
            )

            self._test_adjust_compute_interval(0, 1, "gloo", 1, 15.0, 30.0)
        # Needed to destroy the process group as _test_adjust_compute_interval
        # won't since we initialize the process group for it.
        dist.destroy_process_group()

    def test_adjust_compute_interval_1_30(self) -> None:
        self._test_adjust_compute_interval_launcher(
            batch_time=1,
            min_interval=1.0,
            max_interval=30.0,
        )

    def test_save_and_load_state_dict(self) -> None:
        # Test without batch_size_stages
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        metric_module.update(gen_test_batch(128))

        state_dict_without_bss = metric_module.state_dict()
        # Make sure state loading works and doesn't throw an error
        metric_module.load_state_dict(state_dict_without_bss)
        # Make sure num_batch in the throughput module is not in state_dict
        self.assertFalse("throughput_metric.num_batch" in state_dict_without_bss)

        # Test with batch_size_stages
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            batch_size_stages=[BatchSizeStage(256, 100), BatchSizeStage(512, None)],
        )

        # Update metric 100 times
        for _ in range(100):
            metric_module.update(gen_test_batch(128))

        # Simulate a checkpoint save
        state_dict = metric_module.state_dict()
        # Make sure num_batch is updated correctly to 100
        self.assertEqual(state_dict["throughput_metric.num_batch"], 100)

        # Simulate a checkpoint load
        metric_module.load_state_dict(state_dict)
        # Make sure num_batch is correctly restored
        throughput_metric = metric_module.throughput_metric
        self.assertIsNotNone(throughput_metric)
        self.assertEqual(throughput_metric._num_batch, 100)
        # Make sure num_batch is correctly synchronized
        self.assertEqual(throughput_metric._num_batch, 100)

        # Load the same checkpoint into a module that doesn't use BSS

        no_bss_metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            batch_size_stages=None,
        )

        no_bss_metric_module.load_state_dict(state_dict)
        # Make sure num_batch wasn't created on the throughput module (and no exception was thrown above)
        self.assertFalse(hasattr(no_bss_metric_module.throughput_metric, "_num_batch"))

    def test_batch_size_stages_passed_to_rec_metrics(self) -> None:
        """Verify batch_size_stages flows from generate_metric_module
        through _generate_rec_metrics to individual RecMetric instances."""
        batch_size_stages = [
            BatchSizeStage(256, 100),
            BatchSizeStage(512, None),
        ]
        # Create a config that includes TowerQPSMetric
        config = MetricsConfig(
            rec_tasks=[DefaultTaskInfo],
            rec_metrics={
                RecMetricEnum.NE: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo],
                    window_size=_DEFAULT_WINDOW_SIZE,
                ),
                RecMetricEnum.TOWER_QPS: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo],
                    window_size=_DEFAULT_WINDOW_SIZE,
                ),
            },
            throughput_metric=ThroughputDef(),
            state_metrics=[],
        )

        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            batch_size_stages=batch_size_stages,
        )

        # Verify TowerQPSMetric received batch_size_stages
        found_tower_qps = False
        for metric in metric_module.rec_metrics.rec_metrics:
            if isinstance(metric, TowerQPSMetric):
                found_tower_qps = True
                self.assertIsNotNone(metric._batch_size_stages)
                self.assertEqual(len(metric._batch_size_stages), 2)
                self.assertEqual(metric._batch_size_stages[0].batch_size, 256)
                self.assertEqual(metric._batch_size_stages[0].max_iters, 100)
                self.assertEqual(metric._batch_size_stages[1].batch_size, 512)
                self.assertIsNone(metric._batch_size_stages[1].max_iters)
        self.assertTrue(found_tower_qps, "TowerQPSMetric not found in rec_metrics")

    def test_async_compute_raises_exception(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(
            RecMetricException,
            "async_compute is not supported in RecMetricModule",
        ):
            metric_module.async_compute()

    def test_shutdown(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        # shutdown() should not raise any exception
        metric_module.shutdown()

    def test_local_compute(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        metric_module.update(gen_test_batch(128))
        result = metric_module.local_compute()
        self.assertIsInstance(result, Mapping)

    def test_get_required_inputs(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        # get_required_inputs delegates to rec_metrics
        result = metric_module.get_required_inputs()
        # Result can be None or a list depending on metric configuration
        self.assertTrue(result is None or isinstance(result, list))

    def test_invalid_max_compute_interval(self) -> None:
        with self.assertRaises(ValueError) as context:
            RecMetricModule(
                batch_size=128,
                world_size=1,
                min_compute_interval=5.0,
                max_compute_interval=0.0,  # Invalid: <= 0 when min is set
            )
        self.assertIn("Max compute interval", str(context.exception))

    def test_invalid_min_compute_interval(self) -> None:
        with self.assertRaises(ValueError) as context:
            RecMetricModule(
                batch_size=128,
                world_size=1,
                min_compute_interval=-1.0,  # Invalid: < 0
                max_compute_interval=30.0,
            )
        self.assertIn("Min compute interval", str(context.exception))

    def test_load_state_dict_with_trained_batches_key(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        state_dict = metric_module.state_dict()

        # Add the _trained_batches key to simulate old checkpoint
        state_dict["_trained_batches"] = torch.tensor(42, dtype=torch.long)

        # Load the state_dict with _trained_batches
        # This should not raise an error
        metric_module.load_state_dict(state_dict)
        metric_module.update(gen_test_batch(128))
        result = metric_module.compute()
        self.assertIsInstance(result, Mapping)
        self.assertTrue(len(result) > 0)

    def test_load_state_dict_without_trained_batches_key(self) -> None:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=DefaultMetricsConfig,
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )
        state_dict = metric_module.state_dict()

        # Verify the key is not in the state_dict
        self.assertNotIn("_trained_batches", state_dict)

        # Load the clean state_dict
        # This should not raise an error
        metric_module.load_state_dict(state_dict)
        metric_module.update(gen_test_batch(128))
        result = metric_module.compute()
        self.assertIsInstance(result, Mapping)
        self.assertTrue(len(result) > 0)


def metric_module_gather_state(
    rank: int,
    world_size: int,
    backend: str,
    config: MetricsConfig,
    batch_size: int,
    local_size: Optional[int] = None,
) -> None:
    """
    We compare the computed values of the metric module using the get_pre_compute_states API.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=batch_size,
            world_size=world_size,
            my_rank=rank,
            state_metrics_mapping={},
            device=ctx.device,
            process_group=ctx.pg,
        )

        test_batches = []
        for _ in range(100):
            test_batch = gen_test_batch(batch_size)
            for k in test_batch.keys():
                test_batch[k] = test_batch[k].to(ctx.device)
            # save to re run
            test_batches.append(test_batch)
            metric_module.update(test_batch)

        computed_value = metric_module.compute()
        states = metric_module.get_pre_compute_states(pg=ctx.pg)

        torch.distributed.barrier(ctx.pg)
        # Compare to computing metrics on metric module that loads from pre_compute_states
        new_metric_module = generate_metric_module(
            TestMetricModule,
            metrics_config=config,
            batch_size=batch_size,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device(f"cuda:{rank}"),
            # pyrefly: ignore
            process_group=dist.new_group(ranks=[rank], backend="nccl"),
        )
        new_metric_module.load_pre_compute_states(states)
        new_computed_value = new_metric_module.compute()

        for metric, tensor in computed_value.items():
            new_tensor = new_computed_value[metric]
            torch.testing.assert_close(tensor, new_tensor, check_device=False)

        metric_module.shutdown()


class RecMetricDebugModeTest(unittest.TestCase):
    def _create_config(
        self,
        metric_enums: List[RecMetricEnum],
    ) -> MetricsConfig:
        return MetricsConfig(
            rec_tasks=[DefaultTaskInfo],
            rec_metrics={
                metric_enum: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo],
                    window_size=_DEFAULT_WINDOW_SIZE,
                )
                for metric_enum in metric_enums
            },
        )

    def _create_module(
        self,
        metric_enums: List[RecMetricEnum],
        debug_mode: bool = True,
    ) -> RecMetricModule:
        return generate_metric_module(
            RecMetricModule,
            metrics_config=self._create_config(metric_enums),
            batch_size=4,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            debug_mode=debug_mode,
        )

    def test_debug_mode_disabled_does_not_validate(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NE], debug_mode=False)
        batch = gen_test_batch(
            4,
            label_value=torch.tensor([0.0, float("nan"), 1.0, 0.0]),
        )

        metric_module.update(batch)

        self.assertEqual(1, metric_module.trained_batches)

    def test_nan_label_fails_before_state_mutation(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NE])
        state_before = {
            name: tensor.detach().clone()
            for name, tensor in metric_module.state_dict().items()
        }
        batch = gen_test_batch(
            4,
            label_value=torch.tensor([0.0, float("nan"), 1.0, 0.0]),
        )

        with self.assertRaises(RecMetricValidationError) as context:
            metric_module.update(batch)

        message = str(context.exception)
        self.assertIn("metric=NEMetric", message)
        self.assertIn("task=DefaultTask", message)
        self.assertIn("rank=0 labels shape=[4] dtype=torch.float32", message)
        self.assertIn("numel=4", message)
        self.assertIn("nan=1", message)
        self.assertEqual(0, metric_module.trained_batches)
        for name, tensor in metric_module.state_dict().items():
            torch.testing.assert_close(tensor, state_before[name])

    def test_num_missing_labels_ignores_predictions_and_counts_nan_labels(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NUM_MISSING_LABELS])
        metric_module.update(
            gen_test_batch(
                4,
                label_value=torch.tensor([float("nan"), 1.0, float("nan"), 0.0]),
                prediction_value=torch.tensor([float("nan"), float("inf"), 1.0, 0.0]),
                weight_value=torch.ones(4),
            )
        )

        metrics = metric_module.compute().resolve()
        self.assertEqual(
            2.0,
            metrics[
                "num_missing_labels-DefaultTask|lifetime_num_missing_labels"
            ].item(),
        )

    def test_num_positive_samples_ignores_predictions_and_nan_labels(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NUM_POSITIVE_SAMPLES])
        metric_module.update(
            gen_test_batch(
                4,
                label_value=torch.tensor([float("nan"), 1.0, float("nan"), 0.0]),
                prediction_value=torch.tensor([float("nan"), float("inf"), 1.0, 0.0]),
                weight_value=torch.ones(4),
            )
        )

        metrics = metric_module.compute().resolve()
        self.assertEqual(
            1.0,
            metrics[
                "num_positive_samples-DefaultTask|lifetime_num_positive_samples"
            ].item(),
        )

    def test_weighted_sum_predictions_ignores_labels_and_nan_predictions(
        self,
    ) -> None:
        metric_module = self._create_module([RecMetricEnum.WEIGHTED_SUM_PREDICTIONS])
        metric_module.update(
            gen_test_batch(
                4,
                label_value=torch.tensor([float("nan"), float("inf"), 1.0, 0.0]),
                prediction_value=torch.tensor([float("nan"), 1.0, float("nan"), 2.0]),
                weight_value=torch.ones(4),
            )
        )

        metrics = metric_module.compute().resolve()
        self.assertEqual(
            3.0,
            metrics[
                "weighted_sum_predictions-DefaultTask|lifetime_weighted_sum_predictions"
            ].item(),
        )

    def test_nan_label_shared_with_ne_names_ne(self) -> None:
        metric_module = self._create_module(
            [RecMetricEnum.NUM_MISSING_LABELS, RecMetricEnum.NE]
        )
        batch = gen_test_batch(
            4,
            label_value=torch.tensor([0.0, float("nan"), 1.0, 0.0]),
        )

        with self.assertRaises(RecMetricValidationError) as context:
            metric_module.update(batch)

        self.assertIn("metric=NEMetric", str(context.exception))
        self.assertNotIn("metric=NumMissingLabelsMetric", str(context.exception))

    def test_num_missing_labels_rejects_label_infinity(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NUM_MISSING_LABELS])
        batch = gen_test_batch(
            4,
            label_value=torch.tensor([0.0, float("inf"), 1.0, 0.0]),
        )

        with self.assertRaises(RecMetricValidationError) as context:
            metric_module.update(batch)

        message = str(context.exception)
        self.assertIn("metric=NumMissingLabelsMetric", message)
        self.assertIn("+inf=1", message)

    def test_non_finite_predictions_and_weights_fail(self) -> None:
        cases = [
            ("predictions", "prediction", float("nan"), "nan=1"),
            ("predictions", "prediction", float("inf"), "+inf=1"),
            ("weights", "weight", float("nan"), "nan=1"),
            ("weights", "weight", float("-inf"), "-inf=1"),
        ]
        for tensor_kind, batch_key, value, expected_count in cases:
            with self.subTest(tensor_kind=tensor_kind, value=value):
                metric_module = self._create_module([RecMetricEnum.NE])
                batch = gen_test_batch(4)
                batch[batch_key][0] = value

                with self.assertRaises(RecMetricValidationError) as context:
                    metric_module.update(batch)

                message = str(context.exception)
                self.assertIn(tensor_kind, message)
                self.assertIn(expected_count, message)

    def test_complex_infinity_signs_are_reported(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NE])
        batch = gen_test_batch(4)
        batch["prediction"] = torch.tensor(
            [complex(float("-inf"), 0.0), 0j, 0j, 0j],
            dtype=torch.complex64,
        )

        with self.assertRaises(RecMetricValidationError) as context:
            metric_module.update(batch)

        message = str(context.exception)
        self.assertIn("dtype=torch.complex64", message)
        self.assertIn("+inf=0", message)
        self.assertIn("-inf=1", message)

    def test_complex_non_finite_counts_do_not_overlap(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NE])
        batch = gen_test_batch(4)
        batch["prediction"] = torch.tensor(
            [
                complex(float("inf"), float("nan")),
                complex(float("inf"), float("-inf")),
                0j,
                0j,
            ],
            dtype=torch.complex64,
        )

        with self.assertRaises(RecMetricValidationError) as context:
            metric_module.update(batch)

        message = str(context.exception)
        self.assertIn("nan=1", message)
        self.assertIn("+inf=0", message)
        self.assertIn("-inf=0", message)
        self.assertIn("mixed_inf=1", message)

    def test_debug_mode_does_not_require_process_group(self) -> None:
        metric_module = generate_metric_module(
            RecMetricModule,
            metrics_config=self._create_config([RecMetricEnum.NE]),
            batch_size=4,
            world_size=2,
            my_rank=1,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            debug_mode=True,
        )
        batch = gen_test_batch(
            4,
            label_value=torch.tensor([0.0, float("nan"), 1.0, 0.0]),
        )

        with self.assertRaises(RecMetricValidationError) as context:
            metric_module.update(batch)

        self.assertIn("rank=1", str(context.exception))
        self.assertEqual(0, metric_module.trained_batches)

    def test_preparation_failure_is_not_reclassified(self) -> None:
        metric_module = self._create_module([RecMetricEnum.NE])
        batch = gen_test_batch(4)
        batch["label"] = torch.empty(4, device="meta")

        with self.assertRaises(NotImplementedError):
            metric_module.update(batch)


def _create_distributed_debug_module(
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup],
) -> RecMetricModule:
    config = MetricsConfig(
        rec_tasks=[DefaultTaskInfo],
        rec_metrics={
            RecMetricEnum.NE: RecMetricDef(
                rec_tasks=[DefaultTaskInfo],
                window_size=_DEFAULT_WINDOW_SIZE,
            )
        },
    )
    return generate_metric_module(
        RecMetricModule,
        metrics_config=config,
        batch_size=4,
        world_size=world_size,
        my_rank=rank,
        state_metrics_mapping={},
        device=torch.device("cpu"),
        process_group=process_group,
        debug_mode=True,
    )


def _run_rank_local_input_validation(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    with MultiProcessContext(rank, world_size, backend) as context:
        metric_module = _create_distributed_debug_module(rank, world_size, context.pg)
        batch = gen_test_batch(4)
        if rank == 1:
            batch["label"] = batch["label"].reshape(2, 2)
            batch["prediction"] = batch["prediction"].reshape(2, 2)
            batch["label"][0, 0] = float("nan")
        try:
            metric_module.update(batch)
        except RecMetricValidationError as error:
            assert rank == 1
            assert "rank=1" in str(error)
            assert "metric=NEMetric" in str(error)
            assert "shape=[2, 2]" in str(error)
        else:
            assert rank == 0
            assert metric_module.trained_batches == 1
        dist.barrier()


def _run_rank_local_uneven_updates(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    with MultiProcessContext(rank, world_size, backend) as context:
        metric_module = _create_distributed_debug_module(rank, world_size, context.pg)
        update_count = 3 if rank == 0 else 1
        for _ in range(update_count):
            metric_module.update(gen_test_batch(4))
        assert metric_module.trained_batches == update_count
        dist.barrier()


@skip_if_asan_class
class RecMetricDebugModeDistributedTest(MultiProcessTestBase):
    def test_invalid_input_raises_only_on_offending_rank(self) -> None:
        self._run_multi_process_test(
            callable=_run_rank_local_input_validation,
            world_size=2,
            backend="gloo",
        )

    def test_uneven_update_counts_do_not_collect(self) -> None:
        self._run_multi_process_test(
            callable=_run_rank_local_uneven_updates,
            world_size=2,
            backend="gloo",
        )


class MetricsConfigPostInitTest(unittest.TestCase):
    """Test class for MetricsConfig._post_init() validation functionality."""

    def test_post_init_valid_rec_task_indices(self) -> None:
        """Test that _post_init() passes when rec_task_indices are valid."""
        # Setup: create rec_tasks and valid indices
        task1 = RecTaskInfo(name="task1", label_name="label1", prediction_name="pred1")
        task2 = RecTaskInfo(name="task2", label_name="label2", prediction_name="pred2")
        rec_tasks = [task1, task2]

        # Execute: create MetricsConfig with valid rec_task_indices
        config = MetricsConfig(
            rec_tasks=rec_tasks,
            rec_metrics={
                RecMetricEnum.AUC: RecMetricDef(rec_task_indices=[0, 1]),
                RecMetricEnum.NE: RecMetricDef(rec_task_indices=[0]),
            },
        )

        # Assert: config should be created successfully without raising an exception
        self.assertEqual(len(config.rec_tasks), 2)
        self.assertEqual(len(config.rec_metrics), 2)

    def test_post_init_empty_rec_task_indices(self) -> None:
        """Test that _post_init() passes when rec_task_indices is empty."""
        # Setup: create rec_tasks but use empty indices
        task = RecTaskInfo(name="task", label_name="label", prediction_name="pred")
        rec_tasks = [task]

        # Execute: create MetricsConfig with empty rec_task_indices
        config = MetricsConfig(
            rec_tasks=rec_tasks,
            rec_metrics={
                RecMetricEnum.AUC: RecMetricDef(rec_task_indices=[]),
            },
        )

        # Assert: config should be created successfully with empty indices
        self.assertEqual(len(config.rec_tasks), 1)
        self.assertEqual(config.rec_metrics[RecMetricEnum.AUC].rec_task_indices, [])

    def test_post_init_rejects_k_below_one(self) -> None:
        for k in (0, -1):
            with self.subTest(k=k):
                with self.assertRaises(ValueError) as context:
                    _ = MetricsConfig(num_micro_batches_per_step=k)
                self.assertIn(
                    "num_micro_batches_per_step must be at least 1",
                    str(context.exception),
                )

    def test_post_init_accepts_k_of_one_and_above(self) -> None:
        for k in (1, 4):
            with self.subTest(k=k):
                self.assertEqual(
                    MetricsConfig(
                        num_micro_batches_per_step=k
                    ).num_micro_batches_per_step,
                    k,
                )

    def test_post_init_rejects_a_non_member_loss_aggregation(self) -> None:
        # A plain string compares equal to the member but is not it, and the consumers
        # select with `is` -- so coercing would hide the typo this rejects.
        not_members: Dict[str, Any] = {"task1:loss": "sum"}
        with self.assertRaises(ValueError) as context:
            _ = MetricsConfig(loss_aggregation=not_members)
        error_message = str(context.exception)
        self.assertIn("loss_aggregation['task1:loss']", error_message)
        self.assertIn("must be a LossAggregation member", error_message)

    def test_post_init_accepts_real_loss_aggregation_members(self) -> None:
        config = MetricsConfig(
            loss_aggregation={
                "task1:loss": LossAggregation.SUM,
                "task2:loss": LossAggregation.MERGEABLE_RATIO,
            },
        )
        self.assertIs(config.loss_aggregation["task1:loss"], LossAggregation.SUM)

    def test_post_init_raises_when_rec_tasks_is_none(self) -> None:
        """Test that _post_init() raises ValueError when rec_tasks is None but rec_task_indices is specified."""
        # Setup: prepare to create config with None rec_tasks but specified indices

        # Execute & Assert: should raise ValueError about rec_tasks being None
        with self.assertRaises(ValueError) as context:
            _ = MetricsConfig(
                # pyrefly: ignore[bad-argument-type]
                rec_tasks=None,
                rec_metrics={
                    RecMetricEnum.AUC: RecMetricDef(rec_task_indices=[0]),
                },
            )

        error_message = str(context.exception)
        self.assertIn("rec_task_indices [0] is specified", error_message)
        self.assertIn("but rec_tasks is None", error_message)
        self.assertIn("for metric auc", error_message)

    def test_post_init_raises_when_rec_task_index_out_of_range(self) -> None:
        """Test that _post_init() raises ValueError when rec_task_index is out of range."""
        # Setup: create single rec_task but try to access index 1
        task = RecTaskInfo(name="task", label_name="label", prediction_name="pred")
        rec_tasks = [task]

        # Execute & Assert: should raise ValueError about index out of range
        with self.assertRaises(ValueError) as context:
            _ = MetricsConfig(
                rec_tasks=rec_tasks,
                rec_metrics={
                    RecMetricEnum.NE: RecMetricDef(
                        rec_task_indices=[1]
                    ),  # Index 1 doesn't exist
                },
            )

        error_message = str(context.exception)
        self.assertIn("rec_task_indices 1 is out of range", error_message)
        self.assertIn("of 1 tasks", error_message)
        self.assertIn("for metric ne", error_message)


@skip_if_asan_class
class MetricModuleDistributedTest(MultiProcessTestBase):

    def setUp(self, backend: str = "nccl") -> None:
        super().setUp()
        self.backend = backend

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.skipTest("CUDA required for distributed test")

    def test_metric_module_gather_state(self) -> None:
        world_size = 2
        backend = "nccl"
        # use NE to test torch.Tensor state and AUC to test List[torch.Tensor] state
        metrics_config = MetricsConfig(
            rec_tasks=[DefaultTaskInfo],
            rec_metrics={
                RecMetricEnum.NE: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
                ),
                RecMetricEnum.AUC: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
                ),
            },
            throughput_metric=ThroughputDef(),
            state_metrics=[],
        )
        batch_size = 128

        self._run_multi_process_test(
            callable=metric_module_gather_state,
            world_size=world_size,
            backend=backend,
            batch_size=batch_size,
            config=metrics_config,
        )


@skip_if_asan_class
class MetricModuleGlooDistributedTest(MultiProcessTestBase):
    """
    Distributed tests using GLOO backend (works on CPU).
    Tests _get_metric_states functionality with torch.cat optimization.
    """

    def setUp(self) -> None:
        super().setUp()
        self.device = torch.device("cpu")

    def test_get_metric_states_list_reduction(self) -> None:
        """
        Test _get_metric_states with list states and concatenation reduction.
        Validates the torch.cat optimization for AUC-like metrics.
        """
        world_size = 2
        backend = "gloo"

        self._run_multi_process_test(
            callable=_test_get_metric_states_with_list_reduction,
            world_size=world_size,
            backend=backend,
        )

    def test_get_metric_states_tensor_reduction(self) -> None:
        """
        Test _get_metric_states with tensor states and sum reduction.
        Validates standard reduction for NE-like metrics.
        """
        world_size = 2
        backend = "gloo"

        self._run_multi_process_test(
            callable=_test_get_metric_states_with_tensor_reduction,
            world_size=world_size,
            backend=backend,
        )

    def test_get_metric_states_single_tensor(self) -> None:
        """
        Test _get_metric_states with a single tensor in the list.
        Edge case validation.
        """
        world_size = 2
        backend = "gloo"

        self._run_multi_process_test(
            callable=_test_get_metric_states_with_single_tensor,
            world_size=world_size,
            backend=backend,
        )

    def test_get_metric_states_reduction_fn_none(self) -> None:
        """
        Test _get_metric_states with reduction_fn=None.
        Validates that no TypeError is raised when reduction_fn is None.
        """
        world_size = 2
        backend = "gloo"

        self._run_multi_process_test(
            callable=_test_get_metric_states_with_reduction_fn_none,
            world_size=world_size,
            backend=backend,
        )

    def test_get_metric_states_asymmetric_batches(self) -> None:
        """
        Test _get_metric_states with different batch values across ranks.

        This validates that the torch.cat approach correctly aggregates
        data when ranks have different tensor values (same batch count).
        - Rank 0: 3 batch updates with values 1-6
        - Rank 1: 3 batch updates with values 7-12
        """
        world_size = 2
        backend = "gloo"

        self._run_multi_process_test(
            callable=_test_get_metric_states_with_asymmetric_batches,
            world_size=world_size,
            backend=backend,
        )


def _test_get_metric_states_with_list_reduction(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    """Test _get_metric_states with list states and concatenation reduction (AUC-like)."""
    with MultiProcessContext(rank, world_size, backend) as ctx:
        # Create mock metric with list state using concatenation reduction
        tasks = gen_test_tasks(["task1"])
        mock_metric = MockRecMetric(
            world_size=world_size,
            my_rank=rank,
            batch_size=10,
            tasks=tasks,
            is_tensor_list=True,
            reduction_fn=_state_reduction,
            initial_states={"predictions": []},
        )

        # Each rank appends different local tensors to simulate batch updates
        # Rank 0: [[1, 2], [3, 4]] -> after local concat: [1, 2, 3, 4]
        # Rank 1: [[5, 6], [7, 8]] -> after local concat: [5, 6, 7, 8]
        # After global gather: [[1, 2, 3, 4], [5, 6, 7, 8]] -> reduction -> [[1, 2, 3, 4, 5, 6, 7, 8]]
        if rank == 0:
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[1.0, 2.0]], device=ctx.device)}
            )
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[3.0, 4.0]], device=ctx.device)}
            )
        else:  # rank == 1
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[5.0, 6.0]], device=ctx.device)}
            )
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[7.0, 8.0]], device=ctx.device)}
            )

        # Execute: Call _get_metric_states
        metric_module = RecMetricModule(
            batch_size=10,
            world_size=world_size,
            rec_tasks=tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        result = metric_module._get_metric_states(
            metric=mock_metric,
            world_size=world_size,
            # pyrefly: ignore [bad-argument-type]
            process_group=ctx.pg or dist.group.WORLD,
        )

        # Assert: Verify result matches expected
        # With torch.cat approach:
        # - Local concat: rank0 [1,2,3,4], rank1 [5,6,7,8]
        # - All-gather produces [[1,2,3,4], [5,6,7,8]]
        # - _state_reduction concatenates: [1,2,3,4,5,6,7,8]
        expected = [
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], device=ctx.device)
        ]

        actual = result["task1"]["predictions"]

        assert len(actual) == len(
            expected
        ), f"Expected {len(expected)} tensors, got {len(actual)}"
        torch.testing.assert_close(
            actual[0],
            expected[0],
            msg="Mismatch in gathered predictions",
        )


def _test_get_metric_states_with_tensor_reduction(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    """Test _get_metric_states with tensor states and sum reduction (NE-like)."""
    with MultiProcessContext(rank, world_size, backend) as ctx:
        # Create mock metric with tensor state using sum reduction
        tasks = gen_test_tasks(["task1"])
        initial_value = torch.tensor(
            [float(rank + 1)], device=ctx.device
        )  # Rank 0: [1.0], Rank 1: [2.0]
        mock_metric = MockRecMetric(
            world_size=world_size,
            my_rank=rank,
            batch_size=10,
            tasks=tasks,
            is_tensor_list=False,
            reduction_fn="sum",
            initial_states={"state1": initial_value},
        )

        # Execute: Call _get_metric_states
        metric_module = RecMetricModule(
            batch_size=10,
            world_size=world_size,
            rec_tasks=tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        result = metric_module._get_metric_states(
            metric=mock_metric,
            world_size=world_size,
            # pyrefly: ignore [bad-argument-type]
            process_group=ctx.pg or dist.group.WORLD,
        )

        # Assert: Verify result matches expected
        # Expected: sum([rank0_value, rank1_value]) = sum([1.0, 2.0]) = 3.0
        expected = torch.tensor([3.0], device=ctx.device)

        actual = result["task1"]["state1"]

        torch.testing.assert_close(
            actual,
            expected,
            msg="Mismatch in summed state",
        )


def _test_get_metric_states_with_single_tensor(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    """Test _get_metric_states with a single tensor in the list (edge case)."""
    with MultiProcessContext(rank, world_size, backend) as ctx:

        # Create mock metric with list state containing a single tensor
        tasks = gen_test_tasks(["task1"])
        mock_metric = MockRecMetric(
            world_size=world_size,
            my_rank=rank,
            batch_size=10,
            tasks=tasks,
            is_tensor_list=True,
            reduction_fn=_state_reduction,
            initial_states={"predictions": []},
        )

        # Each rank has a single tensor
        # Rank 0: [[1, 2]]
        # Rank 1: [[3, 4]]
        # After local concat (no-op since single tensor): [1, 2] and [3, 4]
        # After global gather+concat: [[1, 2, 3, 4]]
        if rank == 0:
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[1.0, 2.0]], device=ctx.device)}
            )
        else:  # rank == 1
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[3.0, 4.0]], device=ctx.device)}
            )

        # Execute: Call _get_metric_states
        metric_module = RecMetricModule(
            batch_size=10,
            world_size=world_size,
            rec_tasks=tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        result = metric_module._get_metric_states(
            metric=mock_metric,
            world_size=world_size,
            # pyrefly: ignore [bad-argument-type]
            process_group=ctx.pg or dist.group.WORLD,
        )

        # Assert: Verify result matches expected
        # Expected: [[1, 2, 3, 4]]
        expected = [torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=ctx.device)]

        actual = result["task1"]["predictions"]

        assert len(actual) == len(
            expected
        ), f"Expected {len(expected)} tensors, got {len(actual)}"
        torch.testing.assert_close(
            actual[0],
            expected[0],
            msg="Mismatch in gathered predictions for single tensor case",
        )


def _test_get_metric_states_with_reduction_fn_none(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    """Test _get_metric_states with reduction_fn=None (no reduction applied)."""
    with MultiProcessContext(rank, world_size, backend) as ctx:
        # Create mock metric with list state and reduction_fn=None
        tasks = gen_test_tasks(["task1"])
        mock_metric = MockRecMetric(
            world_size=world_size,
            my_rank=rank,
            batch_size=10,
            tasks=tasks,
            is_tensor_list=True,
            # pyrefly: ignore [bad-argument-type]
            reduction_fn=None,  # No reduction
            initial_states={"predictions": []},
        )

        # Each rank has tensors
        if rank == 0:
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[1.0, 2.0]], device=ctx.device)}
            )
        else:  # rank == 1
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[3.0, 4.0]], device=ctx.device)}
            )

        # Execute: Call _get_metric_states - should NOT raise TypeError
        metric_module = RecMetricModule(
            batch_size=10,
            world_size=world_size,
            rec_tasks=tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        result = metric_module._get_metric_states(
            metric=mock_metric,
            world_size=world_size,
            # pyrefly: ignore [bad-argument-type]
            process_group=ctx.pg or dist.group.WORLD,
        )

        # Assert: With reduction_fn=None, gathered_list is returned as-is
        # After torch.cat locally: rank0 [1,2], rank1 [3,4]
        # After all-gather: [[1,2], [3,4]] (list of 2 tensors)
        actual = result["task1"]["predictions"]

        # Should be a list of 2 tensors (one per rank)
        assert isinstance(actual, list), f"Expected list, got {type(actual)}"
        assert (
            len(actual) == world_size
        ), f"Expected {world_size} tensors, got {len(actual)}"

        # Verify the gathered tensors
        expected_rank0 = torch.tensor([[1.0, 2.0]], device=ctx.device)
        expected_rank1 = torch.tensor([[3.0, 4.0]], device=ctx.device)

        torch.testing.assert_close(
            actual[0],
            expected_rank0,
            msg="Mismatch in rank 0 gathered tensor",
        )
        torch.testing.assert_close(
            actual[1],
            expected_rank1,
            msg="Mismatch in rank 1 gathered tensor",
        )


def _test_get_metric_states_with_asymmetric_batches(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    """
    Test _get_metric_states with different batch values across ranks.

    This validates that the torch.cat approach correctly aggregates data
    when ranks have different tensor values (same batch count).
    """
    with MultiProcessContext(rank, world_size, backend) as ctx:
        tasks = gen_test_tasks(["task1"])
        mock_metric = MockRecMetric(
            world_size=world_size,
            my_rank=rank,
            batch_size=10,
            tasks=tasks,
            is_tensor_list=True,
            reduction_fn=_state_reduction,
            initial_states={"predictions": []},
        )

        # Same batch count per rank (3 batches each), but different values:
        # Rank 0: [[1,2], [3,4], [5,6]] -> local concat -> [1,2,3,4,5,6]
        # Rank 1: [[7,8], [9,10], [11,12]] -> local concat -> [7,8,9,10,11,12]
        # After all_gather: [[1,2,3,4,5,6], [7,8,9,10,11,12]]
        # After reduction (_state_reduction = concat): [1,2,3,4,5,6,7,8,9,10,11,12]
        if rank == 0:
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[1.0, 2.0]], device=ctx.device)}
            )
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[3.0, 4.0]], device=ctx.device)}
            )
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[5.0, 6.0]], device=ctx.device)}
            )
        else:  # rank == 1
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[7.0, 8.0]], device=ctx.device)}
            )
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[9.0, 10.0]], device=ctx.device)}
            )
            mock_metric.append_to_computation_states(
                {"predictions": torch.tensor([[11.0, 12.0]], device=ctx.device)}
            )

        metric_module = RecMetricModule(
            batch_size=10,
            world_size=world_size,
            rec_tasks=tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        result = metric_module._get_metric_states(
            metric=mock_metric,
            world_size=world_size,
            # pyrefly: ignore [bad-argument-type]
            process_group=ctx.pg or dist.group.WORLD,
        )

        # Expected: All values concatenated in rank order
        # [1,2,3,4,5,6,7,8,9,10,11,12]
        expected = [
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
                device=ctx.device,
            )
        ]

        actual = result["task1"]["predictions"]

        assert len(actual) == len(
            expected
        ), f"Expected {len(expected)} tensors, got {len(actual)}"
        torch.testing.assert_close(
            actual[0],
            expected[0],
            msg="Mismatch in gathered predictions with multiple batches per rank",
        )


class RecMetricModuleLossAccumulationTest(unittest.TestCase):
    """Tests for RecMetricModule loss accumulation across micro-batches.

    The _accumulate_loss_metrics() path sums per-task loss values during
    update() / update_micro_batch() and returns their average in compute().
    Accumulators are NOT cleared by compute() — they are cleared by
    reset_loss_metrics() at the start of each optimizer step.
    This ensures compute() always reports the loss from the most recent
    step, not a running average across the compute interval.
    """

    def _make_module(self) -> RecMetricModule:
        module = RecMetricModule(batch_size=128, world_size=1)
        # Declare gradient accumulation. The micro-batch gate is driven by CONFIG
        # (MetricsConfig.num_micro_batches_per_step -> set_under_micro_batching), not by
        # having called update_micro_batch(), so a K>1 sequence has to say it is K>1. The
        # K=1 contract tests below deliberately do NOT call this.
        module.set_under_micro_batching(True)
        return module

    def _make_k1_module(self) -> RecMetricModule:
        """A module for the K=1 contract tests: GA is NOT declared, which is exactly
        what a caller that never enables micro-batching looks like."""
        return RecMetricModule(batch_size=128, world_size=1)

    def _step(
        self, module: RecMetricModule, micros: List[Dict[str, torch.Tensor]]
    ) -> DeferrableMetrics:
        """Drive one optimizer step of len(micros) micro-batches, then compute().

        K-1 x update_micro_batch() then one update(). The chain is armed by
        ``_make_module`` declaring GA in the config, not by this call sequence.
        """
        module.reset_loss_metrics()
        for micro in micros[:-1]:
            module.update_micro_batch(micro)
        module.update(micros[-1])
        return module.compute()

    # The micro-batch loss chain is unreachable at K=1

    def test_k1_bare_update_publishes_no_loss(self) -> None:
        """A K=1 job never declares GA, so the micro-batch loss chain is unreachable and
        no loss key is published."""
        module = self._make_k1_module()
        module.update({"ctr:loss": torch.tensor(0.5)})
        result = module.compute()
        self.assertNotIn("ctr:loss", result)

    def test_k1_published_keys_are_unchanged_by_loss_keys(self) -> None:
        """The K=1 contract: feeding loss values through a pure-update() sequence
        publishes exactly the key set the same sequence publishes without them.

        This is the assertion that makes the loss chain safe to land unconditionally --
        every K=1 caller reaches update(), and none of them may gain a key.
        """
        with_loss = self._make_k1_module()
        with_loss.update(
            {"prediction": torch.tensor(0.5), "ctr:loss": torch.tensor(1.0)}
        )
        with_loss.update(
            {"prediction": torch.tensor(0.7), "ctr:loss": torch.tensor(3.0)}
        )

        without_loss = self._make_k1_module()
        without_loss.update({"prediction": torch.tensor(0.5)})
        without_loss.update({"prediction": torch.tensor(0.7)})

        self.assertEqual(
            sorted(with_loss.compute().keys()), sorted(without_loss.compute().keys())
        )

    # Config-driven gate: the key set does not depend on call history

    def test_published_key_set_is_history_independent_under_ga(self) -> None:
        """The property the config-driven scope selection exists to give us.

        Eval skips the micro-batch loop, so a gate latched by call history would make the
        SAME eval publish a different key set depending on whether an accumulating training
        step had already run. Sums reset correctly either way; only key PRESENCE varied.
        Selecting the scope from config removes the history dependence, so pin it.
        """
        before = self._make_module()
        keys_before = sorted(before.compute().keys())

        after = self._make_module()
        after.reset_loss_metrics()
        after.update_micro_batch({"ctr:loss": torch.tensor(0.5)})
        after.update({"ctr:loss": torch.tensor(0.7)})
        after.compute()
        after.reset_loss_metrics()
        keys_after = sorted(after.compute().keys())

        self.assertEqual(keys_before, keys_after)

    def test_gate_is_off_until_declared(self) -> None:
        """An undeclared scope refuses the micro-batch call rather than latching on it."""
        module = self._make_k1_module()
        self.assertFalse(module.under_micro_batching)
        with self.assertRaises(RecMetricException):
            module.update_micro_batch({"ctr:loss": torch.tensor(0.5)})
        self.assertFalse(
            module.under_micro_batching,
            "update_micro_batch() changed the loss-aggregation scope",
        )

    def test_scope_is_the_declared_policy_and_the_bool_is_its_view(self) -> None:
        """The scope is the value the three read sites share; ``under_micro_batching`` is
        only a boolean view of it, so the two can never disagree."""
        module = self._make_k1_module()
        self.assertIs(
            LossAggregationScope.PER_READER_BATCH, module.loss_aggregation_scope
        )
        self.assertFalse(module.under_micro_batching)

        module.set_under_micro_batching(True)
        self.assertIs(
            LossAggregationScope.PER_OPTIMIZER_STEP, module.loss_aggregation_scope
        )
        self.assertTrue(module.under_micro_batching)

        module.set_under_micro_batching(False)
        self.assertIs(
            LossAggregationScope.PER_READER_BATCH, module.loss_aggregation_scope
        )
        self.assertFalse(module.under_micro_batching)

    def test_eval_key_set_identical_before_and_after_first_ga_step(self) -> None:
        """Consequence-2, pinned on ONE module instance.

        An eval interlude skips the micro-batch loop -- a bare ``update()`` + ``compute()``
        -- so a history-latched scope would make the SAME eval publish a loss key only
        after an accumulating step had run on that module. One eval, one accumulating
        step, the identical eval again: the key sets must match.
        """
        module = self._make_module()

        module.reset_loss_metrics()
        module.update({"ctr:loss": torch.tensor(0.5)})
        keys_eval_before = sorted(module.compute().keys())

        module.reset_loss_metrics()
        module.update_micro_batch({"ctr:loss": torch.tensor(0.4)})
        module.update({"ctr:loss": torch.tensor(0.6)})
        module.compute()

        module.reset_loss_metrics()
        module.update({"ctr:loss": torch.tensor(0.5)})
        keys_eval_after = sorted(module.compute().keys())

        self.assertEqual(keys_eval_before, keys_eval_after)

    # Basic accumulation
    def test_two_updates_returns_average(self) -> None:
        """Two micro-batches -> compute() returns the mean of both."""
        module = self._make_module()
        result = self._step(
            module,
            [{"ctr:loss": torch.tensor(0.4)}, {"ctr:loss": torch.tensor(0.6)}],
        )
        self.assertAlmostEqual(float(result["ctr:loss"]), 0.5, places=6)

    def test_k_updates_returns_average(self) -> None:
        """K micro-batches -> compute() returns mean over K values."""
        module = self._make_module()
        values = [0.1, 0.2, 0.3, 0.4]
        result = self._step(module, [{"task:loss": torch.tensor(v)} for v in values])
        expected = sum(values) / len(values)
        self.assertAlmostEqual(float(result["task:loss"]), expected, places=6)

    # Multiple tasks

    def test_multiple_task_losses_averaged_independently(self) -> None:
        """Each task's loss is averaged independently across micro-batches."""
        module = self._make_module()
        result = self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(1.0),
                    "reels_ctr:loss": torch.tensor(2.0),
                },
                {
                    "ctr:loss": torch.tensor(3.0),
                    "reels_ctr:loss": torch.tensor(4.0),
                },
            ],
        )
        self.assertAlmostEqual(float(result["ctr:loss"]), 2.0, places=6)
        self.assertAlmostEqual(float(result["reels_ctr:loss"]), 3.0, places=6)

    # Bare "loss" key

    def test_bare_loss_key_accumulated(self) -> None:
        """The bare 'loss' key (no task prefix) is also accumulated."""
        module = self._make_module()
        result = self._step(
            module, [{"loss": torch.tensor(1.0)}, {"loss": torch.tensor(3.0)}]
        )
        self.assertAlmostEqual(float(result["loss"]), 2.0, places=6)

    def test_bare_loss_and_task_loss_coexist(self) -> None:
        """Bare 'loss' and per-task 'task:loss' keys are accumulated independently."""
        module = self._make_module()
        result = self._step(
            module,
            [
                {"loss": torch.tensor(10.0), "ctr:loss": torch.tensor(1.0)},
                {"loss": torch.tensor(20.0), "ctr:loss": torch.tensor(3.0)},
            ],
        )
        self.assertAlmostEqual(float(result["loss"]), 15.0, places=6)
        self.assertAlmostEqual(float(result["ctr:loss"]), 2.0, places=6)

    # Reset after compute

    def test_reset_loss_metrics_clears_accumulator(self) -> None:
        """Each _step() begins with reset_loss_metrics(), so a step reports only its
        own micro-batches -- not a running average across steps."""
        module = self._make_module()
        first = self._step(
            module, [{"ctr:loss": torch.tensor(1.0)}, {"ctr:loss": torch.tensor(3.0)}]
        )
        self.assertAlmostEqual(float(first["ctr:loss"]), 2.0, places=6)

        second = self._step(
            module, [{"ctr:loss": torch.tensor(10.0)}, {"ctr:loss": torch.tensor(10.0)}]
        )
        self.assertAlmostEqual(float(second["ctr:loss"]), 10.0, places=6)

    def test_compute_without_reset_returns_last_step_loss(self) -> None:
        """Without reset_loss_metrics(), compute() keeps returning the
        most recent accumulated value — this is the intended behavior
        when compute_interval_steps > 1."""
        module = self._make_module()
        first = self._step(
            module, [{"ctr:loss": torch.tensor(5.0)}, {"ctr:loss": torch.tensor(5.0)}]
        )
        second = module.compute()
        self.assertAlmostEqual(
            float(first["ctr:loss"]), float(second["ctr:loss"]), places=6
        )

    def test_compute_without_losses_returns_empty(self) -> None:
        """An armed step carrying no loss keys publishes no loss keys."""
        module = self._make_module()
        result = self._step(
            module,
            [{"prediction": torch.tensor(0.5)}, {"prediction": torch.tensor(0.7)}],
        )
        loss_keys = [k for k in result if k.endswith(":loss") or k == "loss"]
        self.assertEqual(loss_keys, [])

    # Non-loss keys are ignored

    def test_non_loss_keys_not_accumulated(self) -> None:
        """Keys that don't match the loss pattern are NOT accumulated."""
        module = self._make_module()
        micro = {
            "ctr:loss": torch.tensor(1.0),
            "ctr:prediction": torch.tensor(0.7),
            "reg_loss": torch.tensor(0.01),
        }
        result = self._step(module, [micro, micro])
        self.assertIn("ctr:loss", result)
        self.assertNotIn("ctr:prediction", result)
        self.assertNotIn("reg_loss", result)

    # Detach behavior

    def test_accumulated_values_are_detached(self) -> None:
        """Loss tensors are detached before accumulation so they don't
        participate in the backward graph."""
        module = self._make_module()
        t = torch.tensor(1.0, requires_grad=True)
        result = self._step(module, [{"task:loss": t}, {"task:loss": t}])
        val = result["task:loss"]
        assert isinstance(val, torch.Tensor)
        self.assertFalse(val.requires_grad)

    # Equivalence: K micro-batches == single update with mean

    def test_micro_batch_average_equals_full_batch_loss(self) -> None:
        """The core correctness property: averaging K micro-batch losses
        equals the arithmetic mean computed externally."""
        module = self._make_module()
        losses = [0.12, 0.34, 0.56, 0.78]
        result = self._step(module, [{"ctr:loss": torch.tensor(v)} for v in losses])
        expected = sum(losses) / len(losses)
        self.assertAlmostEqual(
            float(result["ctr:loss"]),
            expected,
            places=6,
        )


class UpdateMicroBatchTest(unittest.TestCase):
    """Tests for RecMetricModule.update_micro_batch().

    Verifies that update_micro_batch():
    - Does NOT increment trained_batches (the optimizer-step counter)
    - DOES advance throughput _steps once per micro-batch: every micro-batch is
      a real reader batch of the full configured metrics batch_size (K does NOT
      auto-divide batch_size), so counting all K is what makes examples/sec
      correct; skipping would under-report QPS by K x
    - Does accumulate rec metric state
    - Does accumulate loss metrics
    """

    def _make_module(self) -> RecMetricModule:
        # K=2 so the config-driven micro-batch gate arms via generate_metric_module ->
        # set_under_micro_batching(). This class tests update_micro_batch(), which is by
        # definition a gradient-accumulation job, and going through the config (rather
        # than calling the setter directly) also covers the MetricsConfig plumbing.
        return generate_metric_module(
            TestMetricModule,
            metrics_config=dataclasses.replace(
                DefaultMetricsConfig, num_micro_batches_per_step=2
            ),
            batch_size=128,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )

    def test_does_not_increment_trained_batches(self) -> None:
        module = self._make_module()
        initial = module.trained_batches
        module.update_micro_batch(gen_test_batch(128))
        module.update_micro_batch(gen_test_batch(128))
        self.assertEqual(module.trained_batches, initial)

    def test_updates_throughput_steps_per_micro_batch(self) -> None:
        """update_micro_batch MUST advance throughput once per micro-batch.

        Every micro-batch is a real reader batch of the full configured batch_size (K does
        not auto-divide it), so throughput must count all K for examples/sec to be right.
        The optimizer-step counter trained_batches is NOT advanced."""
        module = self._make_module()
        assert module.throughput_metric is not None
        initial_steps = module.throughput_metric._steps
        module.update_micro_batch(gen_test_batch(128))
        module.update_micro_batch(gen_test_batch(128))
        self.assertEqual(module.throughput_metric._steps, initial_steps + 2)
        # A follow-up full update() advances throughput by one more.
        module.update(gen_test_batch(128))
        self.assertEqual(module.throughput_metric._steps, initial_steps + 3)

    def test_does_accumulate_rec_metric_state(self) -> None:
        module = self._make_module()
        batch = gen_test_batch(128)

        # Get initial weighted_num_samples
        sd_before = module.state_dict()
        initial_samples = sd_before[
            "rec_metrics.rec_metrics.0._metrics_computations.0.weighted_num_samples"
        ].item()

        module.update_micro_batch(batch)

        sd_after = module.state_dict()
        after_samples = sd_after[
            "rec_metrics.rec_metrics.0._metrics_computations.0.weighted_num_samples"
        ].item()

        self.assertGreater(after_samples, initial_samples)

    def test_does_accumulate_loss_metrics(self) -> None:
        module = self._make_module()
        batch1 = gen_test_batch(128)
        batch1["ctr:loss"] = torch.tensor(0.5)
        batch2 = gen_test_batch(128)
        batch2["ctr:loss"] = torch.tensor(1.5)
        module.update_micro_batch(batch1)
        module.update_micro_batch(batch2)
        result = module.compute()
        loss_value = result["ctr:loss"]
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        self.assertAlmostEqual(float(loss_value), 1.0, places=6)

    def test_micro_batch_then_update_increments_trained_batches_once(self) -> None:
        """K-1 update_micro_batch() + 1 update() increments trained_batches by 1."""
        module = self._make_module()
        initial = module.trained_batches
        batch = gen_test_batch(128)

        module.update_micro_batch(batch)
        module.update_micro_batch(batch)
        module.update_micro_batch(batch)
        module.update(batch)

        self.assertEqual(module.trained_batches, initial + 1)


class ValidateBatchSizeStagesTest(unittest.TestCase):
    def test_none_is_valid(self) -> None:
        # Should not raise
        validate_batch_size_stages(None)

    def test_last_stage_max_iters_not_none_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_batch_size_stages([BatchSizeStage(256, 100)])

    def test_valid_stages(self) -> None:
        # Should not raise
        validate_batch_size_stages(
            [BatchSizeStage(256, 100), BatchSizeStage(512, None)]
        )

    def test_single_stage_valid(self) -> None:
        # Should not raise - single stage with max_iters=None
        validate_batch_size_stages([BatchSizeStage(256, None)])


class LossAggregationContractTest(unittest.TestCase):
    """G-U1a: per-key loss aggregation across the micro-batches of one optimizer step.

    The contract is producer-declared: a loss that is a mergeable ratio emits its
    effective guarded denominator as ``"{key}" + LOSS_DENOM_SUFFIX`` and is recombined
    as ``sum(l_i * d_i) / sum(d_i)``. A key with no denominator keeps the legacy
    per-key mean. ``MetricsConfig.loss_aggregation`` overrides only the exceptions.
    """

    def _make_module(
        self, overrides: Optional[Dict[str, LossAggregation]] = None
    ) -> RecMetricModule:
        module = RecMetricModule(batch_size=128, world_size=1)
        module.set_loss_aggregation(overrides)
        # The K>1 aggregation contract only applies to a GA job, and the gate is config-driven.
        module.set_under_micro_batching(True)
        return module

    def _make_k1_module(
        self, overrides: Optional[Dict[str, LossAggregation]] = None
    ) -> RecMetricModule:
        """A K=1 job: GA is NOT declared, so the micro-batch loss chain stays gated off."""
        module = RecMetricModule(batch_size=128, world_size=1)
        module.set_loss_aggregation(overrides)
        return module

    def test_the_noop_module_accepts_an_override_without_an_accumulator(self) -> None:
        """NoOp has no ``_loss_acc``, so the inherited setter would raise."""
        module = NoOpMetricModule()
        module.set_loss_aggregation({"ctr:loss": LossAggregation.SUM})
        self.assertFalse(hasattr(module, "_loss_acc"))

    def _step(
        self, module: RecMetricModule, micros: List[Dict[str, torch.Tensor]]
    ) -> DeferrableMetrics:
        """Run one optimizer step of len(micros) micro-batches and compute()."""
        module.reset_loss_metrics()
        for micro in micros[:-1]:
            module.update_micro_batch(micro)
        module.update(micros[-1])
        return module.compute()

    # A key present in only j of K micro-batches

    def test_key_missing_from_some_micro_batches_divides_by_its_own_count(
        self,
    ) -> None:
        """A key emitted by 2 of 4 micro-batches is divided by 2, not by 4.

        A single call count shared across every key divides a sparsely-emitted
        auxiliary loss by the window size rather than by its own emission count,
        scaling it down by j/K without any error.
        """
        module = self._make_module()
        result = self._step(
            module,
            [
                {"ctr:loss": torch.tensor(1.0), "aux:loss": torch.tensor(4.0)},
                {"ctr:loss": torch.tensor(2.0)},
                {"ctr:loss": torch.tensor(3.0)},
                {"ctr:loss": torch.tensor(4.0), "aux:loss": torch.tensor(6.0)},
            ],
        )
        self.assertAlmostEqual(float(result["ctr:loss"]), 2.5, places=6)
        # 10 / 2, not 10 / 4.
        self.assertAlmostEqual(float(result["aux:loss"]), 5.0, places=6)

    # Producer-declared mergeable ratio

    def test_denominator_recovers_the_unsplit_batch_value(self) -> None:
        """Sum(l*d)/Sum(d) equals what the un-split batch would have produced.

        Models a `drop_zero_weight` head: micro 0 contributes 6.0 over 3 nonzero
        examples, micro 1 contributes 10.0 over 1. The un-split batch is
        16/4 = 4.0, whereas the legacy unweighted mean of the two micro-batch
        scalars would report (2.0 + 10.0)/2 = 6.0.
        """
        module = self._make_module()
        result = self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(2.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(3.0),
                },
                {
                    "ctr:loss": torch.tensor(10.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(1.0),
                },
            ],
        )
        self.assertAlmostEqual(float(result["ctr:loss"]), 4.0, places=6)

    def test_equal_denominators_match_the_legacy_mean(self) -> None:
        """With equal per-micro denominators the ratio collapses to the mean.

        This is why annotating a producer is safe to do unconditionally: it only
        changes a value where the denominators actually differ.
        """
        annotated = self._make_module()
        ratio = self._step(
            annotated,
            [
                {
                    "ctr:loss": torch.tensor(0.4),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(128.0),
                },
                {
                    "ctr:loss": torch.tensor(0.6),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(128.0),
                },
            ],
        )
        legacy = self._step(
            self._make_module(),
            [{"ctr:loss": torch.tensor(0.4)}, {"ctr:loss": torch.tensor(0.6)}],
        )
        self.assertAlmostEqual(
            float(ratio["ctr:loss"]), float(legacy["ctr:loss"]), places=6
        )

    def test_k1_publishes_no_loss_with_or_without_a_denominator(self) -> None:
        """At K=1 the chain is gated off, so annotating a producer is a no-op.

        This is the whole reason the denominator path ships unconditionally rather than
        behind a flag: a K=1 model cannot tell the difference, because it publishes the
        same key set either way.
        """
        with_denom = self._make_k1_module()
        with_denom.reset_loss_metrics()
        with_denom.update(
            {
                "ctr:loss": torch.tensor(0.37),
                f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(1024.0),
            }
        )
        without = self._make_k1_module()
        without.reset_loss_metrics()
        without.update({"ctr:loss": torch.tensor(0.37)})
        self.assertEqual(
            sorted(with_denom.compute().keys()), sorted(without.compute().keys())
        )
        self.assertNotIn("ctr:loss", with_denom.compute())

    def test_denominator_key_is_not_itself_reported_as_a_loss(self) -> None:
        """`:loss_denom` must not be mistaken for a published loss metric."""
        module = self._make_module()
        result = self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(1.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(8.0),
                },
                {
                    "ctr:loss": torch.tensor(1.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(8.0),
                },
            ],
        )
        self.assertIn("ctr:loss", result)
        self.assertNotIn(f"ctr:loss{LOSS_DENOM_SUFFIX}", result)

    def test_keys_aggregate_independently_within_a_step(self) -> None:
        """An annotated key and a legacy key coexist without interfering."""
        module = self._make_module()
        result = self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(2.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(3.0),
                    "regularization:loss": torch.tensor(1.0),
                },
                {
                    "ctr:loss": torch.tensor(10.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(1.0),
                    "regularization:loss": torch.tensor(3.0),
                },
            ],
        )
        self.assertAlmostEqual(float(result["ctr:loss"]), 4.0, places=6)
        self.assertAlmostEqual(float(result["regularization:loss"]), 2.0, places=6)

    def test_reset_clears_denominator_state_across_steps(self) -> None:
        """Step N+1 must not inherit step N's denominator accumulation."""
        module = self._make_module()
        self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(2.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(3.0),
                },
                {
                    "ctr:loss": torch.tensor(2.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(3.0),
                },
            ],
        )
        second = self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(5.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(7.0),
                },
                {
                    "ctr:loss": torch.tensor(5.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(7.0),
                },
            ],
        )
        self.assertAlmostEqual(float(second["ctr:loss"]), 5.0, places=6)

    # Config overrides (the exceptions)

    def test_sum_override_accumulates_without_dividing(self) -> None:
        module = self._make_module({"cov:loss": LossAggregation.SUM})
        result = self._step(
            module,
            [{"cov:loss": torch.tensor(1.5)}, {"cov:loss": torch.tensor(2.5)}],
        )
        self.assertAlmostEqual(float(result["cov:loss"]), 4.0, places=6)

    def test_sum_override_wins_over_an_emitted_denominator(self) -> None:
        """A SUM pin defends against a producer that emits a bogus denominator."""
        module = self._make_module({"cov:loss": LossAggregation.SUM})
        result = self._step(
            module,
            [
                {
                    "cov:loss": torch.tensor(1.5),
                    f"cov:loss{LOSS_DENOM_SUFFIX}": torch.tensor(4.0),
                },
                {
                    "cov:loss": torch.tensor(2.5),
                    f"cov:loss{LOSS_DENOM_SUFFIX}": torch.tensor(4.0),
                },
            ],
        )
        self.assertAlmostEqual(float(result["cov:loss"]), 4.0, places=6)

    def test_non_mergeable_override_pins_a_key_to_the_legacy_path(self) -> None:
        """NON_MERGEABLE ignores an emitted denominator (mis-annotated producer)."""
        module = self._make_module({"ctr:loss": LossAggregation.NON_MERGEABLE})
        result = self._step(
            module,
            [
                {
                    "ctr:loss": torch.tensor(2.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(3.0),
                },
                {
                    "ctr:loss": torch.tensor(10.0),
                    f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(1.0),
                },
            ],
        )
        # Legacy per-key mean, not the 4.0 the ratio would give.
        self.assertAlmostEqual(float(result["ctr:loss"]), 6.0, places=6)

    def test_declared_mergeable_ratio_without_denominator_raises(self) -> None:
        """Fail closed: the override asserts the producer emits a denominator."""
        module = self._make_module({"ctr:loss": LossAggregation.MERGEABLE_RATIO})
        module.reset_loss_metrics()
        with self.assertRaises(RecMetricException):
            module.update_micro_batch({"ctr:loss": torch.tensor(1.0)})

    def test_denominator_emitted_on_only_some_micro_batches_raises(self) -> None:
        """A half-annotated producer would mix `l*d` and raw `l` terms."""
        module = self._make_module()
        module.reset_loss_metrics()
        module.update_micro_batch(
            {
                "ctr:loss": torch.tensor(2.0),
                f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(3.0),
            }
        )
        with self.assertRaises(RecMetricException):
            module.update({"ctr:loss": torch.tensor(10.0)})

    # Legacy identity

    def test_unannotated_step_is_byte_identical_to_legacy(self) -> None:
        """With no denominators and no overrides, nothing changes."""
        module = self._make_module()
        result = self._step(
            module,
            [
                {"a:loss": torch.tensor(1.0), "b:loss": torch.tensor(2.0)},
                {"a:loss": torch.tensor(3.0), "b:loss": torch.tensor(6.0)},
            ],
        )
        self.assertAlmostEqual(float(result["a:loss"]), 2.0, places=6)
        self.assertAlmostEqual(float(result["b:loss"]), 4.0, places=6)

    # Atomicity: a rejected micro-batch must not leave half of itself behind

    def _accumulator_state(self, module: RecMetricModule) -> Dict[str, Any]:
        """A detached copy of every accumulator the loss chain writes."""
        return {
            "sums": {k: v.clone() for k, v in module._loss_acc.sums.items()},
            "key_counts": dict(module._loss_acc.key_counts),
            "denom_sums": {
                k: v.clone() for k, v in module._loss_acc.denom_sums.items()
            },
            "ratio_keys": set(module._loss_acc.ratio_keys),
            "count": module._loss_acc.count,
        }

    def _assert_accumulators_equal(
        self, module: RecMetricModule, expected: Dict[str, Any]
    ) -> None:
        actual = self._accumulator_state(module)
        self.assertEqual(set(actual["sums"]), set(expected["sums"]), "sums keys")
        for key, value in expected["sums"].items():
            self.assertTrue(
                torch.equal(actual["sums"][key], value),
                f"sums[{key!r}] moved: {value} -> {actual['sums'][key]}",
            )
        self.assertEqual(actual["key_counts"], expected["key_counts"])
        self.assertEqual(set(actual["denom_sums"]), set(expected["denom_sums"]))
        for key, value in expected["denom_sums"].items():
            self.assertTrue(
                torch.equal(actual["denom_sums"][key], value),
                f"denom_sums[{key!r}] moved",
            )
        self.assertEqual(actual["ratio_keys"], expected["ratio_keys"])
        self.assertEqual(actual["count"], expected["count"])

    def test_rejected_micro_batch_does_not_fold_in_its_earlier_keys(self) -> None:
        """A contract violation must be all-or-nothing across the WHOLE model_out.

        The validation walk is per key, so a bad key raises only after earlier keys were
        folded in; the staging/commit split is what makes that atomic. ``ok:loss`` is
        emitted BEFORE the offending ``bad:loss`` in the same dict, so a non-atomic
        implementation leaves ``ok:loss`` in the accumulators and double-counts it.
        """
        module = self._make_module({"bad:loss": LossAggregation.MERGEABLE_RATIO})
        module.reset_loss_metrics()
        before = self._accumulator_state(module)

        with self.assertRaisesRegex(RecMetricException, "MERGEABLE_RATIO"):
            module.update_micro_batch(
                {
                    "ok:loss": torch.tensor(1.0),
                    "bad:loss": torch.tensor(2.0),  # declared ratio, no denominator
                }
            )

        self._assert_accumulators_equal(module, before)

    def test_rejected_micro_batch_leaves_the_previous_micro_intact(self) -> None:
        """The rejection must not corrupt micro-batches that already committed.

        Falsifier for the test above, which starts from empty accumulators and so
        would also pass if a raise *cleared* everything instead of rolling back.
        Here a good micro-batch commits first; the step must still be able to
        publish it after the bad one is rejected.
        """
        module = self._make_module()
        module.reset_loss_metrics()
        module.update_micro_batch(
            {
                "ctr:loss": torch.tensor(1.0),
                f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(4.0),
            }
        )
        after_good = self._accumulator_state(module)

        # Denominator present on the first micro but absent here: no correct divisor.
        with self.assertRaisesRegex(RecMetricException, "on some"):
            module.update_micro_batch({"ctr:loss": torch.tensor(2.0)})

        self._assert_accumulators_equal(module, after_good)

        # And the step still publishes the surviving micro-batch: 4.0 / 4.0.
        module.update(
            {
                "ctr:loss": torch.tensor(3.0),
                f"ctr:loss{LOSS_DENOM_SUFFIX}": torch.tensor(4.0),
            }
        )
        result = module.compute()
        self.assertAlmostEqual(float(result["ctr:loss"]), 2.0, places=6)
