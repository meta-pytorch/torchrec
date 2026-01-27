#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import dataclasses
import logging
import multiprocessing
import os
import tempfile
import unittest
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.metric_module import (
    generate_metric_module,
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
    MetricsConfig,
    RecMetricDef,
    RecMetricEnum,
    ThroughputDef,
)
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecMetricException, RecMetricList, RecTaskInfo
from torchrec.metrics.test_utils import gen_test_batch
from torchrec.metrics.throughput import ThroughputMetric
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
                metric_module.rec_metrics.rec_metrics[0]
                ._metrics_computations[0]
                .predictions
            ),
            1,  # The predictions state is a list containing 1 tensor value
        )

        # 1. After the metric module is created
        tc.assertEqual(
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 1),
            # The 1st 1 is the number of tasks; the 2nd 1 is the default value length
        )

        metric_module.sync()
        tc.assertEqual(
            metric_module.rec_metrics.rec_metrics[0]
            ._metrics_computations[0]
            .labels[0]
            .size(),
            (1, 2),
        )

        metric_module.unsync()
        tc.assertEqual(
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

        # pyre-fixme[53]: Captured variable `batch` is not annotated.
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
        self.assertIsInstance(result, dict)
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
        self.assertIsInstance(result, dict)
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
            process_group=dist.new_group(ranks=[rank], backend="nccl"),
        )
        new_metric_module.load_pre_compute_states(states)
        new_computed_value = new_metric_module.compute()

        for metric, tensor in computed_value.items():
            new_tensor = new_computed_value[metric]
            torch.testing.assert_close(tensor, new_tensor, check_device=False)

        metric_module.shutdown()


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

    def test_post_init_raises_when_rec_tasks_is_none(self) -> None:
        """Test that _post_init() raises ValueError when rec_tasks is None but rec_task_indices is specified."""
        # Setup: prepare to create config with None rec_tasks but specified indices

        # Execute & Assert: should raise ValueError about rec_tasks being None
        with self.assertRaises(ValueError) as context:
            config = MetricsConfig(
                rec_tasks=None,  # pyre-ignore[6]: Intentionally passing None for testing
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
            config = MetricsConfig(
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
