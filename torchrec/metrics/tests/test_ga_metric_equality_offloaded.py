#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import unittest
from typing import Any

import torch
import torch.distributed as dist
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.metric_module import generate_metric_module
from torchrec.metrics.metrics_config import (
    DefaultTaskInfo,
    MetricsConfig,
    RecComputeMode,
    RecMetricDef,
    RecMetricEnum,
)
from torchrec.test_utils import init_process_group_single_rank

_METRICS: list[RecMetricEnum] = [
    RecMetricEnum.NE,
    RecMetricEnum.CTR,
    RecMetricEnum.MSE,
    RecMetricEnum.AUC,
    RecMetricEnum.CALIBRATION,
]

_BATCH_SIZE = 32
_WORLD_SIZE = 1

# Use a worker batch size that differs from each tested accumulation window.
_WORKER_BATCH = 3


def _model_out(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Generate one reader batch from the supplied generator."""
    return {
        DefaultTaskInfo.prediction_name: torch.rand(_BATCH_SIZE, generator=generator),
        DefaultTaskInfo.label_name: torch.randint(
            0, 2, (_BATCH_SIZE,), generator=generator
        ).float(),
        DefaultTaskInfo.weight_name: torch.rand(_BATCH_SIZE, generator=generator),
    }


class OffloadedGANonGAMetricEqualityTest(unittest.TestCase):
    """Compare offloaded metrics with and without gradient accumulation.

    Both configurations use identical reader batches and worker batch sizes, isolating
    gradient accumulation from worker-side batching.
    """

    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        init_process_group_single_rank("gloo")
        self._modules: list[CPUOffloadedRecMetricModule] = []

    def tearDown(self) -> None:
        # Shut down non-daemon workers so the test process can exit.
        for module in self._modules:
            module.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _make_module(
        self, *, k: int, compute_mode: RecComputeMode, worker_batch: int
    ) -> CPUOffloadedRecMetricModule:
        config = MetricsConfig(
            rec_tasks=[DefaultTaskInfo],
            rec_metrics={
                metric: RecMetricDef(
                    rec_tasks=[DefaultTaskInfo], window_size=_BATCH_SIZE * 8
                )
                for metric in _METRICS
            },
            throughput_metric=None,
            rec_compute_mode=compute_mode,
            num_micro_batches_per_step=k,
        )
        module_kwargs: dict[str, Any] = {
            "model_out_device": torch.device("cpu"),
            "update_batch_size": worker_batch,
        }
        module = generate_metric_module(
            CPUOffloadedRecMetricModule,
            metrics_config=config,
            batch_size=_BATCH_SIZE,
            world_size=_WORLD_SIZE,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
            module_kwargs=module_kwargs,
        )
        assert isinstance(module, CPUOffloadedRecMetricModule)
        self._modules.append(module)
        return module

    def _run(
        self,
        *,
        k: int,
        compute_mode: RecComputeMode,
        batches: list[dict[str, torch.Tensor]],
        worker_batch: int = _WORKER_BATCH,
    ) -> dict[str, torch.Tensor]:
        module = self._make_module(
            k=k, compute_mode=compute_mode, worker_batch=worker_batch
        )
        self.assertEqual(
            module.under_micro_batching,
            k > 1,
            "gradient accumulation mode does not match K",
        )
        for i, batch in enumerate(batches):
            if i % k == 0:
                module.reset_loss_metrics()
            # The final reader batch in each accumulation window uses `update()`.
            if (i + 1) % k == 0:
                module.update(batch)
            else:
                module.update_micro_batch(batch)

        # Resolving the asynchronous result waits for all queued updates.
        metrics = dict(module.async_compute().resolve())
        self.assertEqual(
            module._total_updates_processed,
            len(batches),
            "worker did not fold in every reader batch",
        )
        return metrics

    def _rec_metric_keys(self, metrics: dict[str, torch.Tensor]) -> list[str]:
        return sorted(k for k in metrics if not k.endswith(":loss"))

    def _assert_results_equal(
        self,
        baseline: dict[str, torch.Tensor],
        micro_batched: dict[str, torch.Tensor],
        *,
        context: str,
    ) -> None:
        keys = self._rec_metric_keys(baseline)
        self.assertTrue(keys, "expected RecMetric outputs")
        self.assertEqual(keys, self._rec_metric_keys(micro_batched))
        for key in keys:
            torch.testing.assert_close(
                micro_batched[key],
                baseline[key],
                msg=f"{key} differs between configurations ({context})",
            )

    def test_k2_matches_k1_at_a_fixed_worker_batch(self) -> None:
        for compute_mode in (
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_COMPUTATION,
        ):
            with self.subTest(compute_mode=compute_mode):
                generator = torch.Generator().manual_seed(20260918)
                batches = [_model_out(generator) for _ in range(8)]

                baseline = self._run(k=1, compute_mode=compute_mode, batches=batches)
                micro_batched = self._run(
                    k=2, compute_mode=compute_mode, batches=batches
                )

                self._assert_results_equal(
                    baseline, micro_batched, context=f"K=1 vs K=2, {compute_mode}"
                )

    def test_k4_matches_k1_at_a_fixed_worker_batch(self) -> None:
        generator = torch.Generator().manual_seed(20260918)
        batches = [_model_out(generator) for _ in range(8)]

        baseline = self._run(
            k=1, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )
        micro_batched = self._run(
            k=4, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )

        self._assert_results_equal(baseline, micro_batched, context="K=1 vs K=4")

    def test_equality_holds_at_every_fixed_worker_batch(self) -> None:
        """Test worker batch sizes below, equal to, and above the accumulation window."""
        for worker_batch in (1, 2, 5, 8):
            with self.subTest(worker_batch=worker_batch):
                generator = torch.Generator().manual_seed(20260918)
                batches = [_model_out(generator) for _ in range(8)]

                baseline = self._run(
                    k=1,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                    batches=batches,
                    worker_batch=worker_batch,
                )
                micro_batched = self._run(
                    k=2,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                    batches=batches,
                    worker_batch=worker_batch,
                )

                self._assert_results_equal(
                    baseline, micro_batched, context=f"worker_batch={worker_batch}"
                )

    def test_comparison_detects_different_data(self) -> None:
        generator = torch.Generator().manual_seed(20260918)
        batches = [_model_out(generator) for _ in range(8)]
        other = torch.Generator().manual_seed(777)
        different = [_model_out(other) for _ in range(8)]

        baseline = self._run(
            k=1, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )
        skewed = self._run(
            k=2,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            batches=different,
        )

        differing = [
            key
            for key in self._rec_metric_keys(baseline)
            if not torch.allclose(skewed[key], baseline[key])
        ]
        self.assertTrue(
            differing,
            "different data produced identical metrics; comparison is ineffective",
        )

    def test_trained_batches_counts_optimizer_steps_not_reader_batches(self) -> None:
        generator = torch.Generator().manual_seed(20260918)
        batches = [_model_out(generator) for _ in range(8)]

        k1 = self._make_module(
            k=1,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            worker_batch=_WORKER_BATCH,
        )
        for batch in batches:
            k1.update(batch)

        k2 = self._make_module(
            k=2,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            worker_batch=_WORKER_BATCH,
        )
        for i, batch in enumerate(batches):
            if (i + 1) % 2 == 0:
                k2.update(batch)
            else:
                k2.update_micro_batch(batch)

        self.assertEqual(k1.trained_batches, 8)
        self.assertEqual(k2.trained_batches, 4)
