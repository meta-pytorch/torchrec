#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
from torchrec.metrics.deferrable_metrics import DeferrableMetrics
from torchrec.metrics.metric_module import generate_metric_module, RecMetricModule
from torchrec.metrics.metrics_config import (
    DefaultTaskInfo,
    MetricsConfig,
    RecComputeMode,
    RecMetricDef,
    RecMetricEnum,
)

_METRICS: list[RecMetricEnum] = [
    RecMetricEnum.NE,
    RecMetricEnum.CTR,
    RecMetricEnum.MSE,
    RecMetricEnum.AUC,
    RecMetricEnum.CALIBRATION,
]

_BATCH_SIZE = 32
_WORLD_SIZE = 1


def _model_out(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Generate one reader batch from the supplied generator."""
    return {
        DefaultTaskInfo.prediction_name: torch.rand(_BATCH_SIZE, generator=generator),
        DefaultTaskInfo.label_name: torch.randint(
            0, 2, (_BATCH_SIZE,), generator=generator
        ).float(),
        DefaultTaskInfo.weight_name: torch.rand(_BATCH_SIZE, generator=generator),
    }


class GANonGAMetricEqualityTest(unittest.TestCase):
    """Compare RecMetric outputs with and without gradient accumulation.

    Both configurations consume identical reader batches. Loss and ``trained_batches``
    are excluded because their semantics intentionally change with accumulation.
    """

    def _make_module(self, *, k: int, compute_mode: RecComputeMode) -> RecMetricModule:
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
        return generate_metric_module(
            RecMetricModule,
            metrics_config=config,
            batch_size=_BATCH_SIZE,
            world_size=_WORLD_SIZE,
            my_rank=0,
            state_metrics_mapping={},
            device=torch.device("cpu"),
        )

    def _run(
        self,
        *,
        k: int,
        compute_mode: RecComputeMode,
        batches: list[dict[str, torch.Tensor]],
    ) -> DeferrableMetrics:
        module = self._make_module(k=k, compute_mode=compute_mode)
        for i, batch in enumerate(batches):
            if i % k == 0:
                module.reset_loss_metrics()
            # The final reader batch in each accumulation window uses `update()`.
            if (i + 1) % k == 0:
                module.update(batch)
            else:
                module.update_micro_batch(batch)
        return module.compute()

    def _rec_metric_keys(self, metrics: DeferrableMetrics) -> list[str]:
        return sorted(k for k in metrics if not k.endswith(":loss"))

    def test_k2_matches_k1_on_every_rec_metric(self) -> None:
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

                keys = self._rec_metric_keys(baseline)
                self.assertTrue(keys, "expected RecMetric outputs")
                self.assertEqual(keys, self._rec_metric_keys(micro_batched))
                for key in keys:
                    torch.testing.assert_close(
                        micro_batched[key],
                        baseline[key],
                        msg=f"{key} differs between K=1 and K=2 on the same batches",
                    )

    def test_k4_matches_k1_on_every_rec_metric(self) -> None:
        generator = torch.Generator().manual_seed(20260918)
        batches = [_model_out(generator) for _ in range(8)]

        baseline = self._run(
            k=1, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )
        micro_batched = self._run(
            k=4, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )

        keys = self._rec_metric_keys(baseline)
        self.assertTrue(keys)
        for key in keys:
            torch.testing.assert_close(
                micro_batched[key], baseline[key], msg=f"{key} differs at K=4"
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
            k=1, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )
        for batch in batches:
            k1.update(batch)

        k2 = self._make_module(
            k=2, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )
        for i, batch in enumerate(batches):
            if (i + 1) % 2 == 0:
                k2.update(batch)
            else:
                k2.update_micro_batch(batch)

        self.assertEqual(k1.trained_batches, 8)
        self.assertEqual(k2.trained_batches, 4)
