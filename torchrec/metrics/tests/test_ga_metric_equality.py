#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, List

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

_METRICS: List[RecMetricEnum] = [
    RecMetricEnum.NE,
    RecMetricEnum.CTR,
    RecMetricEnum.MSE,
    RecMetricEnum.AUC,
    RecMetricEnum.CALIBRATION,
]

_BATCH_SIZE = 32
_WORLD_SIZE = 1


def _model_out(generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """One reader batch. Drawn from a seeded generator so both arms see the same data."""
    return {
        DefaultTaskInfo.prediction_name: torch.rand(_BATCH_SIZE, generator=generator),
        DefaultTaskInfo.label_name: torch.randint(
            0, 2, (_BATCH_SIZE,), generator=generator
        ).float(),
        DefaultTaskInfo.weight_name: torch.rand(_BATCH_SIZE, generator=generator),
    }


class GANonGAMetricEqualityTest(unittest.TestCase):
    """Identical reader batches must produce identical RecMetric values under GA.

    Micro-batching changes only WHICH call carries a reader batch into the metrics
    module -- ``update_micro_batch()`` for the batches that do not close a window,
    ``update()`` for the one that does. Neither call may change what the metric sees,
    so for the same batches in the same order every RecMetric must compute the same
    value at both K=1 and K>1.

    This is a deterministic replacement for comparing two training jobs: it removes
    world size, data order and nondeterministic kernels from the comparison, so a
    difference here is a real accounting bug and nothing else.

    **Deliberately not compared:** ``trained_batches`` (it counts optimizer steps, so
    K=2 halves it BY DESIGN) and the published ``:loss`` (the GA path publishes the
    K-average for a window, the K=1 path publishes each batch's own value). Those two
    differences are the feature; every RecMetric value is not.
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
        batches: List[Dict[str, torch.Tensor]],
    ) -> DeferrableMetrics:
        module = self._make_module(k=k, compute_mode=compute_mode)
        for i, batch in enumerate(batches):
            if i % k == 0:
                module.reset_loss_metrics()
            # The closing batch of a window is the one that takes update(); at K=1
            # that is every batch, which is exactly the non-GA path.
            if (i + 1) % k == 0:
                module.update(batch)
            else:
                module.update_micro_batch(batch)
        return module.compute()

    def _rec_metric_keys(self, metrics: DeferrableMetrics) -> List[str]:
        return sorted(k for k in metrics if not k.endswith(":loss"))

    def test_k2_matches_k1_on_every_rec_metric(self) -> None:
        for compute_mode in (
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_COMPUTATION,
        ):
            with self.subTest(compute_mode=compute_mode):
                # Eight reader batches: four K=2 windows, or eight K=1 steps.
                generator = torch.Generator().manual_seed(20260918)
                batches = [_model_out(generator) for _ in range(8)]

                baseline = self._run(k=1, compute_mode=compute_mode, batches=batches)
                micro_batched = self._run(
                    k=2, compute_mode=compute_mode, batches=batches
                )

                keys = self._rec_metric_keys(baseline)
                # An empty intersection would make every assertion below vacuous.
                self.assertTrue(keys)
                self.assertEqual(keys, self._rec_metric_keys(micro_batched))
                for key in keys:
                    torch.testing.assert_close(
                        micro_batched[key],
                        baseline[key],
                        msg=f"{key} differs between K=1 and K=2 on the same batches",
                    )

    def test_k4_matches_k1_on_every_rec_metric(self) -> None:
        # K=4 with 8 batches: two windows. A K that does not divide the batch count
        # the same way K=2 does, so an off-by-one in the window arithmetic shows up.
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

    def test_the_comparison_can_fail(self) -> None:
        """Falsifiability: the assertion above is not true of any two runs.

        Feed the K=2 arm DIFFERENT data and the same comparison must fail. Without
        this, a bug that made every metric constant would pass the equality tests.
        """
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
            differing, "different data produced identical metrics -- the arm is inert"
        )

    def test_trained_batches_counts_optimizer_steps_not_reader_batches(self) -> None:
        """The one difference that is supposed to exist, pinned so it stays deliberate."""
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
