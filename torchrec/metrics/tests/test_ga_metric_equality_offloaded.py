#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Any, Dict, List

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

_METRICS: List[RecMetricEnum] = [
    RecMetricEnum.NE,
    RecMetricEnum.CTR,
    RecMetricEnum.MSE,
    RecMetricEnum.AUC,
    RecMetricEnum.CALIBRATION,
]

_BATCH_SIZE = 32
_WORLD_SIZE = 1

# The worker batch. Held identical across the two arms of every comparison, and
# deliberately not a multiple or divisor of any K under test, so that worker-side
# merging and micro-batch accumulation stay independent axes.
_WORKER_BATCH = 3


def _model_out(generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """One reader batch. Drawn from a seeded generator so both arms see the same data."""
    return {
        DefaultTaskInfo.prediction_name: torch.rand(_BATCH_SIZE, generator=generator),
        DefaultTaskInfo.label_name: torch.randint(
            0, 2, (_BATCH_SIZE,), generator=generator
        ).float(),
        DefaultTaskInfo.weight_name: torch.rand(_BATCH_SIZE, generator=generator),
    }


class OffloadedGANonGAMetricEqualityTest(unittest.TestCase):
    """GA vs non-GA equality on the OFFLOADED path, at a fixed worker batch.

    ``GANonGAMetricEqualityTest`` (test_ga_metric_equality.py) proves the same property
    for the in-process ``RecMetricModule``. That leaves the offloaded module untested for
    it, and the offloaded module is where the two mechanisms can interfere: reader batches
    are merged into worker batches of ``update_batch_size`` on a background thread, while
    micro-batching decides which reader batch closes an accumulation window on the caller
    thread.

    Holding the worker batch FIXED and identical across both arms is what makes the
    comparison mean something. The offloaded module's batching legitimately changes when
    state is folded
    in; pinning it removes that as an explanation, so a K=1 vs K>1 difference can only be
    the accumulation contract.

    ``_WORKER_BATCH`` is deliberately coprime to the K values under test. Were it equal to
    K, a bug that merged on window boundaries would be invisible because the two
    boundaries would coincide.

    **Deliberately not asserted:** that the offloaded arm equals the in-process arm.
    Those differ for known batching reasons and comparing them would re-introduce exactly
    the variable this test pins down.
    """

    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        init_process_group_single_rank("gloo")
        self._modules: List[CPUOffloadedRecMetricModule] = []

    def tearDown(self) -> None:
        # Worker threads are non-daemon; leaking them wedges the whole suite.
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
        module_kwargs: Dict[str, Any] = {
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
        # The factory is typed to the base class, and a plain module here would make
        # every comparison below exercise the in-process path under an offloaded name.
        assert isinstance(module, CPUOffloadedRecMetricModule)
        self._modules.append(module)
        return module

    def _run(
        self,
        *,
        k: int,
        compute_mode: RecComputeMode,
        batches: List[Dict[str, torch.Tensor]],
        worker_batch: int = _WORKER_BATCH,
    ) -> Dict[str, torch.Tensor]:
        module = self._make_module(
            k=k, compute_mode=compute_mode, worker_batch=worker_batch
        )
        # The gate is what the whole comparison rests on. If num_micro_batches_per_step
        # ever stopped reaching set_under_micro_batching, both arms would run the non-GA
        # path and every equality below would pass while testing nothing.
        self.assertEqual(
            module.under_micro_batching, k > 1, "the GA gate is not armed from K"
        )
        for i, batch in enumerate(batches):
            if i % k == 0:
                module.reset_loss_metrics()
            # The closing batch of a window is the one that takes update(); at K=1
            # that is every batch, which is exactly the non-GA path.
            if (i + 1) % k == 0:
                module.update(batch)
            else:
                module.update_micro_batch(batch)

        # compute() raises on this module; the offloaded publication is async_compute(),
        # and resolve() is what drains it. The compute marker is enqueued behind every
        # update, so resolving it means the worker has folded in every reader batch --
        # asserted below rather than assumed, since an arm that silently dropped batches
        # would otherwise make the equality below pass for the wrong reason.
        metrics = dict(module.async_compute().resolve())
        self.assertEqual(
            module._total_updates_processed,
            len(batches),
            "worker did not fold in every reader batch",
        )
        return metrics

    def _rec_metric_keys(self, metrics: Dict[str, torch.Tensor]) -> List[str]:
        return sorted(k for k in metrics if not k.endswith(":loss"))

    def _assert_arms_agree(
        self,
        baseline: Dict[str, torch.Tensor],
        micro_batched: Dict[str, torch.Tensor],
        *,
        context: str,
    ) -> None:
        keys = self._rec_metric_keys(baseline)
        # An empty intersection would make every assertion below vacuous.
        self.assertTrue(keys)
        self.assertEqual(keys, self._rec_metric_keys(micro_batched))
        for key in keys:
            torch.testing.assert_close(
                micro_batched[key],
                baseline[key],
                msg=f"{key} differs between the arms ({context})",
            )

    def test_k2_matches_k1_at_a_fixed_worker_batch(self) -> None:
        for compute_mode in (
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_COMPUTATION,
        ):
            with self.subTest(compute_mode=compute_mode):
                # Eight reader batches: four K=2 windows, or eight K=1 steps. Neither
                # divides _WORKER_BATCH=3 evenly, so the final worker batch is short on
                # both arms -- the case where a merge-boundary bug would surface.
                generator = torch.Generator().manual_seed(20260918)
                batches = [_model_out(generator) for _ in range(8)]

                baseline = self._run(k=1, compute_mode=compute_mode, batches=batches)
                micro_batched = self._run(
                    k=2, compute_mode=compute_mode, batches=batches
                )

                self._assert_arms_agree(
                    baseline, micro_batched, context=f"K=1 vs K=2, {compute_mode}"
                )

    def test_k4_matches_k1_at_a_fixed_worker_batch(self) -> None:
        # K=4 with 8 batches: two windows. A K that does not divide the batch count the
        # same way K=2 does, so an off-by-one in the window arithmetic shows up.
        generator = torch.Generator().manual_seed(20260918)
        batches = [_model_out(generator) for _ in range(8)]

        baseline = self._run(
            k=1, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )
        micro_batched = self._run(
            k=4, compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION, batches=batches
        )

        self._assert_arms_agree(baseline, micro_batched, context="K=1 vs K=4")

    def test_equality_holds_at_every_fixed_worker_batch(self) -> None:
        """The isolation claim itself, swept.

        One fixed worker batch shows GA does not perturb the metrics at THAT batching.
        The claim being made is stronger -- that the two mechanisms do not interact --
        so the equality has to survive worker batches on both sides of K and one equal
        to it. B=1 dispatches every reader batch alone; B=8 coalesces all of them into a
        single merged job; B=2 aligns with K, the case the default deliberately avoids.
        """
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

                self._assert_arms_agree(
                    baseline, micro_batched, context=f"worker_batch={worker_batch}"
                )

    def test_the_comparison_can_fail(self) -> None:
        """Falsifiability: the assertions above are not true of any two runs.

        Feed the K=2 arm DIFFERENT data and the same comparison must fail. Without this,
        an offloaded arm that published constants -- or nothing the comparison reads --
        would pass every equality test above.
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
        """The one difference that is supposed to exist, pinned so it stays deliberate.

        Also the reason the equality tests compare published metrics rather than module
        state: this counter is expected to differ, and on the offloaded module it is
        incremented on the caller thread while the metrics land on the worker.
        """
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
