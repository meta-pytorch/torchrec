#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest

import torch
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.metric_module import generate_metric_module, RecMetricModule
from torchrec.metrics.metrics_config import (
    DefaultMetricsConfig,
    DefaultTaskInfo,
    RecMetricEnum,
)
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.test_utils import gen_test_batch


_CUDA_UNAVAILABLE: bool = not torch.cuda.is_available()


class TestGPU(unittest.TestCase):
    @unittest.skipIf(_CUDA_UNAVAILABLE, "Test needs to run on GPU")
    def test_auc_reset(self) -> None:
        batch_size = 64
        auc = AUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=[DefaultTaskInfo],
        )
        # Mimic the case where the metric module is moved to GPU.
        device = torch.device("cuda:0")
        auc.to(device)
        # Mimic that reset is called when checkpointing.
        auc.reset()
        # pyrefly: ignore[bad-argument-type]
        self.assertEqual(len(auc._metrics_computations[0].predictions), 1)
        # pyrefly: ignore[bad-argument-type]
        self.assertEqual(len(auc._metrics_computations[0].labels), 1)
        # pyrefly: ignore[bad-argument-type]
        self.assertEqual(len(auc._metrics_computations[0].weights), 1)
        model_output = gen_test_batch(batch_size)
        model_output = {k: v.to(device) for k, v in model_output.items()}
        auc.update(
            predictions={"DefaultTask": model_output["prediction"]},
            labels={"DefaultTask": model_output["label"]},
            weights={"DefaultTask": model_output["weight"]},
        )
        # pyrefly: ignore[bad-argument-type]
        self.assertEqual(len(auc._metrics_computations[0].predictions), 1)
        # pyrefly: ignore[bad-argument-type]
        self.assertEqual(len(auc._metrics_computations[0].labels), 1)
        # pyrefly: ignore[bad-argument-type]
        self.assertEqual(len(auc._metrics_computations[0].weights), 1)
        # pyrefly: ignore[bad-index]
        self.assertEqual(auc._metrics_computations[0].predictions[0].device, device)
        # pyrefly: ignore[bad-index]
        self.assertEqual(auc._metrics_computations[0].labels[0].device, device)
        # pyrefly: ignore[bad-index]
        self.assertEqual(auc._metrics_computations[0].weights[0].device, device)

    @unittest.skipIf(_CUDA_UNAVAILABLE, "Test needs to run on GPU")
    def test_device_match(self) -> None:
        config = copy.deepcopy(DefaultMetricsConfig)
        config.rec_metrics[RecMetricEnum.NE].window_size = 256
        batch_size = 128
        device = torch.device("cuda:0")
        metric_module = generate_metric_module(
            RecMetricModule,
            metrics_config=config,
            batch_size=batch_size,
            world_size=1,
            my_rank=0,
            state_metrics_mapping={},
            device=device,
        )

        for _ in range(10):
            model_output = gen_test_batch(batch_size)
            model_output = {k: v.to(device) for k, v in model_output.items()}
            metric_module.update(model_output)

    @unittest.skipIf(_CUDA_UNAVAILABLE, "Test needs to run on GPU")
    def test_window_metric(self) -> None:
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=128,
            tasks=[DefaultTaskInfo],
            window_size=3 * 128,
        )
        # Mimic the case where the metric module is moved to GPU.
        device = torch.device("cuda:0")
        ne.to(device)

        ne_computation = ne._metrics_computations[0]
        # test RecMetricComputation._add_window_state
        torch.allclose(
            # pyrefly: ignore[bad-argument-type]
            ne_computation.window_cross_entropy_sum,
            torch.tensor([0.0], dtype=torch.double, device=device),
        )
        torch.allclose(
            # pyrefly: ignore[bad-argument-type]
            ne_computation.window_weighted_num_samples,
            torch.tensor([[0.0]], dtype=torch.double, device=device),
        )

        # test RecMetricComputation._aggregate_window_state
        for i in range(4):
            model_output = gen_test_batch(128)
            model_output = {k: v.to(device) for k, v in model_output.items()}
            ne.update(
                predictions={"DefaultTask": model_output["prediction"]},
                labels={"DefaultTask": model_output["label"]},
                weights={"DefaultTask": model_output["weight"]},
            )

            if i < 3:
                ne_metric = ne.compute()
                name = DefaultTaskInfo.name
                torch.allclose(
                    ne_metric[f"ne-{name}|lifetime_ne"],
                    ne_metric[f"ne-{name}|window_ne"],
                )
                ne_metric = ne.local_compute()
                torch.allclose(
                    ne_metric[f"ne-{name}|local_lifetime_ne"],
                    ne_metric[f"ne-{name}|local_window_ne"],
                )
            else:
                self.assertEqual(
                    # pyrefly: ignore[not-callable]
                    ne_computation.window_cross_entropy_sum.size(),
                    torch.Size([1]),
                )
                self.assertEqual(
                    len(
                        # pyrefly: ignore[bad-index, missing-attribute]
                        ne_computation._batch_window_buffers[
                            # pyrefly: ignore[bad-index]
                            "window_cross_entropy_sum"
                        ].buffers
                    ),
                    3,
                )
