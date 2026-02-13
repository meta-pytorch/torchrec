#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.ne_with_recalibration import RecalibratedNEMetric


WORLD_SIZE = 4
BATCH_SIZE = 10


def generate_model_output() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]]),
        "expected_recalibrated_ne": torch.tensor([2.8214]),
    }


class RecalibratedNEMetricMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ne_with_recalibration = RecalibratedNEMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            recalibration_coefficient=0.1,
        )

    def test_ne_with_recalibration(self) -> None:
        model_output = generate_model_output()
        self.ne_with_recalibration.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
        )
        metric = self.ne_with_recalibration.compute()
        actual_metric = metric[f"recalibrated_ne-{DefaultTaskInfo.name}|lifetime_ne"]
        expected_metric = model_output["expected_recalibrated_ne"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )
