#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.ne_positive import NEPositiveMetric


WORLD_SIZE = 4
BATCH_SIZE = 10


def generate_model_output() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor([[0.8, 0.2, 0.3, 0.6, 0.5]]),
        "labels": torch.tensor([[1, 0, 0, 1, 1]]),
        "weights": torch.tensor([[1, 2, 1, 2, 1]]),
        "expected_ne_positive": torch.tensor([0.4054]),
    }


class NEPositiveValueTest(unittest.TestCase):
    """
    This set of tests verify the computation logic of AUC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    def setUp(self) -> None:
        self.ne_positive = NEPositiveMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
        )

    def test_ne_positive(self) -> None:
        model_output = generate_model_output()
        self.ne_positive.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
        )
        metric = self.ne_positive.compute()
        print(metric)
        actual_metric = metric[
            f"ne_positive-{DefaultTaskInfo.name}|lifetime_ne_positive"
        ]
        expected_metric = model_output["expected_ne_positive"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )
