#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, Optional

import torch
from torchrec.metrics.gauc import compute_gauc_3d, compute_window_auc, GAUCMetric
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.test_utils import TestMetric


class TestGAUCMetric(TestMetric):

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gauc_res = compute_gauc_3d(predictions, labels, weights)
        return {
            "auc_sum": gauc_res["auc_sum"],
            "num_samples": gauc_res["num_samples"],
        }

    @staticmethod
    # pyrefly: ignore[bad-override]
    def _compute(states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return compute_window_auc(
            states["auc_sum"],
            states["num_samples"],
        )


class GAUCMetricValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions: Dict[str, Optional[torch.Tensor]] = {"DefaultTask": None}
        self.labels: Dict[str, Optional[torch.Tensor]] = {"DefaultTask": None}
        # Reassigned to None in some tests to exercise the no-weights path.
        self.weights: Any = {"DefaultTask": None}
        self.num_candidates: Optional[torch.Tensor] = None
        self.batches: Dict[str, Any] = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "weights": self.weights,
        }
        self.gauc = GAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_max_num_candidates_cpu_matches_legacy(self) -> None:
        test_cases = [
            (
                "exact_bound",
                torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]]),
                torch.tensor([[1, 0, 1, 1, 0]]),
                torch.tensor([[1, 1, 1, 1, 1]]),
                torch.tensor([3, 2]),
                3,
            ),
            (
                "loose_bound",
                torch.tensor([[0.3, 0.9, 0.1, 0.8, 0.2, 0.8, 0.7, 0.6, 0.5, 0.5]]),
                torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 1, 0]]),
                torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                torch.tensor([2, 3, 3, 2]),
                8,
            ),
            (
                "zero_weights_and_tied_predictions",
                torch.tensor([[0.8, 0.8, 0.3, 0.2]]),
                torch.tensor([[1, 0, 1, 0]]),
                torch.tensor([[1, 0, 1, 1]]),
                torch.tensor([2, 2]),
                6,
            ),
        ]

        for (
            name,
            predictions,
            labels,
            weights,
            num_candidates,
            max_num_candidates,
        ) in test_cases:
            with self.subTest(name=name):
                legacy = GAUCMetric(
                    world_size=1,
                    my_rank=0,
                    batch_size=100,
                    tasks=[DefaultTaskInfo],
                )
                optimized = GAUCMetric(
                    world_size=1,
                    my_rank=0,
                    batch_size=100,
                    tasks=[DefaultTaskInfo],
                )
                batches = {
                    "predictions": {"DefaultTask": predictions},
                    "labels": {"DefaultTask": labels},
                    "num_candidates": num_candidates,
                    "weights": {"DefaultTask": weights},
                }

                legacy.update(**batches)
                optimized.update(
                    **batches,
                    max_num_candidates_cpu=torch.tensor(max_num_candidates),
                )

                legacy_result = legacy.compute()
                optimized_result = optimized.compute()
                self.assertEqual(legacy_result.keys(), optimized_result.keys())
                for key, legacy_value in legacy_result.items():
                    torch.testing.assert_close(
                        optimized_result[key],
                        legacy_value,
                        equal_nan=True,
                    )

    def test_calc_gauc_simple(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1]])
        self.num_candidates = torch.tensor([3, 2])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "max_num_candidates_cpu": torch.tensor(4),
            "weights": self.weights,
        }

        expected_gauc = torch.tensor([0.75], dtype=torch.double)
        expected_num_samples = torch.tensor([2], dtype=torch.double)
        self.gauc.update(**self.batches)
        gauc_res = self.gauc.compute()
        actual_gauc, num_effective_samples = (
            gauc_res["gauc-DefaultTask|window_gauc"],
            gauc_res["gauc-DefaultTask|window_gauc_num_samples"],
        )
        if not torch.allclose(expected_num_samples, num_effective_samples):
            raise ValueError(
                "actual num sample {} is not equal to expected num sample {}".format(
                    num_effective_samples, expected_num_samples
                )
            )
        if not torch.allclose(expected_gauc, actual_gauc):
            raise ValueError(
                "actual auc {} is not equal to expected auc {}".format(
                    actual_gauc, expected_gauc
                )
            )

    def test_calc_gauc_hard(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor(
            [[0.3, 0.9, 0.1, 0.8, 0.2, 0.8, 0.7, 0.6, 0.5, 0.5]]
        )
        self.labels["DefaultTask"] = torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.num_candidates = torch.tensor([2, 3, 3, 2])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "weights": self.weights,
        }

        expected_gauc = torch.tensor([0.25], dtype=torch.double)
        expected_num_samples = torch.tensor([2], dtype=torch.double)
        self.gauc.update(**self.batches)
        gauc_res = self.gauc.compute()
        actual_gauc, num_effective_samples = (
            gauc_res["gauc-DefaultTask|window_gauc"],
            gauc_res["gauc-DefaultTask|window_gauc_num_samples"],
        )
        if not torch.allclose(expected_num_samples, num_effective_samples):
            raise ValueError(
                "actual num sample {} is not equal to expected num sample {}".format(
                    num_effective_samples, expected_num_samples
                )
            )
        if not torch.allclose(expected_gauc, actual_gauc):
            raise ValueError(
                "actual auc {} is not equal to expected auc {}".format(
                    actual_gauc, expected_gauc
                )
            )

    def test_calc_gauc_all_0_labels(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
        self.labels["DefaultTask"] = torch.tensor([[0, 0, 0, 0, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1]])
        self.num_candidates = torch.tensor([3, 2])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "weights": None,
        }

        expected_gauc = torch.tensor([0.5], dtype=torch.double)
        expected_num_samples = torch.tensor([0], dtype=torch.double)
        self.gauc.update(**self.batches)
        gauc_res = self.gauc.compute()
        actual_gauc, num_effective_samples = (
            gauc_res["gauc-DefaultTask|window_gauc"],
            gauc_res["gauc-DefaultTask|window_gauc_num_samples"],
        )
        if not torch.allclose(expected_num_samples, num_effective_samples):
            raise ValueError(
                "actual num sample {} is not equal to expected num sample {}".format(
                    num_effective_samples, expected_num_samples
                )
            )
        if not torch.allclose(expected_gauc, actual_gauc):
            raise ValueError(
                "actual auc {} is not equal to expected auc {}".format(
                    actual_gauc, expected_gauc
                )
            )

    def test_calc_gauc_all_1_labels(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
        self.labels["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1]])
        self.num_candidates = torch.tensor([3, 2])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "weights": None,
        }

        expected_gauc = torch.tensor([0.5], dtype=torch.double)
        expected_num_samples = torch.tensor([0], dtype=torch.double)
        self.gauc.update(**self.batches)
        gauc_res = self.gauc.compute()
        actual_gauc, num_effective_samples = (
            gauc_res["gauc-DefaultTask|window_gauc"],
            gauc_res["gauc-DefaultTask|window_gauc_num_samples"],
        )
        if not torch.allclose(expected_num_samples, num_effective_samples):
            raise ValueError(
                "actual num sample {} is not equal to expected num sample {}".format(
                    num_effective_samples, expected_num_samples
                )
            )
        if not torch.allclose(expected_gauc, actual_gauc):
            raise ValueError(
                "actual auc {} is not equal to expected auc {}".format(
                    actual_gauc, expected_gauc
                )
            )

    def test_calc_gauc_identical_predictions(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor([[0.8, 0.8, 0.8, 0.8, 0.8]])
        self.labels["DefaultTask"] = torch.tensor([[1, 1, 0, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1]])
        self.num_candidates = torch.tensor([3, 2])
        self.weights = None
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "weights": None,
        }

        expected_gauc = torch.tensor([0.5], dtype=torch.double)
        expected_num_samples = torch.tensor([0], dtype=torch.double)
        self.gauc.update(**self.batches)
        gauc_res = self.gauc.compute()
        actual_gauc, num_effective_samples = (
            gauc_res["gauc-DefaultTask|window_gauc"],
            gauc_res["gauc-DefaultTask|window_gauc_num_samples"],
        )
        if not torch.allclose(expected_num_samples, num_effective_samples):
            raise ValueError(
                "actual num sample {} is not equal to expected num sample {}".format(
                    num_effective_samples, expected_num_samples
                )
            )
        if not torch.allclose(expected_gauc, actual_gauc):
            raise ValueError(
                "actual auc {} is not equal to expected auc {}".format(
                    actual_gauc, expected_gauc
                )
            )

    def test_calc_gauc_weighted(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor(
            [[0.3, 0.9, 0.1, 0.8, 0.2, 0.8, 0.7, 0.6, 0.5, 0.5]]
        )
        self.labels["DefaultTask"] = torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 0, 1, 1]])
        self.num_candidates = torch.tensor([2, 3, 3, 2])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
            "weights": self.weights,
        }

        expected_gauc = torch.tensor([0.5], dtype=torch.double)
        expected_num_samples = torch.tensor([2], dtype=torch.double)
        self.gauc.update(**self.batches)
        gauc_res = self.gauc.compute()
        actual_gauc, num_effective_samples = (
            gauc_res["gauc-DefaultTask|window_gauc"],
            gauc_res["gauc-DefaultTask|window_gauc_num_samples"],
        )
        if not torch.allclose(expected_num_samples, num_effective_samples):
            raise ValueError(
                "actual num sample {} is not equal to expected num sample {}".format(
                    num_effective_samples, expected_num_samples
                )
            )
        if not torch.allclose(expected_gauc, actual_gauc):
            raise ValueError(
                "actual auc {} is not equal to expected auc {}".format(
                    actual_gauc, expected_gauc
                )
            )
