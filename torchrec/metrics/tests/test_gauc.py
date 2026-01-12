#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, Optional

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
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_window_auc(
            states["auc_sum"],
            states["num_samples"],
        )


class GAUCMetricValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = {"DefaultTask": None}
        self.labels = {"DefaultTask": None}
        self.weights = {"DefaultTask": None}
        self.num_candidates = None
        self.batches = {
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

    def test_calc_gauc_simple(self) -> None:
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1, 1, 1, 1, 1]])
        self.num_candidates = torch.tensor([3, 2])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "num_candidates": self.num_candidates,
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
