#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.uauc import UAUCMetric


class UAUCMetricValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = {"DefaultTask": None}
        self.labels = {"DefaultTask": None}
        self.weights = {"DefaultTask": None}
        self.grouping_keys = None
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }
        self.uauc = UAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_uauc_simple(self) -> None:
        """Two users with mixed labels. User 0 has perfect AUC=1.0, user 1 has AUC=0.5."""
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.7, 0.3]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 0, 0, 1]])
        self.weights["DefaultTask"] = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        # User 0: samples 0,1,2 -> preds [0.9, 0.1, 0.8], labels [1, 0, 1] -> AUC = 1.0
        # User 1: samples 3,4,5 -> preds [0.2, 0.7, 0.3], labels [0, 0, 1] -> AUC = 0.5
        self.grouping_keys = torch.tensor([0, 0, 0, 1, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_uauc = torch.tensor([0.75], dtype=torch.double)  # (1.0 + 0.5) / 2
        expected_num_users = torch.tensor([2.0], dtype=torch.double)

        actual_uauc = res["uauc-DefaultTask|window_uauc"]
        actual_num_users = res["uauc-DefaultTask|window_uauc_num_users"]

        self.assertTrue(
            torch.allclose(expected_uauc, actual_uauc, atol=1e-4),
            f"Expected uAUC {expected_uauc}, got {actual_uauc}",
        )
        self.assertTrue(
            torch.allclose(expected_num_users, actual_num_users),
            f"Expected num_users {expected_num_users}, got {actual_num_users}",
        )

    def test_calc_uauc_multi_user(self) -> None:
        """4 users, one with all-same labels (skipped)."""
        self.predictions["DefaultTask"] = torch.tensor(
            [[0.9, 0.1, 0.8, 0.2, 0.5, 0.5, 0.7, 0.3]]
        )
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1]])
        self.weights["DefaultTask"] = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        )
        # User 0: samples 0,1 -> preds [0.9, 0.1], labels [1, 0] -> AUC = 1.0
        # User 1: samples 2,3 -> preds [0.8, 0.2], labels [1, 0] -> AUC = 1.0
        # User 2: samples 4,5 -> preds [0.5, 0.5], labels [1, 1] -> all-same labels, skipped
        # User 3: samples 6,7 -> preds [0.7, 0.3], labels [0, 1] -> AUC = 0.0
        self.grouping_keys = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        # 3 valid users: AUCs = [1.0, 1.0, 0.0], mean = 2/3
        expected_uauc = torch.tensor([2.0 / 3.0], dtype=torch.double)
        expected_num_users = torch.tensor([3.0], dtype=torch.double)

        actual_uauc = res["uauc-DefaultTask|window_uauc"]
        actual_num_users = res["uauc-DefaultTask|window_uauc_num_users"]

        self.assertTrue(
            torch.allclose(expected_uauc, actual_uauc, atol=1e-4),
            f"Expected uAUC {expected_uauc}, got {actual_uauc}",
        )
        self.assertTrue(
            torch.allclose(expected_num_users, actual_num_users),
            f"Expected num_users {expected_num_users}, got {actual_num_users}",
        )

    def test_calc_uauc_all_0_labels(self) -> None:
        """All labels=0, all users skipped."""
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        self.labels["DefaultTask"] = torch.tensor([[0, 0, 0, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.grouping_keys = torch.tensor([0, 0, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_num_users = torch.tensor([0.0], dtype=torch.double)
        actual_num_users = res["uauc-DefaultTask|window_uauc_num_users"]

        self.assertTrue(
            torch.allclose(expected_num_users, actual_num_users),
            f"Expected num_users {expected_num_users}, got {actual_num_users}",
        )

    def test_calc_uauc_all_1_labels(self) -> None:
        """All labels=1, all users skipped."""
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
        self.labels["DefaultTask"] = torch.tensor([[1, 1, 1, 1]])
        self.weights["DefaultTask"] = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.grouping_keys = torch.tensor([0, 0, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_num_users = torch.tensor([0.0], dtype=torch.double)
        actual_num_users = res["uauc-DefaultTask|window_uauc_num_users"]

        self.assertTrue(
            torch.allclose(expected_num_users, actual_num_users),
            f"Expected num_users {expected_num_users}, got {actual_num_users}",
        )

    def test_calc_uauc_identical_predictions(self) -> None:
        """Identical predictions within a user -> user is skipped."""
        self.predictions["DefaultTask"] = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.grouping_keys = torch.tensor([0, 0, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_num_users = torch.tensor([0.0], dtype=torch.double)
        actual_num_users = res["uauc-DefaultTask|window_uauc_num_users"]

        self.assertTrue(
            torch.allclose(expected_num_users, actual_num_users),
            f"Expected num_users {expected_num_users}, got {actual_num_users}",
        )

    def test_calc_uauc_weighted(self) -> None:
        """Non-uniform sample weights."""
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.1, 0.8, 0.2]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 0]])
        self.weights["DefaultTask"] = torch.tensor([[2.0, 1.0, 1.0, 3.0]])
        # User 0: preds [0.9, 0.1], labels [1, 0], weights [2.0, 1.0] -> AUC = 1.0
        # User 1: preds [0.8, 0.2], labels [1, 0], weights [1.0, 3.0] -> AUC = 1.0
        self.grouping_keys = torch.tensor([0, 0, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_uauc = torch.tensor([1.0], dtype=torch.double)
        actual_uauc = res["uauc-DefaultTask|window_uauc"]

        self.assertTrue(
            torch.allclose(expected_uauc, actual_uauc, atol=1e-4),
            f"Expected uAUC {expected_uauc}, got {actual_uauc}",
        )

    def test_calc_uauc_single_valid_user(self) -> None:
        """Only 1 valid user in batch."""
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.1, 0.5, 0.5]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 1]])
        self.weights["DefaultTask"] = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        # User 0: preds [0.9, 0.1], labels [1, 0] -> AUC = 1.0 (valid)
        # User 1: preds [0.5, 0.5], labels [1, 1] -> all-same labels, skipped
        self.grouping_keys = torch.tensor([0, 0, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_uauc = torch.tensor([1.0], dtype=torch.double)
        expected_num_users = torch.tensor([1.0], dtype=torch.double)

        actual_uauc = res["uauc-DefaultTask|window_uauc"]
        actual_num_users = res["uauc-DefaultTask|window_uauc_num_users"]

        self.assertTrue(
            torch.allclose(expected_uauc, actual_uauc, atol=1e-4),
            f"Expected uAUC {expected_uauc}, got {actual_uauc}",
        )
        self.assertTrue(
            torch.allclose(expected_num_users, actual_num_users),
            f"Expected num_users {expected_num_users}, got {actual_num_users}",
        )

    def test_calc_wuauc(self) -> None:
        """Verify weighted uAUC calculation."""
        self.predictions["DefaultTask"] = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.7, 0.3]])
        self.labels["DefaultTask"] = torch.tensor([[1, 0, 1, 0, 0, 1]])
        self.weights["DefaultTask"] = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        # User 0: 3 samples, AUC=1.0, weight=3.0 (w_pos=2 + w_neg=1)
        # User 1: 3 samples, AUC=0.5, weight=3.0 (w_pos=1 + w_neg=2)
        # wuAUC = (1.0*3 + 0.5*3) / (3+3) = 4.5/6 = 0.75
        self.grouping_keys = torch.tensor([0, 0, 0, 1, 1, 1])
        self.batches = {
            "predictions": self.predictions,
            "labels": self.labels,
            "grouping_keys": self.grouping_keys,
            "weights": self.weights,
        }

        self.uauc.update(**self.batches)
        res = self.uauc.compute()

        expected_wuauc = torch.tensor([0.75], dtype=torch.double)
        actual_wuauc = res["uauc-DefaultTask|window_wuauc"]

        self.assertTrue(
            torch.allclose(expected_wuauc, actual_wuauc, atol=1e-4),
            f"Expected wuAUC {expected_wuauc}, got {actual_wuauc}",
        )
