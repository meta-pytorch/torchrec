#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.autograd import Variable
from torchrec.optim.clipping import (
    _batch_cal_norm,
    _compute_total_norm,
    _get_grads,
    GradientClipping,
    GradientClippingOptimizer,
)
from torchrec.optim.test_utils import DummyKeyedOptimizer


class TestGradientClippingOptimizer(unittest.TestCase):
    def test_clip_all_gradients_norm(self) -> None:
        # Clip all gradients to zero
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=0.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([1.0, 2.0])
        gradient_clipping_optimizer.step()

        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.0, 0.0])))

    def test_clip_no_gradients_norm(self) -> None:
        # gradients are too small to be clipped
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([0.5, 0.5])
        gradient_clipping_optimizer.step()

        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.5, 0.5])))

    def test_clip_partial_gradients_norm(self) -> None:
        # test partial clipping
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        gradient_clipping_optimizer.step()

        norm = 2.0**2 + 4.0**2
        expected_grad = torch.tensor([2.0, 4.0]) * norm ** (-0.5)
        self.assertTrue(torch.allclose(param_1.grad, expected_grad))

    def test_clip_partial_gradients_norm_multi_params(self) -> None:
        # test partial clipping
        max_gradient = 2.0
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([2.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1]}, {"params": [param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        param_2.grad = torch.tensor([4.0, 8.0])

        gradient_clipping_optimizer.step()

        print(param_1.grad, param_2.grad)

        norm = (2.0**2 + 4.0**2 + 4.0**2 + 8.0**2) ** (-0.5)
        expected_grad_1 = torch.tensor([2.0, 4.0]) * norm * max_gradient
        expected_grad_2 = torch.tensor([4.0, 8.0]) * norm * max_gradient

        print(param_1.grad, param_2.grad, expected_grad_1, expected_grad_2)

        self.assertTrue(torch.allclose(param_1.grad, expected_grad_1))
        self.assertTrue(torch.allclose(param_2.grad, expected_grad_2))

    def test_clip_all_gradients_value(self) -> None:
        # Clip all gradients to zero
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=0, clipping=GradientClipping.VALUE
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([1.0, 2.0])
        gradient_clipping_optimizer.step()

        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.0, 0.0])))

    def test_clip_no_gradients_value(self) -> None:
        # gradients are too small to be clipped
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.VALUE
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([0.5, 0.5])
        gradient_clipping_optimizer.step()

        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.5, 0.5])))

    def test_clip_gradients_value(self) -> None:
        # test partial clipping
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1, clipping=GradientClipping.VALUE
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        gradient_clipping_optimizer.step()

        expected_grad = torch.tensor([1.0, 1.0])

        self.assertTrue(torch.allclose(param_1.grad, expected_grad))

    def test_clip_partial_gradients_value_multi_params(self) -> None:
        # test partial clipping
        max_gradient = 2.0
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([2.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1]}, {"params": [param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.VALUE,
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        param_2.grad = torch.tensor([4.0, 8.0])

        gradient_clipping_optimizer.step()

        expected_grad_1 = torch.tensor([2.0, 2.0])
        expected_grad_2 = torch.tensor([2.0, 2.0])

        self.assertTrue(torch.allclose(param_1.grad, expected_grad_1))
        self.assertTrue(torch.allclose(param_2.grad, expected_grad_2))

    @patch("torch.nn.utils.clip_grad_norm_")
    def test_clip_no_gradients_norm_meta_device(
        self, mock_clip_grad_norm: MagicMock
    ) -> None:
        # Clip all gradients to zero
        param_1 = Variable(
            torch.tensor([1.0, 2.0], device=torch.device("meta")), requires_grad=True
        )

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=0.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        gradient_clipping_optimizer.step()

        mock_clip_grad_norm.assert_not_called()


class TestGetGrads(unittest.TestCase):
    def test_get_grads_returns_gradients(self) -> None:
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param_2 = torch.tensor([3.0, 4.0], requires_grad=True)
        param_1.grad = torch.tensor([0.1, 0.2])
        param_2.grad = torch.tensor([0.3, 0.4])

        grads = _get_grads([param_1, param_2])

        self.assertEqual(len(grads), 2)
        self.assertTrue(torch.equal(grads[0], torch.tensor([0.1, 0.2])))
        self.assertTrue(torch.equal(grads[1], torch.tensor([0.3, 0.4])))

    def test_get_grads_skips_none_gradients(self) -> None:
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param_2 = torch.tensor([3.0, 4.0], requires_grad=True)
        param_1.grad = torch.tensor([0.1, 0.2])
        param_2.grad = None

        grads = _get_grads([param_1, param_2])

        self.assertEqual(len(grads), 1)
        self.assertTrue(torch.equal(grads[0], torch.tensor([0.1, 0.2])))

    def test_get_grads_skips_empty_gradients(self) -> None:
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param_2 = torch.tensor([], requires_grad=True)
        param_1.grad = torch.tensor([0.1, 0.2])
        param_2.grad = torch.tensor([])

        grads = _get_grads([param_1, param_2])

        self.assertEqual(len(grads), 1)
        self.assertTrue(torch.equal(grads[0], torch.tensor([0.1, 0.2])))

    def test_get_grads_empty_list(self) -> None:
        grads = _get_grads([])

        self.assertEqual(len(grads), 0)


class TestBatchCalNorm(unittest.TestCase):
    def test_batch_cal_norm_l2(self) -> None:
        grad_1 = torch.tensor([3.0, 4.0])
        grad_2 = torch.tensor([6.0, 8.0])

        result = _batch_cal_norm(
            grad_list=[grad_1, grad_2],
            max_norm=1.0,
            norm_type=2.0,
        )

        expected_norm_1 = (3.0**2 + 4.0**2) ** 0.5
        expected_norm_2 = (6.0**2 + 8.0**2) ** 0.5
        expected_total = (expected_norm_1**2 + expected_norm_2**2) ** 0.5
        expected_result = expected_total**2.0

        self.assertTrue(torch.allclose(result, torch.tensor(expected_result)))

    def test_batch_cal_norm_l1(self) -> None:
        grad_1 = torch.tensor([1.0, 2.0])
        grad_2 = torch.tensor([3.0, 4.0])

        result = _batch_cal_norm(
            grad_list=[grad_1, grad_2],
            max_norm=1.0,
            norm_type=1.0,
        )

        expected_norm_1 = 1.0 + 2.0
        expected_norm_2 = 3.0 + 4.0
        expected_total = expected_norm_1 + expected_norm_2
        expected_result = expected_total**1.0

        self.assertTrue(torch.allclose(result, torch.tensor(expected_result)))

    def test_batch_cal_norm_inf(self) -> None:
        grad_1 = torch.tensor([1.0, 5.0])
        grad_2 = torch.tensor([3.0, 2.0])

        result = _batch_cal_norm(
            grad_list=[grad_1, grad_2],
            max_norm=1.0,
            norm_type=torch.inf,
        )

        expected_result = 5.0

        self.assertTrue(torch.allclose(result, torch.tensor(expected_result)))

    def test_batch_cal_norm_inf_with_empty_tensor(self) -> None:
        """Test that infinity norm handles empty tensors without erroring."""
        grad_1 = torch.tensor([1.0, 5.0])
        grad_2 = torch.tensor([])  # empty tensor

        result = _batch_cal_norm(
            grad_list=[grad_1, grad_2],
            max_norm=1.0,
            norm_type=torch.inf,
        )

        # The empty tensor should be filtered out, so the result should be max of grad_1
        expected_result = 5.0

        self.assertTrue(torch.allclose(result, torch.tensor(expected_result)))

    def test_batch_cal_norm_inf_all_empty_tensors(self) -> None:
        """Test that infinity norm returns -inf when all tensors are empty."""
        grad_1 = torch.tensor([])
        grad_2 = torch.tensor([])

        result = _batch_cal_norm(
            grad_list=[grad_1, grad_2],
            max_norm=1.0,
            norm_type=torch.inf,
        )

        # When all tensors are empty, return -inf (identity for max operation)
        self.assertEqual(result.item(), float("-inf"))


class TestComputeTotalNorm(unittest.TestCase):
    def test_compute_total_norm_replicate_only_l2(self) -> None:
        grad_1 = torch.tensor([3.0, 4.0])
        grad_2 = torch.tensor([6.0, 8.0])

        result = _compute_total_norm(
            replicate_grads=[grad_1, grad_2],
            sharded_grads={},
            norm_type=2.0,
            max_grad_norm=1.0,
        )

        expected_norm_1 = (3.0**2 + 4.0**2) ** 0.5
        expected_norm_2 = (6.0**2 + 8.0**2) ** 0.5
        expected_total = (expected_norm_1**2 + expected_norm_2**2) ** 0.5

        self.assertTrue(torch.allclose(result, torch.tensor(expected_total)))

    def test_compute_total_norm_replicate_only_inf(self) -> None:
        grad_1 = torch.tensor([1.0, 5.0])
        grad_2 = torch.tensor([3.0, 2.0])

        result = _compute_total_norm(
            replicate_grads=[grad_1, grad_2],
            sharded_grads={},
            norm_type=torch.inf,
            max_grad_norm=1.0,
        )

        expected_result = 5.0

        self.assertTrue(torch.allclose(result, torch.tensor(expected_result)))

    def test_compute_total_norm_empty_grads(self) -> None:
        result = _compute_total_norm(
            replicate_grads=[],
            sharded_grads={},
            norm_type=2.0,
            max_grad_norm=1.0,
        )

        self.assertTrue(torch.allclose(result, torch.tensor(0.0)))

    def test_compute_total_norm_single_grad(self) -> None:
        grad_1 = torch.tensor([3.0, 4.0])

        result = _compute_total_norm(
            replicate_grads=[grad_1],
            sharded_grads={},
            norm_type=2.0,
            max_grad_norm=1.0,
        )

        expected_norm = (3.0**2 + 4.0**2) ** 0.5

        self.assertTrue(torch.allclose(result, torch.tensor(expected_norm)))


class TestClipGradNorm(unittest.TestCase):
    def test_clip_grad_norm_clips_gradients(self) -> None:
        max_gradient = 1.0
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([6.0, 8.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1, param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        param_2.grad = torch.tensor([6.0, 8.0])

        mock_pg = MagicMock()
        gradient_clipping_optimizer._sharded_params = {(mock_pg,): [param_1]}
        gradient_clipping_optimizer._replicate_params = [param_2]

        with patch("torch.distributed.all_reduce"):
            total_norm = gradient_clipping_optimizer.clip_grad_norm_()

        self.assertIsNotNone(total_norm)
        self.assertTrue(isinstance(total_norm, torch.Tensor))

    def test_clip_grad_norm_returns_total_norm(self) -> None:
        max_gradient = 100.0
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1},
            {},
            [{"params": [param_1]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])

        mock_pg = MagicMock()
        gradient_clipping_optimizer._sharded_params = {(mock_pg,): [param_1]}
        gradient_clipping_optimizer._replicate_params = []

        with patch("torch.distributed.all_reduce"):
            total_norm = gradient_clipping_optimizer.clip_grad_norm_()

        expected_norm = (3.0**2 + 4.0**2) ** 0.5
        self.assertTrue(
            torch.allclose(
                # pyrefly: ignore[bad-argument-type]
                total_norm,
                torch.tensor(expected_norm),
            )
        )

    def test_clip_grad_norm_no_clipping_when_below_threshold(self) -> None:
        max_gradient = 100.0
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1},
            {},
            [{"params": [param_1]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()
        original_grad = torch.tensor([3.0, 4.0])
        param_1.grad = original_grad.clone()

        mock_pg = MagicMock()
        gradient_clipping_optimizer._sharded_params = {(mock_pg,): [param_1]}
        gradient_clipping_optimizer._replicate_params = []

        with patch("torch.distributed.all_reduce"):
            gradient_clipping_optimizer.clip_grad_norm_()

        self.assertTrue(torch.allclose(param_1.grad, original_grad))

    def test_clip_grad_norm_applies_clipping_when_above_threshold(self) -> None:
        max_gradient = 1.0
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1},
            {},
            [{"params": [param_1]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()
        original_grad = torch.tensor([3.0, 4.0])
        param_1.grad = original_grad.clone()

        mock_pg = MagicMock()
        gradient_clipping_optimizer._sharded_params = {(mock_pg,): [param_1]}
        gradient_clipping_optimizer._replicate_params = []

        with patch("torch.distributed.all_reduce"):
            gradient_clipping_optimizer.clip_grad_norm_()

        grad_norm = torch.linalg.vector_norm(original_grad)
        clip_coef = max_gradient / (grad_norm + 1e-6)
        clip_coef_clamped = min(clip_coef.item(), 1.0)
        expected_grad = original_grad * clip_coef_clamped

        self.assertTrue(torch.allclose(param_1.grad, expected_grad))

    def test_clip_grad_norm_with_replicate_and_sharded_params(self) -> None:
        max_gradient = 1.0
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([6.0, 8.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1, param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        param_2.grad = torch.tensor([6.0, 8.0])

        mock_pg = MagicMock()
        gradient_clipping_optimizer._sharded_params = {(mock_pg,): [param_1]}
        gradient_clipping_optimizer._replicate_params = [param_2]

        with patch("torch.distributed.all_reduce"):
            total_norm = gradient_clipping_optimizer.clip_grad_norm_()

        self.assertIsNotNone(total_norm)
        # pyrefly: ignore[missing-attribute]
        self.assertGreater(total_norm.item(), 0)
