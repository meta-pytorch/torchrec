#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import torch
from torch.autograd import Variable
from torch.distributed._tensor.api import DTensor
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

        torch.testing.assert_close(
            param_1.grad, torch.tensor([0.0, 0.0]), rtol=0, atol=0
        )

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

        torch.testing.assert_close(
            param_1.grad, torch.tensor([0.5, 0.5]), rtol=0, atol=0
        )

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
        torch.testing.assert_close(param_1.grad, expected_grad, rtol=1e-05, atol=1e-08)

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

        torch.testing.assert_close(
            param_1.grad, expected_grad_1, rtol=1e-05, atol=1e-08
        )
        torch.testing.assert_close(
            param_2.grad, expected_grad_2, rtol=1e-05, atol=1e-08
        )

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

        torch.testing.assert_close(
            param_1.grad, torch.tensor([0.0, 0.0]), rtol=0, atol=0
        )

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

        torch.testing.assert_close(
            param_1.grad, torch.tensor([0.5, 0.5]), rtol=0, atol=0
        )

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

        torch.testing.assert_close(param_1.grad, expected_grad, rtol=1e-05, atol=1e-08)

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

        torch.testing.assert_close(
            param_1.grad, expected_grad_1, rtol=1e-05, atol=1e-08
        )
        torch.testing.assert_close(
            param_2.grad, expected_grad_2, rtol=1e-05, atol=1e-08
        )

    def test_last_total_grad_norm_none_before_step(self) -> None:
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.NORM
        )

        self.assertIsNone(gradient_clipping_optimizer.last_total_grad_norm)

    def test_last_total_grad_norm_stored_after_step(self) -> None:
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=10.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        gradient_clipping_optimizer.step()

        expected_norm = (3.0**2 + 4.0**2) ** 0.5
        self.assertIsNotNone(gradient_clipping_optimizer.last_total_grad_norm)
        torch.testing.assert_close(
            gradient_clipping_optimizer.last_total_grad_norm,
            torch.tensor(expected_norm),
            rtol=1e-05,
            atol=1e-08,
        )

    def test_last_total_grad_norm_updated_each_step(self) -> None:
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=10.0, clipping=GradientClipping.NORM
        )

        # Step 1
        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        gradient_clipping_optimizer.step()
        norm_1 = gradient_clipping_optimizer.last_total_grad_norm

        # Step 2 with different gradients
        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([6.0, 8.0])
        gradient_clipping_optimizer.step()
        norm_2 = gradient_clipping_optimizer.last_total_grad_norm

        expected_norm_2 = (6.0**2 + 8.0**2) ** 0.5
        self.assertIsNotNone(norm_2)
        torch.testing.assert_close(
            norm_2,
            torch.tensor(expected_norm_2),
            rtol=1e-05,
            atol=1e-08,
        )
        # Verify it changed from step 1
        assert norm_1 is not None
        assert norm_2 is not None
        self.assertNotEqual(norm_1.item(), norm_2.item())

    def test_last_total_grad_norm_stored_for_value_clipping(self) -> None:
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=1.0,
            clipping=GradientClipping.VALUE,
            track_grad_norm=True,
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        gradient_clipping_optimizer.step()

        expected_norm = (3.0**2 + 4.0**2) ** 0.5
        self.assertIsNotNone(gradient_clipping_optimizer.last_total_grad_norm)
        torch.testing.assert_close(
            gradient_clipping_optimizer.last_total_grad_norm,
            torch.tensor(expected_norm),
            rtol=1e-05,
            atol=1e-08,
        )

    def test_last_total_grad_norm_stored_for_no_clipping(self) -> None:
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=1.0,
            clipping=GradientClipping.NONE,
            track_grad_norm=True,
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        gradient_clipping_optimizer.step()

        expected_norm = (3.0**2 + 4.0**2) ** 0.5
        self.assertIsNotNone(gradient_clipping_optimizer.last_total_grad_norm)
        torch.testing.assert_close(
            gradient_clipping_optimizer.last_total_grad_norm,
            torch.tensor(expected_norm),
            rtol=1e-05,
            atol=1e-08,
        )

    def test_last_total_grad_norm_value_clipping_multi_params(self) -> None:
        # VALUE + track_grad_norm aggregates the pre-clip norm across params via
        # the on-device get_total_norm path (not the racy _compute_total_norm).
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1]}, {"params": [param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=1.0,
            clipping=GradientClipping.VALUE,
            track_grad_norm=True,
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        param_2.grad = torch.tensor([0.0, 12.0])
        gradient_clipping_optimizer.step()

        # norm is computed pre-clip over both params: sqrt(3^2+4^2+12^2) = 13
        expected_norm = (3.0**2 + 4.0**2 + 12.0**2) ** 0.5
        self.assertIsNotNone(gradient_clipping_optimizer.last_total_grad_norm)
        torch.testing.assert_close(
            gradient_clipping_optimizer.last_total_grad_norm,
            torch.tensor(expected_norm),
            rtol=1e-05,
            atol=1e-08,
        )

    def test_last_total_grad_norm_multi_params(self) -> None:
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([6.0, 8.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1]}, {"params": [param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=10.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([3.0, 4.0])
        param_2.grad = torch.tensor([6.0, 8.0])
        gradient_clipping_optimizer.step()

        expected_norm = (3.0**2 + 4.0**2 + 6.0**2 + 8.0**2) ** 0.5
        torch.testing.assert_close(
            gradient_clipping_optimizer.last_total_grad_norm,
            torch.tensor(expected_norm),
            rtol=1e-05,
            atol=1e-08,
        )

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

    def test_clip_inf_norm_with_empty_grad_no_sharded_params(self) -> None:
        """Test that infinity norm gradient clipping handles empty gradients
        when there are no sharded parameters (uses torch.nn.utils.clip_grad_norm_ path).

        This is a regression test for T256567466 / T260346508 where
        torch._foreach_norm raises an error on empty tensors with infinity norm
        during optimizer init_state().
        """
        param_1 = Variable(torch.tensor([3.0, 4.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1, param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=1.0,
            clipping=GradientClipping.NORM,
            norm_type=float("inf"),
        )

        # Simulate empty gradient (as happens during init_state)
        param_1.grad = torch.tensor([1.0, 2.0])
        param_2.grad = torch.tensor([])

        # This should not raise RuntimeError about empty tensor infinity norm
        gradient_clipping_optimizer.step()

    def test_clip_inf_norm_all_empty_grads_no_sharded_params(self) -> None:
        """Test that infinity norm gradient clipping handles the case where
        all parameters have empty gradients and there are no sharded params."""
        param_1 = Variable(torch.tensor([]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1},
            {},
            [{"params": [param_1]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=1.0,
            clipping=GradientClipping.NORM,
            norm_type=float("inf"),
        )

        param_1.grad = torch.tensor([])

        # This should not raise - should gracefully handle empty params
        gradient_clipping_optimizer.step()


class TestGetGrads(unittest.TestCase):
    def test_get_grads_returns_gradients(self) -> None:
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param_2 = torch.tensor([3.0, 4.0], requires_grad=True)
        param_1.grad = torch.tensor([0.1, 0.2])
        param_2.grad = torch.tensor([0.3, 0.4])

        grads = _get_grads([param_1, param_2])

        self.assertEqual(len(grads), 2)
        torch.testing.assert_close(grads[0], torch.tensor([0.1, 0.2]), rtol=0, atol=0)
        torch.testing.assert_close(grads[1], torch.tensor([0.3, 0.4]), rtol=0, atol=0)

    def test_get_grads_skips_none_gradients(self) -> None:
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param_2 = torch.tensor([3.0, 4.0], requires_grad=True)
        param_1.grad = torch.tensor([0.1, 0.2])
        param_2.grad = None

        grads = _get_grads([param_1, param_2])

        self.assertEqual(len(grads), 1)
        torch.testing.assert_close(grads[0], torch.tensor([0.1, 0.2]), rtol=0, atol=0)

    def test_get_grads_skips_empty_gradients(self) -> None:
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param_2 = torch.tensor([], requires_grad=True)
        param_1.grad = torch.tensor([0.1, 0.2])
        param_2.grad = torch.tensor([])

        grads = _get_grads([param_1, param_2])

        self.assertEqual(len(grads), 1)
        torch.testing.assert_close(grads[0], torch.tensor([0.1, 0.2]), rtol=0, atol=0)

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

        torch.testing.assert_close(
            result, torch.tensor(expected_result), rtol=1e-05, atol=1e-08
        )

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

        torch.testing.assert_close(
            result, torch.tensor(expected_result), rtol=1e-05, atol=1e-08
        )

    def test_batch_cal_norm_inf(self) -> None:
        grad_1 = torch.tensor([1.0, 5.0])
        grad_2 = torch.tensor([3.0, 2.0])

        result = _batch_cal_norm(
            grad_list=[grad_1, grad_2],
            max_norm=1.0,
            norm_type=torch.inf,
        )

        expected_result = 5.0

        torch.testing.assert_close(
            result, torch.tensor(expected_result), rtol=1e-05, atol=1e-08
        )

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

        torch.testing.assert_close(
            result, torch.tensor(expected_result), rtol=1e-05, atol=1e-08
        )

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

        torch.testing.assert_close(
            result, torch.tensor(expected_total), rtol=1e-05, atol=1e-08
        )

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

        torch.testing.assert_close(
            result, torch.tensor(expected_result), rtol=1e-05, atol=1e-08
        )

    def test_compute_total_norm_empty_grads(self) -> None:
        result = _compute_total_norm(
            replicate_grads=[],
            sharded_grads={},
            norm_type=2.0,
            max_grad_norm=1.0,
        )

        torch.testing.assert_close(result, torch.tensor(0.0), rtol=1e-05, atol=1e-08)

    def test_compute_total_norm_single_grad(self) -> None:
        grad_1 = torch.tensor([3.0, 4.0])

        result = _compute_total_norm(
            replicate_grads=[grad_1],
            sharded_grads={},
            norm_type=2.0,
            max_grad_norm=1.0,
        )

        expected_norm = (3.0**2 + 4.0**2) ** 0.5

        torch.testing.assert_close(
            result, torch.tensor(expected_norm), rtol=1e-05, atol=1e-08
        )


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
        torch.testing.assert_close(
            # pyrefly: ignore[bad-argument-type]
            total_norm,
            torch.tensor(expected_norm),
            rtol=1e-05,
            atol=1e-08,
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

        torch.testing.assert_close(param_1.grad, original_grad, rtol=1e-05, atol=1e-08)

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

        torch.testing.assert_close(param_1.grad, expected_grad, rtol=1e-05, atol=1e-08)

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


def _clipper(
    params: list[torch.Tensor],
    clipping: GradientClipping = GradientClipping.NORM,
    max_gradient: float = 1.0,
    norm_type: float | str = 2.0,
) -> GradientClippingOptimizer:
    keyed_optimizer = DummyKeyedOptimizer(
        {f"param_{i}": p for i, p in enumerate(params)}, {}, [{"params": params}]
    )
    return GradientClippingOptimizer(
        optimizer=keyed_optimizer,
        clipping=clipping,
        max_gradient=max_gradient,
        norm_type=norm_type,
    )


class StepGradScaleFoldTest(unittest.TestCase):
    """Tests the gradient scale a NORM clipper applies inside its clip multiply."""

    def _grads(self, seed: int = 0) -> list[torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return [
            torch.randn(5, generator=generator) * 3,
            torch.randn(3, 2, generator=generator) * 3,
        ]

    def _run(
        self,
        scale: float,
        fold: bool,
        max_gradient: float,
        norm_type: float | str,
        sharded: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        grads = self._grads()
        params: list[torch.Tensor] = [
            Variable(torch.zeros_like(g), requires_grad=True) for g in grads
        ]
        clipper = _clipper(params, max_gradient=max_gradient, norm_type=norm_type)
        if sharded:
            clipper._sharded_params = {(MagicMock(),): [params[0]]}
            clipper._replicate_params = [params[1]]
        for param, grad in zip(params, grads):
            param.grad = grad.clone()
        if fold:
            clipper.set_step_grad_scale(scale)
        else:
            torch._foreach_mul_([cast(torch.Tensor, p.grad) for p in params], scale)
        with patch("torch.distributed.all_reduce"):
            clipper.step()
        norm = clipper.last_total_grad_norm
        assert norm is not None
        return [cast(torch.Tensor, p.grad) for p in params], norm

    def test_power_of_two_scale_folded_equals_separate_multiply_bitwise(self) -> None:
        """A power-of-two scale folded into the clip matches a separate multiply bitwise."""
        for scale in (0.5, 0.25):
            for norm_type in (2.0, "inf"):
                for max_gradient in (0.1, 1e6):
                    for sharded in (False, True):
                        with self.subTest(
                            scale=scale,
                            norm_type=norm_type,
                            max_gradient=max_gradient,
                            sharded=sharded,
                        ):
                            folded, folded_norm = self._run(
                                scale, True, max_gradient, norm_type, sharded
                            )
                            separate, separate_norm = self._run(
                                scale, False, max_gradient, norm_type, sharded
                            )
                            for got, want in zip(folded, separate):
                                self.assertTrue(torch.equal(got, want))
                            self.assertTrue(torch.equal(folded_norm, separate_norm))

    def test_other_scale_folded_matches_separate_multiply(self) -> None:
        """A scale that is not a power of two matches a separate multiply to rounding."""
        for norm_type in (2.0, "inf"):
            for sharded in (False, True):
                with self.subTest(norm_type=norm_type, sharded=sharded):
                    folded, folded_norm = self._run(
                        1 / 3, True, 0.1, norm_type, sharded
                    )
                    separate, separate_norm = self._run(
                        1 / 3, False, 0.1, norm_type, sharded
                    )
                    for got, want in zip(folded, separate):
                        torch.testing.assert_close(got, want)
                    torch.testing.assert_close(folded_norm, separate_norm)

    def test_scale_of_one_is_a_plain_step(self) -> None:
        """A scale of 1 leaves the step bitwise unchanged."""
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                with_scale, with_scale_norm = self._run(1.0, True, 0.1, 2.0, sharded)
                plain, plain_norm = self._run(1.0, False, 0.1, 2.0, sharded)
                for got, want in zip(with_scale, plain):
                    self.assertTrue(torch.equal(got, want))
                self.assertTrue(torch.equal(with_scale_norm, plain_norm))

    def test_scale_applies_to_one_step(self) -> None:
        """The scale is used by the next step only, and a new scale replaces it."""
        param = Variable(torch.zeros(2), requires_grad=True)
        clipper = _clipper([param], max_gradient=1e6)
        clipper.set_step_grad_scale(0.5)
        param.grad = torch.tensor([2.0, 4.0])
        clipper.step()
        torch.testing.assert_close(param.grad, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
        param.grad = torch.tensor([2.0, 4.0])
        clipper.step()
        torch.testing.assert_close(param.grad, torch.tensor([2.0, 4.0]), rtol=0, atol=0)
        clipper.set_step_grad_scale(0.25)
        clipper.step()
        torch.testing.assert_close(param.grad, torch.tensor([0.5, 1.0]), rtol=0, atol=0)

    def test_logged_norm_is_the_scaled_norm(self) -> None:
        """The logged norm is the norm of the scaled gradients."""
        param = Variable(torch.zeros(2), requires_grad=True)
        clipper = _clipper([param], max_gradient=1e6)
        clipper.set_step_grad_scale(0.5)
        param.grad = torch.tensor([3.0, 4.0])
        clipper.step()
        torch.testing.assert_close(
            clipper.last_total_grad_norm, torch.tensor(2.5), rtol=0, atol=0
        )

    def test_only_norm_clipping_takes_a_scale(self) -> None:
        """Only NORM clipping folds a scale; VALUE and NONE reject one."""
        param = Variable(torch.zeros(2), requires_grad=True)
        self.assertTrue(_clipper([param]).can_fold_step_grad_scale)
        for mode in (GradientClipping.VALUE, GradientClipping.NONE):
            with self.subTest(clipping=mode):
                clipper = _clipper([param], clipping=mode)
                self.assertFalse(clipper.can_fold_step_grad_scale)
                clipper.set_step_grad_scale(0.5)
                param.grad = torch.ones(2)
                with self.assertRaisesRegex(ValueError, "needs NORM clipping"):
                    clipper.step()

    def test_dtensor_params_on_the_replicate_path_do_not_fold(self) -> None:
        """The replicate-only path cannot scale DTensor gradients, so it does not fold."""
        param = Variable(torch.zeros(2), requires_grad=True)
        clipper = _clipper([param])
        clipper._replicate_params = [MagicMock(spec=DTensor)]
        self.assertFalse(clipper.can_fold_step_grad_scale)
        clipper._sharded_params = {(MagicMock(),): [param]}
        self.assertTrue(clipper.can_fold_step_grad_scale)
