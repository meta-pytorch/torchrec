#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
import unittest
import weakref
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils._python_dispatch import TorchDispatchMode
from torchrec.optim import clipping
from torchrec.optim.clipping import (
    _batch_cal_norm,
    _clip_flat_split,
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


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _flat_params(
    seed: int, nonfinite: Optional[float] = None, scale: float = 1.0
) -> Tuple[List[torch.nn.Parameter], List[torch.Tensor]]:
    """Params whose grads mimic DDP ``gradient_as_bucket_view``: the grads of
    buffers A and B tile them exactly (listed out of storage order), buffer C
    has a hole no grad covers, plus a non-view fp32 grad and a bf16 grad."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(35 + 11 + 24, generator=gen)
    b = torch.randn(13 + 18, generator=gen)
    c = torch.randn(30, generator=gen)
    if nonfinite is not None:
        a[40] = nonfinite
    for buf in (a, b, c):
        buf.mul_(scale)
    grads = [
        a[35:46].view(11),
        b[13:31].view(2, 9),
        torch.randn(6, generator=gen) * scale,
        a[0:35].view(7, 5),
        c[0:10].view(10),
        (torch.randn(4, generator=gen) * scale).bfloat16(),
        b[0:13].view(13),
        a[46:70].view(3, 4, 2),
        c[15:30].view(3, 5),
    ]
    params = []
    for g in grads:
        p = torch.nn.Parameter(torch.zeros(g.shape, dtype=g.dtype))
        p.grad = g
        params.append(p)
    return params, [a, b, c]


def _flat_optimizer(
    params: List[torch.nn.Parameter], norm_type: float = 2.0
) -> GradientClippingOptimizer:
    keyed_optimizer = DummyKeyedOptimizer(
        {f"param_{i}": p for i, p in enumerate(params)}, {}, [{"params": params}]
    )
    return GradientClippingOptimizer(
        optimizer=keyed_optimizer,
        max_gradient=1.0,
        clipping=GradientClipping.NORM,
        norm_type=norm_type,
    )


def _flat_env(
    flat: bool, check: bool = False, segments: bool = False
) -> Dict[str, str]:
    return {
        "ONEFLOW_CLIP_FLAT": "1" if flat else "0",
        "ONEFLOW_CLIP_FLAT_CHECK": "1" if check else "0",
        "ONEFLOW_CLIP_FLAT_SEGMENTS": "1" if segments else "0",
    }


class _OpLog(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls: List[Tuple[str, List[int]]] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: Any = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        first = args[0] if args else None
        tensors = first if isinstance(first, (list, tuple)) else [first]
        numels = [t.numel() for t in tensors if isinstance(t, torch.Tensor)]
        self.calls.append((func.overloadpacket.__name__, numels))
        return func(*args, **(kwargs or {}))


class TestClipFlat(unittest.TestCase):
    def _step(
        self,
        params: List[torch.nn.Parameter],
        flat: bool,
        check: bool = False,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        optimizer = _flat_optimizer(params, norm_type)
        with mock.patch.dict(os.environ, _flat_env(flat, check)):
            optimizer.step()
        norm = optimizer.last_total_grad_norm
        assert norm is not None
        return norm

    def _assert_clipped_with(
        self,
        params: List[torch.nn.Parameter],
        seed: int,
        total_norm: torch.Tensor,
        nonfinite: Optional[float] = None,
    ) -> None:
        """The grads equal clip_grads_with_norm_ with ``total_norm`` bitwise."""
        expected, _ = _flat_params(seed, nonfinite)
        torch.nn.utils.clip_grads_with_norm_(expected, 1.0, total_norm.clone())
        for p, q in zip(params, expected):
            # pyrefly: ignore [bad-argument-type]
            self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))

    def test_split_finds_tiled_buffers(self) -> None:
        params, (a, b, c) = _flat_params(0)
        # pyrefly: ignore [bad-argument-type]
        bases, rest = _clip_flat_split([p.grad for p in params])
        self.assertEqual([id(base) for base, _ in bases], [id(a), id(b)])
        self.assertEqual(sorted(len(m) for _, m in bases), [2, 3])
        self.assertEqual(len(rest), 4)
        self.assertTrue(all(g._base is not a and g._base is not b for g in rest))
        # Duplicated or overlapping views do not tile.
        g0 = params[3].grad
        # pyrefly: ignore [bad-argument-type]
        self.assertEqual(_clip_flat_split([g0, g0, params[0].grad])[0], [])
        x = torch.zeros(20)
        y = torch.zeros(30)
        for views in (
            [x[0:10], x[5:15]],  # sizes add up, but overlap and a gap
            [x[0:20:2], x[10:20]],  # a strided view
            [z.view(4) for z in torch.zeros(8, dtype=torch.bfloat16).split(4)],
            [y[0:10], y[10:30], y[20:30]],  # overlap in the last slot
            [y[0:10], y[10:30], y[10:30]],  # duplicate of the last view
            [y[0:10], y[10:20]],  # the tail of the buffer is not covered
        ):
            bases, rest = _clip_flat_split(views)
            self.assertEqual(bases, [])
            self.assertEqual(len(rest), len(views))

    def test_flat_clip_matches_per_tensor(self) -> None:
        flat_params, (_, _, c) = _flat_params(0)
        hole = c[10:15].clone()
        ref_params, _ = _flat_params(0)
        flat_norm = self._step(flat_params, flat=True)
        ref_norm = self._step(ref_params, flat=False)
        self.assertGreater(float(ref_norm), 2.0)  # clipping is active
        torch.testing.assert_close(flat_norm, ref_norm, rtol=1e-6, atol=0)
        self._assert_clipped_with(flat_params, 0, flat_norm)
        self.assertTrue(torch.equal(c[10:15], hole))
        for p, q in zip(flat_params, ref_params):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=0)

    def test_flat_clip_below_max_norm_keeps_grads(self) -> None:
        params, _ = _flat_params(0, scale=0.01)
        norm = self._step(params, flat=True)
        self.assertLess(float(norm), 1.0)
        expected, _ = _flat_params(0, scale=0.01)
        for p, q in zip(params, expected):
            # pyrefly: ignore [bad-argument-type]
            self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))

    def test_flat_clip_one_norm_and_mul_per_buffer(self) -> None:
        params, (a, b, _) = _flat_params(0)
        log = _OpLog()
        with log:
            self._step(params, flat=True)
        norms = [n for op, n in log.calls if op == "linalg_vector_norm"]
        muls = [n for op, n in log.calls if op == "mul_"]
        foreach_norms = [n for op, n in log.calls if op == "_foreach_norm"]
        foreach_muls = [n for op, n in log.calls if op == "_foreach_mul_"]
        self.assertIn([a.numel()], norms)
        self.assertIn([b.numel()], norms)
        self.assertIn([a.numel()], muls)
        self.assertIn([b.numel()], muls)
        rest = sorted([6, 10, 15, 4])
        self.assertEqual(sorted(n for c in foreach_norms for n in c), rest)
        self.assertEqual(sorted(n for c in foreach_muls for n in c), rest)

    def test_flat_clip_declines(self) -> None:
        for norm_type, params_fn, reason in (
            (float("inf"), lambda: _flat_params(0)[0], "norm_type"),
            (
                2.0,
                lambda: [
                    # pyrefly: ignore [missing-attribute]
                    torch.nn.Parameter(p.grad.clone())
                    for p in _flat_params(0)[0]
                ],
                "no fp32 grad buffer",
            ),
            (
                2.0,
                lambda: _flat_params(0)[0]
                + [torch.nn.Parameter(torch.ones(3, dtype=torch.float64))],
                "promote to torch.float64",
            ),
        ):
            ref = params_fn()
            for p in ref:
                p.grad = p.detach().clone() if p.grad is None else p.grad
            ref_norm = self._step(ref, flat=False, norm_type=norm_type)
            got = params_fn()
            for p in got:
                p.grad = p.detach().clone() if p.grad is None else p.grad
            with self.assertLogs(level="INFO") as logs:
                got_norm = self._step(got, flat=True, norm_type=norm_type)
            self.assertTrue(any(reason in line for line in logs.output), logs.output)
            self.assertTrue(torch.equal(_bits(got_norm), _bits(ref_norm)))
            for p, q in zip(got, ref):
                # pyrefly: ignore [bad-argument-type]
                self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))

    def test_flat_clip_toggles_per_call(self) -> None:
        params, buffers = _flat_params(0)
        optimizer = _flat_optimizer(params)
        for step, flat in enumerate((True, False, True, False)):
            fresh, fresh_buffers = _flat_params(step)
            for buf, new in zip(buffers, fresh_buffers):
                buf.copy_(new)
            for p, q in zip(params, fresh):
                # pyrefly: ignore [missing-attribute]
                if p.grad._base is None:
                    # pyrefly: ignore [missing-attribute]
                    p.grad.copy_(q.grad)
            log = _OpLog()
            with mock.patch.dict(os.environ, _flat_env(flat)), log:
                optimizer.step()
            norm = optimizer.last_total_grad_norm
            assert norm is not None
            flat_ops = [n for op, n in log.calls if op == "linalg_vector_norm"]
            self.assertEqual([buffers[0].numel()] in flat_ops, flat)
            if not flat:
                ref_params, _ = _flat_params(step)
                ref_norm = self._step(ref_params, flat=False)
                self.assertTrue(torch.equal(_bits(norm), _bits(ref_norm)))
            self._assert_clipped_with(params, step, norm)

    def test_flat_clip_nonfinite(self) -> None:
        for bad in (float("nan"), float("inf")):
            params, _ = _flat_params(0, nonfinite=bad)
            ref_params, _ = _flat_params(0, nonfinite=bad)
            norm = self._step(params, flat=True)
            ref_norm = self._step(ref_params, flat=False)
            self.assertTrue(torch.equal(_bits(norm), _bits(ref_norm)))
            self._assert_clipped_with(params, 0, norm, nonfinite=bad)
            for p, q in zip(params, ref_params):
                # pyrefly: ignore [bad-argument-type]
                self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))

    def test_flat_clip_check_passes(self) -> None:
        params, buffers = _flat_params(0)
        optimizer = _flat_optimizer(params)
        with self.assertLogs(level="INFO") as logs:
            for step in range(4):
                for buf, new in zip(buffers, _flat_params(step)[1]):
                    buf.copy_(new)
                for p, q in zip(params, _flat_params(step)[0]):
                    # pyrefly: ignore [missing-attribute]
                    if p.grad._base is None:
                        # pyrefly: ignore [missing-attribute]
                        p.grad.copy_(q.grad)
                with mock.patch.dict(os.environ, _flat_env(True, check=True)):
                    optimizer.step()
                norm = optimizer.last_total_grad_norm
                assert norm is not None
                self._assert_clipped_with(params, step, norm)
        self.assertFalse(any("ERROR" in line for line in logs.output))
        self.assertEqual(
            sum("matched the per-tensor path" in line for line in logs.output), 1
        )

    def test_flat_clip_check_norm_mismatch_raises(self) -> None:
        params, _ = _flat_params(0)
        optimizer = _flat_optimizer(params)
        with mock.patch.object(clipping, "_CLIP_FLAT_NORM_RTOL", -1.0):
            with self.assertLogs(level="ERROR") as logs, self.assertRaisesRegex(
                RuntimeError, "rank 0: total norm"
            ):
                with mock.patch.dict(os.environ, _flat_env(True, check=True)):
                    optimizer.step()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("rank 0", logs.output[0])
        # Not latched: the next flat step runs the flat path again.
        log = _OpLog()
        with mock.patch.dict(os.environ, _flat_env(True)), log:
            optimizer.step()
        self.assertIn([70], [n for op, n in log.calls if op == "linalg_vector_norm"])

    def test_flat_clip_check_mul_mismatch_raises(self) -> None:
        params, _ = _flat_params(0)
        optimizer = _flat_optimizer(params)
        mul_ = torch.Tensor.mul_

        def bad_mul_(t: torch.Tensor, other: Any) -> torch.Tensor:
            mul_(t, other)
            return t.add_(1.0) if t.numel() == 70 else t

        with mock.patch.object(torch.Tensor, "mul_", bad_mul_):
            with self.assertLogs(level="ERROR") as logs, self.assertRaisesRegex(
                RuntimeError, "rank 0: the clipped buffer of 70 elements"
            ):
                with mock.patch.dict(os.environ, _flat_env(True, check=True)):
                    optimizer.step()
        self.assertEqual(len(logs.output), 1)

    def test_flat_clip_real_ddp_buckets(self) -> None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp.name}", rank=0, world_size=1
        )
        try:
            torch.manual_seed(0)
            model = torch.nn.parallel.DistributedDataParallel(
                torch.nn.Sequential(
                    torch.nn.Linear(64, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 8),
                ),
                gradient_as_bucket_view=True,
                bucket_cap_mb=0.5,
            )
            # DDP rebuilds its buckets after the first backward.
            for _ in range(2):
                model(torch.randn(16, 64)).square().sum().backward()
            params = list(model.parameters())
            # pyrefly: ignore [bad-argument-type]
            bases, rest = _clip_flat_split([p.grad for p in params])
            self.assertGreaterEqual(len(bases), 2)
            self.assertEqual(rest, [])
            # pyrefly: ignore [bad-argument-type]
            ref_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
            # pyrefly: ignore [missing-attribute]
            originals = [p.grad.clone() for p in params]
            norm = self._step(params, flat=True)
            # The CPU vector_norm of a whole bucket accumulates long runs
            # sequentially (about 1e-5 relative here); on GPU it is a tree.
            torch.testing.assert_close(norm, ref_norm, rtol=3e-5, atol=0)
            expected = []
            for g in originals:
                q = torch.nn.Parameter(torch.zeros_like(g))
                q.grad = g
                expected.append(q)
            torch.nn.utils.clip_grads_with_norm_(expected, 1.0, norm)
            for p, q in zip(params, expected):
                # pyrefly: ignore [bad-argument-type]
                self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))
        finally:
            dist.destroy_process_group()
            os.unlink(tmp.name)


class TestClipFlatSegments(unittest.TestCase):
    def test_split_segments(self) -> None:
        d = torch.zeros(100)
        ours = [d[60:100], d[0:30], d[30:50]]  # d[50:60] is another optimizer's
        bases, rest = _clip_flat_split(ours, 1)
        self.assertEqual(
            sorted((b.storage_offset(), b.numel(), len(m)) for b, m in bases),
            [(0, 50, 2), (60, 40, 1)],
        )
        self.assertTrue(all(b._base is d for b, _ in bases))
        self.assertEqual(rest, [])
        bases, rest = _clip_flat_split(ours, 45)
        self.assertEqual([(b.storage_offset(), b.numel()) for b, _ in bases], [(0, 50)])
        self.assertEqual([g.numel() for g in rest], [40])
        bases, _ = _clip_flat_split(ours, 50)
        self.assertEqual([(b.storage_offset(), b.numel()) for b, _ in bases], [(0, 50)])
        bases, rest = _clip_flat_split(ours, None)
        self.assertEqual(bases, [])
        self.assertEqual(len(rest), 3)
        # A tiled base is still the base itself, however short.
        bases, rest = _clip_flat_split([d[50:100], d[0:50]], 1000)
        self.assertEqual(len(bases), 1)
        self.assertIs(bases[0][0], d)
        # Overlapping or duplicated grads: the whole base stays per-tensor.
        for views in (
            [d[0:30], d[20:40], d[60:100]],
            [d[0:30], d[0:30], d[60:100]],
        ):
            bases, rest = _clip_flat_split(views, 1)
            self.assertEqual(bases, [])
            self.assertEqual(len(rest), 3)

    def _step(
        self, params: List[torch.nn.Parameter], **env: bool
    ) -> Tuple[GradientClippingOptimizer, _OpLog]:
        optimizer = _flat_optimizer(params)
        log = _OpLog()
        with mock.patch.object(clipping, "_CLIP_FLAT_MIN_SEGMENT", 8):
            with mock.patch.dict(os.environ, _flat_env(True, **env)), log:
                optimizer.step()
        return optimizer, log

    def test_segments_clip_matches_per_tensor(self) -> None:
        params, (_, _, c) = _flat_params(0)
        hole = c[10:15].clone()
        optimizer, log = self._step(params, segments=True)
        norm = optimizer.last_total_grad_norm
        assert norm is not None
        ref_params, _ = _flat_params(0)
        ref = _flat_optimizer(ref_params)
        with mock.patch.dict(os.environ, _flat_env(False)):
            ref.step()
        ref_norm = ref.last_total_grad_norm
        assert ref_norm is not None
        torch.testing.assert_close(norm, ref_norm, rtol=1e-6, atol=0)
        expected, _ = _flat_params(0)
        torch.nn.utils.clip_grads_with_norm_(expected, 1.0, norm.clone())
        for p, q in zip(params, expected):
            # pyrefly: ignore [bad-argument-type]
            self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))
        self.assertTrue(torch.equal(c[10:15], hole))
        norms = [n for op, n in log.calls if op == "linalg_vector_norm"]
        self.assertIn([10], norms)
        self.assertIn([15], norms)
        foreach_norms = [n for op, n in log.calls if op == "_foreach_norm"]
        self.assertEqual(sorted(n for c_ in foreach_norms for n in c_), [4, 6])

    def test_segments_default_threshold(self) -> None:
        n = 1 << 21  # the default _CLIP_FLAT_MIN_SEGMENT
        d = torch.randn(2 * n)
        params = []
        # A run of n elements, a gap of one, a run of n - 1 elements.
        for g in (d[0:n], d[n + 1 : 2 * n], torch.randn(5)):
            p = torch.nn.Parameter(torch.zeros(g.shape))
            p.grad = g
            params.append(p)
        optimizer = _flat_optimizer(params)
        log = _OpLog()
        with mock.patch.dict(os.environ, _flat_env(True, segments=True)), log:
            optimizer.step()
        norms = [c for op, c in log.calls if op == "linalg_vector_norm"]
        self.assertIn([n], norms)
        self.assertNotIn([n - 1], norms)
        foreach = [c for op, c in log.calls if op == "_foreach_norm"]
        self.assertEqual(sorted(x for c in foreach for x in c), [5, n - 1])

    def test_check_holds_one_clone_at_a_time(self) -> None:
        clone = torch.Tensor.clone
        live: List[Any] = []
        peak = [0]

        def tracking_clone(t: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            out = clone(t, *args, **kwargs)
            live[:] = [r for r in live if r() is not None] + [weakref.ref(out)]
            peak[0] = max(peak[0], len(live))
            return out

        params, _ = _flat_params(0)
        with mock.patch.object(torch.Tensor, "clone", tracking_clone):
            optimizer, _ = self._step(params, segments=True, check=True)
        self.assertEqual(peak[0], 1)
        self.assertGreaterEqual(len(live), 1)

    def test_segments_off_keeps_full_tiles_only(self) -> None:
        params, _ = _flat_params(0)
        _, log = self._step(params, segments=False)
        norms = [n for op, n in log.calls if op == "linalg_vector_norm"]
        self.assertNotIn([10], norms)
        self.assertNotIn([15], norms)

    def test_segments_check(self) -> None:
        params, _ = _flat_params(0)
        with self.assertLogs(level="INFO") as logs:
            self._step(params, segments=True, check=True)
        self.assertFalse(any("ERROR" in line for line in logs.output))
        mul_ = torch.Tensor.mul_

        def bad_mul_(t: torch.Tensor, other: Any) -> torch.Tensor:
            mul_(t, other)
            return t.add_(1.0) if t.numel() == 15 else t

        params, _ = _flat_params(0)
        with mock.patch.object(torch.Tensor, "mul_", bad_mul_):
            with self.assertLogs(level="ERROR"), self.assertRaisesRegex(
                RuntimeError, "rank 0: the clipped buffer of 15 elements"
            ):
                self._step(params, segments=True, check=True)

    def test_segments_two_optimizers_share_ddp_buckets(self) -> None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp.name}", rank=0, world_size=1
        )
        try:
            torch.manual_seed(0)
            model = torch.nn.parallel.DistributedDataParallel(
                torch.nn.Sequential(
                    torch.nn.Linear(64, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 8),
                ),
                gradient_as_bucket_view=True,
            )
            for _ in range(2):  # DDP rebuilds its buckets after the first one
                model(torch.randn(16, 64)).square().sum().backward()
            params = list(model.parameters())
            # Weights in one optimizer, biases in another, as with a separate
            # no-weight-decay group: no bucket is tiled by either.
            groups = [params[0::2], params[1::2]]
            # pyrefly: ignore [missing-attribute]
            originals = [[p.grad.clone() for p in g] for g in groups]
            for group, grads in zip(groups, originals):
                # pyrefly: ignore [bad-argument-type]
                bases, _ = _clip_flat_split([p.grad for p in group])
                self.assertEqual(bases, [])
                optimizer = _flat_optimizer(group)
                with mock.patch.object(clipping, "_CLIP_FLAT_MIN_SEGMENT", 1000):
                    with mock.patch.dict(os.environ, _flat_env(True, segments=True)):
                        optimizer.step()
                norm = optimizer.last_total_grad_norm
                assert norm is not None
                torch.testing.assert_close(
                    norm, torch.nn.utils.get_total_norm(grads), rtol=3e-5, atol=0
                )
                expected = []
                for g in grads:
                    q = torch.nn.Parameter(torch.zeros_like(g))
                    q.grad = g
                    expected.append(q)
                torch.nn.utils.clip_grads_with_norm_(expected, 1.0, norm)
                for p, q in zip(group, expected):
                    # pyrefly: ignore [bad-argument-type]
                    self.assertTrue(torch.equal(_bits(p.grad), _bits(q.grad)))
            # pyrefly: ignore [bad-argument-type]
            segs, _ = _clip_flat_split([p.grad for p in groups[0]], 1000)
            # Each weight is its own segment, between two biases.
            self.assertEqual(
                sorted(m[0].numel() for _, m in segs), [8 * 256, 64 * 256, 256 * 256]
            )
        finally:
            dist.destroy_process_group()
            os.unlink(tmp.name)


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
