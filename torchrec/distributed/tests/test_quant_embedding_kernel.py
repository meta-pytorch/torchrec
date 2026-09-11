#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, cast, Dict
from unittest.mock import MagicMock, patch

import torch
import torchrec.distributed.quant_embedding_kernel as qek
from torchrec.distributed.fused_params import (
    FUSED_PARAM_IS_DEVICE_RO,
    FUSED_PARAM_USE_CPU_KJT_FOR_FX_TRACING,
    is_fused_param_device_ro,
    is_fused_param_use_cpu_kjt_for_fx_tracing,
    tbe_fused_params,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _CpuKJTUnwrapModule(torch.nn.Module):
    def __init__(self, *, use_fx_helper: bool) -> None:
        super().__init__()
        self._use_fx_helper = use_fx_helper

    def forward(
        self, features: KeyedJaggedTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self._use_fx_helper:
            return qek._unwrap_kjt_for_cpu_fx(features, weighted=False)
        return qek._unwrap_kjt_for_cpu(features, weighted=False)


class QuantEmbeddingKernelFxTest(unittest.TestCase):
    def test_cpu_kjt_fx_helper_is_opt_in(self) -> None:
        default_graph = torch.fx.symbolic_trace(
            _CpuKJTUnwrapModule(use_fx_helper=False)
        )
        default_call_targets = [
            node.target
            for node in default_graph.graph.nodes
            if node.op == "call_function"
        ]
        self.assertNotIn(qek._unwrap_kjt_for_cpu, default_call_targets)
        self.assertNotIn(qek._unwrap_kjt_for_cpu_fx, default_call_targets)

        graph_module = torch.fx.symbolic_trace(_CpuKJTUnwrapModule(use_fx_helper=True))

        cpu_unwrap_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target is qek._unwrap_kjt_for_cpu_fx
        ]
        self.assertEqual(len(cpu_unwrap_nodes), 1)

        fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        fake_values = fake_mode.from_tensor(torch.tensor([1, 2], dtype=torch.int64))
        fake_offsets = fake_mode.from_tensor(torch.tensor([0, 2], dtype=torch.int32))
        features = KeyedJaggedTensor(
            keys=["feature"],
            values=fake_values,
            offsets=fake_offsets,
        )

        values, offsets, weights = graph_module(features)

        self.assertIsInstance(values, torch._subclasses.FakeTensor)
        self.assertIsInstance(offsets, torch._subclasses.FakeTensor)
        self.assertEqual(values.dtype, torch.int64)
        self.assertEqual(offsets.dtype, torch.int64)
        self.assertIsNone(weights)

    def test_cpu_kjt_fx_fused_param_is_internal_to_torchrec(self) -> None:
        fused_params: Dict[str, Any] = {
            FUSED_PARAM_USE_CPU_KJT_FOR_FX_TRACING: True,
            "output_dtype": object(),
        }

        self.assertTrue(is_fused_param_use_cpu_kjt_for_fx_tracing(fused_params))
        self.assertFalse(is_fused_param_use_cpu_kjt_for_fx_tracing({}))
        filtered_params = tbe_fused_params(fused_params)
        self.assertIsNotNone(filtered_params)
        self.assertNotIn(FUSED_PARAM_USE_CPU_KJT_FOR_FX_TRACING, filtered_params)
        self.assertIn("output_dtype", filtered_params)


class QuantBatchedEmbeddingBagDeviceRoTest(unittest.TestCase):
    def _make_bag(
        self,
        *,
        is_device_ro: bool,
        lengths_to_tbe: bool = False,
        runtime_device: str = "cuda",
        use_cpu_kjt_for_fx_tracing: bool = False,
    ) -> qek.QuantBatchedEmbeddingBag:
        bag = qek.QuantBatchedEmbeddingBag.__new__(qek.QuantBatchedEmbeddingBag)
        bag._runtime_device = torch.device(runtime_device)
        bag.lengths_to_tbe = lengths_to_tbe
        bag._is_device_ro = is_device_ro
        bag._use_cpu_kjt_for_fx_tracing = use_cpu_kjt_for_fx_tracing
        bag._config = MagicMock(is_weighted=False)
        return bag

    def test_cpu_forward_uses_unwrapped_helper_by_default(self) -> None:
        bag = self._make_bag(is_device_ro=False, runtime_device="cpu")
        output = torch.tensor([1.0])
        indices = torch.tensor([1])
        offsets = torch.tensor([0, 1])
        bag._emb_module_forward = MagicMock(return_value=output)
        features = cast(KeyedJaggedTensor, object())

        with patch.object(
            qek, "_unwrap_kjt_for_cpu", return_value=(indices, offsets, None)
        ) as mock_unwrapped, patch.object(
            qek, "_unwrap_kjt_for_cpu_fx", return_value=(indices, offsets, None)
        ) as mock_fx_wrapped:
            result = bag.forward(features)

        self.assertIs(result, output)
        mock_unwrapped.assert_called_once_with(features, False)
        mock_fx_wrapped.assert_not_called()

    def test_cpu_forward_uses_fx_helper_when_enabled(self) -> None:
        bag = self._make_bag(
            is_device_ro=False,
            runtime_device="cpu",
            use_cpu_kjt_for_fx_tracing=True,
        )
        output = torch.tensor([1.0])
        indices = torch.tensor([1])
        offsets = torch.tensor([0, 1])
        bag._emb_module_forward = MagicMock(return_value=output)
        features = cast(KeyedJaggedTensor, object())

        with patch.object(
            qek, "_unwrap_kjt_for_cpu", return_value=(indices, offsets, None)
        ) as mock_unwrapped, patch.object(
            qek, "_unwrap_kjt_for_cpu_fx", return_value=(indices, offsets, None)
        ) as mock_fx_wrapped:
            result = bag.forward(features)

        self.assertIs(result, output)
        mock_fx_wrapped.assert_called_once_with(features, False)
        mock_unwrapped.assert_not_called()

    def test_forward_uses_ro_unwrap_for_cuda_devicero(self) -> None:
        bag = self._make_bag(is_device_ro=True)
        output = torch.tensor([3.0])
        indices = torch.tensor([1])
        offsets = torch.tensor([0, 1])
        emb_forward = MagicMock(return_value=output)
        bag._emb_module_forward = emb_forward
        features = cast(KeyedJaggedTensor, object())

        with patch.object(
            qek, "_unwrap_ro_kjt", return_value=(indices, offsets, None)
        ) as mock_ro_unwrap, patch.object(
            qek, "_unwrap_kjt", return_value=(indices, offsets, None)
        ) as mock_regular_unwrap:
            result = bag.forward(features)

        self.assertIs(result, output)
        mock_ro_unwrap.assert_called_once_with(features)
        mock_regular_unwrap.assert_not_called()
        emb_forward.assert_called_once_with(indices, offsets, None)

    def test_forward_uses_regular_unwrap_for_cuda_non_devicero(self) -> None:
        bag = self._make_bag(is_device_ro=False)
        output = torch.tensor([5.0])
        indices = torch.tensor([2])
        offsets = torch.tensor([0, 1])
        emb_forward = MagicMock(return_value=output)
        bag._emb_module_forward = emb_forward
        features = cast(KeyedJaggedTensor, object())

        with patch.object(
            qek, "_unwrap_ro_kjt", return_value=(indices, offsets, None)
        ) as mock_ro_unwrap, patch.object(
            qek, "_unwrap_kjt", return_value=(indices, offsets, None)
        ) as mock_regular_unwrap:
            result = bag.forward(features)

        self.assertIs(result, output)
        mock_regular_unwrap.assert_called_once_with(features)
        mock_ro_unwrap.assert_not_called()
        emb_forward.assert_called_once_with(indices, offsets, None)

    def test_lengths_to_tbe_uses_lengths_unwrap_for_devicero(self) -> None:
        bag = self._make_bag(is_device_ro=True, lengths_to_tbe=True)
        output = torch.tensor([7.0])
        indices = torch.tensor([4])
        lengths = torch.tensor([1])
        emb_forward = MagicMock(return_value=output)
        bag._emb_module_forward = emb_forward
        features = cast(KeyedJaggedTensor, object())

        with patch.object(
            qek, "_unwrap_kjt_lengths", return_value=(indices, lengths, None)
        ) as mock_lengths_unwrap, patch.object(
            qek, "_unwrap_ro_kjt", return_value=(indices, lengths, None)
        ) as mock_ro_unwrap:
            result = bag.forward(features)

        self.assertIs(result, output)
        mock_lengths_unwrap.assert_called_once_with(features)
        mock_ro_unwrap.assert_not_called()
        emb_forward.assert_called_once_with(indices, lengths, None)

    def test_device_ro_fused_param_is_internal_to_torchrec(self) -> None:
        fused_params: Dict[str, Any] = {
            FUSED_PARAM_IS_DEVICE_RO: True,
            "output_dtype": object(),
        }

        self.assertTrue(is_fused_param_device_ro(fused_params))
        self.assertFalse(is_fused_param_device_ro({}))
        filtered_params = tbe_fused_params(fused_params)
        self.assertIsNotNone(filtered_params)
        self.assertNotIn(FUSED_PARAM_IS_DEVICE_RO, filtered_params)
        self.assertIn("output_dtype", filtered_params)
