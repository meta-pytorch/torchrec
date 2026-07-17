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
    is_fused_param_device_ro,
    tbe_fused_params,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class QuantBatchedEmbeddingBagDeviceRoTest(unittest.TestCase):
    def _make_bag(
        self,
        *,
        is_device_ro: bool,
        lengths_to_tbe: bool = False,
    ) -> qek.QuantBatchedEmbeddingBag:
        bag = qek.QuantBatchedEmbeddingBag.__new__(qek.QuantBatchedEmbeddingBag)
        bag._runtime_device = torch.device("cuda")
        bag.lengths_to_tbe = lengths_to_tbe
        bag._is_device_ro = is_device_ro
        return bag

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
