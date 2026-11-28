#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any, cast, Dict
from unittest.mock import MagicMock, patch

import torch
import torchrec.distributed.quant_embedding_kernel as qek
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_ENABLE_FS_2D_ASYNC_ALLGATHER,
    fused_param_enable_fs_2d_async_allgather,
    FUSED_PARAM_IS_DEVICE_RO,
    is_fused_param_device_ro,
    tbe_fused_params,
)
from torchrec.modules.embedding_configs import DataType, PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

try:
    from fbgemm_gpu.tbe.cache.kv_zch_gpu_bucket_wise_embedding import (
        KVZCHGPUBucketWiseEmbeddingInference,
    )

    HAS_KV_ZCH_GPU_BUCKET_WISE = True
except ImportError:
    # Released fbgemm_gpu builds may predate this TBE.
    HAS_KV_ZCH_GPU_BUCKET_WISE = False


class QuantizeWeightTest(unittest.TestCase):
    def test_unsupported_dtype_error_names_the_dtype(self) -> None:
        state_dict = {"weight": torch.zeros(4, 8, dtype=torch.bfloat16)}

        with self.assertRaisesRegex(Exception, r"Unsupported dtype: torch\.bfloat16"):
            qek._quantize_weight(state_dict, DataType.INT8)


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
        bag._is_virtual_table = False
        bag._is_kv_zch_gpu_bucket_wise = False
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

    def test_fs_2d_async_allgather_fused_param_is_internal_to_torchrec(self) -> None:
        fused_params: Dict[str, Any] = {
            FUSED_PARAM_ENABLE_FS_2D_ASYNC_ALLGATHER: True,
            "output_dtype": object(),
        }

        self.assertFalse(fused_param_enable_fs_2d_async_allgather(None))
        self.assertFalse(fused_param_enable_fs_2d_async_allgather({}))
        self.assertFalse(
            fused_param_enable_fs_2d_async_allgather(
                {FUSED_PARAM_ENABLE_FS_2D_ASYNC_ALLGATHER: False}
            )
        )
        self.assertFalse(
            fused_param_enable_fs_2d_async_allgather(
                {FUSED_PARAM_ENABLE_FS_2D_ASYNC_ALLGATHER: "shared_arch"}
            )
        )
        self.assertFalse(
            fused_param_enable_fs_2d_async_allgather(
                {FUSED_PARAM_ENABLE_FS_2D_ASYNC_ALLGATHER: 1}
            )
        )
        self.assertTrue(fused_param_enable_fs_2d_async_allgather(fused_params))
        filtered_params = tbe_fused_params(fused_params)
        self.assertIsNotNone(filtered_params)
        self.assertNotIn(FUSED_PARAM_ENABLE_FS_2D_ASYNC_ALLGATHER, filtered_params)
        self.assertIn("output_dtype", filtered_params)


class KVZCHVirtualTableUnwrapTest(unittest.TestCase):
    """KV-ZCH virtual table ids span int64 and must reach the KV TBE unchanged."""

    def _kjt(self) -> KeyedJaggedTensor:
        return KeyedJaggedTensor(
            keys=["cu"],
            values=torch.tensor([5, 2**33 + 7, 562949951477760 + 3], dtype=torch.int64),
            lengths=torch.tensor([1, 2], dtype=torch.int64),
        )

    def test_only_the_kv_unwrap_keeps_int64_ids(self) -> None:
        kjt = self._kjt()
        indices, offsets, _ = qek._unwrap_kv_kjt(kjt)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(offsets.dtype, torch.int64)
        self.assertTrue(torch.equal(indices, kjt.values()))
        indices, offsets, _ = qek._unwrap_kjt(kjt)
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(offsets.dtype, torch.int32)

    def test_cuda_sequence_forward_passes_virtual_table_ids_unchanged(self) -> None:
        emb = qek.QuantBatchedEmbedding.__new__(qek.QuantBatchedEmbedding)
        emb._runtime_device = torch.device("cuda")
        emb._is_virtual_table = True
        emb._is_kv_zch_gpu_bucket_wise = True
        emb._emb_module_registered = True
        tbe = MagicMock(return_value=torch.tensor([1.0]))
        emb._emb_module = tbe
        kjt = self._kjt()

        emb.forward(kjt)

        sent = tbe.call_args.kwargs["indices"]
        self.assertEqual(sent.dtype, torch.int64)
        self.assertTrue(torch.equal(sent, kjt.values()))

    def test_undeclared_cuda_virtual_table_keeps_trunk_unwrap(self) -> None:
        emb = qek.QuantBatchedEmbedding.__new__(qek.QuantBatchedEmbedding)
        emb._runtime_device = torch.device("cuda")
        emb._is_virtual_table = True
        emb._is_kv_zch_gpu_bucket_wise = False
        emb._emb_module_registered = True
        tbe = MagicMock(return_value=torch.tensor([1.0]))
        emb._emb_module = tbe
        emb.forward(self._kjt())
        self.assertEqual(tbe.call_args.kwargs["indices"].dtype, torch.int32)

    def test_declared_cpu_virtual_table_uses_the_kv_unwrap(self) -> None:
        emb = qek.QuantBatchedEmbedding.__new__(qek.QuantBatchedEmbedding)
        emb._runtime_device = torch.device("cpu")
        emb._is_virtual_table = True
        emb._is_kv_zch_gpu_bucket_wise = True
        emb._emb_module_registered = True
        tbe = MagicMock(return_value=torch.tensor([1.0]))
        emb._emb_module = tbe
        kjt = self._kjt()
        emb.forward(kjt)
        self.assertTrue(torch.equal(tbe.call_args.kwargs["indices"], kjt.values()))

    @unittest.skipIf(
        not HAS_KV_ZCH_GPU_BUCKET_WISE,
        "fbgemm_gpu has no KVZCHGPUBucketWiseEmbeddingInference",
    )
    def test_tbe_class_selection_uses_the_table_declaration(self) -> None:
        def config(declared: bool, virtual: bool = True) -> GroupedEmbeddingConfig:
            shard = ShardMetadata(
                shard_offsets=[0, 0], shard_sizes=[16, 8], placement="rank:0/cpu"
            )
            table = ShardedEmbeddingTable(
                num_embeddings=16,
                embedding_dim=8,
                name="cu",
                feature_names=["cu"],
                data_type=DataType.INT8,
                pooling=PoolingType.NONE,
                compute_kernel=EmbeddingComputeKernel.QUANT,
                local_rows=16,
                local_cols=8,
                local_metadata=shard,
                global_metadata=ShardedTensorMetadata([shard], torch.Size([16, 8])),
                use_virtual_table=virtual,
                kv_zch_gpu_bucket_wise=declared,
            )
            return GroupedEmbeddingConfig(
                data_type=DataType.INT8,
                pooling=PoolingType.NONE,
                is_weighted=False,
                has_feature_processor=False,
                compute_kernel=EmbeddingComputeKernel.QUANT,
                embedding_tables=[table],
            )

        ordinary = qek.QuantBatchedEmbedding(
            config(False), device=torch.device("cpu"), shard_index=0
        )
        declared = qek.QuantBatchedEmbedding(
            config(True), device=torch.device("cpu"), shard_index=0
        )
        self.assertIs(type(ordinary.emb_module), qek.KVEmbeddingInference)
        self.assertIs(type(declared.emb_module), KVZCHGPUBucketWiseEmbeddingInference)
        with self.assertRaisesRegex(
            ValueError, "must be declared on every virtual table"
        ):
            qek.QuantBatchedEmbedding(
                config(True, virtual=False), device=torch.device("cpu"), shard_index=0
            )
