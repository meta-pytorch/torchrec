#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from parameterized import parameterized
from torchrec.distributed.batched_embedding_kernel import BatchedTPUEmbedding
from torchrec.distributed.embedding_lookup import GroupedEmbeddingsLookup
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.experimental.torch_tpu.pallas import ops  # noqa: F401
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE = 8

TABLES: List[EmbeddingConfig] = [
    EmbeddingConfig(
        name="user_table",
        embedding_dim=32,
        num_embeddings=100000,
        feature_names=["user"],
    ),
    EmbeddingConfig(
        name="item_table",
        embedding_dim=32,
        num_embeddings=50000,
        feature_names=["item"],
    ),
]


def tpu_is_available() -> bool:
    """Whether a TPU backend is importable and has a device attached."""
    try:
        import torch_tpu  # pyre-ignore[21]  # noqa: F401
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


def _register_tpu_kernels() -> None:
    """Bind the Pallas kernels to the TPU dispatch key.

    Imported lazily: `impl` pulls in `torch_tpu` and `jax`, which are only
    present on a TPU host, so importing it at module scope would break the CPU
    tests.
    """
    from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401


def _grouped_config(
    compute_kernel: EmbeddingComputeKernel, tables: List[EmbeddingConfig]
) -> GroupedEmbeddingConfig:
    sharded = [
        ShardedEmbeddingTable(
            name=table.name,
            num_embeddings=table.num_embeddings,
            embedding_dim=table.embedding_dim,
            data_type=DataType.FP32,
            feature_names=table.feature_names,
            pooling=PoolingType.NONE,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=compute_kernel,
            local_rows=table.num_embeddings,
            local_cols=table.embedding_dim,
        )
        for table in tables
    ]
    return GroupedEmbeddingConfig(
        data_type=DataType.FP32,
        pooling=PoolingType.NONE,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=compute_kernel,
        embedding_tables=sharded,
    )


class ECLookupTest(unittest.TestCase):
    def _setup_device(self, device: str) -> None:
        """Skip the TPU cases off-hardware; bind the Pallas kernels on it.

        The check runs here rather than in a `skipIf` decorator so that
        `torch_tpu` is only imported when a TPU case actually executes.
        """
        if device != "tpu":
            return
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        _register_tpu_kernels()

    @parameterized.expand([("cpu",), ("tpu",)])
    def test_dispatch(self, device: str) -> None:
        """`GroupedEmbeddingsLookup` builds a `BatchedTPUEmbedding` for UNFUSED_TPU."""
        self._setup_device(device)
        lookup = GroupedEmbeddingsLookup(
            grouped_configs=[
                _grouped_config(EmbeddingComputeKernel.UNFUSED_TPU, TABLES)
            ],
            device=torch.device(device),
        )
        self.assertIsInstance(lookup._emb_modules[0], BatchedTPUEmbedding)

    @parameterized.expand([("cpu",), ("tpu",)])
    def test_forward_matches_embedding_collection(self, device: str) -> None:
        self._setup_device(device)
        config = _grouped_config(EmbeddingComputeKernel.UNFUSED_TPU, TABLES)
        ref_ec = EmbeddingCollection(tables=TABLES, device=torch.device("cpu"))
        kernel = BatchedTPUEmbedding(config=config, device=torch.device(device))

        # split_embedding_weights() returns one weight per table, in table order.
        for weight, table in zip(kernel.split_embedding_weights(), TABLES):
            weight.data.copy_(  # pyre-ignore[16]
                ref_ec.embeddings[table.name].weight.data  # pyre-ignore[16]
            )

        model_input, _ = ModelInput.generate(
            batch_size=2048,
            world_size=WORLD_SIZE,
            num_float_features=0,
            tables=TABLES,
            weighted_tables=[],
            pooling_avg=5,
            random_seed=100,
        )
        kjt = model_input.idlist_features  # int64 indices on cpu, keys in table order
        kjt_device = KeyedJaggedTensor(
            keys=kjt.keys(),  # pyre-ignore[16]
            values=kjt.values().to(device=device, dtype=torch.int32),  # pyre-ignore[16]
            lengths=kjt.lengths(),  # pyre-ignore[16]
        )

        ref_jt = ref_ec(kjt)
        expected = torch.cat(
            [ref_jt[key].values() for key in kjt.keys()], dim=0  # pyre-ignore[16]
        )
        actual = kernel(kjt_device).to("cpu")

        self.assertEqual(expected.shape, actual.shape)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @parameterized.expand([("cpu",), ("tpu",)])
    def test_backward_populates_grads(self, device: str) -> None:
        """`loss.backward()` populates every table's gradient."""
        self._setup_device(device)
        config = _grouped_config(EmbeddingComputeKernel.UNFUSED_TPU, TABLES)
        kernel = BatchedTPUEmbedding(config=config, device=torch.device(device))

        model_input, _ = ModelInput.generate(
            batch_size=128,
            world_size=WORLD_SIZE,
            num_float_features=0,
            tables=TABLES,
            weighted_tables=[],
            pooling_avg=5,
            random_seed=100,
        )
        kjt = model_input.idlist_features
        kjt_device = KeyedJaggedTensor(
            keys=kjt.keys(),  # pyre-ignore[16]
            values=kjt.values().to(device=device, dtype=torch.int32),  # pyre-ignore[16]
            lengths=kjt.lengths(),  # pyre-ignore[16]
        )

        kernel(kjt_device).float().sum().backward()

        for weight, table in zip(kernel.split_embedding_weights(), TABLES):
            self.assertIsNotNone(weight.grad, f"no gradient for table {table.name}")
            self.assertGreater(
                torch.count_nonzero(weight.grad).item(),
                0,
                f"all-zero gradient for table {table.name}",
            )


if __name__ == "__main__":
    unittest.main()
