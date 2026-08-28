#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional
from unittest import TestCase
from unittest.mock import patch

import torch
from parameterized import parameterized
from torchrec.distributed.batched_embedding_kernel import BatchedTPUEmbeddingBag
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.experimental.torch_tpu.modules.embedding_modules import (
    PooledLookupKernel,
    TPUEmbeddingBagUnfused,
)
from torchrec.experimental.torch_tpu.pallas import ops  # noqa: F401
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE = 8

TABLES: List[EmbeddingBagConfig] = [
    EmbeddingBagConfig(
        name="user_table",
        embedding_dim=32,
        num_embeddings=100000,
        feature_names=["user"],
        pooling=PoolingType.SUM,
    ),
    EmbeddingBagConfig(
        name="item_table",
        embedding_dim=32,
        num_embeddings=50000,
        feature_names=["item"],
        pooling=PoolingType.SUM,
    ),
]

# Exercised batch / pooling grid — mirrors single_pooled_lookup.py's
# `if __name__ == "__main__"` sweep (BATCH_SIZES x {"sum","mean"}).
BATCH_SIZES = [1, 8, 64, 512]

DEVICE_KERNEL_CASES = [
    (f"{device}_{kernel.value}", device, kernel)
    for device in ("cpu", "tpu")
    for kernel in PooledLookupKernel
]


def _tables_for_pooling(pooling: PoolingType) -> List[EmbeddingBagConfig]:
    """Clone TABLES with the requested pooling (EBC reference depends on it)."""
    return [
        EmbeddingBagConfig(
            name=t.name,
            embedding_dim=t.embedding_dim,
            num_embeddings=t.num_embeddings,
            feature_names=t.feature_names,
            pooling=pooling,
        )
        for t in TABLES
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
    compute_kernel: EmbeddingComputeKernel,
    tables: List[EmbeddingBagConfig],
    pooling: PoolingType,
    fused_params: Optional[Dict[str, Any]] = None,
) -> GroupedEmbeddingConfig:
    sharded = [
        ShardedEmbeddingTable(
            name=table.name,
            num_embeddings=table.num_embeddings,
            embedding_dim=table.embedding_dim,
            data_type=DataType.FP32,
            feature_names=table.feature_names,
            pooling=pooling,
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
        pooling=pooling,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=compute_kernel,
        embedding_tables=sharded,
        fused_params=fused_params,
    )


class EBCLookupTest(TestCase):
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

    def test_rejects_unaligned_embedding_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            TPUEmbeddingBagUnfused(
                num_embeddings=10,
                embedding_dim=15,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_parameter_names_match_state_dict(self) -> None:
        config = _grouped_config(
            EmbeddingComputeKernel.UNFUSED_TPU, TABLES, PoolingType.SUM
        )
        kernel = BatchedTPUEmbeddingBag(config=config, device=torch.device("cpu"))
        expected_names = [f"{table.name}.weight" for table in TABLES]

        self.assertEqual(expected_names, list(dict(kernel.named_parameters())))
        self.assertEqual(expected_names, list(kernel.state_dict()))

    @parameterized.expand([("cpu",), ("tpu",)])
    def test_dispatch(self, device: str) -> None:
        """`GroupedPooledEmbeddingsLookup` builds a `BatchedTPUEmbeddingBag` for UNFUSED_TPU."""
        self._setup_device(device)
        with patch(
            "torchrec.distributed.embedding_lookup.dist.get_world_size",
            return_value=1,
        ):
            lookup = GroupedPooledEmbeddingsLookup(
                grouped_configs=[
                    _grouped_config(
                        EmbeddingComputeKernel.UNFUSED_TPU,
                        TABLES,
                        PoolingType.SUM,
                        fused_params={
                            "pooled_lookup_kernel": PooledLookupKernel.BATCHED_OFFSET
                        },
                    )
                ],
                device=torch.device(device),
            )
        kernel = lookup._emb_modules[0]
        self.assertIsInstance(kernel, BatchedTPUEmbeddingBag)
        self.assertEqual(kernel.pooled_lookup_kernel, PooledLookupKernel.BATCHED_OFFSET)
        self.assertTrue(
            all(weight.requires_grad for weight in kernel.split_embedding_weights())
        )

    @parameterized.expand(DEVICE_KERNEL_CASES)
    def test_forward_matches_embedding_bag_collection(
        self, _case_name: str, device: str, pooled_lookup_kernel: PooledLookupKernel
    ) -> None:
        """Pooled forward matches `EmbeddingBagCollection` across both poolings and batch sizes.

        Mirrors `single_pooled_lookup.py`'s main correctness sweep (sum/mean x
        BATCH_SIZES, fwd `F.embedding_bag` reference, bwd scatter check) but
        exercised through the TorchRec pooled stack:
        `GroupedPooledEmbeddingsLookup` / `BatchedTPUEmbeddingBag`. Each case
        shares identical weights between the CPU `EmbeddingBagCollection`
        reference (one `EmbeddingBagConfig` per feature, pooled per-batch) and
        the kernel, then checks the pooled `[B, sum(dim)]` output.
        """
        self._setup_device(device)
        for pooling in (PoolingType.SUM, PoolingType.MEAN):
            tables_pooled = _tables_for_pooling(pooling)
            config = _grouped_config(
                EmbeddingComputeKernel.UNFUSED_TPU, tables_pooled, pooling
            )
            # Reference pooled collection — one bag per feature, pooled per sample.
            ref_ebc = EmbeddingBagCollection(
                tables=tables_pooled, device=torch.device("cpu")
            )
            kernel = BatchedTPUEmbeddingBag(
                config=config,
                device=torch.device(device),
                pooled_lookup_kernel=pooled_lookup_kernel,
            )

            # split_embedding_weights() returns one weight per table, in table order.
            for weight, table in zip(kernel.split_embedding_weights(), tables_pooled):
                weight.data.copy_(  # pyre-ignore[16]
                    ref_ebc.embedding_bags[table.name].weight.data  # pyre-ignore[16]
                )

            for batch_size in BATCH_SIZES:
                # Generate KJT with jagged bags (pooling_avg controls avg bag length).
                # Use a deterministic seed per (pooling, batch) so cpu/tpu cases compare
                # the same ids/offsets.
                model_input, _ = ModelInput.generate(
                    batch_size=batch_size,
                    world_size=WORLD_SIZE,
                    num_float_features=0,
                    tables=tables_pooled,
                    weighted_tables=[],
                    pooling_avg=5,
                    random_seed=100
                    + batch_size
                    + (10 if pooling == PoolingType.MEAN else 0),
                )
                kjt = (
                    model_input.idlist_features
                )  # int64 indices on cpu, keys in table order
                assert isinstance(kjt, KeyedJaggedTensor)
                # The caller owns device placement; the module only normalizes dtype.
                kjt_device = KeyedJaggedTensor(
                    keys=kjt.keys(),  # pyre-ignore[16]
                    values=kjt.values().to(
                        device=device, dtype=torch.int32
                    ),  # pyre-ignore[16]
                    lengths=kjt.lengths().to(device=device),  # pyre-ignore[16]
                )

                ref_kt = ref_ebc(kjt)  # KeyedTensor: ref_kt[key] -> [B, dim]
                # Pooled features concatenate along the embedding dim -> [B, sum(dim)].
                expected = torch.cat(
                    [ref_kt[key] for key in kjt.keys()], dim=1  # pyre-ignore[16]
                )
                actual = kernel(kjt_device).to("cpu")

                self.assertEqual(
                    expected.shape,
                    actual.shape,
                    f"pooling={pooling} B={batch_size}: shape {expected.shape} != {actual.shape}",
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-4,
                    rtol=1e-4,
                    msg=lambda m, p=pooling, b=batch_size: f"pooling={p} B={b}: {m}",
                )

    @parameterized.expand(DEVICE_KERNEL_CASES)
    def test_backward_populates_grads(
        self, _case_name: str, device: str, pooled_lookup_kernel: PooledLookupKernel
    ) -> None:
        """`loss.backward()` populates every table's gradient for each pooling."""
        self._setup_device(device)
        for pooling in (PoolingType.SUM, PoolingType.MEAN):
            config = _grouped_config(
                EmbeddingComputeKernel.UNFUSED_TPU, TABLES, pooling
            )
            kernel = BatchedTPUEmbeddingBag(
                config=config,
                device=torch.device(device),
                pooled_lookup_kernel=pooled_lookup_kernel,
            )

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
            assert isinstance(kjt, KeyedJaggedTensor)
            kjt_device = KeyedJaggedTensor(
                keys=kjt.keys(),  # pyre-ignore[16]
                values=kjt.values().to(
                    device=device, dtype=torch.int32
                ),  # pyre-ignore[16]
                lengths=kjt.lengths().to(device=device),  # pyre-ignore[16]
            )

            # Zero grads from previous pooling iteration if reused.
            for w in kernel.split_embedding_weights():
                w.grad = None
            kernel(kjt_device).float().sum().backward()

            for weight, table in zip(kernel.split_embedding_weights(), TABLES):
                self.assertIsNotNone(
                    weight.grad,
                    f"pooling={pooling}: no gradient for table {table.name}",
                )
                self.assertGreater(
                    torch.count_nonzero(weight.grad).item(),
                    0,
                    f"pooling={pooling}: all-zero gradient for table {table.name}",
                )
