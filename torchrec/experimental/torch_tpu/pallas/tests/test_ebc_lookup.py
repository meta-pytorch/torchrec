#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, cast, Dict, List, Optional
from unittest import TestCase
from unittest.mock import patch

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
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
    if kernel != PooledLookupKernel.BATCHED_OFFSET_FUSED_ADAGRAD
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
    from torchrec.experimental.torch_tpu.pallas import dispatcher  # noqa: F401


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


def _cpu_vbe_lookup(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
    row_offsets: torch.Tensor,
    emb_dim: int,
    row_output_offsets: torch.Tensor,
) -> torch.Tensor:
    """CPU stand-in for the Pallas VBE forward kernel."""
    feature_major = ops.embedding_pooled_batched_lookup_offset_cpu(
        indices, offsets, weights, row_offsets, emb_dim
    )
    output_rows = torch.div(row_output_offsets, emb_dim, rounding_mode="floor").long()
    return torch.zeros_like(feature_major).index_copy(0, output_rows, feature_major)


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

    @parameterized.expand(
        [
            (
                "explicit",
                PooledLookupKernel.BATCHED_OFFSET,
                {
                    "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD,
                    "learning_rate": 0.2,
                    "eps": 0.01,
                },
                0.01,
            ),
            (
                "legacy",
                PooledLookupKernel.BATCHED_OFFSET_FUSED_ADAGRAD,
                {"lr": 0.2},
                1e-8,
            ),
        ]
    )
    def test_fused_adagrad_config(
        self,
        _name: str,
        pooled_lookup_kernel: PooledLookupKernel,
        fused_params: Dict[str, Any],
        expected_eps: float,
    ) -> None:
        table = EmbeddingBagConfig(
            name="table",
            embedding_dim=16,
            num_embeddings=4,
            feature_names=["feature"],
        )
        kernel = BatchedTPUEmbeddingBag(
            config=_grouped_config(
                EmbeddingComputeKernel.UNFUSED_TPU,
                [table],
                PoolingType.SUM,
                fused_params=fused_params,
            ),
            device=torch.device("cpu"),
            pooled_lookup_kernel=pooled_lookup_kernel,
        )
        self.assertEqual(kernel.emb_module.optimizer, OptimType.EXACT_ROWWISE_ADAGRAD)
        self.assertEqual(kernel.emb_module.learning_rate, 0.2)
        self.assertEqual(kernel.emb_module.eps, expected_eps)
        kernel(
            KeyedJaggedTensor(
                keys=["feature"],
                values=torch.tensor([1, 1, 2], dtype=torch.int32),
                lengths=torch.tensor([2, 1], dtype=torch.int32),
            )
        ).sum().backward()
        self.assertIsNone(kernel.emb_module.weights.grad)
        torch.testing.assert_close(
            kernel.emb_module.momentum, torch.tensor([0.0, 4.0, 1.0, 0.0])
        )

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

    @parameterized.expand(
        [
            (
                "unfused",
                PooledLookupKernel.BATCHED_OFFSET,
                "embedding_pooled_batched_lookup_offset",
            ),
            (
                "fused_adagrad",
                PooledLookupKernel.BATCHED_OFFSET_FUSED_ADAGRAD,
                "embedding_pooled_batched_lookup_offset_fused_adagrad",
            ),
        ]
    )
    def test_vbe_flat_output_merges_and_backpropagates(
        self, _name: str, kernel: PooledLookupKernel, op_name: str
    ) -> None:
        tables = [
            EmbeddingBagConfig(
                name=f"table_{t}",
                embedding_dim=dim,
                num_embeddings=8,
                feature_names=[f"feature_{t}"],
            )
            for t, dim in enumerate((16, 32))
        ]
        with patch(
            "torchrec.distributed.embedding_lookup.dist.get_world_size",
            return_value=2,
        ):
            lookup = GroupedPooledEmbeddingsLookup(
                grouped_configs=[
                    _grouped_config(
                        EmbeddingComputeKernel.UNFUSED_TPU,
                        [table],
                        PoolingType.SUM,
                        fused_params={
                            "pooled_lookup_kernel": kernel,
                            "learning_rate": 0.1,
                        },
                    )
                    for table in tables
                ],
                device=torch.device("cpu"),
            )
        features = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.arange(6, dtype=torch.int32),
            lengths=torch.ones(6, dtype=torch.int32),
            stride_per_key_per_rank=[[2, 1], [1, 2]],
        )
        # CPU VBE kernels are unsupported; supply the Pallas kernel's 2D output.
        pooled = [
            torch.arange(3 * dim, dtype=torch.float32).reshape(3, dim).requires_grad_()
            for dim in (16, 32)
        ]
        with patch.object(torch.ops.torchrec, op_name, side_effect=pooled) as op:
            actual = lookup(features)
        self.assertEqual(op.call_count, 2)
        self.assertEqual(actual.shape, torch.Size([144]))
        expected = torch.cat(
            [
                pooled[0][:2].flatten(),
                pooled[1][:1].flatten(),
                pooled[0][2:].flatten(),
                pooled[1][1:].flatten(),
            ]
        )
        torch.testing.assert_close(actual, expected)

        grad = torch.arange(actual.numel(), dtype=torch.float32)
        actual.backward(grad)
        torch.testing.assert_close(
            pooled[0].grad, torch.cat([grad[:32], grad[64:80]]).reshape(3, 16)
        )
        torch.testing.assert_close(
            pooled[1].grad, torch.cat([grad[32:64], grad[80:]]).reshape(3, 32)
        )

    @parameterized.expand(
        [
            (
                "unfused",
                PooledLookupKernel.BATCHED_OFFSET,
                "embedding_pooled_batched_lookup_offset",
            ),
            (
                "fused_adagrad",
                PooledLookupKernel.BATCHED_OFFSET_FUSED_ADAGRAD,
                "embedding_pooled_batched_lookup_offset_fused_adagrad",
            ),
        ]
    )
    def test_vbe_mean_uses_rank_major_lengths(
        self, _name: str, pooled_lookup_kernel: PooledLookupKernel, op_name: str
    ) -> None:
        tables = [
            EmbeddingBagConfig(
                name=f"table_{t}",
                embedding_dim=16,
                num_embeddings=8,
                feature_names=[f"feature_{t}"],
                pooling=PoolingType.MEAN,
            )
            for t in range(2)
        ]
        kernel = BatchedTPUEmbeddingBag(
            config=_grouped_config(
                EmbeddingComputeKernel.UNFUSED_TPU,
                tables,
                PoolingType.MEAN,
                fused_params={"learning_rate": 0.1},
            ),
            device=torch.device("cpu"),
            pooled_lookup_kernel=pooled_lookup_kernel,
        )
        features = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.zeros(15, dtype=torch.int32),
            lengths=torch.tensor([0, 2, 3, 4, 1, 5], dtype=torch.int32),
            stride_per_key_per_rank=[[2, 1], [1, 2]],
        )
        pooled = (
            torch.tensor([0, 4, 20, 12, 6, 35], dtype=torch.float32)[:, None]
            .repeat(1, 16)
            .requires_grad_()
        )
        with (
            patch.object(
                ops,
                "generate_vbe_row_output_offsets",
                wraps=ops.generate_vbe_row_output_offsets,
            ) as generate_offsets,
            patch.object(torch.ops.torchrec, op_name, return_value=pooled) as op,
        ):
            actual = kernel(features)
        generate_offsets.assert_called_once()
        assert op.call_args is not None
        torch.testing.assert_close(
            op.call_args.kwargs["row_output_offsets"],
            torch.tensor([0, 16, 48, 32, 64, 80], dtype=torch.int64),
        )
        expected = torch.tensor([0, 2, 5, 4, 6, 7], dtype=torch.float32)[:, None]
        torch.testing.assert_close(actual, expected.expand(6, 16).reshape(-1))

        actual.sum().backward()
        rank_major_lengths = torch.tensor([1, 2, 4, 3, 1, 5], dtype=torch.float32)
        torch.testing.assert_close(
            pooled.grad, rank_major_lengths.reciprocal()[:, None].expand(6, 16)
        )

    @parameterized.expand(
        [(device, groups) for device in ("cpu", "tpu") for groups in (1, 2)]
    )
    def test_vbe_rank_major_output_and_grad(self, device: str, num_groups: int) -> None:
        self._setup_device(device)
        tables = [
            EmbeddingBagConfig(
                name=f"table_{t}",
                embedding_dim=16 if num_groups == 1 or t == 0 else 32,
                num_embeddings=16,
                feature_names=[f"feature_{t}"],
                pooling=PoolingType.SUM,
            )
            for t in range(2)
        ]
        # Citrine C3: create input and expected tensors on the lookup device.
        features = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device=device),
            lengths=torch.ones(6, dtype=torch.int32, device=device),
            stride_per_key_per_rank=[[2, 1], [1, 2]],
        )
        second_dim = tables[1].embedding_dim
        expected = torch.tensor(
            [11, 12, 24, 13, 25, 26], dtype=torch.float32, device=device
        ).repeat_interleave(
            torch.tensor(
                [16, 16, second_dim, 16, second_dim, second_dim],
                dtype=torch.int32,
                device=device,
            )
        )

        groups = [tables] if num_groups == 1 else [[table] for table in tables]
        with patch(
            "torchrec.distributed.embedding_lookup.dist.get_world_size",
            return_value=2,
        ):
            lookup = GroupedPooledEmbeddingsLookup(
                grouped_configs=[
                    _grouped_config(
                        EmbeddingComputeKernel.UNFUSED_TPU,
                        group,
                        PoolingType.SUM,
                        fused_params={
                            "pooled_lookup_kernel": PooledLookupKernel.BATCHED_OFFSET
                        },
                    )
                    for group in groups
                ],
                device=torch.device(device),
            )
        tpu_modules = [cast(BatchedTPUEmbeddingBag, m) for m in lookup._emb_modules]
        with torch.no_grad():
            weights = [
                weight
                for emb_module in tpu_modules
                for weight in emb_module.split_embedding_weights()
            ]
            for table_idx, weight in enumerate(weights):
                weight.copy_(
                    torch.arange(16, dtype=torch.float32, device=device)[:, None]
                    + 10 * (table_idx + 1)
                )

        if device == "cpu":
            with patch.object(
                torch.ops.torchrec,
                "embedding_pooled_batched_lookup_offset",
                side_effect=_cpu_vbe_lookup,
            ) as op:
                actual = lookup(features)
            self.assertEqual(op.call_count, num_groups)
        else:
            actual = lookup(features)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected)

        # FBGEMM's CPU sparse permute has no backward; TPU dispatch uses gather.
        if device == "tpu":
            actual.sum().backward()
            grads = [
                weight.grad
                for emb_module in tpu_modules
                for weight in emb_module.split_embedding_weights()
            ]
            for table_idx, grad in enumerate(grads):
                self.assertIsNotNone(grad)
                assert grad is not None
                row_start = 1 if table_idx == 0 else 4
                torch.testing.assert_close(
                    grad[row_start : row_start + 3],
                    torch.ones((3, tables[table_idx].embedding_dim), device=device),
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
