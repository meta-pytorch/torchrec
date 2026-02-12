#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Optional

import torch
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
    EmbeddingBagCollectionSharder,
    EmbeddingCollectionSharder,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.models.dlrm import DLRM_DCN
from torchrec.modules.debug_embedding_modules import (
    DebugEmbeddingBagCollection,
    DebugEmbeddingCollection,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


torch.autograd.detect_anomaly(check_nan=True)


class TestDebugEmbedding(MultiProcessTestBase):
    """
    Test to check for anomaly such as NaN in gradients.

    Works for Normal embeddings (not virtual tables). This is because we have
    grads for Normal embeddings (not virtual tables).
    """

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs",
    )
    def test_embedding(
        self,
    ) -> None:
        WORLD_SIZE = 2
        # we do not use virtual tables because they do not have grads (use_virtual_table)
        # ec_tables is to test Debug Embedding Collection model
        tables = [
            EmbeddingConfig(
                num_embeddings=8000,
                embedding_dim=64,
                name="table_0",
                feature_names=["feature_0", "feature_1"],
                total_num_buckets=20,
            ),
            EmbeddingConfig(
                num_embeddings=8000,
                embedding_dim=64,
                name="table_1",
                feature_names=["feature_2"],
                total_num_buckets=20,
            ),
            EmbeddingConfig(
                num_embeddings=8000,
                embedding_dim=64,
                name="table_2",
                feature_names=["feature_3"],
            ),
        ]
        backend = "nccl"
        # Generate unique inputs for all 8 ranks with different random seeds
        torch.manual_seed(42)  # Set seed for reproducibility
        inputs_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2", "feature_3"],
                values=torch.randint(
                    0, 8000, (14,), generator=torch.Generator().manual_seed(100 + i)
                ),
                lengths=torch.LongTensor([2, 1, 2, 1, 1, 1, 2, 0, 1, 1, 2, 0]),
            )
            for i in range(WORLD_SIZE)
        ]
        self._run_multi_process_test(
            callable=run_embedding_collection,
            world_size=WORLD_SIZE,
            tables=tables,
            backend=backend,
            inputs_per_rank=inputs_per_rank,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs",
    )
    def test_embedding_bag(
        self,
    ) -> None:
        WORLD_SIZE = 2
        # ebc_tables is to test Debug Embedding Bag Collection model
        tables = [
            EmbeddingBagConfig(
                name="table_0",
                embedding_dim=64,
                num_embeddings=8000,
                feature_names=["feature_0", "feature_1"],
                pooling=PoolingType.SUM,
            ),
            EmbeddingBagConfig(
                name="table_1",
                embedding_dim=64,
                num_embeddings=8000,
                feature_names=["feature_1", "feature_2"],
                pooling=PoolingType.SUM,
            ),
            EmbeddingBagConfig(
                name="table_2",
                embedding_dim=64,
                num_embeddings=8000,
                feature_names=["feature_2", "feature_3"],
                pooling=PoolingType.SUM,
            ),
        ]
        backend = "nccl"
        torch.manual_seed(42)  # Set seed for reproducibility
        inputs_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2", "feature_3"],
                values=torch.randint(
                    0, 8000, (14,), generator=torch.Generator().manual_seed(100 + i)
                ),
                lengths=torch.LongTensor([2, 1, 2, 1, 1, 1, 2, 0, 1, 1, 2, 0]),
            )
            for i in range(WORLD_SIZE)
        ]
        self._run_multi_process_test(
            callable=run_embedding_bag_collection,
            world_size=WORLD_SIZE,
            tables=tables,
            backend=backend,
            inputs_per_rank=inputs_per_rank,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs",
    )
    def test_model(
        self,
    ) -> None:
        WORLD_SIZE = 2
        backend = "nccl"
        self._run_multi_process_test(
            callable=run_debug_model,
            world_size=WORLD_SIZE,
            backend=backend,
        )


def run_debug_model(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """In progress"""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        B, D = 2, 8
        set_propogate_device(True)

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = DebugEmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            device=ctx.device,
            debug_mode=True,
        )
        model = DLRM_DCN(
            # pyrefly: ignore[bad-argument-type]
            embedding_bag_collection=ebc,
            dense_in_features=100,
            dense_arch_layer_sizes=[20, D],
            dcn_num_layers=2,
            dcn_low_rank_dim=8,
            over_arch_layer_sizes=[5, 1],
            dense_device=ctx.device,
        ).to(ctx.device)

        def insert_nan_grad(grad) -> torch.Tensor:
            """Hook to insert nan into the gradient"""
            return torch.full_like(grad, float("nan"))

        features = torch.rand((B, 100), device=ctx.device)

        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 7, 8, 6]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 9, 11]),
        ).to(ctx.device)

        logits = model(
            dense_features=features,
            sparse_features=sparse_features,
        )
        logits.register_hook(insert_nan_grad)
        loss = torch.sum(logits)

        tc = unittest.TestCase()
        with torch.autograd.detect_anomaly():
            with tc.assertRaisesRegex(RuntimeError, "returned nan values"):
                loss.backward()


def run_embedding_collection(
    rank: int,
    world_size: int,
    tables: List[EmbeddingConfig],
    backend: str,
    inputs_per_rank: List[KeyedJaggedTensor],
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        # debug model is off
        model = DebugEmbeddingCollection(tables=tables, device=ctx.device)

        sharder = EmbeddingCollectionSharder()
        # Use FUSED kernel which supports embedding updates via .write()
        # DENSE kernel does not support .update() method required for .write()
        per_param_sharding = {
            "table_0": row_wise(compute_kernel=EmbeddingComputeKernel.DENSE.value),
            "table_1": row_wise(compute_kernel=EmbeddingComputeKernel.DENSE.value),
            "table_2": data_parallel(),
        }
        sharding_plan = construct_module_sharding_plan(
            model.ec,
            per_param_sharding=per_param_sharding,
            local_size=local_size,
            world_size=world_size,
            device_type=ctx.device.type,
            # pyrefly: ignore[bad-argument-type]
            sharder=sharder,
        )

        set_propogate_device(True)

        # Case 1: everything works as usual
        sharded_model = DistributedModelParallel(
            model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ec": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )

        # Typical backward (no NaNs)
        kjts = inputs_per_rank[rank]
        out = sharded_model(
            kjts.to(ctx.device)
        )  # Returns EmbeddingCollectionAwaitable object

        # compute a scalar loss upon which we can call backward()
        loss = sum(torch.sum(jt.values()) for jt in out.values())
        # pyrefly: ignore[missing-attribute]
        loss.backward()

        torch.cuda.synchronize()

        # Case 2: torch.autograd.set_detect_anomaly(True), we insert NaN in the gradient
        # debug model is False, now if nans are found in backward, torch.autograd.set_detect_anomaly(True)
        # should throw an error
        debug_model = DebugEmbeddingCollection(
            tables=tables, device=ctx.device, debug_mode=False
        )
        debug_sharded_model = DistributedModelParallel(
            debug_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ec": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )
        debug_out = debug_sharded_model(kjts.to(ctx.device))

        candidates = [
            (k, jt.values()) for k, jt in debug_out.items() if jt.values().requires_grad
        ]

        assert (
            candidates
        ), "No outputs require grad; ensure all tables use DENSE kernels"

        k, first_tensor = candidates[0]

        def insert_nan_grad(grad) -> torch.Tensor:
            """Hook to insert nan into the gradient"""
            return torch.full_like(grad, float("nan"))

        first_tensor.register_hook(insert_nan_grad)

        debug_loss = sum(torch.sum(v.values()) for k, v in debug_out.items())

        with torch.autograd.detect_anomaly():
            tc = unittest.TestCase()
            with tc.assertRaisesRegex(
                RuntimeError, "Function 'SplitWithSizesBackward0' returned nan values"
            ):
                # pyrefly: ignore[missing-attribute]
                debug_loss.backward()

        torch.cuda.synchronize()

        debug_model = DebugEmbeddingCollection(
            tables=tables, device=ctx.device, debug_mode=True
        )
        debug_sharded_model = DistributedModelParallel(
            debug_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ec": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )
        debug_out = debug_sharded_model(kjts.to(ctx.device))

        candidates = [
            (k, jt.values()) for k, jt in debug_out.items() if jt.values().requires_grad
        ]
        assert len(candidates) == len(
            debug_out.keys()
        ), "All jt.values() should require grad; ensure all tables use DENSE kernels"

        k, first_tensor = candidates[0]

        first_tensor.register_hook(insert_nan_grad)

        debug_loss = sum(torch.sum(v.values()) for k, v in debug_out.items())

        tc = unittest.TestCase()
        with tc.assertRaisesRegex(
            RuntimeError, "NaN/Inf detected in gradient entering"
        ):
            # pyrefly: ignore[missing-attribute]
            debug_loss.backward()

        torch.cuda.synchronize()


def run_embedding_bag_collection(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    backend: str,
    inputs_per_rank: List[KeyedJaggedTensor],
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        # debug model is off
        model = DebugEmbeddingBagCollection(tables=tables, device=ctx.device)

        sharder = EmbeddingBagCollectionSharder()
        # Use FUSED kernel which supports embedding updates via .write()
        # DENSE kernel does not support .update() method required for .write()
        per_param_sharding = {
            "table_0": row_wise(compute_kernel=EmbeddingComputeKernel.DENSE.value),
            "table_1": row_wise(compute_kernel=EmbeddingComputeKernel.DENSE.value),
            "table_2": data_parallel(),
        }
        sharding_plan = construct_module_sharding_plan(
            model.ebc,
            per_param_sharding=per_param_sharding,
            local_size=local_size,
            world_size=world_size,
            device_type=ctx.device.type,
            # pyrefly: ignore[bad-argument-type]
            sharder=sharder,
        )

        set_propogate_device(True)

        # Case 1: everything works as usual
        sharded_model = DistributedModelParallel(
            model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ec": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )

        # Typical backward (no NaNs)
        kjts = inputs_per_rank[rank]
        out = sharded_model(kjts.to(ctx.device))

        # compute a scalar loss upon which we can call backward()
        loss = sum(torch.sum(v) for v in out.values())
        # pyrefly: ignore[missing-attribute]
        loss.backward()

        torch.cuda.synchronize()

        # Case 2: torch.autograd.set_detect_anomaly(True), we insert NaN in the gradient
        # debug model is False, now if nans are found in backward, torch.autograd.set_detect_anomaly(True)
        # should throw an error
        debug_model = DebugEmbeddingBagCollection(
            tables=tables, device=ctx.device, debug_mode=False
        )
        debug_sharded_model = DistributedModelParallel(
            debug_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ebc": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )
        # Returns a EmbeddingBagCollectionAwaitable
        debug_out_awaitable = debug_sharded_model(kjts.to(ctx.device))
        debug_out = debug_out_awaitable.wait()

        values = list(debug_out.values())
        first_tensor = values[0]
        assert first_tensor.requires_grad, "requires_grad should be True"

        def insert_nan_grad(grad: torch.Tensor) -> torch.Tensor:
            """Hook to insert nan into the gradient"""
            return torch.full_like(grad, float("nan"))

        first_tensor.register_hook(insert_nan_grad)

        debug_loss = sum(torch.sum(v) for v in values)

        tc = unittest.TestCase()
        with torch.autograd.detect_anomaly():
            with tc.assertRaisesRegex(
                RuntimeError, "Function 'UnbindBackward0' returned nan values in"
            ):
                # pyrefly: ignore[missing-attribute]
                debug_loss.backward()

        torch.cuda.synchronize()

        debug_model = DebugEmbeddingBagCollection(
            tables=tables, device=ctx.device, debug_mode=True
        )
        debug_sharded_model = DistributedModelParallel(
            debug_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ebc": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )
        debug_out = debug_sharded_model(kjts.to(ctx.device))
        # we do not call .wait() on the returned object, because
        # _GradCheck has called .wait() already

        values = list(debug_out.values())
        first_tensor = values[0]
        assert first_tensor.requires_grad, "requires_grad should be True"

        first_tensor.register_hook(insert_nan_grad)

        debug_loss = sum(torch.sum(v) for v in values)

        with tc.assertRaisesRegex(
            RuntimeError, "NaN/Inf detected in gradient entering"
        ):
            # pyrefly: ignore[missing-attribute]
            debug_loss.backward()

        torch.cuda.synchronize()
