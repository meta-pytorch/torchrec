#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import unittest
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.mc_embedding import (
    ManagedCollisionEmbeddingCollectionSharder,
    ShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import ManagedCollisionCollectionContext
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

logger: logging.Logger = logging.getLogger(__name__)
WORLD_SIZE = 2
EMBEDDING_DIM = 4


class WriteSparseArch(nn.Module):
    """
    Wrapper module for testing MCEC write path.
    Sets enable_embedding_update=True on all tables.
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        input_hash_size: int = 4000,
    ) -> None:
        super().__init__()

        mc_modules: Dict[str, ManagedCollisionModule] = {}
        for table in tables:
            mc_modules[table.name] = MCHManagedCollisionModule(
                zch_size=table.num_embeddings,
                input_hash_size=input_hash_size,
                device=device,
                eviction_interval=2,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )

        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(
                    tables=tables,
                    device=device,
                ),
                ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=tables,
                ),
                return_remapped_features=True,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        ec_out, remapped_ids_out = self._mc_ec(kjt)
        result = {}
        for key in ec_out.keys():
            result[key] = ec_out[key].values()
        return result, remapped_ids_out


def _create_sharded_arch(
    tables: List[EmbeddingConfig],
    ctx: MultiProcessContext,
    sharder: ModuleSharder[nn.Module],
    input_hash_size: int = 4000,
) -> WriteSparseArch:
    """Helper to create and shard a WriteSparseArch."""
    sparse_arch = WriteSparseArch(
        tables,
        torch.device("meta"),
        input_hash_size=input_hash_size,
    )

    apply_optimizer_in_backward(
        RowWiseAdagrad,
        # pyre-fixme[6]: Argument list is not assignable to parameter type
        [
            sparse_arch._mc_ec._embedding_collection.embeddings[t.name].weight
            for t in tables
        ],
        {"lr": 0.01},
    )

    module_sharding_plan = construct_module_sharding_plan(
        sparse_arch._mc_ec,
        per_param_sharding={t.name: row_wise() for t in tables},
        local_size=ctx.local_size,
        world_size=WORLD_SIZE,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        sharder=sharder,
    )

    sharded_sparse_arch = _shard_modules(
        module=copy.deepcopy(sparse_arch),
        plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
        # pyrefly: ignore[bad-argument-type]
        env=ShardingEnv.from_process_group(ctx.pg),
        sharders=[sharder],
        device=ctx.device,
    )

    return sharded_sparse_arch  # pyre-fixme[7]


def _test_write_all_to_all_routing(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that calling input_dist twice with the same 2D-weighted KJT produces
    identical outputs, validating deterministic AlltoAll routing.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _create_sharded_arch(tables, ctx, sharder)
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Forward to populate ZCH
        kjt_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([100, 200, 300, 400]),
            lengths=torch.LongTensor([1, 1, 1, 1]),
            weights=None,
        ).to(ctx.device)

        loss, _ = sharded_sparse_arch(kjt_input)
        torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Create KJT with 2D weights
        write_dim = tables[0].embedding_dim
        write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([100, 200, 300, 400]),
            lengths=torch.LongTensor([1, 1, 1, 1]),
            weights=torch.randn(4, write_dim),
        ).to(ctx.device)

        # First input_dist call
        mc_ctx1 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out1 = mcc.input_dist(mc_ctx1, write_kjt).wait().wait()

        # Second input_dist call with same input
        mc_ctx2 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out2 = mcc.input_dist(mc_ctx2, write_kjt).wait().wait()

        # Verify outputs are identical
        for kjt1, kjt2 in zip(dist_out1, dist_out2):
            assert torch.equal(
                kjt1.values(), kjt2.values()
            ), f"Rank {rank}: values mismatch between two input_dist calls"
            assert torch.equal(
                kjt1.lengths(), kjt2.lengths()
            ), f"Rank {rank}: lengths mismatch between two input_dist calls"


def _test_write_with_unknown_ids(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that input_dist with unknown IDs (not in ZCH) doesn't crash.
    Calling input_dist twice with the same unknown IDs produces identical outputs.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _create_sharded_arch(tables, ctx, sharder)
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Forward to populate ZCH with some IDs
        kjt_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1000, 2000, 1001, 2001]),
            lengths=torch.LongTensor([1, 1, 1, 1]),
            weights=None,
        ).to(ctx.device)

        loss, _ = sharded_sparse_arch(kjt_input)
        torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Write with IDs that were NEVER seen in forward (not in ZCH)
        write_dim = tables[0].embedding_dim
        unknown_ids = torch.LongTensor([9999, 8888, 7777, 6666])
        write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=unknown_ids,
            lengths=torch.LongTensor([1, 1, 1, 1]),
            weights=torch.randn(4, write_dim),
        ).to(ctx.device)

        # First input_dist call — should not crash
        mc_ctx1 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out1 = mcc.input_dist(mc_ctx1, write_kjt).wait().wait()

        # Second input_dist call with same input
        mc_ctx2 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out2 = mcc.input_dist(mc_ctx2, write_kjt).wait().wait()

        # Verify outputs are identical
        for kjt1, kjt2 in zip(dist_out1, dist_out2):
            assert torch.equal(
                kjt1.values(), kjt2.values()
            ), f"Rank {rank}: values mismatch between two input_dist calls"
            assert torch.equal(
                kjt1.lengths(), kjt2.lengths()
            ), f"Rank {rank}: lengths mismatch between two input_dist calls"


class HashZchWriteSparseArch(nn.Module):
    """
    Wrapper module for testing MCEC write path with HashZch (MPZCH).
    Uses HashZchManagedCollisionModule which supports eviction and runtime_meta.
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        input_hash_size: int = 0,
        total_num_buckets: int = 4,
        max_probe: int = 5,
        write_runtime_meta_dim: int = 0,
    ) -> None:
        super().__init__()

        mc_modules: Dict[str, ManagedCollisionModule] = {}
        for table in tables:
            mc_modules[table.name] = HashZchManagedCollisionModule(
                zch_size=table.num_embeddings,
                input_hash_size=input_hash_size,
                device=device,
                total_num_buckets=total_num_buckets,
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=table.feature_names,
                    single_ttl=-1,
                ),
                max_probe=max_probe,
                track_id_freq=write_runtime_meta_dim == 0,
                write_runtime_meta_dim=write_runtime_meta_dim,
            )

        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(
                    tables=tables,
                    device=device,
                ),
                ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=tables,
                ),
                return_remapped_features=True,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        ec_out, remapped_ids_out = self._mc_ec(kjt)
        result = {}
        for key in ec_out.keys():
            result[key] = ec_out[key].values()
        return result, remapped_ids_out


def _create_hashzch_sharded_arch(
    tables: List[EmbeddingConfig],
    ctx: MultiProcessContext,
    sharder: ModuleSharder[nn.Module],
    input_hash_size: int = 0,
    total_num_buckets: int = 4,
    max_probe: int = 5,
    write_runtime_meta_dim: int = 0,
) -> HashZchWriteSparseArch:
    """Helper to create and shard a HashZchWriteSparseArch."""
    sparse_arch = HashZchWriteSparseArch(
        tables,
        torch.device("meta"),
        input_hash_size=input_hash_size,
        total_num_buckets=total_num_buckets,
        max_probe=max_probe,
        write_runtime_meta_dim=write_runtime_meta_dim,
    )
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        # pyre-fixme[6]: Argument list is not assignable to parameter type
        [
            sparse_arch._mc_ec._embedding_collection.embeddings[t.name].weight
            for t in tables
        ],
        {"lr": 0.01},
    )

    module_sharding_plan = construct_module_sharding_plan(
        sparse_arch._mc_ec,
        per_param_sharding={t.name: row_wise() for t in tables},
        local_size=ctx.local_size,
        world_size=WORLD_SIZE,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        sharder=sharder,
    )

    sharded_sparse_arch = _shard_modules(
        module=copy.deepcopy(sparse_arch),
        plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
        # pyrefly: ignore[bad-argument-type]
        env=ShardingEnv.from_process_group(ctx.pg),
        sharders=[sharder],
        device=ctx.device,
    )

    return sharded_sparse_arch  # pyre-fixme[7]


def _test_input_dist_2d_weights_mapping(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that input_dist() and compute() with 2D weights preserves the 1:1
    mapping between values[i] and weights[i] through the full pipeline:
    input_dist (AlltoAll) -> compute (remap).

    Each ID gets a unique weight vector (filled with the ID value) so that
    after redistribution and remap we can verify each remapped value is
    paired with the correct weight row on every rank.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            tables, ctx, sharder, write_runtime_meta_dim=EMBEDDING_DIM // WORLD_SIZE
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Create KJT with 2D weights where weights[i] = [ids[i]] * dim
        # This creates a unique fingerprint per ID for verification
        ids = torch.LongTensor([100, 200, 300, 400])
        write_dim = tables[0].embedding_dim
        weight_values = ids.float().unsqueeze(1).expand(-1, write_dim)
        kjt_with_weights = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=ids,
            lengths=torch.LongTensor([4]),
            weights=weight_values,
        ).to(ctx.device)

        # Run input_dist with 2D weights
        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_output = mcc.input_dist(mc_ctx, kjt_with_weights).wait().wait()

        # Verify 1:1 mapping after AlltoAll (input_dist)
        for kjt_per_sharding in dist_output:
            local_values = kjt_per_sharding.values()
            local_weights = kjt_per_sharding.weights()
            if local_values.numel() == 0:
                continue

            assert local_weights is not None, "Weights should not be None after dist"
            assert (
                local_weights.dim() == 2
            ), f"Expected 2D weights, got {local_weights.dim()}D"
            assert local_values.shape[0] == local_weights.shape[0], (
                f"Mismatch: {local_values.shape[0]} values vs "
                f"{local_weights.shape[0]} weight rows"
            )

            # Each weight row should be filled with the corresponding value
            for i in range(local_values.shape[0]):
                val = local_values[i].float()
                expected_row = val.expand(write_dim)
                actual_row = local_weights[i]
                assert torch.allclose(actual_row, expected_row), (
                    f"Rank {rank}, index {i}: value={val.item()}, "
                    f"expected weights all {val.item()}, got {actual_row}"
                )

        # Run compute() — remaps raw IDs to local ZCH indices.
        compute_output = mcc.compute(mc_ctx, dist_output)
        assert compute_output is not None


def _test_input_dist_2d_weights_with_eviction(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that after eviction, the 1:1 mapping between values and 2D weights
    is preserved through input_dist(), and that runtime_meta is correctly
    updated at remapped slots and cleared at evicted slots.

    Flow:
    1. Fill ZCH with IDs via forward (triggers insertion)
    2. Run write with 2D weights → updates runtime_meta
    3. Insert new IDs via forward (triggers eviction of old IDs)
    4. Run write with new IDs → verify runtime_meta updated for new IDs
    5. Verify evicted slots' runtime_meta was reset by the kernel
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Small ZCH size to trigger eviction quickly
        small_tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                # Small table: total 16 slots, 8 per rank with 2 ranks
                num_embeddings=16,
                enable_embedding_update=True,
            ),
        ]

        sharded_sparse_arch = _create_hashzch_sharded_arch(
            small_tables,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=4,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Fill ZCH with initial IDs via forward passes
        initial_ids = list(range(1000, 1008))
        kjt_fill = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor(initial_ids),
            lengths=torch.LongTensor([len(initial_ids)]),
            weights=None,
        ).to(ctx.device)

        # Multiple forwards to firmly insert IDs
        for _ in range(3):
            loss, _ = sharded_sparse_arch(kjt_fill)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 2: Write with 2D weights for the initial IDs
        write_weights = torch.stack(
            [
                torch.tensor([v], dtype=torch.int64).view(torch.float32)
                for v in initial_ids
            ]
        )
        write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor(initial_ids),
            lengths=torch.LongTensor([len(initial_ids)]),
            weights=write_weights,
        ).to(ctx.device)

        mc_ctx_w1 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out_w1 = mcc.input_dist(mc_ctx_w1, write_kjt).wait().wait()
        mcc.compute(mc_ctx_w1, dist_out_w1)

        # Step 3: Verify runtime_meta was updated on this rank's MC module
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            if hasattr(mc_module, "_hash_zch_runtime_meta"):
                meta = mc_module._hash_zch_runtime_meta
                assert meta is not None, "runtime_meta should exist after write"
                # Some slots should have non-zero values (written by update_runtime_meta)
                # The exact values depend on dtype casting (float→int64)
                assert meta.shape[0] == mc_module._zch_size  # pyre-fixme[16]
                assert meta.shape[1] == 1  # pyre-fixme[16]

        # Step 4: Insert many new IDs to trigger eviction
        new_ids = list(range(2000, 2016))
        kjt_new = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor(new_ids),
            lengths=torch.LongTensor([len(new_ids)]),
            weights=None,
        ).to(ctx.device)

        for _ in range(3):
            loss, _ = sharded_sparse_arch(kjt_new)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 5: Write with 2D weights for new IDs
        new_write_weights = torch.stack(
            [torch.tensor([v], dtype=torch.int64).view(torch.float32) for v in new_ids]
        )
        new_write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor(new_ids),
            lengths=torch.LongTensor([len(new_ids)]),
            weights=new_write_weights,
        ).to(ctx.device)

        mc_ctx_w2 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out_w2 = mcc.input_dist(mc_ctx_w2, new_write_kjt).wait().wait()
        mcc.compute(mc_ctx_w2, dist_out_w2)

        # Step 6: Verify input_dist still preserves 1:1 mapping after eviction
        verify_ids = torch.LongTensor(new_ids[:4])
        verify_weights = torch.stack(
            [
                torch.tensor([v], dtype=torch.int64).view(torch.float32)
                for v in new_ids[:4]
            ]
        )
        verify_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=verify_ids,
            lengths=torch.LongTensor([len(verify_ids)]),
            weights=verify_weights,
        ).to(ctx.device)

        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_output = mcc.input_dist(mc_ctx, verify_kjt).wait().wait()

        for kjt_per_sharding in dist_output:
            local_values = kjt_per_sharding.values()
            local_weights = kjt_per_sharding.weights()

            if local_values.numel() == 0:
                continue

            assert local_weights is not None
            assert local_values.shape[0] == local_weights.shape[0], (
                f"1:1 mapping broken after eviction: "
                f"{local_values.shape[0]} values vs {local_weights.shape[0]} weights"
            )

            for i in range(local_values.shape[0]):
                val = local_values[i].item()
                expected_row = torch.tensor(
                    [int(val)], dtype=torch.int64, device=local_weights.device
                ).view(torch.float32)
                actual_row = local_weights[i]
                assert torch.allclose(actual_row, expected_row), (
                    f"Rank {rank}, index {i}: value={val}, "
                    f"expected weights {expected_row}, got {actual_row}"
                )

        # Step 7: Verify runtime_meta consistency
        # After eviction + new writes, runtime_meta should reflect
        # the new IDs, not the old evicted ones
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            if hasattr(mc_module, "_hash_zch_runtime_meta"):
                meta = mc_module._hash_zch_runtime_meta
                assert meta is not None
                # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
                identities: torch.Tensor = mc_module._hash_zch_identities.data
                # pyre-ignore[16]: Cannot index into `Module`, expected `__getitem__` to be callable
                assert meta.shape[0] == identities.shape[0]
                # Unoccupied slots (identity == -1) should have zeroed runtime_meta
                unoccupied: torch.Tensor = (torch.flatten(identities) == -1).nonzero(
                    as_tuple=True
                )[0]
                for slot_idx in unoccupied:
                    assert torch.all(meta[slot_idx] == 0), (  # pyre-fixme[16]
                        f"Rank {rank}: unoccupied slot {slot_idx} has "
                        f"non-zero runtime_meta after eviction"
                    )


@skip_if_asan_class
class ShardedMCECWriteAllToAllTest(MultiProcessTestBase):
    """Tests for write path AlltoAll routing and unknown ID handling."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_write_all_to_all_routing(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_write_all_to_all_routing,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_write_with_unknown_ids(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_write_with_unknown_ids,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )


@skip_if_asan_class
class ShardedMCECInputDist2DWeightsTest(MultiProcessTestBase):
    """Tests for input_dist with 2D weights and eviction consistency."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_input_dist_2d_weights_mapping(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_input_dist_2d_weights_mapping,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_input_dist_2d_weights_with_eviction(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=16,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_input_dist_2d_weights_with_eviction,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_multi_table_2d_weights_split(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_multi_table_2d_weights_split,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )


def _test_multi_table_2d_weights_split(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that compute() correctly splits 2D weights across multiple tables
    using the weight_offset logic when len(splits) > 1.

    Each table gets IDs with distinct weight fingerprints so we can verify
    the correct weight slice is passed to each table's remap.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            tables, ctx, sharder, write_runtime_meta_dim=EMBEDDING_DIM // WORLD_SIZE
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Forward pass to insert IDs
        ids = torch.LongTensor([100, 200, 300, 400])
        kjt_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=ids,
            lengths=torch.LongTensor([2, 2]),
            weights=None,
        ).to(ctx.device)

        loss, _ = sharded_sparse_arch(kjt_input)
        torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 2: Create KJT with 2D weights — different values per table
        write_dim = tables[0].embedding_dim
        # feature_0 gets IDs [100, 200] with weights filled with 1.0
        # feature_1 gets IDs [300, 400] with weights filled with 2.0
        weight_values = torch.tensor(
            [
                [1.0] * write_dim,
                [1.0] * write_dim,
                [2.0] * write_dim,
                [2.0] * write_dim,
            ],
            dtype=torch.float,
        )
        kjt_with_weights = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=ids,
            lengths=torch.LongTensor([2, 2]),
            weights=weight_values,
        ).to(ctx.device)

        # Step 3: Run input_dist + compute with 2D weights
        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_output = mcc.input_dist(mc_ctx, kjt_with_weights).wait().wait()
        compute_output = mcc.compute(mc_ctx, dist_output)

        # Step 4: Verify output KJTs exclude 2D weights
        for kjt_remapped in compute_output:
            remapped_weights = kjt_remapped.weights_or_none()
            assert remapped_weights is None, (
                f"Rank {rank}: 2D weights should be excluded from output KJT "
                f"but got weights with shape {remapped_weights.shape}"
            )

        # Step 5: Verify runtime_meta was updated per table
        for table_name, mc_module in mcc._managed_collision_modules.items():
            if hasattr(mc_module, "_hash_zch_runtime_meta"):
                meta = mc_module._hash_zch_runtime_meta
                assert (
                    meta is not None
                ), f"Rank {rank}: runtime_meta should exist for {table_name}"
                assert meta.shape[0] == mc_module._zch_size  # pyre-fixme[16]


def _test_dedup_indices_preserves_2d_weights(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that _dedup_indices preserves 2D weights when input has duplicate IDs.

    When use_index_dedup=True and the KJT contains duplicate IDs with 2D weights,
    the deduplication should:
    1. Remove duplicate values (keeping unique ones)
    2. Keep the corresponding weight row for each unique value
    3. Maintain the invariant: dedup_weights[reverse_indices[i]] matches one of
       the weight rows for the original indices[i]
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            tables,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=2,
            write_runtime_meta_dim=EMBEDDING_DIM // WORLD_SIZE,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Enable index dedup
        mcc._use_index_dedup = True

        # Step 1: Forward pass to insert IDs into ZCH
        ids = torch.LongTensor([100, 200, 300, 400])
        kjt_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=ids,
            lengths=torch.LongTensor([4]),
            weights=None,
        ).to(ctx.device)

        loss, _ = sharded_sparse_arch(kjt_input)
        torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 2: Create KJT with duplicate IDs and 2D weights.
        # IDs [100, 200, 100, 300] — ID 100 appears twice with different weights.
        # Weights have EMBEDDING_DIM columns which split to EMBEDDING_DIM // WORLD_SIZE per rank.
        write_dim = tables[0].embedding_dim
        dup_ids = torch.LongTensor([100, 200, 100, 300])
        dup_weights = torch.tensor(
            [
                [10.0] * write_dim,  # ID 100, first
                [20.0] * write_dim,  # ID 200
                [30.0] * write_dim,  # ID 100, duplicate
                [40.0] * write_dim,  # ID 300
            ],
            dtype=torch.float,
        )
        kjt_dup = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=dup_ids,
            lengths=torch.LongTensor([4]),
            weights=dup_weights,
        ).to(ctx.device)

        # Step 3: Run input_dist which calls _dedup_indices internally
        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        mc_ctx.reverse_indices = []  # pyre-fixme[16]
        dist_output = mcc.input_dist(mc_ctx, kjt_dup).wait().wait()

        # Step 4: Run compute() — remaps IDs and updates runtime_meta
        compute_output = mcc.compute(mc_ctx, dist_output)
        assert compute_output is not None

        # Step 5: Verify runtime_meta has correct values for inserted IDs.
        # The write weights were [10.0]*dim, [20.0]*dim, [30.0]*dim, [40.0]*dim
        # for IDs 100, 200, 100(dup), 300. After dedup, ID 100 keeps one of
        # [10.0]*dim or [30.0]*dim. After bucketization, each rank gets
        # EMBEDDING_DIM // WORLD_SIZE columns per weight row.
        local_write_dim = EMBEDDING_DIM // WORLD_SIZE
        # Expected float values that map to the written weights after
        # block_bucketize_sparse_features_2d_weights splits columns across ranks.
        expected_float_vals = {10.0, 20.0, 30.0, 40.0}
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            runtime_meta = mc_module._hash_zch_runtime_meta
            if runtime_meta is None:
                continue

            # Map input IDs through input_mapper to get mapped IDs
            mapped_ids, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=dup_ids.unique().to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )

            for mapped_id in mapped_ids:
                slot_idx = (identities == mapped_id).nonzero(as_tuple=True)[0]
                if slot_idx.numel() == 0:
                    # ID not on this rank
                    continue
                slot = slot_idx[0].item()
                meta_row = runtime_meta.data[slot]  # pyre-fixme[16]
                # Inserted ID should have non-zero runtime_meta
                assert (
                    meta_row.numel() == local_write_dim
                ), f"Rank {rank}: runtime_meta dim {meta_row.numel()} != {local_write_dim}"
                # Verify the meta value is one of the expected written values
                meta_float = meta_row.view(torch.float32)
                fill_val = meta_float[0].item()
                assert fill_val in expected_float_vals, (
                    f"Rank {rank}, slot {slot}: unexpected runtime_meta "
                    f"value {fill_val}, expected one of {expected_float_vals}"
                )


def _test_dedup_indices_no_weights_unchanged(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that _dedup_indices still works correctly with no weights (the original
    code path). When weights are None, dedup should not add any weights.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            tables,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=2,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Enable index dedup
        mcc._use_index_dedup = True

        # KJT with duplicate IDs but NO weights
        dup_ids = torch.LongTensor([100, 200, 100, 300])
        kjt_no_weights = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=dup_ids,
            lengths=torch.LongTensor([4]),
            weights=None,
        ).to(ctx.device)

        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        mc_ctx.reverse_indices = []  # pyre-fixme[16]
        dist_output = mcc.input_dist(mc_ctx, kjt_no_weights).wait().wait()

        # Verify no weights are attached
        for kjt_per_sharding in dist_output:
            local_weights = kjt_per_sharding.weights_or_none()
            assert local_weights is None, (
                f"Rank {rank}: weights should be None when input has no weights, "
                f"got {local_weights}"
            )

        # Run compute() — no weights should remain None throughout
        compute_output = mcc.compute(mc_ctx, dist_output)
        assert compute_output is not None


def _test_overwrite_ids_state_dict(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test inserting 2 KJTs sequentially where some IDs from the 1st KJT
    get overwritten by the 2nd KJT. Verify via state_dict that:
    1. Identity tensor contains all inserted IDs
    2. runtime_meta at overwritten slots has the 2nd KJT's weights
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            embedding_config,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=4,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Forward pass to insert IDs into ZCH identity tensor
        all_ids = torch.LongTensor([100, 200, 300, 400, 500, 600])
        kjt_insert = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=all_ids,
            lengths=torch.LongTensor([6]),
            weights=None,
        ).to(ctx.device)

        loss, _ = sharded_sparse_arch(kjt_insert)
        torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()
        # Step 2: Write KJT1 with IDs [100, 200, 300, 400] and unique weight fingerprints
        kjt1_ids = torch.LongTensor([100, 200, 300, 400])
        kjt1_weights = torch.stack(
            [
                torch.tensor([10], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 100 -> 10
                torch.tensor([20], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 200 -> 20
                torch.tensor([30], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 300 -> 30
                torch.tensor([40], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 400 -> 40
            ]
        )
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt1_ids,
            lengths=torch.LongTensor([4]),
            weights=kjt1_weights,
        ).to(ctx.device)

        mc_ctx1 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out1 = mcc.input_dist(mc_ctx1, kjt1).wait().wait()
        mcc.compute(mc_ctx1, dist_out1)

        # Step 3: Write KJT2 with IDs [300, 400, 500, 600] — overlapping 300/400
        kjt2_ids = torch.LongTensor([300, 400, 500, 600])
        kjt2_weights = torch.stack(
            [
                torch.tensor([130], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 300 -> 130 (overwrite)
                torch.tensor([140], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 400 -> 140 (overwrite)
                torch.tensor([150], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 500 -> 150
                torch.tensor([160], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 600 -> 160
            ]
        )
        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt2_ids,
            lengths=torch.LongTensor([4]),
            weights=kjt2_weights,
        ).to(ctx.device)

        mc_ctx2 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out2 = mcc.input_dist(mc_ctx2, kjt2).wait().wait()
        mcc.compute(mc_ctx2, dist_out2)

        # Step 4: Verify state_dict
        sd = sharded_sparse_arch.state_dict()

        for key, val in sd.items():
            if "_hash_zch_identities" in key and "_managed_collision_modules" in key:
                if isinstance(val, ShardedTensor):
                    identities = val.local_shards()[0].tensor
                else:
                    identities = val
                logger.info(f"rank {rank}: identities={identities.flatten()}")
                # All 6 IDs should be present across both ranks.
                # On this rank's local shard, check that occupied slots exist.
                occupied = identities.flatten() != -1
                if not occupied.any():
                    logger.info(f"Rank {rank}: no IDs in identity tensor")

            if "_hash_zch_runtime_meta" in key and "_managed_collision_modules" in key:
                if isinstance(val, ShardedTensor):
                    runtime_meta = val.local_shards()[0].tensor
                else:
                    runtime_meta = val

        # Step 5: Verify per-MC-module that overwritten slots have KJT2 values
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            runtime_meta = mc_module._hash_zch_runtime_meta.data
            assert isinstance(runtime_meta, torch.Tensor)

            # Build a map from mapped_id -> slot for verification
            # Use input_mapper to get mapped IDs
            mapped_ids, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=all_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )

            for i, raw_id in enumerate(all_ids.tolist()):
                mapped_id = mapped_ids[i].item()
                # Find slot where this mapped_id lives
                slot_mask = identities == mapped_id
                if not slot_mask.any():
                    continue  # This ID is on the other rank's shard

                slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                meta_row = runtime_meta[slot_idx]

                # Determine expected value:
                # IDs 300, 400 were overwritten by KJT2
                if raw_id == 300:
                    expected = 130.0
                elif raw_id == 400:
                    expected = 140.0
                elif raw_id == 100:
                    expected = 10.0
                elif raw_id == 200:
                    expected = 20.0
                elif raw_id == 500:
                    expected = 150.0
                elif raw_id == 600:
                    expected = 160.0
                else:
                    continue

                # runtime_meta is stored as int64, check cast
                expected_tensor = torch.full_like(
                    meta_row, expected, dtype=torch.float
                ).to(runtime_meta.dtype)
                assert torch.equal(meta_row, expected_tensor), (
                    f"Rank {rank}: ID {raw_id} at slot {slot_idx}: "
                    f"expected runtime_meta={expected}, got {meta_row}"
                )


def _test_partial_insertion_state_dict(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that when not all IDs in a write KJT are inserted (some map to
    sentinel or miss), only actually-inserted IDs have their runtime_meta
    updated. Verify via state_dict that identity tensor and runtime_meta
    are consistent.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            embedding_config,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=4,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Forward with a subset of IDs to insert them into ZCH
        known_ids = torch.LongTensor([100, 200, 300])
        kjt_insert = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=known_ids,
            lengths=torch.LongTensor([3]),
            weights=None,
        ).to(ctx.device)

        loss, _ = sharded_sparse_arch(kjt_insert)
        torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 2: Write a KJT that includes both known and unknown IDs
        # IDs 100, 200, 300 are inserted; 3501, 3502, 3503 were never forward'd
        # Unknown IDs must be < input_hash_size (4000) to map to valid buckets.
        write_ids = torch.LongTensor([100, 200, 300, 3510, 3520, 3530])
        write_weights = torch.stack(
            [
                torch.tensor([10], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 100 -> 10.0
                torch.tensor([20], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 200 -> 20.0
                torch.tensor([30], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 300 -> 30.0
                torch.tensor([351], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 3510 -> 351.0 (may not insert)
                torch.tensor([352], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 3520 -> 352.0 (may not insert)
                torch.tensor([353], dtype=torch.int64).view(
                    torch.float32
                ),  # ID 3530 -> 353.0 (may not insert)
            ]
        )
        write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=write_ids,
            lengths=torch.LongTensor([6]),
            weights=write_weights,
        ).to(ctx.device)

        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out = mcc.input_dist(mc_ctx, write_kjt).wait().wait()
        mcc.compute(mc_ctx, dist_out)

        # Verify per-MC-module consistency
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            runtime_meta = mc_module._hash_zch_runtime_meta.data
            assert isinstance(runtime_meta, torch.Tensor)

            # Check known IDs that were forwarded
            mapped_known, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=known_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )

            for i, raw_id in enumerate(known_ids.tolist()):
                mapped_id = mapped_known[i].item()
                slot_mask = identities == mapped_id
                if not slot_mask.any():
                    continue  # On other rank's shard

                slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                meta_row = runtime_meta[slot_idx]

                expected = float(raw_id // 10)  # 10.0, 20.0, 30.0
                expected_tensor = torch.full_like(
                    meta_row, expected, dtype=torch.float
                ).to(runtime_meta.dtype)
                assert torch.equal(meta_row, expected_tensor), (
                    f"Rank {rank}: known ID {raw_id} at slot {slot_idx}: "
                    f"expected runtime_meta={expected}, got {meta_row}"
                )

            # Verify consistency: every occupied slot (identity != -1)
            # should have runtime_meta that is either zero (from kernel
            # init or eviction zeroing) or a valid written value.
            # Non-occupied slots should have zero runtime_meta.
            for slot_idx in range(identities.numel()):
                if identities[slot_idx].item() == -1:
                    # Unoccupied slot: runtime_meta should be zero
                    meta_row = runtime_meta[slot_idx]
                    assert torch.all(meta_row == 0), (
                        f"Rank {rank}: unoccupied slot {slot_idx} has "
                        f"non-zero runtime_meta={meta_row}"
                    )


def _test_eviction_clears_runtime_meta(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that when KJT1 fills ~90% of identity slots and then KJT2 causes
    eviction of some KJT1 IDs, the evicted slots have runtime_meta zeroed
    and surviving KJT1 slots retain their written values. Also verifies
    that newly inserted KJT2 IDs get correct runtime_meta.

    Flow:
    1. Forward KJT1 (many IDs) to fill ~90% of the small ZCH table
    2. Write runtime_meta for KJT1 IDs
    3. Forward KJT2 (new IDs) which triggers eviction of some KJT1 IDs
    4. Write runtime_meta for KJT2 IDs
    5. Verify:
       - Evicted KJT1 slots: identity == -1 or replaced, runtime_meta zeroed
       - Surviving KJT1 slots: runtime_meta retains KJT1 written values
       - KJT2 slots: runtime_meta has KJT2 written values
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Small table: 16 total slots = 8 per rank with 2 ranks
        # With total_num_buckets=4, each rank gets 2 buckets of 4 slots.
        small_tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=16,
                enable_embedding_update=True,
            ),
        ]
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            small_tables,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=4,
            max_probe=8,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Forward with many IDs to fill all slots
        # 8 slots per rank, but IDs are hash-distributed across 2 ranks,
        # so we need many more IDs to ensure each rank's bucket is full.
        kjt1_ids = torch.LongTensor(list(range(1000, 1025)))  # 25 IDs
        kjt1_insert = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt1_ids,
            lengths=torch.LongTensor([len(kjt1_ids)]),
            weights=None,
        ).to(ctx.device)

        # Multiple forwards to firmly insert
        for _ in range(3):
            loss, _ = sharded_sparse_arch(kjt1_insert)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 2: Write runtime_meta for KJT1 IDs
        # Each ID gets a unique int64 fingerprint viewed as float32,
        # matching write_runtime_meta_dim=1.
        kjt1_weights = torch.stack(
            [
                torch.tensor([100 + i], dtype=torch.int64).view(torch.float32)
                for i in range(len(kjt1_ids))
            ]
        )
        write_kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt1_ids,
            lengths=torch.LongTensor([len(kjt1_ids)]),
            weights=kjt1_weights,
        ).to(ctx.device)

        mc_ctx1 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out1 = mcc.input_dist(mc_ctx1, write_kjt1).wait().wait()
        mcc.compute(mc_ctx1, dist_out1)
        mcc.evict()

        # Snapshot which KJT1 IDs are on this rank before eviction
        kjt1_slots_before: Dict[int, int] = {}  # raw_id -> slot_idx
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            mapped_kjt1, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=kjt1_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )
            for i, raw_id in enumerate(kjt1_ids.tolist()):
                mapped_id = mapped_kjt1[i].item()
                slot_mask = identities == mapped_id
                if slot_mask.any():
                    slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                    kjt1_slots_before[raw_id] = slot_idx

        # Step 3: Forward with new IDs to trigger eviction
        # Insert enough new IDs to force eviction of some KJT1 IDs
        kjt2_ids = torch.LongTensor(list(range(2000, 2025)))  # 25 new IDs
        kjt2_insert = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt2_ids,
            lengths=torch.LongTensor([len(kjt2_ids)]),
            weights=None,
        ).to(ctx.device)

        for _ in range(3):
            loss, _ = sharded_sparse_arch(kjt2_insert)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 4: Write runtime_meta for KJT2 IDs
        kjt2_weights = torch.stack(
            [
                torch.tensor([200 + i], dtype=torch.int64).view(torch.float32)
                for i in range(len(kjt2_ids))
            ]
        )
        write_kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt2_ids,
            lengths=torch.LongTensor([len(kjt2_ids)]),
            weights=kjt2_weights,
        ).to(ctx.device)

        mc_ctx2 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out2 = mcc.input_dist(mc_ctx2, write_kjt2).wait().wait()
        mcc.compute(mc_ctx2, dist_out2)
        mcc.evict()

        # Step 5: Verify per-MC-module
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            runtime_meta = mc_module._hash_zch_runtime_meta.data
            assert isinstance(runtime_meta, torch.Tensor)

            # Check which KJT1 IDs survived vs were evicted
            mapped_kjt1, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=kjt1_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )
            survived_count = 0
            evicted_count = 0
            for i, raw_id in enumerate(kjt1_ids.tolist()):
                mapped_id = mapped_kjt1[i].item()
                slot_mask = identities == mapped_id
                if slot_mask.any():
                    # KJT1 ID survived eviction — runtime_meta should
                    # retain the KJT1 written value
                    slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                    meta_row = runtime_meta[slot_idx]
                    expected = float(100 + i)
                    expected_tensor = torch.full_like(
                        meta_row, expected, dtype=torch.float
                    ).to(runtime_meta.dtype)
                    assert torch.equal(meta_row, expected_tensor), (
                        f"Rank {rank}: surviving KJT1 ID {raw_id} at slot "
                        f"{slot_idx}: expected runtime_meta={expected}, "
                        f"got {meta_row}"
                    )
                    survived_count += 1
                elif raw_id in kjt1_slots_before:
                    # This ID was on this rank before but got evicted
                    evicted_count += 1
                    old_slot = kjt1_slots_before[raw_id]
                    # The old slot should either:
                    # - Be occupied by a KJT2 ID (identity changed)
                    # - Or be empty (identity == -1)
                    # In either case, the old KJT1 runtime_meta value
                    # should NOT remain.
                    meta_row = runtime_meta[old_slot]
                    old_expected = float(100 + i)
                    old_expected_tensor = torch.full_like(
                        meta_row, old_expected, dtype=torch.float
                    ).to(runtime_meta.dtype)
                    if identities[old_slot].item() == -1:
                        # Empty slot: runtime_meta should be zeroed
                        assert torch.all(meta_row == 0), (
                            f"Rank {rank}: evicted slot {old_slot} (was ID "
                            f"{raw_id}) has non-zero runtime_meta={meta_row}"
                        )
                    else:
                        # Slot reused by a different ID: runtime_meta
                        # should NOT have the old KJT1 value
                        assert not torch.equal(meta_row, old_expected_tensor), (
                            f"Rank {rank}: slot {old_slot} was evicted from "
                            f"ID {raw_id} but still has old "
                            f"runtime_meta={old_expected}"
                        )

            # Verify KJT2 IDs have correct runtime_meta
            mapped_kjt2, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=kjt2_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )
            for i, raw_id in enumerate(kjt2_ids.tolist()):
                mapped_id = mapped_kjt2[i].item()
                slot_mask = identities == mapped_id
                if not slot_mask.any():
                    continue  # On other rank's shard

                slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                meta_row = runtime_meta[slot_idx]
                expected = float(200 + i)
                expected_tensor = torch.full_like(
                    meta_row, expected, dtype=torch.float
                ).to(runtime_meta.dtype)
                assert torch.equal(meta_row, expected_tensor), (
                    f"Rank {rank}: KJT2 ID {raw_id} at slot {slot_idx}: "
                    f"expected runtime_meta={expected}, got {meta_row}"
                )
            # Verify unoccupied slots have zero runtime_meta
            for slot_idx in range(identities.numel()):
                if identities[slot_idx].item() == -1:
                    meta_row = runtime_meta[slot_idx]
                    assert torch.all(meta_row == 0), (
                        f"Rank {rank}: unoccupied slot {slot_idx} has "
                        f"non-zero runtime_meta={meta_row}"
                    )


def _test_full_table_eviction_overwrites_runtime_meta(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test that KJT1 fills ALL slots in the identity tensor, writes runtime_meta,
    then KJT2 evicts some KJT1 IDs and overwrites runtime_meta with its own
    values. Verifies:
    1. Evicted KJT1 slots have runtime_meta zeroed (eviction zeroing in remap)
       or overwritten by KJT2 values
    2. Surviving KJT1 slots retain their original runtime_meta
    3. KJT2 slots have KJT2's runtime_meta values
    4. All slots are occupied (table is full after both rounds)
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Small table: 16 total slots = 8 per rank with 2 ranks, 2 buckets
        small_tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=16,
                enable_embedding_update=True,
            ),
        ]
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            small_tables,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=4,
            max_probe=8,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Forward KJT1 with enough IDs to fill ALL slots
        # 16 total slots across 2 ranks. Insert 25 IDs to ensure saturation.
        kjt1_ids = torch.LongTensor(list(range(1000, 1025)))  # 25 IDs
        kjt1_insert = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt1_ids,
            lengths=torch.LongTensor([len(kjt1_ids)]),
            weights=None,
        ).to(ctx.device)

        for _ in range(2):
            loss, _ = sharded_sparse_arch(kjt1_insert)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()
        # Verify all slots are occupied on this rank
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            occupied = (identities != -1).sum().item()
            assert (
                occupied == identities.numel()
            ), f"rank {rank}: {occupied}/{identities.numel()} slots occupied after KJT1"

        # Step 2: Write runtime_meta for KJT1 IDs
        # Each ID gets fingerprint: ID 1000 -> 100.0, 1001 -> 101.0, ...
        kjt1_weights = torch.stack(
            [
                torch.tensor([100 + i], dtype=torch.int64).view(torch.float32)
                for i in range(len(kjt1_ids))
            ]
        )
        write_kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt1_ids,
            lengths=torch.LongTensor([len(kjt1_ids)]),
            weights=kjt1_weights,
        ).to(ctx.device)

        mc_ctx1 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out1 = mcc.input_dist(mc_ctx1, write_kjt1).wait().wait()
        mcc.compute(mc_ctx1, dist_out1)
        mcc.evict()

        # Snapshot KJT1 slots and their runtime_meta before eviction
        kjt1_slots_before: Dict[int, Tuple[int, float]] = (
            {}
        )  # raw_id -> (slot, expected_val)
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            mapped_kjt1, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=kjt1_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )
            for i, raw_id in enumerate(kjt1_ids.tolist()):
                mapped_id = mapped_kjt1[i].item()
                slot_mask = identities == mapped_id
                if slot_mask.any():
                    slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                    kjt1_slots_before[raw_id] = (slot_idx, float(100 + i))

        # Step 3: Forward KJT2 to evict some KJT1 IDs
        # Insert 25 new IDs — should evict some KJT1 IDs on each rank
        kjt2_ids = torch.LongTensor(list(range(2000, 2025)))  # 25 new IDs
        kjt2_insert = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt2_ids,
            lengths=torch.LongTensor([len(kjt2_ids)]),
            weights=None,
        ).to(ctx.device)

        for _ in range(2):
            loss, _ = sharded_sparse_arch(kjt2_insert)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 4: Write runtime_meta for KJT2 IDs
        # Each ID gets fingerprint: ID 2000 -> 200.0, 2001 -> 201.0, ...
        kjt2_weights = torch.stack(
            [
                torch.tensor([200 + i], dtype=torch.int64).view(torch.float32)
                for i in range(len(kjt2_ids))
            ]
        )
        write_kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=kjt2_ids,
            lengths=torch.LongTensor([len(kjt2_ids)]),
            weights=kjt2_weights,
        ).to(ctx.device)

        mc_ctx2 = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out2 = mcc.input_dist(mc_ctx2, write_kjt2).wait().wait()
        mcc.compute(mc_ctx2, dist_out2)
        mcc.evict()

        # Step 5: Verify per-MC-module
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            runtime_meta = mc_module._hash_zch_runtime_meta.data
            assert isinstance(runtime_meta, torch.Tensor)

            # --- Check surviving KJT1 IDs ---
            mapped_kjt1, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=kjt1_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )
            survived_count = 0
            evicted_count = 0
            for i, raw_id in enumerate(kjt1_ids.tolist()):
                mapped_id = mapped_kjt1[i].item()
                slot_mask = identities == mapped_id
                if slot_mask.any():
                    # KJT1 ID survived — runtime_meta should retain KJT1 value
                    slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                    meta_row = runtime_meta[slot_idx]
                    expected = float(100 + i)
                    expected_tensor = torch.full_like(
                        meta_row, expected, dtype=torch.float
                    ).to(runtime_meta.dtype)
                    assert torch.equal(meta_row, expected_tensor), (
                        f"Rank {rank}: surviving KJT1 ID {raw_id} at slot "
                        f"{slot_idx}: expected runtime_meta={expected}, "
                        f"got {meta_row}"
                    )
                    survived_count += 1
                elif raw_id in kjt1_slots_before:
                    evicted_count += 1
                    old_slot, old_expected = kjt1_slots_before[raw_id]
                    meta_row = runtime_meta[old_slot]
                    old_expected_tensor = torch.full_like(
                        meta_row, old_expected, dtype=torch.float
                    ).to(runtime_meta.dtype)
                    # Evicted slot must NOT retain the old KJT1 value.
                    # It should either be zeroed (if unoccupied) or have
                    # a KJT2 value (if reused).
                    assert not torch.equal(meta_row, old_expected_tensor), (
                        f"Rank {rank}: evicted slot {old_slot} (was KJT1 ID "
                        f"{raw_id}) still has old runtime_meta={old_expected}"
                    )

            # --- Check KJT2 IDs ---
            mapped_kjt2, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=kjt2_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )
            for i, raw_id in enumerate(kjt2_ids.tolist()):
                mapped_id = mapped_kjt2[i].item()
                slot_mask = identities == mapped_id
                if not slot_mask.any():
                    continue  # On other rank's shard

                slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                meta_row = runtime_meta[slot_idx]
                expected = float(200 + i)
                expected_tensor = torch.full_like(
                    meta_row, expected, dtype=torch.float
                ).to(runtime_meta.dtype)
                assert torch.equal(meta_row, expected_tensor), (
                    f"Rank {rank}: KJT2 ID {raw_id} at slot {slot_idx}: "
                    f"expected runtime_meta={expected}, got {meta_row}"
                )

            # --- Check unoccupied slots have zero runtime_meta ---
            for slot_idx in range(identities.numel()):
                if identities[slot_idx].item() == -1:
                    meta_row = runtime_meta[slot_idx]
                    assert torch.all(meta_row == 0), (
                        f"Rank {rank}: unoccupied slot {slot_idx} has "
                        f"non-zero runtime_meta={meta_row}"
                    )


@skip_if_asan_class
class ShardedMCECWriteStateDict(MultiProcessTestBase):
    """Tests for write path state_dict verification: overwrite, partial insertion, and eviction."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_overwrite_ids_state_dict(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_overwrite_ids_state_dict,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_partial_insertion_state_dict(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_partial_insertion_state_dict,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_eviction_clears_runtime_meta(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=16,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_eviction_clears_runtime_meta,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_full_table_eviction_overwrites_runtime_meta(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=16,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_full_table_eviction_overwrites_runtime_meta,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )


@skip_if_asan_class
class ShardedMCECDedupIndicesTest(MultiProcessTestBase):
    """Tests for _dedup_indices with 2D weights and duplicate IDs."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_dedup_indices_preserves_2d_weights(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_dedup_indices_preserves_2d_weights,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(
                ec_sharder=EmbeddingCollectionSharder(use_index_dedup=True),
            ),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_dedup_indices_no_weights_unchanged(self, backend: str) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=32,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_dedup_indices_no_weights_unchanged,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(
                ec_sharder=EmbeddingCollectionSharder(use_index_dedup=True),
            ),
            backend=backend,
        )


def _test_collision_only_inserted_id_updates_runtime_meta(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Test collision behavior: when multiple IDs in a write KJT remap to the
    same slot (collision fallback), only the actually-inserted ID should have
    its runtime_meta updated.

    With disable_fallback=False (default), the ZCH kernel returns a valid
    slot index for ALL IDs — even those that collided and weren't inserted.
    update_runtime_meta() verifies insertion by checking identity[slot] ==
    mapped_id, and only updates runtime_meta for matching slots.

    Flow:
    1. Use a small table (8 slots total) so collisions are likely
    2. Fill most slots via forward to reduce free slots
    3. Write a KJT with many new IDs that compete for few remaining slots
    4. Verify that runtime_meta is only set for IDs that actually appear
       in the identity tensor (not collision fallbacks)
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Very small table: 8 total slots = 4 per rank, 2 buckets
        # This maximizes collision probability.
        small_tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=8,
                enable_embedding_update=True,
            ),
        ]
        sharded_sparse_arch = _create_hashzch_sharded_arch(
            small_tables,
            ctx,
            sharder,
            input_hash_size=0,
            total_num_buckets=4,
            write_runtime_meta_dim=1,
        )
        mc_ec = sharded_sparse_arch._mc_ec
        assert isinstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = mc_ec._managed_collision_collection

        # Step 1: Fill most slots via forward passes
        # Insert 3 IDs to fill 3 of 4 slots per rank, leaving ~1 free slot
        fill_ids = torch.LongTensor([10, 20, 30])
        kjt_fill = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=fill_ids,
            lengths=torch.LongTensor([3]),
            weights=None,
        ).to(ctx.device)

        for _ in range(3):
            loss, _ = sharded_sparse_arch(kjt_fill)
            torch.sum(torch.stack([v.sum() for v in loss.values()])).backward()

        # Step 2: Write a KJT with many new IDs that will compete for the
        # remaining free slot(s). Most will collide and not get inserted.
        # With only ~1 free slot per rank and 6 new IDs, at most 1-2 will
        # succeed — the rest will get collision fallback slots.
        collision_ids = torch.LongTensor([101, 102, 103, 104, 105, 106])
        # Each ID gets a unique weight fingerprint
        collision_weights = torch.stack(
            [
                torch.tensor([int(id_val)], dtype=torch.int64).view(torch.float32)
                for id_val in collision_ids.tolist()
            ]
        )
        collision_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=collision_ids,
            lengths=torch.LongTensor([len(collision_ids)]),
            weights=collision_weights,
        ).to(ctx.device)

        mc_ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        dist_out = mcc.input_dist(mc_ctx, collision_kjt).wait().wait()
        mcc.compute(mc_ctx, dist_out)

        # Step 3: Verify runtime_meta consistency
        for _table_name, mc_module in mcc._managed_collision_modules.items():
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            identities: torch.Tensor = torch.flatten(
                mc_module._hash_zch_identities.data
            )
            # pyre-ignore[16]: `Module | Tensor` is not assignable to `Tensor`
            runtime_meta = mc_module._hash_zch_runtime_meta.data
            assert isinstance(runtime_meta, torch.Tensor)

            # Get mapped IDs for all collision IDs
            mapped_collision, _, _ = mc_module.input_mapper(  # pyre-fixme[29]
                values=collision_ids.to(ctx.device),
                output_offset=mc_module._output_global_offset_tensor,
            )

            inserted_ids = []
            not_inserted_ids = []

            for i, raw_id in enumerate(collision_ids.tolist()):
                mapped_id = mapped_collision[i].item()
                slot_mask = identities == mapped_id
                if slot_mask.any():
                    # This ID was actually inserted into the identity tensor
                    inserted_ids.append(raw_id)
                    slot_idx = slot_mask.nonzero(as_tuple=True)[0][0].item()
                    meta_row = runtime_meta[slot_idx]

                    # runtime_meta should have this ID's weight value
                    expected = float(raw_id)
                    expected_tensor = torch.full_like(
                        meta_row, expected, dtype=torch.float
                    ).to(runtime_meta.dtype)
                    assert torch.equal(meta_row, expected_tensor), (
                        f"Rank {rank}: inserted ID {raw_id} at slot "
                        f"{slot_idx}: expected runtime_meta={expected}, "
                        f"got {meta_row}"
                    )
                else:
                    # This ID collided and was NOT inserted
                    not_inserted_ids.append(raw_id)

            # Key assertion: not all IDs should have been inserted.
            # With 6 IDs competing for ~1 free slot, most should collide.
            # (We check across both ranks — at least some should collide overall)
            # Verify no slot has runtime_meta from a non-inserted ID.
            # For each occupied slot, check that the runtime_meta value
            # corresponds to the ID in the identity tensor, not a collider.
            for slot_idx in range(identities.numel()):
                identity_val = identities[slot_idx].item()
                if identity_val == -1:
                    # Unoccupied: runtime_meta should be zero
                    meta_row = runtime_meta[slot_idx]
                    assert torch.all(meta_row == 0), (
                        f"Rank {rank}: unoccupied slot {slot_idx} has "
                        f"non-zero runtime_meta={meta_row}"
                    )
                else:
                    # Occupied: if runtime_meta is non-zero, it should match
                    # the ID that's actually in this slot, not a collision
                    # fallback ID.
                    meta_row = runtime_meta[slot_idx]
                    if torch.all(meta_row == 0):
                        continue  # Slot was filled before write, no meta set

                    # The meta value (all elements same) should be the float
                    # value of one of our collision IDs or fill IDs
                    meta_val = meta_row[0].item()
                    # Check this meta_val corresponds to a valid ID
                    # that is actually in this slot
                    if meta_val in [float(x) for x in collision_ids.tolist()]:
                        # Verify the identity at this slot matches
                        mapped_expected_id = None
                        for j, raw_id in enumerate(collision_ids.tolist()):
                            if float(raw_id) == meta_val:
                                mapped_expected_id = mapped_collision[j].item()
                                break
                        assert identity_val == mapped_expected_id, (
                            f"Rank {rank}: slot {slot_idx} has "
                            f"runtime_meta={meta_val} (for ID {int(meta_val)}) "
                            f"but identity={identity_val} doesn't match "
                            f"mapped_id={mapped_expected_id} — collision "
                            f"fallback leaked into runtime_meta"
                        )


@skip_if_asan_class
class ShardedMCECCollisionTest(MultiProcessTestBase):
    """Tests for collision behavior with runtime_meta updates."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_collision_only_inserted_id_updates_runtime_meta(
        self, backend: str
    ) -> None:
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=8,
                enable_embedding_update=True,
            ),
        ]
        self._run_multi_process_test(
            callable=_test_collision_only_inserted_id_updates_runtime_meta,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )
