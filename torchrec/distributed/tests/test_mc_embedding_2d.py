#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.mc_embedding import ManagedCollisionEmbeddingCollectionSharder
from torchrec.distributed.mc_embedding_modules import (
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.tests.test_mc_embedding_utils import SparseArch
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.test_utils import skip_if_asan_class
from torchrec.types import DataType


def _test_2d_mc_sharding_syncing_identical_identities_and_tables(  # noqa: C901
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    world_size_2D: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    kjt_input_per_rank: List[torch.Tensor],
    local_size: Optional[int] = None,
    use_inter_host_allreduce: bool = False,
    apply_optimizer_in_backward_config: Optional[
        Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
    ] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        num_tables = 2
        num_replica = world_size // world_size_2D

        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            use_mpzch=True,
            input_hash_size=80,
        )

        if apply_optimizer_in_backward_config is not None:
            for apply_optim_name, (
                optimizer_type,
                optimizer_kwargs,
            ) in apply_optimizer_in_backward_config.items():
                for name, param in sparse_arch.named_parameters():
                    if apply_optim_name not in name:
                        continue
                    apply_optimizer_in_backward(
                        optimizer_type,
                        [param],
                        optimizer_kwargs,
                    )

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size_2D,
                compute_device=ctx.device.type,
                local_world_size=None,
            ),
        )
        plan = planner.collective_plan(sparse_arch, [sharder], ctx.pg)
        assert ctx.pg is not None

        dmp = DMPCollection(
            module=sparse_arch,
            device=ctx.device,
            plan=plan,
            sharding_group_size=world_size_2D,
            world_size=world_size,
            global_pg=ctx.pg,
            sharders=[sharder],
            use_inter_host_allreduce=use_inter_host_allreduce,
        )

        # Test that "HashZch" modules are in `modules_to_sync`
        for module, _ in dmp._ctxs[0].modules_to_sync:
            # The first module is SplitTable, but the second/third should be HashZch
            if type(module).__name__ != "SplitTableBatchedEmbeddingBagsCodegen":
                assert isinstance(
                    module, BaseShardedManagedCollisionEmbeddingCollection
                ), f"Modules {module} should be 'BaseShardedManagedCollisionEmbeddingCollection'"

        # Test that hash_identities and hash_metadata is inside the context
        assert hasattr(dmp._ctxs[0], "hash_zch_modules")
        assert len(dmp._ctxs[0].hash_zch_modules) == num_tables

        # pyrefly: ignore[missing-attribute]
        mc_collection = dmp.module._mc_ec._managed_collision_collection  # ShardedMCC

        # Test that sharding metadata is identical across replicas, this is created from
        #   _create_managed_collision_modules
        metadata = torch.tensor(
            [
                mc_collection._mc_module_name_shard_metadata[f"table_{i_table}"]
                for i_table in range(num_tables)
            ],
            device=ctx.device,
            dtype=torch.int32,
        )
        other_metadata = [torch.empty_like(metadata) for _ in range(num_replica)]
        dist.all_gather(
            other_metadata,
            metadata,
            group=dmp._default_ctx.replica_pg,
        )
        for m in other_metadata[1:]:
            assert torch.equal(other_metadata[0], m)

        # Do a forward pass
        kjt_input = kjt_input_per_rank[rank]
        loss, _ = dmp(kjt_input.to(ctx.device))
        loss.backward()

        # Randomize the metadata to get realistic scenario
        for t in ["table_0", "table_1"]:
            m = mc_collection._managed_collision_modules[t]._hash_zch_metadata
            m.copy_(torch.randint_like(m, low=10, high=100))

        # Perform syncing
        dmp.sync()

        # Test that identities/metadata is identical across replicas
        for i_table in range(num_tables):
            # Grab the sharded tensors for this rank
            identities = (
                mc_collection._model_parallel_mc_buffer_name_to_sharded_tensor[
                    f"_managed_collision_modules.table_{i_table}._hash_zch_identities"
                ]
                ._local_shards[0]
                .tensor
            )
            metadata = (
                mc_collection._model_parallel_mc_buffer_name_to_sharded_tensor[
                    f"_managed_collision_modules.table_{i_table}._hash_zch_metadata"
                ]
                ._local_shards[0]
                .tensor
            )

            # Assert that it is unique elements after removing -1, tests buckets/local_sizes
            torch.testing.assert_close(
                torch.unique(identities[identities != -1]).numel(),
                identities[identities != -1].numel(),
            )

            # Grab the other sharded tensors from other ranks
            identities_other_ranks = [
                torch.empty_like(identities) for _ in range(num_replica)
            ]
            dist.all_gather(
                identities_other_ranks,
                identities,
                group=dmp._default_ctx.replica_pg,
            )
            metadata_other_ranks = [
                torch.empty_like(metadata) for _ in range(num_replica)
            ]
            dist.all_gather(
                metadata_other_ranks,
                metadata,
                group=dmp._default_ctx.replica_pg,
            )

            # Make sure identities/metadata are identical across ranks after syncing
            for i in range(1, num_replica):
                torch.testing.assert_allclose(
                    identities_other_ranks[0], identities_other_ranks[i]
                )
                torch.testing.assert_allclose(
                    metadata_other_ranks[0], metadata_other_ranks[i]
                )

            # Test that embedding table are equal too.
            # pyrefly: ignore[missing-attribute]
            tbe, table_id = dmp.module._mc_ec._table_to_tbe_and_index[
                f"table_{i_table}"
            ]
            emb_table = tbe.split_embedding_weights()[table_id.item()]
            embedding_tables_other_ranks = [
                torch.empty_like(emb_table) for _ in range(num_replica)
            ]
            dist.all_gather(
                embedding_tables_other_ranks,
                emb_table,
                group=dmp._default_ctx.replica_pg,
            )
            for rank_emb in embedding_tables_other_ranks[1:]:
                torch.testing.assert_allclose(rank_emb, embedding_tables_other_ranks[0])

            # Test optimizers are equal too.
            optim = tbe.get_optimizer_state()
            if len(optim) > 0:
                optim = optim[table_id.item()]["sum"]
                optimizers_other_ranks = [
                    torch.empty_like(optim) for _ in range(num_replica)
                ]
                dist.all_gather(
                    optimizers_other_ranks, optim, group=dmp._default_ctx.replica_pg
                )
                for rank_optim in optimizers_other_ranks[1:]:
                    torch.testing.assert_allclose(rank_optim, optimizers_other_ranks[0])


def _test_2d_mc_syncing_preserves_top_indices(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    world_size_2D: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    kjt_input_per_rank: List[torch.Tensor],
    local_size: Optional[int] = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            use_mpzch=True,
            input_hash_size=80,
        )
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size_2D,
                compute_device=ctx.device.type,
                local_world_size=None,
            ),
        )
        assert ctx.pg is not None

        dmp = DMPCollection(
            module=sparse_arch,
            device=ctx.device,
            plan=planner.collective_plan(sparse_arch, [sharder], ctx.pg),
            sharding_group_size=world_size_2D,
            world_size=world_size,
            global_pg=ctx.pg,
            sharders=[sharder],
        )

        # pyrefly: ignore[missing-attribute]
        mc_collection = dmp.module._mc_ec._managed_collision_collection

        kjt_input = kjt_input_per_rank[rank]
        loss, _ = dmp(kjt_input.to(ctx.device))
        loss.backward()

        # Randomize metadata to simulate realistic situation
        for t in ["table_0", "table_1"]:
            m = mc_collection._managed_collision_modules[t]._hash_zch_metadata
            m.copy_(torch.randint_like(m, low=10, high=100))

        # Find root of this replica group
        mesh = dmp._default_ctx.device_mesh.mesh
        is_root_node = rank in mesh[0]
        index_root = torch.where(mesh == rank)
        replica_ranks = mesh[:, index_root[1][0]].tolist()

        # Force high metadata only on the root node
        # it survives the merge (they have the highest metadata values)
        indices_preserved = {}
        if is_root_node:
            for t in ["table_0", "table_1"]:
                model = mc_collection._managed_collision_modules[t]
                m = model._hash_zch_metadata
                # This is higher than original metadata values of 100
                m = mc_collection._managed_collision_modules[t]._hash_zch_metadata
                m.copy_(torch.randint_like(m, low=101, high=200))
                indices_preserved[t] = model._hash_zch_identities.clone()

        dmp.sync()

        # Send preserved indices from root to all replica
        for t in ["table_0", "table_1"]:
            model = mc_collection._managed_collision_modules[t]
            preserved = torch.empty_like(model._hash_zch_identities)
            if is_root_node:
                preserved = indices_preserved[t]

            dist.broadcast(
                preserved,
                src=replica_ranks[0],
                group=dmp._default_ctx.replica_pg,
            )

            # Verify all elements match replica 0 after sync
            final_identities = model._hash_zch_identities.squeeze()
            preserved = preserved.squeeze()
            nonempty = preserved != -1
            torch.testing.assert_close(preserved[nonempty], final_identities[nonempty])


@skip_if_asan_class
class ShardedMCECWith2DSharding(MultiProcessTestBase):
    """Tests for 2D parallelism of MCEC tables"""

    WORLD_SIZE = 8

    def setUp(self) -> None:
        super().setUp()

        self.embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=80,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=60,
                data_type=DataType.FP16,
            ),
        ]

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least 8 GPUs",
    )
    # pyre-ignore
    @given(
        backend=st.sampled_from(["nccl"]),
        use_inter_host_allreduce=st.booleans(),
        world_size_2D=st.sampled_from([2, 4]),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embeddings": (torch.optim.Adagrad, {"lr": 0.2}),
                },
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_2d_mc_zch_sharding_syncing_is_correct(
        self,
        backend: str,
        use_inter_host_allreduce: bool,
        world_size_2D: int,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        _, local_inputs = ModelInput.generate(
            batch_size=40,
            world_size=self.WORLD_SIZE,
            num_float_features=0,
            tables=self.embedding_config,
            weighted_tables=[],
            pooling_avg=5,
            random_seed=100,
        )
        # Extract global and local KJT from ModelInput
        kjt_input_per_rank = [mi.idlist_features for mi in local_inputs]

        self._run_multi_process_test(
            callable=_test_2d_mc_sharding_syncing_identical_identities_and_tables,
            world_size=self.WORLD_SIZE,
            tables=self.embedding_config,
            world_size_2D=world_size_2D,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
            kjt_input_per_rank=kjt_input_per_rank,
            use_inter_host_allreduce=use_inter_host_allreduce,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least 8 GPUs",
    )
    @given(
        backend=st.sampled_from(["nccl"]),
        world_size_2D=st.sampled_from([2, 4]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_2d_mc_zch_syncing_preserves_top_indices(
        self,
        backend: str,
        world_size_2D: int,
    ) -> None:
        """One rank has all largest metadata, and so checks it didn't get evicted."""
        _, local_inputs = ModelInput.generate(
            batch_size=40,
            world_size=self.WORLD_SIZE,
            num_float_features=0,
            tables=self.embedding_config,
            weighted_tables=[],
            pooling_avg=5,
            random_seed=100,
        )
        kjt_input_per_rank = [mi.idlist_features for mi in local_inputs]

        self._run_multi_process_test(
            callable=_test_2d_mc_syncing_preserves_top_indices,
            world_size=self.WORLD_SIZE,
            tables=self.embedding_config,
            world_size_2D=world_size_2D,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
            kjt_input_per_rank=kjt_input_per_rank,
        )
