#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.mc_embedding import (
    KJTList,
    ManagedCollisionEmbeddingCollectionSharder,
    ShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_embedding_modules import (
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import (
    ManagedCollisionCollectionContext,
    ShardedManagedCollisionCollection,
)
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    EmbeddingCollectionSharder,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
)
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
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class
from torchrec.types import DataType


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        return_remapped: bool = False,
        input_hash_size: int = 4000,
        allow_in_place_embed_weight_update: bool = False,
        use_mpzch: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        mc_modules: dict[str, ManagedCollisionModule] = {}
        if use_mpzch:
            # Parameters hard-coded from test_quant_mc_embedding
            mc_modules["table_0"] = HashZchManagedCollisionModule(
                zch_size=(tables[0].num_embeddings),
                input_hash_size=input_hash_size,
                device=device,
                total_num_buckets=4,
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_0"],
                    single_ttl=1,
                ),
                max_probe=5,
            )

            mc_modules["table_1"] = HashZchManagedCollisionModule(
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=input_hash_size,
                total_num_buckets=4,
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_1"],
                    single_ttl=1,
                ),
                max_probe=5,
            )
        else:
            mc_modules["table_0"] = MCHManagedCollisionModule(
                zch_size=(tables[0].num_embeddings),
                input_hash_size=input_hash_size,
                device=device,
                eviction_interval=2,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )
            mc_modules["table_1"] = MCHManagedCollisionModule(
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=input_hash_size,
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
                    # pyrefly: ignore[bad-argument-type]
                    managed_collision_modules=mc_modules,
                    embedding_configs=tables,
                ),
                return_remapped_features=self._return_remapped,
                allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, JaggedTensor]]]:
        ec_out, remapped_ids_out = self._mc_ec(kjt)
        pred = torch.cat(
            [ec_out[key].values() for key in ["feature_0", "feature_1"]],
            dim=0,
        )
        loss = pred.mean()
        return loss, remapped_ids_out


def _test_sharding_and_remapping(  # noqa C901
    output_keys: List[str],
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]],
    initial_state_per_rank: List[Dict[str, torch.Tensor]],
    final_state_per_rank: List[Dict[str, torch.Tensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
    input_hash_size: int = 4000,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        kjt_out_per_iter = [
            kjt[rank].to(ctx.device) for kjt in kjt_out_per_iter_per_rank
        ]
        return_remapped: bool = True
        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            return_remapped=return_remapped,
            input_hash_size=input_hash_size,
        )

        apply_optimizer_in_backward(
            RowWiseAdagrad,
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
            [
                sparse_arch._mc_ec._embedding_collection.embeddings["table_0"].weight,
                sparse_arch._mc_ec._embedding_collection.embeddings["table_1"].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ec,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ec, ShardedManagedCollisionEmbeddingCollection
        )
        assert isinstance(
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )
        assert (
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection._has_uninitialized_input_dist
            is False
        )
        assert (
            not hasattr(
                #  attribute `_embedding_collection`.
                sharded_sparse_arch._mc_ec._embedding_collection,
                "_input_dists",
            )
            #  `_embedding_collection`.
            or len(sharded_sparse_arch._mc_ec._embedding_collection._input_dists) == 0
        )

        assert isinstance(
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )

        assert (
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            #  `_embedding_collection`.
            == sharded_sparse_arch._mc_ec._embedding_collection._use_index_dedup
        )

        initial_state_dict = sharded_sparse_arch.state_dict()
        for key, sharded_tensor in initial_state_dict.items():
            postfix = ".".join(key.split(".")[-2:])
            if postfix in initial_state_per_rank[ctx.rank]:
                tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                torch.testing.assert_close(
                    tensor,
                    initial_state_per_rank[ctx.rank][postfix],
                    rtol=0,
                    atol=0,
                )

        sharded_sparse_arch.load_state_dict(initial_state_dict)

        # sharded model
        # each rank gets a subbatch
        loss1, remapped_ids1 = sharded_sparse_arch(kjt_input)
        loss1.backward()
        loss2, remapped_ids2 = sharded_sparse_arch(kjt_input)
        loss2.backward()

        final_state_dict = sharded_sparse_arch.state_dict()
        for key, sharded_tensor in final_state_dict.items():
            postfix = ".".join(key.split(".")[-2:])
            if postfix in final_state_per_rank[ctx.rank]:
                tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                torch.testing.assert_close(
                    tensor,
                    final_state_per_rank[ctx.rank][postfix],
                    rtol=0,
                    atol=0,
                )

        remapped_ids = [remapped_ids1, remapped_ids2]
        for key in output_keys:
            for i, kjt_out in enumerate(kjt_out_per_iter):
                torch.testing.assert_close(
                    remapped_ids[i][key].values(),
                    kjt_out[key].values(),
                    rtol=0,
                    atol=0,
                )

        # TODO: validate embedding rows, and eviction


def _test_in_place_embd_weight_update(  # noqa C901
    output_keys: List[str],
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]],
    initial_state_per_rank: List[Dict[str, torch.Tensor]],
    final_state_per_rank: List[Dict[str, torch.Tensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
    input_hash_size: int = 4000,
    allow_in_place_embed_weight_update: bool = True,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        kjt_out_per_iter = [
            kjt[rank].to(ctx.device) for kjt in kjt_out_per_iter_per_rank
        ]
        return_remapped: bool = True
        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            return_remapped=return_remapped,
            input_hash_size=input_hash_size,
            allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
        )
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
            [
                sparse_arch._mc_ec._embedding_collection.embeddings["table_0"].weight,
                sparse_arch._mc_ec._embedding_collection.embeddings["table_1"].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ec,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        initial_state_dict = sharded_sparse_arch.state_dict()
        for key, sharded_tensor in initial_state_dict.items():
            postfix = ".".join(key.split(".")[-2:])
            if postfix in initial_state_per_rank[ctx.rank]:
                tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                torch.testing.assert_close(
                    tensor,
                    initial_state_per_rank[ctx.rank][postfix],
                    rtol=0,
                    atol=0,
                )

        sharded_sparse_arch.load_state_dict(initial_state_dict)

        # sharded model
        # each rank gets a subbatch
        loss1, remapped_ids1 = sharded_sparse_arch(kjt_input)
        loss2, remapped_ids2 = sharded_sparse_arch(kjt_input)

        if not allow_in_place_embed_weight_update:
            # Without in-place overwrite the backward pass will fail due to tensor version mismatch
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError,
                "one of the variables needed for gradient computation has been modified by an inplace operation",
            ):
                loss1.backward()
        else:
            loss1.backward()
            loss2.backward()
            final_state_dict = sharded_sparse_arch.state_dict()
            for key, sharded_tensor in final_state_dict.items():
                postfix = ".".join(key.split(".")[-2:])
                if postfix in final_state_per_rank[ctx.rank]:
                    tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                    torch.testing.assert_close(
                        tensor,
                        final_state_per_rank[ctx.rank][postfix],
                        rtol=0,
                        atol=0,
                    )

            remapped_ids = [remapped_ids1, remapped_ids2]
            for key in output_keys:
                for i, kjt_out in enumerate(kjt_out_per_iter):
                    torch.testing.assert_close(
                        remapped_ids[i][key].values(),
                        kjt_out[key].values(),
                        rtol=0,
                        atol=0,
                    )


def _test_sharding_and_resharding(  # noqa C901
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]],
    initial_state_per_rank: List[Dict[str, torch.Tensor]],
    final_state_per_rank: List[Dict[str, torch.Tensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:

        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        kjt_out_per_iter = [
            kjt[rank].to(ctx.device) for kjt in kjt_out_per_iter_per_rank
        ]
        return_remapped: bool = True
        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            return_remapped=return_remapped,
        )

        apply_optimizer_in_backward(
            RowWiseAdagrad,
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
            [
                sparse_arch._mc_ec._embedding_collection.embeddings["table_0"].weight,
                sparse_arch._mc_ec._embedding_collection.embeddings["table_1"].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ec,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ec, ShardedManagedCollisionEmbeddingCollection
        )
        assert isinstance(
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )
        assert (
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection._has_uninitialized_input_dist
            is False
        )
        assert (
            not hasattr(
                #  attribute `_embedding_collection`.
                sharded_sparse_arch._mc_ec._embedding_collection,
                "_input_dists",
            )
            #  `_embedding_collection`.
            or len(sharded_sparse_arch._mc_ec._embedding_collection._input_dists) == 0
        )

        assert isinstance(
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )
        # sharded model
        # each rank gets a subbatch
        loss1, remapped_ids1 = sharded_sparse_arch(kjt_input)
        loss1.backward()
        loss2, remapped_ids2 = sharded_sparse_arch(kjt_input)
        loss2.backward()
        remapped_ids = [remapped_ids1, remapped_ids2]
        for key in kjt_input.keys():
            for i, kjt_out in enumerate(kjt_out_per_iter[:2]):  # first two iterations
                torch.testing.assert_close(
                    remapped_ids[i][key].values(),
                    kjt_out[key].values(),
                    rtol=0,
                    atol=0,
                )

        state_dict = sharded_sparse_arch.state_dict()
        cpu_state_dict = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, ShardedTensor):
                tensor = tensor.local_shards()[0].tensor
            cpu_state_dict[key] = tensor.to("cpu")
        gather_list = [None, None] if ctx.rank == 0 else None
        torch.distributed.gather_object(cpu_state_dict, gather_list)

    if rank == 0:
        with MultiProcessContext(rank, 1, backend, 1) as ctx:
            kjt_input = kjt_input_per_rank[rank].to(ctx.device)
            sparse_arch = SparseArch(
                tables,
                torch.device("meta"),
                return_remapped=return_remapped,
            )

            apply_optimizer_in_backward(
                RowWiseAdagrad,
                #  got `Iterable[Union[Module, Tensor]]`.
                # pyrefly: ignore[bad-argument-type]
                [
                    sparse_arch._mc_ec._embedding_collection.embeddings[
                        "table_0"
                    ].weight,
                    sparse_arch._mc_ec._embedding_collection.embeddings[
                        "table_1"
                    ].weight,
                ],
                {"lr": 0.01},
            )
            module_sharding_plan = construct_module_sharding_plan(
                sparse_arch._mc_ec,
                per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
                local_size=1,
                world_size=1,
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                sharder=sharder,
            )

            sharded_sparse_arch = _shard_modules(
                module=copy.deepcopy(sparse_arch),
                plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
                #  `Optional[ProcessGroup]`.
                # pyrefly: ignore[bad-argument-type]
                env=ShardingEnv.from_process_group(ctx.pg),
                sharders=[sharder],
                device=ctx.device,
            )
            state_dict = sharded_sparse_arch.state_dict()

            for key in state_dict.keys():
                if isinstance(state_dict[key], ShardedTensor):
                    replacement_tensor = torch.cat(
                        # pyrefly: ignore[unsupported-operation]
                        [gather_list[0][key], gather_list[1][key]],
                        dim=0,
                    ).to(ctx.device)
                    state_dict[key].local_shards()[0].tensor.copy_(replacement_tensor)
                else:
                    # pyrefly: ignore[unsupported-operation]
                    state_dict[key] = gather_list[0][key].to(ctx.device)

            sharded_sparse_arch.load_state_dict(state_dict)
            loss3, remapped_ids3 = sharded_sparse_arch(kjt_input)
            final_state_dict = sharded_sparse_arch.state_dict()
            for key, sharded_tensor in final_state_dict.items():
                postfix = ".".join(key.split(".")[-2:])
                if postfix in final_state_per_rank[ctx.rank]:
                    tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                    torch.testing.assert_close(
                        tensor,
                        final_state_per_rank[ctx.rank][postfix],
                        rtol=0,
                        atol=0,
                    )

            remapped_ids = [remapped_ids3]
            for key in kjt_input.keys():
                for i, kjt_out in enumerate(kjt_out_per_iter[-1:]):  # last iteration
                    torch.testing.assert_close(
                        remapped_ids[i][key].values(),
                        kjt_out[key].values(),
                        rtol=0,
                        atol=0,
                    )


def _test_sharding_dedup(  # noqa C901
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    dedup_sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
    input_hash_size: int = 4000,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        return_remapped: bool = True
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            return_remapped=return_remapped,
            input_hash_size=input_hash_size,
        )
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
            [
                sparse_arch._mc_ec._embedding_collection.embeddings["table_0"].weight,
                sparse_arch._mc_ec._embedding_collection.embeddings["table_1"].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ec,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )
        dedup_sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[dedup_sharder],
            device=ctx.device,
        )

        assert (
            #  `_managed_collision_collection`.
            # pyrefly: ignore[missing-attribute]
            sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            #  `_embedding_collection`.
            # pyrefly: ignore[missing-attribute]
            == sharded_sparse_arch._mc_ec._embedding_collection._use_index_dedup
        )

        assert (
            #  `_managed_collision_collection`.
            # pyrefly: ignore[missing-attribute]
            sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            is False
        )

        assert (
            #  `_managed_collision_collection`.
            # pyrefly: ignore[missing-attribute]
            dedup_sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            #  `_embedding_collection`.
            # pyrefly: ignore[missing-attribute]
            == dedup_sharded_sparse_arch._mc_ec._embedding_collection._use_index_dedup
        )

        assert (
            #  `_managed_collision_collection`.
            # pyrefly: ignore[missing-attribute]
            dedup_sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            is True
        )

        # sync state_dict()
        state_dict = sharded_sparse_arch.state_dict()
        dedup_state_dict = dedup_sharded_sparse_arch.state_dict()
        for key, sharded_tensor in state_dict.items():
            if isinstance(sharded_tensor, ShardedTensor):
                dedup_state_dict[key].local_shards()[
                    0
                ].tensor = sharded_tensor.local_shards()[0].tensor.clone()
            dedup_state_dict[key] = sharded_tensor.clone()
        dedup_sharded_sparse_arch.load_state_dict(dedup_state_dict)

        loss1, remapped_1 = sharded_sparse_arch(kjt_input)
        loss1.backward()
        dedup_loss1, dedup_remapped_1 = dedup_sharded_sparse_arch(kjt_input)
        dedup_loss1.backward()

        torch.testing.assert_close(loss1, dedup_loss1, rtol=1e-05, atol=1e-08)
        # deduping is not being used right now
        # assert torch.allclose(remapped_1.values(), dedup_remapped_1.values())
        # assert torch.allclose(remapped_1.lengths(), dedup_remapped_1.lengths())


def _test_2d_mc_sharding_syncing_identical_identities_and_tables(
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
) -> None:  # noqa: C901
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
class ShardedMCEmbeddingCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_mc_ec_reshard(self, backend: str) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    [1000, 2000, 1001, 2000, 2001, 2002],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    [
                        1000,
                        1002,
                        1004,
                        2000,
                        2002,
                        2004,
                    ],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
        ]

        kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]] = []
        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [7, 15, 7, 31, 31, 31],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [7, 7, 7, 31, 31, 31],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
            ]
        )
        # TODO: cleanup sorting so more dedugable/logical initial fill

        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 14, 4, 27, 29, 28],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 5, 6, 27, 28, 30],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
            ]
        )

        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 14, 4, 27, 29, 28],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.empty(),
            ]
        )

        max_int = torch.iinfo(torch.int64).max

        final_state_per_rank = [
            {
                "table_0._mch_sorted_raw_ids": torch.LongTensor(
                    [1000, 1001, 1002, 1004, 2000] + [max_int] * (16 - 5)
                ),
                "table_1._mch_sorted_raw_ids": torch.LongTensor(
                    [2000, 2001, 2002, 2004] + [max_int] * (32 - 4)
                ),
                "table_0._mch_remapped_ids_mapping": torch.LongTensor(
                    [3, 4, 5, 6, 14, 0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 15],
                ),
                "table_1._mch_remapped_ids_mapping": torch.LongTensor(
                    [
                        27,
                        29,
                        28,
                        30,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        31,
                    ],
                ),
            },
        ]

        self._run_multi_process_test(
            callable=_test_sharding_and_resharding,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            kjt_out_per_iter_per_rank=kjt_out_per_iter_per_rank,
            initial_state_per_rank=None,
            final_state_per_rank=final_state_per_rank,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_mc_ec_remap(self, backend: str) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [1000, 2000, 1001, 2000, 2001, 2002, 1, 1, 1],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [
                        1000,
                        1002,
                        1004,
                        2000,
                        2002,
                        2004,
                        2,
                        2,
                        2,
                    ],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
        ]

        kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]] = []
        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [7, 15, 7, 31, 31, 31],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [7, 7, 7, 31, 31, 31],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
            ]
        )
        # TODO: cleanup sorting so more dedugable/logical initial fill

        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 14, 4, 27, 29, 28],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 5, 6, 27, 28, 30],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
            ]
        )

        initial_state_per_rank = [
            {
                "table_0._mch_remapped_ids_mapping": torch.arange(8, dtype=torch.int64),
                "table_1._mch_remapped_ids_mapping": torch.arange(
                    16, dtype=torch.int64
                ),
            },
            {
                "table_0._mch_remapped_ids_mapping": torch.arange(
                    start=8, end=16, dtype=torch.int64
                ),
                "table_1._mch_remapped_ids_mapping": torch.arange(
                    start=16, end=32, dtype=torch.int64
                ),
            },
        ]
        max_int = torch.iinfo(torch.int64).max

        final_state_per_rank = [
            {
                "table_0._mch_sorted_raw_ids": torch.LongTensor(
                    [1000, 1001, 1002, 1004] + [max_int] * 4
                ),
                "table_1._mch_sorted_raw_ids": torch.LongTensor([max_int] * 16),
                "table_0._mch_remapped_ids_mapping": torch.LongTensor(
                    [3, 4, 5, 6, 0, 1, 2, 7]
                ),
                "table_1._mch_remapped_ids_mapping": torch.arange(
                    16, dtype=torch.int64
                ),
            },
            {
                "table_0._mch_sorted_raw_ids": torch.LongTensor([2000] + [max_int] * 7),
                "table_1._mch_sorted_raw_ids": torch.LongTensor(
                    [2000, 2001, 2002, 2004] + [max_int] * 12
                ),
                "table_0._mch_remapped_ids_mapping": torch.LongTensor(
                    [14, 8, 9, 10, 11, 12, 13, 15]
                ),
                "table_1._mch_remapped_ids_mapping": torch.LongTensor(
                    [27, 29, 28, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31]
                ),
            },
        ]

        self._run_multi_process_test(
            callable=_test_sharding_and_remapping,
            output_keys=["feature_0", "feature_1"],
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            kjt_out_per_iter_per_rank=kjt_out_per_iter_per_rank,
            initial_state_per_rank=initial_state_per_rank,
            final_state_per_rank=final_state_per_rank,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_mc_ec_dedup(self, backend: str) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0", "feature_2"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [1000, 1000, 2000, 1001, 1000, 2001, 2002, 3000, 2000, 1000],
                ),
                lengths=torch.LongTensor([2, 1, 1, 1, 1, 1, 2, 0, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [
                        1002,
                        1002,
                        1004,
                        2000,
                        1002,
                        2004,
                        3999,
                        2000,
                        2000,
                    ],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 3]),
                weights=None,
            ),
        ]

        self._run_multi_process_test(
            callable=_test_sharding_dedup,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=ManagedCollisionEmbeddingCollectionSharder(
                ec_sharder=EmbeddingCollectionSharder(
                    use_index_dedup=False,
                )
            ),
            dedup_sharder=ManagedCollisionEmbeddingCollectionSharder(
                ec_sharder=EmbeddingCollectionSharder(
                    use_index_dedup=True,
                )
            ),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_mc_ec_dedup_input_error(self, backend: str) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0", "feature_2"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [1000, 1000, 2000, 1001, 1000, 2001, 2002, 3000, 2000, 1000],
                ),
                lengths=torch.LongTensor([2, 1, 1, 1, 1, 1, 2, 0, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [
                        1002,
                        1002,
                        1004,
                        2000,
                        1002,
                        2004,
                        3999,
                        2000,
                        2000,
                    ],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 3]),
                weights=None,
            ),
        ]

        try:
            self._run_multi_process_test(
                callable=_test_sharding_dedup,
                world_size=WORLD_SIZE,
                tables=embedding_config,
                kjt_input_per_rank=kjt_input_per_rank,
                sharder=ManagedCollisionEmbeddingCollectionSharder(
                    ec_sharder=EmbeddingCollectionSharder(
                        use_index_dedup=False,
                    )
                ),
                dedup_sharder=ManagedCollisionEmbeddingCollectionSharder(
                    ec_sharder=EmbeddingCollectionSharder(
                        use_index_dedup=True,
                    )
                ),
                backend=backend,
                input_hash_size=(2**52) - 1 + 10,
            ),
        except AssertionError as e:
            self.assertTrue("0 != 1" in str(e))

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        backend=st.sampled_from(["nccl"]),
        allow_in_place_embed_weight_update=st.booleans(),
    )
    @settings(deadline=None)
    def test_in_place_embd_weight_update(
        self, backend: str, allow_in_place_embed_weight_update: bool
    ) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [1000, 2000, 1001, 2000, 2001, 2002, 1, 1, 1],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor(
                    [
                        1000,
                        1002,
                        1004,
                        2000,
                        2002,
                        2004,
                        2,
                        2,
                        2,
                    ],
                ),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
        ]

        kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]] = []
        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [7, 15, 7, 31, 31, 31],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [7, 7, 7, 31, 31, 31],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
            ]
        )
        # TODO: cleanup sorting so more dedugable/logical initial fill

        kjt_out_per_iter_per_rank.append(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 14, 4, 27, 29, 28],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor(
                        [3, 5, 6, 27, 28, 30],
                    ),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                    weights=None,
                ),
            ]
        )

        initial_state_per_rank = [
            {
                "table_0._mch_remapped_ids_mapping": torch.arange(8, dtype=torch.int64),
                "table_1._mch_remapped_ids_mapping": torch.arange(
                    16, dtype=torch.int64
                ),
            },
            {
                "table_0._mch_remapped_ids_mapping": torch.arange(
                    start=8, end=16, dtype=torch.int64
                ),
                "table_1._mch_remapped_ids_mapping": torch.arange(
                    start=16, end=32, dtype=torch.int64
                ),
            },
        ]
        max_int = torch.iinfo(torch.int64).max

        final_state_per_rank = [
            {
                "table_0._mch_sorted_raw_ids": torch.LongTensor(
                    [1000, 1001, 1002, 1004] + [max_int] * 4
                ),
                "table_1._mch_sorted_raw_ids": torch.LongTensor([max_int] * 16),
                "table_0._mch_remapped_ids_mapping": torch.LongTensor(
                    [3, 4, 5, 6, 0, 1, 2, 7]
                ),
                "table_1._mch_remapped_ids_mapping": torch.arange(
                    16, dtype=torch.int64
                ),
            },
            {
                "table_0._mch_sorted_raw_ids": torch.LongTensor([2000] + [max_int] * 7),
                "table_1._mch_sorted_raw_ids": torch.LongTensor(
                    [2000, 2001, 2002, 2004] + [max_int] * 12
                ),
                "table_0._mch_remapped_ids_mapping": torch.LongTensor(
                    [14, 8, 9, 10, 11, 12, 13, 15]
                ),
                "table_1._mch_remapped_ids_mapping": torch.LongTensor(
                    [27, 29, 28, 30, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31]
                ),
            },
        ]

        self._run_multi_process_test(
            callable=_test_in_place_embd_weight_update,
            output_keys=["feature_0", "feature_1"],
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            kjt_out_per_iter_per_rank=kjt_out_per_iter_per_rank,
            initial_state_per_rank=initial_state_per_rank,
            final_state_per_rank=final_state_per_rank,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
            allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
        )


class ComputeOutputLengthTest(unittest.TestCase):
    """Test that ShardedManagedCollisionCollection.compute uses
    mc_module output lengths rather than input feature lengths."""

    def test_compute_single_table_uses_mc_output_lengths(self) -> None:
        """Single-table path (len(splits) == 1): compute should use
        mc_input[table].lengths(), not features.lengths()."""
        smcc = object.__new__(ShardedManagedCollisionCollection)
        smcc._sharding_tables = [["table_0"]]
        smcc._sharding_per_table_feature_splits = [[2]]
        smcc._sharding_features = [["feature_0", "feature_1"]]

        input_lengths = torch.tensor([2, 1, 1, 3])
        input_kjt = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.arange(7, dtype=torch.long),
            lengths=input_lengths,
        )

        # MC module returns different lengths than input
        mc_output_lengths = torch.tensor([1, 2, 2, 2])

        def mock_get_lookup_value(
            table: str, features: KeyedJaggedTensor
        ) -> Dict[str, JaggedTensor]:
            return {
                table: JaggedTensor(
                    values=torch.arange(7, dtype=torch.long),
                    lengths=mc_output_lengths,
                )
            }

        smcc.get_lookup_value = mock_get_lookup_value

        ctx = ManagedCollisionCollectionContext(
            sharding_contexts=[SequenceShardingContext()]
        )

        result = smcc.compute(ctx, KJTList([input_kjt]))

        self.assertTrue(
            torch.equal(result[0].lengths(), mc_output_lengths),
            f"Expected lengths {mc_output_lengths}, got {result[0].lengths()}. "
            "compute should use lengths from mc_module output.",
        )

    def test_compute_multi_table_uses_mc_output_lengths(self) -> None:
        """Multi-table path (len(splits) > 1): compute should use
        concatenated lengths from mc_module outputs, not features.lengths()."""
        smcc = object.__new__(ShardedManagedCollisionCollection)
        smcc._sharding_tables = [["table_0", "table_1"]]
        smcc._sharding_per_table_feature_splits = [[1, 1]]
        smcc._sharding_features = [["feature_0", "feature_1"]]

        input_lengths = torch.tensor([2, 1, 1, 3])
        input_kjt = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.arange(7, dtype=torch.long),
            lengths=input_lengths,
        )

        mc_lengths_table_0 = torch.tensor([1, 2])
        mc_lengths_table_1 = torch.tensor([2, 2])

        def mock_get_lookup_value(
            table: str, features: KeyedJaggedTensor
        ) -> Dict[str, JaggedTensor]:
            if table == "table_0":
                return {
                    table: JaggedTensor(
                        values=torch.tensor([10, 20, 30], dtype=torch.long),
                        lengths=mc_lengths_table_0,
                    )
                }
            return {
                table: JaggedTensor(
                    values=torch.tensor([40, 50, 60, 70], dtype=torch.long),
                    lengths=mc_lengths_table_1,
                )
            }

        smcc.get_lookup_value = mock_get_lookup_value

        ctx = ManagedCollisionCollectionContext(
            sharding_contexts=[SequenceShardingContext()]
        )

        result = smcc.compute(ctx, KJTList([input_kjt]))

        expected_lengths = torch.cat([mc_lengths_table_0, mc_lengths_table_1])
        self.assertTrue(
            torch.equal(result[0].lengths(), expected_lengths),
            f"Expected lengths {expected_lengths}, got {result[0].lengths()}. "
            "compute should use concatenated mc_module output lengths.",
        )

    def test_compute_preserves_lengths_when_mc_unchanged(self) -> None:
        """When mc_module preserves lengths (standard MCH behavior),
        output lengths should match input lengths."""
        smcc = object.__new__(ShardedManagedCollisionCollection)
        smcc._sharding_tables = [["table_0"]]
        smcc._sharding_per_table_feature_splits = [[1]]
        smcc._sharding_features = [["feature_0"]]

        input_lengths = torch.tensor([2, 1])
        input_kjt = KeyedJaggedTensor(
            keys=["feature_0"],
            values=torch.arange(3, dtype=torch.long),
            lengths=input_lengths,
        )

        def mock_get_lookup_value(
            table: str, features: KeyedJaggedTensor
        ) -> Dict[str, JaggedTensor]:
            return {
                table: JaggedTensor(
                    values=torch.tensor([10, 20, 30], dtype=torch.long),
                    lengths=input_lengths.clone(),
                )
            }

        smcc.get_lookup_value = mock_get_lookup_value

        ctx = ManagedCollisionCollectionContext(
            sharding_contexts=[SequenceShardingContext()]
        )

        result = smcc.compute(ctx, KJTList([input_kjt]))

        self.assertTrue(
            torch.equal(result[0].lengths(), input_lengths),
            f"Expected lengths {input_lengths}, got {result[0].lengths()}.",
        )


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
