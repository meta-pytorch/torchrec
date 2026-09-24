#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from hypothesis import assume, given, settings, strategies as st
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.mc_embedding import (
    KJTList,
    ManagedCollisionEmbeddingCollectionSharder,
    ShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import (
    _cat_jagged_values,
    _fx_global_to_local_index,
    _fx_jt_dict_add_offset,
    _get_length_per_key,
    create_mc_sharding,
    EmbeddingCollectionContext,
    input_dist_permute,
    ManagedCollisionCollectionContext,
    ManagedCollisionCollectionSharder,
    ShardedManagedCollisionCollection,
    ShardedMCCRemapper,
    ShardedQuantManagedCollisionCollection,
    update_jagged_tensor_dict,
)
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
from torchrec.distributed.tests.test_mc_embedding_utils import SparseArch
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


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
        gather_list: Optional[List[Any]] = [None, None] if ctx.rank == 0 else None
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


def _test_uneven_buckets_per_rank(
    tables: List[EmbeddingConfig],
    rank: int,
    hash_sizes: List[int],
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    indices_to_rank: List[int],
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        features = ["feature_0", "feature_1"]
        sparse_arch = SparseArch(
            tables,
            torch.device("meta"),
            use_mpzch=True,
            input_hash_size=hash_sizes,
        )
        assert ctx.pg is not None
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ec,
            per_param_sharding={
                "table_0": row_wise(num_buckets=tables[0].total_num_buckets),
                "table_1": row_wise(num_buckets=tables[1].total_num_buckets),
            },
            local_size=None,
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

        assert hasattr(sharded_sparse_arch._mc_ec, "_managed_collision_collection")
        mc_module = sharded_sparse_arch._mc_ec._managed_collision_collection
        assert isinstance(mc_module, ShardedManagedCollisionCollection)

        kjt_rank = kjt_input_per_rank[rank].to(ctx.device)
        mc_module._create_input_dists(features)
        input_dists = mc_module._input_dists
        assert len(input_dists) == 1
        output = input_dists[0](kjt_rank).wait().wait()

        torch.testing.assert_close(output.values().tolist(), indices_to_rank[rank])


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

    @unittest.skip("Temporarily disabled")
    @unittest.skipIf(
        torch.cuda.device_count() <= 2,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_uneven_buckets_per_rank(self, backend: str) -> None:
        WORLD_SIZE = 2
        # Test on one feature having infinite input space, and other finite
        HASH_SIZES = [0, 40]  # hash sizes per feature
        BUCKETS = [3, 5]  # uneven buckets for worldsize=2
        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=9,
                total_num_buckets=BUCKETS[0],
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=10,
                total_num_buckets=BUCKETS[1],
            ),
        ]
        # feature_0 does interleave mapping to rank
        # feature_1 does block mapping to rank
        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    [
                        0,  # feature_0, rank 0
                        1,  # feature_0, rank 0
                        2,  # feature_0, rank 1
                        3,  # feature_0, rank 0
                        4,  # feature_0, rank 0
                        5,  # feature_0, rank 1
                        0,  # feature_1, rank 0,
                        10,  # feature_1, rank 0
                        23,  # feature_1, rank 0
                        30,  # feature_1, rank 1
                        32,  # feature_1, rank 1
                        39,  # feature_1, rank 1
                    ],
                ),
                lengths=torch.LongTensor([1] * 12),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor(
                    [
                        0,  # feature_0, rank 0
                        10,  # feature_1, rank 0
                    ],
                ),
                lengths=torch.LongTensor([1, 1]),
            ),
        ]

        # Final answer that maps rank to indices
        rank_to_indices = [[0, 1, 3, 4, 0, 0, 10, 23, 10], [2, 5, 30, 32, 39]]

        self._run_multi_process_test(
            callable=_test_uneven_buckets_per_rank,
            world_size=WORLD_SIZE,
            hash_sizes=HASH_SIZES,
            tables=embedding_config,
            backend=backend,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            kjt_input_per_rank=kjt_input_per_rank,
            indices_to_rank=rank_to_indices,
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
    @given(backend=st.sampled_from(["nccl"]), uneven_buckets=st.booleans())
    @settings(deadline=None)
    def test_sharding_zch_mc_ec_dedup(self, backend: str, uneven_buckets: bool) -> None:
        # Temporarily disabled for the uneven buckets case
        assume(not uneven_buckets)
        WORLD_SIZE = 2
        total_num_buckets = [4, 4]
        if uneven_buckets:
            total_num_buckets = [3, 4]

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0", "feature_2"],
                embedding_dim=8,
                num_embeddings=16,
                total_num_buckets=total_num_buckets[0],
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
                total_num_buckets=total_num_buckets[1],
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
        smcc._use_2d_weights = False
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
            table: str,
            features: KeyedJaggedTensor,
            write_weights: Optional[torch.Tensor] = None,
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
        smcc._use_2d_weights = False
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
            table: str,
            features: KeyedJaggedTensor,
            write_weights: Optional[torch.Tensor] = None,
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
        smcc._use_2d_weights = False
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
            table: str,
            features: KeyedJaggedTensor,
            write_weights: Optional[torch.Tensor] = None,
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


class DummyMCModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestInputDistPermute(unittest.TestCase):
    def _create_kjt(self) -> KeyedJaggedTensor:
        return KeyedJaggedTensor(
            keys=["f0", "f1", "f2"],
            values=torch.tensor([10, 20, 30, 40, 50, 60]),
            lengths=torch.tensor([2, 1, 1, 1, 0, 1]),
        )

    def test_input_dist_permute_correctness(self) -> None:
        kjt = self._create_kjt()
        features_order = [2, 0, 1]
        features_order_tensor = torch.tensor(features_order, dtype=torch.int32)

        result = input_dist_permute(kjt, features_order, features_order_tensor)
        expected = kjt.permute(features_order, features_order_tensor)

        self.assertEqual(result.keys(), expected.keys())
        torch.testing.assert_close(result.values(), expected.values())
        torch.testing.assert_close(result.lengths(), expected.lengths())

    def test_input_dist_permute_fx_node_type(self) -> None:
        _features_order_tensor = torch.tensor([2, 0, 1], dtype=torch.int32)

        class _Module(torch.nn.Module):
            def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
                return input_dist_permute(features, [2, 0, 1], _features_order_tensor)

        tracer = torch.fx.Tracer(
            autowrap_functions=(input_dist_permute,),
            autowrap_modules=(torch.fx,),
        )
        graph = tracer.trace(_Module())

        found = any(
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "input_dist_permute"
            for node in graph.nodes
        )
        self.assertTrue(found)

    def test_input_dist_permute_identity_order(self) -> None:
        kjt = self._create_kjt()
        features_order = [0, 1, 2]
        features_order_tensor = torch.tensor(features_order, dtype=torch.int32)

        result = input_dist_permute(kjt, features_order, features_order_tensor)

        self.assertEqual(result.keys(), kjt.keys())
        torch.testing.assert_close(result.values(), kjt.values())

    def test_input_dist_permute_subset(self) -> None:
        kjt = self._create_kjt()
        result = input_dist_permute(kjt, [1], torch.tensor([1], dtype=torch.int32))

        self.assertEqual(result.keys(), ["f1"])


class TestMCModuleUtilityFunctions(unittest.TestCase):
    def test_fx_global_to_local_index(self) -> None:
        feature_dict = {
            "t_a": JaggedTensor(
                values=torch.tensor([100, 200]), lengths=torch.tensor([2])
            ),
            "t_b": JaggedTensor(values=torch.tensor([500]), lengths=torch.tensor([1])),
        }
        result = _fx_global_to_local_index(feature_dict, {"t_a": 100, "t_b": 500})

        torch.testing.assert_close(result["t_a"].values(), torch.tensor([0, 100]))
        torch.testing.assert_close(result["t_b"].values(), torch.tensor([0]))

    def test_fx_jt_dict_add_offset(self) -> None:
        feature_dict = {
            "t_a": JaggedTensor(
                values=torch.tensor([0, 10]), lengths=torch.tensor([2])
            ),
        }
        result = _fx_jt_dict_add_offset(feature_dict, {"t_a": 100})

        torch.testing.assert_close(result["t_a"].values(), torch.tensor([100, 110]))

    def test_get_length_per_key(self) -> None:
        kjt = KeyedJaggedTensor(
            keys=["f0", "f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            lengths=torch.tensor([2, 1, 1, 1, 0, 1]),
        )
        self.assertEqual(_get_length_per_key(kjt).tolist(), [3, 2, 1])

    def test_cat_jagged_values(self) -> None:
        jd = {
            "a": JaggedTensor(values=torch.tensor([1, 2]), lengths=torch.tensor([2])),
            "b": JaggedTensor(values=torch.tensor([3]), lengths=torch.tensor([1])),
        }
        torch.testing.assert_close(_cat_jagged_values(jd), torch.tensor([1, 2, 3]))

    def test_update_jagged_tensor_dict(self) -> None:
        jt_a = JaggedTensor(values=torch.tensor([1]), lengths=torch.tensor([1]))
        jt_b = JaggedTensor(values=torch.tensor([2]), lengths=torch.tensor([1]))
        result = update_jagged_tensor_dict({"a": jt_a}, {"b": jt_b})

        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_create_mc_sharding_unsupported(self) -> None:
        from unittest.mock import MagicMock

        with self.assertRaises(ValueError):
            create_mc_sharding(
                sharding_type=ShardingType.TABLE_WISE.value,
                sharding_infos=[],
                env=MagicMock(),
            )


class TestMCModuleContexts(unittest.TestCase):
    def test_embedding_collection_context_defaults(self) -> None:
        ctx = EmbeddingCollectionContext(sharding_contexts=[])

        self.assertEqual(ctx.input_features, [])
        self.assertIsNone(ctx.inverse_indices)
        self.assertFalse(ctx.variable_batch_per_feature)

    def test_mcc_context_is_subclass(self) -> None:
        ctx = ManagedCollisionCollectionContext(sharding_contexts=[])

        self.assertIsInstance(ctx, EmbeddingCollectionContext)


class TestMCModuleSharder(unittest.TestCase):
    def test_sharding_types(self) -> None:
        sharder = ManagedCollisionCollectionSharder()

        self.assertEqual(sharder.sharding_types("cuda"), [ShardingType.ROW_WISE.value])

    def test_module_type(self) -> None:
        sharder = ManagedCollisionCollectionSharder()

        self.assertEqual(sharder.module_type, ManagedCollisionCollection)

    def test_shardable_parameters_raises(self) -> None:
        from unittest.mock import MagicMock

        sharder = ManagedCollisionCollectionSharder()

        with self.assertRaises(NotImplementedError):
            sharder.shardable_parameters(MagicMock())


class TestMCCRemapperUnit(unittest.TestCase):
    def test_init_feature_to_offset(self) -> None:
        mc_modules = torch.nn.ModuleDict({"table_a": DummyMCModule()})

        remapper = ShardedMCCRemapper(
            table_feature_splits=[2],
            fns=["f0", "f1"],
            managed_collision_modules=mc_modules,
            shard_metadata={"table_a": [100, 50]},
        )

        self.assertEqual(remapper._feature_to_offset["f0"], 100)
        self.assertEqual(remapper._feature_to_offset["f1"], 100)

    def test_global_to_local_index(self) -> None:
        mc_modules = torch.nn.ModuleDict({"table_a": DummyMCModule()})
        remapper = ShardedMCCRemapper(
            table_feature_splits=[1],
            fns=["f0"],
            managed_collision_modules=mc_modules,
            shard_metadata={"table_a": [100, 50]},
        )

        jt = JaggedTensor(values=torch.tensor([150, 175]), lengths=torch.tensor([1, 1]))
        result = remapper.global_to_local_index({"table_a": jt})

        torch.testing.assert_close(result["table_a"].values(), torch.tensor([50, 75]))


class TestShardedQuantMCCOutputDistUnit(unittest.TestCase):
    def _make_sqmcc(
        self, feature_names: list[str]
    ) -> ShardedQuantManagedCollisionCollection:
        # pyre-ignore[20]
        obj = object.__new__(ShardedQuantManagedCollisionCollection)
        obj._feature_names = feature_names
        return obj

    def test_output_dist_empty(self) -> None:
        sqmcc = self._make_sqmcc(["f0"])
        ctx = ManagedCollisionCollectionContext(sharding_contexts=[])

        result = sqmcc.output_dist(ctx, KJTList([]))

        self.assertEqual(result.keys(), [])

    def test_output_dist_single(self) -> None:
        sqmcc = self._make_sqmcc(["f0", "f1"])
        ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        kjt = KeyedJaggedTensor(
            keys=["f0", "f1"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([2, 1]),
        )

        result = sqmcc.output_dist(ctx, KJTList([kjt]))

        self.assertEqual(result.keys(), ["f0", "f1"])
        torch.testing.assert_close(result.values(), torch.tensor([1, 2, 3]))

    def test_compute_raises(self) -> None:
        sqmcc = self._make_sqmcc(["f0"])
        ctx = ManagedCollisionCollectionContext(sharding_contexts=[])

        with self.assertRaises(NotImplementedError):
            sqmcc.compute(ctx, 0, KJTList([]))

    def test_create_context(self) -> None:
        sqmcc = self._make_sqmcc(["f0"])
        ctx = sqmcc.create_context()

        self.assertIsInstance(ctx, ManagedCollisionCollectionContext)

    def test_unsharded_module_type(self) -> None:
        sqmcc = self._make_sqmcc(["f0"])

        self.assertEqual(sqmcc.unsharded_module_type, ManagedCollisionCollection)


class _IdentityDistModule(torch.nn.Module):
    def forward(self, x: KeyedJaggedTensor) -> KeyedJaggedTensor:
        return x


class _SplitWithIndexAccessModule(torch.nn.Module):
    """Mimics the fixed ShardedQuantManagedCollisionCollection.input_dist() pattern."""

    def __init__(self, num_splits: int) -> None:
        super().__init__()
        self._dists = torch.nn.ModuleList(
            [_IdentityDistModule() for _ in range(num_splits)]
        )
        self._splits = [1] * num_splits

    def forward(self, features: KeyedJaggedTensor) -> List[KeyedJaggedTensor]:
        feature_splits = features.split(self._splits)
        results: List[KeyedJaggedTensor] = []
        for i in range(len(self._dists)):
            results.append(self._dists[i](feature_splits[i]))
        return results


class _SplitWithZipModule(torch.nn.Module):
    """Mimics the old broken ShardedQuantManagedCollisionCollection.input_dist() pattern."""

    def __init__(self, num_splits: int) -> None:
        super().__init__()
        self._dists = torch.nn.ModuleList(
            [_IdentityDistModule() for _ in range(num_splits)]
        )
        self._splits = [1] * num_splits

    def forward(self, features: KeyedJaggedTensor) -> List[KeyedJaggedTensor]:
        feature_splits = features.split(self._splits)
        results: List[KeyedJaggedTensor] = []
        for feature_split, dist_mod in zip(feature_splits, self._dists):
            results.append(dist_mod(feature_split))
        return results


class TestMccInputDistFxTracing(unittest.TestCase):
    """Tests that ShardedQuantManagedCollisionCollection.input_dist() iteration
    pattern is FX-traceable.

    The fix changed zip(feature_splits, self._input_dists) to index-based
    range(len(...)) + feature_splits[i], because FX Proxies support
    __getitem__ but not __iter__.
    """

    def _create_kjt(self, num_features: int = 3) -> KeyedJaggedTensor:
        keys = [f"f{i}" for i in range(num_features)]
        values = torch.arange(num_features * 2, dtype=torch.int64)
        lengths = torch.ones(num_features, dtype=torch.int64) * 2
        return KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)

    def test_index_access_traces_without_error(self) -> None:
        module = _SplitWithIndexAccessModule(num_splits=3)
        tracer = torch.fx.Tracer()
        graph = tracer.trace(module)

        split_nodes = [
            n for n in graph.nodes if n.op == "call_method" and n.target == "split"
        ]
        self.assertGreaterEqual(len(split_nodes), 1)

        getitem_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and "getitem" in str(n.target)
        ]
        self.assertEqual(len(getitem_nodes), 3)

    def test_zip_iteration_raises_trace_error(self) -> None:
        module = _SplitWithZipModule(num_splits=3)
        tracer = torch.fx.Tracer()
        with self.assertRaises(torch.fx.proxy.TraceError):
            tracer.trace(module)

    def test_index_access_correctness(self) -> None:
        kjt = self._create_kjt(num_features=3)
        module = _SplitWithIndexAccessModule(num_splits=3)

        results = module(kjt)

        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result.keys(), [f"f{i}"])
            self.assertEqual(len(result.values()), 2)

    def test_index_access_matches_zip_access(self) -> None:
        kjt = self._create_kjt(num_features=3)
        index_module = _SplitWithIndexAccessModule(num_splits=3)
        zip_module = _SplitWithZipModule(num_splits=3)

        index_results = index_module(kjt)
        zip_results = zip_module(kjt)

        self.assertEqual(len(index_results), len(zip_results))
        for idx_r, zip_r in zip(index_results, zip_results):
            self.assertEqual(idx_r.keys(), zip_r.keys())
            torch.testing.assert_close(idx_r.values(), zip_r.values())
            torch.testing.assert_close(idx_r.lengths(), zip_r.lengths())

    def test_traced_module_executes_correctly(self) -> None:
        module = _SplitWithIndexAccessModule(num_splits=2)
        tracer = torch.fx.Tracer()
        graph = tracer.trace(module)
        gm = torch.fx.GraphModule(module, graph)

        kjt = KeyedJaggedTensor(
            keys=["f0", "f1"],
            values=torch.tensor([10, 20, 30, 40]),
            lengths=torch.tensor([2, 2]),
        )

        original_results = module(kjt)
        traced_results = gm(kjt)

        self.assertEqual(len(traced_results), len(original_results))
        for orig, traced in zip(original_results, traced_results):
            self.assertEqual(orig.keys(), traced.keys())
            torch.testing.assert_close(orig.values(), traced.values())
            torch.testing.assert_close(orig.lengths(), traced.lengths())

    def test_single_split_traces_correctly(self) -> None:
        module = _SplitWithIndexAccessModule(num_splits=1)
        tracer = torch.fx.Tracer()
        graph = tracer.trace(module)
        gm = torch.fx.GraphModule(module, graph)

        kjt = KeyedJaggedTensor(
            keys=["f0"],
            values=torch.tensor([10, 20]),
            lengths=torch.tensor([2]),
        )
        results = gm(kjt)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].keys(), ["f0"])
