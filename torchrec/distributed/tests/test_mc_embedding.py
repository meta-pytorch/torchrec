#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.mc_embedding import (
    ManagedCollisionEmbeddingCollectionSharder,
    ShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import ShardedManagedCollisionCollection
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    EmbeddingCollectionSharder,
    row_wise,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        return_remapped: bool = False,
        input_hash_size: int = 4000,
        allow_in_place_embed_weight_update: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        mc_modules = {}
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
            # pyre-fixme[6]: For 2nd argument expected `Iterable[Parameter]` but got
            #  `Iterable[Union[Module, Tensor]]`.
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
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ec, ShardedManagedCollisionEmbeddingCollection
        )
        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )
        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection._has_uninitialized_input_dist
            is False
        )
        assert (
            not hasattr(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `_embedding_collection`.
                sharded_sparse_arch._mc_ec._embedding_collection,
                "_input_dists",
            )
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            or len(sharded_sparse_arch._mc_ec._embedding_collection._input_dists) == 0
        )

        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )

        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            == sharded_sparse_arch._mc_ec._embedding_collection._use_index_dedup
        )

        initial_state_dict = sharded_sparse_arch.state_dict()
        for key, sharded_tensor in initial_state_dict.items():
            postfix = ".".join(key.split(".")[-2:])
            if postfix in initial_state_per_rank[ctx.rank]:
                tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                assert torch.equal(
                    tensor, initial_state_per_rank[ctx.rank][postfix]
                ), f"initial state {key} on {ctx.rank} does not match, got {tensor}, expect {initial_state_per_rank[rank][postfix]}"

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
                assert torch.equal(
                    tensor, final_state_per_rank[ctx.rank][postfix]
                ), f"initial state {key} on {ctx.rank} does not match, got {tensor}, expect {final_state_per_rank[rank][postfix]}"

        remapped_ids = [remapped_ids1, remapped_ids2]
        for key in output_keys:
            for i, kjt_out in enumerate(kjt_out_per_iter):
                assert torch.equal(
                    remapped_ids[i][key].values(),
                    kjt_out[key].values(),
                ), f"feature {key} on {ctx.rank} iteration {i} does not match, got {remapped_ids[i][key].values()}, expect {kjt_out[key].values()}"

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
            # pyre-fixme[6]: For 2nd argument expected `Iterable[Parameter]` but got
            #  `Iterable[Union[Module, Tensor]]`.
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
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        initial_state_dict = sharded_sparse_arch.state_dict()
        for key, sharded_tensor in initial_state_dict.items():
            postfix = ".".join(key.split(".")[-2:])
            if postfix in initial_state_per_rank[ctx.rank]:
                tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                assert torch.equal(
                    tensor, initial_state_per_rank[ctx.rank][postfix]
                ), f"initial state {key} on {ctx.rank} does not match, got {tensor}, expect {initial_state_per_rank[rank][postfix]}"

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
                    assert torch.equal(
                        tensor, final_state_per_rank[ctx.rank][postfix]
                    ), f"initial state {key} on {ctx.rank} does not match, got {tensor}, expect {final_state_per_rank[rank][postfix]}"

            remapped_ids = [remapped_ids1, remapped_ids2]
            for key in output_keys:
                for i, kjt_out in enumerate(kjt_out_per_iter):
                    assert torch.equal(
                        remapped_ids[i][key].values(),
                        kjt_out[key].values(),
                    ), f"feature {key} on {ctx.rank} iteration {i} does not match, got {remapped_ids[i][key].values()}, expect {kjt_out[key].values()}"


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
            # pyre-fixme[6]: For 2nd argument expected `Iterable[Parameter]` but got
            #  `Iterable[Union[Module, Tensor]]`.
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
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ec, ShardedManagedCollisionEmbeddingCollection
        )
        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )
        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            sharded_sparse_arch._mc_ec._embedding_collection._has_uninitialized_input_dist
            is False
        )
        assert (
            not hasattr(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `_embedding_collection`.
                sharded_sparse_arch._mc_ec._embedding_collection,
                "_input_dists",
            )
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            or len(sharded_sparse_arch._mc_ec._embedding_collection._input_dists) == 0
        )

        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
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
                assert torch.equal(
                    remapped_ids[i][key].values(),
                    kjt_out[key].values(),
                ), f"feature {key} on {ctx.rank} iteration {i} does not match, got {remapped_ids[i][key].values()}, expect {kjt_out[key].values()}"

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
                # pyre-fixme[6]: For 2nd argument expected `Iterable[Parameter]` but
                #  got `Iterable[Union[Module, Tensor]]`.
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
                # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
                #  `Optional[ProcessGroup]`.
                env=ShardingEnv.from_process_group(ctx.pg),
                sharders=[sharder],
                device=ctx.device,
            )
            state_dict = sharded_sparse_arch.state_dict()

            for key in state_dict.keys():
                if isinstance(state_dict[key], ShardedTensor):
                    replacement_tensor = torch.cat(
                        # pyre-ignore [16]
                        [gather_list[0][key], gather_list[1][key]],
                        dim=0,
                    ).to(ctx.device)
                    state_dict[key].local_shards()[0].tensor.copy_(replacement_tensor)
                else:
                    state_dict[key] = gather_list[0][key].to(ctx.device)

            sharded_sparse_arch.load_state_dict(state_dict)
            loss3, remapped_ids3 = sharded_sparse_arch(kjt_input)
            final_state_dict = sharded_sparse_arch.state_dict()
            for key, sharded_tensor in final_state_dict.items():
                postfix = ".".join(key.split(".")[-2:])
                if postfix in final_state_per_rank[ctx.rank]:
                    tensor = sharded_tensor.local_shards()[0].tensor.cpu()
                    assert torch.equal(
                        tensor, final_state_per_rank[ctx.rank][postfix]
                    ), f"initial state {key} on {ctx.rank} does not match, got {tensor}, expect {final_state_per_rank[rank][postfix]}"

            remapped_ids = [remapped_ids3]
            for key in kjt_input.keys():
                for i, kjt_out in enumerate(kjt_out_per_iter[-1:]):  # last iteration
                    assert torch.equal(
                        remapped_ids[i][key].values(),
                        kjt_out[key].values(),
                    ), f"feature {key} on {ctx.rank} iteration {i} does not match, got {remapped_ids[i][key].values()}, expect {kjt_out[key].values()}"


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
            # pyre-fixme[6]: For 2nd argument expected `Iterable[Parameter]` but got
            #  `Iterable[Union[Module, Tensor]]`.
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
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )
        dedup_sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[dedup_sharder],
            device=ctx.device,
        )

        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            == sharded_sparse_arch._mc_ec._embedding_collection._use_index_dedup
        )

        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            is False
        )

        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            dedup_sharded_sparse_arch._mc_ec._managed_collision_collection._use_index_dedup
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_collection`.
            == dedup_sharded_sparse_arch._mc_ec._embedding_collection._use_index_dedup
        )

        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
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

        assert torch.allclose(loss1, dedup_loss1)
        # deduping is not being used right now
        # assert torch.allclose(remapped_1.values(), dedup_remapped_1.values())
        # assert torch.allclose(remapped_1.lengths(), dedup_remapped_1.lengths())


class SparseArchSingleTableWithReadonly(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        return_remapped: bool = False,
        input_hash_size: int = 4000,
        enable_per_feature_lookups: bool = True,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        from torchrec.fb.modules.hash_mc_modules import HashZchManagedCollisionModule

        mc_modules = {}
        mc_modules["table_0"] = HashZchManagedCollisionModule(
            zch_size=(tables[0].num_embeddings),
            input_hash_size=input_hash_size,
            device=device,
            total_num_buckets=2,
            read_only_suffix="_readonly",
            enable_per_feature_lookups=enable_per_feature_lookups,
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
                return_remapped_features=self._return_remapped,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, JaggedTensor]]]:
        ec_out, remapped_ids_out = self._mc_ec(kjt)
        pred = torch.cat(
            [ec_out[key].values() for key in ["feature_0", "feature_0_readonly"]],
            dim=0,
        )
        loss = pred.mean()
        return loss, remapped_ids_out


def _test_readonly_feature_metadata(  # noqa C901
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
    input_hash_size: int = 4000,
) -> None:
    """
    Test that verifies the readonly feature behavior in HashZchManagedCollisionModule.

    The test validates that:
    1. Regular features (feature_0) can add new IDs to the ZCH identities table
    2. Readonly features (feature_0_readonly) do not add new IDs to the table

    Note: The readonly feature lookup for IDs not in the table returns -1, which would
    cause CUDA errors during embedding lookup. So we test by comparing state changes.

    Bucket assignment with input_hash_size=4000, total_num_buckets=2:
    - IDs 0-1999 go to bucket 0 (rank 0)
    - IDs 2000-3999 go to bucket 1 (rank 1)
    """
    for enable_per_feature_lookups in [True, False]:
        with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
            kjt_input = kjt_input_per_rank[rank].to(ctx.device)
            return_remapped: bool = True
            sparse_arch = SparseArchSingleTableWithReadonly(
                tables,
                torch.device("meta"),
                return_remapped=return_remapped,
                input_hash_size=input_hash_size,
                enable_per_feature_lookups=enable_per_feature_lookups,
            )

            apply_optimizer_in_backward(
                RowWiseAdagrad,
                [
                    sparse_arch._mc_ec._embedding_collection.embeddings[
                        "table_0"
                    ].weight,
                ],
                {"lr": 0.01},
            )
            module_sharding_plan = construct_module_sharding_plan(
                sparse_arch._mc_ec,
                per_param_sharding={"table_0": row_wise()},
                local_size=local_size,
                world_size=world_size,
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                sharder=sharder,
            )

            sharded_sparse_arch = _shard_modules(
                module=copy.deepcopy(sparse_arch),
                plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
                env=ShardingEnv.from_process_group(ctx.pg),
                sharders=[sharder],
                device=ctx.device,
            )

            assert isinstance(
                sharded_sparse_arch._mc_ec, ShardedManagedCollisionEmbeddingCollection
            )
            assert isinstance(
                sharded_sparse_arch._mc_ec._embedding_collection,
                ShardedEmbeddingCollection,
            )
            assert isinstance(
                sharded_sparse_arch._mc_ec._managed_collision_collection,
                ShardedManagedCollisionCollection,
            )

            # First run with both features to initialize the input dists and add initial IDs
            # The kjt_input has IDs for both feature_0 and feature_0_readonly
            loss_init, _ = sharded_sparse_arch(kjt_input)

            # Get initial identities state after first forward
            initial_state_dict = sharded_sparse_arch.state_dict()
            initial_identities = None
            cpu_state_dict = {}
            for key, sharded_tensor in initial_state_dict.items():
                if "table_0._hash_zch_identities" in key:
                    target_key = key
                    initial_identities = sharded_tensor.local_shards()[0].tensor.clone()
                    cpu_state_dict[key] = initial_identities.to("cpu")

            assert initial_identities is not None
            gather_list = [None, None] if ctx.rank == 0 else None
            torch.distributed.gather_object(cpu_state_dict, gather_list)

            # only expect 3 write elements in the initial state
            if rank == 0:
                write_element_cnt: int = sum(
                    [(i[target_key] != -1).sum() for i in gather_list]
                )
                assert torch.equal(
                    torch.tensor(write_element_cnt),
                    torch.tensor(6) if enable_per_feature_lookups else torch.tensor(12),
                ), f"identity tensor expect {6 if enable_per_feature_lookups else 12} element when enable_per_feature_lookups={enable_per_feature_lookups} , now get {write_element_cnt}"


@skip_if_asan_class
class ShardedMCEmbeddingCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore
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
    # pyre-ignore
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
    # pyre-ignore
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
    # pyre-ignore
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
    # pyre-ignore
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_mc_ec_readonly_feature(self, backend: str) -> None:
        """
        Test that readonly features do not change MCH metadata while regular features do.

        This test verifies that:
        1. When running forward with a feature_0_readonly, the MCH metadata
           (_mch_sorted_raw_ids and _mch_remapped_ids_mapping) should NOT change.
        2. When running forward with feature_0 (regular feature), the MCH metadata
           SHOULD change.
        """
        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0", "feature_0_readonly"],
                embedding_dim=8,
                num_embeddings=1000,
            ),
        ]

        kjt_input_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_0_readonly"],
                values=torch.LongTensor([1000, 2000, 3000, 1001, 2001, 3001]),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_0_readonly"],
                values=torch.LongTensor([1002, 2002, 3002, 1003, 2003, 3003]),
                lengths=torch.LongTensor([1, 1, 1, 1, 1, 1]),
                weights=None,
            ),
        ]

        self._run_multi_process_test(
            callable=_test_readonly_feature_metadata,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend=backend,
        )
