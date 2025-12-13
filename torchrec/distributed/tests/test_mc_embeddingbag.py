#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Dict, Final, List, Optional, Tuple

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.mc_embeddingbag import (
    ManagedCollisionEmbeddingBagCollectionSharder,
    ShardedManagedCollisionEmbeddingBagCollection,
)
from torchrec.distributed.mc_modules import ShardedManagedCollisionCollection
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


# Global constants for testing ShardedManagedCollisionEmbeddingBagCollection

WORLD_SIZE = 2

# Input KeyedJaggedTensors for each rank in distributed tests
embedding_bag_config: Final[List[EmbeddingBagConfig]] = [
    EmbeddingBagConfig(
        name="table_0",
        feature_names=["feature_0"],
        embedding_dim=8,
        num_embeddings=16,
    ),
    EmbeddingBagConfig(
        name="table_1",
        feature_names=["feature_1"],
        embedding_dim=8,
        num_embeddings=32,
    ),
]

# Expected remapped outputs per iteration per rank for validation
kjt_input_per_rank: Final[List[KeyedJaggedTensor]] = [
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
                1,
                1,
                1,
            ],
        ),
        lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        weights=None,
    ),
]

kjt_out_per_iter_per_rank: Final[List[List[KeyedJaggedTensor]]] = [
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
    ],
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
    ],
]


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        device: torch.device,
        return_remapped: bool = False,
        allow_in_place_embed_weight_update: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        mc_modules = {}
        mc_modules["table_0"] = MCHManagedCollisionModule(
            zch_size=tables[0].num_embeddings,
            input_hash_size=4000,
            device=device,
            eviction_interval=2,
            eviction_policy=DistanceLFU_EvictionPolicy(),
        )

        mc_modules["table_1"] = MCHManagedCollisionModule(
            zch_size=tables[1].num_embeddings,
            device=device,
            input_hash_size=4000,
            eviction_interval=2,
            eviction_policy=DistanceLFU_EvictionPolicy(),
        )

        self._mc_ebc: ManagedCollisionEmbeddingBagCollection = (
            ManagedCollisionEmbeddingBagCollection(
                EmbeddingBagCollection(
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
        if self._return_remapped:
            ebc_out, remapped_ids_out = self._mc_ebc(kjt)
        else:
            ebc_out = self._mc_ebc(kjt)
            remapped_ids_out = None
        pred = torch.cat(
            [ebc_out[key] for key in ["feature_0", "feature_1"]],
            dim=1,
        )
        loss = pred.mean()
        return loss, remapped_ids_out


def _test_sharding(  # noqa C901
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
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
                sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_0"
                ].weight,
                sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_1"
                ].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ebc,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ebc": module_sharding_plan}),
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ebc, ShardedManagedCollisionEmbeddingBagCollection
        )
        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ebc._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )


def _test_sharding_and_remapping(  # noqa C901
    rank: int,
    world_size: int,
    output_keys: List[str],
    tables: List[EmbeddingBagConfig],
    kjt_input_per_rank: List[KeyedJaggedTensor],
    kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]],
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
                sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_0"
                ].weight,
                sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_1"
                ].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ebc,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ebc": module_sharding_plan}),
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ebc, ShardedManagedCollisionEmbeddingBagCollection
        )
        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_bag_collection`.
            sharded_sparse_arch._mc_ebc._embedding_bag_collection,
            ShardedEmbeddingBagCollection,
        )
        assert (
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_bag_collection`.
            sharded_sparse_arch._mc_ebc._embedding_bag_collection._has_uninitialized_input_dist
            is False
        )
        assert (
            not hasattr(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `_embedding_bag_collection`.
                sharded_sparse_arch._mc_ebc._embedding_bag_collection,
                "_input_dists",
            )
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_embedding_bag_collection`.
            or len(sharded_sparse_arch._mc_ebc._embedding_bag_collection._input_dists)
            == 0
        )

        assert isinstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ebc._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )

        test_state_dict = sharded_sparse_arch.state_dict()
        sharded_sparse_arch.load_state_dict(test_state_dict)

        # sharded model
        # each rank gets a subbatch
        loss1, remapped_ids1 = sharded_sparse_arch(kjt_input)
        loss1.backward()
        loss2, remapped_ids2 = sharded_sparse_arch(kjt_input)
        loss2.backward()
        remapped_ids = [remapped_ids1, remapped_ids2]
        for key in output_keys:
            for i, kjt_out in enumerate(kjt_out_per_iter):
                assert torch.equal(
                    remapped_ids[i][key].values(),
                    kjt_out[key].values(),
                ), f"feature {key} on {ctx.rank} iteration {i} does not match, got {remapped_ids[i][key].values()}, expect {kjt_out[key].values()}"

        # TODO: validate embedding rows, and eviction


def _test_in_place_embd_weight_update(  # noqa C901
    rank: int,
    world_size: int,
    output_keys: List[str],
    tables: List[EmbeddingBagConfig],
    kjt_input_per_rank: List[KeyedJaggedTensor],
    kjt_out_per_iter_per_rank: List[List[KeyedJaggedTensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
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
            allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
        )
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            # pyre-fixme[6]: For 2nd argument expected `Iterable[Parameter]` but got
            #  `Iterable[Union[Module, Tensor]]`.
            [
                sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_0"
                ].weight,
                sparse_arch._mc_ebc._embedding_bag_collection.embedding_bags[
                    "table_1"
                ].weight,
            ],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ebc,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ebc": module_sharding_plan}),
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        test_state_dict = sharded_sparse_arch.state_dict()
        sharded_sparse_arch.load_state_dict(test_state_dict)

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
            remapped_ids = [remapped_ids1, remapped_ids2]
            for key in output_keys:
                for i, kjt_out in enumerate(kjt_out_per_iter):
                    assert torch.equal(
                        remapped_ids[i][key].values(),
                        kjt_out[key].values(),
                    ), f"feature {key} on {ctx.rank} iteration {i} does not match, got {remapped_ids[i][key].values()}, expect {kjt_out[key].values()}"


@skip_if_asan_class
class ShardedMCEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_uneven_sharding(self, backend: str) -> None:
        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=17,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=33,
            ),
        ]

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            sharder=ManagedCollisionEmbeddingBagCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_even_sharding(self, backend: str) -> None:

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            sharder=ManagedCollisionEmbeddingBagCollectionSharder(),
            backend=backend,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore
    @given(backend=st.sampled_from(["nccl"]))
    @settings(deadline=None)
    def test_sharding_zch_mc_ebc(self, backend: str) -> None:
        self._run_multi_process_test(
            callable=_test_sharding_and_remapping,
            output_keys=["feature_0", "feature_1"],
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            kjt_input_per_rank=kjt_input_per_rank,
            kjt_out_per_iter_per_rank=kjt_out_per_iter_per_rank,
            sharder=ManagedCollisionEmbeddingBagCollectionSharder(),
            backend=backend,
        )

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

        self._run_multi_process_test(
            callable=_test_in_place_embd_weight_update,
            output_keys=["feature_0", "feature_1"],
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            kjt_input_per_rank=kjt_input_per_rank,
            kjt_out_per_iter_per_rank=kjt_out_per_iter_per_rank,
            sharder=ManagedCollisionEmbeddingBagCollectionSharder(),
            backend=backend,
            allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
        )
