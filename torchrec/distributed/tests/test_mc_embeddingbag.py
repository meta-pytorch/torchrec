#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import multiprocessing
import unittest
from collections import OrderedDict
from typing import Any, Dict, Final, List, Optional, Tuple

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
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
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
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
from torchrec.types import DataType


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
        use_mpzch: bool = False,
        is_inference: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        mc_modules = {}
        if use_mpzch:
            # Parameters hard-coded from test_quant_mc_embedding
            mc_modules["table_0"] = HashZchManagedCollisionModule(
                zch_size=(tables[0].num_embeddings),
                input_hash_size=0,
                device=device,
                total_num_buckets=100,
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_0"],
                    single_ttl=1,
                ),
                is_inference=is_inference,
            )

            mc_modules["table_1"] = HashZchManagedCollisionModule(
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=0,
                total_num_buckets=2,
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_1"],
                    single_ttl=1,
                ),
                is_inference=is_inference,
            )
        else:
            # pyrefly: ignore[unsupported-operation]
            mc_modules["table_0"] = MCHManagedCollisionModule(
                zch_size=tables[0].num_embeddings,
                input_hash_size=4000,
                device=device,
                eviction_interval=2,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )

            # pyrefly: ignore[unsupported-operation]
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
        if self._return_remapped:
            ebc_out, remapped_ids_out = self._mc_ebc(kjt)
        else:
            ebc_out, _ = self._mc_ebc(kjt)
            remapped_ids_out = None

        pred = torch.cat(
            [ebc_out[key] for key in ["feature_0", "feature_1"]],
            dim=1,
        )
        loss = pred.mean()

        return loss, remapped_ids_out


def _test_sharding(  # noqa C901
    tables: List[EmbeddingBagConfig],
    rank: int,
    world_size: int,
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
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
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
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ebc, ShardedManagedCollisionEmbeddingBagCollection
        )
        assert isinstance(
            #  `_managed_collision_collection`.
            sharded_sparse_arch._mc_ebc._managed_collision_collection,
            ShardedManagedCollisionCollection,
        )


def _test_sharding_and_remapping(  # noqa C901
    output_keys: List[str],
    tables: List[EmbeddingBagConfig],
    rank: int,
    world_size: int,
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
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
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
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(
            sharded_sparse_arch._mc_ebc, ShardedManagedCollisionEmbeddingBagCollection
        )
        assert isinstance(
            #  `_embedding_bag_collection`.
            sharded_sparse_arch._mc_ebc._embedding_bag_collection,
            ShardedEmbeddingBagCollection,
        )
        assert (
            #  `_embedding_bag_collection`.
            sharded_sparse_arch._mc_ebc._embedding_bag_collection._has_uninitialized_input_dist
            is False
        )
        assert (
            not hasattr(
                #  attribute `_embedding_bag_collection`.
                sharded_sparse_arch._mc_ebc._embedding_bag_collection,
                "_input_dists",
            )
            #  `_embedding_bag_collection`.
            or len(sharded_sparse_arch._mc_ebc._embedding_bag_collection._input_dists)
            == 0
        )

        assert isinstance(
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
    output_keys: List[str],
    tables: List[EmbeddingBagConfig],
    rank: int,
    world_size: int,
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
            #  `Iterable[Union[Module, Tensor]]`.
            # pyrefly: ignore[bad-argument-type]
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
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
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


def _merge_sharded_return_state_dict(
    sharded_return_state_dict: OrderedDict[str, torch.Tensor],
    is_with_metadata: bool = True,  # whether to save metadata in the state dict as well
) -> OrderedDict[str, torch.Tensor]:
    # stack the output of train_state_dict into a single tensor
    ## sort the state dict keys by table sharding indices, and then stack the tensors
    ## for every key name as AAAA.BBBB.CCCC.*.XXXX_<sharding_idx>, we extract the table name as AAAA.BBBB.CCCC.*.XXXX, and idx as <idx>
    ## we build a dict like {table_name: [{'idx': <idx>, 'tensor': tensor}]}
    ## then we sort the table_name dict by the idx, and stack the tensors
    table_sharding_key_dict = {}  # {table_name: [{'idx': <idx>, 'tensor': tensor}]}
    for state_dict_key, state_dict_value in sharded_return_state_dict.items():
        table_name = "_".join(state_dict_key.split("_")[0:-1])
        sharding_idx = int(state_dict_key.split("_")[-1])

        if table_name not in table_sharding_key_dict:
            table_sharding_key_dict[table_name] = []
        table_sharding_key_dict[table_name].append(
            {"idx": sharding_idx, "tensor": state_dict_value}
        )
    # sort the second level key list
    for table_name in table_sharding_key_dict:
        table_sharding_key_dict[table_name].sort(key=lambda x: x["idx"])
    # stack the tensors into init_model_state_dict
    merged_state_dict = OrderedDict()  # {first_level_key: tensor}
    for table_name in table_sharding_key_dict:
        if not is_with_metadata and "metadata" in table_name:
            continue
        stacked_tensor_list = []
        for table_sharding_key_dict_item in table_sharding_key_dict[table_name]:
            stacked_tensor_list.append(table_sharding_key_dict_item["tensor"])

        # _hash_zch_bucket should not be concatenated as it's a scalar configuration value
        # that should be the same across all ranks
        if "_hash_zch_bucket" in table_name:
            merged_state_dict[table_name] = stacked_tensor_list[0]
        else:
            merged_state_dict[table_name] = torch.concat(stacked_tensor_list)
    return merged_state_dict


def _run_single_rank_training_step(
    tables: List[EmbeddingBagConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    return_dict: Dict[str, Any],
    return_loss: Dict[str, Any],
    local_size: Optional[int] = None,
    use_mpzch: bool = False,
    kernel_type: Optional[str] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)

        train_model = SparseArch(
            tables=tables,
            device=torch.device("cuda"),
            return_remapped=False,
            use_mpzch=use_mpzch,
        )

        train_sharding_plan = construct_module_sharding_plan(
            train_model._mc_ebc,
            per_param_sharding={
                "table_0": row_wise(compute_kernel=kernel_type),
                "table_1": row_wise(compute_kernel=kernel_type),
            },
            local_size=local_size,
            world_size=world_size,
            device_type="cuda",
            sharder=sharder,
        )
        sharded_train_model = _shard_modules(
            module=copy.deepcopy(train_model),
            plan=ShardingPlan({"_mc_ebc": train_sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        # Forward Pass
        loss, _ = sharded_train_model(kjt_input.to(ctx.device))
        return_loss[f"loss_{rank}"] = loss.item()

        # Store managed collision module state
        mc_state = (
            # pyrefly: ignore[missing-attribute]
            sharded_train_model._mc_ebc._managed_collision_collection._managed_collision_modules.state_dict()
        )
        for key, value in mc_state.items():
            return_dict[
                f"_mc_ebc._managed_collision_collection._managed_collision_modules.{key}_{rank}"
            ] = value.cpu()

        # Store embedding bag collection state
        # pyrefly: ignore[missing-attribute]
        ebc_state = sharded_train_model._mc_ebc._embedding_bag_collection.state_dict()
        for key, value in ebc_state.items():
            tensors = []
            for i in range(len(value.local_shards())):
                tensors.append(value.local_shards()[i].tensor.cpu())
            return_dict[f"_mc_ebc._embedding_module.{key}_{rank}"] = torch.cat(
                tensors, dim=0
            )

        loss.backward()  # Test if backward pass gave no errors


@skip_if_asan_class
class ShardedMCEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        backend=st.sampled_from(["nccl"]),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(deadline=None)
    def test_mc_zch_with_sharded_versus_unsharded_vbe(
        self, backend: str, kernel_type: str
    ) -> None:
        WORLD_SIZE = 2
        embedding_bag_config: Final[List[EmbeddingBagConfig]] = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=64,
                num_embeddings=1000,
                data_type=DataType.FP16,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        global_input, local_inputs = ModelInput.generate_variable_batch_input(
            average_batch_size=10,
            world_size=WORLD_SIZE,
            num_float_features=0,
            tables=embedding_bag_config,
            weighted_tables=None,
            pooling_avg=5,
            global_constant_batch=False,
            use_offsets=False,
            random_seed=100,
        )

        # Extract global and local KJT from ModelInput
        global_kjt = global_input.idlist_features
        kjt_input_per_rank = [mi.idlist_features for mi in local_inputs]

        # Create dictionaries to store sharded model state
        train_state_dict = multiprocessing.Manager().dict()
        return_loss = multiprocessing.Manager().dict()

        self._run_multi_process_test(
            callable=_run_single_rank_training_step,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=ManagedCollisionEmbeddingBagCollectionSharder(),
            return_dict=train_state_dict,
            return_loss=return_loss,
            backend=backend,
            use_mpzch=True,
            kernel_type=kernel_type,
        )

        merged_state_dict = _merge_sharded_return_state_dict(
            # pyrefly: ignore[bad-argument-type]
            train_state_dict,
            is_with_metadata=False,
        )

        # Global model that loads sharded model state
        unsharded_model = SparseArch(
            tables=embedding_bag_config,
            device=torch.device("cpu"),
            return_remapped=False,
            use_mpzch=True,
            is_inference=True,
        )
        unsharded_model.load_state_dict(merged_state_dict)

        # Run forward passes on the unsharded model on local inputs
        unsharded_losses = []
        for kjt_input in kjt_input_per_rank:
            loss, _ = unsharded_model(kjt_input)
            unsharded_losses.append(loss.item())

        # Compare losses from sharded vs unsharded models
        for i in range(WORLD_SIZE):
            torch.testing.assert_close(
                torch.tensor(return_loss[f"loss_{i}"]),
                torch.tensor(unsharded_losses[i]),
                msg=f"Rank {i} Sharded model loss {return_loss[f'loss_{i}']} does not match "
                "unsharded model loss {unsharded_losses[i]}",
            )
        # Run forward pass on the unsharded model with global input
        global_loss, _ = unsharded_model(global_kjt)

        # Compute expected global loss from sharded model per-rank losses
        sharded_losses = [return_loss[f"loss_{i}"] for i in range(WORLD_SIZE)]
        expected_global_loss = sum(sharded_losses) / len(sharded_losses)

        # Compare global loss from unsharded model vs averaged sharded losses
        torch.testing.assert_close(
            global_loss,
            torch.tensor(expected_global_loss),
            msg=f"Unsharded global loss {global_loss.item()} does not match averaged sharded losses {expected_global_loss}",
        )
