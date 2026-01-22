#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from functools import partial
from typing import Any, Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import assume, given, settings, Verbosity
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec
from torch.distributed._tensor.api import DTensor
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import distributed as trec_dist
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.shard import shard
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    ModuleSharder,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import (
    assert_state_buffers_parameters_equal,
    skip_if_asan_class,
)


def _optional_equals(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]) -> bool:
    if t1 is None:
        return t2 is None
    return t2 is not None and torch.equal(t1, t2)


def _test_sharding(  # noqa C901
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    is_data_parallel: bool = False,
    use_apply_optimizer_in_backward: bool = False,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )
        unsharded_model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        if use_apply_optimizer_in_backward:
            apply_optimizer_in_backward(
                torch.optim.SGD,
                model.embedding_bags["table_0"].parameters(),
                {"lr": 1.0},
            )
            apply_optimizer_in_backward(
                torch.optim.SGD,
                model.embedding_bags["table_1"].parameters(),
                {"lr": 4.0},
            )
            apply_optimizer_in_backward(
                torch.optim.SGD,
                unsharded_model.embedding_bags["table_0"].parameters(),
                {"lr": 1.0},
            )
            apply_optimizer_in_backward(
                torch.optim.SGD,
                unsharded_model.embedding_bags["table_1"].parameters(),
                {"lr": 4.0},
            )
        plan: ShardingPlan = planner.collective_plan(model, [sharder], ctx.pg)
        sharded_model = shard(
            module=model,
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan.get_plan_for_module(""),
            sharder=sharder,
            device=ctx.device,
        )

        if not use_apply_optimizer_in_backward:
            unsharded_model_optimizer = torch.optim.SGD(
                unsharded_model.parameters(), lr=0.01
            )
            sharded_model_optimizer = torch.optim.SGD(
                sharded_model.parameters(), lr=0.01
            )

        assert isinstance(sharded_model, ShardedEmbeddingBagCollection)

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        for _it in range(5):
            unsharded_model_params = dict(unsharded_model.named_parameters())

            if not use_apply_optimizer_in_backward:
                unsharded_model_optimizer.zero_grad()
                sharded_model_optimizer.zero_grad()

            if is_data_parallel:
                for fqn, param in sharded_model.named_parameters():
                    assert _optional_equals(
                        param.grad, unsharded_model_params[fqn].grad
                    )

            unsharded_model_pred_kt = []
            for unsharded_rank in range(ctx.world_size):
                # simulate the unsharded model run on the entire batch
                unsharded_model_pred_kt.append(
                    unsharded_model(kjt_input_per_rank[unsharded_rank])
                )

            all_unsharded_preds = []
            for unsharded_rank in range(ctx.world_size):
                unsharded_model_pred_kt_mini_batch = unsharded_model_pred_kt[
                    unsharded_rank
                ].to_dict()

                all_unsharded_preds.extend(
                    [
                        unsharded_model_pred_kt_mini_batch[feature]
                        for feature in feature_keys
                    ]
                )
                if unsharded_rank == ctx.rank:
                    unsharded_model_pred = torch.stack(
                        [
                            unsharded_model_pred_kt_mini_batch[feature]
                            for feature in feature_keys
                        ]
                    )
            # sharded model
            # each rank gets a subbatch
            sharded_model_pred_kt = sharded_model(
                kjt_input_per_rank[ctx.rank]
            ).to_dict()
            sharded_model_pred = torch.stack(
                [sharded_model_pred_kt[feature] for feature in feature_keys]
            )

            # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
            # in normal author modelling code this won't be an issue because each rank would individually create
            # their model. output from sharded_pred is correctly on the correct device.
            # Compare predictions of sharded vs unsharded models.
            torch.testing.assert_close(
                sharded_model_pred.cpu(), unsharded_model_pred.cpu()
            )

            sharded_model_pred.sum().backward()

            all_unsharded_preds = torch.stack(all_unsharded_preds)
            _sum = all_unsharded_preds.sum()
            if is_data_parallel:
                _sum /= world_size
            _sum.backward()

            if is_data_parallel:
                for fqn, param in sharded_model.named_parameters():
                    assert _optional_equals(
                        param.grad, unsharded_model_params[fqn].grad
                    )

            if not use_apply_optimizer_in_backward:
                unsharded_model_optimizer.step()
                sharded_model_optimizer.step()

        # check nn.Module APIs look the same
        assert_state_buffers_parameters_equal(unsharded_model, sharded_model)

        for fqn in unsharded_model.state_dict():
            unsharded_state = unsharded_model.state_dict()[fqn]
            sharded_state = sharded_model.state_dict()[fqn]
            if is_data_parallel:
                torch.testing.assert_close(unsharded_state, sharded_state)
            else:
                out = (
                    torch.zeros(size=unsharded_state.shape, device=ctx.device)
                    if ctx.rank == 0
                    else None
                )
                if isinstance(sharded_state, DTensor):
                    out = sharded_state.full_tensor()
                else:
                    sharded_state.gather(out=out)

                if ctx.rank == 0:
                    torch.testing.assert_close(
                        unsharded_state,
                        out,
                    )


class TestEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class LookupOrderByShardTypeTest(unittest.TestCase):
    """Tests for _sharded_module_order_overwrite functionality in ShardedEmbeddingBagCollection."""

    def _create_sharding_spec(
        self, sharding_type: str, num_embeddings: int, embedding_dim: int
    ):
        """
        Create a mock sharding spec with proper shards for the given sharding type.
        Different sharding types have different requirements.
        """

        if sharding_type == ShardingType.DATA_PARALLEL.value:
            return None

        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[num_embeddings, embedding_dim],
                placement="rank:0/cpu",
            )
        ]
        return EnumerableShardingSpec(shards=shards)

    def _create_ebc_and_sharding_plan(
        self,
        sharding_types: List[str],
    ) -> tuple:
        """
        Helper to create an EmbeddingBagCollection and parameter sharding plan
        with multiple tables assigned to different sharding types.
        """
        from torchrec.distributed.types import ParameterSharding

        num_embeddings = 100
        embedding_dim = 16
        tables = []
        table_name_to_parameter_sharding: Dict[str, ParameterSharding] = {}

        for i, sharding_type in enumerate(sharding_types):
            table_name = f"table_{i}"
            tables.append(
                EmbeddingBagConfig(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    name=table_name,
                    feature_names=[f"feature_{i}"],
                )
            )
            table_name_to_parameter_sharding[table_name] = ParameterSharding(
                sharding_type=sharding_type,
                compute_kernel="fused",
                ranks=[0],
                sharding_spec=self._create_sharding_spec(
                    sharding_type, num_embeddings, embedding_dim
                ),
            )

        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        return ebc, table_name_to_parameter_sharding

    def _create_mock_sharding_env(self) -> ShardingEnv:
        """Create a mock ShardingEnv for testing."""
        return ShardingEnv(
            world_size=1,
            rank=0,
            pg=None,
        )

    @unittest.mock.patch.object(
        ShardedEmbeddingBagCollection, "_create_lookups", lambda self: None
    )
    @unittest.mock.patch.object(
        ShardedEmbeddingBagCollection, "_create_output_dist", lambda self: None
    )
    def test_sharding_types_ordered_by_lookup_order(self) -> None:
        """
        Test that _sharding_types are ordered according to sharded_module_order_overwrite.
        """
        custom_order = [
            ShardingType.COLUMN_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.TABLE_WISE.value,
        ]

        ebc, table_name_to_parameter_sharding = self._create_ebc_and_sharding_plan(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        )
        env = self._create_mock_sharding_env()

        sharded_ebc = ShardedEmbeddingBagCollection(
            module=ebc,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            env=env,
            device=torch.device("meta"),
            sharded_module_order_overwrite=custom_order,
        )

        self.assertEqual(
            sharded_ebc._sharding_types,
            [
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_WISE.value,
            ],
        )

    @unittest.mock.patch.object(
        ShardedEmbeddingBagCollection, "_create_lookups", lambda self: None
    )
    @unittest.mock.patch.object(
        ShardedEmbeddingBagCollection, "_create_output_dist", lambda self: None
    )
    def test_sharding_types_not_in_lookup_order_appended(self) -> None:
        """
        Test that sharding types not in sharded_module_order_overwrite are appended
        after the ordered ones.
        """
        custom_order = [
            ShardingType.COLUMN_WISE.value,
        ]

        ebc, table_name_to_parameter_sharding = self._create_ebc_and_sharding_plan(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        )
        env = self._create_mock_sharding_env()

        sharded_ebc = ShardedEmbeddingBagCollection(
            module=ebc,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            env=env,
            device=torch.device("meta"),
            sharded_module_order_overwrite=custom_order,
        )

        # COLUMN_WISE should be first since it's in the lookup order
        self.assertEqual(sharded_ebc._sharding_types[0], ShardingType.COLUMN_WISE.value)
        # Other types should be present
        self.assertIn(ShardingType.TABLE_WISE.value, sharded_ebc._sharding_types)
        self.assertIn(ShardingType.ROW_WISE.value, sharded_ebc._sharding_types)

    @unittest.mock.patch.object(
        ShardedEmbeddingBagCollection, "_create_lookups", lambda self: None
    )
    @unittest.mock.patch.object(
        ShardedEmbeddingBagCollection, "_create_output_dist", lambda self: None
    )
    def test_lookup_order_with_unused_sharding_types(self) -> None:
        """
        Test that sharding types in lookup_order that are not used are skipped.
        """
        custom_order = [
            ShardingType.GRID_SHARD.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.TABLE_WISE.value,
        ]

        ebc, table_name_to_parameter_sharding = self._create_ebc_and_sharding_plan(
            [ShardingType.COLUMN_WISE.value]
        )
        env = self._create_mock_sharding_env()

        sharded_ebc = ShardedEmbeddingBagCollection(
            module=ebc,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            env=env,
            device=torch.device("meta"),
            sharded_module_order_overwrite=custom_order,
        )

        # Only COLUMN_WISE should be in _sharding_types
        self.assertEqual(sharded_ebc._sharding_types, [ShardingType.COLUMN_WISE.value])


@skip_if_asan_class
class ShardedEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
        use_apply_optimizer_in_backward=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_sharding_ebc(
        self,
        sharding_type: str,
        use_apply_optimizer_in_backward: bool,
    ) -> None:

        # TODO DistributedDataParallel needs full support of registering fused optims before we can enable this.
        assume(
            not (
                use_apply_optimizer_in_backward
                and sharding_type == ShardingType.DATA_PARALLEL.value
            ),
        )

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=4,
                num_embeddings=4,
                init_fn=partial(torch.nn.init.normal_, mean=0.0, std=1.5),
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=4,
                num_embeddings=4,
                init_fn=partial(torch.nn.init.uniform_, a=-0.036, b=0.036),
            ),
        ]

        constraints = {
            "table_0": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LRU,
                    load_factor=0.1,
                    reserved_memory=8.0,
                    precision=DataType.FP16,
                ),
                enforce_hbm=True,
                stochastic_rounding=False,
                bounds_check_mode=BoundsCheckMode.IGNORE,
            ),
            "table_1": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.2,
                    reserved_memory=0.0,
                    precision=DataType.FP16,
                ),
                enforce_hbm=False,
                stochastic_rounding=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            ),
        }

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
        # "feature_1"   [2, 3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1]),
            ),
        ]
        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            initial_state_dict={
                "embedding_bags.table_0.weight": torch.Tensor(
                    [
                        [1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [4, 4, 4, 4],
                        [8, 8, 8, 8],
                    ]
                ),
                "embedding_bags.table_1.weight": torch.Tensor(
                    [
                        [101, 101, 101, 101],
                        [102, 102, 102, 102],
                        [104, 104, 104, 104],
                        [108, 108, 108, 108],
                    ]
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=TestEmbeddingBagCollectionSharder(sharding_type=sharding_type),
            backend=(
                "nccl"
                if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
                else "gloo"
            ),
            constraints=constraints,
            is_data_parallel=(sharding_type == ShardingType.DATA_PARALLEL.value),
            use_apply_optimizer_in_backward=use_apply_optimizer_in_backward,
        )
