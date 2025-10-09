#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import random
import unittest

from typing import Any, Dict, List, Optional

import hypothesis.strategies as st

import torch

from hypothesis import assume, given, settings, Verbosity

from torch import nn, optim

from torchrec import (
    distributed as trec_dist,
    EmbeddingBagCollection,
    KeyedJaggedTensor,
    optim as trec_optim,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.sharding.dynamic_sharding import (
    output_sharding_plan_delta_single,
)

from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    get_module_to_default_sharders,
    table_wise,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    create_test_sharder,
    generate_rank_placements,
    SharderType,
)

from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import data_type_to_dtype, EmbeddingBagConfig

from torchrec.test_utils import skip_if_asan_class
from torchrec.types import DataType


# Utils:
def table_name(i: int) -> str:
    return "table_" + str(i)


def feature_name(i: int) -> str:
    return "feature_" + str(i)


def generate_embedding_bag_config(
    data_type: DataType,
    num_tables: int = 3,
    embedding_dim: int = 16,
    num_embeddings: int = 4,
) -> List[EmbeddingBagConfig]:
    embedding_bag_config = []
    for i in range(num_tables):
        embedding_bag_config.append(
            EmbeddingBagConfig(
                name=table_name(i),
                feature_names=[feature_name(i)],
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                data_type=data_type,
            ),
        )
    return embedding_bag_config


def create_test_initial_state_dict(
    sharded_module_type: nn.Module,
    num_tables: int,
    data_type: DataType,
    embedding_dim: int = 16,
    num_embeddings: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Helpful for debugging:

    initial_state_dict = {
        "embedding_bags.table_0.weight": torch.tensor(
            [
                [1] * 16,
                [2] * 16,
                [3] * 16,
                [4] * 16,
            ],
        ),
        "embedding_bags.table_1.weight": torch.tensor(
            [
                [101] * 16,
                [102] * 16,
                [103] * 16,
                [104] * 16,
            ],
            dtype=data_type_to_dtype(data_type),
        ),
        ...
    }
    """

    initial_state_dict = {}
    for i in range(num_tables):
        # pyre-ignore
        extended_name = sharded_module_type.extend_shard_name(table_name(i))
        initial_state_dict[extended_name] = torch.tensor(
            [[j + (i * 100)] * embedding_dim for j in range(num_embeddings)],
            dtype=data_type_to_dtype(data_type),
        )

    return initial_state_dict


def are_sharded_ebc_modules_identical(
    module1: ShardedEmbeddingBagCollection,
    module2: ShardedEmbeddingBagCollection,
) -> None:
    # Check if both modules have the same parameters
    params1 = list(module1.named_parameters())
    params2 = list(module2.named_parameters())

    assert len(params1) == len(params2)

    for param1, param2 in zip(params1, params2):
        # Check parameter names
        assert param1[0] == param2[0]
        # Check parameter values
        assert torch.allclose(param1[1], param2[1])

    # Check if both modules have the same buffers
    buffers1 = list(module1.named_buffers())
    buffers2 = list(module2.named_buffers())

    assert len(buffers1) == len(buffers2)

    for buffer1, buffer2 in zip(buffers1, buffers2):
        assert buffer1[0] == buffer2[0]  # Check buffer names
        assert torch.allclose(buffer1[1], buffer2[1])  # Check buffer values

    # Hard-coded attributes for EmbeddingBagCollection
    attribute_list = [
        "_module_fqn",
        "_table_names",
        "_pooling_type_to_rs_features",
        "_output_dtensor",
        "_sharding_types",
        "_is_weighted",
        "_embedding_names",
        "_embedding_dims",
        "_feature_splits",
        "_features_order",
        "_uncombined_embedding_names",
        "_uncombined_embedding_dims",
        "_has_mean_pooling_callback",
        "_kjt_key_indices",
        "_has_uninitialized_input_dist",
        "_has_features_permute",
        "_dim_per_key",  # Tensor
        "_inverse_indices_permute_indices",  # Tensor
        "_kjt_inverse_order",  # Tensor
        "_kt_key_ordering",  # Tensor
        # Non-primitive types which can be compared
        "module_sharding_plan",
        "_table_name_to_config",
        # Excluding the non-primitive types that cannot be compared
        # "sharding_type_to_sharding_infos",
        # "_embedding_shardings"
        # "_input_dists",
        # "_lookups",
        # "_output_dists",
        # "_optim",
    ]

    for attr in attribute_list:
        assert hasattr(module1, attr) and hasattr(module2, attr)

        val1 = getattr(module1, attr)
        val2 = getattr(module2, attr)

        assert type(val1) is type(val2)
        if isinstance(val1, torch.Tensor):
            torch.testing.assert_close(val1, val2)
        else:
            assert val1 == val2


def _test_ebc_resharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    module_sharding_plan: EmbeddingModuleShardingPlan,
    new_module_sharding_plan: EmbeddingModuleShardingPlan,
    local_size: Optional[int] = None,
) -> None:
    """
    Distributed call to test resharding for ebc by creating 2 models with identical config and
    states:
        m1 sharded with new_module_sharding_plan
        m2 sharded with module_sharding_plan, then resharded with new_module_sharding_plan

    Expects m1 and resharded m2 to be the same, and predictions outputted from the same KJT
    inputs to be the same.

    TODO: modify to include other modules once dynamic sharding is built out.
    """
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        # Set seed to be 0 to ensure models have the same initialization across ranks
        torch.manual_seed(0)
        m1 = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        m2 = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )
        if initial_state_dict is not None:
            initial_state_dict = {
                fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
            }

            # Load initial State - making sure models are identical
            m1.load_state_dict(initial_state_dict)

            m2.load_state_dict(initial_state_dict)

        else:
            # Note this is the only correct behavior due to setting random seed to 0 above
            # Otherwise the weights generated in EBC initialization will be different on
            # Each rank, resulting in different behavior after resharding
            copy_state_dict(
                loc=m2.state_dict(),
                glob=m1.state_dict(),
            )

        sharder = get_module_to_default_sharders()[type(m1)]

        # pyre-ignore
        env = ShardingEnv.from_process_group(ctx.pg)

        sharded_m1 = sharder.shard(
            module=m1,
            params=new_module_sharding_plan,
            env=env,
            device=ctx.device,
        )

        sharded_m2 = sharder.shard(
            module=m1,
            params=module_sharding_plan,
            env=env,
            device=ctx.device,
        )

        _, new_module_sharding_plan_delta = output_sharding_plan_delta_single(
            module_sharding_plan, new_module_sharding_plan
        )

        # pyre-ignore
        resharded_m2 = sharder.reshard(
            sharded_module=sharded_m2,
            changed_shard_to_params=new_module_sharding_plan_delta,
            env=env,
            device=ctx.device,
        )

        are_sharded_ebc_modules_identical(sharded_m1, resharded_m2)

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        # For current test model and inputs, the prediction should be the exact same
        # rtol = 0
        # atol = 0

        for _ in range(world_size):
            # sharded model
            # each rank gets a subbatch
            sharded_m1_pred_kt_no_dict = sharded_m1(kjt_input_per_rank[ctx.rank])
            resharded_m2_pred_kt_no_dict = resharded_m2(kjt_input_per_rank[ctx.rank])

            sharded_m1_pred_kt = sharded_m1_pred_kt_no_dict.to_dict()
            resharded_m2_pred_kt = resharded_m2_pred_kt_no_dict.to_dict()
            sharded_m1_pred = torch.stack(
                [sharded_m1_pred_kt[feature] for feature in feature_keys]
            )

            resharded_m2_pred = torch.stack(
                [resharded_m2_pred_kt[feature] for feature in feature_keys]
            )
            # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
            # in normal author modelling code this won't be an issue because each rank would individually create
            # their model. output from sharded_pred is correctly on the correct device.

            # Compare predictions of sharded vs unsharded models.
            torch.testing.assert_close(sharded_m1_pred.cpu(), resharded_m2_pred.cpu())

            sharded_m1_pred.sum().backward()
            resharded_m2_pred.sum().backward()


@skip_if_asan_class
class MultiRankEBCDynamicShardingTest(MultiProcessTestBase):
    def _run_ebc_resharding_test(
        self,
        per_param_sharding: Dict[str, ParameterSharding],
        new_per_param_sharding: Dict[str, ParameterSharding],
        num_tables: int,
        world_size: int,
        data_type: DataType,
        embedding_dim: int = 16,
        num_embeddings: int = 4,
        use_debug_state_dict: bool = False,  # Turn on to use dummy values for initial state dict
    ) -> None:
        embedding_bag_config = generate_embedding_bag_config(
            data_type, num_tables, embedding_dim, num_embeddings
        )

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            # pyre-ignore
            per_param_sharding=per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            # pyre-ignore
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Row-wise not supported on gloo
        if (
            not torch.cuda.is_available()
            and new_module_sharding_plan["table_0"].sharding_type
            == ShardingType.ROW_WISE.value
        ):
            return

        kjt_input_per_rank = [
            ModelInput.create_standard_kjt(
                batch_size=2,
                tables=embedding_bag_config,
            )
            for _ in range(world_size)
        ]

        initial_state_dict = None
        if use_debug_state_dict:
            # initial_state_dict filled with deterministic dummy values
            initial_state_dict = create_test_initial_state_dict(
                ShardedEmbeddingBagCollection,  # pyre-ignore
                num_tables,
                data_type,
                embedding_dim,
                num_embeddings,
            )

        self._run_multi_process_test(
            callable=_test_ebc_resharding,
            world_size=world_size,
            tables=embedding_bag_config,
            initial_state_dict=initial_state_dict,
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl" if torch.cuda.is_available() else "gloo",
            module_sharding_plan=module_sharding_plan,
            new_module_sharding_plan=new_module_sharding_plan,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(  # pyre-ignore
        num_tables=st.sampled_from([2, 3, 4]),
        data_type=st.sampled_from([DataType.FP32, DataType.FP16]),
        world_size=st.sampled_from([2, 4]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_dynamic_sharding_ebc_tw(
        self,
        num_tables: int,
        data_type: DataType,
        world_size: int,
    ) -> None:
        # Tests EBC dynamic sharding implementation for TW

        # Table wise can only have 1 rank allocated per table:
        ranks_per_tables = [1 for _ in range(num_tables)]
        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)

        while new_ranks == old_ranks:
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = table_wise(rank=old_ranks[i][0])
            new_per_param_sharding[table_name(i)] = table_wise(rank=new_ranks[i][0])

        self._run_ebc_resharding_test(
            per_param_sharding,
            new_per_param_sharding,
            num_tables,
            world_size,
            data_type,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(  # pyre-ignore
        num_tables=st.sampled_from([2, 3, 4]),
        data_type=st.sampled_from([DataType.FP32, DataType.FP16]),
        world_size=st.sampled_from([3, 4]),
        embedding_dim=st.sampled_from([16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_dynamic_sharding_ebc_cw(
        self,
        num_tables: int,
        data_type: DataType,
        world_size: int,
        embedding_dim: int,
    ) -> None:
        # Tests EBC dynamic sharding implementation for CW

        # Force the ranks per table to be consistent
        valid_ranks = [i for i in range(1, world_size) if embedding_dim % i == 0]
        ranks_per_tables = [random.choice(valid_ranks) for _ in range(num_tables)]
        new_ranks_per_tables = [random.choice(valid_ranks) for _ in range(num_tables)]

        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size

        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(
            world_size, num_tables, new_ranks_per_tables
        )

        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        while new_ranks == old_ranks:
            old_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = column_wise(ranks=old_ranks[i])
            new_per_param_sharding[table_name(i)] = column_wise(ranks=new_ranks[i])

        self._run_ebc_resharding_test(
            per_param_sharding,
            new_per_param_sharding,
            num_tables,
            world_size,
            data_type,
            embedding_dim,
        )


@skip_if_asan_class
class MultiRankDMPDynamicShardingTest(ModelParallelTestShared):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(  # Pyre-ignore
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
            ]
        ),
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        random_seed=st.integers(0, 1000),
        world_size=st.sampled_from([2, 4]),
        skip_passing_resharding_fqn=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        data_type: DataType,
        random_seed: int,
        world_size: int,
        skip_passing_resharding_fqn: bool,
    ) -> None:
        """
        Tests resharding from DMP module interface with conditional optimizer selection:
        - For Table-Wise sharding: All optimizers including RowWiseAdagrad
        - For Column-Wise sharding: Only Adagrad and SGD (no RowWiseAdagrad)
        """
        assume(
            self.device != torch.device("cpu")
            or kernel_type == EmbeddingComputeKernel.FUSED.value
        )

        # Fixed to False as variable batch size with CW is more complex
        variable_batch_size = False

        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )

        # Define optimizer options based on sharding type
        if sharding_type == ShardingType.COLUMN_WISE.value:
            # For Column-Wise sharding, exclude RowWiseAdagrad
            # TODO: fix rowise adagrad logic similar to 2D sharding
            optimizer_options = [
                None,
                {"embedding_bags": (optim.Adagrad, {"lr": 0.04})},
                {"embedding_bags": (torch.optim.SGD, {"lr": 0.01})},
            ]
        else:
            # For Table-Wise sharding, include all optimizers
            optimizer_options = [
                None,
                {"embedding_bags": (optim.Adagrad, {"lr": 0.04})},
                {"embedding_bags": (torch.optim.SGD, {"lr": 0.01})},
                {"embedding_bags": (trec_optim.RowWiseAdagrad, {"lr": 0.01})},
            ]

        # Randomly select one optimizer from the appropriate options
        apply_optimizer_in_backward_config = random.choice(optimizer_options)

        sharding_type_e = ShardingType(sharding_type)

        self._test_dynamic_sharding(
            sharders=[  # Pyre-ignore
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=self.device,
                ),
            ],
            backend=self.backend,
            qcomms_config=qcomms_config,
            # Pyre-ignore
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            data_type=data_type,
            sharding_type=sharding_type_e,
            random_seed=random_seed,
            world_size=world_size,
            skip_passing_resharding_fqn=skip_passing_resharding_fqn,
        )


class SingleRankDynamicShardingUtilsTest(unittest.TestCase):
    def test_output_sharding_plan_delta_single(self) -> None:
        """
        Tests output_sharding_plan_delta function
        """
        num_tables = 2
        # NOTE: even though this is a single rank DS test, setting world_size to 2 here to check
        # output_sharding_plan_delta function to see if it works correctly at pruning a sharding plan.
        # Since we don't actually run resharding in this UT, no need to ensure world_size is exactly as ranks used.
        world_size = 2
        data_type = DataType.FP32
        embedding_dim = 16
        num_embeddings = 4

        # Table wise can only have 1 rank allocated per table:
        ranks_per_tables = [1 for _ in range(num_tables)]
        # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)

        while new_ranks == old_ranks:
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )
        per_param_sharding = {}
        new_per_param_sharding = {}

        # Construct parameter shardings
        for i in range(num_tables):
            per_param_sharding[table_name(i)] = table_wise(rank=old_ranks[i][0])
            new_per_param_sharding[table_name(i)] = table_wise(rank=new_ranks[i][0])

        embedding_bag_config = generate_embedding_bag_config(
            data_type, num_tables, embedding_dim, num_embeddings
        )

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        _, new_module_sharding_plan_delta = output_sharding_plan_delta_single(
            module_sharding_plan, new_module_sharding_plan
        )

        assert len(new_module_sharding_plan_delta) <= len(new_module_sharding_plan)

        # using t_name instead of table_name to avoid clashing with helper method
        for t_name, new_sharding in new_module_sharding_plan.items():
            if new_sharding.ranks != module_sharding_plan[t_name].ranks:
                assert t_name in new_module_sharding_plan_delta
                assert (
                    new_module_sharding_plan_delta[t_name].ranks == new_sharding.ranks
                )
                assert (
                    new_module_sharding_plan_delta[t_name].sharding_type
                    == new_sharding.sharding_type
                )
                assert (
                    new_module_sharding_plan_delta[t_name].compute_kernel
                    == new_sharding.compute_kernel
                )
                # NOTE there are other attributes to test for equivalence in ParameterSharding type
                # but the ones included here are the most important.

    def test_output_sharding_plans_delta(self) -> None:
        """
        Tests output_sharding_plan_delta_single with multiple tables and sharding types.
        """
        num_tables = 3
        world_size = 4
        data_type = DataType.FP32
        embedding_dim = 16
        num_embeddings = 8

        # Generate random ranks for table-wise and column-wise sharding
        # Table 0: Table-wise, Table 1: Column-wise, Table 2: Table-wise
        ranks_per_tables = [1, 2, 1]
        old_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        new_ranks = generate_rank_placements(world_size, num_tables, ranks_per_tables)
        # Ensure at least one table changes ranks
        while new_ranks == old_ranks:
            new_ranks = generate_rank_placements(
                world_size, num_tables, ranks_per_tables
            )

        per_param_sharding = {}
        new_per_param_sharding = {}

        # Table 0: Table-wise, Table 1: Column-wise, Table 2: Table-wise
        per_param_sharding[table_name(0)] = table_wise(rank=old_ranks[0][0])
        per_param_sharding[table_name(1)] = column_wise(ranks=old_ranks[1])
        per_param_sharding[table_name(2)] = table_wise(rank=old_ranks[2][0])

        new_per_param_sharding[table_name(0)] = table_wise(rank=new_ranks[0][0])
        new_per_param_sharding[table_name(1)] = column_wise(ranks=new_ranks[1])
        new_per_param_sharding[table_name(2)] = table_wise(rank=new_ranks[2][0])

        embedding_bag_config = generate_embedding_bag_config(
            data_type, num_tables, embedding_dim, num_embeddings
        )

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        new_module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        _, new_module_sharding_plan_delta = output_sharding_plan_delta_single(
            module_sharding_plan, new_module_sharding_plan
        )

        # The delta should only contain tables whose sharding/ranks changed
        for t_name, new_sharding in new_module_sharding_plan.items():
            old_sharding = module_sharding_plan[t_name]
            if (
                new_sharding.ranks != old_sharding.ranks
                or new_sharding.sharding_type != old_sharding.sharding_type
                or new_sharding.compute_kernel != old_sharding.compute_kernel
            ):
                assert t_name in new_module_sharding_plan_delta
                assert (
                    new_module_sharding_plan_delta[t_name].ranks == new_sharding.ranks
                )
                assert (
                    new_module_sharding_plan_delta[t_name].sharding_type
                    == new_sharding.sharding_type
                )
                assert (
                    new_module_sharding_plan_delta[t_name].compute_kernel
                    == new_sharding.compute_kernel
                )
            else:
                assert t_name not in new_module_sharding_plan_delta

        # The delta should not contain more keys than the new plan
        assert len(new_module_sharding_plan_delta) <= len(new_module_sharding_plan)
