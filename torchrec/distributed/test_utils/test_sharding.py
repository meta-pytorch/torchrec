#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import random
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Protocol, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    EmbeddingTableConfig,
)
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.model_parallel import DistributedModelParallel, DMPCollection
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.sharding.dynamic_sharding import output_sharding_plans_delta
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    get_sharding_constructor_from_type,
)
from torchrec.distributed.test_utils.emb_sharder import (
    TestEBCSharder,
    TestEBSharder,
    TestECSharder,
    TestETCSharder,
    TestETSharder,
)
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNNBase
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
)
from torchrec.distributed.types import (
    DMPCollectionConfig,
    EmbeddingModuleShardingPlan,
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    DataType,
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


class SharderType(Enum):
    EMBEDDING_BAG = "embedding_bag"
    EMBEDDING_BAG_COLLECTION = "embedding_bag_collection"
    EMBEDDING_TOWER = "embedding_tower"
    EMBEDDING_TOWER_COLLECTION = "embedding_tower_collection"
    EMBEDDING_COLLECTION = "embedding_collection"


def _gather_all_tensors(
    local_tensor: torch.Tensor,
    world_size: int,
    pg: Optional[dist.ProcessGroup] = None,
) -> List[torch.Tensor]:
    """
    Gathers tensors from all processes in a distributed group.

    This function collects tensors from all processes in the specified
    process group and returns a list of tensors, where each tensor
    corresponds to the data from one process.

    Args:
        local_tensor (torch.Tensor): The tensor to be gathered from the local process.
        world_size (int): The number of processes in the distributed group.
        pg (Optional[ProcessGroup]): The process group to use for communication.
                                                If not provided, a default ProcessGroup will be created.

    Returns:
        List[torch.Tensor]: A list of tensors gathered from all processes.
    """
    all_local_tensors: List[torch.Tensor] = []
    for _ in range(world_size):
        all_local_tensors.append(torch.empty_like(local_tensor))
    dist.all_gather(all_local_tensors, local_tensor, pg)
    return all_local_tensors


def create_test_sharder(
    sharder_type: str,
    sharding_type: str,
    kernel_type: str,
    fused_params: Optional[Dict[str, Any]] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    device: Optional[torch.device] = None,
) -> Union[TestEBSharder, TestEBCSharder, TestETSharder, TestETCSharder, TestECSharder]:
    if fused_params is None:
        fused_params = {}
    qcomm_codecs_registry = {}
    if qcomms_config is not None:
        qcomm_codecs_registry = get_qcomm_codecs_registry(qcomms_config, device=device)
    if "learning_rate" not in fused_params:
        fused_params["learning_rate"] = 0.1
    if sharder_type == SharderType.EMBEDDING_BAG.value:
        return TestEBSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value:
        return TestEBCSharder(
            sharding_type,
            kernel_type,
            fused_params,
            qcomm_codecs_registry,
        )
    elif sharder_type == SharderType.EMBEDDING_COLLECTION.value:
        return TestECSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_TOWER.value:
        return TestETSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_TOWER_COLLECTION.value:
        return TestETCSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    else:
        raise ValueError(f"Sharder not supported {sharder_type}")


class ModelInputCallable(Protocol):
    def __call__(
        self,
        batch_size: int,
        world_size: int,
        num_float_features: int,
        tables: Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]],
        weighted_tables: Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]],
        pooling_avg: int = 10,
        dedup_tables: Optional[
            Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]]
        ] = None,
        variable_batch_size: bool = False,
        use_offsets: bool = False,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        random_seed: Optional[int] = None,
    ) -> Tuple["ModelInput", List["ModelInput"]]: ...


class VariableBatchModelInputCallable(Protocol):
    def __call__(
        self,
        average_batch_size: int,
        world_size: int,
        num_float_features: int,
        tables: Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]],
        weighted_tables: Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]],
        pooling_avg: int = 10,
        global_constant_batch: bool = False,
        use_offsets: bool = False,
        indices_dtype: torch.dtype = torch.int64,
        offsets_dtype: torch.dtype = torch.int64,
        lengths_dtype: torch.dtype = torch.int64,
        random_seed: Optional[int] = None,
    ) -> Tuple["ModelInput", List["ModelInput"]]: ...


def gen_model_and_input(
    model_class: TestSparseNNBase,
    tables: List[EmbeddingTableConfig],
    embedding_groups: Dict[str, List[str]],
    world_size: int,
    # pyre-ignore [9]
    generate: Union[
        ModelInputCallable, VariableBatchModelInputCallable
    ] = ModelInput.generate,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    num_float_features: int = 16,
    dense_device: Optional[torch.device] = None,
    sparse_device: Optional[torch.device] = None,
    dedup_feature_names: Optional[List[str]] = None,
    dedup_tables: Optional[List[EmbeddingTableConfig]] = None,
    variable_batch_size: bool = False,
    batch_size: int = 4,
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    use_offsets: bool = False,
    indices_dtype: torch.dtype = torch.int64,
    offsets_dtype: torch.dtype = torch.int64,
    lengths_dtype: torch.dtype = torch.int64,
    global_constant_batch: bool = False,
    num_inputs: int = 1,
    input_type: str = "kjt",  # "kjt" or "td"
    random_seed: Optional[int] = None,
) -> Tuple[nn.Module, List[Tuple[ModelInput, List[ModelInput]]]]:
    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        torch.manual_seed(0)
    if dedup_feature_names:
        model = model_class(
            tables=cast(
                List[BaseEmbeddingConfig],
                tables + dedup_tables if dedup_tables else tables,
            ),
            num_float_features=num_float_features,
            dedup_feature_names=dedup_feature_names,
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
            feature_processor_modules=feature_processor_modules,
        )
    else:
        model = model_class(
            tables=cast(
                List[BaseEmbeddingConfig],
                tables,
            ),
            num_float_features=num_float_features,
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
            feature_processor_modules=feature_processor_modules,
        )
    inputs = []
    if input_type == "kjt" and generate == ModelInput.generate_variable_batch_input:
        for _ in range(num_inputs):
            inputs.append(
                cast(VariableBatchModelInputCallable, generate)(
                    average_batch_size=batch_size,
                    world_size=world_size,
                    num_float_features=num_float_features,
                    tables=tables,
                    weighted_tables=weighted_tables or [],
                    global_constant_batch=global_constant_batch,
                    use_offsets=use_offsets,
                    indices_dtype=indices_dtype,
                    offsets_dtype=offsets_dtype,
                    lengths_dtype=lengths_dtype,
                    random_seed=random_seed,
                )
            )
    elif generate == ModelInput.generate:
        for _ in range(num_inputs):
            inputs.append(
                ModelInput.generate(
                    world_size=world_size,
                    tables=tables,
                    dedup_tables=dedup_tables,
                    weighted_tables=weighted_tables or [],
                    num_float_features=num_float_features,
                    variable_batch_size=variable_batch_size,
                    batch_size=batch_size,
                    input_type=input_type,
                    use_offsets=use_offsets,
                    indices_dtype=indices_dtype,
                    offsets_dtype=offsets_dtype,
                    lengths_dtype=lengths_dtype,
                    random_seed=random_seed,
                )
            )
    else:
        for _ in range(num_inputs):
            inputs.append(
                cast(ModelInputCallable, generate)(
                    world_size=world_size,
                    tables=tables,
                    dedup_tables=dedup_tables,
                    weighted_tables=weighted_tables or [],
                    num_float_features=num_float_features,
                    variable_batch_size=variable_batch_size,
                    batch_size=batch_size,
                    use_offsets=use_offsets,
                    indices_dtype=indices_dtype,
                    offsets_dtype=offsets_dtype,
                    lengths_dtype=lengths_dtype,
                    random_seed=random_seed,
                )
            )
    return (model, inputs)


def dynamic_sharding_test(
    rank: int,
    world_size: int,
    model_class: TestSparseNNBase,
    embedding_groups: Dict[str, List[str]],
    tables: List[EmbeddingTableConfig],
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    optim: EmbOptimType,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    apply_optimizer_in_backward_config: Optional[
        Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
    ] = None,
    variable_batch_size: bool = False,  # variable batch per rank
    batch_size: int = 4,
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    variable_batch_per_feature: bool = False,  # VBE
    global_constant_batch: bool = False,
    world_size_2D: Optional[int] = None,  # 2D parallel
    node_group_size: Optional[int] = None,  # 2D parallel
    use_inter_host_allreduce: bool = False,  # 2D parallel
    input_type: str = "kjt",  # "kjt" or "td"
    allow_zero_batch_size: bool = False,
    custom_all_reduce: bool = False,  # 2D parallel
    use_offsets: bool = False,
    indices_dtype: torch.dtype = torch.int64,
    offsets_dtype: torch.dtype = torch.int64,
    lengths_dtype: torch.dtype = torch.int64,
    sharding_type: ShardingType = None,  # pyre-ignore
    random_seed: int = 0,
    skip_passing_resharding_fqn: bool = False,
) -> None:
    """
    Test case for dynamic sharding:
        1. Generate global model and inputs
        2. Create 2 identical local models based on global model
        3. Use planner to generate sharding plan for local model
        4. Based on planner output, generate a second, different sharding plan
        5. Shard both local models 1 and 2 through DMP with plan 1 and 2 respectively
        6. Reshard (dynamic sharding API) model 1 with plan 2
        7. Generate predictions for local models and compare them to global model prediction. Expect to be the same.

    For debugging specific differences in model 1 and 2, use compare_model_pred_one_step instead of generate_model_pred_one_step.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # TODO: support non-sharded forward with zero batch size KJT
        assert (
            not allow_zero_batch_size
        ), "Have not yet implemented non-sharded forward with zero batch size KJT"

        batch_size = (
            random.randint(0, batch_size) if allow_zero_batch_size else batch_size
        )
        num_steps = 2
        num_float_features = 16
        # Generate model & inputs.
        (global_model, inputs) = gen_model_and_input(
            model_class=model_class,
            tables=tables,
            # pyre-ignore [6]
            generate=(
                cast(
                    VariableBatchModelInputCallable,
                    ModelInput.generate_variable_batch_input,
                )
                if variable_batch_per_feature
                else ModelInput.generate
            ),
            # TODO: support weighted tables in the future
            # weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
            num_float_features=num_float_features,
            variable_batch_size=variable_batch_size,
            batch_size=batch_size,
            feature_processor_modules=feature_processor_modules,
            global_constant_batch=global_constant_batch,
            input_type=input_type,
            use_offsets=use_offsets,
            indices_dtype=indices_dtype,
            offsets_dtype=offsets_dtype,
            lengths_dtype=lengths_dtype,
            num_inputs=num_steps,
        )
        global_model = global_model.to(ctx.device)
        global_input_0 = inputs[0][0].to(ctx.device)
        local_input_0 = inputs[0][1][rank].to(ctx.device)

        global_input_1 = inputs[1][0].to(ctx.device)
        local_input_1 = inputs[1][1][rank].to(ctx.device)

        # Shard model.
        local_m1 = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            # TODO: support weighted tables in the future
            # weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=ctx.device,
            sparse_device=torch.device("meta"),
            num_float_features=num_float_features,
            feature_processor_modules=feature_processor_modules,
        )

        local_m2 = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            # TODO: support weighted tables in the future
            # weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=ctx.device,
            sparse_device=torch.device("meta"),
            num_float_features=num_float_features,
            feature_processor_modules=feature_processor_modules,
        )

        global_model_named_params_as_dict = dict(global_model.named_parameters())
        local_m1_named_params_as_dict = dict(local_m1.named_parameters())
        local_m2_named_params_as_dict = dict(local_m2.named_parameters())

        if apply_optimizer_in_backward_config is not None:
            for apply_optim_name, (
                optimizer_type,
                optimizer_kwargs,
            ) in apply_optimizer_in_backward_config.items():
                for name, param in global_model_named_params_as_dict.items():
                    if apply_optim_name not in name:
                        continue
                    assert name in local_m1_named_params_as_dict
                    assert name in local_m2_named_params_as_dict
                    local_m1_param = local_m1_named_params_as_dict[name]
                    local_m2_param = local_m2_named_params_as_dict[name]
                    apply_optimizer_in_backward(
                        optimizer_type,
                        [param],
                        optimizer_kwargs,
                    )
                    apply_optimizer_in_backward(
                        optimizer_type, [local_m1_param], optimizer_kwargs
                    )
                    apply_optimizer_in_backward(
                        optimizer_type, [local_m2_param], optimizer_kwargs
                    )

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,  # TODO world_size_2D gap
                compute_device=ctx.device.type,
                local_world_size=node_group_size if node_group_size else ctx.local_size,
            ),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.collective_plan(local_m1, sharders, ctx.pg)

        """
        Overwriting certain shard placements for certain sharding types
        TODO: Support following sharding types for dynamic sharding (currently will error out)
        """
        for group in plan.plan:
            for _, parameter_sharding in cast(
                EmbeddingModuleShardingPlan, plan.plan[group]
            ).items():
                if (
                    parameter_sharding.sharding_type
                    in {
                        ShardingType.TABLE_ROW_WISE.value,
                        ShardingType.TABLE_COLUMN_WISE.value,
                        ShardingType.GRID_SHARD.value,
                    }
                    and ctx.device.type != "cpu"
                ):
                    sharding_spec = parameter_sharding.sharding_spec
                    if sharding_spec is not None:
                        # pyre-ignore
                        for shard in sharding_spec.shards:
                            placement = shard.placement
                            rank: Optional[int] = placement.rank()
                            assert rank is not None
                            shard.placement = torch.distributed._remote_device(
                                f"rank:{rank}/cuda:{rank}"
                            )

        assert ctx.pg is not None

        num_tables = len(tables)

        ranks_per_tables = [1 for _ in range(num_tables)]
        new_ranks = generate_rank_placements(
            world_size, num_tables, ranks_per_tables, random_seed
        )

        ranks_per_tables_for_CW = []
        for table in tables:

            # CW sharding
            valid_candidates = [
                i for i in range(1, world_size + 1) if table.embedding_dim % i == 0
            ]
            ranks_per_tables_for_CW.append(random.choice(valid_candidates))

        new_ranks_cw = generate_rank_placements(
            world_size, num_tables, ranks_per_tables_for_CW, random_seed
        )

        new_per_param_sharding = {}

        assert len(sharders) == 1
        # pyre-ignore
        kernel_type = sharders[0]._kernel_type
        # Construct parameter shardings
        for i in range(num_tables):
            table_name = tables[i].name
            table_constraint = constraints[table_name]  # pyre-ignore
            assert hasattr(table_constraint, "sharding_types")
            assert (
                len(table_constraint.sharding_types) == 1
            ), "Dynamic Sharding currently only supports 1 sharding type per table"
            sharding_type = ShardingType(table_constraint.sharding_types[0])
            sharding_type_constructor = get_sharding_constructor_from_type(
                sharding_type
            )

            if sharding_type == ShardingType.TABLE_WISE:
                new_per_param_sharding[table_name] = sharding_type_constructor(
                    rank=new_ranks[i][0], compute_kernel=kernel_type
                )
            elif sharding_type == ShardingType.COLUMN_WISE:
                new_per_param_sharding[table_name] = sharding_type_constructor(
                    ranks=new_ranks_cw[i], compute_kernel=kernel_type
                )
            else:
                raise NotImplementedError(
                    f"Dynamic Sharding currently does not support {sharding_type}"
                )

        new_module_sharding_plan = construct_module_sharding_plan(
            local_m2.sparse.ebc,
            sharder=sharders[0],
            per_param_sharding=new_per_param_sharding,
            local_size=world_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        plan_1 = copy.deepcopy(plan)
        plan_1.plan["sparse.ebc"] = new_module_sharding_plan

        local_m1_dmp = DistributedModelParallel(
            local_m1,
            env=ShardingEnv.from_process_group(ctx.pg),  # pyre-ignore
            plan=plan,
            sharders=sharders,
            device=ctx.device,
        )

        local_m2_dmp = DistributedModelParallel(
            local_m2,
            env=ShardingEnv.from_process_group(ctx.pg),  # pyre-ignore
            plan=plan_1,
            sharders=sharders,
            device=ctx.device,
        )

        dense_m2_optim = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(local_m2_dmp.named_parameters())),
            lambda params: torch.optim.SGD(params, lr=0.1),
        )
        local_m2_opt = CombinedOptimizer([local_m2_dmp.fused_optimizer, dense_m2_optim])

        # Load model state from the global model.
        copy_state_dict(
            local_m1_dmp.state_dict(),
            global_model.state_dict(),
            exclude_predfix="sparse.pooled_embedding_arch.embedding_modules._itp_iter",
        )

        copy_state_dict(
            local_m2_dmp.state_dict(),
            global_model.state_dict(),
            exclude_predfix="sparse.pooled_embedding_arch.embedding_modules._itp_iter",
        )

        new_module_sharding_plan_delta = output_sharding_plans_delta(
            plan.plan, plan_1.plan  # pyre-ignore
        )

        dense_m1_optim = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(local_m1_dmp.named_parameters())),
            lambda params: torch.optim.SGD(params, lr=0.1),
        )

        local_m1_opt = CombinedOptimizer([local_m1_dmp.fused_optimizer, dense_m1_optim])

        # Run a single training step of the sharded model.
        local_m1_pred = gen_full_pred_after_one_step(
            local_m1_dmp,
            local_m1_opt,
            local_input_0,
            True,
        )
        local_m2_pred = gen_full_pred_after_one_step(
            local_m2_dmp,
            local_m2_opt,
            local_input_0,
            True,
        )

        local_m1_dmp.reshard(
            sharded_module_fqn=None if skip_passing_resharding_fqn else "sparse.ebc",
            changed_shard_to_params=new_module_sharding_plan_delta,
        )

        # Must recreate local_m1_opt, because current local_m1_opt is a copy of underlying fused_opt

        local_m1_opt = CombinedOptimizer([local_m1_dmp.fused_optimizer, dense_m1_optim])

        local_m1_pred = gen_full_pred_after_one_step(
            local_m1_dmp, local_m1_opt, local_input_1
        )
        local_m2_pred = gen_full_pred_after_one_step(
            local_m2_dmp, local_m2_opt, local_input_1
        )

        # TODO: support non-sharded forward with zero batch size KJT

        all_local_pred_m1 = _gather_all_tensors(local_m1_pred, world_size, ctx.pg)

        all_local_pred_m2 = _gather_all_tensors(local_m2_pred, world_size, ctx.pg)

        # Compare predictions of sharded vs unsharded models.
        if qcomms_config is None:
            torch.testing.assert_close(
                torch.cat(all_local_pred_m1),
                torch.cat(all_local_pred_m2),
                atol=1e-4,
                rtol=1e-4,
            )
        else:
            # With quantized comms, we can relax constraints a bit
            rtol = 0.003
            if CommType.FP8 in [
                qcomms_config.forward_precision,
                qcomms_config.backward_precision,
            ]:
                rtol = 0.05

            cat_1 = torch.cat(all_local_pred_m1)
            atol = cat_1.max().item() * rtol
            torch.testing.assert_close(
                cat_1, torch.cat(all_local_pred_m2), atol=atol, rtol=rtol
            )


def copy_state_dict(
    loc: Dict[str, Union[torch.Tensor, ShardedTensor, DTensor]],
    glob: Dict[str, torch.Tensor],
    exclude_predfix: Optional[str] = None,
) -> None:
    """
    Copies the contents of the global tensors in glob to the local tensors in loc.
    """
    for name, tensor in loc.items():
        if exclude_predfix is not None and name.startswith(exclude_predfix):
            continue
        else:
            assert name in glob, name
        global_tensor = glob[name]
        if isinstance(global_tensor, ShardedTensor):
            global_tensor = global_tensor.local_shards()[0].tensor
        if isinstance(global_tensor, DTensor):
            # pyre-ignore[16]
            global_tensor = global_tensor.to_local().local_shards()[0]

        if isinstance(tensor, ShardedTensor):
            for local_shard in tensor.local_shards():
                # Tensors like `PartiallyMaterializedTensor` do not provide
                # `ndim` property, so use shape length here as a workaround
                ndim = len(local_shard.tensor.shape)
                assert (
                    global_tensor.ndim == ndim
                ), f"global_tensor.ndim: {global_tensor.ndim}, local_shard.tensor.ndim: {ndim}"
                assert (
                    global_tensor.dtype == local_shard.tensor.dtype
                ), f"global tensor dtype: {global_tensor.dtype}, local tensor dtype: {local_shard.tensor.dtype}"

                shard_meta = local_shard.metadata
                t = global_tensor.detach()
                if t.ndim == 1:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0]
                    ]
                elif t.ndim == 2:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0],
                        shard_meta.shard_offsets[1] : shard_meta.shard_offsets[1]
                        + local_shard.tensor.shape[1],
                    ]
                else:
                    raise ValueError("Tensors with ndim > 2 are not supported")

                if isinstance(local_shard.tensor, PartiallyMaterializedTensor):
                    local_shard.tensor.wrapped.set_range(
                        0, 0, t.size(0), t.to(device="cpu")
                    )
                else:
                    local_shard.tensor.copy_(t)
        elif isinstance(tensor, DTensor):
            for local_shard, global_offset in zip(
                tensor.to_local().local_shards(),
                tensor.to_local().local_offsets(),  # pyre-ignore[16]
            ):
                assert (
                    global_tensor.ndim == local_shard.ndim
                ), f"global_tensor.ndim: {global_tensor.ndim}, local_shard.ndim: {local_shard.ndim}"
                assert (
                    global_tensor.dtype == local_shard.dtype
                ), f"global_tensor.dtype: {global_tensor.dtype}, local_shard.dtype: {local_shard.tensor.dtype}"

                t = global_tensor.detach()
                local_shape = local_shard.shape
                if t.ndim == 1:
                    t = t[global_offset[0] : global_offset[0] + local_shape[0]]
                elif t.ndim == 2:
                    t = t[
                        global_offset[0] : global_offset[0] + local_shape[0],
                        global_offset[1] : global_offset[1] + local_shape[1],
                    ]
                else:
                    raise ValueError("Tensors with ndim > 2 are not supported")
                local_shard.copy_(t)
        else:
            tensor.copy_(global_tensor)


# alter the ebc dtype to float32 in-place.
def alter_global_ebc_dtype(model: nn.Module) -> None:
    for _name, ebc in model.named_modules():
        if isinstance(ebc, EmbeddingBagCollection) and ebc._is_weighted:
            with torch.no_grad():
                for bag in ebc.embedding_bags.values():
                    # pyre-fixme[16]: `Module` has no attribute `weight`.
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    bag.weight = torch.nn.Parameter(bag.weight.float())


def sharding_single_rank_test_single_process(
    pg: dist.ProcessGroup,
    device: torch.device,
    rank: int,
    world_size: int,
    model_class: TestSparseNNBase,
    embedding_groups: Dict[str, List[str]],
    tables: List[EmbeddingTableConfig],
    sharders: List[ModuleSharder[nn.Module]],
    optim: EmbOptimType,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    apply_optimizer_in_backward_config: Optional[
        Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
    ] = None,
    variable_batch_size: bool = False,  # variable batch per rank
    batch_size: int = 4,
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    variable_batch_per_feature: bool = False,  # VBE
    global_constant_batch: bool = False,
    world_size_2D: Optional[int] = None,  # 2D parallel
    node_group_size: Optional[int] = None,  # 2D parallel
    use_inter_host_allreduce: bool = False,  # 2D parallel
    input_type: str = "kjt",  # "kjt" or "td"
    allow_zero_batch_size: bool = False,
    custom_all_reduce: bool = False,  # 2D parallel
    submodule_configs: Optional[List[DMPCollectionConfig]] = None,
    use_offsets: bool = False,
    indices_dtype: torch.dtype = torch.int64,
    offsets_dtype: torch.dtype = torch.int64,
    lengths_dtype: torch.dtype = torch.int64,
    random_seed: Optional[int] = None,
) -> None:
    batch_size = random.randint(0, batch_size) if allow_zero_batch_size else batch_size
    # Generate model & inputs.
    (global_model, inputs) = gen_model_and_input(
        model_class=model_class,
        tables=tables,
        # pyre-ignore [6]
        generate=(
            cast(
                VariableBatchModelInputCallable,
                ModelInput.generate_variable_batch_input,
            )
            if variable_batch_per_feature
            else ModelInput.generate
        ),
        weighted_tables=weighted_tables,
        embedding_groups=embedding_groups,
        world_size=world_size,
        num_float_features=16,
        variable_batch_size=variable_batch_size,
        batch_size=batch_size,
        feature_processor_modules=feature_processor_modules,
        global_constant_batch=global_constant_batch,
        input_type=input_type,
        use_offsets=use_offsets,
        indices_dtype=indices_dtype,
        offsets_dtype=offsets_dtype,
        lengths_dtype=lengths_dtype,
        random_seed=random_seed,
    )

    global_model = global_model.to(device)
    global_input = inputs[0][0].to(device)
    local_input = inputs[0][1][rank].to(device)

    # Shard model.
    local_model = model_class(
        tables=cast(List[BaseEmbeddingConfig], tables),
        weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
        embedding_groups=embedding_groups,
        dense_device=device,
        sparse_device=torch.device("meta"),
        num_float_features=16,
        feature_processor_modules=feature_processor_modules,
    )

    global_model_named_params_as_dict = dict(global_model.named_parameters())
    local_model_named_params_as_dict = dict(local_model.named_parameters())
    # Registers a hook to update parameters in the backward pass, when gradients are computed.
    if apply_optimizer_in_backward_config is not None:
        for apply_optim_name, (
            optimizer_type,
            optimizer_kwargs,
        ) in apply_optimizer_in_backward_config.items():
            for name, param in global_model_named_params_as_dict.items():
                if apply_optim_name not in name:
                    continue
                assert name in local_model_named_params_as_dict
                local_param = local_model_named_params_as_dict[name]
                apply_optimizer_in_backward(
                    optimizer_type,
                    [param],
                    optimizer_kwargs,
                )
                apply_optimizer_in_backward(
                    optimizer_type, [local_param], optimizer_kwargs
                )

    # For 2D parallelism, we use single group world size and local world size
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            world_size=world_size_2D if world_size_2D else world_size,
            compute_device=device.type,
            local_world_size=node_group_size if node_group_size else local_size,
        ),
        constraints=constraints,
    )
    plan: ShardingPlan = planner.collective_plan(local_model, sharders, pg)

    if submodule_configs is not None:
        # Dynamic 2D parallel, create a new plan for each submodule
        for config in submodule_configs:
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=config.sharding_group_size,
                    compute_device=device.type,
                    local_world_size=config.node_group_size,
                ),
                constraints=constraints,
            )
            sharder = TestEmbeddingCollectionSharder(
                sharding_type=ShardingType.ROW_WISE.value,
                kernel_type=EmbeddingComputeKernel.FUSED.value,
                qcomms_config=qcomms_config,
            )
            sharders.append(sharder)  # pyre-ignore[6]
            config.plan = planner.collective_plan(
                local_model, [sharder], pg  # pyre-ignore[6]
            )

    """
    Simulating multiple nodes on a single node. However, metadata information and
    tensor placement must still be consistent. Here we overwrite this to do so.

    NOTE:
        inter/intra process groups should still behave as expected.

    TODO: may need to add some checks that only does this if we're running on a
    single GPU (which should be most cases).
    """
    for group in plan.plan:
        for _, parameter_sharding in cast(
            EmbeddingModuleShardingPlan, plan.plan[group]
        ).items():
            if (
                parameter_sharding.sharding_type
                in {
                    ShardingType.TABLE_ROW_WISE.value,
                    ShardingType.TABLE_COLUMN_WISE.value,
                    ShardingType.GRID_SHARD.value,
                }
                and device.type != "cpu"
            ):
                sharding_spec = parameter_sharding.sharding_spec
                if sharding_spec is not None:
                    # pyre-ignore
                    for shard in sharding_spec.shards:
                        placement = shard.placement
                        rank: Optional[int] = placement.rank()
                        assert rank is not None
                        shard.placement = torch.distributed._remote_device(
                            f"rank:{rank}/cuda:{rank}"
                        )

    hook_called: bool = False
    if world_size_2D is not None:
        all_reduce_func = None
        if custom_all_reduce:
            all_reduce_pg: dist.ProcessGroup = create_device_mesh_for_2D(
                use_inter_host_allreduce,
                world_size=world_size,
                local_size=world_size_2D,
            ).get_group(mesh_dim="replicate")

            def _custom_hook(input: List[torch.Tensor]) -> None:
                nonlocal hook_called
                opts = dist.AllreduceCoalescedOptions()
                opts.reduceOp = dist.ReduceOp.AVG
                handle = all_reduce_pg.allreduce_coalesced(input, opts=opts)
                handle.wait()
                hook_called = True

            all_reduce_func = _custom_hook

        local_model = DMPCollection(
            module=local_model,
            sharding_group_size=world_size_2D,
            world_size=world_size,
            global_pg=pg,
            node_group_size=node_group_size,
            plan=plan,
            sharders=sharders,
            device=device,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=all_reduce_func,
            submodule_configs=submodule_configs,
        )
    else:
        local_model = DistributedModelParallel(
            local_model,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )

    dense_optim = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(local_model.named_parameters())),
        lambda params: torch.optim.SGD(params, lr=0.1),
    )
    local_opt = CombinedOptimizer([local_model.fused_optimizer, dense_optim])

    # Load model state from the global model.
    copy_state_dict(
        local_model.state_dict(),
        global_model.state_dict(),
        exclude_predfix="sparse.pooled_embedding_arch.embedding_modules._itp_iter",
    )
    alter_global_ebc_dtype(global_model)

    # Run a single training step of the sharded model.
    local_pred = gen_full_pred_after_one_step(
        local_model,
        local_opt,
        local_input,
    )

    if world_size_2D is not None and custom_all_reduce:
        assert hook_called, "custom all reduce hook was not called"

    # TODO: support non-sharded forward with zero batch size KJT
    if not allow_zero_batch_size:
        all_local_pred = _gather_all_tensors(local_pred, world_size, pg)

        # Run second training step of the unsharded model.
        assert optim == EmbOptimType.EXACT_SGD
        global_opt = torch.optim.SGD(global_model.parameters(), lr=0.1)

        global_pred = gen_full_pred_after_one_step(
            global_model, global_opt, global_input
        )

        # Compare predictions of sharded vs unsharded models.
        if qcomms_config is not None:
            # With quantized comms, we can relax constraints a bit
            rtol = 0.003
            if CommType.FP8 in [
                qcomms_config.forward_precision,
                qcomms_config.backward_precision,
            ]:
                rtol = 0.05
            atol = global_pred.max().item() * rtol
            torch.testing.assert_close(
                global_pred, torch.cat(all_local_pred), rtol=rtol, atol=atol
            )
        elif (
            weighted_tables is not None
            and weighted_tables[0].data_type == DataType.FP16
        ):
            # we relax this accuracy test because when the embedding table weights is FP16,
            # the sharded EBC would upscale the precision to FP32 for the returned embedding
            # KJT.weights (FP32) + sharded_EBC (FP16) ==> embeddings (FP32)
            # the test uses the unsharded EBC for reference to compare the results, but the unsharded EBC
            #  uses EmbeddingBags can only handle same precision, i.e.,
            # KJT.weights (FP32) + unsharded_EBC (FP32) ==> embeddings (FP32)
            # therefore, the discrepancy leads to a relaxed tol level.
            torch.testing.assert_close(
                global_pred,
                torch.cat(all_local_pred),
                atol=1e-4,  # relaxed atol due to FP16 in weights
                rtol=1e-4,  # relaxed rtol due to FP16 in weights
            )
        else:
            torch.testing.assert_close(global_pred, torch.cat(all_local_pred))


def sharding_single_rank_test(
    rank: int,
    world_size: int,
    model_class: TestSparseNNBase,
    embedding_groups: Dict[str, List[str]],
    tables: List[EmbeddingTableConfig],
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    optim: EmbOptimType,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    apply_optimizer_in_backward_config: Optional[
        Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
    ] = None,
    variable_batch_size: bool = False,  # variable batch per rank
    batch_size: int = 4,
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    variable_batch_per_feature: bool = False,  # VBE
    global_constant_batch: bool = False,
    world_size_2D: Optional[int] = None,  # 2D parallel
    node_group_size: Optional[int] = None,  # 2D parallel
    use_inter_host_allreduce: bool = False,  # 2D parallel
    input_type: str = "kjt",  # "kjt" or "td"
    allow_zero_batch_size: bool = False,
    custom_all_reduce: bool = False,  # 2D parallel
    submodule_configs: Optional[List[DMPCollectionConfig]] = None,  # 2D parallel
    use_offsets: bool = False,
    indices_dtype: torch.dtype = torch.int64,
    offsets_dtype: torch.dtype = torch.int64,
    lengths_dtype: torch.dtype = torch.int64,
    random_seed: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        sharding_single_rank_test_single_process(
            pg=ctx.pg,
            device=ctx.device,
            rank=rank,
            world_size=world_size,
            model_class=model_class,
            embedding_groups=embedding_groups,
            tables=tables,
            sharders=sharders,
            optim=optim,
            weighted_tables=weighted_tables,
            constraints=constraints,
            local_size=local_size,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            batch_size=batch_size,
            feature_processor_modules=feature_processor_modules,
            variable_batch_per_feature=variable_batch_per_feature,
            global_constant_batch=global_constant_batch,
            world_size_2D=world_size_2D,
            node_group_size=node_group_size,
            use_inter_host_allreduce=use_inter_host_allreduce,
            input_type=input_type,
            allow_zero_batch_size=allow_zero_batch_size,
            custom_all_reduce=custom_all_reduce,
            submodule_configs=submodule_configs,
            use_offsets=use_offsets,
            indices_dtype=indices_dtype,
            offsets_dtype=offsets_dtype,
            lengths_dtype=lengths_dtype,
            random_seed=random_seed,
        )


def create_device_mesh_for_2D(
    use_inter_host_allreduce: bool, world_size: int, local_size: int
) -> DeviceMesh:
    if use_inter_host_allreduce:
        peer_matrix = [
            list(range(i, i + local_size)) for i in range(0, world_size, local_size)
        ]
    else:
        peer_matrix = []
        step = world_size // local_size
        for group_rank in range(world_size // local_size):
            peer_matrix.append([step * r + group_rank for r in range(local_size)])

    mesh = DeviceMesh(
        device_type="cuda",
        mesh=peer_matrix,
        mesh_dim_names=("replicate", "shard"),
    )

    return mesh


def gen_full_pred_after_one_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    input: ModelInput,
    skip_inference: bool = False,
    skip_training: bool = False,
) -> torch.Tensor:
    if skip_training:
        model.train(False)
        output = model(input)
        return output
    # Run a single training step of the global model.
    opt.zero_grad()
    model.train(True)
    output = model(input)
    loss = output[0]
    loss.backward()
    opt.step()
    if skip_inference:
        return output

    # Sync embedding weights if 2D paralleism is used.
    if isinstance(model, DMPCollection):
        model.sync()

    # Run a forward pass of the global model.
    with torch.no_grad():
        model.train(False)
        full_pred = model(input)
        return full_pred


def gen_full_model_pred_after_x_steps(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    input: List[ModelInput],
    skip_inference: bool = False,
    steps: int = 1,
) -> torch.Tensor:
    output = torch.Tensor()
    for i in range(steps):
        output = gen_full_pred_after_one_step(
            model,
            opt,
            input[i],
            skip_inference,
        )
    return output


def compare_models_pred_one_step(
    model_1: nn.Module,
    model_2: nn.Module,
    opt_1: torch.optim.Optimizer,
    opt_2: torch.optim.Optimizer,
    input: ModelInput,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """
    Helper function to compare the model predictions of two models after one training step.
    Useful for debugging sharding tests to see which model weights are different
    """
    # pyre-ignore
    compare_model_weights(model_1.module.sparse, model_2.module.sparse)
    # Run a single training step of the global model.
    output_1 = gen_full_pred_after_one_step(model_1, opt_1, input, skip_inference=True)
    output_2 = gen_full_pred_after_one_step(model_2, opt_2, input, skip_inference=True)

    torch.testing.assert_close(output_1, output_2)
    compare_model_weights(
        # pyre-ignore
        model_1.module.sparse,
        # pyre-ignore
        model_2.module.sparse,
    )  # Module weights are the same

    # Run a forward pass of the global model.
    with torch.no_grad():
        model_1.train(False)
        full_pred_1 = model_1(input)
        model_2.train(False)
        full_pred_2 = model_2(input)
        torch.testing.assert_close(full_pred_1, full_pred_2, rtol=rtol, atol=atol)


def compare_model_weights(
    m1: nn.Module,
    m2: nn.Module,
) -> None:
    """
    Compare weights between two models, and asserts differences with useful logs.
    """
    # For ShardedTensor and DTensor, we need to compare local shards
    for name, param1 in m1.named_parameters():
        if name not in dict(m2.named_parameters()):
            raise ValueError(f"Parameter {name} not found in model 2")

        param2 = dict(m2.named_parameters())[name]

        # Handle different tensor types
        if isinstance(param1, ShardedTensor) and isinstance(param2, ShardedTensor):
            # Compare local shards
            for local_shard1 in param1.local_shards():
                shard_meta1 = local_shard1.metadata
                found_match = False
                for local_shard2 in param2.local_shards():
                    shard_meta2 = local_shard2.metadata
                    if shard_meta1.shard_offsets == shard_meta2.shard_offsets:
                        torch.testing.assert_close(
                            local_shard1.tensor,
                            local_shard2.tensor,
                            msg=f"ShardedTensor {name} not equal at offset {shard_meta1.shard_offsets}",
                        )
                        found_match = True
                        break

                if not found_match:
                    raise ValueError(
                        f"No matching shard found for {name} at offset {shard_meta1.shard_offsets}"
                    )

        elif isinstance(param1, DTensor) and isinstance(param2, DTensor):
            # Compare local tensors for DTensor
            local1 = param1.to_local()
            local2 = param2.to_local()
            for shard1, offset1 in zip(local1.local_shards(), local1.local_offsets()):
                found_match = False

                for shard2, offset2 in zip(
                    local2.local_shards(), local2.local_offsets()
                ):
                    if offset1 == offset2:
                        torch.testing.assert_close(
                            shard1,
                            shard2,
                            msg=f"DTensor {name} not equal at offset {offset1}",
                        )
                        found_match = True
                        break

                if not found_match:
                    raise ValueError(
                        f"No matching DTensor shard found for {name} at offset {offset1}"
                    )
        else:
            param1 = param1.cpu()
            param2 = param2.cpu()
            # Regular tensor comparison
            torch.testing.assert_close(
                param1,
                param2,
                msg=f"Parameter {name} not equal",  # rtol=1e-4, atol=1e-4 would make it pass...
            )


def generate_rank_placements(
    world_size: int,
    num_tables: int,
    ranks_per_tables: List[int],
    random_seed: int = None,  # pyre-ignore
) -> List[List[int]]:
    # Cannot include old/new rank generation with hypothesis library due to depedency on world_size
    if random_seed is None:
        # Generate a random seed to ensure that the output rank placements can be different each time
        random_seed = random.randint(0, 10000)
    placements = []
    max_rank = world_size - 1
    random.seed(random_seed)
    if ranks_per_tables == [0]:
        ranks_per_tables = [random.randint(1, max_rank) for _ in range(num_tables)]
    for i in range(num_tables):
        ranks_per_table = ranks_per_tables[i]
        placement = sorted(random.sample(range(world_size), ranks_per_table))
        placements.append(placement)
    return placements


def compare_opt_local_t(
    opt_1: CombinedOptimizer,
    opt_2: CombinedOptimizer,
    table_id: int,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """
    Helper function to compare the optimizer state of two models after one training step.
    Useful for debugging sharding tests to see which model weights are different
    """
    # TODO: update logic to be generic other embedding modules
    t1 = (
        opt_1.state_dict()["state"][
            "sparse.ebc.embedding_bags.table_" + str(table_id) + ".weight"
        ]["table_" + str(table_id) + ".momentum1"]
        .local_shards()[0]
        .tensor
    )
    t2 = (
        opt_2.state_dict()["state"][
            "sparse.ebc.embedding_bags.table_" + str(table_id) + ".weight"
        ]["table_" + str(table_id) + ".momentum1"]
        .local_shards()[0]
        .tensor
    )
    torch.testing.assert_close(t1, t2, rtol=rtol, atol=atol)
