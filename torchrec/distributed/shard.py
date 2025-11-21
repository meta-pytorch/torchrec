#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, Dict, List, Optional, Type, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.contract import contract
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.global_settings import get_propogate_device
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.sharding_plan import (
    get_module_to_default_sharders,
    ParameterShardingGenerator,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ModuleShardingPlan,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.distributed.utils import init_parameters
from torchrec.modules.utils import reset_module_states_post_sharding


def _join_module_path(path: str, name: str) -> str:
    return (path + "." + name) if path else name


# pyre-ignore
@contract()
def shard(
    module: nn.Module,
    plan: Union[
        ModuleShardingPlan,
        Dict[str, ParameterShardingGenerator],
        ParameterShardingGenerator,
    ],
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    sharder: Optional[ModuleSharder[nn.Module]] = None,
) -> nn.Module:
    """
    Replaces this module with its sharded variant

    This will leave the other parts of the model unaffected.

    It returns the original module

    Args:
        module (nn.Module): module to wrap.
        env (Optional[ShardingEnv]): sharding environment that has the process group.
        device (Optional[torch.device]): compute device, defaults to cpu.
        plan (Union[ModuleShardingPlan, Dict[str, ParameterShardingGenerator], ParameterShardingGenerator]):
            dict of ParameterSharding (materized plan) or ParameterShardingGenerator (which will be run to produce ParameterSharding).
            If single ParameterShardingGenerator is supplied, it will be applied to all module parameters.
        sharder (Optional[List[ModuleSharder[nn.Module]]]): sharder to use, default is picked from `get_default_sharders()`.

    Example:

        ebc = EmbeddingBagCollection()
        sharded_ebc = shard(ebc, table_row_wise(host_index=0))
        assert isinstance(sharded_ebc, ShardedEmbeddingBagCollection)
    """
    torch._C._log_api_usage_once("torchrec.distributed.shard")
    return _shard(module, plan, env, device, sharder)


def _shard(
    module: nn.Module,
    plan: Union[
        ModuleShardingPlan,
        Dict[str, ParameterShardingGenerator],
        ParameterShardingGenerator,
    ],
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    sharder: Optional[ModuleSharder[nn.Module]] = None,
) -> nn.Module:
    """
    See shard
    """
    if sharder is None:
        sharder = get_module_to_default_sharders().get(type(module), None)
    assert (
        sharder is not None
    ), f"Could not find a valid sharder type for {type(module)}"

    if env is None:
        pg = dist.GroupMember.WORLD
        assert pg is not None, "Process group is not initialized"
        env = ShardingEnv.from_process_group(pg)

    if device is None:
        if get_propogate_device():
            device = torch.device(
                "cpu"
            )  # TODO: replace hardcoded cpu with DEFAULT_DEVICE_TYPE in torchrec.distributed.types when torch package issue resolved
        else:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device("cpu")

    if isinstance(plan, ModuleShardingPlan):
        return sharder.shard(module, plan, env, device)

    # Run sharding generators.
    shardable_parameters = sharder.shardable_parameters(module)
    if isinstance(plan, Callable):
        gen = plan
        plan = {}
        for table_name, param in shardable_parameters.items():
            plan[table_name] = gen(
                param,
                get_local_size(env.world_size),
                env.world_size,
                device.type,
                sharder,
            )
    else:
        for table_name, sharding in plan.items():
            if isinstance(sharding, Callable):
                param = shardable_parameters[table_name]
                # pyre-fixme[6]: For 2nd argument expected `(Parameter, int, int,
                #  str, ModuleSharder[Module]) -> ParameterSharding` but got
                #  `ParameterSharding`.
                plan[table_name] = sharding(
                    param,
                    get_local_size(env.world_size),
                    env.world_size,
                    device.type,
                    sharder,
                )

    return sharder.shard(module, plan, env, device)


# pyre-ignore
@contract()
def shard_modules(
    module: nn.Module,
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    init_params: bool = False,
) -> nn.Module:
    """
    Replaces all sub_modules that are embedding modules with their sharded variants. This embedding_module -> sharded_embedding_module mapping
    is derived from the passed in sharders.

    This will leave the other parts of the model unaffected.

    It returns the original module

    Args:
        module (nn.Module): module to wrap.
        env (Optional[ShardingEnv]): sharding environment that has the process group.
        device (Optional[torch.device]): compute device, defaults to cpu.
        plan (Optional[ShardingPlan]): plan to use when sharding, defaults to
            `EmbeddingShardingPlanner.collective_plan()`.
        sharders (Optional[List[ModuleSharder[nn.Module]]]): `ModuleSharders` available
            to shard with, defaults to `get_default_sharders()`.
        init_params: (Optional[bool]): If ``True``, will materialize parameters and
            buffers that are on meta device, and will move module to ``device``. Note that
            this only applies if `device.type != "meta"``. Default: `False`.


    Example::

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection):
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        m = shard(m)
        assert isinstance(m.embedding_bag_collection, ShardedEmbeddingBagCollection)
    """

    torch._C._log_api_usage_once("torchrec.distributed.shard_modules")
    return _shard_modules(module, env, device, plan, sharders, init_params)


def _shard_modules(  # noqa: C901
    module: nn.Module,
    # TODO: Consolidate to using Dict[str, ShardingEnv]
    env: Optional[
        Union[ShardingEnv, Dict[str, ShardingEnv]]
    ] = None,  # Support hybrid sharding
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    init_params: Optional[bool] = False,
    planner: Optional[EmbeddingShardingPlanner] = None,
) -> nn.Module:
    """
    See shard_modules
    """

    torch._C._log_api_usage_once("torchrec.distributed.shard_modules")

    if sharders is None:
        sharders = get_default_sharders()

    if env is None:
        pg = dist.GroupMember.WORLD
        assert pg is not None, "Process group is not initialized"
        env = ShardingEnv.from_process_group(pg)

    if device is None:
        if get_propogate_device():
            device = torch.device(
                "cpu"
            )  # TODO: replace hardcoded cpu with DEFAULT_DEVICE_TYPE in torchrec.distributed.types when torch package issue resolved
        else:
            if torch.cuda.is_available():
                device = torch.device(torch.cuda.current_device())
            else:
                device = torch.device("cpu")

    sharder_map: Dict[Type[nn.Module], ModuleSharder[nn.Module]] = {
        sharder.module_type: sharder for sharder in sharders
    }

    if plan is None:
        assert isinstance(
            env, ShardingEnv
        ), "Currently hybrid sharding only support use manual sharding plan"
        if planner is None:
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    local_world_size=get_local_size(env.world_size),
                    world_size=env.world_size,
                    compute_device=device.type,
                )
            )
        pg = env.process_group
        if pg is not None:
            plan = planner.collective_plan(module, sharders, pg)
        else:
            plan = planner.plan(module, sharders)

    if type(module) in sharder_map:
        # If the top level module is itself a shardable module, return the sharded variant.
        # Note, we cannot do an inplace replacement in this case.
        return sharder_map[type(module)].shard(
            module, plan.get_plan_for_module(""), env, device, ""
        )

    def _replace(_model: nn.Module, path: str = "") -> None:
        for child_name, child in _model.named_children():
            child_path = _join_module_path(path, child_name)
            if type(child) in sharder_map:
                assert plan is not None
                sharded_params = plan.get_plan_for_module(child_path)
                if sharded_params is not None:
                    sharded_module = sharder_map[type(child)].shard(
                        child,
                        sharded_params,
                        env,
                        device,
                        child_path,
                    )
                    _model.register_module(
                        child_name,
                        sharded_module,
                    )
            else:
                _replace(child, child_path)

    _replace(module)
    if init_params and device is not None and device.type != "meta":
        init_parameters(module, device)
        module = module.to(device)

    reset_module_states_post_sharding(module)

    return module
