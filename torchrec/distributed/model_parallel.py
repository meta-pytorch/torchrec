#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import copy
import logging as logger
from collections import defaultdict, OrderedDict
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Set, Tuple, Type

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DeviceMesh
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_tracker.model_delta_tracker import ModelDeltaTracker
from torchrec.distributed.model_tracker.types import DeltaRows, ModelTrackerConfig

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import (
    DMPCollectionConfig,
    DMPCollectionContext,
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ShardedModule,
    ShardingEnv,
    ShardingEnv2D,
    ShardingPlan,
)
from torchrec.distributed.utils import (
    add_prefix_to_state_dict,
    append_prefix,
    copy_to_device,
    filter_state_dict,
    sharded_model_copy,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except (OSError, RuntimeError):
    pass


_DDP_STATE_DICT_PREFIX = "module."


class DataParallelWrapper(abc.ABC):
    """
    Interface implemented by custom data parallel wrappers.
    """

    @abc.abstractmethod
    def wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        pass


class DefaultDataParallelWrapper(DataParallelWrapper):
    """
    Default data parallel wrapper, which applies data parallel to all unsharded modules.
    """

    def __init__(
        self,
        bucket_cap_mb: int = 25,
        static_graph: bool = True,
        find_unused_parameters: bool = False,
        allreduce_comm_precision: Optional[str] = None,
        params_to_ignore: Optional[List[str]] = None,
        ddp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bucket_cap_mb: int = bucket_cap_mb
        self._static_graph: bool = static_graph
        self._find_unused_parameters: bool = find_unused_parameters
        self._allreduce_comm_precision = allreduce_comm_precision
        self._additional_params_to_ignore: Set[str] = set(params_to_ignore or [])
        self._ddp_kwargs: Dict[str, Any] = ddp_kwargs or {}

    def _ddp_wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
        ddp_ignore_param_names: Set[str],
    ) -> None:
        pg = env.process_group
        if pg is None:
            raise RuntimeError("Can only init DDP for ProcessGroup-based ShardingEnv")
        all_parameter_names = set(dict(dmp.named_parameters()).keys())
        if len(all_parameter_names - ddp_ignore_param_names) == 0:
            return
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=dmp._dmp_wrapped_module,
            params_and_buffers_to_ignore=ddp_ignore_param_names,
        )
        # initialize DDP
        dmp._dmp_wrapped_module = cast(
            nn.Module,
            DistributedDataParallel(
                module=dmp._dmp_wrapped_module.to(device),
                device_ids=None if device.type == "cpu" else [device],
                process_group=pg,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=self._static_graph,
                find_unused_parameters=self._find_unused_parameters,
                bucket_cap_mb=self._bucket_cap_mb,
                **self._ddp_kwargs,
            ),
        )
        if self._allreduce_comm_precision == "fp16":
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            dmp._dmp_wrapped_module.register_comm_hook(
                None, ddp_default_hooks.fp16_compress_hook
            )
        elif self._allreduce_comm_precision == "bf16":
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            dmp._dmp_wrapped_module.register_comm_hook(
                None, ddp_default_hooks.bf16_compress_hook
            )

    def wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        if isinstance(dmp._dmp_wrapped_module, DistributedDataParallel) or isinstance(
            dmp._dmp_wrapped_module, FullyShardedDataParallel
        ):
            return
        sharded_parameter_names = set(
            DistributedModelParallel._sharded_parameter_names(dmp._dmp_wrapped_module)
        )
        params_to_ignore = sharded_parameter_names.union(
            self._additional_params_to_ignore
        )
        self._ddp_wrap(dmp, env, device, params_to_ignore)


def get_unwrapped_module(module: nn.Module) -> nn.Module:
    """
    Unwraps module wrapped by DMP, DDP, or FSDP.
    """
    while (
        isinstance(module, DistributedModelParallel)
        or isinstance(module, DistributedDataParallel)
        or isinstance(module, FullyShardedDataParallel)
    ):
        if isinstance(module, DistributedModelParallel):
            module = module._dmp_wrapped_module
        elif isinstance(module, FullyShardedDataParallel):
            module = module._fsdp_wrapped_module
        else:
            module = module.module
    return module


def get_module(module: nn.Module) -> nn.Module:
    """
    Unwraps DMP module.

    Does not unwrap data parallel wrappers (i.e. DDP/FSDP), so overriding
    implementations by the wrappers can be used.
    """
    while isinstance(module, DistributedModelParallel):
        module = module._dmp_wrapped_module
    return module


class DistributedModelParallel(nn.Module, FusedOptimizerModule):
    """
    Entry point to model parallelism.

    Args:
        module (nn.Module): module to wrap.
        env (Optional[ShardingEnv]): sharding environment that has the process group.
        device (Optional[torch.device]): compute device, defaults to cpu.
        plan (Optional[ShardingPlan]): plan to use when sharding, defaults to
            `EmbeddingShardingPlanner.collective_plan()`.
        sharders (Optional[List[ModuleSharder[nn.Module]]]): `ModuleSharders` available
            to shard with, defaults to `EmbeddingBagCollectionSharder()`.
        init_data_parallel (bool): data-parallel modules can be lazy, i.e. they delay
            parameter initialization until the first forward pass. Pass `True` to delay
            initialization of data parallel modules. Do first forward pass and then call
            DistributedModelParallel.init_data_parallel().
        init_parameters (bool): initialize parameters for modules still on meta device.
        data_parallel_wrapper (Optional[DataParallelWrapper]): custom wrapper for data
            parallel modules.
        model_tracker_config (Optional[DeltaTrackerConfig]): config for model tracker.

    Example::

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection):
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        m = DistributedModelParallel(m)
        m.apply(init_weights)
    """

    def __init__(
        self,
        module: nn.Module,
        env: Optional[ShardingEnv] = None,
        device: Optional[torch.device] = None,
        plan: Optional[ShardingPlan] = None,
        sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
        init_data_parallel: bool = True,
        init_parameters: bool = True,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
        model_tracker_config: Optional[ModelTrackerConfig] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        self.init_parameters = init_parameters

        self._ddp_wrapped: bool = False

        if env is None:
            pg = dist.GroupMember.WORLD
            assert pg is not None, "Process group is not initialized"
            env = ShardingEnv.from_process_group(pg)
        self._env: ShardingEnv = env

        if device is None:
            device = torch.device("cpu")
        self.device: torch.device = device

        self.sharders: List[ModuleSharder[nn.modules.module.Module]] = (
            get_default_sharders() if sharders is None else sharders
        )

        self._sharder_map: Dict[Type[nn.Module], ModuleSharder[nn.Module]] = {
            sharder.module_type: sharder for sharder in self.sharders
        }

        if data_parallel_wrapper is None:
            data_parallel_wrapper = DefaultDataParallelWrapper()
        self._data_parallel_wrapper: DataParallelWrapper = data_parallel_wrapper

        if plan is None:
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    local_world_size=get_local_size(self._env.world_size),
                    world_size=self._env.world_size,
                    compute_device=self.device.type,
                )
            )
            pg = self._env.process_group
            if pg is not None:
                plan = planner.collective_plan(module, self.sharders, pg)
            else:
                plan = planner.plan(module, self.sharders)
        self._plan: ShardingPlan = plan
        self._dmp_wrapped_module: nn.Module = self._init_dmp(module)
        self._optim: CombinedOptimizer = self._init_optim(self._dmp_wrapped_module)

        if init_parameters:
            self._init_parameters(self.module)

        if init_data_parallel:
            self.init_data_parallel()

        self.model_delta_tracker: Optional[ModelDeltaTracker] = (
            self._init_delta_tracker(model_tracker_config, self._dmp_wrapped_module)
            if model_tracker_config is not None
            else None
        )

    @property
    def module(self) -> nn.Module:
        """
        Property to directly access sharded module, which will not be wrapped in DDP,
        FSDP, DMP, or any other parallelism wrappers.
        """
        return get_unwrapped_module(self)

    @module.setter
    def module(self, value: nn.Module) -> None:
        if isinstance(self.module, DistributedDataParallel) or isinstance(
            self.module, FullyShardedDataParallel
        ):
            raise RuntimeError(
                "module can't be set after calling init_data_parallel(...)"
            )
        else:
            self._dmp_wrapped_module = value

    # pyre-ignore [2, 3]
    def forward(self, *args, **kwargs) -> Any:
        if self.model_delta_tracker is not None:
            # The step() call advances the internal batch counter so that subsequent ID tracking and delta
            # retrieval operations can be properly organized by batch boundaries.

            # Context: ModelDeltaTracker tracks unique embedding IDs (and optionally embeddings / states)
            # useful for calculating topk rows for checkpointing and for updating fresh embedding weights
            # between predictors and trainers in online training scenarios.
            self.model_delta_tracker.step()

        # Model forward for DMP wrapped model
        return self._dmp_wrapped_module(*args, **kwargs)

    def init_data_parallel(self) -> None:
        """
        See init_data_parallel c-tor argument for usage.
        It's safe to call this method multiple times.
        """
        if not self._ddp_wrapped:
            # Allocate any 'meta' tensors
            if self.init_parameters:
                self._init_parameters(self._dmp_wrapped_module)
            self._data_parallel_wrapper.wrap(self, self._env, self.device)
            self._ddp_wrapped = True

    def copy(
        self,
        device: torch.device,
    ) -> "DistributedModelParallel":
        """
        Recursively copy submodules to new device by calling per-module customized copy
        process, since some modules needs to use the original references (like
        `ShardedModule` for inference).
        """
        assert isinstance(device, torch.device)
        # dmp code deep copy
        with sharded_model_copy(device=None):
            copy_dmp = copy.deepcopy(self)
        # tensor resident module deep copy
        copy_dmp_wrapped_module = copy_to_device(
            self._dmp_wrapped_module, self.device, device
        )
        copy_dmp._dmp_wrapped_module = copy_dmp_wrapped_module
        return copy_dmp

    def _init_dmp(self, module: nn.Module) -> nn.Module:
        return self._shard_modules_impl(module)

    def _init_delta_tracker(
        self, model_tracker_config: ModelTrackerConfig, module: nn.Module
    ) -> ModelDeltaTracker:
        # Init delta tracker if config is provided
        return ModelDeltaTracker(
            model=module,
            consumers=model_tracker_config.consumers,
            delete_on_read=model_tracker_config.delete_on_read,
            auto_compact=model_tracker_config.auto_compact,
            mode=model_tracker_config.tracking_mode,
            fqns_to_skip=model_tracker_config.fqns_to_skip,
        )

    def _init_optim(self, module: nn.Module) -> CombinedOptimizer:
        # pyre-ignore [6]
        return CombinedOptimizer(self._fused_optim_impl(module, []))

    def _fused_optim_impl(
        self,
        module: nn.Module,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
        path: str = "",
    ) -> List[Tuple[str, KeyedOptimizer]]:
        if isinstance(module, FusedOptimizerModule):
            fused_optims.append((path, module.fused_optimizer))
            return fused_optims

        for name, child in module.named_children():
            self._fused_optim_impl(
                child,
                fused_optims,
                path + "." + name if path else name,
            )
        return fused_optims

    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str = "",
    ) -> nn.Module:
        # pre-sharded module
        if isinstance(module, ShardedModule):
            return module

        # shardable module
        module_sharding_plan = self._plan.get_plan_for_module(path)
        if module_sharding_plan:
            sharder_key = type(module)
            module = self._sharder_map[sharder_key].shard(
                module,
                module_sharding_plan,
                self._env,
                self.device,
                path,
            )
            return module

        for name, child in module.named_children():
            child = self._shard_modules_impl(
                child,
                path + "." + name if path else name,
            )
            setattr(module, name, child)

        return module

    def _init_parameters(self, module: nn.Module) -> None:
        @torch.no_grad()
        def init_parameters(module: nn.Module) -> None:
            # Allocate parameters and buffers if over 'meta' device.
            has_meta_param = False
            for name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.device.type == "meta":
                    module._parameters[name] = nn.Parameter(
                        torch.empty_like(param, device=self.device),
                        requires_grad=param.requires_grad,
                    )
                    has_meta_param = True
            for name, buffer in module._buffers.items():
                if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
                    module._buffers[name] = torch.zeros_like(buffer, device=self.device)

            # Init parameters if at least one parameter is over 'meta' device.
            if has_meta_param and hasattr(module, "reset_parameters"):
                # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
                module.reset_parameters()

        module.apply(init_parameters)

    def get_model_tracker(self) -> ModelDeltaTracker:
        """
        Returns the model tracker if it exists.
        """

        assert (
            self.model_delta_tracker is not None
        ), "Model tracker is not initialized. Add ModelTrackerConfig at DistributedModelParallel init."
        return self.model_delta_tracker

    def get_delta(self, consumer: Optional[str] = None) -> Dict[str, DeltaRows]:
        """
        Returns the delta rows for the given consumer.
        """
        assert (
            self.model_delta_tracker is not None
        ), "Model tracker is not initialized. Add ModelTrackerConfig at DistributedModelParallel init."
        return self.model_delta_tracker.get_delta(consumer)

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return self._sparse_grad_parameter_names(self.module, destination, prefix)

    def _sparse_grad_parameter_names(
        self, module: nn.Module, destination: List[str], prefix: str = ""
    ) -> List[str]:
        module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            pass
        elif isinstance(module, nn.Embedding):
            if module.sparse:
                destination.append(append_prefix(prefix, "weight"))
        elif isinstance(module, nn.EmbeddingBag):
            if module.sparse:
                destination.append(append_prefix(prefix, "weight"))
        else:
            for name, child in module.named_children():
                self._sparse_grad_parameter_names(
                    child, destination, append_prefix(prefix, name)
                )
        return destination

    # pyre-ignore [14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        state_dict = get_module(self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix + _DDP_STATE_DICT_PREFIX
        )
        add_prefix_to_state_dict(state_dict, prefix)
        return state_dict

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str = "",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        return self._load_state_dict(self, state_dict, prefix, strict)

    def _load_state_dict(
        self,
        module: nn.Module,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str = "",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        module = get_module(module)
        if isinstance(module, DistributedDataParallel):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                state_dict, prefix
            )
            add_prefix_to_state_dict(state_dict, prefix + _DDP_STATE_DICT_PREFIX)
        if getattr(module, "_FORCE_STATE_DICT_LOAD", False):
            return module.load_state_dict(state_dict, strict=strict)
        else:
            module._load_from_state_dict(
                state_dict, prefix, {}, strict, missing_keys, unexpected_keys, []
            )
            for name, child in module.named_children():
                m_keys, u_keys = self._load_state_dict(
                    child,
                    filter_state_dict(state_dict, prefix + name),
                    "",
                    strict,
                )
                missing_keys.extend(m_keys)
                unexpected_keys.extend(u_keys)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def _named_parameters(
        self,
        module: nn.Module,
        prefix: str = "",
        recurse: bool = True,
        strip_ddp: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        if strip_ddp:
            module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            yield from module.named_parameters(prefix, recurse)
        else:
            yield from module.named_parameters(prefix, recurse=False)
            for name, child in module.named_children():
                yield from self._named_parameters(
                    child,
                    append_prefix(prefix, name),
                    recurse,
                    strip_ddp,
                )

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        gen = self._named_parameters(
            self.module,
            prefix,
            recurse,
        )
        memo = set()
        for key, param in gen:
            if param in memo:
                continue
            if remove_duplicate:
                memo.add(param)
            yield key, param

    def bare_named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        gen = self._named_parameters(
            self.module,
            prefix,
            recurse,
        )
        memo = set()
        for key, param in gen:
            if param in memo:
                continue
            memo.add(param)
            yield key, param

    @staticmethod
    def _sharded_parameter_names(module: nn.Module, prefix: str = "") -> Iterator[str]:
        module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            yield from module.sharded_parameter_names(prefix)
        else:
            for name, child in module.named_children():
                yield from DistributedModelParallel._sharded_parameter_names(
                    child, append_prefix(prefix, name)
                )

    def _named_buffers(
        self, module: nn.Module, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            yield from module.named_buffers(prefix, recurse)
        else:
            yield from module.named_buffers(prefix, recurse=False)
            for name, child in module.named_children():
                yield from self._named_buffers(
                    child, append_prefix(prefix, name), recurse
                )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        gen = self._named_buffers(self.module, prefix, recurse)
        memo = set()
        for key, param in gen:
            if param in memo:
                continue
            if remove_duplicate:
                memo.add(param)
            yield key, param

    @property
    def fused_optimizer(self) -> CombinedOptimizer:
        return self._optim

    @property
    def plan(self) -> ShardingPlan:
        return self._plan

    @staticmethod
    def _reset_parameters(module: nn.Module) -> None:
        for _, m in module.named_modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def reshard(
        self,
        changed_shard_to_params: Dict[str, Tuple[float, EmbeddingModuleShardingPlan]],
        sharded_module_fqn: Optional[str] = None,
    ) -> float:
        """
        Reshards an already-sharded module in the DMP given a set of ParameterShardings to change placements.
        Returns the data volume resharded

        This method allows you to dynamically change the sharding strategy for a specific module
        without recreating the entire DMP. It's particularly useful for:
        1. Adapting to changing requirements during training
        2. Implementing progressive sharding strategies
        3. Rebalancing load across devices
        4. A/B Testing different sharding plans

        Args:
            path_to_sharded_module (str): The path to the sharded module in the DMP.
                For example, "sparse.ebc".
            changed_shard_to_params (Dict[str, ParameterSharding]): A dictionary mapping
                parameter names to their new ParameterSharding configurations. Includes
                only the shards that needs to be moved.

        Example:
            ```
            # Original sharding plan might have table sharded across 2 GPUs
            original_plan = {
                "table_0': ParameterSharding(
                    sharding_type="table_wise",
                    ranks=[0, 1, 2, 3],
                    sharding_spec=EnumerableShardingSpec(...)
                )
            }

            # New sharding plan to shard across 4 GPUs
            new_plan = {
                "weight": ParameterSharding(
                    sharding_type="table_wise",
                    ranks=[0, 1, 2, 3],
                    sharding_spec=EnumerableShardingSpec(...)
                )
            }

            # Helper function for only selecting the delta between original and new plan
            changed_sharding_params = output_sharding_plan_delta(new_plan)

            # Reshard the module and redistribute the tensors
            model.reshard("embedding_module", changed_sharding_params)
            ```

        Notes:
            - The sharder for the module must implement a `reshard` method
            - Resharding involves redistributing tensor data across devices, which can be expensive
            - After resharding, the optimizer state is maintained for the module
            - The sharding plan is updated to reflect the new configuration
        """
        sharder = None
        sharded_module = None

        if sharded_module_fqn is None:
            named_modules_queue = [("", self.module)]
            while named_modules_queue:
                child_path, child_module = named_modules_queue.pop(0)
                if isinstance(child_module, ShardedModule):
                    sharder_key = child_module.unsharded_module_type
                    sharder = self._sharder_map.get(sharder_key, None)
                if not sharder:
                    for n, m in child_module.named_children():
                        if child_path != "":
                            named_modules_queue.append((child_path + "." + n, m))
                        else:
                            named_modules_queue.append((n, m))
                    continue
                if hasattr(sharder, "reshard"):
                    sharded_module = child_module
                    sharded_module_fqn = child_path
                    break
        else:  # Parse the fqn to identify module to be resharded
            steps = sharded_module_fqn.split(".")
            sharded_module = self.module
            for s in steps:
                sharded_module = getattr(sharded_module, s)

            # TODO: consider sharding unsharded module
            assert isinstance(
                sharded_module, ShardedModule
            ), "Given module is unsharded"
            assert changed_shard_to_params is not None
            sharder_key = sharded_module.unsharded_module_type
            sharder = self._sharder_map[sharder_key]
            assert hasattr(
                sharder, "reshard"
            ), "reshard is not implemented for this sharder"

        assert sharder is not None, "Could not find sharder to reshard"
        assert (
            sharded_module is not None and sharded_module_fqn is not None
        ), "Could not find sharded_module to reshard"
        data_volume, delta_plan = changed_shard_to_params[sharded_module_fqn]

        sharded_module = sharder.reshard(  # pyre-ignore
            sharded_module,
            delta_plan,
            self._env,
            self.device,
        )

        # Need to use .module to maintain FQN consistency
        self._optim: CombinedOptimizer = self._init_optim(
            self._dmp_wrapped_module.module  # pyre-ignore
        )
        self._plan.plan[sharded_module_fqn] = sharded_module.module_sharding_plan
        return data_volume


class DMPCollection(DistributedModelParallel):
    """
    A wrapper around DistributedModelParallel that allows for multiple DMPs to be created and managed together.

    This class implements a 2D parallelism model where a DMP is sharded over a subset of ranks.
    The current implementation shards the model such that, for a given shard, its replicated shards lie on the ranks within the node.
    This significantly improves the performance of the all-reduce communication (parameter sync) by utilizing intra-node bandwidth.

    Example Use Case:
        Consider a setup with 2 nodes, each with 4 GPUs. The sharding groups could be:
            - Group 0, DMP 0: [0, 2, 4, 6]
            - Group 1, DMP 1: [1, 3, 5, 7]

        Each group receives an identical sharding plan for their local world size and ranks.
        If we have one table sharded in each DMP, with one shard on each rank in the group,
        each shard in DMP0 will have a duplicate shard on its corresponding rank in DMP1.
        The replication groups would be: [0, 1], [2, 3], [4, 5], [6, 7].

    Notes:
        - DTensor must be used for state dict for checkpointing to work correctly.
        - The expected sharding plan should be sharded across sharding_group_size (sharding group world size)
          and broadcasted to all ranks (`planner.collective_plan(..)`).

    Args:
            module (nn.Module): The module to be sharded.
            device (torch.device): The device to use for the sharded module.
            plan (ShardingPlan): The sharding plan to use, created for sharding group world size.
            sharding_group_size (int): The number of GPUs to model parallel shard the embedding tables over
            world_size (int): The total number of GPUs.
            global_pg (dist.ProcessGroup): The global process group.
            node_group_size (Optional[int]): Specify a logical group size for a node for TWRW/GRID sharding schemes
            sharders (Optional[List[ModuleSharder[torch.nn.Module]]]): The sharders to use.
            init_data_parallel (bool): Whether to initialize data parallelism.
            init_parameters (bool): Whether to initialize parameters.
            data_parallel_wrapper (Optional[DataParallelWrapper]): The data parallel wrapper to use.

    Example::

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection):
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=global_world_size,
                local_world_size=sharding_group_size,
            ),
            constraints=constraints,
        )
        plan = planner.collective_plan(m, sharders, global_pg)
        m = DMPCollection(
            module=m,
            sharding_group_size=sharding_group_size,
            world_size=global_world_size,
            global_pg=global_pg,
            plan=plan,
        )
        m.apply(init_weights)
    """

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        plan: ShardingPlan,
        world_size: int,
        sharding_group_size: int,
        global_pg: dist.ProcessGroup,
        node_group_size: Optional[int] = None,
        sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
        init_data_parallel: bool = True,
        init_parameters: bool = True,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
        use_inter_host_allreduce: bool = False,
        custom_all_reduce: Optional[Callable[[List[torch.Tensor]], None]] = None,
        submodule_configs: Optional[List[DMPCollectionConfig]] = None,
    ) -> None:
        assert (
            device.type == "cuda" or device.type == "mtia"
        ), "DMPCollection only supports CUDA or MTIA"
        self._device = device
        self._pg: dist.ProcessGroup = global_pg
        self._global_rank: int = dist.get_rank(global_pg)
        self._custom_all_reduce = custom_all_reduce

        if sharders is None:
            sharders = get_default_sharders()
        self._sharder_map: Dict[Type[nn.Module], ModuleSharder[nn.Module]] = {
            sharder.module_type: sharder for sharder in sharders
        }

        # the args provided by the users are used for default modules
        # default context is index 0, TODO - if cleaner way to distinguish
        self._ctxs: List[DMPCollectionContext] = [
            DMPCollectionContext(
                # default context has module type None
                module=None,  # pyre-ignore[6]
                plan=plan,
                sharding_group_size=sharding_group_size,
                node_group_size=node_group_size,
                use_inter_host_allreduce=use_inter_host_allreduce,
            )
        ]

        if submodule_configs is not None:
            for submodule_config in submodule_configs:
                self._ctxs.append(
                    DMPCollectionContext(
                        module=submodule_config.module,
                        plan=submodule_config.plan,
                        sharding_group_size=submodule_config.sharding_group_size,
                        use_inter_host_allreduce=submodule_config.use_inter_host_allreduce,
                    )
                )

        # create process groups and remap sharding plans per module context
        for ctx in self._ctxs:
            (
                device_mesh,
                sharding_pg,
                replica_pg,
            ) = self._create_process_groups(
                global_rank=self._global_rank,
                world_size=world_size,
                local_size=ctx.sharding_group_size,
                use_inter_host_allreduce=ctx.use_inter_host_allreduce,
            )

            ctx.device_mesh = device_mesh
            ctx.sharding_pg = sharding_pg
            ctx.replica_pg = replica_pg

            step = world_size // ctx.sharding_group_size
            self._remap_sharding_plan(
                plan=ctx.plan,
                rank=self._global_rank,
                step=step,
                sharding_group_size=ctx.sharding_group_size,
                use_inter_host_allreduce=ctx.use_inter_host_allreduce,
            )

            if ctx.module:
                # pyre-ignore[16]
                ctx.sharded_module = self._sharder_map[ctx.module].sharded_module_type

        consolidated_plan = self._ctxs[0].plan
        for ctx in self._ctxs[1:]:
            for key, val in ctx.plan.plan.items():
                consolidated_plan.plan[key] = val

        logger.info(
            "[TorchRec 2D Parallel] Consolidated sharding plan:\n%s", consolidated_plan
        )

        default_env = ShardingEnv2D(
            global_pg=self._pg,
            sharding_pg=self._ctxs[0].sharding_pg,
            device_mesh=self._ctxs[0].device_mesh,
            node_group_size=node_group_size,
            use_inter_host_allreduce=self._ctxs[0].use_inter_host_allreduce,
        )

        super().__init__(  # type: ignore[misc]
            module,
            default_env,
            device,
            consolidated_plan,
            sharders,
            init_data_parallel,
            init_parameters,
            data_parallel_wrapper,
        )

        # post DMP init, we group sharded modules for parameter sync, stored in the context
        self._group_sharded_modules(self._ctxs)

    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str = "",
    ) -> nn.Module:

        # pre-sharded module
        if isinstance(module, ShardedModule):
            return module

        # shardable module
        module_sharding_plan = self._plan.get_plan_for_module(path)
        if module_sharding_plan:
            env = self._env
            sharder_key = type(module)

            for ctx in self._ctxs[1:]:
                if ctx.module == sharder_key:
                    env = ShardingEnv2D(
                        global_pg=self._pg,
                        sharding_pg=ctx.sharding_pg,
                        device_mesh=ctx.device_mesh,
                        node_group_size=ctx.sharding_group_size,
                        use_inter_host_allreduce=ctx.use_inter_host_allreduce,
                    )
                    break

            module = self._sharder_map[sharder_key].shard(
                module,
                module_sharding_plan,
                env,
                self.device,
                path,
            )
            return module

        for name, child in module.named_children():
            child = self._shard_modules_impl(
                child,
                path + "." + name if path else name,
            )
            setattr(module, name, child)

        return module

    def sync(self, include_optimizer_state: bool = True) -> None:
        """
        Syncs the DMP weights across the allreduce (inter) process group

        This method is called after each train step to synchronize the weights of the sharded modules.
        It uses the `dist.AllreduceCoalescedOptions` to perform an all-reduce operation on the weights,
        which averages the weights across all processes in the inter-process group.

        The default CUDA stream is used for the all-reduce operation, and the method does not return any value.

        Args:
            include_optimizer_state (bool): Flag to include optimizer state syncing upon call
        """
        # we sync per context to use the right all reduce process group
        for ctx in self._ctxs:
            self._sync(
                ctx.replica_pg,
                ctx.modules_to_sync,
                include_optimizer_state,
            )

    def _sync(
        self,
        replica_pg: dist.ProcessGroup,
        modules_to_sync: List[nn.Module],
        include_optimizer_state: bool = True,
    ) -> None:
        assert replica_pg is not None, "replica_pg is not initialized!"
        all_weights_by_dtype: dict[torch.dtype, List[torch.Tensor]] = defaultdict(list)

        for emb_kernel in modules_to_sync:
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            for w in emb_kernel.split_embedding_weights():
                all_weights_by_dtype[w.dtype].append(w)

        opts = None
        if self._custom_all_reduce is None:
            opts = dist.AllreduceCoalescedOptions()
            opts.reduceOp = dist.ReduceOp.AVG
        self._allreduce_tensors(
            replica_pg, all_weights_by_dtype, "## 2d_weight_sync ##", opts
        )

        if include_optimizer_state:
            optimizer_tensors_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = (
                defaultdict(list)
            )
            for emb_kernel in modules_to_sync:
                # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
                optimizer_states = emb_kernel.get_optimizer_state()
                for state in optimizer_states:
                    opt_tensor = state["sum"]
                    optimizer_tensors_by_dtype[opt_tensor.dtype].append(opt_tensor)
            if optimizer_tensors_by_dtype:
                self._allreduce_tensors(
                    replica_pg,
                    optimizer_tensors_by_dtype,
                    "## 2d_optimizer_sync ##",
                    opts,
                )

    def _allreduce_tensors(
        self,
        pg: dist.ProcessGroup,
        tensors_dict: Dict[torch.dtype, List[torch.Tensor]],
        annotation: str,
        opts: Optional[dist.AllreduceCoalescedOptions] = None,
    ) -> None:
        """
        Helper to perform all reduce on given tensors, uses custom all reduce function if provided
        We perform all reduce per tensor dtype per collective constraints.
        """

        custom_all_reduce = self._custom_all_reduce
        if custom_all_reduce is not None:

            def _all_reduce(tensors: List[torch.Tensor]) -> None:
                with record_function(f"{annotation}_custom_hook"):
                    custom_all_reduce(tensors)

        else:

            def _all_reduce(tensors: List[torch.Tensor]) -> None:
                with record_function(annotation):
                    pg.allreduce_coalesced(tensors, opts=opts).wait()

        for tensor_list in tensors_dict.values():
            _all_reduce(tensor_list)

    def set_all_reduce_hook(
        self,
        reduce_hook: Callable[[List[torch.Tensor]], None],
    ) -> None:
        """
        Replace default all reduce with custom callable. Users can alternatively
        pass in the custom all reduce function through the constructor. The hook
        expects the user to handle distributed communication call, associated
        process group, and stream synchronization.

        Args:
            reduce_hook (Callable[[List[torch.Tensor]], torch.Tensor]): The custom all reduce function to use for
                embedding weights and optimizer states
        """
        if self._custom_all_reduce is not None:
            logger.warning(
                "[TorchRec 2D Parallel] Custom all reduce function already defined, overriding with new callable"
            )
        self._custom_all_reduce = reduce_hook

    def _create_process_groups(
        self,
        global_rank: int,
        world_size: int,
        local_size: int,
        use_inter_host_allreduce: bool = False,
    ) -> Tuple[DeviceMesh, dist.ProcessGroup, dist.ProcessGroup]:
        """
        Creates process groups for sharding and replication, the process groups
        are created using the DeviceMesh API.

        Args:
            global_rank (int): The global rank of the current process.
            world_size (int): The total number of ranks.
            local_size (int): The number of ranks per sharding group.

        Returns:
            Tuple[DeviceMesh, dist.ProcessGroup, dist.ProcessGroup]: A tuple containing the device mesh,
                replication process group, and allreduce process group.
        """
        peer_matrix = []
        mesh, sharding_pg, replica_pg = None, None, None

        logger.warning(f"[2D] Use inter host all reduce: {use_inter_host_allreduce}")

        if use_inter_host_allreduce:
            # We shard on continuous set of ranks and nodes. Thereby forcing our all reduce to be inter host.
            # Under this scheme sharding types such as TWRW and GRID will now take
            # advantage of intra node comms as a result of the continuous set of ranks.
            peer_matrix = [
                list(range(i, i + local_size)) for i in range(0, world_size, local_size)
            ]
        else:
            step = world_size // local_size
            for group_rank in range(world_size // local_size):
                peers = [step * r + group_rank for r in range(local_size)]
                peer_matrix.append(peers)

        mesh = DeviceMesh(
            device_type=self._device.type,
            mesh=peer_matrix,
            mesh_dim_names=("replicate", "shard"),
        )

        logger.warning(f"[Connection] 2D Device Mesh created: {mesh}")
        sharding_pg = mesh.get_group(mesh_dim="shard")
        logger.warning(
            f"[Connection] 2D sharding_group: [{global_rank}] -> [{mesh['shard']}]"
        )
        replica_pg = mesh.get_group(mesh_dim="replicate")
        logger.warning(
            f"[Connection] 2D replica_group: [{global_rank}] -> [{mesh['replicate']}]"
        )

        return mesh, sharding_pg, replica_pg

    def _remap_sharding_plan(
        self,
        plan: ShardingPlan,
        rank: int,
        step: int,
        sharding_group_size: int,
        use_inter_host_allreduce: bool = False,
    ) -> None:
        """
        Remaps the sharding plan to the local replica process group ranks
        ShardingPlan is remapped inplace.

        As an example,
            ShardingPlan for created for ranks [0, 2, 4, 6] is remapped to ranks [1, 3, 5, 7]

        Args:
            plan (ShardingPlan): The original sharding plan.
            global_rank (int): The global rank of the current process.
            num_nodes (int): The number of nodes.
        """
        group_start = rank % step
        for key in plan.plan:
            # pyre-ignore[16]
            for _, param_sharding in plan.plan[key].items():
                new_ranks = []
                if use_inter_host_allreduce:
                    group = rank // sharding_group_size
                    new_ranks = [
                        shard_rank + (group * sharding_group_size)
                        for shard_rank in param_sharding.ranks
                    ]
                else:
                    for shard_rank in param_sharding.ranks:
                        new_ranks.append(shard_rank * step + group_start)
                param_sharding.ranks = new_ranks

                if isinstance(param_sharding.sharding_spec, EnumerableShardingSpec):
                    shards = param_sharding.sharding_spec.shards
                    if shards is not None:
                        for shard in shards:
                            if use_inter_host_allreduce:
                                shard_rank = shard.placement._rank + (
                                    (rank // sharding_group_size) * sharding_group_size
                                )
                            else:
                                shard_rank = shard.placement._rank * step + group_start
                            shard.placement = _remote_device(
                                f"rank:{shard_rank}/{self._device.type}:{shard_rank % get_local_size()}"
                            )
        return

    def _group_sharded_modules(
        self,
        contexts: List[DMPCollectionContext],
    ) -> None:
        # Post init DMP, save the embedding kernels, with respect to contexts
        for context in contexts[1:]:
            context.modules_to_sync = self._group_sharded_module(
                context.sharded_module  # pyre-ignore[6]
            )

        # Group leftover embedding kernels, with respect to default context
        # pyre-ignore[9]
        modules_to_skip: List[nn.Module] = [c.sharded_module for c in contexts[1:]]
        sharded_modules: List[nn.Module] = []

        def _find_sharded_modules(
            module: nn.Module,
        ) -> None:
            if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
                sharded_modules.append(module)
            if not isinstance(
                module, tuple(modules_to_skip)  # pyre-ignore[6]
            ) and hasattr(module, "_lookups"):
                for lookup in module._lookups:  # pyre-ignore[29]
                    _find_sharded_modules(lookup)

            for _, child in module.named_children():
                _find_sharded_modules(child)

        _find_sharded_modules(self._dmp_wrapped_module)
        contexts[0].modules_to_sync = sharded_modules

    def _group_sharded_module(
        self,
        sharded_module: nn.Module,
    ) -> List[nn.Module]:
        # Traverse module and find all sharded module kernels matching the sharded module
        # Post init DMP, save the embedding kernels
        sharded_modules: List[nn.Module] = []

        def _find_sharded_modules(
            module: nn.Module,
        ) -> None:
            if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
                sharded_modules.append(module)
            if isinstance(module, sharded_module):  # pyre-ignore[6]
                for lookup in module._lookups:  # pyre-ignore[29]
                    _find_sharded_modules(lookup)

            for _, child in module.named_children():
                _find_sharded_modules(child)

        _find_sharded_modules(self._dmp_wrapped_module)
        return sharded_modules

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        Returns the device mesh used for 2D parallelism.
        Contains two dimensions: "replicate" and "shard".
        """
        return self._ctxs[0].device_mesh
