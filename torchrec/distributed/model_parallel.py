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
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Set, Tuple, Type

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    res_bitmap_compact,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)
from torch.distributed.device_mesh import BackendConfig
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DeviceMesh
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.collective_utils import (
    create_on_rank_and_share_result,
    init_collective_validation,
)
from torchrec.distributed.comm import get_local_size, get_topology_domain_multiple
from torchrec.distributed.ddp_comm_hooks import (
    bf16_stream_compress_hook,
    CommHook,
    fp16_stream_compress_hook,
    StreamCompressState,
)
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.logging_handlers import log_two_dim_sharding_config
from torchrec.distributed.mc_embedding_modules import (
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.model_tracker.model_delta_tracker import (
    ModelDeltaTracker,
    ModelDeltaTrackerTrec,
)
from torchrec.distributed.model_tracker.trackers.raw_id_tracker import RawIdTracker
from torchrec.distributed.model_tracker.types import (
    DeltaTrackerConfig,
    ModelTrackerConfigs,
    RawIdTrackerConfig,
    Trackers,
)
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
    ShardingStrategy,
)
from torchrec.distributed.utils import (
    add_prefix_to_state_dict,
    append_prefix,
    copy_to_device,
    filter_state_dict,
    init_data_parallel_state,
    sharded_model_copy,
)
from torchrec.modules.embedding_modules import mark_data_parallel_skip_grad_sync
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


_DDP_STATE_DICT_PREFIX = "module."


def _is_triton_tbe(module: nn.Module) -> bool:
    return getattr(module, "_is_triton_tbe", False) is True


def _is_embedding_kernel_with_sync_tensors(module: nn.Module) -> bool:
    return isinstance(module, SplitTableBatchedEmbeddingBagsCodegen) or _is_triton_tbe(
        module
    )


@dataclass
class _PackedUsedRows:
    target: torch.Tensor
    row_ids: torch.Tensor
    payload: torch.Tensor


@dataclass
class DMPModuleSyncHandle:
    """Outstanding replica synchronization for one planned sharded module."""

    module_fqn: str
    weight_works: List[dist.Work] = field(default_factory=list)
    optimizer_works: List[dist.Work] = field(default_factory=list)
    packed_weight_rows: List[_PackedUsedRows] = field(default_factory=list)
    packed_optimizer_rows: List[_PackedUsedRows] = field(default_factory=list)


@dataclass
class _DMPModuleSyncUnit:
    kernel: nn.Module
    parent: nn.Module
    context: DMPCollectionContext
    weights_by_dtype: Dict[torch.dtype, List[torch.Tensor]]
    optimizer_by_dtype: Dict[torch.dtype, List[torch.Tensor]]


@dataclass
class _UsedRowKernelState:
    kernel: SplitTableBatchedEmbeddingBagsCodegen
    context: DMPCollectionContext
    weights: List[torch.Tensor]
    optimizer_states: List[Optional[torch.Tensor]]
    table_offsets: List[int]
    table_boundaries: torch.Tensor
    rows_seen: torch.Tensor
    count_cpu: torch.Tensor
    prep_stream: torch.Stream
    ready_event: torch.Event
    compacted_rows: Optional[torch.Tensor] = None


def _populate_updatable_modules(
    func: Callable[..., nn.Module],
) -> Callable[..., nn.Module]:
    """
    Decorator that populates the list of modules that can be updated with kjt.
    Specifically, modules with enable_embedding_update flag set to True.

    Applied to _shard_modules_impl to automatically process returned modules.
    """

    @wraps(func)
    def wrapper(
        self: "DistributedModelParallel",
        module: nn.Module,
        path: str = "",
        module_id_cache: Optional[Dict[str, "ShardedModule"]] = None,
    ) -> nn.Module:
        result = func(self, module, path, module_id_cache)

        module_id = id(result)
        if module_id_cache and module_id in module_id_cache:
            # skip adding duplicate one
            return result

        if isinstance(result, ShardedEmbeddingCollection) and getattr(
            result, "enable_embedding_update", False
        ):
            self._writable_sharded_modules.append(result)

        return result

    return wrapper


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

    def _select_allreduce_comm_hook(
        self,
        pg: dist.ProcessGroup,
        device: torch.device,
    ) -> Optional[Tuple[Any, CommHook]]:
        """
        Returns the state and hook for this precision, or None for FP32.

        The `_stream` precisions additionally require a CUDA device to create
        their stream on, and are a no-op without one. Anything unrecognized is
        a no-op too, as it has always been.

        Args:
            pg: the process group DDP was built with.
            device: the device DDP is running on.

        Returns:
            The `(state, hook)` pair to register, or None to leave gradients in
            FP32.
        """
        stream_hook: CommHook
        match self._allreduce_comm_precision:
            # The stock hooks take the process group as their state and fall
            # back to WORLD when it is None. Keep passing None, as torchrec
            # always has.
            case "fp16":
                return None, ddp_default_hooks.fp16_compress_hook
            case "bf16":
                return None, ddp_default_hooks.bf16_compress_hook
            case "fp16_stream":
                stream_hook = fp16_stream_compress_hook
            case "bf16_stream":
                stream_hook = bf16_stream_compress_hook
            case _:
                return None

        if device.type != "cuda":
            logger.warning(
                f"No comm hook was registered for "
                f"allreduce_comm_precision={self._allreduce_comm_precision!r}: "
                f"the stream compress hooks need a CUDA device to create their "
                f"decompression stream, but got {device}. Gradients will "
                "allreduce in FP32."
            )
            return None

        return StreamCompressState(process_group=pg), stream_hook

    def _register_allreduce_comm_hook(
        self,
        ddp: DistributedDataParallel,
        pg: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        """
        Registers the comm hook for this precision on `ddp`, if any.

        Args:
            ddp: the freshly constructed DDP module to hook.
            pg: the process group DDP was built with.
            device: the device DDP is running on.
        """
        selected = self._select_allreduce_comm_hook(pg, device)
        if selected is None:
            return

        state, hook = selected
        ddp.register_comm_hook(state, hook)

        logger.info(f"Registered {hook.__name__} for {self._allreduce_comm_precision}")

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
        if dmp._skip_data_parallel_grad_sync:
            init_data_parallel_state(
                dmp._dmp_wrapped_module,
                pg,
                params_to_ignore=ddp_ignore_param_names,
                broadcast_buffers=False,
            )
            return
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=dmp._dmp_wrapped_module,
            params_and_buffers_to_ignore=ddp_ignore_param_names,
        )
        # initialize DDP
        ddp = DistributedDataParallel(
            module=dmp._dmp_wrapped_module.to(device),
            device_ids=None if device.type == "cpu" else [device],
            process_group=pg,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=self._static_graph,
            find_unused_parameters=self._find_unused_parameters,
            bucket_cap_mb=self._bucket_cap_mb,
            **self._ddp_kwargs,
        )
        dmp._dmp_wrapped_module = ddp
        self._register_allreduce_comm_hook(ddp, pg, device)

    def wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        if dmp._skip_data_parallel_grad_sync and env.process_group is None:
            return
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


def find_ddp_modules(module: nn.Module) -> List[DistributedDataParallel]:
    """Returns all DDP instances in a module tree, including hidden lookups."""
    ddp_modules: List[DistributedDataParallel] = []
    visited_module_ids: Set[int] = set()

    def _visit(module: nn.Module) -> None:
        if id(module) in visited_module_ids:
            return
        visited_module_ids.add(id(module))
        if isinstance(module, DistributedDataParallel):
            ddp_modules.append(module)
        lookups = getattr(module, "_lookups", None) or ()
        for child in (*module.children(), *lookups):
            if isinstance(child, nn.Module):
                _visit(child)

    _visit(get_module(module))
    return ddp_modules


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
        skip_data_parallel_grad_sync (bool): evaluation-only mode that omits DDP
            gradient reducers. With a process group, startup broadcasts dense
            parameters and supported DATA_PARALLEL lookup parameters and buffers
            from rank 0. Source weights loaded afterward must match across ranks.
            Backward produces local, unsynchronized gradients; fused embedding
            optimizers update independently on each rank. Do not use this mode
            for distributed training. Calling init_data_parallel() raises.

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
        model_tracker_configs: Optional[ModelTrackerConfigs] = None,
        skip_data_parallel_grad_sync: bool = False,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        self.init_parameters = init_parameters
        self._skip_data_parallel_grad_sync: bool = skip_data_parallel_grad_sync
        if skip_data_parallel_grad_sync and data_parallel_wrapper is not None:
            raise ValueError(
                "skip_data_parallel_grad_sync=True cannot be combined with a custom "
                "data_parallel_wrapper because the wrapper would be skipped"
            )
        if skip_data_parallel_grad_sync and not init_data_parallel:
            raise ValueError(
                "skip_data_parallel_grad_sync=True requires "
                "init_data_parallel=True for device placement"
            )

        self._ddp_wrapped: bool = False

        if env is None:
            pg = dist.GroupMember.WORLD
            assert pg is not None, "Process group is not initialized"
            env = ShardingEnv.from_process_group(pg)
        self._env: ShardingEnv = env

        if self._env.process_group is not None:
            # Only validate on WORLD or groups that contain all ranks.
            # Subgroups (e.g., sharding_pg/replica_pg in DMPCollection) must not
            # issue constructor-time collectives as they can deadlock if DMP
            # instances are created with different process groups or ordering.
            pg = self._env.process_group
            if pg is dist.GroupMember.WORLD or pg.size() == dist.get_world_size():
                init_collective_validation(pg)

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
                    pod_size=get_topology_domain_multiple(),
                )
            )
            pg = self._env.process_group
            if pg is not None:
                plan = planner.collective_plan(module, self.sharders, pg)
            else:
                plan = planner.plan(module, self.sharders)
        self._plan: ShardingPlan = plan
        self._writable_sharded_modules: list[ShardedEmbeddingCollection] = []
        self._dmp_wrapped_module: nn.Module = self._init_dmp(module)
        if self._skip_data_parallel_grad_sync and find_ddp_modules(
            self._dmp_wrapped_module
        ):
            raise RuntimeError(
                "skip_data_parallel_grad_sync=True is not supported by every "
                "sharder in this DMP plan"
            )
        self._optim: CombinedOptimizer = self._init_optim(self._dmp_wrapped_module)

        if init_parameters:
            self._init_parameters(self.module)

        if init_data_parallel and not self._skip_data_parallel_grad_sync:
            self.init_data_parallel()
        elif (
            init_data_parallel
            and self._skip_data_parallel_grad_sync
            and self.device.type != "meta"
        ):
            self._dmp_wrapped_module.to(self.device)
            self._data_parallel_wrapper.wrap(self, self._env, self.device)

        self.model_trackers: Dict[str, ModelDeltaTracker] = {}

        if (
            model_tracker_configs is not None
            and model_tracker_configs.raw_id_tracker_config is not None
        ):
            self.model_trackers[Trackers.RAW_ID_TRACKER.name] = (
                self._init_raw_id_tracker(
                    model_tracker_configs.raw_id_tracker_config,
                    self._dmp_wrapped_module,
                )
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
        for tracker in self.model_trackers.values():
            # The step() call advances the internal batch counter so that subsequent ID tracking and delta
            # retrieval operations can be properly organized by batch boundaries.

            # Context: ModelDeltaTracker tracks unique embedding IDs (and optionally embeddings / states)
            # useful for calculating topk rows for checkpointing and for updating fresh embedding weights
            # between predictors and trainers in online training scenarios.
            tracker.step()

        # Model forward for DMP wrapped model
        return self._dmp_wrapped_module(*args, **kwargs)

    def init_data_parallel(self) -> None:
        """
        See init_data_parallel c-tor argument for usage.
        It's safe to call this method multiple times.
        """
        if self._skip_data_parallel_grad_sync:
            raise RuntimeError(
                "init_data_parallel() cannot be called on a DMP constructed with "
                "skip_data_parallel_grad_sync=True"
            )
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
        module_id_cache: Dict[int, ShardedModule] = {}
        return self._shard_modules_impl(module, module_id_cache=module_id_cache)

    def _init_delta_tracker(
        self, delta_tracker_config: DeltaTrackerConfig, module: nn.Module
    ) -> ModelDeltaTracker:
        # Init delta tracker if config is provided
        return ModelDeltaTrackerTrec(
            model=module,
            consumers=delta_tracker_config.consumers,
            delete_on_read=delta_tracker_config.delete_on_read,
            auto_compact=delta_tracker_config.auto_compact,
            mode=delta_tracker_config.tracking_mode,
            fqns_to_skip=delta_tracker_config.fqns_to_skip,
        )

    def _init_raw_id_tracker(
        self, raw_id_tracker_config: RawIdTrackerConfig, module: nn.Module
    ) -> RawIdTracker:
        return RawIdTracker(
            model=module,
            delete_on_read=raw_id_tracker_config.delete_on_read,
            fqns_to_skip=raw_id_tracker_config.fqns_to_skip,
        )

    def _init_optim(self, module: nn.Module) -> CombinedOptimizer:
        module_id_cache: Dict[int, KeyedOptimizer] = {}
        return CombinedOptimizer(
            # pyre-ignore [6]
            self._fused_optim_impl(module, [], module_id_cache=module_id_cache)
        )

    def _fused_optim_impl(
        self,
        module: nn.Module,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
        path: str = "",
        module_id_cache: Optional[Dict[int, KeyedOptimizer]] = None,
    ) -> List[Tuple[str, KeyedOptimizer]]:
        if isinstance(module, FusedOptimizerModule):
            if module_id_cache is not None:
                module_id = id(module)
                if module_id in module_id_cache:
                    # This module's optimizer was already added at a different path.
                    # Skip adding it again to avoid duplicate optimizer state in
                    # CombinedOptimizer, which would cause save/load asymmetry.
                    logger.error(
                        f"Module {path} optimizer already collected "
                        f"(module ID {module_id} at different path)"
                    )
                    return fused_optims
                module_id_cache[module_id] = module.fused_optimizer
            fused_optims.append((path, module.fused_optimizer))
            return fused_optims

        for name, child in module.named_children():
            self._fused_optim_impl(
                child,
                fused_optims,
                path + "." + name if path else name,
                module_id_cache,
            )
        return fused_optims

    @_populate_updatable_modules
    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str = "",
        module_id_cache: Optional[Dict[int, ShardedModule]] = None,
    ) -> nn.Module:
        # pre-sharded module
        if isinstance(module, ShardedModule):
            return module

        # Only used when module_id_cache is provided
        module_id = id(module)
        if module_id_cache is not None:
            if module_id in module_id_cache:
                """
                This is likely due to a single sparse module being used in multiple places in the model,
                which results in multiple FQNs for the same sparse module. The dedup logic is applied on
                the sharded module, i.e., multiple FQNs will refer to the same sharded module, as it is in
                eager-mode sparse module. However, there could be potential issues in other places where
                model is travesed via `named_children()`, the same sparse module will be visited multiple
                times again.
                """
                logger.error(
                    f"Module {path} is already in cache (replaced by sharded module already)"
                )
                return module_id_cache[module_id]

        # shardable module
        module_sharding_plan = self._plan.get_plan_for_module(path)
        if module_sharding_plan:
            if self._skip_data_parallel_grad_sync:
                mark_data_parallel_skip_grad_sync(module)
            sharder_key = type(module)
            sharded_module = self._sharder_map[sharder_key].shard(
                module,
                module_sharding_plan,  # pyre-ignore[6]
                self._env,
                self.device,
                path,
            )
            if module_id_cache is not None:
                module_id_cache[module_id] = sharded_module
            return sharded_module

        for name, child in module.named_children():
            child = self._shard_modules_impl(
                child,
                path + "." + name if path else name,
                module_id_cache,
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

    def init_torchrec_delta_tracker(
        self, delta_tracker_config: DeltaTrackerConfig
    ) -> ModelDeltaTracker:
        """
        Initializes the model delta tracker if it doesn't exists.
        """
        if Trackers.DELTA_TRACKER.name not in self.model_trackers:
            self.model_trackers[Trackers.DELTA_TRACKER.name] = self._init_delta_tracker(
                delta_tracker_config, self._dmp_wrapped_module
            )

        return self.model_trackers[Trackers.DELTA_TRACKER.name]

    def get_delta_tracker(self) -> Optional[ModelDeltaTracker]:
        """
        Returns the delta tracker if it exists.
        """
        if Trackers.DELTA_TRACKER.name in self.model_trackers:
            return self.model_trackers[Trackers.DELTA_TRACKER.name]
        return None

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
        # pyre-ignore[6]
        state_dict = get_module(self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        assert state_dict is not None
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

    def write(self, *input, **kwargs) -> None:
        """
        Write features to the sharded module if it has enable_embedding_update flag.
        """
        if len(self._writable_sharded_modules) == 0:
            raise RuntimeError(
                "No writable sharded modules found. Please check `enable_embedding_update` flag in your embedding config"
            )

        for module in self._writable_sharded_modules:
            module.write(*input, **kwargs)

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
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

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


class HybridEvalDMP(DistributedModelParallel):
    """
    DMP for eval-only workflows with split-device placement
    (e.g., CPU embeddings + GPU dense).

    - Defaults to ``init_data_parallel=False`` (no DDP gradient sync for eval)
    - Calls ``self.eval()`` on construction
    - Recursive ``.to()`` skips entire ``ShardedModule`` subtrees, preserving
      the device placement of sharded embedding modules
    - ``share_embedding_memory()`` deduplicates CPU embedding storage across
      ranks on the same host via POSIX shared memory (``/dev/shm``)
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        init_data_parallel: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(module, init_data_parallel=init_data_parallel, **kwargs)

    # pyrefly: ignore[bad-override]: inconsistent override
    def to(self, *args: Any, **kwargs: Any) -> "HybridEvalDMP":
        def _selective_to(mod: nn.Module) -> None:
            for key, param in mod._parameters.items():
                if param is not None:
                    mod._parameters[key] = nn.Parameter(
                        param.data.to(*args, **kwargs),
                        requires_grad=param.requires_grad,
                    )
            for key, buf in mod._buffers.items():
                if buf is not None:
                    mod._buffers[key] = buf.to(*args, **kwargs)
            for child in mod.children():
                if not isinstance(child, ShardedModule):
                    _selective_to(child)

        _selective_to(self.module)
        return self

    def share_embedding_memory(self, pg: dist.ProcessGroup) -> None:
        """
        Share CPU embedding parameters across ranks via POSIX shared memory.

        Walks all ``ShardedModule`` children, finds CPU parameters, deduplicates
        by underlying storage, and uses ``create_on_rank_and_share_result`` to
        place the storage in ``/dev/shm``. Rank 0 is the creator; other ranks
        map the same physical memory — no data is copied.

        Args:
            pg: Process group whose members share memory. Should be an
                intra-node (host-local) group for multi-host jobs.
        """
        # Step 1: Collect CPU params from ShardedModules, dedup by storage ptr
        seen_ptrs: Dict[int, torch.Tensor] = {}
        param_info: List[Tuple[nn.Parameter, int]] = []

        for _fqn, module in self.module.named_modules():
            if isinstance(module, ShardedModule):
                for _pname, param in module.named_parameters():
                    if param.device.type == "cpu":
                        ptr = param.data.untyped_storage().data_ptr()
                        if ptr not in seen_ptrs:
                            seen_ptrs[ptr] = param.data
                        param_info.append((param, ptr))

        if not seen_ptrs:
            return

        unique_ptrs = list(seen_ptrs.keys())
        unique_tensors = [seen_ptrs[ptr] for ptr in unique_ptrs]

        # Step 2: Share unique storages via create_on_rank_and_share_result.
        # On rank 0, _share_filename_cpu_() moves storages to shm in-place;
        # all existing views (params) automatically point to shared memory.
        shared = create_on_rank_and_share_result(
            pg,
            0,
            creator=lambda: unique_tensors,
            extractor=lambda ts: list(ts),  # List[Tensor] -> List[Optional[Tensor]]
            constructor=lambda ts: [t for t in ts if t is not None],
        )

        # Step 3: On non-creator ranks, remap params to shared storages
        if pg.rank() != 0:
            ptr_to_shared_storage = {
                old_ptr: shared[i].untyped_storage()
                for i, old_ptr in enumerate(unique_ptrs)
            }
            for param, old_ptr in param_info:
                new_storage = ptr_to_shared_storage[old_ptr]
                param.data.set_(
                    new_storage,
                    param.data.storage_offset(),
                    param.data.shape,
                    param.data.stride(),
                )


class DMPCollection(DistributedModelParallel):
    """
    A wrapper around DistributedModelParallel that allows for multiple DMPs to be created and managed together.

    This class implements a 2D parallelism model where a DMP is sharded over a subset of ranks.
    The current implementation shards the model such that, for a given shard, its replicated shards lie on the ranks within the node.
    This significantly improves the performance of the all-reduce communication (parameter sync) by utilizing intra-node bandwidth.

    Collective Communications:
        The collective communications depend on the ShardingStrategy:

        ShardingStrategy.DEFAULT / ShardingStrategy.PER_MODULE:
            - sync(): allreduce_coalesced (ReduceOp.AVG) on replica_pg for weights
            - sync(): allreduce_coalesced (ReduceOp.AVG) on replica_pg for optimizer states

        ShardingStrategy.FULLY_SHARDED:
            - Forward pass: reduce_scatter_tensor (ReduceOp.AVG, async) on replica_pg for weights
            - Backward pass: all_gather_into_tensor on replica_pg for weights
            - Note: sync() is typically not called for FULLY_SHARDED since weight sync happens
              automatically via reduce_scatter/all_gather per iteration

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
            device (torch.device): The device to use for the sharded module. CUDA, MTIA
                and CPU are supported. CPU runs the replica-dim collectives over gloo and
                is intended for offline jobs (e.g. transfer learning) that build a sharded
                module without training on accelerators.
            plan (ShardingPlan): The sharding plan to use, created for sharding group world size.
            sharding_group_size (int): The number of GPUs to model parallel shard the embedding tables over
            world_size (int): The total number of GPUs.
            global_pg (dist.ProcessGroup): The global process group.
            node_group_size (Optional[int]): Specify a logical group size for a node for TWRW/GRID sharding schemes
            sharders (Optional[List[ModuleSharder[torch.nn.Module]]]): The sharders to use.
            init_data_parallel (bool): Whether to initialize data parallelism.
            init_parameters (bool): Whether to initialize parameters.
            data_parallel_wrapper (Optional[DataParallelWrapper]): The data parallel wrapper to use.
            use_inter_host_allreduce (bool): If True, construct sharding and replica groups to force
                inter-host all-reduce communication by assigning continuous rank ranges per shard group.
                If False, use alternating ranks to prefer intra-host all-reduce where possible.
            custom_all_reduce (Optional[Callable[[List[torch.Tensor]], None]]): Custom all-reduce
                function to override the default dist.allreduce_coalesced behavior during sync().
                The callable must perform the collective across the appropriate process group and
                handle stream synchronization.
            submodule_configs (Optional[List[DMPCollectionConfig]]): Optional per-submodule configuration
                list allowing different sharding plans and strategies for specific submodules within
                the model.
            rs_awaitable_hook_module (Optional[str]): Name of a first-level child submodule on which to
                register a forward hook that ensures reduce-scatter completion and weight resize when
                using FULLY_SHARDED strategy. Useful to avoid peak memory pressure prior to the selected
                module's forward pass.
            device_mesh_backend_override (Optional[Tuple[BackendConfig, ...]]): Backend override for
                the device mesh.


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

    MESH_DIM_NAMES: Tuple[str, str] = ("replicate", "shard")

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        plan: ShardingPlan,
        world_size: int,
        sharding_group_size: int,
        global_pg: dist.ProcessGroup,
        sharding_strategy: ShardingStrategy = ShardingStrategy.DEFAULT,
        node_group_size: Optional[int] = None,
        sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
        init_data_parallel: bool = True,
        init_parameters: bool = True,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
        use_inter_host_allreduce: bool = False,
        custom_all_reduce: Optional[Callable[[List[torch.Tensor]], None]] = None,
        submodule_configs: Optional[List[DMPCollectionConfig]] = None,
        rs_awaitable_hook_module: Optional[str] = None,
        device_mesh_backend_override: Optional[Tuple[BackendConfig, ...]] = None,
    ) -> None:
        assert device.type in (
            "cuda",
            "mtia",
            "cpu",
        ), f"DMPCollection only supports CUDA, MTIA or CPU, got {device.type}"
        # TODO: Add assertion that world_size != sharding_group_size
        self._device = device
        self._pg: dist.ProcessGroup = global_pg
        self._global_rank: int = dist.get_rank(global_pg)
        self._custom_all_reduce = custom_all_reduce
        self._all_reduce_hook: Optional[
            Callable[[dist.ProcessGroup, List[torch.Tensor]], None]
        ] = None

        if sharders is None:
            sharders = get_default_sharders()
        self._sharder_map: Dict[Type[nn.Module], ModuleSharder[nn.Module]] = {
            sharder.module_type: sharder for sharder in sharders
        }

        # Create the default context for modules without submodule configs
        self._default_ctx: DMPCollectionContext = DMPCollectionContext(
            # default context has module type None
            module=None,  # pyre-ignore[6]
            plan=plan,
            sharding_group_size=sharding_group_size,
            node_group_size=node_group_size,
            use_inter_host_allreduce=use_inter_host_allreduce,
            sharding_strategy=sharding_strategy,
            device_mesh_backend_override=device_mesh_backend_override,
        )

        self._submodule_ctxs: List[DMPCollectionContext] = []
        if submodule_configs is not None:
            for submodule_config in submodule_configs:
                self._submodule_ctxs.append(
                    DMPCollectionContext(
                        module=submodule_config.module,
                        plan=submodule_config.plan,
                        sharding_group_size=submodule_config.sharding_group_size,
                        use_inter_host_allreduce=submodule_config.use_inter_host_allreduce,
                        sharding_strategy=submodule_config.sharding_strategy,
                        device_mesh_backend_override=submodule_config.device_mesh_backend_override,
                    )
                )

        self._ctxs: List[DMPCollectionContext] = [
            self._default_ctx
        ] + self._submodule_ctxs
        self._module_sync_units: Dict[str, List[_DMPModuleSyncUnit]] = {}
        self._module_sync_units_initialized = False
        self._pending_module_syncs: Dict[str, DMPModuleSyncHandle] = {}

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
                device_mesh_backend_override=ctx.device_mesh_backend_override,
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

        consolidated_plan = self._default_ctx.plan
        for ctx in self._submodule_ctxs:
            for key, val in ctx.plan.plan.items():
                consolidated_plan.plan[key] = val

        logger.info(
            "[TorchRec 2D Parallel] Consolidated sharding plan:\n%s", consolidated_plan
        )

        # Log once per job; all ranks share the same config.
        if self._global_rank == 0:
            log_two_dim_sharding_config(
                {
                    "world_size": str(world_size),
                    "sharding_group_size": str(sharding_group_size),
                    "node_group_size": str(node_group_size),
                    "sharding_strategy": sharding_strategy.name,
                    "use_inter_host_allreduce": str(use_inter_host_allreduce),
                    "has_submodule_configs": str(bool(submodule_configs)),
                    "num_parallel_worlds": str(world_size // sharding_group_size),
                }
            )

        default_env = ShardingEnv2D(
            global_pg=self._pg,
            sharding_pg=self._default_ctx.sharding_pg,
            replica_pg=self._default_ctx.replica_pg,
            device_mesh=self._default_ctx.device_mesh,
            node_group_size=node_group_size,
            use_inter_host_allreduce=self._default_ctx.use_inter_host_allreduce,
            sharding_strategy=self._default_ctx.sharding_strategy,
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
        self._cache_sync_tensors(self._ctxs)
        if (
            sharding_strategy == ShardingStrategy.FULLY_SHARDED
            and rs_awaitable_hook_module is not None
        ):
            self._register_sparse_arch_forward_hook(rs_awaitable_hook_module)

    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str = "",
        module_id_cache: Optional[Dict[int, ShardedModule]] = None,
    ) -> nn.Module:

        # pre-sharded module
        if isinstance(module, ShardedModule):
            return module

        # Only used only when module_id_cache is provided
        module_id: int = id(module)
        if module_id_cache is not None:
            if module_id in module_id_cache:
                """
                This is likely due to a single sparse module being used in multiple places in the model,
                which results in multiple FQNs for the same sparse module. The dedup logic is applied on
                the sharded module, i.e., multiple FQNs will refer to the same sharded module, as it is in
                eager-mode sparse module. However, there could be potential issues in other places where
                model is travesed via `named_children()`, the same sparse module will be visited multiple
                times again.
                """
                logger.error(
                    f"Module {path} is already in cache (replaced by sharded module already)"
                )
                return module_id_cache[module_id]

        # shardable module
        module_sharding_plan = self._plan.get_plan_for_module(path)
        if module_sharding_plan:
            env = self._env
            sharder_key = type(module)

            for ctx in self._submodule_ctxs:
                if ctx.module == sharder_key:
                    env = ShardingEnv2D(
                        global_pg=self._pg,
                        sharding_pg=ctx.sharding_pg,
                        replica_pg=ctx.replica_pg,
                        device_mesh=ctx.device_mesh,
                        node_group_size=ctx.sharding_group_size,
                        use_inter_host_allreduce=ctx.use_inter_host_allreduce,
                        sharding_strategy=ctx.sharding_strategy,
                    )
                    break

            sharded_module = self._sharder_map[sharder_key].shard(
                module,
                module_sharding_plan,  # pyre-ignore[6]
                env,
                self.device,
                path,
            )
            if module_id_cache is not None:
                module_id_cache[module_id] = sharded_module
            return sharded_module

        for name, child in module.named_children():
            child = self._shard_modules_impl(
                child,
                path + "." + name if path else name,
                module_id_cache,
            )
            setattr(module, name, child)

        return module

    def sync(self, include_optimizer_state: bool = True) -> None:
        """
        Syncs the DMP weights across the allreduce (inter) process group

        This method is called after each train step to synchronize the weights of the sharded modules.
        It uses the `dist.AllreduceCoalescedOptions` to perform an all-reduce operation on the weights,
        which averages the weights across all processes in the inter-process group.

        Args:
            include_optimizer_state (bool): Flag to include optimizer state syncing upon call
        """
        # We sync per context to use the right all reduce process group.
        for ctx in self._ctxs:
            if len(ctx.hash_zch_modules) > 0:
                # Do syncing of managed collision modules.
                self._sync_mcc_modules(ctx, include_optimizer_state)

            self._sync(ctx, include_optimizer_state)

    def _sync_mcc_modules(
        self,
        ctx: DMPCollectionContext,
        include_optimizer_state: bool = True,
    ) -> None:
        """
        Syncs the DMP identites/metadata/weights of ManagedCollisionModule across process group.

        It syncs the hash identities across replica by merging them all together.
        The weights of the identities that did not exist in the final merge are zeroed out.
        The weights of the identities that are in final merge are averaged across those replica
        that have them.
        """
        mesh = ctx.device_mesh.mesh
        # The list of root nodes that all identities get sent to
        replica_gp_root = mesh[0]
        is_root_node = self._global_rank in replica_gp_root
        # The index of replica_gp_root to send all identities/metadata to
        index_root = torch.where(mesh == self._global_rank)
        replica_ranks = mesh[:, index_root[1][0]].tolist()

        for emb_kernel, table_name in ctx.hash_zch_modules:
            emb_kernel = cast(
                BaseShardedManagedCollisionEmbeddingCollection, emb_kernel
            )
            mpzch = emb_kernel._managed_collision_collection._managed_collision_modules[
                table_name
            ]
            # pyre-ignore[16]
            reserved_indices = mpzch.get_indices_of_reserved_slots_per_bucket()

            # Sync hash identities and metadata and get information about mapping
            survived, rank_to_global = mpzch.sync_identities(  # pyre-ignore[16]
                is_root_node,
                num_replica_gp=len(mesh),
                replica_ranks=replica_ranks,
                replica_pg=ctx.replica_pg,
            )

            # Sync the weights and optimizers of the table
            emb_kernel.sync_hash_zch_weights(
                table_name,
                rank_to_global,
                survived,
                include_optimizer_state,
                allreduce_fn=lambda dict_tensors, annotation, opts: self._allreduce_tensors(
                    ctx, dict_tensors, annotation, opts
                ),
                indices_slots=reserved_indices,
            )

    def _restore_stashed_sync_tensors(
        self,
        ctx: DMPCollectionContext,
        include_optimizer_state: bool = True,
    ) -> None:
        """Restore memory-stashed sync tensors to HBM before the 2D allreduce.

        Memory stashing (e.g. EMS) frees a tensor's HBM via
        ``storage.resize_(0)`` to reduce peak memory. If ``sync()`` runs while a
        tensor it is about to allreduce is still stashed, the collective reads
        freed memory and fails with ``cudaErrorIllegalAddress``. Restore any
        stashed sync tensors first.

        Gated so it costs nothing on the hot path: a no-op when memory stashing
        is disabled and when the sync tensors are already resident (the
        steady-state case, where the pipeline restores them during backward
        before this post-backward sync). It restores only in the window that
        would otherwise crash and does NOT re-stash -- the next forward
        re-stashes as usual, so peak memory is unchanged and no extra D2H is
        added.
        """
        # Local import to avoid any import cycle at module import time.
        from torchrec.distributed.memory_stashing import MemoryStashingManager

        if not MemoryStashingManager.is_enabled():
            return

        def _count_stashed(
            tensors_by_dtype: Dict[torch.dtype, List[torch.Tensor]],
        ) -> int:
            return sum(
                1
                for tensors in tensors_by_dtype.values()
                for t in tensors
                if t.is_cuda and t.untyped_storage().size() == 0
            )

        num_weights_stashed = _count_stashed(ctx.weights_by_dtype)
        num_optimizer_stashed = (
            _count_stashed(ctx.optimizer_tensors_by_dtype)
            if include_optimizer_state and ctx.optimizer_tensors_by_dtype
            else 0
        )

        if num_weights_stashed == 0 and num_optimizer_stashed == 0:
            # Steady state: tensors already resident -> nothing to do, no extra IO.
            return

        logger.warning(
            "[DMP 2D-sync] memory-stashed tensors detected at sync "
            "(stashed weights=%d, optimizer=%d); restoring to HBM before "
            "allreduce to avoid an illegal memory access. If this fires every "
            "step, the stash/restore ordering for these tensors needs review.",
            num_weights_stashed,
            num_optimizer_stashed,
        )

        if num_weights_stashed > 0:
            MemoryStashingManager.restore_embedding_weights()
        if num_optimizer_stashed > 0:
            MemoryStashingManager.restore_optimizer_state()

        # The restore H2D copies run on the stashing H2D stream; make the
        # current (collective) stream wait for them before the allreduce reads
        # the tensors.
        torch.cuda.current_stream().wait_stream(MemoryStashingManager.h2d_stream())

    def _sync(
        self,
        ctx: DMPCollectionContext,
        include_optimizer_state: bool = True,
    ) -> None:
        """
        Sync weights and optimizer states for a given context.

        Collective Communications:
            - allreduce_coalesced (ReduceOp.AVG) on replica_pg for weights
            - allreduce_coalesced (ReduceOp.AVG) on replica_pg for optimizer states
              (if include_optimizer_state is True)
        """
        assert ctx.replica_pg is not None, "replica_pg is not initialized!"

        sync_modules = [module for module, _ in ctx.modules_to_sync]
        if (
            self._custom_all_reduce is None
            and ctx.replica_pg.size() == 1
            and (
                not sync_modules
                or all(_is_triton_tbe(module) for module in sync_modules)
            )
        ):
            return

        # Memory stashing (e.g. EMS) may have freed the HBM of the TBE weight /
        # fused-optimizer tensors this sync allreduces. Restore any that are
        # currently stashed before the collective to avoid cudaErrorIllegalAddress.
        # No-op when stashing is off or the tensors are already resident.
        self._restore_stashed_sync_tensors(ctx, include_optimizer_state)

        opts = None
        if self._custom_all_reduce is None and self._all_reduce_hook is None:
            opts = dist.AllreduceCoalescedOptions()
            opts.reduceOp = dist.ReduceOp.AVG

        self._allreduce_tensors(ctx, ctx.weights_by_dtype, "## 2d_weight_sync ##", opts)

        if include_optimizer_state and ctx.optimizer_tensors_by_dtype:
            self._allreduce_tensors(
                ctx,
                ctx.optimizer_tensors_by_dtype,
                "## 2d_optimizer_sync ##",
                opts,
            )

    def _allreduce_tensors(
        self,
        ctx: DMPCollectionContext,
        tensors_dict: Dict[torch.dtype, List[torch.Tensor]],
        annotation: str,
        opts: Optional[dist.AllreduceCoalescedOptions] = None,
    ) -> None:
        """
        Helper to perform all reduce on given tensors, uses custom all reduce function if provided.
        We perform all reduce per tensor dtype per collective constraints.

        Collective Communication:
            - allreduce_coalesced on the provided process group (pg)

        Args:
            ctx: The DMPCollectionContext containing the process group
            tensors_dict: Dictionary of tensors grouped by dtype
            annotation: Annotation string for profiling
            opts: Optional allreduce coalesced options
        """
        all_reduce_hook = self._all_reduce_hook
        custom_all_reduce = self._custom_all_reduce
        if all_reduce_hook is not None:

            def _all_reduce(tensors: List[torch.Tensor]) -> None:
                with record_function(f"{annotation}_custom_hook"):
                    all_reduce_hook(ctx.replica_pg, tensors)

        elif custom_all_reduce is not None:
            # Custom all reduce hook
            def _all_reduce(tensors: List[torch.Tensor]) -> None:
                with record_function(f"{annotation}_custom_hook"):
                    custom_all_reduce(tensors)

        else:
            # Default allreduce_coalesced path (blocking)
            def _all_reduce(tensors: List[torch.Tensor]) -> None:
                with record_function(annotation):
                    ctx.replica_pg.allreduce_coalesced(tensors, opts=opts).wait()

        for tensor_list in tensors_dict.values():
            _all_reduce(tensor_list)

    def set_all_reduce_hook(
        self,
        reduce_hook: Callable[[dist.ProcessGroup, List[torch.Tensor]], None],
    ) -> None:
        """
        Replace the default all reduce with a process-group-aware callable.
        The hook must handle the distributed communication call and stream
        synchronization.

        Args:
            reduce_hook: Custom all reduce function for embedding weights and
                optimizer states. It receives the replication process group and
                tensors for the current DMP collection context.
        """
        if self._custom_all_reduce is not None or self._all_reduce_hook is not None:
            logger.warning(
                "[TorchRec 2D Parallel] Custom all reduce function already defined, overriding with new callable"
            )
        self._custom_all_reduce = None
        self._all_reduce_hook = reduce_hook

    def ensure_reduce_scatter_complete(self) -> None:
        """
        Ensure all reduce scatter and resize operations are complete for FULLY_SHARDED modules.

        This method can be called during a training step to guarantee that all pending
        async reduce scatter operations have finished and weight tensors have been resized.

        Collective Communication:
            Waits for completion of async reduce_scatter_tensor operations launched
            during the forward pass for FULLY_SHARDED strategy modules.

        This is a no-op if:
        - No modules are using FULLY_SHARDED strategy
        - All reduce scatter operations are already complete

        Example usage in training loop::

            model = DMPCollection(...)
            for batch in dataloader:
                output = model(batch)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                model.sync()
                # Optionally ensure reduce scatter is complete before next iteration
                model.ensure_reduce_scatter_complete()
        """
        for ctx in self._ctxs:
            if ctx.sharding_strategy == ShardingStrategy.FULLY_SHARDED:
                for _, sharded_module in ctx.modules_to_sync:
                    sharded_module.ensure_reduce_scatter_complete()  # pyre-ignore[16]

    def _register_sparse_arch_forward_hook(self, rs_awaitable_hook_module) -> None:
        """
        Registers a forward hook on a specified first-level submodule to ensure that
        reduce-scatter and subsequent weight resize operations are completed when using
        FULLY_SHARDED mode. The hook is attached to the module named by
        rs_awaitable_hook_module and runs immediately before that module’s forward pass.

        We need to select module before peak memory so that we can ensure that we release table weights memory before peak memory.
        The module selected is case by case. By default, we select sparse_arch module.

        Note: The exact hook placement could be made configurable in the future.
        """

        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], _output: Any) -> None:
            self.ensure_reduce_scatter_complete()

        # Only check first-level children for rs_awaitable_hook_module
        target: Optional[nn.Module] = None
        for name, child in self.module.named_children():
            if name == rs_awaitable_hook_module:
                target = child
                break
        assert target is not None and isinstance(
            target, nn.Module
        ), f"[TorchRec 2D Parallel] First-level submodule {rs_awaitable_hook_module} not found; forward hook not registered."
        target.register_forward_hook(_hook)
        logger.info(
            f"[TorchRec 2D Parallel] Registered reduce-scatter awaitable hook on submodule: {rs_awaitable_hook_module}"
        )

    def _create_process_groups(
        self,
        global_rank: int,
        world_size: int,
        local_size: int,
        use_inter_host_allreduce: bool = False,
        device_mesh_backend_override: Optional[Tuple[BackendConfig, ...]] = None,
    ) -> Tuple[DeviceMesh, dist.ProcessGroup, dist.ProcessGroup]:
        """
        Creates process groups for sharding and replication, the process groups
        are created using the DeviceMesh API.

        Args:
            global_rank (int): The global rank of the current process.
            world_size (int): The total number of ranks.
            local_size (int): The number of ranks per sharding group.
            device_mesh_backend_override (Optional[Tuple[BackendConfig, ...]]):
                Backend override for the device mesh.

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
            mesh_dim_names=self.MESH_DIM_NAMES,
            backend_override=device_mesh_backend_override,
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
                            assert shard.placement is not None
                            shard_rank_val = cast(int, shard.placement._rank)
                            if use_inter_host_allreduce:
                                shard_rank = shard_rank_val + (
                                    (rank // sharding_group_size) * sharding_group_size
                                )
                            else:
                                shard_rank = shard_rank_val * step + group_start
                            # Only accelerators carry a device index, matching
                            # placement() in sharding_plan.py and planner/utils.py.
                            # A "cpu:0" placement does not compare equal to a tensor's
                            # plain "cpu" device, so Shard.__post_init__ would reject
                            # every shard built from this plan.
                            param_device = self._device.type
                            if param_device in {"cuda", "mtia", "tpu"}:
                                param_device = torch.device(
                                    param_device, shard_rank % get_local_size()
                                )
                            shard.placement = _remote_device(
                                f"rank:{shard_rank}/{param_device}"
                            )
        return

    def _group_sharded_modules(
        self,
        contexts: List[DMPCollectionContext],
    ) -> None:
        """
        Group sharded modules by context for parameter synchronization.

        Args:
            contexts: List of contexts where contexts[0] is the default context
                and contexts[1:] are submodule-specific contexts.
        """
        # Process submodule-specific contexts first (contexts[1:])
        for context in contexts[1:]:
            context.modules_to_sync = self._group_sharded_module(
                context.sharded_module  # pyre-ignore[6]
            )

        # Group leftover embedding kernels, with respect to default context
        # pyre-ignore[9]
        modules_to_skip: List[nn.Module] = [c.sharded_module for c in contexts[1:]]
        sharded_modules: List[Tuple[nn.Module, nn.Module]] = []
        visited_module_ids: Set[int] = set()

        def _find_sharded_modules(
            module: nn.Module,
            prev_module: nn.Module,
        ) -> None:
            module_id = id(module)
            if module_id in visited_module_ids:
                return
            visited_module_ids.add(module_id)

            if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
                sharded_modules.append((module, prev_module))
            elif _is_triton_tbe(module):
                sharded_modules.append((module, prev_module))
            if isinstance(module, BaseShardedManagedCollisionEmbeddingCollection):
                sharded_modules.append((module, prev_module))
                return  # Stop here don't go to the children
            if not isinstance(
                module, tuple(modules_to_skip)  # pyre-ignore[6]
            ) and hasattr(module, "_lookups"):
                for lookup in module._lookups:  # pyre-ignore[29]
                    _find_sharded_modules(lookup, module)

            for _, child in module.named_children():
                _find_sharded_modules(child, module)

        # pyre-ignore[6]
        _find_sharded_modules(self._dmp_wrapped_module, None)
        contexts[0].modules_to_sync = sharded_modules

    def _group_sharded_module(
        self,
        sharded_module: nn.Module,
    ) -> List[Tuple[nn.Module, nn.Module]]:
        # Traverse module and find all sharded module kernels matching the sharded module
        # Post init DMP, save the embedding kernels
        sharded_modules: List[Tuple[nn.Module, nn.Module]] = []
        visited_module_ids: Set[int] = set()

        def _find_sharded_modules(module: nn.Module, prev_module: nn.Module) -> None:
            module_id = id(module)
            if module_id in visited_module_ids:
                return
            visited_module_ids.add(module_id)

            if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
                sharded_modules.append((module, prev_module))
            elif _is_triton_tbe(module):
                sharded_modules.append((module, prev_module))
            if isinstance(module, sharded_module):  # pyre-ignore[6]
                for lookup in module._lookups:  # pyre-ignore[29]
                    _find_sharded_modules(lookup, module)

            for _, child in module.named_children():
                _find_sharded_modules(child, module)

        _find_sharded_modules(self._dmp_wrapped_module, None)  # pyre-ignore[6]
        return sharded_modules

    def _cache_sync_tensors(
        self,
        contexts: List[DMPCollectionContext],
    ) -> None:
        """
        Pre-compute and cache the weight and optimizer tensor mappings by dtype.

        This is called once after _group_sharded_modules() to avoid rebuilding
        these mappings on every sync() call. The cached mappings are stored
        in each context for use during sync operations.
        """
        for context in contexts:
            weights_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = defaultdict(list)
            optimizer_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = defaultdict(
                list
            )
            hash_zch_modules: List[Tuple[nn.Module, str]] = []
            for emb_kernel, parent_module in context.modules_to_sync:
                if isinstance(
                    emb_kernel, BaseShardedManagedCollisionEmbeddingCollection
                ):
                    # Managed-collision metadata still needs periodic replica sync,
                    # including when table weights use fully sharded collectives.
                    for table_name in emb_kernel._table_to_tbe_and_index.keys():
                        hash_zch_modules.append((emb_kernel, table_name))

                elif _is_embedding_kernel_with_sync_tensors(emb_kernel):
                    sync_kernel = cast(Any, emb_kernel)
                    get_sync_weight_tensors = getattr(
                        parent_module, "get_sync_weight_tensors", None
                    )
                    sync_weights = (
                        get_sync_weight_tensors()
                        if get_sync_weight_tensors is not None
                        else sync_kernel.split_embedding_weights()
                    )
                    for w in sync_weights:
                        weights_by_dtype[w.dtype].append(w)
                    for state in sync_kernel.get_optimizer_state():
                        opt_tensor = state.get("sum")
                        if opt_tensor is not None:
                            optimizer_by_dtype[opt_tensor.dtype].append(opt_tensor)

            context.weights_by_dtype = dict(weights_by_dtype)
            context.optimizer_tensors_by_dtype = dict(optimizer_by_dtype)
            context.hash_zch_modules = hash_zch_modules

    @staticmethod
    def _reachable_module_ids(root: nn.Module) -> Set[int]:
        pending = [root]
        visited: Set[int] = set()
        while pending:
            module = pending.pop()
            module_id = id(module)
            if module_id in visited:
                continue
            visited.add(module_id)
            pending.extend(module.children())
            lookups = getattr(module, "_lookups", None)
            if lookups is not None:
                pending.extend(
                    lookup for lookup in lookups if isinstance(lookup, nn.Module)
                )
        return visited

    @staticmethod
    def _group_tensors_by_dtype(
        tensors: List[torch.Tensor],
    ) -> Dict[torch.dtype, List[torch.Tensor]]:
        grouped: Dict[torch.dtype, List[torch.Tensor]] = defaultdict(list)
        for tensor in tensors:
            grouped[tensor.dtype].append(tensor)
        return dict(grouped)

    @classmethod
    def _module_sync_tensors(
        cls,
        kernel: nn.Module,
        parent: nn.Module,
    ) -> Tuple[
        Dict[torch.dtype, List[torch.Tensor]],
        Dict[torch.dtype, List[torch.Tensor]],
    ]:
        sync_kernel = cast(Any, kernel)
        get_sync_weight_tensors = getattr(parent, "get_sync_weight_tensors", None)
        weights = list(
            get_sync_weight_tensors()
            if get_sync_weight_tensors is not None
            else sync_kernel.split_embedding_weights()
        )
        optimizer = [
            state["sum"]
            for state in sync_kernel.get_optimizer_state()
            if state.get("sum") is not None
        ]
        return (
            cls._group_tensors_by_dtype(weights),
            cls._group_tensors_by_dtype(optimizer),
        )

    def _build_module_sync_units(self) -> None:
        """Retain planned-module ownership for physical embedding kernels."""
        if getattr(self, "_module_sync_units_initialized", False):
            return
        claimed_kernel_ids: Set[int] = set()
        units: Dict[str, List[_DMPModuleSyncUnit]] = {}
        module_fqns_to_discover = sorted(
            self._plan.plan,
            key=lambda fqn: (-fqn.count("."), fqn),
        )
        for module_fqn in module_fqns_to_discover:
            owner = self.module.get_submodule(module_fqn)
            reachable_ids = self._reachable_module_ids(owner)
            module_units = []
            for context in self._ctxs:
                for kernel, parent in context.modules_to_sync:
                    if id(kernel) in claimed_kernel_ids or not (
                        id(kernel) in reachable_ids or id(parent) in reachable_ids
                    ):
                        continue
                    if not _is_embedding_kernel_with_sync_tensors(kernel):
                        continue
                    weights, optimizer = self._module_sync_tensors(kernel, parent)
                    module_units.append(
                        _DMPModuleSyncUnit(
                            kernel=kernel,
                            parent=parent,
                            context=context,
                            weights_by_dtype=weights,
                            optimizer_by_dtype=optimizer,
                        )
                    )
                    claimed_kernel_ids.add(id(kernel))
            if module_units:
                units[module_fqn] = module_units
        self._module_sync_units = {
            module_fqn: units[module_fqn]
            for module_fqn in self._plan.plan
            if module_fqn in units
        }
        self._module_sync_units_initialized = True

    def _ensure_module_sync_units(self) -> None:
        if not getattr(self, "_module_sync_units_initialized", False):
            self._build_module_sync_units()

    @property
    def module_sync_fqns(self) -> Tuple[str, ...]:
        """Planned module FQNs with independently syncable physical kernels."""
        return tuple(self._module_sync_units)

    def start_module_sync(
        self,
        module_fqn: str,
        include_optimizer_state: bool = True,
    ) -> DMPModuleSyncHandle:
        """Launch asynchronous replica synchronization for one planned module.

        All ranks in a replica process group must call this method for module
        FQNs in the same order. `wait_all_module_syncs` drains the outstanding
        weight and optimizer-state collectives before their next use.
        """
        self._ensure_module_sync_units()
        if self._custom_all_reduce is not None or self._all_reduce_hook is not None:
            raise RuntimeError(
                "module-granular sync does not support a custom all-reduce hook"
            )
        if module_fqn not in self._module_sync_units:
            raise KeyError(f"no syncable embedding kernels for module '{module_fqn}'")
        pending = self._pending_module_syncs.get(module_fqn)
        if pending is not None and (pending.weight_works or pending.optimizer_works):
            raise RuntimeError(f"module '{module_fqn}' already has pending sync work")

        handle = DMPModuleSyncHandle(module_fqn=module_fqn)
        restored_contexts: Set[int] = set()
        with record_function(f"## 2d_module_sync_start {module_fqn} ##"):
            for unit in self._module_sync_units[module_fqn]:
                replica_pg = unit.context.replica_pg
                assert replica_pg is not None, "replica_pg is not initialized"
                if replica_pg.size() == 1:
                    continue
                context_id = id(unit.context)
                if context_id not in restored_contexts:
                    self._restore_stashed_sync_tensors(
                        unit.context,
                        include_optimizer_state,
                    )
                    restored_contexts.add(context_id)
                weights_by_dtype = unit.weights_by_dtype
                optimizer_by_dtype = unit.optimizer_by_dtype
                if getattr(self, "_module_sync_used_row_sync_enabled", False):
                    (
                        weights_by_dtype,
                        optimizer_by_dtype,
                    ) = self._pack_module_used_rows(
                        module_fqn,
                        unit,
                        handle,
                        include_optimizer_state,
                    )
                opts = dist.AllreduceCoalescedOptions()
                opts.reduceOp = dist.ReduceOp.AVG
                for dtype in sorted(weights_by_dtype, key=str):
                    handle.weight_works.append(
                        replica_pg.allreduce_coalesced(
                            weights_by_dtype[dtype],
                            opts=opts,
                        )
                    )
                if include_optimizer_state:
                    for dtype in sorted(optimizer_by_dtype, key=str):
                        handle.optimizer_works.append(
                            replica_pg.allreduce_coalesced(
                                optimizer_by_dtype[dtype],
                                opts=opts,
                            )
                        )
        self._pending_module_syncs[module_fqn] = handle
        return handle

    def _clear_completed_module_sync(self, module_fqn: str) -> None:
        handle = self._pending_module_syncs.get(module_fqn)
        if (
            handle is not None
            and not handle.weight_works
            and not handle.optimizer_works
        ):
            del self._pending_module_syncs[module_fqn]

    def wait_module_weights(self, module_fqn: str) -> None:
        """Make the current stream wait for one module's synchronized weights."""
        handle = self._pending_module_syncs.get(module_fqn)
        if handle is None:
            return
        with record_function(f"## 2d_module_weight_wait {module_fqn} ##"):
            for work in handle.weight_works:
                work.wait()
            for packed in handle.packed_weight_rows:
                packed.target.index_copy_(0, packed.row_ids, packed.payload)
            handle.weight_works.clear()
            handle.packed_weight_rows.clear()
        self._clear_completed_module_sync(module_fqn)

    def wait_module_optimizer_state(self, module_fqn: str) -> None:
        """Wait before the module's next fused optimizer update."""
        handle = self._pending_module_syncs.get(module_fqn)
        if handle is None:
            return
        with record_function(f"## 2d_module_optimizer_wait {module_fqn} ##"):
            for work in handle.optimizer_works:
                work.wait()
            for packed in handle.packed_optimizer_rows:
                packed.target.index_copy_(0, packed.row_ids, packed.payload)
            handle.optimizer_works.clear()
            handle.packed_optimizer_rows.clear()
        self._clear_completed_module_sync(module_fqn)

    def wait_all_module_syncs(self) -> None:
        """Drain all module-granular synchronization work."""
        for module_fqn in tuple(self._pending_module_syncs):
            self.wait_module_weights(module_fqn)
            self.wait_module_optimizer_state(module_fqn)

    @staticmethod
    def _create_used_row_kernel_state(
        unit: _DMPModuleSyncUnit,
    ) -> _UsedRowKernelState:
        if not isinstance(unit.kernel, SplitTableBatchedEmbeddingBagsCodegen):
            raise ValueError(
                "used-row module sync requires SplitTableBatchedEmbeddingBagsCodegen"
            )
        kernel = unit.kernel
        weights = list(kernel.split_embedding_weights())
        optimizer_states = [
            optimizer_state.get("sum")
            for optimizer_state in kernel.get_optimizer_state()
        ]
        if len(weights) != len(optimizer_states):
            raise ValueError("TBE weights and optimizer states must align by table")
        table_offsets = [0]
        for weight in weights:
            table_offsets.append(table_offsets[-1] + weight.size(0))
        if table_offsets[-1] != kernel.total_hash_size:
            raise ValueError("TBE linear rows do not match split embedding weights")
        device = weights[0].device
        return _UsedRowKernelState(
            kernel=kernel,
            context=unit.context,
            weights=weights,
            optimizer_states=optimizer_states,
            table_offsets=table_offsets,
            table_boundaries=torch.tensor(
                table_offsets[1:-1],
                device=device,
                dtype=torch.long,
            ),
            rows_seen=torch.zeros(
                kernel.total_hash_size + 1,
                device=device,
                dtype=torch.uint8,
            ),
            count_cpu=torch.empty((), dtype=torch.long, pin_memory=True),
            prep_stream=torch.cuda.Stream(),
            ready_event=torch.cuda.Event(),
        )

    def _prepare_module_used_rows(
        self,
        module_fqn: str,
        kernel_id: int,
    ) -> None:
        used_rows_by_module = getattr(self, "_module_sync_used_row_kernels", None)
        if used_rows_by_module is None:
            raise RuntimeError("used-row module sync is not enabled")
        used_rows = used_rows_by_module[module_fqn][kernel_id]
        kernel = used_rows.kernel
        replica_pg = used_rows.context.replica_pg
        assert replica_pg is not None, "replica_pg is not initialized"
        with record_function(f"## 2d_used_rows_{module_fqn}_union ##"):
            used_rows.prep_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(used_rows.prep_stream):
                used_rows.rows_seen.zero_()
                linear_indices = torch.ops.fbgemm.linearize_cache_indices(
                    kernel.hash_size_cumsum,
                    kernel._indices,
                    kernel._offsets,
                    kernel._vbe_B_offsets,
                    kernel._vbe_max_B,
                ).clamp(max=kernel.total_hash_size)
                used_rows.rows_seen.index_fill_(0, linear_indices, 1)
                union_rows_seen = used_rows.rows_seen[: kernel.total_hash_size]
                opts = dist.AllreduceOptions()
                opts.reduceOp = dist.ReduceOp.MAX
                replica_pg.allreduce([union_rows_seen], opts=opts).wait()
                compacted_rows, count = res_bitmap_compact(union_rows_seen.bool())
                used_rows.count_cpu.copy_(count, non_blocking=True)
                used_rows.ready_event.record()
                used_rows.compacted_rows = compacted_rows

    def _pack_module_used_rows(
        self,
        module_fqn: str,
        unit: _DMPModuleSyncUnit,
        handle: DMPModuleSyncHandle,
        include_optimizer_state: bool,
    ) -> Tuple[
        Dict[torch.dtype, List[torch.Tensor]],
        Dict[torch.dtype, List[torch.Tensor]],
    ]:
        used_rows = self._module_sync_used_row_kernels[module_fqn][id(unit.kernel)]
        used_rows.ready_event.synchronize()
        count = int(used_rows.count_cpu.item())
        if used_rows.compacted_rows is None:
            raise RuntimeError("used-row union was not prepared before backward")
        rows = used_rows.compacted_rows[:count]
        split_positions = [0]
        if used_rows.table_boundaries.numel() > 0:
            split_positions.extend(
                torch.searchsorted(rows, used_rows.table_boundaries).cpu().tolist()
            )
        split_positions.append(count)

        weights_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = defaultdict(list)
        optimizer_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = defaultdict(list)
        with record_function(f"## 2d_used_rows_{module_fqn}_pack ##"):
            for table_index, (start, end) in enumerate(
                zip(split_positions, split_positions[1:])
            ):
                if start == end:
                    continue
                row_ids = rows[start:end].sub(used_rows.table_offsets[table_index])
                weight = used_rows.weights[table_index]
                weight_payload = weight.index_select(0, row_ids)
                weights_by_dtype[weight.dtype].append(weight_payload)
                handle.packed_weight_rows.append(
                    _PackedUsedRows(weight, row_ids, weight_payload)
                )
                optimizer = used_rows.optimizer_states[table_index]
                if include_optimizer_state and optimizer is not None:
                    optimizer_payload = optimizer.index_select(0, row_ids)
                    optimizer_by_dtype[optimizer.dtype].append(optimizer_payload)
                    handle.packed_optimizer_rows.append(
                        _PackedUsedRows(optimizer, row_ids, optimizer_payload)
                    )
        used_rows.compacted_rows = None
        return dict(weights_by_dtype), dict(optimizer_by_dtype)

    def enable_module_sync(
        self,
        *,
        preserve_forward_order: bool = False,
        consumer_waits: bool = True,
        used_row_sync: bool = False,
    ) -> None:
        """Launch module syncs from fused updates.

        The sharding plan provides the stable forward module order shared by
        every replica rank. The collective order follows backward execution:
        reverse plan order by default, or plan order when forward order is
        preserved. Each module starts as soon as all of its physical kernels
        have completed their fused update and every preceding module in that
        order has started.
        """
        if getattr(self, "_module_sync_hook_handles", None):
            raise RuntimeError("module sync hooks are already enabled")
        self._ensure_module_sync_units()
        if not self._module_sync_units:
            raise ValueError("DMPCollection has no syncable planned embedding modules")
        module_sync_fqns = self.module_sync_fqns
        self._module_sync_collective_order = (
            module_sync_fqns
            if preserve_forward_order
            else tuple(reversed(module_sync_fqns))
        )
        self._module_sync_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._module_sync_step_enabled = False
        self._module_sync_ready_fqns: Set[str] = set()
        self._module_sync_updated_kernels: Dict[str, Set[int]] = {}
        self._module_sync_next_index = 0
        self._module_sync_consumer_waits_enabled = consumer_waits
        self._module_sync_used_row_sync_enabled = used_row_sync
        self._module_sync_used_row_kernels: Dict[
            str, Dict[int, _UsedRowKernelState]
        ] = {}
        if used_row_sync:
            if self.device.type != "cuda":
                raise ValueError("used-row module sync requires CUDA")
            self._module_sync_used_row_kernels = {
                module_fqn: {
                    id(unit.kernel): self._create_used_row_kernel_state(unit)
                    for unit in self._module_sync_units[module_fqn]
                }
                for module_fqn in module_sync_fqns
            }

        for module_fqn in module_sync_fqns:
            for unit in self._module_sync_units[module_fqn]:
                register_post_update = getattr(
                    unit.kernel,
                    "register_post_fused_update_hook",
                    None,
                )
                if register_post_update is None:
                    raise ValueError(
                        f"kernel {type(unit.kernel).__name__} for '{module_fqn}' "
                        "does not support post-fused-update hooks"
                    )
                kernel_id = id(unit.kernel)

                def post_update(
                    fqn: str = module_fqn,
                    kernel_identity: int = kernel_id,
                ) -> None:
                    self._mark_module_kernel_updated(fqn, kernel_identity)

                self._module_sync_hook_handles.append(register_post_update(post_update))

                if self._module_sync_used_row_sync_enabled:

                    def prepare_used_rows(
                        _module: nn.Module,
                        _inputs: Any,
                        _output: Any,
                        fqn: str = module_fqn,
                        kernel_identity: int = kernel_id,
                    ) -> None:
                        self._prepare_module_used_rows(fqn, kernel_identity)

                    self._module_sync_hook_handles.append(
                        unit.kernel.register_forward_hook(prepare_used_rows)
                    )

                if not self._module_sync_consumer_waits_enabled:
                    continue

                def before_lookup(
                    _module: nn.Module,
                    _args: Tuple[Any, ...],
                    fqn: str = module_fqn,
                ) -> None:
                    self.wait_module_weights(fqn)

                self._module_sync_hook_handles.append(
                    unit.kernel.register_forward_pre_hook(before_lookup)
                )

                def before_backward(
                    _module: nn.Module,
                    _grad_output: Any,
                    fqn: str = module_fqn,
                ) -> None:
                    self.wait_module_optimizer_state(fqn)

                self._module_sync_hook_handles.append(
                    unit.kernel.register_full_backward_pre_hook(before_backward)
                )

    @property
    def module_sync_uses_consumer_waits(self) -> bool:
        return getattr(self, "_module_sync_consumer_waits_enabled", False)

    def begin_module_sync_step(self, enabled: bool) -> None:
        """Arm post-update launches for one backward pass."""
        if not getattr(self, "_module_sync_hook_handles", None):
            raise RuntimeError("module sync hooks are not enabled")
        if self._module_sync_step_enabled:
            raise RuntimeError("the previous module sync step has not finished")
        self._module_sync_step_enabled = enabled
        self._module_sync_ready_fqns.clear()
        self._module_sync_updated_kernels.clear()
        self._module_sync_next_index = 0

    def _mark_module_kernel_updated(self, module_fqn: str, kernel_id: int) -> None:
        if not self._module_sync_step_enabled:
            return
        updated = self._module_sync_updated_kernels.setdefault(module_fqn, set())
        if kernel_id in updated:
            raise RuntimeError(
                f"kernel for '{module_fqn}' completed more than once in one step"
            )
        updated.add(kernel_id)
        if len(updated) == len(self._module_sync_units[module_fqn]):
            self._module_sync_ready_fqns.add(module_fqn)
        self._drain_ready_module_syncs()

    def _drain_ready_module_syncs(self) -> None:
        """Launch the contiguous ready prefix of the collective order.

        Every replica rank must enqueue module collectives in exactly the same
        order even if its autograd hooks become ready in a different order.
        Readiness is therefore buffered in `_module_sync_ready_fqns`; it never
        directly determines which collective launches next.
        """
        while self._module_sync_next_index < len(self._module_sync_collective_order):
            next_module_fqn = self._module_sync_collective_order[
                self._module_sync_next_index
            ]
            if next_module_fqn not in self._module_sync_ready_fqns:
                return
            self.start_module_sync(next_module_fqn)
            self._module_sync_next_index += 1

    def finish_module_sync_step(self) -> None:
        """Verify that every configured module launched on this sync step."""
        if self._module_sync_step_enabled and self._module_sync_next_index != len(
            self._module_sync_collective_order
        ):
            missing = self._module_sync_collective_order[self._module_sync_next_index :]
            raise RuntimeError(f"module sync hooks did not fire for {missing}")
        self._module_sync_step_enabled = False

    def disable_module_sync(self) -> None:
        """Drain outstanding work and remove automatic module-sync hooks."""
        self.wait_all_module_syncs()
        for handle in getattr(self, "_module_sync_hook_handles", []):
            handle.remove()
        self._module_sync_hook_handles = []
        self._module_sync_consumer_waits_enabled = False
        self._module_sync_used_row_sync_enabled = False
        self._module_sync_used_row_kernels = {}
        self._module_sync_step_enabled = False

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        Returns the device mesh used for 2D parallelism.
        Contains two dimensions: "replicate" and "shard".
        """
        return self._default_ctx.device_mesh
