#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from functools import partial
from itertools import zip_longest
from typing import (
    Any,
    cast,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from tensordict import TensorDict
from torch import distributed as dist, nn, Tensor
from torch.autograd.profiler import record_function
from torch.distributed._shard.sharded_tensor import TensorProperties
from torch.distributed._tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_lookup import PartiallyMaterializedTensor
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
    Multistreamable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_IS_SSD_TABLE,
    FUSED_PARAM_SSD_TABLE_LIST,
)
from torchrec.distributed.sharding.cw_sharding import CwPooledEmbeddingSharding
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.sharding.dynamic_sharding import (
    CommP2PMetadata,
    CommStrategy,
    prepare_comm_ops,
    transfer_data,
    update_module_sharding_plan,
    update_state_dictionaries,
)
from torchrec.distributed.sharding.grid_sharding import GridPooledEmbeddingSharding
from torchrec.distributed.sharding.rw_sharding import RwPooledEmbeddingSharding
from torchrec.distributed.sharding.tw_sharding import TwPooledEmbeddingSharding
from torchrec.distributed.sharding.twcw_sharding import TwCwPooledEmbeddingSharding
from torchrec.distributed.sharding.twrw_sharding import TwRwPooledEmbeddingSharding
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import (
    Awaitable,
    EmbeddingEvent,
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    LazyAwaitable,
    LazyGetItemMixin,
    NullShardedModuleContext,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedTensor,
    ShardingEnv,
    ShardingEnv2D,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import (
    add_params_from_parameter_sharding,
    append_prefix,
    convert_to_fbgemm_types,
    create_global_tensor_shape_stride_from_metadata,
    maybe_annotate_embedding_event,
    merge_fused_params,
    none_throws,
    optimizer_type_to_emb_opt_type,
)
from torchrec.modules.embedding_configs import (
    data_type_to_dtype,
    EmbeddingBagConfig,
    EmbeddingTableConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
)
from torchrec.optim.fused import EmptyFusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import _to_offsets, KeyedJaggedTensor, KeyedTensor
from torchrec.sparse.jagged_tensor_validator import validate_keyed_jagged_tensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops")
except (OSError, RuntimeError):
    pass


logger: logging.Logger = logging.getLogger(__name__)


def _pin_and_move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (
        tensor
        if device.type == "cpu"
        else tensor.pin_memory().to(device=device, non_blocking=True)
    )


def get_device_from_parameter_sharding(
    ps: ParameterSharding,
) -> Union[str, Tuple[str, ...]]:
    """
    Returns list of device type per shard if table is sharded across different
    device type, else reutrns single device type for the table parameter
    """
    if not isinstance(ps.sharding_spec, EnumerableShardingSpec):
        raise ValueError("Expected EnumerableShardingSpec as input to the function")

    device_type_list: Tuple[str, ...] = tuple(
        # pyre-fixme[16]: `Optional` has no attribute `device`
        [shard.placement.device().type for shard in ps.sharding_spec.shards]
    )
    if len(set(device_type_list)) == 1:
        return device_type_list[0]
    else:
        assert (
            ps.sharding_type == "row_wise"
        ), "Only row_wise sharding supports sharding across multiple device types for a table"
        return device_type_list


def replace_placement_with_meta_device(
    sharding_infos: List[EmbeddingShardingInfo],
) -> None:
    """Placement device and tensor device could be unmatched in some
    scenarios, e.g. passing meta device to DMP and passing cuda
    to EmbeddingShardingPlanner. We need to make device consistent
    after getting sharding planner.
    """
    for info in sharding_infos:
        sharding_spec = info.param_sharding.sharding_spec
        if sharding_spec is None:
            continue
        if isinstance(sharding_spec, EnumerableShardingSpec):
            for shard_metadata in sharding_spec.shards:
                placement = shard_metadata.placement
                if isinstance(placement, str):
                    placement = torch.distributed._remote_device(placement)
                assert isinstance(placement, torch.distributed._remote_device)
                placement._device = torch.device("meta")
                shard_metadata.placement = placement
        else:
            # We only support EnumerableShardingSpec at present.
            raise RuntimeError(
                f"Unsupported ShardingSpec {type(sharding_spec)} with meta device"
            )


def create_sharding_infos_by_sharding_device_group(
    module: EmbeddingBagCollectionInterface,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    prefix: str,
    fused_params: Optional[Dict[str, Any]],
    suffix: Optional[str] = "weight",
) -> Dict[Tuple[str, Union[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]]:

    if fused_params is None:
        fused_params = {}

    shared_feature: Dict[str, bool] = {}
    for embedding_config in module.embedding_bag_configs():
        if not embedding_config.feature_names:
            embedding_config.feature_names = [embedding_config.name]
        for feature_name in embedding_config.feature_names:
            if feature_name not in shared_feature:
                shared_feature[feature_name] = False
            else:
                shared_feature[feature_name] = True

    sharding_type_device_group_to_sharding_infos: Dict[
        Tuple[str, Union[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]
    ] = {}

    # state_dict returns parameter.Tensor, which loses parameter level attributes
    parameter_by_name = dict(module.named_parameters())
    # QuantEBC registers weights as buffers (since they are INT8), and so we need to grab it there
    state_dict = module.state_dict()

    for config in module.embedding_bag_configs():
        table_name = config.name
        assert (
            table_name in table_name_to_parameter_sharding
        ), f"{table_name} not in table_name_to_parameter_sharding"
        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.compute_kernel not in [
            kernel.value for kernel in EmbeddingComputeKernel
        ]:
            raise ValueError(
                f"Compute kernel not supported {parameter_sharding.compute_kernel}"
            )
        embedding_names: List[str] = []
        for feature_name in config.feature_names:
            if shared_feature[feature_name]:
                embedding_names.append(feature_name + "@" + config.name)
            else:
                embedding_names.append(feature_name)

        param_name = prefix + table_name
        if suffix is not None:
            param_name = f"{param_name}.{suffix}"

        assert param_name in parameter_by_name or param_name in state_dict
        param = parameter_by_name.get(param_name, state_dict[param_name])

        # if a table name is overridden to be offloaded to ssd storage for inference
        # update the device group accordingly
        if fused_params and table_name in fused_params.get(
            FUSED_PARAM_SSD_TABLE_LIST, {}
        ):
            device_group: Union[str, Tuple[str, ...]] = "ssd"
        else:
            device_group: Union[str, Tuple[str, ...]] = (
                get_device_from_parameter_sharding(parameter_sharding)
            )

        if (
            parameter_sharding.sharding_type,
            device_group,
        ) not in sharding_type_device_group_to_sharding_infos:
            sharding_type_device_group_to_sharding_infos[
                (parameter_sharding.sharding_type, device_group)
            ] = []

        optimizer_params = getattr(param, "_optimizer_kwargs", [{}])
        optimizer_classes = getattr(param, "_optimizer_classes", [None])

        assert (
            len(optimizer_classes) == 1 and len(optimizer_params) == 1
        ), f"Only support 1 optimizer, given {len(optimizer_classes)} optimizer classes \
        and {len(optimizer_params)} optimizer kwargs."

        optimizer_class = optimizer_classes[0]
        optimizer_params = optimizer_params[0]
        if optimizer_class:
            optimizer_params["optimizer"] = optimizer_type_to_emb_opt_type(
                optimizer_class
            )

        per_table_fused_params = merge_fused_params(fused_params, optimizer_params)
        per_table_fused_params = add_params_from_parameter_sharding(
            per_table_fused_params, parameter_sharding
        )
        per_table_fused_params = convert_to_fbgemm_types(per_table_fused_params)
        if device_group == "ssd":
            per_table_fused_params.update({FUSED_PARAM_IS_SSD_TABLE: True})

        sharding_type_device_group_to_sharding_infos[
            (parameter_sharding.sharding_type, device_group)
        ].append(
            EmbeddingShardingInfo(
                embedding_config=EmbeddingTableConfig(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    name=config.name,
                    data_type=config.data_type,
                    feature_names=copy.deepcopy(config.feature_names),
                    pooling=config.pooling,
                    is_weighted=module.is_weighted(),
                    has_feature_processor=False,
                    embedding_names=embedding_names,
                    weight_init_max=config.weight_init_max,
                    weight_init_min=config.weight_init_min,
                    num_embeddings_post_pruning=(
                        getattr(config, "num_embeddings_post_pruning", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                    total_num_buckets=(
                        getattr(config, "total_num_buckets", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                    use_virtual_table=(
                        getattr(config, "use_virtual_table", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                    virtual_table_eviction_policy=(
                        getattr(config, "virtual_table_eviction_policy", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                ),
                param_sharding=parameter_sharding,
                param=param,
                fused_params=per_table_fused_params,
            )
        )
    return sharding_type_device_group_to_sharding_infos


def construct_output_kt(
    embeddings: List[torch.Tensor],
    embedding_names: List[str],
    embedding_dims: List[int],
) -> KeyedTensor:
    cat_embeddings: torch.Tensor
    if len(embeddings) == 1:
        cat_embeddings = embeddings[0]
    else:
        cat_embeddings = torch.cat(embeddings, dim=1)
    return KeyedTensor(
        keys=embedding_names,
        length_per_key=embedding_dims,
        values=cat_embeddings,
        key_dim=1,
    )


class VariableBatchEmbeddingBagCollectionAwaitable(
    LazyGetItemMixin[str, torch.Tensor], LazyAwaitable[KeyedTensor]
):
    def __init__(
        self,
        awaitables: List[Awaitable[torch.Tensor]],
        inverse_indices: Tuple[List[str], torch.Tensor],
        inverse_indices_permute_indices: Optional[torch.Tensor],
        batch_size_per_feature_pre_a2a: List[int],
        uncombined_embedding_dims: List[int],
        embedding_names: List[str],
        embedding_dims: List[int],
        permute_op: PermutePooledEmbeddings,
        module_fqn: Optional[str] = None,
        sharding_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._awaitables = awaitables
        self._inverse_indices = inverse_indices
        self._inverse_indices_permute_indices = inverse_indices_permute_indices
        self._batch_size_per_feature_pre_a2a = batch_size_per_feature_pre_a2a
        self._uncombined_embedding_dims = uncombined_embedding_dims
        self._embedding_names = embedding_names
        self._embedding_dims = embedding_dims
        self._permute_op = permute_op
        self._module_fqn = module_fqn
        self._sharding_types = sharding_types

    def _wait_impl(self) -> KeyedTensor:
        embeddings = []
        for i, w in enumerate(self._awaitables):
            with maybe_annotate_embedding_event(
                EmbeddingEvent.OUTPUT_DIST_WAIT,
                self._module_fqn,
                self._sharding_types[i] if self._sharding_types else None,
            ):
                embeddings.append(w.wait())
        batch_size = self._inverse_indices[1].numel() // len(self._inverse_indices[0])
        permute_indices = self._inverse_indices_permute_indices
        if permute_indices is not None:
            indices = torch.index_select(self._inverse_indices[1], 0, permute_indices)
        else:
            indices = self._inverse_indices[1]
        reindex_output = torch.ops.fbgemm.batch_index_select_dim0(
            inputs=embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings),
            indices=indices.view(-1),
            input_num_indices=[batch_size] * len(self._uncombined_embedding_dims),
            input_rows=self._batch_size_per_feature_pre_a2a,
            input_columns=self._uncombined_embedding_dims,
            permute_output_dim_0_1=True,
        ).view(batch_size, -1)
        return construct_output_kt(
            embeddings=[self._permute_op(reindex_output)],
            embedding_names=self._embedding_names,
            embedding_dims=self._embedding_dims,
        )


class EmbeddingBagCollectionAwaitable(
    LazyGetItemMixin[str, Tensor], LazyAwaitable[KeyedTensor]
):
    def __init__(
        self,
        awaitables: List[Awaitable[torch.Tensor]],
        embedding_dims: List[int],
        embedding_names: List[str],
        module_fqn: Optional[str] = None,
        sharding_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._awaitables = awaitables
        self._embedding_dims = embedding_dims
        self._embedding_names = embedding_names
        self._module_fqn = module_fqn
        self._sharding_types = sharding_types

    def _wait_impl(self) -> KeyedTensor:
        embeddings = []
        for i, w in enumerate(self._awaitables):
            with maybe_annotate_embedding_event(
                EmbeddingEvent.OUTPUT_DIST_WAIT,
                self._module_fqn,
                self._sharding_types[i] if self._sharding_types else None,
            ):
                embeddings.append(w.wait())

        return construct_output_kt(
            embeddings=embeddings,
            embedding_names=self._embedding_names,
            embedding_dims=self._embedding_dims,
        )


@dataclass
class EmbeddingBagCollectionContext(Multistreamable):
    sharding_contexts: List[Optional[EmbeddingShardingContext]] = field(
        default_factory=list
    )
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None
    variable_batch_per_feature: bool = False
    divisor: Optional[torch.Tensor] = None

    def record_stream(self, stream: torch.Stream) -> None:
        for ctx in self.sharding_contexts:
            if ctx:
                ctx.record_stream(stream)
        if self.inverse_indices is not None:
            self.inverse_indices[1].record_stream(stream)
        if self.divisor is not None:
            self.divisor.record_stream(stream)


class ShardedEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        KeyedTensor,
        EmbeddingBagCollectionContext,
    ],
    # TODO remove after compute_kernel X sharding decoupling
    FusedOptimizerModule,
):
    """
    Sharded implementation of EmbeddingBagCollection.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        module_fqn: Optional[str] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._module_fqn = module_fqn
        self._embedding_bag_configs: List[EmbeddingBagConfig] = (
            module.embedding_bag_configs()
        )

        self._table_names: List[str] = []
        self._pooling_type_to_rs_features: Dict[str, List[str]] = defaultdict(list)
        self._table_name_to_config: Dict[str, EmbeddingBagConfig] = {}

        for config in self._embedding_bag_configs:
            self._table_names.append(config.name)
            self._table_name_to_config[config.name] = config

            if table_name_to_parameter_sharding[config.name].sharding_type in [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.ROW_WISE.value,
            ]:
                self._pooling_type_to_rs_features[config.pooling.value].extend(
                    config.feature_names
                )

        self.module_sharding_plan: EmbeddingModuleShardingPlan = cast(
            EmbeddingModuleShardingPlan,
            {
                table_name: parameter_sharding
                for table_name, parameter_sharding in table_name_to_parameter_sharding.items()
                if table_name in self._table_names
            },
        )
        self._env = env
        # output parameters as DTensor in state dict
        self._output_dtensor: bool = env.output_dtensor
        self.sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = (
            self.create_grouped_sharding_infos(
                module,
                table_name_to_parameter_sharding,
                "embedding_bags.",
                fused_params,
            )
        )
        self._sharding_types: List[str] = list(
            self.sharding_type_to_sharding_infos.keys()
        )
        self._embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = [
            self.create_embedding_bag_sharding(
                embedding_configs,
                env,
                device,
                permute_embeddings=True,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for embedding_configs in self.sharding_type_to_sharding_infos.values()
        ]

        self._is_weighted: bool = module.is_weighted()
        self._device = device
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []
        self._uncombined_embedding_names: List[str] = []
        self._uncombined_embedding_dims: List[int] = []
        self._inverse_indices_permute_indices: Optional[torch.Tensor] = None
        # to support mean pooling callback hook
        self._has_mean_pooling_callback: bool = (
            PoolingType.MEAN.value in self._pooling_type_to_rs_features
        )
        self._dim_per_key: Optional[torch.Tensor] = None
        self._kjt_key_indices: Dict[str, int] = {}
        self._kjt_inverse_order: Optional[torch.Tensor] = None
        self._kt_key_ordering: Optional[torch.Tensor] = None
        # to support the FP16 hook
        self._create_output_dist()

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_features_permute: bool = True
        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, tbe_module in lookup.named_modules():
                if isinstance(tbe_module, FusedOptimizerModule):
                    # modify param keys to match EmbeddingBagCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in tbe_module.fused_optimizer.params.items():
                        # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`
                        params["embedding_bags." + param_key] = weight
                    tbe_module.fused_optimizer.params = params
                    optims.append(("", tbe_module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._skip_missing_weight_key: List[str] = []

        for i, (sharding, lookup) in enumerate(
            zip(self._embedding_shardings, self._lookups)
        ):
            # TODO: can move this into DpPooledEmbeddingSharding once all modules are composable
            if isinstance(sharding, DpPooledEmbeddingSharding):
                self._lookups[i] = DistributedDataParallel(
                    module=lookup,
                    device_ids=(
                        [self._device]
                        if self._device is not None
                        and (self._device.type in {"cuda", "mtia"})
                        else None
                    ),
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )

        if env.process_group and dist.get_backend(env.process_group) != "fake":
            self._initialize_torch_state()

        if module.device not in ["meta", "cpu"] and module.device.type not in [
            "meta",
            "cpu",
        ]:
            self.load_state_dict(module.state_dict(), strict=False)

    @classmethod
    def create_grouped_sharding_infos(
        cls,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        prefix: str,
        fused_params: Optional[Dict[str, Any]],
        suffix: Optional[str] = "weight",
    ) -> Dict[str, List[EmbeddingShardingInfo]]:
        """
        convert ParameterSharding (table_name_to_parameter_sharding: Dict[str, ParameterSharding]) to
        EmbeddingShardingInfo that are grouped by sharding_type, and propagate the configs/parameters
        """

        if fused_params is None:
            fused_params = {}

        shared_feature: Dict[str, bool] = {}
        for embedding_config in module.embedding_bag_configs():
            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            for feature_name in embedding_config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = (
            defaultdict(list)
        )

        # state_dict returns parameter.Tensor, which loses parameter level attributes
        parameter_by_name = dict(module.named_parameters())
        # QuantEBC registers weights as buffers (since they are INT8), and so we need to grab it there
        state_dict = module.state_dict()

        for config in module.embedding_bag_configs():
            table_name = config.name
            assert (
                table_name in table_name_to_parameter_sharding
            ), f"{table_name} not in table_name_to_parameter_sharding"
            parameter_sharding = table_name_to_parameter_sharding[table_name]
            if parameter_sharding.compute_kernel not in [
                kernel.value for kernel in EmbeddingComputeKernel
            ]:
                raise ValueError(
                    f"Compute kernel not supported {parameter_sharding.compute_kernel}"
                )
            embedding_names: List[str] = []
            for feature_name in config.feature_names:
                if shared_feature[feature_name]:
                    embedding_names.append(feature_name + "@" + config.name)
                else:
                    embedding_names.append(feature_name)

            param_name = prefix + table_name
            if suffix is not None:
                param_name = f"{param_name}.{suffix}"

            assert param_name in parameter_by_name or param_name in state_dict
            param = parameter_by_name.get(param_name, state_dict[param_name])

            optimizer_params = getattr(param, "_optimizer_kwargs", [{}])
            optimizer_classes = getattr(param, "_optimizer_classes", [None])

            assert (
                len(optimizer_classes) == 1 and len(optimizer_params) == 1
            ), f"Only support 1 optimizer, given {len(optimizer_classes)} optimizer classes \
            and {len(optimizer_params)} optimizer kwargs."

            optimizer_class = optimizer_classes[0]
            optimizer_params = optimizer_params[0]
            if optimizer_class:
                optimizer_params["optimizer"] = optimizer_type_to_emb_opt_type(
                    optimizer_class
                )

            per_table_fused_params = merge_fused_params(fused_params, optimizer_params)
            per_table_fused_params = add_params_from_parameter_sharding(
                per_table_fused_params, parameter_sharding
            )
            per_table_fused_params = convert_to_fbgemm_types(per_table_fused_params)

            sharding_info = EmbeddingShardingInfo(
                embedding_config=EmbeddingTableConfig(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    name=config.name,
                    data_type=config.data_type,
                    feature_names=copy.deepcopy(config.feature_names),
                    pooling=config.pooling,
                    is_weighted=module.is_weighted(),
                    has_feature_processor=False,
                    embedding_names=embedding_names,
                    weight_init_max=config.weight_init_max,
                    weight_init_min=config.weight_init_min,
                    num_embeddings_post_pruning=(
                        getattr(config, "num_embeddings_post_pruning", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                    total_num_buckets=(
                        getattr(config, "total_num_buckets", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                    use_virtual_table=(
                        getattr(config, "use_virtual_table", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                    virtual_table_eviction_policy=(
                        getattr(config, "virtual_table_eviction_policy", None)
                        # TODO: Need to check if attribute exists for BC
                    ),
                ),
                param_sharding=parameter_sharding,
                param=param,
                fused_params=per_table_fused_params,
            )
            sharding_type_to_sharding_infos[parameter_sharding.sharding_type].append(
                sharding_info
            )
        return sharding_type_to_sharding_infos

    @classmethod
    def create_embedding_bag_sharding(
        cls,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        """
        This is the main function to generate `EmbeddingSharding` instances based on sharding_type
        so that the same sharding_type in one EBC would be fused.
        """
        sharding_type = sharding_infos[0].param_sharding.sharding_type

        if device is not None and device.type == "meta":
            replace_placement_with_meta_device(sharding_infos)
        if sharding_type == ShardingType.TABLE_WISE.value:
            return TwPooledEmbeddingSharding(
                sharding_infos,
                env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            return RwPooledEmbeddingSharding(
                sharding_infos,
                env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.DATA_PARALLEL.value:
            return DpPooledEmbeddingSharding(sharding_infos, env, device)
        elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
            return TwRwPooledEmbeddingSharding(
                sharding_infos,
                env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.COLUMN_WISE.value:
            return CwPooledEmbeddingSharding(
                sharding_infos,
                env,
                device,
                permute_embeddings=permute_embeddings,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
            return TwCwPooledEmbeddingSharding(
                sharding_infos,
                env,
                device,
                permute_embeddings=permute_embeddings,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.GRID_SHARD.value:
            return GridPooledEmbeddingSharding(
                sharding_infos,
                env,
                device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            raise ValueError(f"Sharding type not supported {sharding_type}")

    @staticmethod
    def _pre_state_dict_hook(
        self: "ShardedEmbeddingBagCollection",
        prefix: str = "",
        keep_vars: bool = False,
    ) -> None:
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.flush()

    @staticmethod
    def _pre_load_state_dict_hook(
        self: "ShardedEmbeddingBagCollection",
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        Modify the destination state_dict for model parallel
        to transform from ShardedTensors/DTensors into tensors
        """
        for table_name in self._model_parallel_name_to_local_shards.keys():
            if self._table_name_to_config[table_name].use_virtual_table:
                # weight_id and bucket are generated at the runtime of state_dict instead of registered class
                # so we need to erase them before passing into load_state_dict
                weight_key = f"{prefix}embedding_bags.{table_name}.weight"
                weight_id_key = f"{prefix}embedding_bags.{table_name}.weight_id"
                bucket_key = f"{prefix}embedding_bags.{table_name}.bucket"
                metadata_key = f"{prefix}embedding_bags.{table_name}.metadata"
                if weight_id_key in state_dict:
                    del state_dict[weight_id_key]
                if bucket_key in state_dict:
                    del state_dict[bucket_key]
                if metadata_key in state_dict:
                    del state_dict[metadata_key]
                assert weight_key in state_dict
                assert (
                    len(self._model_parallel_name_to_local_shards[table_name]) == 1
                ), "currently only support 1 shard per rank"

                # for loading state_dict into virtual table, we skip the weights assignment
                # if needed, for now this should be handled separately outside of load_state_dict call
                self._skip_missing_weight_key.append(weight_key)
                del state_dict[weight_key]
                continue

            key = f"{prefix}embedding_bags.{table_name}.weight"

            # If key not in state dict, continue
            if key not in state_dict:
                continue

            # gather model shards from both DTensor and ShardedTensor maps
            model_shards_sharded_tensor = self._model_parallel_name_to_local_shards[
                table_name
            ]
            model_shards_dtensor = self._model_parallel_name_to_shards_wrapper[
                table_name
            ]
            # If state_dict[key] is already a ShardedTensor, use its local shards
            if isinstance(state_dict[key], ShardedTensor):
                local_shards = state_dict[key].local_shards()
                if len(local_shards) == 0:
                    state_dict[key] = torch.empty(0)
                else:
                    dim = state_dict[key].metadata().shards_metadata[0].shard_sizes[1]
                    # CW multiple shards are merged
                    if len(local_shards) > 1:
                        state_dict[key] = torch.cat(
                            [s.tensor.view(-1) for s in local_shards], dim=0
                        ).view(-1, dim)
                    else:
                        state_dict[key] = local_shards[0].tensor.view(-1, dim)
            elif isinstance(state_dict[key], DTensor):
                shards_wrapper = state_dict[key].to_local()
                local_shards = shards_wrapper.local_shards()
                if len(local_shards) == 0:
                    state_dict[key] = torch.empty(0)
                else:
                    dim = shards_wrapper.local_sizes()[0][1]
                    # CW multiple shards are merged
                    if len(local_shards) > 1:
                        state_dict[key] = torch.cat(
                            [s.view(-1) for s in local_shards], dim=0
                        ).view(-1, dim)
                    else:
                        state_dict[key] = local_shards[0].view(-1, dim)
            elif isinstance(state_dict[key], torch.Tensor):
                local_shards = []
                if model_shards_sharded_tensor:
                    # splice according to sharded tensor metadata
                    for shard in model_shards_sharded_tensor:
                        # Extract shard size and offsets for splicing
                        shard_size = shard.metadata.shard_sizes
                        shard_offset = shard.metadata.shard_offsets

                        # Prepare tensor by splicing and placing on appropriate device
                        spliced_tensor = state_dict[key][
                            shard_offset[0] : shard_offset[0] + shard_size[0],
                            shard_offset[1] : shard_offset[1] + shard_size[1],
                        ]

                        # Append spliced tensor into local shards
                        local_shards.append(spliced_tensor)
                elif model_shards_dtensor:
                    # splice according to dtensor metadata
                    for tensor, shard_offset in zip(
                        model_shards_dtensor["local_tensors"],
                        model_shards_dtensor["local_offsets"],
                    ):
                        shard_size = tensor.size()
                        spliced_tensor = state_dict[key][
                            shard_offset[0] : shard_offset[0] + shard_size[0],
                            shard_offset[1] : shard_offset[1] + shard_size[1],
                        ]
                        local_shards.append(spliced_tensor)
                state_dict[key] = (
                    torch.empty(0)
                    if not local_shards
                    else torch.cat(local_shards, dim=0)
                )
            else:
                raise RuntimeError(
                    f"Unexpected state_dict key type {type(state_dict[key])} found for {key}"
                )

        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.purge()

    def _initialize_torch_state(self, skip_registering: bool = False) -> None:  # noqa
        """
        This provides consistency between this class and the EmbeddingBagCollection's
        nn.Module API calls (state_dict, named_modules, etc)

        Args:
            skip_registering (bool): If True, skips registering state_dict hooks. This is useful
                for dynamic sharding where the state_dict hooks do not need to be
                reregistered when being resharded. Default is False.

        """
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for table_name in self._table_names:
            self.embedding_bags[table_name] = nn.Module()

        self._model_parallel_name_to_local_shards = OrderedDict()
        self._model_parallel_name_to_shards_wrapper = OrderedDict()
        self._model_parallel_name_to_sharded_tensor = OrderedDict()
        self._model_parallel_name_to_dtensor = OrderedDict()

        _model_parallel_name_to_compute_kernel: Dict[str, str] = {}
        for (
            table_name,
            parameter_sharding,
        ) in self.module_sharding_plan.items():
            if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            _model_parallel_name_to_compute_kernel[table_name] = (
                parameter_sharding.compute_kernel
            )
            if (
                parameter_sharding.compute_kernel
                == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
            ):
                # Skip state_dict handling for CUSTOMIZED_KERNEL, this should be implemented
                # in child class for the CUSTOMIZED_KERNEL
                continue
            self._model_parallel_name_to_local_shards[table_name] = []
            self._model_parallel_name_to_shards_wrapper[table_name] = OrderedDict(
                [("local_tensors", []), ("local_offsets", [])]
            )

        self._name_to_table_size = {}
        for table in self._embedding_bag_configs:
            self._name_to_table_size[table.name] = (
                table.num_embeddings,
                table.embedding_dim,
            )

        for lookup, sharding in zip(self._lookups, self._embedding_shardings):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                # unwrap DDP
                lookup = lookup.module
            else:
                # save local_shards for transforming MP params to DTensor
                for key, v in lookup.state_dict().items():
                    table_name = key[: -len(".weight")]
                    if (
                        _model_parallel_name_to_compute_kernel[table_name]
                        == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
                    ):
                        continue
                    if isinstance(v, DTensor):
                        shards_wrapper = self._model_parallel_name_to_shards_wrapper[
                            table_name
                        ]
                        local_shards_wrapper = v._local_tensor
                        shards_wrapper["local_tensors"].extend(
                            # pyre-ignore[16]
                            local_shards_wrapper.local_shards()
                        )
                        shards_wrapper["local_offsets"].extend(
                            # pyre-ignore[16]
                            local_shards_wrapper.local_offsets()
                        )
                        shards_wrapper["global_size"] = v.size()
                        shards_wrapper["global_stride"] = v.stride()
                        shards_wrapper["placements"] = v.placements
                    elif isinstance(v, ShardedTensor):
                        # for virtual table, we only populate the shardedTensor for Embedding Table during
                        # initial state_dict calls, skip weight id and bucket tensor
                        self._model_parallel_name_to_local_shards[table_name].extend(
                            v.local_shards()
                        )
            for (
                table_name,
                tbe_slice,
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
                #  `named_parameters_by_table`.
            ) in lookup.named_parameters_by_table():
                # for virtual table, currently we don't expose id tensor and bucket tensor
                # because they are not updated in real time, and they are created on the fly
                # whenever state_dict is called
                # reference: ƒbgs _gen_named_parameters_by_table_ssd_pmt
                self.embedding_bags[table_name].register_parameter("weight", tbe_slice)

        for table_name in self._model_parallel_name_to_local_shards.keys():
            local_shards = self._model_parallel_name_to_local_shards[table_name]
            shards_wrapper_map = self._model_parallel_name_to_shards_wrapper[table_name]
            # for shards that don't exist on this rank, register with empty tensor
            if not hasattr(self.embedding_bags[table_name], "weight"):
                self.embedding_bags[table_name].register_parameter(
                    "weight", nn.Parameter(torch.empty(0))
                )
                if (
                    _model_parallel_name_to_compute_kernel[table_name]
                    != EmbeddingComputeKernel.DENSE.value
                ):
                    # pyre-fixme[16]: `Module` has no attribute
                    #  `_in_backward_optimizers`.
                    # pyre-fixme[16]: `Tensor` has no attribute
                    #  `_in_backward_optimizers`.
                    self.embedding_bags[table_name].weight._in_backward_optimizers = [
                        EmptyFusedOptimizer()
                    ]

            if self._output_dtensor:
                assert _model_parallel_name_to_compute_kernel[table_name] not in {
                    EmbeddingComputeKernel.KEY_VALUE.value
                }
                if shards_wrapper_map["local_tensors"]:
                    self._model_parallel_name_to_dtensor[table_name] = (
                        DTensor.from_local(
                            local_tensor=LocalShardsWrapper(
                                local_shards=shards_wrapper_map["local_tensors"],
                                local_offsets=shards_wrapper_map["local_offsets"],
                            ),
                            device_mesh=self._env.device_mesh,
                            placements=shards_wrapper_map["placements"],
                            shape=shards_wrapper_map["global_size"],
                            stride=shards_wrapper_map["global_stride"],
                            run_check=False,
                        )
                    )
                else:
                    shape, stride = create_global_tensor_shape_stride_from_metadata(
                        none_throws(self.module_sharding_plan[table_name]),
                        (
                            self._env.node_group_size
                            if isinstance(self._env, ShardingEnv2D)
                            else get_local_size(self._env.world_size)
                        ),
                    )
                    # empty shard case
                    self._model_parallel_name_to_dtensor[table_name] = (
                        DTensor.from_local(
                            local_tensor=LocalShardsWrapper(
                                local_shards=[],
                                local_offsets=[],
                            ),
                            device_mesh=self._env.device_mesh,
                            run_check=False,
                            shape=shape,
                            stride=stride,
                        )
                    )
            else:
                # created ShardedTensors once in init, use in post_state_dict_hook
                # note: at this point kvstore backed tensors don't own valid snapshots, so no read
                # access is allowed on them.

                if self._table_name_to_config[table_name].use_virtual_table:
                    # virtual table size will be recalculated before checkpointing. Here we cannot
                    # use sharding spec to build tensor metadata which will exceed the checkpoint capacity limit
                    self._model_parallel_name_to_sharded_tensor[table_name] = (
                        ShardedTensor._init_from_local_shards(
                            local_shards,
                            (
                                [
                                    # assuming virtual table only supports rw sharding for now
                                    # When backend return whole row, need to respect dim(1)
                                    # otherwise will see shard dim exceeded tensor dim error
                                    (
                                        0
                                        if dim == 0
                                        else (
                                            local_shards[0].metadata.shard_sizes[1]
                                            if dim == 1
                                            else dim_size
                                        )
                                    )
                                    for dim, dim_size in enumerate(
                                        self._name_to_table_size[table_name]
                                    )
                                ]
                            ),
                            process_group=(
                                self._env.sharding_pg
                                if isinstance(self._env, ShardingEnv2D)
                                else self._env.process_group
                            ),
                        )
                    )
                else:
                    # create ShardedTensor from local shards and metadata avoding all_gather collective
                    sharding_spec = none_throws(
                        self.module_sharding_plan[table_name].sharding_spec
                    )

                    tensor_properties = TensorProperties(
                        dtype=(
                            data_type_to_dtype(
                                self._table_name_to_config[table_name].data_type
                            )
                        ),
                    )

                    self._model_parallel_name_to_sharded_tensor[table_name] = (
                        ShardedTensor._init_from_local_shards_and_global_metadata(
                            local_shards=local_shards,
                            sharded_tensor_metadata=sharding_spec.build_metadata(
                                tensor_sizes=self._name_to_table_size[table_name],
                                tensor_properties=tensor_properties,
                            ),
                            process_group=(
                                self._env.sharding_pg
                                if isinstance(self._env, ShardingEnv2D)
                                else self._env.process_group
                            ),
                        )
                    )

        def extract_sharded_kvtensors(
            module: ShardedEmbeddingBagCollection,
        ) -> OrderedDict[str, ShardedTensor]:
            # retrieve all kvstore backed tensors
            ret = OrderedDict()
            for (
                table_name,
                sharded_t,
            ) in module._model_parallel_name_to_sharded_tensor.items():
                if _model_parallel_name_to_compute_kernel[table_name] in {
                    EmbeddingComputeKernel.KEY_VALUE.value,
                    EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
                    EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
                }:
                    ret[table_name] = sharded_t
            return ret

        def post_state_dict_hook(
            module: ShardedEmbeddingBagCollection,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            # Adjust dense MP
            for (
                table_name,
                sharded_t,
            ) in module._model_parallel_name_to_sharded_tensor.items():
                destination_key = f"{prefix}embedding_bags.{table_name}.weight"
                destination[destination_key] = sharded_t
            for (
                table_name,
                d_tensor,
            ) in module._model_parallel_name_to_dtensor.items():
                destination_key = f"{prefix}embedding_bags.{table_name}.weight"
                destination[destination_key] = d_tensor

            # kvstore backed tensors do not have a valid backing snapshot at this point. Fill in a valid
            # snapshot for read access.
            sharded_kvtensors = extract_sharded_kvtensors(module)
            if len(sharded_kvtensors) == 0:
                return

            sharded_kvtensors_copy = copy.deepcopy(sharded_kvtensors)
            virtual_table_sharded_t_map: Optional[
                Dict[str, Tuple[ShardedTensor, ShardedTensor]]
            ] = None
            for lookup, sharding in zip(module._lookups, module._embedding_shardings):
                if not isinstance(sharding, DpPooledEmbeddingSharding):
                    for (
                        table_name,
                        weights_t,
                        weight_ids_sharded_t,
                        id_cnt_per_bucket_sharded_t,
                        metadata_sharded_t,
                    ) in (
                        lookup.get_named_split_embedding_weights_snapshot()  # pyre-ignore
                    ):
                        assert table_name in sharded_kvtensors_copy
                        if self._table_name_to_config[table_name].use_virtual_table:
                            assert isinstance(weights_t, ShardedTensor)
                            if virtual_table_sharded_t_map is None:
                                virtual_table_sharded_t_map = {}
                            assert (
                                weight_ids_sharded_t is not None
                                and id_cnt_per_bucket_sharded_t is not None
                            )
                            # The logic here assumes there is only one shard per table on any particular rank
                            # if there are cases each rank has >1 shards, we need to update here accordingly
                            sharded_kvtensors_copy[table_name] = weights_t
                            virtual_table_sharded_t_map[table_name] = (
                                weight_ids_sharded_t,
                                id_cnt_per_bucket_sharded_t,
                                metadata_sharded_t,
                            )
                        else:
                            assert isinstance(weights_t, PartiallyMaterializedTensor)
                            assert (
                                weight_ids_sharded_t is None
                                and id_cnt_per_bucket_sharded_t is None
                                and metadata_sharded_t is None
                            )
                            # The logic here assumes there is only one shard per table on any particular rank
                            # if there are cases each rank has >1 shards, we need to update here accordingly
                            # pyre-ignore
                            sharded_kvtensors_copy[table_name].local_shards()[
                                0
                            ].tensor = weights_t

            def update_destination(
                table_name: str,
                tensor_name: str,
                destination: Dict[str, torch.Tensor],
                value: torch.Tensor,
            ) -> None:
                destination_key = f"{prefix}embedding_bags.{table_name}.{tensor_name}"
                destination[destination_key] = value

            for (
                table_name,
                sharded_kvtensor,
            ) in sharded_kvtensors_copy.items():
                update_destination(table_name, "weight", destination, sharded_kvtensor)
                if (
                    virtual_table_sharded_t_map
                    and table_name in virtual_table_sharded_t_map
                ):
                    update_destination(
                        table_name,
                        "weight_id",
                        destination,
                        virtual_table_sharded_t_map[table_name][0],
                    )
                    update_destination(
                        table_name,
                        "bucket",
                        destination,
                        virtual_table_sharded_t_map[table_name][1],
                    )
                    if virtual_table_sharded_t_map[table_name][2] is not None:
                        update_destination(
                            table_name,
                            "metadata",
                            destination,
                            virtual_table_sharded_t_map[table_name][2],
                        )

        def _post_load_state_dict_hook(
            module: "ShardedEmbeddingBagCollection",
            incompatible_keys: _IncompatibleKeys,
        ) -> None:
            if incompatible_keys.missing_keys:
                # has to remove the key inplace
                for skip_key in module._skip_missing_weight_key:
                    if skip_key in incompatible_keys.missing_keys:
                        incompatible_keys.missing_keys.remove(skip_key)

        if not skip_registering:
            self.register_state_dict_pre_hook(self._pre_state_dict_hook)
            self._register_state_dict_hook(post_state_dict_hook)
            self._register_load_state_dict_pre_hook(
                self._pre_load_state_dict_hook, with_module=True
            )
            self.register_load_state_dict_post_hook(_post_load_state_dict_hook)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._device and self._device.type == "meta":
            return

        # Initialize embedding bags weights with init_fn
        for table_config in self._embedding_bag_configs:
            if self.module_sharding_plan[table_config.name].compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE.value,
                EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
                EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
            }:
                continue
            assert table_config.init_fn is not None
            param = self.embedding_bags[f"{table_config.name}"].weight
            if param.data.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                tmp_param = torch.zeros(param.shape, device=param.device)  # pyre-ignore
                table_config.init_fn(tmp_param).to(param.data.dtype)  # pyre-ignore
                param.data.copy_(tmp_param)  # pyre-ignore
            else:
                # pyre-ignore
                table_config.init_fn(param)

            sharding_type = self.module_sharding_plan[table_config.name].sharding_type
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                pg = self._env.process_group
                with torch.no_grad():
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Module, Tensor]`.
                    dist.broadcast(param.data, src=0, group=pg)

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._embedding_shardings:
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))

        if feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=self._device, dtype=torch.int32
                ),
                persistent=False,
            )

    def _init_mean_pooling_callback(
        self,
        input_feature_names: List[str],
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]],
    ) -> None:
        # account for shared features
        feature_names: List[str] = [
            feature_name
            for sharding in self._embedding_shardings
            for feature_name in sharding.feature_names()
        ]

        for i, key in enumerate(feature_names):
            if key not in self._kjt_key_indices:  # index of first occurence
                self._kjt_key_indices[key] = i

        keyed_tensor_ordering = []
        for key in self._embedding_names:
            if "@" in key:
                key = key.split("@")[0]
            keyed_tensor_ordering.append(self._kjt_key_indices[key])
        self._kt_key_ordering = torch.tensor(keyed_tensor_ordering, device=self._device)

        if inverse_indices:
            key_to_inverse_index = {
                name: i for i, name in enumerate(inverse_indices[0])
            }
            self._kjt_inverse_order = torch.tensor(
                [key_to_inverse_index[key] for key in feature_names],
                device=self._device,
            )

    def _softcopy_lookups(self) -> List[nn.Module]:
        old_modules: List[nn.Module] = [lookup for lookup in self._lookups]
        self._lookups.clear()
        return old_modules

    def _create_lookups(
        self,
    ) -> None:
        for sharding in self._embedding_shardings:
            self._lookups.append(sharding.create_lookup())

    def _create_output_dist(self) -> None:
        embedding_shard_metadata: List[Optional[ShardMetadata]] = []
        for sharding in self._embedding_shardings:
            self._output_dists.append(sharding.create_output_dist(device=self._device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())
            self._uncombined_embedding_names.extend(
                sharding.uncombined_embedding_names()
            )
            self._uncombined_embedding_dims.extend(sharding.uncombined_embedding_dims())
            embedding_shard_metadata.extend(sharding.embedding_shard_metadata())
        self._dim_per_key = torch.tensor(self._embedding_dims, device=self._device)

        embedding_shard_offsets: List[int] = [
            meta.shard_offsets[1] if meta is not None else 0
            for meta in embedding_shard_metadata
        ]
        embedding_name_order: Dict[str, int] = {}
        for i, name in enumerate(self._uncombined_embedding_names):
            embedding_name_order.setdefault(name, i)

        permute_indices = sorted(
            range(len(self._uncombined_embedding_names)),
            key=lambda i: (
                embedding_name_order[self._uncombined_embedding_names[i]],
                embedding_shard_offsets[i],
            ),
        )

        self._permute_op: PermutePooledEmbeddings = PermutePooledEmbeddings(
            self._uncombined_embedding_dims, permute_indices, self._device
        )

    def _update_output_dist(self) -> None:
        """
        This function is only used in update update_shards
        """
        embedding_shard_metadata: List[Optional[ShardMetadata]] = []
        # TODO: Optimize to only go through embedding shardings with new ranks
        self._output_dists: List[nn.Module] = []
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._uncombined_embedding_names: List[str] = []
        self._uncombined_embedding_dims: List[int] = []
        for sharding in self._embedding_shardings:
            # TODO: if sharding type of table completely changes, need to regenerate everything
            self._embedding_names.extend(sharding.embedding_names())
            self._output_dists.append(sharding.create_output_dist(device=self._device))
            embedding_shard_metadata.extend(sharding.embedding_shard_metadata())
            self._embedding_dims.extend(sharding.embedding_dims())
            self._uncombined_embedding_names.extend(
                sharding.uncombined_embedding_names()
            )
            self._uncombined_embedding_dims.extend(sharding.uncombined_embedding_dims())

        embedding_shard_offsets: List[int] = [
            meta.shard_offsets[1] if meta is not None else 0
            for meta in embedding_shard_metadata
        ]
        embedding_name_order: Dict[str, int] = {}
        for i, name in enumerate(self._uncombined_embedding_names):
            embedding_name_order.setdefault(name, i)

        permute_indices = sorted(
            range(len(self._uncombined_embedding_names)),
            key=lambda i: (
                embedding_name_order[self._uncombined_embedding_names[i]],
                embedding_shard_offsets[i],
            ),
        )

        self._permute_op: PermutePooledEmbeddings = PermutePooledEmbeddings(
            self._uncombined_embedding_dims, permute_indices, self._device
        )

    def _create_inverse_indices_permute_indices(
        self, inverse_indices: Optional[Tuple[List[str], torch.Tensor]]
    ) -> None:
        assert (
            inverse_indices is not None
        ), "inverse indices must be provided from KJT if using variable batch size per feature."
        index_per_name = {name: i for i, name in enumerate(inverse_indices[0])}
        permute_indices = [
            index_per_name[name.split("@")[0]]
            for name in self._uncombined_embedding_names
        ]
        if len(permute_indices) != len(index_per_name) or permute_indices != sorted(
            permute_indices
        ):
            self._inverse_indices_permute_indices = _pin_and_move(
                torch.tensor(permute_indices),
                inverse_indices[1].device,
            )

    def _is_optimizer_enabled(
        self,
        has_local_optimizer: bool,
        env: ShardingEnv,
        device: Optional[torch.device],
    ) -> bool:
        flag = torch.tensor(
            [has_local_optimizer], dtype=torch.uint8, device=device
        )  # example: True
        # Reduce with MAX to check if any process has True
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=env.process_group)
        return bool(flag.item())

    # pyre-ignore [14]
    def input_dist(
        self,
        ctx: EmbeddingBagCollectionContext,
        features: Union[KeyedJaggedTensor, TensorDict],
    ) -> Awaitable[Awaitable[KJTList]]:
        """
        This is the main API called in train_pipeline where we want to do the input_dist
        in advance
        """
        if isinstance(features, TensorDict):
            feature_keys = list(features.keys())  # pyre-ignore[6]
            if len(self._features_order) > 0:
                feature_keys = [feature_keys[i] for i in self._features_order]
                self._has_features_permute = False  # feature_keys are in order
            features = maybe_td_to_kjt(features, feature_keys)  # pyre-ignore[6]
        ctx.variable_batch_per_feature = features.variable_stride_per_key()
        ctx.inverse_indices = features.inverse_indices_or_none()

        if self._has_uninitialized_input_dist:
            if torch._utils_internal.justknobs_check(
                "pytorch/torchrec:enable_kjt_validation"
            ):
                logger.info("Validating input features...")
                validate_keyed_jagged_tensor(features)

            self._create_input_dist(features.keys())
            self._has_uninitialized_input_dist = False
            if ctx.variable_batch_per_feature:
                self._create_inverse_indices_permute_indices(ctx.inverse_indices)
            if self._has_mean_pooling_callback:
                self._init_mean_pooling_callback(features.keys(), ctx.inverse_indices)

        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    # pyre-fixme[6]: For 2nd argument expected `Optional[Tensor]`
                    #  but got `Union[Module, Tensor]`.
                    self._features_order_tensor,
                )
            if self._has_mean_pooling_callback:
                ctx.divisor = _create_mean_pooling_divisor(
                    lengths=features.lengths(),
                    stride=features.stride(),
                    keys=features.keys(),
                    offsets=features.offsets(),
                    pooling_type_to_rs_features=self._pooling_type_to_rs_features,
                    stride_per_key=features.stride_per_key(),
                    dim_per_key=self._dim_per_key,  # pyre-ignore[6]
                    embedding_names=self._embedding_names,
                    embedding_dims=self._embedding_dims,
                    variable_batch_per_feature=ctx.variable_batch_per_feature,
                    kjt_inverse_order=self._kjt_inverse_order,  # pyre-ignore[6]
                    kjt_key_indices=self._kjt_key_indices,
                    kt_key_ordering=self._kt_key_ordering,  # pyre-ignore[6]
                    inverse_indices=ctx.inverse_indices,
                    weights=features.weights_or_none(),
                )

            features_by_shards = features.split(
                self._feature_splits,
            )
            awaitables = []
            for input_dist, features_by_shard, sharding_type in zip(
                self._input_dists,
                features_by_shards,
                self._sharding_types,
            ):
                with maybe_annotate_embedding_event(
                    EmbeddingEvent.KJT_SPLITS_DIST,
                    self._module_fqn,
                    sharding_type,
                ):
                    awaitables.append(input_dist(features_by_shard))

                ctx.sharding_contexts.append(
                    EmbeddingShardingContext(
                        batch_size_per_feature_pre_a2a=features_by_shard.stride_per_key(),
                        variable_batch_per_feature=features_by_shard.variable_stride_per_key(),
                    )
                )
            return KJTListSplitsAwaitable(
                awaitables, ctx, self._module_fqn, self._sharding_types
            )

    def compute(
        self,
        ctx: EmbeddingBagCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        """
        this function is not used in general practice, it's only called by the base class
        ShardedModule.compute_and_output_dist to do the basic function
        """
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self,
        ctx: EmbeddingBagCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:
        batch_size_per_feature_pre_a2a = []
        awaitables = []
        for dist, sharding_context, embeddings in zip(
            self._output_dists,
            ctx.sharding_contexts,
            output,
        ):
            awaitables.append(dist(embeddings, sharding_context))
            if sharding_context:
                batch_size_per_feature_pre_a2a.extend(
                    sharding_context.batch_size_per_feature_pre_a2a
                )

        if ctx.variable_batch_per_feature:
            assert (
                ctx.inverse_indices is not None
            ), "inverse indices must be provided from KJT if using variable batch size per feature."
            awaitable = VariableBatchEmbeddingBagCollectionAwaitable(
                awaitables=awaitables,
                inverse_indices=ctx.inverse_indices,
                inverse_indices_permute_indices=self._inverse_indices_permute_indices,
                batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
                uncombined_embedding_dims=self._uncombined_embedding_dims,
                embedding_names=self._embedding_names,
                embedding_dims=self._embedding_dims,
                permute_op=self._permute_op,
            )
        else:
            awaitable = EmbeddingBagCollectionAwaitable(
                awaitables=awaitables,
                embedding_dims=self._embedding_dims,
                embedding_names=self._embedding_names,
            )

        # register callback if there are features that need mean pooling
        if self._has_mean_pooling_callback:
            awaitable.callbacks.append(
                partial(_apply_mean_pooling, divisor=ctx.divisor)
            )

        return awaitable

    def compute_and_output_dist(
        self, ctx: EmbeddingBagCollectionContext, input: KJTList
    ) -> LazyAwaitable[KeyedTensor]:
        """
        the main API called in PipelineForward, where the shardedEBC's forward is swapped
        see _rewrite_model in train_pipeline for details
        """
        batch_size_per_feature_pre_a2a = []
        awaitables = []

        # No usage of zip for dynamo
        for i in range(len(self._lookups)):
            lookup = self._lookups[i]
            dist = self._output_dists[i]
            sharding_context = ctx.sharding_contexts[i]
            features = input[i]
            sharding_type = self._sharding_types[i]

            with maybe_annotate_embedding_event(
                EmbeddingEvent.LOOKUP,
                self._module_fqn,
                sharding_type,
            ):
                embs = lookup(features)
                if self.post_lookup_tracker_fn is not None:
                    self.post_lookup_tracker_fn(features, embs)

            with maybe_annotate_embedding_event(
                EmbeddingEvent.OUTPUT_DIST,
                self._module_fqn,
                sharding_type,
            ):
                awaitables.append(dist(embs, sharding_context))
                if self.post_odist_tracker_fn is not None:
                    self.post_odist_tracker_fn()

            if sharding_context:
                batch_size_per_feature_pre_a2a.extend(
                    sharding_context.batch_size_per_feature_pre_a2a
                )

        if ctx.variable_batch_per_feature:
            assert (
                ctx.inverse_indices is not None
            ), "inverse indices must be provided from KJT if using variable batch size per feature."
            awaitable = VariableBatchEmbeddingBagCollectionAwaitable(
                awaitables=awaitables,
                inverse_indices=ctx.inverse_indices,
                inverse_indices_permute_indices=self._inverse_indices_permute_indices,
                batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
                uncombined_embedding_dims=self._uncombined_embedding_dims,
                embedding_names=self._embedding_names,
                embedding_dims=self._embedding_dims,
                permute_op=self._permute_op,
                module_fqn=self._module_fqn,
                sharding_types=self._sharding_types,
            )
        else:
            awaitable = EmbeddingBagCollectionAwaitable(
                awaitables=awaitables,
                embedding_dims=self._embedding_dims,
                embedding_names=self._embedding_names,
                module_fqn=self._module_fqn,
                sharding_types=self._sharding_types,
            )

        # register callback if there are features that need mean pooling
        if self._has_mean_pooling_callback:
            awaitable.callbacks.append(
                partial(_apply_mean_pooling, divisor=ctx.divisor)
            )

        return awaitable

    def update_shards(
        self,
        changed_sharding_params: EmbeddingModuleShardingPlan,  # NOTE: only delta
        env: ShardingEnv,
        device: Optional[torch.device],
    ) -> None:
        """
        Updates the sharded embedding module in place based on the changed_sharding_params,
        which contains the new ParameterSharding with different shard placements.

        This method handles resharding of embedding tables, optimizer state transfer,
        and updates the internal lookup and distribution modules to reflect the new sharding.

        Args:
            changed_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping
                table names to their new parameter sharding configs. This should only
                contain shards/table names that need to be moved.
            env (ShardingEnv): The sharding environment.
            device (Optional[torch.device]): The device to place the updated module on.

        Returns:
            None
        Raises:
            RuntimeError: If DTensor output is enabled, as resharding is not yet supported for DTensor.
        """
        if env.output_dtensor:
            raise RuntimeError("We do not yet support DTensor for resharding yet")
            return

        current_state = self.state_dict()

        # Check if local optimizer state exists and is non-empty for all optimizers.
        has_local_optimizer = len(self._optim._optims) > 0 and all(
            len(i) > 0 for i in self._optim.state_dict()["state"].values()
        )

        # Communicate optimizer state across all ranks to ensure consistency.
        has_optimizer = self._is_optimizer_enabled(has_local_optimizer, env, device)

        # Save old lookup modules for cleanup.
        old_lookups: List[nn.Module] = self._softcopy_lookups()

        # Save old optimizer state if present.
        old_optimizer_state = self._optim.state_dict() if has_local_optimizer else None

        assert hasattr(self, "module_sharding_plan")
        current_module_sharding_plan = copy.deepcopy(self.module_sharding_plan)

        # Update the module sharding plan with the changed sharding parameters.
        update_module_sharding_plan(
            self, changed_sharding_params, self.sharding_type_to_sharding_infos
        )

        self._sharding_types: List[str] = list(
            self.sharding_type_to_sharding_infos.keys()
        )
        # TODO: Optimize to update only the changed embedding shardings

        # Recreate embedding sharding modules based on the new sharding infos.
        self._embedding_shardings: List[
            EmbeddingSharding[
                EmbeddingShardingContext,
                KeyedJaggedTensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = [
            self.create_embedding_bag_sharding(
                embedding_configs,
                env,
                device,
                permute_embeddings=True,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for embedding_configs in self.sharding_type_to_sharding_infos.values()
        ]

        # Reset input distribution and feature ordering.
        self._has_uninitialized_input_dist = True
        self._input_dists: List[nn.Module] = []
        self._features_order: List[int] = []
        self._feature_splits: List[int] = []

        self._create_lookups()
        self._update_output_dist()

        # Re-initialize torch state if in a distributed environment.
        if env.process_group and dist.get_backend(env.process_group) != "fake":
            self._initialize_torch_state(skip_registering=True)

        # Update optimizer to reflect new parameters.
        optims = []
        for lookup in self._lookups:
            for _, tbe_module in lookup.named_modules():
                if isinstance(tbe_module, FusedOptimizerModule):
                    # Modify param keys to match EmbeddingBagCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for (
                        param_key,
                        weight,
                    ) in tbe_module.fused_optimizer.params.items():
                        # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`
                        params["embedding_bags." + param_key] = weight
                    tbe_module.fused_optimizer.params = params
                    optims.append(("", tbe_module.fused_optimizer))

        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        new_state = self.state_dict()

        optimizer_state: Dict[str, Dict[str, Dict[str, Any]]] = self._optim.state_dict()

        # Prepare and execute communication operations for state transfer.
        shard_keys = list(changed_sharding_params.keys())
        comms_op: Dict[CommStrategy, List[CommP2PMetadata]] = {}
        reqs: List[Tuple[dist.Work, CommP2PMetadata]] = []
        # Pipeline for communication and computation overlapping
        # move shards of current table while loading next table shards for communiucation
        for i, (shard_name, nxt_shard_name) in enumerate(
            zip_longest(shard_keys, shard_keys[1:])
        ):
            if i == 0:
                # Prepare communication P2P operations
                comms_op = prepare_comm_ops(
                    module_sharding_plan=current_module_sharding_plan,
                    current_state_dict=current_state,
                    new_state_dict=new_state,
                    changed_sharding_params=changed_sharding_params,
                    shard_name=shard_name,
                    env=env,
                    current_opt_state=old_optimizer_state,
                    new_opt_state=optimizer_state,
                    extend_shard_name=self.extend_shard_name,
                    has_optimizer=has_optimizer,
                )

            if comms_op:
                # call underlying batch_isend_irecv primitives
                reqs = transfer_data(comms_op=comms_op)

            if nxt_shard_name:
                comms_op = prepare_comm_ops(
                    module_sharding_plan=current_module_sharding_plan,
                    current_state_dict=current_state,
                    new_state_dict=new_state,
                    changed_sharding_params=changed_sharding_params,
                    shard_name=nxt_shard_name,
                    env=env,
                    current_opt_state=old_optimizer_state,
                    new_opt_state=optimizer_state,
                    extend_shard_name=self.extend_shard_name,
                    has_optimizer=has_optimizer,
                )
            else:
                break
            # Update state and optimizer states
            update_state_dictionaries(
                reqs=reqs,
                old_optimizer_state=old_optimizer_state,
                new_optimizer_state=optimizer_state,
                old_state=current_state,
                new_state=new_state,
                changed_sharding_params=changed_sharding_params,
                extend_shard_name=self.extend_shard_name,
            )

        update_state_dictionaries(
            reqs=reqs,
            old_optimizer_state=old_optimizer_state,
            new_optimizer_state=optimizer_state,
            old_state=current_state,
            new_state=new_state,
            changed_sharding_params=changed_sharding_params,
            extend_shard_name=self.extend_shard_name,
            update_local=True,
        )

        # Clean up old lookup modules.
        for lookup in old_lookups:
            del lookup
        old_lookups.clear()
        self.load_state_dict(new_state, assign=True)
        if has_optimizer:
            self._optim.load_state_dict(optimizer_state)
        return

    def create_rocksdb_hard_link_snapshot(self) -> None:
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            if hasattr(lookup, "create_rocksdb_hard_link_snapshot") and callable(
                lookup.create_rocksdb_hard_link_snapshot()
            ):
                lookup.create_rocksdb_hard_link_snapshot()

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    def create_context(self) -> EmbeddingBagCollectionContext:
        return EmbeddingBagCollectionContext()

    @staticmethod
    def extend_shard_name(shard_name: str) -> str:
        return f"embedding_bags.{shard_name}.weight"

    @property
    def unsharded_module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection


class EmbeddingBagCollectionSharder(BaseEmbeddingSharder[EmbeddingBagCollection]):
    """
    This implementation uses non-fused `EmbeddingBagCollection`
    """

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingBagCollection:
        return ShardedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=self.fused_params,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            module_fqn=module_fqn,
        )

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }

    def reshard(
        self,
        sharded_module: ShardedEmbeddingBagCollection,
        changed_shard_to_params: EmbeddingModuleShardingPlan,
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBagCollection:
        """
        Updates the sharded module in place based on the changed_shard_to_params
        which contains the new ParameterSharding with different shard placements.

        Args:
            sharded_module (ShardedEmbeddingBagCollection): The module to update
            changed_shard_to_params (Dict[str, ParameterSharding]): A dictionary mapping
                table names to their new parameter sharding configs. This should only
                contain shards/table names that need to be moved
            env (ShardingEnv): The sharding environment
            device (Optional[torch.device]): The device to place the updated module on

        Returns:
            ShardedEmbeddingBagCollection: The updated sharded module
        """

        if len(changed_shard_to_params) > 0:
            sharded_module.update_shards(changed_shard_to_params, env, device)
        return sharded_module

    @property
    def module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection

    @property
    def sharded_module_type(self) -> Type[ShardedEmbeddingBagCollection]:
        return ShardedEmbeddingBagCollection


class EmbeddingAwaitable(LazyAwaitable[torch.Tensor]):
    def __init__(
        self,
        awaitable: Awaitable[torch.Tensor],
    ) -> None:
        super().__init__()
        self._awaitable = awaitable

    def _wait_impl(self) -> torch.Tensor:
        embedding = self._awaitable.wait()
        return embedding


class ShardedEmbeddingBag(
    ShardedEmbeddingModule[
        KeyedJaggedTensor, torch.Tensor, torch.Tensor, NullShardedModuleContext
    ],
    FusedOptimizerModule,
):
    """
    Sharded implementation of `nn.EmbeddingBag`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: nn.EmbeddingBag,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        assert (
            len(table_name_to_parameter_sharding) == 1
        ), "expect 1 table, but got len(table_name_to_parameter_sharding)"
        assert module.mode == "sum", "ShardedEmbeddingBag only supports sum pooling"

        self._dummy_embedding_table_name = "dummy_embedding_table_name"
        self._dummy_feature_name = "dummy_feature_name"
        self.parameter_sharding: ParameterSharding = next(
            iter(table_name_to_parameter_sharding.values())
        )
        embedding_table_config = EmbeddingTableConfig(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            name=self._dummy_embedding_table_name,
            feature_names=[self._dummy_feature_name],
            pooling=PoolingType.SUM,
            # We set is_weighted to True for now,
            # if per_sample_weights is None in forward(),
            # we could assign a all-one vector to per_sample_weights
            is_weighted=True,
            embedding_names=[self._dummy_feature_name],
        )

        if self.parameter_sharding.sharding_type == ShardingType.TABLE_WISE.value:
            # TODO: enable it with correct semantics, see T104397332
            raise RuntimeError(
                "table-wise sharding on a single EmbeddingBag is not supported yet"
            )

        self._embedding_sharding: EmbeddingSharding[
            EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
        ] = self.create_embedding_bag_sharding(
            sharding_infos=[
                EmbeddingShardingInfo(
                    embedding_config=embedding_table_config,
                    param_sharding=self.parameter_sharding,
                    param=next(iter(module.parameters())),
                    fused_params=fused_params,
                ),
            ],
            env=env,
            device=device,
            permute_embeddings=True,
        )
        self._input_dist: nn.Module = self._embedding_sharding.create_input_dist()
        self._lookup: nn.Module = self._embedding_sharding.create_lookup()
        self._output_dist: nn.Module = self._embedding_sharding.create_output_dist()

        # Get all fused optimizers and combine them.
        optims = []
        for _, module in self._lookup.named_modules():
            if isinstance(module, FusedOptimizerModule):
                # modify param keys to match EmbeddingBag
                params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                for param_key, weight in module.fused_optimizer.params.items():
                    # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`.
                    params[param_key.split(".")[-1]] = weight
                module.fused_optimizer.params = params
                optims.append(("", module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

    @classmethod
    def create_embedding_bag_sharding(
        cls,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        return ShardedEmbeddingBagCollection.create_embedding_bag_sharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            permute_embeddings=permute_embeddings,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    # pyre-ignore [14]
    def input_dist(
        self,
        ctx: NullShardedModuleContext,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        if per_sample_weights is None:
            per_sample_weights = torch.ones_like(input, dtype=torch.float)
        features = KeyedJaggedTensor(
            keys=[self._dummy_feature_name],
            values=input,
            offsets=offsets,
            weights=per_sample_weights,
        )
        return self._input_dist(features)

    def compute(
        self, ctx: NullShardedModuleContext, dist_input: KeyedJaggedTensor
    ) -> torch.Tensor:
        return self._lookup(dist_input)

    def output_dist(
        self, ctx: NullShardedModuleContext, output: torch.Tensor
    ) -> LazyAwaitable[torch.Tensor]:
        return EmbeddingAwaitable(
            awaitable=self._output_dist(output),
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        # pyre-fixme[19]: Expected 0 positional arguments.
        lookup_state_dict = self._lookup.state_dict(None, "", keep_vars)
        # update key to match embeddingBag state_dict key
        for key, item in lookup_state_dict.items():
            new_key = prefix + key.split(".")[-1]
            destination[new_key] = item
        return destination

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        yield from [(prefix, self)]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        # TODO: add remove_duplicate
        for name, parameter in self._lookup.named_parameters("", recurse):
            # update name to match embeddingBag parameter name
            yield append_prefix(prefix, name.split(".")[-1]), parameter

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        if self.parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
            yield from []
        else:
            for name, _ in self._lookup.named_parameters(""):
                yield append_prefix(prefix, name.split(".")[-1])

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        # TODO: add remove_duplicate
        for name, buffer in self._lookup.named_buffers("", recurse):
            yield append_prefix(prefix, name.split(".")[-1]), buffer

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        # update key to match  embeddingBag state_dict key
        for key, value in state_dict.items():
            new_key = ".".join([self._dummy_embedding_table_name, key])
            state_dict[new_key] = value
            state_dict.pop(key)
        missing, unexpected = self._lookup.load_state_dict(
            state_dict,
            strict,
        )
        missing_keys.extend(missing)
        unexpected_keys.extend(unexpected)

        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    def create_context(self) -> NullShardedModuleContext:
        return NullShardedModuleContext()

    @property
    def unsharded_module_type(self) -> Type[nn.EmbeddingBag]:
        return nn.EmbeddingBag


class EmbeddingBagSharder(BaseEmbeddingSharder[nn.EmbeddingBag]):
    """
    This implementation uses non-fused `nn.EmbeddingBag`
    """

    def shard(
        self,
        module: nn.EmbeddingBag,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingBag:
        return ShardedEmbeddingBag(module, params, env, self.fused_params, device)

    def shardable_parameters(self, module: nn.EmbeddingBag) -> Dict[str, nn.Parameter]:
        return {name: param for name, param in module.named_parameters()}

    @property
    def module_type(self) -> Type[nn.EmbeddingBag]:
        return nn.EmbeddingBag


def _create_mean_pooling_divisor(
    lengths: torch.Tensor,
    keys: List[str],
    offsets: torch.Tensor,
    stride: int,
    stride_per_key: List[int],
    dim_per_key: torch.Tensor,
    pooling_type_to_rs_features: Dict[str, List[str]],
    embedding_names: List[str],
    embedding_dims: List[int],
    variable_batch_per_feature: bool,
    kjt_inverse_order: torch.Tensor,
    kjt_key_indices: Dict[str, int],
    kt_key_ordering: torch.Tensor,
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with record_function("## ebc create mean pooling callback ##"):
        batch_size = (
            none_throws(inverse_indices)[1].size(dim=1)
            if variable_batch_per_feature
            else stride
        )

        if weights is not None:
            # if we have weights, lengths is the sum of weights by offsets for feature
            lengths = torch.ops.fbgemm.segment_sum_csr(1, offsets.int(), weights)

        if variable_batch_per_feature:
            inverse_indices = none_throws(inverse_indices)
            device = inverse_indices[1].device
            inverse_indices_t = inverse_indices[1]
            if len(keys) != len(inverse_indices[0]):
                inverse_indices_t = torch.index_select(
                    inverse_indices[1], 0, kjt_inverse_order
                )
            offsets = _to_offsets(torch.tensor(stride_per_key, device=device))[
                :-1
            ].unsqueeze(-1)
            indices = (inverse_indices_t + offsets).flatten()
            lengths = torch.index_select(input=lengths, dim=0, index=indices)

        # only convert the sum pooling features to be 1 lengths
        lengths = lengths.clone()
        for feature in pooling_type_to_rs_features[PoolingType.SUM.value]:
            feature_index = kjt_key_indices[feature]
            feature_index = feature_index * batch_size
            lengths[feature_index : feature_index + batch_size] = 1

        if len(embedding_names) != len(keys):
            lengths = torch.index_select(
                lengths.reshape(-1, batch_size),
                0,
                kt_key_ordering,
            ).reshape(-1)

        # transpose to align features with keyed tensor dim_per_key
        lengths = lengths.reshape(-1, batch_size).T  # [batch_size, num_features]
        output_size = sum(embedding_dims)

        divisor = torch.repeat_interleave(
            input=lengths,
            repeats=dim_per_key,
            dim=1,
            output_size=output_size,
        )
        eps = 1e-6  # used to safe guard against 0 division
        divisor = divisor + eps
        return divisor.detach()


def _apply_mean_pooling(
    keyed_tensor: KeyedTensor, divisor: torch.Tensor
) -> KeyedTensor:
    """
    Apply mean pooling to pooled embeddings in RW/TWRW sharding schemes.
    This function is applied as a callback to the awaitable
    """
    with record_function("## ebc apply mean pooling ##"):
        mean_pooled_values = (
            keyed_tensor.values() / divisor
        )  # [batch size, num_features * embedding dim]
        return KeyedTensor(
            keys=keyed_tensor.keys(),
            values=mean_pooled_values,
            length_per_key=keyed_tensor.length_per_key(),
            key_dim=1,
        )
