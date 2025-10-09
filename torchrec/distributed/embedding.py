#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import warnings
from collections import defaultdict, deque, OrderedDict
from itertools import accumulate
from typing import (
    Any,
    cast,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    Union as TypeUnion,
)

import torch
from tensordict import TensorDict
from torch import distributed as dist, nn
from torch.autograd.profiler import record_function
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torch.distributed._tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_lookup import PartiallyMaterializedTensor
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    KJTList,
    ShardedEmbeddingModule,
    ShardingType,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_IS_SSD_TABLE,
    FUSED_PARAM_SSD_TABLE_LIST,
)
from torchrec.distributed.sharding.cw_sequence_sharding import (
    CwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.dp_sequence_sharding import (
    DpSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    RwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sharding import RwSparseFeaturesDist
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.sharding.tw_sequence_sharding import (
    TwSequenceEmbeddingSharding,
)
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import (
    Awaitable,
    EmbeddingEvent,
    EmbeddingModuleShardingPlan,
    LazyAwaitable,
    Multistreamable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedTensor,
    ShardingEnv,
    ShardingEnv2D,
    ShardMetadata,
)
from torchrec.distributed.utils import (
    add_params_from_parameter_sharding,
    convert_to_fbgemm_types,
    create_global_tensor_shape_stride_from_metadata,
    maybe_annotate_embedding_event,
    merge_fused_params,
    none_throws,
    optimizer_type_to_emb_opt_type,
)
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
    EmbeddingTableConfig,
    FeatureScoreBasedEvictionPolicy,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.modules.utils import construct_jagged_tensors, SequenceVBEContext
from torchrec.optim.fused import EmptyFusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import _to_offsets, JaggedTensor, KeyedJaggedTensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except (OSError, RuntimeError):
    pass


logger: logging.Logger = logging.getLogger(__name__)


EC_INDEX_DEDUP: bool = False


def get_device_from_parameter_sharding(
    ps: ParameterSharding,
) -> TypeUnion[str, Tuple[str, ...]]:
    """
    Returns list of device type per shard if table is sharded across different device type
    else reutrns single device type for the table parameter
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
        assert ps.sharding_type in [
            ShardingType.ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ], "Only row_wise or column_wise sharding supports sharding across multiple device types for a table"
        return device_type_list


def set_ec_index_dedup(val: bool) -> None:
    warnings.warn(
        "Please set use_index_dedup in EmbeddingCollectionSharder during __init__ instead",
        DeprecationWarning,
        stacklevel=2,
    )
    global EC_INDEX_DEDUP
    EC_INDEX_DEDUP = val


def get_ec_index_dedup() -> bool:
    global EC_INDEX_DEDUP
    return EC_INDEX_DEDUP


def create_sharding_infos_by_sharding_device_group(
    module: EmbeddingCollectionInterface,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    fused_params: Optional[Dict[str, Any]],
) -> Dict[Tuple[str, TypeUnion[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]]:

    if fused_params is None:
        fused_params = {}

    sharding_type_device_group_to_sharding_infos: Dict[
        Tuple[str, TypeUnion[str, Tuple[str, ...]]], List[EmbeddingShardingInfo]
    ] = {}
    # state_dict returns parameter.Tensor, which loses parameter level attributes
    parameter_by_name = dict(module.named_parameters())
    # QuantEBC registers weights as buffers (since they are INT8), and so we need to grab it there
    state_dict = module.state_dict()

    for (
        config,
        embedding_names,
    ) in zip(module.embedding_configs(), module.embedding_names_by_table()):
        table_name = config.name
        assert table_name in table_name_to_parameter_sharding

        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.compute_kernel not in [
            kernel.value for kernel in EmbeddingComputeKernel
        ]:
            raise ValueError(
                f"Compute kernel not supported {parameter_sharding.compute_kernel}"
            )

        param_name = "embeddings." + config.name + ".weight"
        assert param_name in parameter_by_name or param_name in state_dict
        param = parameter_by_name.get(param_name, state_dict[param_name])

        # if a table name is overridden to be offloaded to ssd storage for inference
        # update the device group accordingly
        if fused_params and table_name in fused_params.get(
            FUSED_PARAM_SSD_TABLE_LIST, {}
        ):
            device_group: TypeUnion[str, Tuple[str, ...]] = "ssd"
        else:
            device_group: TypeUnion[str, Tuple[str, ...]] = (
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
        ), f"Only support 1 optimizer, given {len(optimizer_classes)}"

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
            (
                EmbeddingShardingInfo(
                    embedding_config=EmbeddingTableConfig(
                        num_embeddings=config.num_embeddings,
                        embedding_dim=config.embedding_dim,
                        name=config.name,
                        data_type=config.data_type,
                        feature_names=copy.deepcopy(config.feature_names),
                        pooling=PoolingType.NONE,
                        is_weighted=False,
                        has_feature_processor=False,
                        embedding_names=embedding_names,
                        weight_init_max=config.weight_init_max,
                        weight_init_min=config.weight_init_min,
                        total_num_buckets=config.total_num_buckets,
                        use_virtual_table=config.use_virtual_table,
                        virtual_table_eviction_policy=config.virtual_table_eviction_policy,
                    ),
                    param_sharding=parameter_sharding,
                    param=param,
                    fused_params=per_table_fused_params,
                )
            )
        )
    return sharding_type_device_group_to_sharding_infos


def pad_vbe_kjt_lengths(features: KeyedJaggedTensor) -> KeyedJaggedTensor:
    max_stride = max(features.stride_per_key())
    new_lengths = torch.zeros(
        max_stride * len(features.keys()),
        device=features.device(),
        dtype=features.lengths().dtype,
    )
    cum_stride = 0
    for i, stride in enumerate(features.stride_per_key()):
        new_lengths[i * max_stride : i * max_stride + stride] = features.lengths()[
            cum_stride : cum_stride + stride
        ]
        cum_stride += stride

    return KeyedJaggedTensor(
        keys=features.keys(),
        values=features.values(),
        lengths=new_lengths,
        stride=max_stride,
        length_per_key=features.length_per_key(),
        offset_per_key=features.offset_per_key(),
    )


def _pin_and_move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    TODO: remove and import from `jagged_tensor.py` once packaging issue is resolved
    """
    return (
        tensor.pin_memory().to(device=device, non_blocking=True)
        if device.type == "cuda" and tensor.device.type == "cpu"
        else tensor.to(device=device, non_blocking=True)
    )


class EmbeddingCollectionContext(Multistreamable):
    # Torch Dynamo does not support default_factory=list:
    # https://github.com/pytorch/pytorch/issues/120108
    # TODO(ivankobzarev): Make this a dataclass once supported

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        seq_vbe_ctx: Optional[List[SequenceVBEContext]] = None,
    ) -> None:
        super().__init__()
        self.sharding_contexts: List[SequenceShardingContext] = sharding_contexts or []
        self.input_features: List[KeyedJaggedTensor] = input_features or []
        self.reverse_indices: List[torch.Tensor] = reverse_indices or []
        self.seq_vbe_ctx: List[SequenceVBEContext] = seq_vbe_ctx or []
        self.table_name_to_unpruned_hash_sizes: Dict[str, int] = {}

    def record_stream(self, stream: torch.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)
        for f in self.input_features:
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            f.record_stream(stream)
        for r in self.reverse_indices:
            r.record_stream(stream)
        for s in self.seq_vbe_ctx:
            s.record_stream(stream)


class EmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    def __init__(
        self,
        awaitables_per_sharding: List[Awaitable[torch.Tensor]],
        features_per_sharding: List[KeyedJaggedTensor],
        embedding_names_per_sharding: List[List[str]],
        ctx: EmbeddingCollectionContext,
        need_indices: bool = False,
        features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
        module_fqn: Optional[str] = None,
        sharding_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._awaitables_per_sharding = awaitables_per_sharding
        self._features_per_sharding = features_per_sharding
        self._need_indices = need_indices
        self._features_to_permute_indices = features_to_permute_indices
        self._embedding_names_per_sharding = embedding_names_per_sharding
        self._ctx = ctx
        self._module_fqn = module_fqn
        self._sharding_types = sharding_types

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}
        for i, (w, f, e) in enumerate(
            zip(
                self._awaitables_per_sharding,
                self._features_per_sharding,
                self._embedding_names_per_sharding,
            )
        ):
            original_features = (
                None
                if i >= len(self._ctx.input_features)
                else self._ctx.input_features[i]
            )
            reverse_indices = (
                None
                if i >= len(self._ctx.reverse_indices)
                else self._ctx.reverse_indices[i]
            )
            seq_vbe_ctx = (
                None if i >= len(self._ctx.seq_vbe_ctx) else self._ctx.seq_vbe_ctx[i]
            )

            with maybe_annotate_embedding_event(
                EmbeddingEvent.OUTPUT_DIST_WAIT,
                self._module_fqn,
                self._sharding_types[i] if self._sharding_types else None,
            ):
                embeddings = w.wait()

            jt_dict.update(
                construct_jagged_tensors(
                    embeddings=embeddings,
                    features=f,
                    embedding_names=e,
                    need_indices=self._need_indices,
                    features_to_permute_indices=self._features_to_permute_indices,
                    original_features=original_features,
                    reverse_indices=reverse_indices,
                    seq_vbe_ctx=seq_vbe_ctx,
                )
            )
        return jt_dict


class ShardedEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        Dict[str, JaggedTensor],
        EmbeddingCollectionContext,
    ],
    # TODO remove after compute_kernel X sharding decoupling
    FusedOptimizerModule,
):
    """
    Sharded implementation of `EmbeddingCollection`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
        module_fqn: Optional[str] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._enable_feature_score_weight_accumulation: bool = False

        self._module_fqn = module_fqn
        self._embedding_configs: List[EmbeddingConfig] = module.embedding_configs()
        self._table_names: List[str] = [
            config.name for config in self._embedding_configs
        ]
        self._table_name_to_config: Dict[str, EmbeddingConfig] = {
            config.name: config for config in self._embedding_configs
        }
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
        # TODO get rid of get_ec_index_dedup global flag
        self._use_index_dedup: bool = use_index_dedup or get_ec_index_dedup()
        sharding_type_to_sharding_infos = self.create_grouped_sharding_infos(
            module,
            table_name_to_parameter_sharding,
            fused_params,
        )

        self._sharding_types: List[str] = list(sharding_type_to_sharding_infos.keys())

        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
            ],
        ] = {
            sharding_type: self.create_embedding_sharding(
                sharding_type=sharding_type,
                sharding_infos=embedding_confings,
                env=env,
                device=device,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

        self.enable_embedding_update: bool = any(
            config.enable_embedding_update for config in self._embedding_configs
        )
        self._device = device
        self._input_dists: List[nn.Module] = []
        self._write_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._updates: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
        self._create_output_dist()

        self._write_splits: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True
        logger.info(f"EC index dedup enabled: {self._use_index_dedup}.")

        for config in self._embedding_configs:
            virtual_table_eviction_policy = config.virtual_table_eviction_policy
            if virtual_table_eviction_policy is not None and isinstance(
                virtual_table_eviction_policy, FeatureScoreBasedEvictionPolicy
            ):
                self._enable_feature_score_weight_accumulation = True
                break

        logger.info(
            f"EC feature score weight accumulation enabled: {self._enable_feature_score_weight_accumulation}."
        )

        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, m in lookup.named_modules():
                if isinstance(m, FusedOptimizerModule):
                    # modify param keys to match EmbeddingCollection
                    params: MutableMapping[
                        str, TypeUnion[torch.Tensor, ShardedTensor]
                    ] = {}
                    for param_key, weight in m.fused_optimizer.params.items():
                        params["embeddings." + param_key] = weight
                    m.fused_optimizer.params = params
                    optims.append(("", m.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._embedding_dim: int = module.embedding_dim()
        self._embedding_names_per_sharding: List[List[str]] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._embedding_names_per_sharding.append(sharding.embedding_names())
        self._local_embedding_dim: int = self._embedding_dim
        self._features_to_permute_indices: Dict[str, List[int]] = {}
        if ShardingType.COLUMN_WISE.value in self._sharding_type_to_sharding:
            sharding = self._sharding_type_to_sharding[ShardingType.COLUMN_WISE.value]
            # CW partition must be same for all CW sharded parameters
            self._local_embedding_dim = cast(
                ShardMetadata, sharding.embedding_shard_metadata()[0]
            ).shard_sizes[1]
            self._generate_permute_indices_per_feature(
                module.embedding_configs(), table_name_to_parameter_sharding
            )
        self._need_indices: bool = module.need_indices()
        self._inverse_indices_permute_per_sharding: Optional[List[torch.Tensor]] = None
        self._skip_missing_weight_key: List[str] = []

        for index, (sharding, lookup) in enumerate(
            zip(
                self._sharding_type_to_sharding.values(),
                self._lookups,
            )
        ):
            # TODO: can move this into DpPooledEmbeddingSharding once all modules are composable
            if isinstance(sharding, DpSequenceEmbeddingSharding):
                self._lookups[index] = DistributedDataParallel(
                    module=lookup,
                    device_ids=(
                        [self._device]
                        if self._device is not None and self._device.type == "cuda"
                        else None
                    ),
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )
        self._initialize_torch_state()

        if module.device != torch.device("meta"):
            self.load_state_dict(module.state_dict())

    @classmethod
    def create_grouped_sharding_infos(
        cls,
        module: EmbeddingCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        fused_params: Optional[Dict[str, Any]],
    ) -> Dict[str, List[EmbeddingShardingInfo]]:
        """
        convert ParameterSharding (table_name_to_parameter_sharding: Dict[str, ParameterSharding]) to
        EmbeddingShardingInfo that are grouped by sharding_type, and propagate the configs/parameters
        """
        if fused_params is None:
            fused_params = {}

        sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}
        # state_dict returns parameter.Tensor, which loses parameter level attributes
        parameter_by_name = dict(module.named_parameters())
        # QuantEBC registers weights as buffers (since they are INT8), and so we need to grab it there
        state_dict = module.state_dict()

        for (
            config,
            embedding_names,
        ) in zip(module.embedding_configs(), module.embedding_names_by_table()):
            table_name = config.name
            assert table_name in table_name_to_parameter_sharding

            parameter_sharding = table_name_to_parameter_sharding[table_name]
            if parameter_sharding.compute_kernel not in [
                kernel.value for kernel in EmbeddingComputeKernel
            ]:
                raise ValueError(
                    f"Compute kernel not supported {parameter_sharding.compute_kernel}"
                )

            param_name = "embeddings." + config.name + ".weight"
            assert param_name in parameter_by_name or param_name in state_dict
            param = parameter_by_name.get(param_name, state_dict[param_name])

            if parameter_sharding.sharding_type not in sharding_type_to_sharding_infos:
                sharding_type_to_sharding_infos[parameter_sharding.sharding_type] = []

            optimizer_params = getattr(param, "_optimizer_kwargs", [{}])
            optimizer_classes = getattr(param, "_optimizer_classes", [None])

            assert (
                len(optimizer_classes) == 1 and len(optimizer_params) == 1
            ), f"Only support 1 optimizer, given {len(optimizer_classes)}"

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

            sharding_type_to_sharding_infos[parameter_sharding.sharding_type].append(
                (
                    EmbeddingShardingInfo(
                        embedding_config=EmbeddingTableConfig(
                            num_embeddings=config.num_embeddings,
                            embedding_dim=config.embedding_dim,
                            name=config.name,
                            data_type=config.data_type,
                            feature_names=copy.deepcopy(config.feature_names),
                            pooling=PoolingType.NONE,
                            is_weighted=False,
                            has_feature_processor=False,
                            embedding_names=embedding_names,
                            weight_init_max=config.weight_init_max,
                            weight_init_min=config.weight_init_min,
                            total_num_buckets=config.total_num_buckets,
                            use_virtual_table=config.use_virtual_table,
                            virtual_table_eviction_policy=config.virtual_table_eviction_policy,
                            enable_embedding_update=config.enable_embedding_update,
                        ),
                        param_sharding=parameter_sharding,
                        param=param,
                        fused_params=per_table_fused_params,
                    )
                )
            )
        return sharding_type_to_sharding_infos

    @classmethod
    def create_embedding_sharding(
        cls,
        sharding_type: str,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        """
        This is the main function to generate `EmbeddingSharding` instances based on sharding_type
        so that the same sharding_type in one EC would be fused.
        """
        if sharding_type == ShardingType.TABLE_WISE.value:
            return TwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            return RwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.DATA_PARALLEL.value:
            return DpSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
            )
        elif sharding_type == ShardingType.COLUMN_WISE.value:
            return CwSequenceEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            raise ValueError(f"Sharding not supported {sharding_type}")

    @staticmethod
    def _pre_state_dict_hook(
        self: "ShardedEmbeddingCollection",
        prefix: str = "",
        keep_vars: bool = False,
    ) -> None:
        for lookup in self._lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            lookup.flush()

    @staticmethod
    def _pre_load_state_dict_hook(
        self: "ShardedEmbeddingCollection",
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
                weight_key = f"{prefix}embeddings.{table_name}.weight"
                weight_id_key = f"{prefix}embeddings.{table_name}.weight_id"
                bucket_key = f"{prefix}embeddings.{table_name}.bucket"
                metadata_key = f"{prefix}embeddings.{table_name}.metadata"
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

            key = f"{prefix}embeddings.{table_name}.weight"
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
                dim = shards_wrapper.local_sizes()[0][1]
                if len(local_shards) == 0:
                    state_dict[key] = torch.empty(0)
                elif len(local_shards) > 1:
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

    def _initialize_torch_state(self) -> None:  # noqa
        """
        This provides consistency between this class and the EmbeddingCollection's
        nn.Module API calls (state_dict, named_modules, etc)
        """

        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        for table_name in self._table_names:
            self.embeddings[table_name] = nn.Module()
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
                # Don't need to use sharded/distributed state tensor for DATA_PARALLEL
                # because each rank has a full copy of the table in DATA_PARALLEL
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
        for table in self._embedding_configs:
            self._name_to_table_size[table.name] = (
                table.num_embeddings,
                table.embedding_dim,
            )

        for sharding_type, lookup in zip(
            self._sharding_type_to_sharding.keys(), self._lookups
        ):
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                # unwrap DDP
                lookup = lookup.module
            else:
                # save local_shards for transforming MP params to shardedTensor
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
                self.embeddings[table_name].register_parameter("weight", tbe_slice)
        for table_name in self._model_parallel_name_to_local_shards.keys():
            local_shards = self._model_parallel_name_to_local_shards[table_name]
            shards_wrapper_map = self._model_parallel_name_to_shards_wrapper[table_name]

            # for shards that don't exist on this rank, register with empty tensor
            if not hasattr(self.embeddings[table_name], "weight"):
                self.embeddings[table_name].register_parameter(
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
                    self.embeddings[table_name].weight._in_backward_optimizers = [
                        EmptyFusedOptimizer()
                    ]

            if self._output_dtensor:
                assert _model_parallel_name_to_compute_kernel[table_name] not in {
                    EmbeddingComputeKernel.KEY_VALUE.value,
                    EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
                    EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
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
                            if self._table_name_to_config[table_name].use_virtual_table
                            else self._name_to_table_size[table_name]
                        ),
                        process_group=(
                            self._env.sharding_pg
                            if isinstance(self._env, ShardingEnv2D)
                            else self._env.process_group
                        ),
                    )
                )

        def extract_sharded_kvtensors(
            module: ShardedEmbeddingCollection,
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
            module: ShardedEmbeddingCollection,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            # Adjust dense MP
            for (
                table_name,
                sharded_t,
            ) in module._model_parallel_name_to_sharded_tensor.items():
                destination_key = f"{prefix}embeddings.{table_name}.weight"
                destination[destination_key] = sharded_t
            for (
                table_name,
                d_tensor,
            ) in module._model_parallel_name_to_dtensor.items():
                destination_key = f"{prefix}embeddings.{table_name}.weight"
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
            for lookup, sharding_type in zip(
                module._lookups, module._sharding_type_to_sharding.keys()
            ):
                if sharding_type != ShardingType.DATA_PARALLEL.value:
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
                destination_key = f"{prefix}embeddings.{table_name}.{tensor_name}"
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
            module: "ShardedEmbeddingCollection",
            incompatible_keys: _IncompatibleKeys,
        ) -> None:
            if incompatible_keys.missing_keys:
                # has to remove the key inplace
                for skip_key in module._skip_missing_weight_key:
                    if skip_key in incompatible_keys.missing_keys:
                        incompatible_keys.missing_keys.remove(skip_key)

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
        # Initialize embedding weights with init_fn
        for table_config in self._embedding_configs:
            if self.module_sharding_plan[table_config.name].compute_kernel in {
                EmbeddingComputeKernel.KEY_VALUE.value,
                EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
                EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
            }:
                continue
            assert table_config.init_fn is not None
            param = self.embeddings[f"{table_config.name}"].weight
            # pyre-ignore
            table_config.init_fn(param)

            sharding_type = self.module_sharding_plan[table_config.name].sharding_type
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                pg = self._env.process_group
                with torch.no_grad():
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `TypeUnion[Module, Tensor]`.
                    dist.broadcast(param.data, src=0, group=pg)

    def _generate_permute_indices_per_feature(
        self,
        embedding_configs: List[EmbeddingConfig],
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    ) -> None:
        """
        Generates permute indices per feature for column-wise sharding.

        Since outputs are stored in order of rank, column-wise shards of a table on the
        same rank will be seen as adjacent, which may not be correct.

        The permute indices store the correct ordering of outputs relative to the
        provided ordering.

        Example::
            rank_0 = [f_0(shard_0), f_0(shard_2)]
            rank_1 = [f_0(shard_1)]
            output = [f_0(shard_0), f_0(shard_2), f_0(shard_1)]

            shard_ranks = [0, 1, 0]
            output_ranks = [0, 0, 1]

            # To get the correct order from output_ranks -> shard_ranks
            permute_indices = [0, 2, 1]
        """
        shared_feature: Dict[str, bool] = {}
        for table in embedding_configs:
            for feature_name in table.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        for table in embedding_configs:
            sharding = table_name_to_parameter_sharding[table.name]
            if sharding.sharding_type != ShardingType.COLUMN_WISE.value:
                continue
            ranks = cast(List[int], sharding.ranks)
            rank_to_indices = defaultdict(deque)
            for i, rank in enumerate(sorted(ranks)):
                rank_to_indices[rank].append(i)
            permute_indices = [rank_to_indices[rank].popleft() for rank in ranks]
            for feature_name in table.feature_names:
                if shared_feature[feature_name]:
                    self._features_to_permute_indices[
                        feature_name + "@" + table.name
                    ] = permute_indices
                else:
                    self._features_to_permute_indices[feature_name] = permute_indices

    def _create_hash_size_info(
        self,
        feature_names: List[str],
        ctx: Optional[EmbeddingCollectionContext] = None,
    ) -> None:
        feature_index = 0
        table_to_unpruned_size_mapping: Optional[Dict[str, int]] = None
        if (
            ctx is not None
            and getattr(ctx, "table_name_to_unpruned_hash_sizes", None)
            and len(ctx.table_name_to_unpruned_hash_sizes) > 0
        ):
            table_to_unpruned_size_mapping = ctx.table_name_to_unpruned_hash_sizes
        for i, sharding in enumerate(self._sharding_type_to_sharding.values()):
            feature_hash_size: List[int] = []
            feature_hash_size_lengths: List[int] = []
            for table in sharding.embedding_tables():
                table_hash_size = [0] * table.num_features()
                if table_to_unpruned_size_mapping and table.name:
                    table_hash_size[-1] = table_to_unpruned_size_mapping[table.name]
                else:
                    table_hash_size[-1] = table.num_embeddings
                feature_hash_size.extend(table_hash_size)

                table_hash_size = [0] * table.num_features()
                table_hash_size[0] = table.num_features()
                feature_hash_size_lengths.extend(table_hash_size)

                # Sanity check for feature orders
                for f in range(table.num_features()):
                    assert feature_names[feature_index + f] == table.feature_names[f]
                feature_index += table.num_features()

            feature_hash_size_cumsum: List[int] = [0] + list(
                accumulate(feature_hash_size)
            )
            feature_hash_size_offset: List[int] = [0] + list(
                accumulate(feature_hash_size_lengths)
            )

            # Register buffers for this shard
            self.register_buffer(
                f"_hash_size_cumsum_tensor_{i}",
                torch.tensor(
                    feature_hash_size_cumsum, device=self._device, dtype=torch.int64
                ),
                persistent=False,
            )
            self.register_buffer(
                f"_hash_size_offset_tensor_{i}",
                torch.tensor(
                    feature_hash_size_offset, device=self._device, dtype=torch.int64
                ),
                persistent=False,
            )

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        ctx: Optional[EmbeddingCollectionContext] = None,
    ) -> None:
        feature_names: List[str] = []
        self._feature_splits: List[int] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))
        self._features_order: List[int] = []
        for f in feature_names:
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=self._device, dtype=torch.int32),
            persistent=False,
        )

        if self._use_index_dedup:
            self._create_hash_size_info(feature_names, ctx)

    def _create_lookups(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            lookup = sharding.create_lookup()
            if self.enable_embedding_update and sharding.enable_embedding_update:
                self._updates.append(sharding.create_update(lookup))
            self._lookups.append(lookup)

    def _create_output_dist(
        self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist())

    def _dedup_indices(
        self,
        ctx: EmbeddingCollectionContext,
        input_feature_splits: List[KeyedJaggedTensor],
    ) -> List[KeyedJaggedTensor]:
        with record_function("## dedup_ec_indices ##"):
            features_by_shards = []
            for i, input_feature in enumerate(input_feature_splits):
                hash_size_cumsum = self.get_buffer(f"_hash_size_cumsum_tensor_{i}")
                hash_size_offset = self.get_buffer(f"_hash_size_offset_tensor_{i}")
                (
                    lengths,
                    offsets,
                    unique_indices,
                    reverse_indices,
                ) = torch.ops.fbgemm.jagged_unique_indices(
                    hash_size_cumsum,
                    hash_size_offset,
                    input_feature.offsets().to(torch.int64),
                    input_feature.values().to(torch.int64),
                )
                acc_weights = None
                if (
                    self._enable_feature_score_weight_accumulation
                    and input_feature.weights_or_none() is not None
                ):
                    source_weights = input_feature.weights()
                    assert (
                        source_weights.dtype == torch.float32
                    ), "Only float32 weights are supported for feature score eviction weights."

                    acc_weights = torch.ops.fbgemm.jagged_acc_weights_and_counts(
                        source_weights.view(-1),
                        reverse_indices,
                        unique_indices.numel(),
                    )

                dedup_features = KeyedJaggedTensor(
                    keys=input_feature.keys(),
                    lengths=lengths,
                    offsets=offsets,
                    values=unique_indices,
                    weights=(
                        acc_weights.view(torch.float64).view(-1)
                        if acc_weights is not None
                        else None
                    ),
                )

                ctx.input_features.append(input_feature)
                ctx.reverse_indices.append(reverse_indices)
                features_by_shards.append(dedup_features)

        return features_by_shards

    def _create_inverse_indices_permute_per_sharding(
        self, inverse_indices: Tuple[List[str], torch.Tensor]
    ) -> None:
        if (
            len(self._embedding_names_per_sharding) == 1
            and self._embedding_names_per_sharding[0] == inverse_indices[0]
        ):
            return
        index_per_name = {name: i for i, name in enumerate(inverse_indices[0])}
        permute_per_sharding = []
        for emb_names in self._embedding_names_per_sharding:
            permute = _pin_and_move(
                torch.tensor(
                    [index_per_name[name.split("@")[0]] for name in emb_names]
                ),
                inverse_indices[1].device,
            )
            permute_per_sharding.append(permute)
        self._inverse_indices_permute_per_sharding = permute_per_sharding

    def _compute_sequence_vbe_context(
        self,
        ctx: EmbeddingCollectionContext,
        unpadded_features: KeyedJaggedTensor,
    ) -> None:
        assert (
            unpadded_features.inverse_indices_or_none() is not None
        ), "inverse indices must be provided from KJT if using variable batch size per feature."

        inverse_indices = unpadded_features.inverse_indices()
        stride = inverse_indices[1].numel() // len(inverse_indices[0])
        if self._inverse_indices_permute_per_sharding is None:
            self._create_inverse_indices_permute_per_sharding(inverse_indices)

        if self._features_order:
            unpadded_features = unpadded_features.permute(
                self._features_order,
                # pyre-fixme[6]: For 2nd argument expected `Optional[Tensor]` but
                #  got `TypeUnion[Module, Tensor]`.
                self._features_order_tensor,
            )

        features_by_sharding = unpadded_features.split(self._feature_splits)
        for i, feature in enumerate(features_by_sharding):
            if self._inverse_indices_permute_per_sharding is not None:
                permute = self._inverse_indices_permute_per_sharding[i]
                permuted_indices = torch.index_select(inverse_indices[1], 0, permute)
            else:
                permuted_indices = inverse_indices[1]
            stride_per_key = _pin_and_move(
                torch.tensor(feature.stride_per_key()), feature.device()
            )
            offsets = _to_offsets(stride_per_key)[:-1].unsqueeze(-1)
            recat = (permuted_indices + offsets).flatten().int()

            if self._need_indices:
                reindexed_lengths, reindexed_values, _ = (
                    torch.ops.fbgemm.permute_1D_sparse_data(
                        recat,
                        feature.lengths(),
                        feature.values(),
                    )
                )
            else:
                reindexed_lengths = torch.index_select(feature.lengths(), 0, recat)
                reindexed_values = None

            reindexed_lengths = reindexed_lengths.view(-1, stride)
            reindexed_length_per_key = torch.sum(reindexed_lengths, dim=1).tolist()

            ctx.seq_vbe_ctx.append(
                SequenceVBEContext(
                    recat=recat,
                    unpadded_lengths=feature.lengths(),
                    reindexed_lengths=reindexed_lengths,
                    reindexed_length_per_key=reindexed_length_per_key,
                    reindexed_values=reindexed_values,
                )
            )

    # pyre-ignore [14]
    def input_dist(
        self,
        ctx: EmbeddingCollectionContext,
        features: TypeUnion[KeyedJaggedTensor, TensorDict],
    ) -> Awaitable[Awaitable[KJTList]]:
        need_permute: bool = True
        if isinstance(features, TensorDict):
            feature_keys = list(features.keys())  # pyre-ignore[6]
            if self._features_order:
                feature_keys = [feature_keys[i] for i in self._features_order]
                need_permute = False
            features = maybe_td_to_kjt(features, feature_keys)  # pyre-ignore[6]
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys(), ctx=ctx)
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            unpadded_features = None
            if features.variable_stride_per_key():
                unpadded_features = features
                features = pad_vbe_kjt_lengths(unpadded_features)

            if need_permute and self._features_order:
                features = features.permute(
                    self._features_order,
                    # pyre-fixme[6]: For 2nd argument expected `Optional[Tensor]`
                    #  but got `TypeUnion[Module, Tensor]`.
                    self._features_order_tensor,
                )
            features_by_shards = features.split(self._feature_splits)
            if self._use_index_dedup:
                features_by_shards = self._dedup_indices(ctx, features_by_shards)

            awaitables = []
            for input_dist, features, sharding_type in zip(
                self._input_dists, features_by_shards, self._sharding_type_to_sharding
            ):
                with maybe_annotate_embedding_event(
                    EmbeddingEvent.KJT_SPLITS_DIST, self._module_fqn, sharding_type
                ):
                    awaitables.append(input_dist(features))

                ctx.sharding_contexts.append(
                    SequenceShardingContext(
                        features_before_input_dist=features,
                        unbucketize_permute_tensor=(
                            input_dist.unbucketize_permute_tensor
                            if isinstance(input_dist, RwSparseFeaturesDist)
                            else None
                        ),
                    )
                )
            if unpadded_features is not None:
                self._compute_sequence_vbe_context(ctx, unpadded_features)

        return KJTListSplitsAwaitable(
            awaitables,
            ctx,
            self._module_fqn,
            list(self._sharding_type_to_sharding.keys()),
        )

    def compute(
        self, ctx: EmbeddingCollectionContext, dist_input: KJTList
    ) -> List[torch.Tensor]:
        ret: List[torch.Tensor] = []
        for lookup, features, sharding_ctx, sharding_type in zip(
            self._lookups,
            dist_input,
            ctx.sharding_contexts,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)
            ret.append(lookup(features).view(-1, embedding_dim))
        return ret

    def output_dist(
        self, ctx: EmbeddingCollectionContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for odist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
            ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(embeddings, sharding_ctx))
            features_before_all2all_per_sharding.append(
                # pyre-fixme[6]: For 1st argument expected `KeyedJaggedTensor` but
                #  got `Optional[KeyedJaggedTensor]`.
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
            ctx=ctx,
        )

    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, input: KJTList
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for lookup, odist, features, sharding_ctx, sharding_type in zip(
            self._lookups,
            self._output_dists,
            input,
            ctx.sharding_contexts,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)

            with maybe_annotate_embedding_event(
                EmbeddingEvent.LOOKUP, self._module_fqn, sharding_type
            ):
                embs = lookup(features)
                if self.post_lookup_tracker_fn is not None:
                    self.post_lookup_tracker_fn(features, embs)

            with maybe_annotate_embedding_event(
                EmbeddingEvent.OUTPUT_DIST, self._module_fqn, sharding_type
            ):
                awaitables_per_sharding.append(
                    odist(embs.view(-1, embedding_dim), sharding_ctx)
                )
                if self.post_odist_tracker_fn is not None:
                    self.post_odist_tracker_fn()

            features_before_all2all_per_sharding.append(
                # pyre-fixme[6]: For 1st argument expected `KeyedJaggedTensor` but
                #  got `Optional[KeyedJaggedTensor]`.
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
            ctx=ctx,
            module_fqn=self._module_fqn,
            sharding_types=list(self._sharding_type_to_sharding.keys()),
        )

    def _embedding_dim_for_sharding_type(self, sharding_type: str) -> int:
        return (
            self._local_embedding_dim
            if sharding_type == ShardingType.COLUMN_WISE.value
            else self._embedding_dim
        )

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

    def create_context(self) -> EmbeddingCollectionContext:
        return EmbeddingCollectionContext(sharding_contexts=[])

    def _create_write_dist(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            if sharding.enable_embedding_update:
                self._write_dists.append(sharding.create_write_dist())
                self._write_splits.append(sharding._get_num_writable_features())

    # pyre-ignore [14]
    def write_dist(
        self, ctx: EmbeddingCollectionContext, embeddings: KeyedJaggedTensor
    ) -> Awaitable[Awaitable[KJTList]]:
        if not self.enable_embedding_update:
            raise ValueError("enable_embedding_update is False for this collection")
        if not self._write_dists:
            self._create_write_dist()
        with torch.no_grad():
            embeddings_by_shards = embeddings.split(self._write_splits)
            awaitables = []
            for write_dist, embeddings in zip(self._write_dists, embeddings_by_shards):
                awaitables.append(write_dist(embeddings))

        return KJTListSplitsAwaitable(
            awaitables,
            ctx,
            self._module_fqn,
            list(self._sharding_type_to_sharding.keys()),
        )

    def update(self, ctx: EmbeddingCollectionContext, dist_input: KJTList) -> None:
        for update, embeddings in zip(
            self._updates,
            dist_input,
        ):
            update(embeddings)


class EmbeddingCollectionSharder(BaseEmbeddingSharder[EmbeddingCollection]):
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry)
        self._use_index_dedup = use_index_dedup

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingCollection:
        return ShardedEmbeddingCollection(
            module,
            params,
            env,
            self.fused_params,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            use_index_dedup=self._use_index_dedup,
            module_fqn=module_fqn,
        )

    def shardable_parameters(
        self, module: EmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embeddings.named_parameters()
        }

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.ROW_WISE.value,
        ]
        return types

    @property
    def module_type(self) -> Type[EmbeddingCollection]:
        return EmbeddingCollection

    @property
    def sharded_module_type(self) -> Type[ShardedEmbeddingCollection]:
        return ShardedEmbeddingCollection
