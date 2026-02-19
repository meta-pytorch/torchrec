#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torchrec
from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn, quantization as quant, Tensor
from torch.distributed._shard.sharding_spec import ShardingSpec
from torch.utils import _pytree as pytree
from torchrec import (
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import DEFAULT_PERF_ESTIMATOR
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.estimator import EmbeddingPerfEstimatorFactory
from torchrec.distributed.planner.shard_estimators import EmbeddingStorageEstimator
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollectionSharder,
    ShardedQuantEmbeddingCollection,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    QuantEmbeddingBagCollectionSharder,
    QuantFeatureProcessedEmbeddingBagCollectionSharder,
    ShardedQuantEmbeddingBagCollection,
    ShardedQuantFeatureProcessedEmbeddingBagCollection,
)
from torchrec.distributed.quant_state import WeightSpec
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.types import (
    ModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.distributed.utils import CopyableMixin
from torchrec.inference.modules import set_pruning_data, shard_quant_model
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingBagConfig,
    QuantConfig,
    VirtualTableEvictionPolicy,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import PositionWeightedModuleCollection
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    FeatureProcessedEmbeddingBagCollection as QuantFeatureProcessedEmbeddingBagCollection,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    MODULE_ATTR_REGISTER_TBES_BOOL,
    quant_prep_enable_quant_state_dict_split_scale_bias_for_types,
    quant_prep_enable_register_tbes,
    QuantManagedCollisionEmbeddingCollection,
)


@dataclass
class TestModelInfo:
    sparse_device: torch.device
    dense_device: torch.device
    num_features: int
    num_float_features: int
    num_weighted_features: int
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]] = field(
        default_factory=list
    )
    weighted_tables: List[EmbeddingBagConfig] = field(default_factory=list)
    model: torch.nn.Module = torch.nn.Module()
    quant_model: torch.nn.Module = torch.nn.Module()
    sharders: List[ModuleSharder] = field(default_factory=list)
    topology: Optional[Topology] = None
    planner: Optional[EmbeddingShardingPlanner] = None


class KJTInputExportWrapper(torch.nn.Module):
    def __init__(
        self,
        module_kjt_input: torch.nn.Module,
        kjt_keys: List[str],
    ) -> None:
        super().__init__()
        self._module_kjt_input = module_kjt_input
        self._kjt_keys = kjt_keys

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        kjt = KeyedJaggedTensor(
            keys=self._kjt_keys,
            values=values,
            lengths=lengths,
            weights=weights,
        )
        output = self._module_kjt_input(kjt, *args, **kwargs)
        # TODO(ivankobzarev): Support of None leaves in dynamo/export (e.g. KJT offsets)
        return [leaf for leaf in pytree.tree_leaves(output) if leaf is not None]


class KJTInputExportDynamicShapeWrapper(torch.nn.Module):
    def __init__(
        self,
        kjt_input_wrapper: KJTInputExportWrapper,
    ) -> None:
        super().__init__()
        self.kjt_input_wrapper = kjt_input_wrapper

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        # Generate unbacked symints to represent sizes
        # for values and weights, constrain them reasonably
        values_size = values[0].item()
        torch._check_is_size(values_size)
        torch._check(values_size >= lengths.shape[0])
        # pyrefly: ignore[no-matching-overload]
        values = torch.ones(values_size).to(values.device)
        if weights is not None:
            weights_size = weights.int()[0].item()
            torch._check_is_size(weights_size)
            torch._check(weights_size >= lengths.shape[0])
            # pyrefly: ignore[no-matching-overload]
            weights = torch.ones(weights_size).to(weights.device)

        return self.kjt_input_wrapper(values, lengths, weights, *args, **kwargs)


def prep_inputs(
    model_info: TestModelInfo,
    world_size: int,
    batch_size: int = 1,
    count: int = 5,
    long_indices: bool = True,
) -> List[ModelInput]:
    inputs = []
    if long_indices:
        indices_dtype = torch.int64
        lengths_dtype = torch.int64
    else:
        indices_dtype = torch.int32
        lengths_dtype = torch.int32
    for _ in range(count):
        inputs.append(
            ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=model_info.num_float_features,
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
                indices_dtype=indices_dtype,
                lengths_dtype=lengths_dtype,
            )[1][0],
        )

    return inputs


class KJTInputExportWrapperWithStrides(torch.nn.Module):
    """
    Version of KJTInputExportWrapper with stride_per_key_per_rank_tensor argument for VB path.
    """

    def __init__(
        self,
        module_kjt_input: torch.nn.Module,
        kjt_keys: List[str],
    ) -> None:
        super().__init__()
        self._module_kjt_input = module_kjt_input
        self._kjt_keys = kjt_keys

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        stride_per_key_per_rank: Optional[List[List[int]]],
        *args,
        **kwargs,
    ):
        kjt = KeyedJaggedTensor(
            keys=self._kjt_keys,
            values=values,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )
        output = self._module_kjt_input(kjt, *args, **kwargs)
        return [leaf for leaf in pytree.tree_leaves(output) if leaf is not None]


def prep_inputs_multiprocess(
    model_info: TestModelInfo, world_size: int, batch_size: int = 1, count: int = 5
) -> List[Tuple[ModelInput, List[ModelInput]]]:
    inputs = []
    for _ in range(count):
        inputs.append(
            ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=model_info.num_float_features,
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
            )
        )
    return inputs


def model_input_to_forward_args_kjt(
    mi: ModelInput,
) -> Tuple[
    List[str],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    kjt = mi.idlist_features
    assert isinstance(kjt, KeyedJaggedTensor)
    return (
        kjt._keys,
        kjt._values,
        kjt._weights,
        kjt._lengths,
        kjt._offsets,
    )


# We want to be torch types bound, args for TorchTypesModelInputWrapper
def model_input_to_forward_args(
    mi: ModelInput,
) -> Tuple[
    torch.Tensor,
    List[str],
    torch.Tensor,
    List[str],
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    idlist_kjt = mi.idlist_features
    idscore_kjt = mi.idscore_features
    assert isinstance(idlist_kjt, KeyedJaggedTensor)
    assert isinstance(idscore_kjt, KeyedJaggedTensor)
    return (
        mi.float_features,
        idlist_kjt._keys,
        idlist_kjt._values,
        idscore_kjt._keys,
        idscore_kjt._values,
        idscore_kjt._weights,
        mi.label,
        idlist_kjt._lengths,
        idlist_kjt._offsets,
        idscore_kjt._lengths,
        idscore_kjt._offsets,
    )


def create_cw_min_partition_constraints(
    table_min_partition_pairs: List[Tuple[str, int]],
) -> Dict[str, ParameterConstraints]:
    return {
        name: ParameterConstraints(
            sharding_types=[ShardingType.COLUMN_WISE.value],
            min_partition=min_partition,
        )
        for name, min_partition in table_min_partition_pairs
    }


def quantize(
    module: torch.nn.Module,
    inplace: bool,
    output_type: torch.dtype = torch.float,
    register_tbes: bool = False,
    quant_state_dict_split_scale_bias: bool = False,
    weight_dtype: torch.dtype = torch.qint8,
    per_table_weight_dtypes: Optional[Dict[str, torch.dtype]] = None,
) -> torch.nn.Module:
    module_types: List[Type[torch.nn.Module]] = [
        # pyrefly: ignore[implicit-import]
        torchrec.modules.embedding_modules.EmbeddingBagCollection,
        # pyrefly: ignore[implicit-import]
        torchrec.modules.embedding_modules.EmbeddingCollection,
        # pyrefly: ignore[implicit-import]
        torchrec.modules.mc_embedding_modules.ManagedCollisionEmbeddingCollection,
    ]
    if register_tbes:
        quant_prep_enable_register_tbes(module, module_types)
    if quant_state_dict_split_scale_bias:
        quant_prep_enable_quant_state_dict_split_scale_bias_for_types(
            module, module_types
        )

    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_type),
        weight=quant.PlaceholderObserver.with_args(dtype=weight_dtype),
    )

    if per_table_weight_dtypes:
        qconfig = QuantConfig(
            activation=quant.PlaceholderObserver.with_args(dtype=output_type),
            weight=quant.PlaceholderObserver.with_args(dtype=torch.quint8),
            per_table_weight_dtype=per_table_weight_dtypes,
        )

    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            EmbeddingBagCollection: qconfig,
            EmbeddingCollection: qconfig,
            ManagedCollisionEmbeddingCollection: qconfig,
        },
        mapping={
            EmbeddingBagCollection: QuantEmbeddingBagCollection,
            EmbeddingCollection: QuantEmbeddingCollection,
            ManagedCollisionEmbeddingCollection: QuantManagedCollisionEmbeddingCollection,
        },
        inplace=inplace,
    )


def quantize_fpebc(
    module: torch.nn.Module,
    inplace: bool,
    output_type: torch.dtype = torch.float,
    register_tbes: bool = False,
    quant_state_dict_split_scale_bias: bool = False,
    weight_dtype: torch.dtype = torch.qint8,
    per_table_weight_dtypes: Optional[Dict[str, torch.dtype]] = None,
) -> torch.nn.Module:
    module_types: List[Type[torch.nn.Module]] = [
        # pyrefly: ignore[implicit-import]
        torchrec.modules.fp_embedding_modules.FeatureProcessedEmbeddingBagCollection,
    ]
    if register_tbes:
        quant_prep_enable_register_tbes(module, module_types)
    if quant_state_dict_split_scale_bias:
        quant_prep_enable_quant_state_dict_split_scale_bias_for_types(
            module, module_types
        )

    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_type),
        weight=quant.PlaceholderObserver.with_args(dtype=weight_dtype),
    )

    if per_table_weight_dtypes:
        qconfig = QuantConfig(
            activation=quant.PlaceholderObserver.with_args(dtype=output_type),
            weight=quant.PlaceholderObserver.with_args(dtype=torch.quint8),
            per_table_weight_dtype=per_table_weight_dtypes,
        )

    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            FeatureProcessedEmbeddingBagCollection: qconfig,
        },
        mapping={
            FeatureProcessedEmbeddingBagCollection: QuantFeatureProcessedEmbeddingBagCollection,
        },
        inplace=inplace,
    )


class TestQuantFPEBCSharder(QuantFeatureProcessedEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__(fused_params=fused_params, shardable_params=shardable_params)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    def shard(
        self,
        module: QuantFeatureProcessedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantFeatureProcessedEmbeddingBagCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        # pyrefly: ignore[unsupported-operation]
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
            module, MODULE_ATTR_REGISTER_TBES_BOOL, False
        )
        # pyrefly: ignore[unsupported-operation]
        fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
            module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
        )
        return ShardedQuantFeatureProcessedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=fused_params,
            device=device,
            feature_processor=module.feature_processor,
        )


class TestQuantEBCSharder(QuantEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__(fused_params=fused_params, shardable_params=shardable_params)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    def shard(
        self,
        module: QuantEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        # pyrefly: ignore[unsupported-operation]
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
            module, MODULE_ATTR_REGISTER_TBES_BOOL, False
        )
        # pyrefly: ignore[unsupported-operation]
        fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
            module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
        )
        return ShardedQuantEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=fused_params,
            device=device,
        )


class TestQuantECSharder(QuantEmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__(fused_params=fused_params, shardable_params=shardable_params)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    def shard(
        self,
        module: QuantEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: Union[Dict[str, ShardingEnv], ShardingEnv],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        # pyrefly: ignore[unsupported-operation]
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
            module, MODULE_ATTR_REGISTER_TBES_BOOL, False
        )
        # pyrefly: ignore[unsupported-operation]
        fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
            module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
        )
        return ShardedQuantEmbeddingCollection(
            module, params, env, fused_params, device
        )


class KJTInputWrapper(torch.nn.Module):
    def __init__(
        self,
        module_kjt_input: torch.nn.Module,
    ) -> None:
        super().__init__()
        self._module_kjt_input = module_kjt_input
        self.add_module("_module_kjt_input", self._module_kjt_input)

    def forward(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ):
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
        )
        return self._module_kjt_input(kjt)


# Wrapper for module that accepts ModelInput to avoid jit scripting of ModelInput (dataclass) and be fully torch types bound.
class TorchTypesModelInputWrapper(CopyableMixin):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(
        self,
        float_features: torch.Tensor,
        idlist_features_keys: List[str],
        idlist_features_values: torch.Tensor,
        idscore_features_keys: List[str],
        idscore_features_values: torch.Tensor,
        idscore_features_weights: torch.Tensor,
        label: torch.Tensor,
        idlist_features_lengths: Optional[torch.Tensor] = None,
        idlist_features_offsets: Optional[torch.Tensor] = None,
        idscore_features_lengths: Optional[torch.Tensor] = None,
        idscore_features_offsets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        idlist_kjt = KeyedJaggedTensor(
            keys=idlist_features_keys,
            values=idlist_features_values,
            lengths=idlist_features_lengths,
            offsets=idlist_features_offsets,
        )
        idscore_kjt = KeyedJaggedTensor(
            keys=idscore_features_keys,
            values=idscore_features_values,
            weights=idscore_features_weights,
            lengths=idscore_features_lengths,
            offsets=idscore_features_offsets,
        )
        mi = ModelInput(
            float_features=float_features,
            idlist_features=idlist_kjt,
            idscore_features=idscore_kjt,
            label=label,
        )
        return self._module(mi)


def create_test_model(
    num_embeddings: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    dense_device: torch.device,
    sparse_device: torch.device,
    quant_state_dict_split_scale_bias: bool = False,
    num_features: int = 1,
    num_float_features: int = 8,
    num_weighted_features: int = 1,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    weight_dtype: torch.dtype = torch.qint8,
    pruning_dict: Optional[Dict[str, int]] = None,
    virtual_table_eviction_policy: Optional[VirtualTableEvictionPolicy] = None,
) -> TestModelInfo:
    topology: Topology = Topology(
        world_size=world_size, compute_device=sparse_device.type
    )
    mi = TestModelInfo(
        dense_device=dense_device,
        sparse_device=sparse_device,
        num_features=num_features,
        num_float_features=num_float_features,
        num_weighted_features=num_weighted_features,
        topology=topology,
    )

    mi.planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            estimator=[
                EmbeddingPerfEstimatorFactory.create(
                    DEFAULT_PERF_ESTIMATOR, topology=topology, is_inference=True
                ),
                EmbeddingStorageEstimator(topology=topology),
            ],
            constraints=constraints,
        ),
    )

    mi.tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
            use_virtual_table=True if virtual_table_eviction_policy else False,
            virtual_table_eviction_policy=virtual_table_eviction_policy,
        )
        for i in range(mi.num_features)
    ]

    mi.weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
            use_virtual_table=True if virtual_table_eviction_policy else False,
            virtual_table_eviction_policy=virtual_table_eviction_policy,
        )
        for i in range(mi.num_weighted_features)
    ]

    if pruning_dict:
        for config in mi.tables + mi.weighted_tables:
            if config.name in pruning_dict:
                config.num_embeddings_post_pruning = pruning_dict[config.name]

    mi.model = TorchTypesModelInputWrapper(
        TestSparseNN(
            tables=mi.tables,
            weighted_tables=mi.weighted_tables,
            num_float_features=mi.num_float_features,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
    )
    mi.model.training = False

    if pruning_dict:
        set_pruning_data(mi.model, pruning_dict)

    mi.quant_model = quantize(
        module=mi.model,
        inplace=False,
        quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        weight_dtype=weight_dtype,
    )
    return mi


def create_test_model_ebc_only_no_quantize(
    num_embeddings: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    dense_device: torch.device,
    sparse_device: torch.device,
    num_features: int = 1,
    num_float_features: int = 8,
    num_weighted_features: int = 1,
    compute_device: str = "cuda",
    feature_processor: bool = False,
) -> TestModelInfo:
    topology: Topology = Topology(world_size=world_size, compute_device=compute_device)
    mi = TestModelInfo(
        dense_device=dense_device,
        sparse_device=sparse_device,
        num_features=num_features,
        num_float_features=num_float_features,
        num_weighted_features=num_weighted_features,
        topology=topology,
    )

    mi.planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            estimator=[
                EmbeddingPerfEstimatorFactory.create(
                    DEFAULT_PERF_ESTIMATOR, topology=topology, is_inference=True
                ),
                EmbeddingStorageEstimator(topology=topology),
            ],
        ),
    )

    mi.tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(mi.num_features)
    ]

    mi.weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=emb_dim,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(mi.num_weighted_features)
    ]

    if feature_processor:
        max_feature_lengths = {config.feature_names[0]: 100 for config in mi.tables}
        fp = PositionWeightedModuleCollection(
            max_feature_lengths=max_feature_lengths, device=mi.sparse_device
        )
        ebc = FeatureProcessedEmbeddingBagCollection(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=mi.tables,
                device=mi.sparse_device,
                is_weighted=True,
            ),
            feature_processors=fp,
        )
    else:
        ebc = EmbeddingBagCollection(
            tables=mi.tables,
            device=mi.sparse_device,
        )

    mi.model = torch.nn.Sequential(ebc)
    mi.model.training = False
    return mi


def create_test_model_ebc_only(
    num_embeddings: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    dense_device: torch.device,
    sparse_device: torch.device,
    num_features: int = 1,
    num_float_features: int = 8,
    num_weighted_features: int = 1,
    quant_state_dict_split_scale_bias: bool = False,
    compute_device: str = "cuda",
    feature_processor: bool = False,
) -> TestModelInfo:
    mi = create_test_model_ebc_only_no_quantize(
        num_embeddings=num_embeddings,
        emb_dim=emb_dim,
        world_size=world_size,
        batch_size=batch_size,
        dense_device=dense_device,
        sparse_device=sparse_device,
        num_features=num_features,
        num_float_features=num_float_features,
        num_weighted_features=num_weighted_features,
        compute_device=compute_device,
        feature_processor=feature_processor,
    )

    if feature_processor:
        mi.quant_model = quantize_fpebc(
            module=mi.model,
            inplace=True,
            register_tbes=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
    else:
        mi.quant_model = quantize(
            module=mi.model,
            inplace=False,
            register_tbes=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
    return mi


def shard_qebc(
    mi: TestModelInfo,
    sharding_type: ShardingType,
    device: torch.device,
    expected_shards: Optional[List[List[Tuple[Tuple[int, int, int, int], str]]]] = None,
    plan: Optional[ShardingPlan] = None,
    ebc_fqn: str = "_module.sparse.ebc",
    shard_score_ebc: bool = False,
    feature_processor: bool = False,
) -> torch.nn.Module:
    if feature_processor:
        sharder = TestQuantFPEBCSharder(
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=(
                [table.name for table in mi.tables]
                + ([table.name for table in mi.weighted_tables])
            ),
        )
    else:
        sharder = TestQuantEBCSharder(
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=(
                [table.name for table in mi.tables]
                + (
                    [table.name for table in mi.weighted_tables]
                    if shard_score_ebc
                    else []
                )
            ),
        )
    if not plan:
        # pyrefly: ignore[missing-attribute]
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

    if expected_shards is not None:
        msp = plan.plan[ebc_fqn]
        for i in range(mi.num_features):
            # pyrefly: ignore[bad-index]
            ps: ParameterSharding = msp[f"table_{i}"]
            assert ps.sharding_type == sharding_type.value
            assert ps.sharding_spec is not None
            sharding_spec: ShardingSpec = ps.sharding_spec
            # pyrefly: ignore[missing-attribute]
            assert len(sharding_spec.shards) == len(expected_shards[i])
            # pyrefly: ignore[no-matching-overload]
            for shard, ((offset_r, offset_c, size_r, size_c), placement) in zip(
                sharding_spec.shards, expected_shards[i]
            ):
                assert shard.shard_offsets == [offset_r, offset_c]
                assert shard.shard_sizes == [size_r, size_c]
                assert str(shard.placement) == placement

    # We want to leave quant_model unchanged to compare the results with it
    quant_model_copy = copy.deepcopy(mi.quant_model)
    sharded_model, _ = shard_quant_model(
        model=quant_model_copy,
        #  `Optional[List[ModuleSharder[Module]]]` but got `List[TestQuantEBCSharder]`.
        # pyrefly: ignore[bad-argument-type]
        sharders=[sharder],
        sharding_device=str(device),
        # pyrefly: ignore[missing-attribute]
        world_size=mi.topology.world_size,
        sharding_plan=plan,
    )
    return sharded_model


def shard_qec(
    mi: TestModelInfo,
    sharding_type: ShardingType,
    device: torch.device,
    expected_shards: Optional[List[List[Tuple[Tuple[int, int, int, int], str]]]],
    plan: Optional[ShardingPlan] = None,
) -> torch.nn.Module:
    sharder = TestQuantECSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
    )

    if not plan:
        # pyrefly: ignore[missing-attribute]
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

    if expected_shards is not None:
        msp: ModuleShardingPlan = plan.plan["_module_kjt_input.0"]  # TODO: hardcoded
        for i in range(mi.num_features):
            # pyrefly: ignore[bad-index]
            ps: ParameterSharding = msp[f"table_{i}"]
            assert ps.sharding_type == sharding_type.value
            assert ps.sharding_spec is not None
            sharding_spec: ShardingSpec = ps.sharding_spec
            # pyrefly: ignore[missing-attribute]
            assert len(sharding_spec.shards) == len(expected_shards[i])
            # pyrefly: ignore[no-matching-overload]
            for shard, ((offset_r, offset_c, size_r, size_c), placement) in zip(
                sharding_spec.shards, expected_shards[i]
            ):
                assert shard.shard_offsets == [offset_r, offset_c]
                assert shard.shard_sizes == [size_r, size_c]
                assert str(shard.placement) == placement

    # We want to leave quant_model unchanged to compare the results with it
    quant_model_copy = copy.deepcopy(mi.quant_model)
    sharded_model = _shard_modules(
        module=quant_model_copy,
        #  `Optional[List[ModuleSharder[Module]]]` but got `List[TestQuantECSharder]`.
        # pyrefly: ignore[bad-argument-type]
        sharders=[sharder],
        device=device,
        plan=plan,
        # pyrefly: ignore[missing-attribute]
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model


def assert_close(expected, actual) -> None:
    if isinstance(expected, KeyedTensor):
        assert isinstance(actual, KeyedTensor)
        assert len(expected.keys()) == len(actual.keys())
        torch.testing.assert_close(expected.values(), actual.values())
        torch.testing.assert_close(expected.length_per_key(), actual.length_per_key())
    elif isinstance(expected, dict):
        assert sorted(expected.keys()) == sorted(actual.keys())
        for feature, jt_e in expected.items():
            jt_got = actual[feature]
            if isinstance(jt_e, torch.Tensor) and isinstance(jt_got, torch.Tensor):
                if jt_got.device != jt_e.device:
                    jt_got = actual.to(jt_e.device)
                assert_close(jt_e, jt_got)
            else:
                assert_close(jt_e.lengths(), jt_got.lengths())
                assert_close(jt_e.values(), jt_got.values())
                assert_close(jt_e.offsets(), jt_got.offsets())
    else:
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if actual.device != expected.device:
                actual = actual.to(expected.device)

        torch.testing.assert_close(expected, actual)


def assert_weight_spec(
    weights_spec: Dict[str, WeightSpec],
    all_expected_shards: List[List[Tuple[Tuple[int, int, int, int], str]]],
    ebc_fqn: str,
    weights_prefix: str,
    all_table_names: List[str],
    sharding_type: str,
) -> None:
    tbe_table_idxs = [0, 0]
    for table_name, expected_shards in zip(all_table_names, all_expected_shards):
        unsharded_weight_fqn = f"{ebc_fqn}.{weights_prefix}.{table_name}.weight"
        for (offset_r, offset_c, size_r, size_c), placement in expected_shards:
            tbe_idx: int = 0
            # Assumption of only one TBE per rank
            if "rank:1" in placement:
                tbe_idx = 1
            sharded_weight_fqn: str = (
                f"{ebc_fqn}.tbes.{tbe_idx}.{tbe_table_idxs[tbe_idx]}.{table_name}.weight"
            )
            tbe_table_idxs[tbe_idx] += 1
            assert sharded_weight_fqn in weights_spec
            wspec = weights_spec[sharded_weight_fqn]
            assert wspec.fqn == unsharded_weight_fqn
            assert wspec.shard_sizes == [size_r, size_c]
            assert wspec.shard_offsets == [offset_r, offset_c]
            assert wspec.sharding_type == sharding_type

            for qcomp in ["qscale", "qbias"]:
                sharded_weight_qcomp_fqn: str = f"{sharded_weight_fqn}_{qcomp}"
                assert sharded_weight_qcomp_fqn in weights_spec
                wqcomp_spec = weights_spec[sharded_weight_qcomp_fqn]
                assert wqcomp_spec.fqn == f"{unsharded_weight_fqn}_{qcomp}"
                assert wqcomp_spec.shard_sizes == [size_r, 2]
                assert wqcomp_spec.shard_offsets == [offset_r, 0]
                assert wqcomp_spec.sharding_type == sharding_type


class MockTBE(nn.Module):
    def __init__(
        self,
        embedding_specs: List[Tuple[str, int, int, SparseType, EmbeddingLocation]],
        device: torch.device,
        output_dtype: int,
        pooling_mode: PoolingMode,
    ) -> None:
        super(MockTBE, self).__init__()
        self.embedding_specs: List[
            Tuple[str, int, int, SparseType, EmbeddingLocation]
        ] = embedding_specs
        self.pooling_mode = pooling_mode
        self.device = device
        self.output_dtype: torch.dtype = SparseType.from_int(output_dtype).as_dtype()
        self.D: int = max([D for _, _, D, _, _ in embedding_specs])

        self.weights: List[torch.Tensor] = [
            torch.arange(N).view(N, 1).expand(N, D) for _, N, D, _, _ in embedding_specs
        ]
        self.split_embedding_weights: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = [
            (
                torch.zeros(N, D, dtype=torch.uint8),
                torch.zeros(N, 2, dtype=torch.uint8),
                torch.zeros(N, 2, dtype=torch.uint8),
            )
            for _, N, D, _, _ in embedding_specs
        ]

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pooling_mode == PoolingMode.SUM:
            return torch.ones(1, self.D, device=self.device, dtype=self.output_dtype)

        return torch.zeros(
            indices.size(0), self.D, device=self.device, dtype=self.output_dtype
        )

    def split_embedding_weights_with_scale_bias(
        self, split_scale_bias_mode: int = 1
    ) -> List[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
        if split_scale_bias_mode == 2:
            # pyrefly: ignore[bad-return]
            return self.split_embedding_weights
        raise NotImplementedError()


def mock_tbe_from_tbe(tbe: IntNBitTableBatchedEmbeddingBagsCodegen) -> MockTBE:
    return MockTBE(
        tbe.embedding_specs,
        tbe.current_device,
        tbe.output_dtype,
        tbe.pooling_mode,
    )


def replace_registered_tbes_with_mock_tbes(M: torch.nn.Module, path: str = "") -> None:
    for child_name, child in M.named_children():
        child_path = f"{path}.{child_name}" if path else child_name
        if isinstance(child, IntNBitTableBatchedEmbeddingBagsCodegen):
            M.register_module(
                child_name,
                mock_tbe_from_tbe(child),
            )
        else:
            replace_registered_tbes_with_mock_tbes(child, child_path)


def replace_sharded_quant_modules_tbes_with_mock_tbes(M: torch.nn.Module) -> None:
    for m in M.modules():
        if isinstance(m, ShardedQuantEmbeddingBagCollection):
            for lookup in m._lookups:
                #  not a function.
                # pyrefly: ignore[not-iterable]
                for lookup_per_rank in lookup._embedding_lookups_per_rank:
                    replace_registered_tbes_with_mock_tbes(lookup_per_rank)
