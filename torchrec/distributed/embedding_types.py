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
from dataclasses import dataclass
from enum import Enum, unique
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import EmbeddingLocation
from torch import fx, nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Placement
from torch.nn.modules.module import _addindent
from torch.nn.parallel import DistributedDataParallel

from torchrec.distributed.types import (
    get_tensor_size_bytes,
    ModuleSharder,
    ParameterStorage,
    QuantizedCommCodecs,
    ShardedModule,
    ShardedTensorMetadata,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingTableConfig,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


@unique
class OptimType(Enum):
    SGD = "SGD"
    LARS_SGD = "LARS_SGD"
    LAMB = "LAMB"
    PARTIAL_ROWWISE_LAMB = "PARTIAL_ROWWISE_LAMB"
    ADAM = "ADAM"
    PARTIAL_ROWWISE_ADAM = "PARTIAL_ROWWISE_ADAM"
    ADAGRAD = "ADAGRAD"
    ROWWISE_ADAGRAD = "ROWWISE_ADAGRAD"
    SHAMPOO = "SHAMPOO"
    SHAMPOO_V2 = "SHAMPOO_V2"
    LION = "LION"
    ADAMW = "ADAMW"
    SHAMPOO_V2_MRS = "SHAMPOO_V2_MRS"
    SHAMPOO_MRS = "SHAMPOO_MRS"


@unique
class EmbeddingComputeKernel(Enum):
    DENSE = "dense"
    FUSED = "fused"
    FUSED_UVM = "fused_uvm"
    FUSED_UVM_CACHING = "fused_uvm_caching"
    QUANT = "quant"
    QUANT_UVM = "quant_uvm"
    QUANT_UVM_CACHING = "quant_uvm_caching"
    KEY_VALUE = (
        "key_value"  # ssd as kv backend storage for fully materialized embedding table
    )
    CUSTOMIZED_KERNEL = "customized_kernel"
    SSD_VIRTUAL_TABLE = (
        "ssd_virtual_table"  # ssd as kv backend storage for virtual table
    )
    DRAM_VIRTUAL_TABLE = (
        "dram_virtual_table"  # dram as kv backend storage for virtual table
    )


def compute_kernel_to_embedding_location(
    compute_kernel: EmbeddingComputeKernel,
) -> EmbeddingLocation:
    if compute_kernel in [
        EmbeddingComputeKernel.DENSE,
        EmbeddingComputeKernel.FUSED,
        EmbeddingComputeKernel.QUANT,
        EmbeddingComputeKernel.KEY_VALUE,  # use hbm for cache
        EmbeddingComputeKernel.SSD_VIRTUAL_TABLE,  # use hbm for cache
        EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE,  # use hbm for cache
    ]:
        return EmbeddingLocation.DEVICE
    elif compute_kernel in [
        EmbeddingComputeKernel.FUSED_UVM,
        EmbeddingComputeKernel.QUANT_UVM,
    ]:
        return EmbeddingLocation.MANAGED
    elif compute_kernel in [
        EmbeddingComputeKernel.FUSED_UVM_CACHING,
        EmbeddingComputeKernel.QUANT_UVM_CACHING,
    ]:
        return EmbeddingLocation.MANAGED_CACHING
    else:
        raise ValueError(f"Invalid EmbeddingComputeKernel {compute_kernel}")


class KJTList(Multistreamable):
    def __init__(self, features: List[KeyedJaggedTensor]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __setitem__(self, key: int, item: KeyedJaggedTensor) -> None:
        self.features[key] = item

    def __getitem__(self, key: int) -> KeyedJaggedTensor:
        return self.features[key]

    @torch.jit._drop
    def __iter__(self) -> Iterator[KeyedJaggedTensor]:
        return iter(self.features)

    @torch.jit._drop
    # pyre-fixme[14]: `record_stream` overrides method defined in `Multistreamable`
    #  inconsistently.
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for feature in self.features:
            feature.record_stream(stream)

    @torch.jit._drop
    def __fx_create_arg__(self, tracer: torch.fx.Tracer) -> fx.node.Argument:
        return tracer.create_node(
            "call_function",
            KJTList,
            args=(tracer.create_arg(self.features),),
            kwargs={},
        )


@dataclass
class InputDistOutputs(Multistreamable):
    features: KJTList
    unbucketize_permute_tensor: Optional[torch.Tensor] = (
        None  # only used in RW sharding
    )
    bucket_mapping_tensor: Optional[torch.Tensor] = None  # only used in RW sharding
    bucketized_length: Optional[torch.Tensor] = None  # only used in RW sharding

    def record_stream(self, stream: torch.Stream) -> None:
        for feature in self.features:
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            feature.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.bucket_mapping_tensor is not None:
            self.bucket_mapping_tensor.record_stream(stream)
        if self.bucketized_length is not None:
            self.bucketized_length.record_stream(stream)


class ListOfKJTList(Multistreamable):
    def __init__(self, features: List[KJTList]) -> None:
        self.features_list = features

    def __len__(self) -> int:
        return len(self.features_list)

    def __setitem__(self, key: int, item: KJTList) -> None:
        self.features_list[key] = item

    def __getitem__(self, key: int) -> KJTList:
        return self.features_list[key]

    @torch.jit._drop
    def __iter__(self) -> Iterator[KJTList]:
        return iter(self.features_list)

    @torch.jit._drop
    def record_stream(self, stream: torch.Stream) -> None:
        for feature in self.features_list:
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            feature.record_stream(stream)

    @torch.jit._drop
    def __fx_create_arg__(self, tracer: torch.fx.Tracer) -> fx.node.Argument:
        return tracer.create_node(
            "call_function",
            ListOfKJTList,
            args=(tracer.create_arg(self.features_list),),
            kwargs={},
        )


@dataclass
class ShardedConfig:
    local_rows: int = 0
    local_cols: int = 0


@dataclass
class DTensorMetadata:
    mesh: Optional[DeviceMesh] = None
    placements: Optional[Tuple[Placement, ...]] = None
    size: Optional[Tuple[int, ...]] = None
    stride: Optional[Tuple[int, ...]] = None


@dataclass
class ShardedMetaConfig(ShardedConfig):
    local_metadata: Optional[ShardMetadata] = None
    global_metadata: Optional[ShardedTensorMetadata] = None
    dtensor_metadata: Optional[DTensorMetadata] = None


@dataclass
class EmbeddingAttributes:
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.DENSE


@dataclass
class ShardedEmbeddingTable(
    ShardedMetaConfig,
    EmbeddingAttributes,
    EmbeddingTableConfig,
):
    fused_params: Optional[Dict[str, Any]] = None


@dataclass
class GroupedEmbeddingConfig:
    data_type: DataType
    pooling: PoolingType
    is_weighted: bool
    has_feature_processor: bool
    compute_kernel: EmbeddingComputeKernel
    embedding_tables: List[ShardedEmbeddingTable]
    fused_params: Optional[Dict[str, Any]] = None
    # Write-enabled Embedding Tables cannot be grouped with read-only Embedding Tables TBE needs to be separate.
    enable_embedding_update: bool = False

    def feature_hash_sizes(self) -> List[int]:
        feature_hash_sizes = []
        for table in self.embedding_tables:
            feature_hash_sizes.extend(table.num_features() * [table.num_embeddings])
        return feature_hash_sizes

    def num_features(self) -> int:
        num_features = 0
        for table in self.embedding_tables:
            num_features += table.num_features()
        return num_features

    def dim_sum(self) -> int:
        dim_sum = 0
        for table in self.embedding_tables:
            dim_sum += table.num_features() * table.local_cols
        return dim_sum

    def table_names(self) -> List[str]:
        table_names = []
        for table in self.embedding_tables:
            table_names.append(table.name)
        return table_names

    def feature_names(self) -> List[str]:
        feature_names = []
        for table in self.embedding_tables:
            feature_names.extend(table.feature_names)
        return feature_names

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for table in self.embedding_tables:
            embedding_dims.extend([table.local_cols] * table.num_features())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for table in self.embedding_tables:
            embedding_names.extend(table.embedding_names)
        return embedding_names

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata: List[Optional[ShardMetadata]] = []
        for table in self.embedding_tables:
            for _ in table.feature_names:
                embedding_shard_metadata.append(table.local_metadata)
        return embedding_shard_metadata

    def is_using_virtual_table(self) -> bool:
        return any(table.use_virtual_table for table in self.embedding_tables)


F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")


class BaseEmbeddingLookup(abc.ABC, nn.Module, Generic[F, T]):
    """
    Interface implemented by different embedding implementations:
    e.g. one, which relies on `nn.EmbeddingBag` or table-batched one, etc.
    """

    @abc.abstractmethod
    def forward(
        self,
        sparse_features: F,
    ) -> T:
        pass


class BaseEmbeddingUpdate(abc.ABC, nn.Module, Generic[F]):
    """
    Interface implemented by different embedding implementations for updating the weights
    """

    @abc.abstractmethod
    def forward(
        self,
        embeddings: F,
    ) -> None:
        pass


class FeatureShardingMixIn:
    """
    Feature Sharding Interface to provide sharding-aware feature metadata.
    """

    def feature_names(self) -> List[str]:
        raise NotImplementedError

    def feature_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def features_per_rank(self) -> List[int]:
        raise NotImplementedError


class ModuleShardingMixIn:
    """
    The interface to access a sharded module's sharding scheme.
    """

    @property
    def shardings(self) -> Dict[str, FeatureShardingMixIn]:
        raise NotImplementedError


Out = TypeVar("Out")
CompIn = TypeVar("CompIn", KJTList, ListOfKJTList, KeyedJaggedTensor)
DistOut = TypeVar("DistOut")
ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


class ShardedEmbeddingModule(
    ShardedModule[CompIn, DistOut, Out, ShrdCtx],
    ModuleShardingMixIn,
):
    """
    All model-parallel embedding modules implement this interface.
    Inputs and outputs are data-parallel.

    Args::
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]) : Mapping of CommOp name to QuantizedCommCodecs
    """

    @abc.abstractmethod
    def __init__(
        self, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    ) -> None:
        super().__init__(qcomm_codecs_registry)

        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._output_dists: List[nn.Module] = []
        self.post_lookup_tracker_fn: Optional[
            Callable[[KeyedJaggedTensor, torch.Tensor], None]
        ] = None
        self.post_odist_tracker_fn: Optional[Callable[..., None]] = None

    def prefetch(
        self,
        dist_input: CompIn,
        forward_stream: Optional[Union[torch.cuda.Stream, torch.mtia.Stream]] = None,
        ctx: Optional[ShrdCtx] = None,
    ) -> None:
        """
        Prefetch input features for each lookup module.
        """
        assert isinstance(dist_input, Iterable)
        for feature, emb_lookup in zip(dist_input, self._lookups):
            while isinstance(emb_lookup, DistributedDataParallel):
                emb_lookup = emb_lookup.module
            emb_lookup.prefetch(sparse_features=feature, forward_stream=forward_stream)

    def extra_repr(self) -> str:
        """
        Pretty prints representation of the module's lookup modules, input_dists and output_dists
        """

        def loop(key: str, modules: List[nn.Module]) -> List[str]:
            child_lines = []
            if len(modules) > 0:
                child_lines.append("(" + key + "): ")
            for module in modules:
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append(mod_str)
            return child_lines

        rep = []
        rep.extend(loop("lookups", self._lookups))
        rep.extend(loop("_input_dists", self._input_dists))
        rep.extend(loop("_output_dists", self._output_dists))

        return "\n ".join(rep)

    def train(self, mode: bool = True):  # pyre-ignore[3]
        r"""Set the module in training mode."""
        super().train(mode)

        # adding additional handling for lookups
        for lookup in self._lookups:
            lookup.train(mode)

        return self

    def register_post_lookup_tracker_fn(
        self,
        record_fn: Callable[[KeyedJaggedTensor, torch.Tensor], None],
    ) -> None:
        """
        Register a function to be called after lookup is done. This is used for
        tracking the lookup results and optimizer states.

        Args:
            record_fn (Callable[[KeyedJaggedTensor, torch.Tensor], None]): A custom record function to be called after lookup is done.

        """
        if self.post_lookup_tracker_fn is not None:
            logger.warning(
                "[ModelDeltaTracker] Custom record function already defined, overriding with new callable"
            )
        self.post_lookup_tracker_fn = record_fn

    def register_post_odist_tracker_fn(
        self,
        record_fn: Callable[..., None],
    ) -> None:
        """
        Register a function to be called after registering odist awaitable.

        Args:
            record_fn (Callable[Callable[..., None]):

        """
        if self.post_odist_tracker_fn is not None:
            logger.warning(
                "[ModelDeltaTracker] Compaction function already defined, overriding with new callable"
            )
        self.post_odist_tracker_fn = record_fn

    @property
    def unsharded_module_type(self) -> Type[nn.Module]:
        """
        As this is the generic ShardedEmbeddingModule class, simply
        return the generic nn.Module type. In the inherited classes of
        ShardedEmbeddingModule, the specific unsharded module type will
        be returned.
        """
        return nn.Module


M = TypeVar("M", bound=nn.Module)


class BaseEmbeddingSharder(ModuleSharder[M]):
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)

        # TODO remove after decoupling
        self._fused_params = fused_params

    def sharding_types(self, compute_device_type: str) -> List[str]:

        if compute_device_type in {"mtia"}:
            return [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.GRID_SHARD.value,
            ]

        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]
        if compute_device_type in {"cuda"}:
            types += [
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.GRID_SHARD.value,
            ]

        return types

    def compute_kernels(
        # TODO remove after decoupling
        self,
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        ret = [
            EmbeddingComputeKernel.DENSE.value,
        ]
        if sharding_type != ShardingType.DATA_PARALLEL.value:
            ret += [
                EmbeddingComputeKernel.FUSED.value,
            ]
            if compute_device_type in {"cuda"}:
                ret += [
                    EmbeddingComputeKernel.FUSED_UVM.value,
                    EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                    EmbeddingComputeKernel.KEY_VALUE.value,
                    EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
                    EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
                ]
        return ret

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        """
        List of system resources and corresponding usage given a compute device and
        compute kernel
        """
        tensor_bytes = get_tensor_size_bytes(tensor)
        if compute_kernel in {
            EmbeddingComputeKernel.FUSED_UVM.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        }:
            assert compute_device_type in {"cuda"}
            return {ParameterStorage.DDR.value: tensor_bytes}
        else:
            assert compute_device_type in {"cuda", "cpu", "mtia"}
            storage_map = {
                "cuda": ParameterStorage.HBM,
                "cpu": ParameterStorage.DDR,
                "mtia": ParameterStorage.HBM,
            }
            return {
                storage_map[compute_device_type].value: get_tensor_size_bytes(tensor)
            }


class BaseGroupedFeatureProcessor(nn.Module):
    """
    Abstract base class for grouped feature processor
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        pass


class BaseQuantEmbeddingSharder(ModuleSharder[M]):
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._fused_params: Optional[Dict[str, Any]] = (
            copy.deepcopy(fused_params) if fused_params is not None else fused_params
        )
        if not shardable_params:
            shardable_params = []
        self._shardable_params: List[str] = shardable_params

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ]

        return types

    def shardable_parameters(self, module: M) -> Dict[str, nn.Parameter]:

        shardable_params: Dict[str, nn.Parameter] = {}
        for name, param in module.state_dict().items():
            if name.endswith(".weight"):
                table_name = name.split(".")[-2]
                shardable_params[table_name] = param

        if self._shardable_params:
            assert all(
                table_name in self._shardable_params
                for table_name in shardable_params.keys()
            ) or all(
                table_name not in self._shardable_params
                for table_name in shardable_params.keys()
            ), f"Cannot partially shard {type(module)}, please check sharder kwargs"
            shardable_params = {
                table_name: param
                for table_name, param in shardable_params.items()
                if table_name in self._shardable_params
            }

        return shardable_params

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        ret = [
            EmbeddingComputeKernel.QUANT.value,
        ]
        if compute_device_type in {"cuda"}:
            ret += [
                EmbeddingComputeKernel.QUANT_UVM.value,
                EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
            ]
        return ret

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        """
        List of system resources and corresponding usage given a compute device and
        compute kernel
        """
        tensor_bytes = get_tensor_size_bytes(tensor) + tensor.shape[0] * 4
        if compute_kernel in {
            EmbeddingComputeKernel.QUANT_UVM.value,
            EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
        }:
            assert compute_device_type in {"cuda"}
            return {ParameterStorage.DDR.value: tensor_bytes}
        else:
            assert compute_device_type in {"cuda", "cpu", "mtia"}
            storage_map = {
                "cuda": ParameterStorage.HBM,
                "cpu": ParameterStorage.DDR,
                # TODO: Update it later. Setting for MTIA is same as CPU's for now.
                "mtia": ParameterStorage.DDR,
            }
            return {storage_map[compute_device_type].value: tensor_bytes}
