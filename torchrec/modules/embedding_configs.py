#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from enum import Enum, unique
from functools import partial
from math import sqrt
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import PoolingMode
from torchrec.types import DataType


@unique
class PoolingType(Enum):
    """
    Pooling type for embedding table.

    Args:
        SUM (str): sum pooling.
        MEAN (str): mean pooling.
        NONE (str): no pooling.
    """

    SUM = "SUM"
    MEAN = "MEAN"
    NONE = "NONE"


# TODO - duplicated, move elsewhere to remove circular dependencies
class ShardingType(Enum):
    """
    Well-known sharding types, used by inter-module optimizations.
    """

    # Replicated on all ranks
    DATA_PARALLEL = "data_parallel"
    # Placed on a single rank
    TABLE_WISE = "table_wise"
    # Placed on multiple ranks as different sharded tables
    COLUMN_WISE = "column_wise"
    # Range-split on the first dimension across all ranks
    ROW_WISE = "row_wise"
    # Row-wise on the same node and table-wise across nodes
    # Useful when having multiple ranks per node
    # and comms within a single node are more efficient than across nodes.
    TABLE_ROW_WISE = "table_row_wise"
    # Column-wise on the same node and table-wise across nodes
    TABLE_COLUMN_WISE = "table_column_wise"


DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
    DataType.FP32: 32,
    DataType.FP16: 16,
    DataType.BF16: 16,
    DataType.INT8: 8,
    DataType.UINT8: 8,
    DataType.NFP8: 8,
    DataType.INT4: 4,
    DataType.INT2: 2,
}


def dtype_to_data_type(dtype: torch.dtype) -> DataType:
    if dtype == torch.float:
        return DataType.FP32
    elif dtype == torch.float16 or dtype == torch.half:
        return DataType.FP16
    elif dtype == torch.bfloat16:
        return DataType.BF16
    elif dtype in {torch.int, torch.int32}:
        return DataType.INT32
    elif dtype in {torch.long, torch.int64}:
        return DataType.INT64
    elif dtype in {torch.quint8, torch.qint8, torch.int8}:
        return DataType.INT8
    elif dtype == torch.uint8:
        return DataType.UINT8
    elif dtype == torch.quint4x2:
        return DataType.INT4
    elif dtype == torch.quint2x4:
        return DataType.INT2
    elif dtype == torch.float8_e4m3fn:
        return DataType.NFP8
    else:
        raise Exception(f"Invalid data type {dtype}")


def pooling_type_to_pooling_mode(
    pooling_type: PoolingType, sharding_type: Optional[ShardingType] = None
) -> PoolingMode:
    if pooling_type.value == PoolingType.SUM.value:
        return PoolingMode.SUM
    elif pooling_type.value == PoolingType.MEAN.value:
        if sharding_type is not None and sharding_type.value in [
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.ROW_WISE.value,
        ]:
            # Mean pooling is not supported in TBE for TWRW/RW sharding.
            # Pass 'SUM' as a workaround, and apply mean pooling as a callback in EBC.
            return PoolingMode.SUM
        return PoolingMode.MEAN
    elif pooling_type.value == PoolingType.NONE.value:
        return PoolingMode.NONE
    else:
        raise Exception(f"Invalid pooling type {pooling_type}")


def pooling_type_to_str(pooling_type: PoolingType) -> str:
    if pooling_type.value == PoolingType.SUM.value:
        return "sum"
    elif pooling_type.value == PoolingType.MEAN.value:
        return "mean"
    else:
        raise ValueError(f"Unsupported pooling type {pooling_type}")


def data_type_to_sparse_type(data_type: DataType) -> SparseType:
    if data_type == DataType.FP32:
        return SparseType.FP32
    elif data_type == DataType.FP16:
        return SparseType.FP16
    elif data_type == DataType.BF16:
        return SparseType.BF16
    elif data_type == DataType.INT8 or data_type == DataType.UINT8:
        return SparseType.INT8
    elif data_type == DataType.INT4:
        return SparseType.INT4
    elif data_type == DataType.INT2:
        return SparseType.INT2
    elif data_type == DataType.NFP8:
        return SparseType.NFP8
    else:
        raise ValueError(f"Invalid DataType {data_type}")


def data_type_to_dtype(data_type: DataType) -> torch.dtype:
    if data_type.value == DataType.FP32.value:
        return torch.float32
    elif data_type.value == DataType.FP16.value:
        return torch.float16
    elif data_type.value == DataType.BF16.value:
        return torch.bfloat16
    elif data_type.value == DataType.NFP8.value:
        return torch.float8_e4m3fn
    elif data_type.value == DataType.INT64.value:
        return torch.int64
    elif data_type.value == DataType.INT32.value:
        return torch.int32
    elif data_type.value == DataType.INT8.value:
        return torch.int8
    elif data_type.value == DataType.UINT8.value:
        return torch.uint8
    elif data_type.value == DataType.INT4.value:
        return torch.quint4x2
    elif data_type.value == DataType.INT2.value:
        return torch.quint2x4
    else:
        raise ValueError(f"DataType {data_type} cannot be converted to dtype")


@dataclass
class VirtualTableEvictionPolicy:
    # metadata header length in element size for virtual table in weight tensor value
    meta_header_len: int = 0
    embedding_dim: int = 0
    initialized: bool = False

    """
    Eviction policy for virtual table.
    """

    def init_metaheader_config(self, data_type: DataType, embedding_dim: int) -> None:
        # the eviction metaheader is set for training data type only. Once initialized, we don't need to reinitialize again
        if self.initialized:
            return
        # 8 bytes for key, 4 bytes timestamp, 4 bytes shared by used and count: 1 bit for used, 31 bits for count
        # for more details, please refer to: https://github.com/pytorch/FBGEMM/pull/4187
        self.meta_header_len = 16 // data_type_to_dtype(data_type).itemsize
        self.embedding_dim = embedding_dim
        self.initialized = True

    def get_meta_header_len(self) -> int:
        return self.meta_header_len

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


@dataclass
class CountBasedEvictionPolicy(VirtualTableEvictionPolicy):
    """
    Count based eviction policy for virtual table.
    """

    training_id_eviction_trigger_count: int = (
        0  # max number of training ids per rank to trigger eviction
    )
    eviction_threshold: int = (
        15  # eviction threshold for count based eviction policy. 0 means no eviction
    )
    decay_rate: float = 0.99  # default decay by default
    inference_eviction_threshold: Optional[int] = (
        None  # eviction threshold for inference count based eviction policy. 0 means no eviction
    )

    def __post_init__(self) -> None:
        if self.inference_eviction_threshold is None:
            self.inference_eviction_threshold = self.eviction_threshold


@dataclass
class FeatureScoreBasedEvictionPolicy(VirtualTableEvictionPolicy):
    """
    Feature score based eviction policy for virtual table.
    """

    decay_rate: float = 0.99  # default decay by default #TODO: Change to real value
    training_id_eviction_trigger_count: int = (
        0  # max number of training ids per rank to trigger eviction
    )
    training_id_keep_count: int = (
        0  # number of training ids per rank to keep after eviction
    )
    eviction_ttl_mins: int = (
        0  # if not 0, means we will use timestamp based policy but not feature score policy
    )
    max_inference_id_num_per_rank: int = (
        0  # max number of inference ids per rank, default is training_id_keep_count
    )
    inference_eviction_feature_score_threshold: Optional[float] = (
        None  # 0 means no eviction
    )
    inference_eviction_ttl_mins: Optional[int] = None  # 0 means no eviction

    def __post_init__(self) -> None:
        if self.inference_eviction_feature_score_threshold is None:
            self.inference_eviction_feature_score_threshold = 0
        if self.inference_eviction_ttl_mins is None:
            self.inference_eviction_ttl_mins = self.eviction_ttl_mins
        if self.max_inference_id_num_per_rank == 0:
            self.max_inference_id_num_per_rank = self.training_id_keep_count


@dataclass
class TimestampBasedEvictionPolicy(VirtualTableEvictionPolicy):
    """
    Timestamp based eviction policy for virtual table.
    """

    training_id_eviction_trigger_count: int = (
        0  # max number of training ids per rank to trigger eviction
    )
    eviction_ttl_mins: int = 24 * 60  # 1 day. 0 means no eviction
    inference_eviction_ttl_mins: Optional[int] = None  # 0 means no eviction

    def __post_init__(self) -> None:
        if self.inference_eviction_ttl_mins is None:
            self.inference_eviction_ttl_mins = self.eviction_ttl_mins


@dataclass
class CountTimestampMixedEvictionPolicy(VirtualTableEvictionPolicy):
    """
    Count timestamp mixed eviction policy for virtual table.
    """

    training_id_eviction_trigger_count: int = (
        0  # max number of training ids per rank to trigger eviction
    )
    eviction_threshold: int = (
        15  # eviction threshold for count based eviction policy. 0 means no eviction based on count
    )
    decay_rate: float = 0.99  # default decay by default
    eviction_ttl_mins: int = 24 * 60  # 1 day. 0 means no eviction based on timestamp
    inference_eviction_threshold: Optional[int] = (
        None  # eviction threshold for inference count based eviction policy. 0 means no eviction based on count
    )

    inference_eviction_ttl_mins: Optional[int] = (
        None  # 0 means no eviction based on timestamp
    )

    def __post_init__(self) -> None:
        if self.inference_eviction_ttl_mins is None:
            self.inference_eviction_ttl_mins = self.eviction_ttl_mins

        if self.inference_eviction_threshold is None:
            self.inference_eviction_threshold = self.eviction_threshold


@dataclass
class FeatureL2NormBasedEvictionPolicy(VirtualTableEvictionPolicy):
    """
    Feature L2 norm based eviction policy for virtual table.
    """

    training_id_eviction_trigger_count: int = (
        0  # max number of training ids per rank to trigger eviction
    )
    eviction_threshold: float = (
        0.0  # eviction threshold for feature l2 norm based eviction policy. 0.0 means no eviction
    )
    inference_eviction_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.inference_eviction_threshold is None:
            self.inference_eviction_threshold = self.eviction_threshold


@dataclass
class NoEvictionPolicy(VirtualTableEvictionPolicy):
    """
    No eviction policy for virtual table.
    """

    pass


@dataclass
class BaseEmbeddingConfig:
    """
    Base class for embedding configs.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): embedding dimension.
        name (str): name of the embedding table.
        data_type (DataType): data type of the embedding table.
        feature_names (List[str]): list of feature names.
        weight_init_max (Optional[float]): max value for weight initialization.
        weight_init_min (Optional[float]): min value for weight initialization.
        num_embeddings_post_pruning (Optional[int]): number of embeddings after pruning for inference.
            If None, no pruning is applied.
        init_fn (Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]]): init function for embedding weights.
        need_pos (bool): whether table is position weighted.

        total_num_buckets (Optional[int]): number of bucket globally, unchanged through model lifetime
        use_virtual_table (bool): indicator of whether table uses virtual space(magnitude like 2^50)
            for number embedding memory for virtual table is dynamic and only materialized when
            id is trained this needs to be paired with SSD/DRAM Virtual talbe in EmbeddingComputeKernel
        virtual_table_eviction_policy (Optional[VirtualTableEvictionPolicy]): eviction policy for virtual table.
    """

    num_embeddings: int
    embedding_dim: int
    name: str = ""
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    num_embeddings_post_pruning: Optional[int] = None

    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    # when the position_weighted feature is in this table config,
    # enable this flag to support rw_sharding
    need_pos: bool = False

    # handle the special case
    input_dim: Optional[int] = None
    total_num_buckets: Optional[int] = None
    use_virtual_table: bool = False
    virtual_table_eviction_policy: Optional[VirtualTableEvictionPolicy] = None
    enable_embedding_update: bool = False

    def get_weight_init_max(self) -> float:
        if self.weight_init_max is None:
            return sqrt(1 / self.num_embeddings)
        else:
            return self.weight_init_max

    def get_weight_init_min(self) -> float:
        if self.weight_init_min is None:
            return -sqrt(1 / self.num_embeddings)
        else:
            return self.weight_init_min

    def num_features(self) -> int:
        return len(self.feature_names)

    def __post_init__(self) -> None:
        if self.init_fn is None:
            self.init_fn = partial(
                torch.nn.init.uniform_,
                a=self.get_weight_init_min(),
                b=self.get_weight_init_max(),
            )


# this class will be deprecated after migration
# and all the following code in sharding itself
# which contains has_feature_processor
@dataclass
class EmbeddingTableConfig(BaseEmbeddingConfig):
    pooling: PoolingType = PoolingType.SUM
    is_weighted: bool = False
    has_feature_processor: bool = False
    embedding_names: List[str] = field(default_factory=list)


@dataclass
class EmbeddingBagConfig(BaseEmbeddingConfig):
    """
    EmbeddingBagConfig is a dataclass that represents a single embedding table,
    where outputs are meant to be pooled.

    Args:
        pooling (PoolingType): pooling type.
    """

    pooling: PoolingType = PoolingType.SUM


@dataclass
class EmbeddingConfig(BaseEmbeddingConfig):
    """
    EmbeddingConfig is a dataclass that represents a single embedding table.

    """

    pass


class QuantConfig(NamedTuple):
    activation: torch.quantization.PlaceholderObserver
    weight: torch.quantization.PlaceholderObserver
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None
