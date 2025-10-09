#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import operator
from dataclasses import asdict, dataclass, field
from enum import Enum, unique
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from fbgemm_gpu.runtime_monitor import TBEStatsReporterConfig
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    MultiPassPrefetchConfig,
)

from torch.autograd.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import _get_pg_default_device
from torchrec.tensor_types import UInt2Tensor, UInt4Tensor
from torchrec.types import DataType, ModuleNoCopyMixin

try:
    # For python 3.6 and below, GenericMeta will be used by
    # other metaclasses (i.e. AwaitableMeta) for customized
    # behaviors, as Generic is non-trival metaclass in
    # python 3.6 and below
    from typing import GenericMeta
except ImportError:
    # In python 3.7+, GenericMeta doesn't exist as it's no
    # longer a non-trival metaclass,
    #   see https://www.python.org/dev/peps/pep-0560/
    # So we make a fake type here in order to share the same
    # code with python 3.6 or below, it will just be used as
    # a placeholder for customized metaclass behaviors
    # (i.e. Awaitable)
    class GenericMeta(type):
        pass


import torch
import torch.distributed as dist
import torch.fx
from torch import nn

# @manual
from torch.distributed._shard.sharded_tensor import (  # noqa
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)

# @manual
from torch.distributed._shard.sharding_spec import (  # noqa
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torchrec.streamable import Multistreamable

_T = TypeVar("_T")


def _tabulate(
    table: List[List[Union[str, int]]], headers: Optional[List[str]] = None
) -> str:
    """
    Format a table as a string.
    Parameters:
        table (list of lists or list of tuples): The data to be formatted as a table.
        headers (list of strings, optional): The column headers for the table. If not provided, the first row of the table will be used as the headers.
    Returns:
        str: A string representation of the table.
    """
    if headers is None:
        headers = table[0]
        table = table[1:]
    headers = cast(List[str], headers)
    rows = []
    # Determine the maximum width of each column
    col_widths = [max([len(str(item)) for item in column]) for column in zip(*table)]
    col_widths = [max(i, len(j)) for i, j in zip(col_widths, headers)]
    # Format each row of the table
    for row in table:
        row_str = " | ".join(
            [str(item).ljust(width) for item, width in zip(row, col_widths)]
        )
        rows.append(row_str)
    # Add the header row and the separator line
    rows.insert(
        0,
        " | ".join(
            [header.center(width) for header, width in zip(headers, col_widths)]
        ),
    )
    rows.insert(1, " | ".join(["-" * width for width in col_widths]))
    return "\n".join(rows)


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
    # Grid sharding, where each rank gets a subset of columns and rows in a CW and TWRW style
    GRID_SHARD = "grid_shard"


class EmbeddingEvent(Enum):
    """
    Events in sharded embedding module's forward, used for trace annotations
    """

    KJT_SPLITS_DIST = "splits_dist"
    KJT_TENSORS_DIST = "tensors_dist"
    LOOKUP = "lookup"
    OUTPUT_DIST = "output_dist"
    # When .wait() is called on output_dist awaitable
    # Useful for linking backward comms event in trace to forward event
    OUTPUT_DIST_WAIT = "output_dist_wait"


class PipelineType(Enum):
    """
    Known pipeline types.
    Check out //torchrec/distributed/train_pipeline/train_pipelines.py
    for details about pipelines.
    """

    NONE = "none"
    TRAIN_BASE = "train_base"
    TRAIN_SPARSE_DIST = "train_sparse_dist"
    TRAIN_PREFETCH_SPARSE_DIST = "train_prefetch_sparse_dist"


class ParameterStorage(Enum):
    """
    Well-known physical resources, which can be used as constraints by ShardingPlanner.
    """

    # GPU-attached memory
    HBM = "hbm"
    # CPU-attached memory
    DDR = "ddr"


@unique
class ComputeKernel(Enum):
    DEFAULT = "default"


QuantizationContext = TypeVar("QuantizationContext")


# Once we only support python3.8+ use
# from typing import protocol.
# We can't use from typing_extensions import Protocol due to torch.deploy restrictions.
class QuantizedCommCodec(Generic[QuantizationContext]):
    """
    Provide an implementation to quantized, or apply mixed precision, to the tensors used in collective calls (pooled_all_to_all, reduce_scatter, etc).
    The dtype is the dtype of the tensor called from encode.

    This makes the assumption that the input tensor has type torch.float32

    >>>
        quantized_tensor = quantized_comm_codec.encode(input_tensor)
        quantized_tensor.dtype == quantized_comm_codec.quantized_dtype
        collective_call(output_tensors, input_tensors=tensor)
        output_tensor = decode(output_tensors)

        torch.assert_close(input_tensors, output_tensor)

    """

    def encode(
        self, input_tensor: torch.Tensor, ctx: Optional[QuantizationContext] = None
    ) -> torch.Tensor: ...

    def decode(
        self, input_grad: torch.Tensor, ctx: Optional[QuantizationContext] = None
    ) -> torch.Tensor: ...

    @property
    def quantized_dtype(self) -> torch.dtype:
        """
        tensor.dtype of the resultant encode(input_tensor)
        """
        ...

    def calc_quantized_size(
        self,
        input_len: int,
        ctx: Optional[QuantizationContext] = None,
    ) -> int:
        """
        Given the length of input tensor, returns the length of tensor after
        quantization. Used by INT8 codecs where the quantized tensor have
        some additional parameters. For other cases, the quantized tensor should
        have the same length with input.
        """
        ...

    def create_context(self) -> Optional[QuantizationContext]:
        """
        Create a context object that can be used to carry session-based
        parameters between encoder and decoder.
        """
        ...

    def padded_size(
        self,
        input_tensor: torch.Tensor,
        dim_per_rank: List[int],
        my_rank: int,
        qcomm_ctx: QuantizationContext,
    ) -> Tuple[int, int]:
        """
        Return (padded_dim_sum, padding_size) of the input tensor for quantization.
        """
        ...


class NoOpQuantizedCommCodec(Generic[QuantizationContext]):
    """
    Default No-Op implementation of QuantizedCommCodec
    """

    def encode(
        self,
        input_tensor: torch.Tensor,
        ctx: Optional[QuantizationContext] = None,
    ) -> torch.Tensor:
        return input_tensor

    def decode(
        self,
        input_grad: torch.Tensor,
        ctx: Optional[QuantizationContext] = None,
    ) -> torch.Tensor:
        return input_grad

    def quantized_dtype(self) -> torch.dtype:
        return torch.float

    def calc_quantized_size(
        self,
        input_len: int,
        ctx: Optional[QuantizationContext] = None,
    ) -> int:
        return input_len

    def create_context(self) -> Optional[QuantizationContext]:
        return None

    def padded_size(
        self,
        input_tensor: torch.Tensor,
        dim_per_rank: List[int],
        my_rank: int,
        qcomm_ctx: QuantizationContext,
    ) -> Tuple[int, int]:
        dim_sum = (
            input_tensor.shape[0] if input_tensor.dim() == 1 else input_tensor.shape[1]
        )
        return dim_sum, 0


@dataclass
class QuantizedCommCodecs:
    """
    The quantization codecs to use for the forward and backward pass respectively of a comm op (e.g. pooled_all_to_all, reduce_scatter, sequence_all_to_all).
    """

    # pyre-ignore
    forward: QuantizedCommCodec = NoOpQuantizedCommCodec()
    # pyre-ignore
    backward: QuantizedCommCodec = NoOpQuantizedCommCodec()


class CommOp(Enum):
    # For detailed descriptions of each of these, see their doc strings in dist_data.
    # These are commonly used inside of a QuantizedCommsRegistry
    POOLED_EMBEDDINGS_ALL_TO_ALL = "pooled_embeddings_all_to_all"
    POOLED_EMBEDDINGS_REDUCE_SCATTER = "pooled_embeddings_reduce_scatter"
    SEQUENCE_EMBEDDINGS_ALL_TO_ALL = "sequence_embeddings_all_to_all"


W = TypeVar("W")
M = TypeVar("M", bound=nn.Module)


class Awaitable(abc.ABC, Generic[W]):
    def __init__(self) -> None:
        self._callbacks: List[Callable[[W], W]] = []

    @abc.abstractmethod
    def _wait_impl(self) -> W:
        pass

    def wait(self) -> W:
        with record_function(f"## {self.__class__.__name__} wait() ##"):
            ret: W = self._wait_impl()
            for callback in self.callbacks:
                ret = callback(ret)
        return ret

    @property
    def callbacks(self) -> List[Callable[[W], W]]:
        return self._callbacks


class NoWait(Awaitable[W]):
    def __init__(self, obj: W) -> None:
        super().__init__()
        self._obj = obj

    def _wait_impl(self) -> W:
        return self._obj


class _LazyAwaitableMeta(
    GenericMeta, abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta
):
    """
    The _LazyAwaitableMeta class that inherits both ABCMeta and ProxyableClassMeta
    This is because ABCMeta/ProxyableClassMeta are both non-trival metaclasses
    Declaring this separately to ensure the init order of metaclasses

    XXX: Generics are non-trival metaclass before python 3.7 but are removed
    afterwards. we add GenericsMeta here to support version before 3.7.
    """

    pass


class LazyAwaitable(Awaitable[W], metaclass=_LazyAwaitableMeta):
    """
    The LazyAwaitable type which exposes a `wait()` API, concrete types
    can control how to initialize and how the `wait()` behavior should
    be in order to achieve specific async operation.

    This base LazyAwaitable type is a "lazy" async type, which means it will
    delay `wait()` as late as possible, see details in `__torch_function__`
    below. This could help the model automatically enable computation and
    communication overlap, model author doesn't need to manually call
    `wait()` if the results is used by a pytorch function, or by other python
    operations (NOTE: need to implement corresponding magic methods
    like __getattr__ below)

    Some caveats:

    * This works with Pytorch functions, but not any generic method, if
      you would like to do arbitrary python operations, you need to
      implement the corresponding magic methods

    * In the case that one function have two or more arguments are LazyAwaitable,
      the lazy wait mechanism can't ensure perfect computation/communication
      overlap (i.e. quickly waited the first one but long wait on the second)
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        # _result is used to cache the results after the wait() is called.
        self._result: Optional[W] = None

    @staticmethod
    # pyre-ignore [2, 3]
    def _wait_async(obj: Any) -> Any:
        """
        This method is used internally to automatically wait when necessary
        and cache the results of the `LazyAwaitable.wait()` call
        """
        if isinstance(obj, LazyAwaitable):
            if obj._result is None:
                obj._result = obj.wait()
            return obj._result
        else:
            return obj

    @classmethod
    # pyre-ignore [2, 3]
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        The LazyAwaitable type has a `__torch_function__` implementation.
        This means when this type is seens as an argument to a PyTorch
        function in a position where it expects a W, the PyTorch's
        dispatcher will call into this function for special handling

        Our `__torch_function__` implementation goes through all of the
        args and kwargs and checks if any of them are `LazyAwaitable`.
        If it is, it will call `wait()` on it and replace the LazyAwaitable
        type object with the result of wait. In this way, async values
        are waited on when the concrete value is first needed and without
        the user having to write an explicit `wait()` call.
        """
        kwargs = kwargs or {}

        # wait() on all LazyAwaitable args/kwargs and replace
        # them with the resulting value.
        new_args = torch.fx.node.map_aggregate(args, LazyAwaitable._wait_async)
        new_kwargs = torch.fx.node.map_aggregate(kwargs, LazyAwaitable._wait_async)

        return func(*new_args, **new_kwargs)

    # pyre-ignore [2, 3]
    def __getattr__(self, name):
        """
        Overrides __getattr__ to allow LazyAwaitable to first wait and then call getattr
        on the wait results.
        """
        if name == "_result":
            raise RuntimeError(
                f"LazyAwaitable type {type(self)} has not been initialized properly, "
                f"did you forget to call 'super()'?"
            )
        elif name == "__setstate__":
            return super().__getattr__(name)  # pyre-ignore [16]

        res = LazyAwaitable._wait_async(self)
        return getattr(res, name)


class LazyNoWait(LazyAwaitable[W]):
    def __init__(self, obj: W) -> None:
        super().__init__()
        self._obj = obj

    def _wait_impl(self) -> W:
        return self._obj


KT = TypeVar("KT")
VT_co = TypeVar("VT_co")
ParentW = TypeVar("ParentW")


class LazyGetItemMixin(Generic[KT, VT_co]):
    """Augments the base LazyAwaitable with a lazy __getitem__ method.

    Instead of triggering a wait() on a __getitem__ call, KeyedLazyAwaitable
    will return another awaitable. This can achieve better
    communication/computation overlap by deferring the wait() until the
    tensor data is actually needed.

    This is intended for Awaitables that model keyed collections, like
    dictionaries or EmbeddingBagCollectionAwaitable.

    NOTE: if using this mixin, please include it before LazyAwaitable in the
    inheritance list, so that Python MRO can properly select this __getitem__
    implementation.
    """

    def __getitem__(self, key: KT) -> LazyAwaitable[VT_co]:
        return GetItemLazyAwaitable(self, key)


class GetItemLazyAwaitable(LazyAwaitable[W], Generic[W, ParentW, KT]):
    """The LazyAwaitable returned from a __getitem__ call on `LazyGetItemMixin`.

    When the actual value of this awaitable is requested, wait on the parent and
    then call __getitem__ on the result.
    """

    def __init__(self, parent: LazyAwaitable[ParentW], key: KT) -> None:
        super().__init__()
        self._parent = parent
        self._key = key

    def _wait_impl(self) -> W:
        kt = LazyAwaitable._wait_async(self._parent)
        return kt[self._key]


# install magic methods
for orig_method_name in torch.fx.graph.magic_methods:
    as_magic = f"__{orig_method_name}__"

    def scope(method):
        def impl(*args, **kwargs):
            lhs = args[0]
            op_fn = getattr(operator, method)
            if len(args) == 1:
                return op_fn(LazyAwaitable._wait_async(lhs))
            elif len(args) == 2:
                rhs = args[1]
                return op_fn(
                    LazyAwaitable._wait_async(lhs), LazyAwaitable._wait_async(rhs)
                )
            else:
                raise RuntimeError(f"magic method {as_magic} not supported!")

        impl.__name__ = as_magic
        setattr(LazyAwaitable, as_magic, impl)

    # pyre-ignore [16]
    scope(orig_method_name)

# install reflective magic methods
for orig_method_name in torch.fx.graph.reflectable_magic_methods:
    as_magic = f"__r{orig_method_name}__"

    # pyre-ignore [2, 3]
    def scope(method):
        # pyre-ignore [2, 3, 53]
        def impl(self, rhs):
            op_fn = getattr(operator, method)
            return op_fn(
                LazyAwaitable._wait_async(rhs), LazyAwaitable._wait_async(self)
            )

        impl.__name__ = as_magic
        impl.__qualname__ = as_magic
        setattr(LazyAwaitable, as_magic, impl)

    # pyre-ignore [16]
    scope(orig_method_name)


class ModuleShardingPlan:
    def _serialize(self) -> dict[str, Any]:
        raise NotImplementedError()


class CacheStatistics(abc.ABC):
    @property
    @abc.abstractmethod
    def expected_lookups(self) -> float:
        """Number of expected cache lookups per training step.

        This is the expected number of distinct values in a global training batch."""

    @abc.abstractmethod
    def expected_miss_rate(self, clf: float) -> float:
        """Expected cache lookup miss rate for a given cache size.

        When clf (cache load factor) is 0, returns 1.0 (100% miss). When clf is 1.0,
        returns 0 (100% hit). For values of clf between these extremes, returns the
        estimated miss rate of the cache, e.g. based on knowledge of the statistical
        properties of the training data set."""

    @property
    @abc.abstractmethod
    def cacheability(self) -> float:
        """Summarized measure of the difficulty to cache a dataset that is independent of
        cache size. A score of 0 means the dataset is very cacheable (e.g. high locality
        between accesses), a score of 1 is very difficult to cache."""


@dataclass
class CacheParams:
    """Caching related fused params for an embedding table. Most of these are
    passed to FBGEMM's Split TBE. These are useful for when uvm caching is used.

    Attributes:
        algorithm (Optional[CacheAlgorithm]): cache algorithm to use. Options
            include LRU and LFU.
        load_factor (Optional[float]): cache load factor per table. This decides
            the size of the cache space for the table, and is crucial for
            performance when using uvm caching.
        reserved_memory (Optional[float]): reserved memory for the cache.
        precision (Optional[DataType]): precision of the cache. Ideally this
            should be the same as the data type of the weights (aka table).
        prefetch_pipeline (Optional[bool]): whether to prefetch pipeline is
            used.
        stats (Optional[CacheStatistics]): cache statistics which has table
            related metadata. Used to create a better plan and tune the load
            factor.
    """

    algorithm: Optional[CacheAlgorithm] = None
    load_factor: Optional[float] = None
    reserved_memory: Optional[float] = None
    precision: Optional[DataType] = None
    prefetch_pipeline: Optional[bool] = None
    stats: Optional[CacheStatistics] = None
    multipass_prefetch_config: Optional[MultiPassPrefetchConfig] = None

    def __hash__(self) -> int:
        return hash(
            (
                self.algorithm,
                self.load_factor,
                self.reserved_memory,
                self.precision,
                self.prefetch_pipeline,
                self.multipass_prefetch_config,
            )
        )


@dataclass
class KeyValueParams:
    """
    Params for SSD TBE aka SSDTableBatchedEmbeddingBags.

    Attributes:
        ssd_storage_directory (Optional[str]): Directory for SSD. If we want directory
            to be f"data00_nvidia{local_rank}", pass in "data00_nvidia@local_rank".
        ssd_rocksdb_write_buffer_size: Optional[int]: rocksdb write buffer size per tbe,
            relavant to rocksdb compaction frequency
        ssd_rocksdb_shards: Optional[int]: rocksdb shards number
        gather_ssd_cache_stats: bool: whether enable ssd stats collection, std reporter and ods reporter
        stats_reporter_config
            report_interval: int: report interval in train iteration if gather_ssd_cache_stats is enabled
        use_passed_in_path: bool: whether to use passed in path for rocksdb shard or default one on SSD mount path
        l2_cache_size: Optional[int]: size in GB for l2 cache size per tbe
        max_l1_cache_size: Optional[int]: size in MB for max allocated l1 cache size per tbe
        enable_async_update: Optional[bool]: whether to enable async update for l2 cache
        bulk_init_chunk_size: int: number of rows to insert into rocksdb in each chunk
        lazy_bulk_init_enabled: bool: whether to enable lazy(async) bulk init for SSD TBE
        enable_raw_embedding_streaming: Optional[bool]: enable raw embedding streaming for SSD TBE
        res_store_shards: Optional[int] = None: the number of shards to store the raw embeddings
        kvzch_eviction_trigger_mode: Optional[int]: eviction trigger mode for KVZCH

        # Parameter Server (PS) Attributes
        ps_hosts (Optional[Tuple[Tuple[str, int]]]): List of PS host ip addresses
            and ports. Example: (("::1", 2000), ("::1", 2001), ("::1", 2002)).
            Reason for using tuple is we want it hashable.
        ps_client_thread_num (int): Number of threads to use for PS client
        ps_max_key_per_request (int): Maximum number of keys to send per request
        ps_max_local_index_length(int): Maximum local index length
    """

    ssd_storage_directory: Optional[str] = None
    ssd_rocksdb_write_buffer_size: Optional[int] = None
    ssd_rocksdb_shards: Optional[int] = None
    gather_ssd_cache_stats: Optional[bool] = None
    stats_reporter_config: Optional[TBEStatsReporterConfig] = None
    use_passed_in_path: bool = True
    l2_cache_size: Optional[int] = None  # size in GB
    max_l1_cache_size: Optional[int] = None  # size in MB
    enable_async_update: Optional[bool] = None  # enable L2 cache async update
    bulk_init_chunk_size: Optional[int] = None  # number of rows
    lazy_bulk_init_enabled: Optional[bool] = None  # enable lazy bulk init
    enable_raw_embedding_streaming: Optional[bool] = (
        None  # enable raw embedding streaming for SSD TBE
    )
    res_store_shards: Optional[int] = None  # shards to store the raw embeddings
    kvzch_eviction_trigger_mode: Optional[int] = None  # eviction trigger mode for KVZCH

    # Parameter Server (PS) Attributes
    ps_hosts: Optional[Tuple[Tuple[str, int], ...]] = None
    ps_client_thread_num: Optional[int] = None
    ps_max_key_per_request: Optional[int] = None
    ps_max_local_index_length: Optional[int] = None

    def __hash__(self) -> int:
        return hash(
            (
                self.ssd_storage_directory,
                self.ssd_rocksdb_write_buffer_size,
                self.ssd_rocksdb_shards,
                # Parameter Server (PS) Attributes
                self.ps_hosts,
                self.ps_client_thread_num,
                self.ps_max_key_per_request,
                self.ps_max_local_index_length,
                # tbe attributes
                self.gather_ssd_cache_stats,
                self.stats_reporter_config,
                self.l2_cache_size,
                self.max_l1_cache_size,
                self.enable_async_update,
                self.bulk_init_chunk_size,
                self.lazy_bulk_init_enabled,
                self.enable_raw_embedding_streaming,
                self.res_store_shards,
                self.kvzch_eviction_trigger_mode,
            )
        )


@dataclass
class ParameterSharding:
    """
        Describes (configures) the sharding of a parameter, which usually corresponds to a (feature) table.

        sharding_type (str): how this parameter is sharded. See ShardingType for well-known types.
        compute_kernel (str): compute kernel to be used by this parameter.
        ranks (Optional[List[int]]): rank of each shard.
        sharding_spec (Optional[ShardingSpec]): list of ShardMetadata for each shard.
        cache_params (Optional[CacheParams]): cache params for embedding lookup.
        enforce_hbm (Optional[bool]): whether to use HBM.
        stochastic_rounding (Optional[bool]): whether to use stochastic rounding.
        bounds_check_mode (Optional[BoundsCheckMode]): bounds check mode.
        output_dtype (Optional[DataType]): output dtype.
        key_value_params (Optional[KeyValueParams]): key value params for SSD TBE or PS.

    NOTE:
      ShardingType.TABLE_WISE - rank where this embedding is placed
      ShardingType.COLUMN_WISE - rank where the embedding shards are placed, seen as individual tables
      ShardingType.TABLE_ROW_WISE  - first rank when this embedding is placed
      ShardingType.ROW_WISE, ShardingType.DATA_PARALLEL - unused

    """

    sharding_type: str
    compute_kernel: str
    ranks: Optional[List[int]] = None
    sharding_spec: Optional[ShardingSpec] = None
    cache_params: Optional[CacheParams] = None
    enforce_hbm: Optional[bool] = None
    stochastic_rounding: Optional[bool] = None
    bounds_check_mode: Optional[BoundsCheckMode] = None
    output_dtype: Optional[DataType] = None
    key_value_params: Optional[KeyValueParams] = None


class EmbeddingModuleShardingPlan(ModuleShardingPlan, Dict[str, ParameterSharding]):
    """
    Map of ParameterSharding per parameter (usually a table). This describes the sharding plan for a torchrec module (e.g. `EmbeddingBagCollection`)
    """

    def __str__(self) -> str:
        out = ""
        param_table = []
        shard_table = []
        for param_name, param_sharding in self.items():
            param_table.append(
                [
                    param_name,
                    param_sharding.sharding_type,
                    param_sharding.compute_kernel,
                    param_sharding.ranks,
                ]
            )
            if isinstance(param_sharding.sharding_spec, EnumerableShardingSpec):
                shards = param_sharding.sharding_spec.shards
                if shards is not None:
                    for shard in shards:
                        shard_table.append(
                            [
                                param_name,
                                shard.shard_offsets,
                                shard.shard_sizes,
                                shard.placement,
                            ]
                        )
        out += "\n\n" + _tabulate(
            param_table, ["param", "sharding type", "compute kernel", "ranks"]
        )
        out += "\n\n" + _tabulate(
            shard_table, ["param", "shard offsets", "shard sizes", "placement"]
        )
        return out

    def _serialize(self) -> dict[str, Any]:
        sharding_plan_dict = {}
        for param_name, param_sharding in self.items():
            sharding_plan_dict[param_name] = {
                "sharding_type": param_sharding.sharding_type,
                "compute_kernel": param_sharding.compute_kernel,
                "ranks": param_sharding.ranks,
            }
            if isinstance(param_sharding.sharding_spec, EnumerableShardingSpec):
                shards = param_sharding.sharding_spec.shards
                if shards is not None:
                    sharding_plan_dict[param_name]["shards"] = []
                    for shard in shards:
                        shard_dict = asdict(shard)
                        shard_dict["placement"] = str(shard_dict["placement"])
                        sharding_plan_dict[param_name]["shards"].append(shard_dict)

        return sharding_plan_dict


@dataclass
class ShardingPlan:
    """
    Representation of sharding plan. This uses the FQN of the larger wrapped model (i.e the model that is wrapped using `DistributedModelParallel`)
    EmbeddingModuleShardingPlan should be used when TorchRec composability is desired.

    Attributes:
        plan (Dict[str, EmbeddingModuleShardingPlan]): dict keyed by module path of
            dict of parameter sharding specs keyed by parameter name.
    """

    plan: Dict[str, ModuleShardingPlan]

    def get_plan_for_module(self, module_path: str) -> Optional[ModuleShardingPlan]:
        """
        Args:
            module_path (str):

        Returns:
            Optional[ModuleShardingPlan]: dict of parameter sharding specs keyed by parameter name. None if sharding specs do not exist for given module_path.
        """
        return self.plan.get(module_path, None)

    def __str__(self) -> str:
        out = ""
        for i, (module_path, module_plan) in enumerate(self.plan.items()):
            if i > 0:
                out += "\n\n"
            out += "module: " + module_path
            out += str(module_plan)
        return out

    def _serialize(self) -> dict[str, Any]:
        sharding_plan_dict = {
            "plan": {
                module_path: module_plan._serialize()
                for module_path, module_plan in self.plan.items()
            }
        }
        return sharding_plan_dict


ShardedModuleContext = Multistreamable


class NullShardedModuleContext(Multistreamable):
    def record_stream(self, stream: Optional[torch.Stream]) -> None:
        pass

    # pyre-ignore [2]
    def __setattr__(self, key: str, value: Any) -> None:
        raise NotImplementedError()


class ShardingEnv:
    """
    Provides an abstraction over `torch.distributed.ProcessGroup`, which practically
    enables `DistributedModelParallel` to be used during inference.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        pg: Optional[dist.ProcessGroup] = None,
        output_dtensor: bool = False,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.process_group: Optional[dist.ProcessGroup] = pg
        self.device_mesh: Optional[DeviceMesh] = (
            init_device_mesh(
                device_type=_get_pg_default_device(pg).type,
                mesh_shape=(dist.get_world_size(pg),),
            )
            if pg
            else None
        )
        self.output_dtensor: bool = output_dtensor

    @classmethod
    def from_process_group(cls, pg: dist.ProcessGroup) -> "ShardingEnv":
        """
        Creates ProcessGroup-based sharding environment.

        NOTE:
            Typically used during training.
        """
        return cls(dist.get_world_size(pg), dist.get_rank(pg), pg)

    @classmethod
    def from_local(cls, world_size: int, rank: int) -> "ShardingEnv":
        """
        Creates a local host-based sharding environment.

        NOTE:
            Typically used during single host inference.
        """
        return cls(world_size, rank, None)


class ShardingEnv2D(ShardingEnv):
    """
    Creates a sharding environment for 2D parallelism, enables usage of 2D parallelism in sharding
    by seamlessly switching to the sub process group (sharding_pg) for a rank. This class is used
    as source of truth for TorchRec to understand if we're in a 2D parallel environment.

    NOTE:
        - global pg is part of `process_group` attribute to keep the same API as ShardingEnv,
        some parts of TorchRec require the global pg to work appropriately (ie: `DDPWrapper` in `DistributedModelParallel`)
        - `world_size` and `rank` attributes return values relative to `sharding_pg`, this is different
        from default ShardingEnv returning values relative to `global_pg`

    Attributes:
        sharding_pg: The process group containing the ranks to shard on.
        global_pg: The process group representing global ranks.
        device_mesh: A 2D device mesh representing the topology of the global world size
            on "replicate" and "shard" dimensions.
        node_group_size (Optional[int]): The size of each node group. If not provided, it will be inferred
            from env var `LOCAL_WORLD_SIZE`.
    """

    def __init__(
        self,
        sharding_pg: dist.ProcessGroup,
        global_pg: dist.ProcessGroup,
        device_mesh: DeviceMesh,
        node_group_size: Optional[int] = None,
        use_inter_host_allreduce: bool = False,
    ) -> None:
        assert device_mesh.ndim == 2, "DeviceMesh must be two dimensional!"
        self.world_size: int = dist.get_world_size(sharding_pg)
        self.global_world_size: int = dist.get_world_size(global_pg)
        self.rank: int = dist.get_rank(sharding_pg)
        self.global_rank: int = dist.get_rank(global_pg)
        self.process_group: dist.ProcessGroup = (
            global_pg  # to keep consistent naming between ShardingEnv and ShardingEnv2D
        )
        self.sharding_pg: dist.ProcessGroup = sharding_pg
        self.device_mesh: DeviceMesh = device_mesh
        self.node_group_size: Optional[int] = node_group_size
        self.output_dtensor: bool = True
        self.use_inter_host_allreduce: bool = use_inter_host_allreduce

    def num_sharding_groups(self) -> int:
        """
        Return number of sharding groups, also known as the number of times model parallel is replicated
        """
        return self.global_world_size // self.world_size

    def remap_rank(self, rank: int, sharding_type: ShardingType) -> int:
        """
        Remap from current rank to the appropriate rank in a continuous [0, ..., world size] array for the given sharding type.

        Args:
            rank (int): rank to remap.
            sharding_type (ShardingType): sharding type to remap to.

        Returns:
            int: remapped rank.
        """
        if sharding_type in (
            ShardingType.COLUMN_WISE,
            ShardingType.TABLE_WISE,
            ShardingType.GRID_SHARD,
        ):
            return (
                rank % self.world_size
                if self.use_inter_host_allreduce
                else rank // self.num_sharding_groups()
            )
        else:
            raise ValueError(
                f"Do not need 2D specific remapping logic for sharding type: {sharding_type}"
            )


class NullShardingContext(Multistreamable):
    def record_stream(self, stream: torch.Stream) -> None:
        pass


Out = TypeVar("Out")
CompIn = TypeVar("CompIn")
DistOut = TypeVar("DistOut")
ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


class ShardedModule(
    abc.ABC,
    nn.Module,
    Generic[CompIn, DistOut, Out, ShrdCtx],
    ModuleNoCopyMixin,
):
    """
    All model-parallel modules implement this interface.
    Inputs and outputs are data-parallel.

    Args::
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]) : Mapping of CommOp name to QuantizedCommCodecs

    NOTE:
        'input_dist' / 'output_dist' are responsible of transforming inputs / outputs
        from data-parallel to model parallel and vise-versa.
    """

    _FORCE_STATE_DICT_LOAD = True

    @abc.abstractmethod
    def __init__(
        self, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    ) -> None:

        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        if qcomm_codecs_registry is None:
            qcomm_codecs_registry = {}
        self._qcomm_codecs_registry = qcomm_codecs_registry
        # In pipelining, this flag is flipped in rewrite_model when the forward is replaced with the pipelined forward
        self.is_pipelined = False

    @abc.abstractmethod
    def create_context(self) -> ShrdCtx:
        pass

    @property
    def qcomm_codecs_registry(self) -> Optional[Dict[str, QuantizedCommCodecs]]:
        return self._qcomm_codecs_registry

    @abc.abstractmethod
    def input_dist(
        self,
        ctx: ShrdCtx,
        # pyre-ignore[2]
        *input,
        # pyre-ignore[2]
        **kwargs,
    ) -> Awaitable[Awaitable[CompIn]]:
        pass

    @abc.abstractmethod
    def compute(self, ctx: ShrdCtx, dist_input: CompIn) -> DistOut:
        pass

    @abc.abstractmethod
    def output_dist(self, ctx: ShrdCtx, output: DistOut) -> LazyAwaitable[Out]:
        pass

    def prefetch(
        self,
        dist_input: CompIn,
        forward_stream: Optional[torch.Stream] = None,
        ctx: Optional[ShrdCtx] = None,
    ) -> None:
        return None

    def compute_and_output_dist(
        self, ctx: ShrdCtx, input: CompIn
    ) -> LazyAwaitable[Out]:
        """
        In case of multiple output distributions it makes sense to override this method
        and initiate the output distibution as soon as the corresponding compute
        completes.
        """
        output = self.compute(ctx, input)
        return self.output_dist(ctx, output)

    # pyre-ignore[2]
    def forward(self, *input, **kwargs) -> LazyAwaitable[Out]:
        """
        Executes the input dist, compute, and output dist steps.

        Args:
            *input: input.
            **kwargs: keyword arguments.

        Returns:
            LazyAwaitable[Out]: awaitable of output from output dist.
        """
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input, **kwargs).wait().wait()
        return self.compute_and_output_dist(ctx, dist_input)

    def update(self, ctx: ShrdCtx, dist_input: CompIn) -> None:
        """
        Updates the sharded module with the given input.

        Args:
            ctx (ShrdCtx): sharding context.
            dist_input (CompIn): distributed input.
        """
        raise NotImplementedError(
            "The update method is not implemented for this collection. Please make sure you are using the correct compute kernel and sharding type."
        )

    def write_dist(
        self, ctx: ShrdCtx, *input, **kwargs  # pyre-ignore[2]
    ) -> Awaitable[Awaitable[CompIn]]:
        raise NotImplementedError(
            "The write_dist method is not implemented for this collection. Please make sure you are using the correct compute kernel and sharding type."
        )

    # pyre-ignore[2]
    def write(self, *input, **kwargs) -> None:
        """
        Executes the write dist and update steps.

        Args:
            *input: input.
            **kwargs: keyword arguments.

        Returns:
            LazyAwaitable[Out]: awaitable of output from output dist.
        """
        ctx = self.create_context()
        dist_input = self.write_dist(ctx, *input, **kwargs).wait().wait()
        self.update(ctx, dist_input)

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for key, _ in self.named_parameters(prefix):
            yield key

    def preprocess_input(
        self,
        args: List[_T],
        kwargs: Mapping[str, _T],
    ) -> Tuple[List[_T], Mapping[str, _T]]:
        """
        This function can be used to preprocess the input arguments prior to module forward call.

        For example, it is used in ShardedFeatureProcessorEmbeddingBagCollection to transform the input data
        prior to the forward call.
        """
        return args, kwargs

    @property
    @abc.abstractmethod
    def unsharded_module_type(self) -> Type[nn.Module]:
        """
        This property is added as part of dynamic sharding implementation.

        When resharding an already-sharded module wrapped in DMP, the unsharded
        module type is needed to identify the proper sharder to reshard. This is
        due to DistributedModelParellel (DMP) references module Sharders based
        on the unsharded module type.
        """
        ...


def get_tensor_size_bytes(t: torch.Tensor) -> int:
    b: int = t.numel() * t.element_size()
    if isinstance(t, UInt4Tensor):
        assert (
            b % 2 == 0
        ), f"UInt4Tensor must have number of elements that is divisible by 2, got {t.numel()}"
        b = b // 2
    elif isinstance(t, UInt2Tensor):
        assert (
            b % 4 == 0
        ), f"UInt2Tensor must have number of elements that is divisible by 4, got {t.numel()}"
        b = b // 4

    return b


class ModuleSharder(abc.ABC, Generic[M]):
    """
    `ModuleSharder` is per each module, which supports sharding,
    e.g. `EmbeddingBagCollection`.

    Args::
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]) : Mapping of CommOp name to QuantizedCommCodecs
    """

    def __init__(
        self, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    ) -> None:
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._qcomm_codecs_registry = qcomm_codecs_registry

    # pyre-fixme[56]: Pyre doesn't yet support decorators with ParamSpec applied to
    #  generic functions. Consider using a context manager instead of a decorator, if
    #  possible.
    @abc.abstractclassmethod
    # pyre-ignore [3]
    def shard(
        self,
        module: M,
        params: EmbeddingModuleShardingPlan,
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedModule[Any, Any, Any, Any]:
        """
        Does the actual sharding. It will allocate parameters on the requested locations
        as specified by corresponding ParameterSharding.

        Default implementation is data-parallel replication.

        Args:
            module (M): module to shard.
            params (EmbeddingModuleShardingPlan): dict of fully qualified parameter names
                (module path + parameter name, '.'-separated) to its sharding spec.
            env (ShardingEnv): sharding environment that has the process group.
            device (Optional[torch.device]): compute device.
            path (Optional[str]): fully qualified name of the module. used for trace annotations in embedding modules

        Returns:
            ShardedModule[Any, Any, Any]: sharded module implementation.
        """
        ...

    @property
    @abc.abstractmethod
    def module_type(self) -> Type[M]: ...

    @property
    def qcomm_codecs_registry(self) -> Optional[Dict[str, QuantizedCommCodecs]]:
        return self._qcomm_codecs_registry

    def shardable_parameters(self, module: M) -> Dict[str, nn.Parameter]:
        """
        List of parameters that can be sharded.
        """
        return dict(module.named_parameters())

    def sharding_types(self, compute_device_type: str) -> List[str]:
        """
        List of supported sharding types. See `ShardingType` for well-known examples.
        """
        return [ShardingType.DATA_PARALLEL.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        """
        List of supported compute kernels for a given sharding type and compute device.
        """

        return [ComputeKernel.DEFAULT.value]

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        """
        List of system resources and corresponding usage given a compute device and
        compute kernel.
        """

        assert compute_device_type in {"cuda", "cpu", "mtia"}
        storage_map = {
            "cuda": ParameterStorage.HBM,
            "cpu": ParameterStorage.DDR,
            "mtia": ParameterStorage.HBM,
        }
        return {storage_map[compute_device_type].value: get_tensor_size_bytes(tensor)}


class ShardingPlanner(abc.ABC):
    """
    Plans sharding.
    This plan can be saved and re-used to ensure sharding stability.
    """

    @abc.abstractmethod
    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Plans sharding for provided module and given sharders.

        Args:
            module (nn.Module): module that sharding is planned for.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            ShardingPlan: the computed sharding plan.
        """
        ...

    @abc.abstractmethod
    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Calls self.plan(...) on rank 0 and broadcasts.

        Args:
            module (nn.Module): module that sharding is planned for.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            ShardingPlan: the computed sharding plan.
        """
        ...


def rank_device(device_type: str, rank: int) -> torch.device:
    if device_type == "cpu":
        return torch.device("cpu")

    return torch.device(f"{device_type}:{rank}")


class ObjectPoolShardingType(Enum):
    """
    Sharding type for object pool
    """

    ROW_WISE = "row_wise"
    # across nodes, state will be replicated. On lookup, all2alls will happen intranode.
    # State is synced via update a2a being global internode.
    REPLICATED_ROW_WISE = "replicated_row_wise"


@dataclass
class ObjectPoolShardingPlan(ModuleShardingPlan):
    sharding_type: ObjectPoolShardingType
    inference: bool = False

    def _serialize(self) -> dict[str, Any]:
        return {
            "sharding_type": self.sharding_type.name,
            "inference": self.inference,
        }


@dataclass
class ShardingBucketMetadata:
    """
    If a table is row-wise sharded with bucketization, this class contains the bucket information for the table.

    Attributes:
        num_buckets_per_shard (List[int]): Number of buckets in each shard of the table.
        bucket_offsets_per_shard (List[int]): Index of the first bucket in each shard.
        bucket_size (int): No. of rows in one bucket.
    """

    num_buckets_per_shard: List[int]
    bucket_offsets_per_shard: List[int]
    bucket_size: int


class DMPCollectionConfig:
    module: Type[nn.Module]
    plan: "ShardingPlan" = field(repr=False)  # sub-tree-specific sharding plan
    sharding_group_size: int
    node_group_size: Optional[int] = None
    use_inter_host_allreduce: bool = False

    def __init__(
        self,
        module: Type[nn.Module],
        plan: "ShardingPlan",
        sharding_group_size: int,
        node_group_size: Optional[int] = None,
        use_inter_host_allreduce: bool = False,
    ) -> None:
        self.module = module
        self.plan = plan
        self.sharding_group_size = sharding_group_size
        self.node_group_size = node_group_size
        self.use_inter_host_allreduce = use_inter_host_allreduce

    def __post_init__(self) -> None:
        if isinstance(self.module, ShardedModule):
            raise ValueError(
                f"ShardedModule should not be passed into DMPCollectionConfig: got {type(self.module)}"
            )


# for internal use in DMPCollection
class DMPCollectionContext(DMPCollectionConfig):
    device_mesh: "DeviceMesh" = field(init=False)
    sharding_pg: "dist.ProcessGroup" = field(init=False)
    replica_pg: "dist.ProcessGroup" = field(init=False)
    modules_to_sync: List[nn.Module] = field(init=False, default_factory=list)
    sharded_module: Optional[nn.Module] = field(init=False, default=None)
