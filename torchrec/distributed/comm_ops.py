#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, cast, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives
from fbgemm_gpu.split_embedding_configs import SparseType
from torch import Tensor
from torch.autograd import Function
from torch.autograd.profiler import record_function
from torchrec.distributed.types import Awaitable, NoWait, QuantizedCommCodecs
from torchrec.distributed.utils import none_throws
from torchrec.pt2.checks import is_torchdynamo_compiling

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

logger: logging.Logger = logging.getLogger(__name__)


# Opt-in, off by default. Routes the three all2all-sequence permutes below
# through torchrec's Triton kernel instead of fbgemm's.
#
# Deliberately here and not on KeyedJaggedTensor: that class is pulled into
# TorchScript by dper_lib/silvertorch/modules/serving/value_model.py's
# @torch.jit.interface, which makes a mutable class attribute, a classmethod
# reading one, and a module-level helper naming the class all fail to compile.
# comm_ops is not on the scripting path, so none of that applies. These are also
# the sites that matter: the shape they launch is ~75% of the model's total
# permute_2D_sparse_data GPU time.
def _permute_2D_sparse_data(
    permute: Optional[Tensor],
    lengths: Tensor,
    values: Tensor,
    weights: Optional[Tensor],
    permuted_lengths_sum: Optional[int],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """fbgemm's permute_2D_sparse_data, or the Triton one when enabled.

    `should_use_triton` decides from shapes alone, so the dispatch itself costs
    no device synchronisation, and it defers to fbgemm on the segment counts
    where fbgemm is faster.
    """
    if get_use_triton_permute_2d() and values.is_cuda and permute is not None:
        from torchrec.sparse.triton_permute_2d import (
            should_use_triton,
            triton_permute_2d_sparse_data,
        )

        if should_use_triton(permute, lengths, permuted_lengths_sum):
            return triton_permute_2d_sparse_data(
                permute, lengths, values, weights, permuted_lengths_sum
            )
    return torch.ops.fbgemm.permute_2D_sparse_data(
        permute, lengths, values, weights, permuted_lengths_sum
    )


# OSS
try:
    pass
except ImportError:
    pass


W = TypeVar("W")

# TODO: T96382816, NE Parity Backward compatibility
GRADIENT_DIVISION: bool = True
USE_SYNC_COLLECTIVES: bool = False
USE_TRITON_PERMUTE_2D: bool = False
# Kill switch for the fbgemm fast paths in `_recat_pooled_embedding_grad_out`. Set
# `TORCHREC_USE_FBGEMM_RECAT=0` to fall back to the split/contiguous/cat reference.
# Read once at import, so it has to be set before the process starts;
# `set_use_fbgemm_recat` still overrides it in-process.
USE_FBGEMM_RECAT: bool = os.environ.get("TORCHREC_USE_FBGEMM_RECAT", "0") == "1"


def set_gradient_division(val: bool) -> None:
    global GRADIENT_DIVISION
    GRADIENT_DIVISION = val


def get_gradient_division() -> bool:
    global GRADIENT_DIVISION
    return GRADIENT_DIVISION


def set_use_sync_collectives(val: bool) -> None:
    global USE_SYNC_COLLECTIVES
    USE_SYNC_COLLECTIVES = val


def get_use_sync_collectives() -> bool:
    global USE_SYNC_COLLECTIVES
    return USE_SYNC_COLLECTIVES or is_torchdynamo_compiling()


def set_use_triton_permute_2d(val: bool) -> None:
    global USE_TRITON_PERMUTE_2D
    USE_TRITON_PERMUTE_2D = val


def get_use_triton_permute_2d() -> bool:
    return USE_TRITON_PERMUTE_2D


def set_use_fbgemm_recat(val: bool) -> None:
    global USE_FBGEMM_RECAT
    USE_FBGEMM_RECAT = val


def get_use_fbgemm_recat() -> bool:
    return USE_FBGEMM_RECAT


@contextmanager
def torchrec_use_sync_collectives():
    original_use_sync_collectives: bool = get_use_sync_collectives()
    set_use_sync_collectives(True)
    yield
    set_use_sync_collectives(original_use_sync_collectives)


def _is_fp8_rowwise_padding_enabled() -> bool:
    return bool(int(os.environ.get("TORCHREC_ENABLE_FP8_ROWWISE_PADDING", 0)))


def _is_mx4_padding_enabled() -> bool:
    return bool(int(os.environ.get("TORCHREC_ENABLE_MX4_PADDING", 0)))


def _qcomm_padded_splits(
    codec: Any, qcomm_ctx: Any, splits: List[int]
) -> Optional[List[int]]:
    """
    Rounds each split up to a multiple of `row_dim` for FP8 rowwise or MX4 qcomms
    (MX4 contexts set `row_dim` to the group size). Each precision has its own
    opt-in flag, which must match on every rank. Returns None if padding is off for
    the codec's precision or every split is already aligned; the comm is unchanged.
    """
    fp8_enabled = _is_fp8_rowwise_padding_enabled()
    mx4_enabled = _is_mx4_padding_enabled()
    if (
        not (fp8_enabled or mx4_enabled)
        or qcomm_ctx is None
        or getattr(qcomm_ctx, "row_dim", -1) <= 0
    ):
        return None
    precision = getattr(codec, "_comm_precision", None)
    if not (
        (fp8_enabled and precision == SparseType.FP8)
        or (mx4_enabled and precision == SparseType.MX4)
    ):
        return None
    row_dim = qcomm_ctx.row_dim
    padded = [split + (row_dim - split % row_dim) % row_dim for split in splits]
    return padded if padded != splits else None


def _pad_splits(tensor: Tensor, splits: List[int], padded: List[int]) -> Tensor:
    return torch.cat(
        [
            torch.nn.functional.pad(chunk, (0, p - s))
            for chunk, s, p in zip(tensor.split(splits), splits, padded)
        ]
    )


def _strip_splits(tensor: Tensor, padded: List[int], splits: List[int]) -> Tensor:
    return torch.cat([chunk[:s] for chunk, s in zip(tensor.split(padded), splits)])


SEQ_PERMUTE_BEFORE_DECODE_ENV = "TORCHREC_SEQ_A2A_PERMUTE_BEFORE_DECODE"
SIDE_STREAM_OUTPUT_ENCODE_ENV = "TORCHREC_KVZCH_SIDE_STREAM_OUTPUT_ENCODE"
SEQ_A2A_BWD_RECORD_STREAM_ENV = "TORCHREC_SEQ_A2A_BWD_RECORD_STREAM"


def _seq_a2a_bwd_record_stream() -> bool:
    return os.environ.get(SEQ_A2A_BWD_RECORD_STREAM_ENV, "0") == "1"


"""
Some commonly used notations for comm ops:
    B - batch size
    T - number of embedding tables
    D - embedding dimension
"""


class Comm(ABC):
    """
    Interface for communication primitives.
    A primitive primarily needs to handle 3 tasks:
    1. **Memory allocation strategy**:
       - Associate each call to a temporary buffer (flexible, simple)
       - Reuse a persistent buffer (efficient, complex)
    2. **Memory location**:
       - NCCL memory pool
       - Regular CUDA caching allocator
    3. **Communication execution**:
       - Actual communication primitive implementation
    See `All2AllSingle` for a concrete example.
    """

    @abstractmethod
    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Allocate memory for communication buffers.
        Args:
            size (Sequence[Union[int, torch.SymInt]]): size of the tensor buffer
            dtype (torch.dtype): dtype of the tensor buffer
            device (torch.device): which device to allocate the tensor onto
        Returns:
            torch.Tensor: Allocated tensor buffer
        Example:
            ```python
            tensor = comm.allocate([1024, 512], dtype=torch.float32, device="cuda:0")
            ```
        """
        ...


class Request(Awaitable[W]):
    """
    Defines a collective operation request for a process group on a tensor.

    Args:
        pg (dist.ProcessGroup): The process group the request is for.
    """

    def __init__(self, pg: dist.ProcessGroup, device: torch.device) -> None:
        super().__init__()
        self.pg: dist.ProcessGroup = pg
        self.req: Optional[dist.Work] = None
        self.tensor: Optional[W] = None
        # Input of an in-flight async collective; the collective still reads it
        # after the op that launched it returns.
        self.send_tensor: Optional[torch.Tensor] = None
        self.a2ai = None  # type: ignore
        self.qcomm_ctx = None  # type: ignore
        self.rsi = None  # type: ignore
        self.agi = None  # type: ignore
        self.wait_function = None  # type: ignore
        # TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP: the row-wise unbucketize permute
        # that SequenceEmbeddingsAwaitable hands to All2All_Seq_Req_Wait.forward
        # for one wait, and whether that forward applied it.
        self.output_row_map: Optional[Tensor] = None
        self.row_map_applied: bool = False

        # This dummy tensor is used to build the autograd graph between
        # CommOp-Req and CommOp-Await. The actual forward tensors, and backwards gradient tensors
        # are stored in self.tensor
        # torch.zeros is a call_function, not placeholder, hence fx.trace incompatible.
        self.dummy_tensor: torch.Tensor = torch.zeros_like(
            torch.empty(
                1,
                requires_grad=True,
                device=device,
            )
        )

    def _wait_impl(self) -> W:
        """
        Calls the wait function for this request.
        """

        # pyrefly: ignore[missing-attribute]
        ret = self.wait_function.apply(self.pg, self, self.dummy_tensor)
        if isinstance(ret, torch.Tensor) and ret.device.type == "cuda":
            ret.record_stream(torch.get_device_module(ret.device).current_stream())
        self.req = None
        self.tensor = None
        # pyrefly: ignore [bad-return]
        return ret


@dataclass
class All2AllPooledInfo:
    """
    The data class that collects the attributes when calling the `alltoall_pooled`
    operation.

    Attributes:
        batch_size_per_rank (List[int]): batch size in each rank
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        dim_sum_per_rank_tensor (Optional[Tensor]): the tensor version of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        cumsum_dim_sum_per_rank_tensor (Optional[Tensor]): cumulative sum of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.
        recat_dim_sum_per_rank_tensor (Optional[Tensor]): int64 device tensor of
            `dim_sum_per_rank`, consumed by the fbgemm recat kernel in
            `_recat_pooled_embedding_grad_out`.
        recat_cumsum_dim_sum_per_rank_tensor (Optional[Tensor]): int64 device tensor
            holding the *exclusive* cumulative sum of `dim_sum_per_rank`
            (`[0, d0, d0 + d1, ...]`), i.e. each rank's starting column. Note this
            differs from `cumsum_dim_sum_per_rank_tensor`, which is inclusive.
    """

    batch_size_per_rank: List[int]
    dim_sum_per_rank: List[int]
    dim_sum_per_rank_tensor: Optional[Tensor]
    cumsum_dim_sum_per_rank_tensor: Optional[Tensor]
    codecs: Optional[QuantizedCommCodecs] = None
    recat_dim_sum_per_rank_tensor: Optional[Tensor] = None
    recat_cumsum_dim_sum_per_rank_tensor: Optional[Tensor] = None


@dataclass
class VariableBatchAll2AllPooledInfo:
    """
    The data class that collects the attributes when calling the
    `variable_batch_alltoall_pooled` operation.

    Attributes:
        batch_size_per_rank_per_feature (List[List[int]]): batch size per rank per
            feature.
        batch_size_per_feature_pre_a2a (List[int]): local batch size before scattering.
        emb_dim_per_rank_per_feature (List[List[int]]): embedding dimension per rank
            per feature
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.
        input_splits (Optional[List[int]]): input splits of tensor all to all.
        output_splits (Optional[List[int]]): output splits of tensor all to all.
    """

    batch_size_per_rank_per_feature: List[List[int]]
    batch_size_per_feature_pre_a2a: List[int]
    emb_dim_per_rank_per_feature: List[List[int]]
    codecs: Optional[QuantizedCommCodecs] = None
    input_splits: Optional[List[int]] = None
    output_splits: Optional[List[int]] = None


@dataclass
class All2AllSequenceInfo:
    """
    The data class that collects the attributes when calling the `alltoall_sequence`
    operation.

    Attributes:
        embedding_dim (int): embedding dimension.
        lengths_after_sparse_data_all2all (Tensor): lengths of sparse features after
            AlltoAll.
        forward_recat_tensor (Optional[Tensor]): recat tensor for forward.
        backward_recat_tensor (Tensor): recat tensor for backward.
        input_splits (List[int]): input splits.
        output_splits (List[int]): output splits.
        variable_batch_size (bool): whether variable batch size is enabled.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.
        permuted_lengths_after_sparse_data_all2all (Optional[Tensor]): lengths of sparse
            features before AlltoAll.
        decode_row_permute (Optional[Tensor]): rows of the received embeddings to
            select before decoding; set by `fold_row_permute_into_seq_decode`.
        precomputed_send (Optional[SeqA2APrecomputedSend]): forward permute and
            encode already run by `precompute_seq_a2a_send`.
    """

    embedding_dim: int
    lengths_after_sparse_data_all2all: Tensor
    forward_recat_tensor: Optional[Tensor]
    backward_recat_tensor: Tensor
    input_splits: List[int]
    output_splits: List[int]
    variable_batch_size: bool = False
    codecs: Optional[QuantizedCommCodecs] = None
    permuted_lengths_after_sparse_data_all2all: Optional[Tensor] = None
    decode_row_permute: Optional[Tensor] = None
    precomputed_send: Optional["SeqA2APrecomputedSend"] = None


@dataclass
class All2AllVInfo:
    """
    The data class that collects the attributes when calling the `alltoallv` operation.

    Attributes:
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        B_global (int): global batch size for each rank.
        B_local (int): local batch size before scattering.
        B_local_list: (List[int]): local batch sizes for each embedding table locally
            (in my current rank).
        D_local_list (List[int]): embedding dimension of each embedding table locally
            (in my current rank).
        input_split_sizes (List[int]): The input split sizes for each rank, this
            remembers how to split the input when doing the `all_to_all_single` operation.
        output_split_sizes (List[int]): The output split sizes for each rank, this
            remembers how to fill the output when doing the `all_to_all_single` operation.
    """

    dims_sum_per_rank: List[int]
    B_global: int
    B_local: int
    B_local_list: List[int]
    D_local_list: List[int]
    input_split_sizes: List[int] = field(default_factory=list)
    output_split_sizes: List[int] = field(default_factory=list)
    codecs: Optional[QuantizedCommCodecs] = None


@dataclass
class ReduceScatterInfo:
    """
    The data class that collects the attributes when calling the `reduce_scatter_pooled`
    operation.

    Attributes:
        input_sizes (List[torch.Size]): the sizes of the input tensors. This remembers the
            sizes of the input tensors when running the backward pass and producing the
            gradient.
    """

    input_sizes: List[torch.Size]
    codecs: Optional[QuantizedCommCodecs] = None


@dataclass
class ReduceScatterBaseInfo:
    """
    The data class that collects the attributes when calling the
    `reduce_scatter_base_pooled` operation.

    Attributes:
        input_sizes (torch.Size): the sizes of the input flatten tensor.
    """

    input_sizes: torch.Size
    codecs: Optional[QuantizedCommCodecs] = None


@dataclass
class AllGatherBaseInfo:
    """
    The data class that collects the attributes when calling the
    `all_gatther_base_pooled` operation.

    Attributes:
        input_size (int): the size of the input tensor.
    """

    input_size: torch.Size
    codecs: Optional[QuantizedCommCodecs] = None


@dataclass
class ReduceScatterVInfo:
    """
    The data class that collects the attributes when calling the `reduce_scatter_v_pooled`
    operation.

    Attributes:
        input_sizes (List[List[int]]): the sizes of the input tensors. This saves the
            sizes of the input tensors when running the backward pass and producing the
            gradient.
        input_splits (List[int]): the splits of the input tensors along dim 0.
        equal_splits (bool): ...
        total_input_size: (List[int]): total input size.
        codecs (Optional[QuantizedCommCodecs]): ...
    """

    input_sizes: List[List[int]]
    input_splits: List[int]
    equal_splits: bool
    total_input_size: List[int]
    codecs: Optional[QuantizedCommCodecs]


@dataclass
class All2AllDenseInfo:
    """
    The data class that collects the attributes when calling the `alltoall_dense`
    operation.
    """

    output_splits: List[int]
    batch_size: int
    input_shape: List[int]
    input_splits: List[int]


def _get_split_lengths_by_len(
    world_size: int, my_rank: int, n: int
) -> Tuple[int, List[int]]:
    k = n // world_size
    m = n % world_size
    splits = []
    if m == 0:
        for _ in range(world_size):
            splits.append(k)

        my_len = k
    else:
        for i in range(world_size):
            splits.append((k + 1) if i < m else k)
        my_len = splits[my_rank]
    return (my_len, splits)


class All2AllSingle(Comm):
    """
    Interface for all-to-all single tensor communication.
    This primitive performs an all-to-all scatter/gather operation where each
    rank sends and receives data from all other ranks in a single operation.
    """

    @abstractmethod
    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[list[int]] = None,
        input_split_sizes: Optional[list[int]] = None,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """
        Execute all-to-all single operation.
        Args:
            output_tensor: Pre-allocated output tensor
            input_tensor: Input tensor to scatter
            output_split_sizes: Sizes for splitting the output (if None, split evenly)
            input_split_sizes: Sizes for splitting the input (if None, split evenly)
            group: Process group (if None, uses default group)
            async_op: Whether to perform asynchronously
        Returns:
            Optional work handle if async_op=True, None otherwise
        """
        ...


class DefaultAllocMixin:
    """
    Mixin providing default tensor allocation using PyTorch's standard allocator.
    """

    def allocate(
        self,
        size: Sequence[Union[int, torch.SymInt]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate tensor using torch.empty with standard CUDA caching allocator."""
        return torch.empty(size, dtype=dtype, device=device)


class DefaultAll2AllSingle(DefaultAllocMixin, All2AllSingle):
    """
    Default implementation of all-to-all single using PyTorch's distributed primitives.

    This implementation uses:
    - Standard CUDA caching allocator for memory allocation
    - PyTorch's dist.all_to_all_single for communication
    """

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[list[int]] = None,
        input_split_sizes: Optional[list[int]] = None,
        group: Optional[dist.ProcessGroup] = None,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """Execute all-to-all using PyTorch's native implementation."""
        return dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes,
            input_split_sizes,
            group=group,
            async_op=async_op,
        )


def alltoall_pooled(
    a2a_pooled_embs_tensor: Tensor,
    batch_size_per_rank: List[int],
    dim_sum_per_rank: List[int],
    dim_sum_per_rank_tensor: Optional[Tensor] = None,
    cumsum_dim_sum_per_rank_tensor: Optional[Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
    all_to_all_single_comm: Optional[All2AllSingle] = None,
    recat_dim_sum_per_rank_tensor: Optional[Tensor] = None,
    recat_cumsum_dim_sum_per_rank_tensor: Optional[Tensor] = None,
) -> Awaitable[Tensor]:
    """
    Performs AlltoAll operation for a single pooled embedding tensor.

    Each process splits the input pooled embeddings tensor based on the world size,
    and then scatters the split list to all processes in the group. Then concatenates
    the received tensors from all processes in the group and returns a single output tensor.

    Args:
        a2a_pooled_embs_tensor (Tensor): input pooled embeddings. Must be pooled
            together before passing into this function. Its shape is `B x D_local_sum`,
            where `D_local_sum` is the dimension sum of all the local embedding tables.
        batch_size_per_rank (List[int]): batch size in each rank.
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        dim_sum_per_rank_tensor (Optional[Tensor]): the tensor version of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        cumsum_dim_sum_per_rank_tensor (Optional[Tensor]): cumulative sum of
            `dim_sum_per_rank`, this is only used by the fast kernel of
            `_recat_pooled_embedding_grad_out`.
        group (Optional[dist.ProcessGroup]): the process group to work on. If None, the
            default process group will be used.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.
        recat_dim_sum_per_rank_tensor (Optional[Tensor]): int64 device tensor of
            `dim_sum_per_rank`, consumed by the fbgemm recat kernel in
            `_recat_pooled_embedding_grad_out`.
        recat_cumsum_dim_sum_per_rank_tensor (Optional[Tensor]): int64 device tensor
            holding the *exclusive* cumulative sum of `dim_sum_per_rank`.

    Returns:
        Awaitable[Tensor]: Async work handle which can be `wait()`-ed later to
            get the resulting tensor.

    Example:
        ```python
        # Using default implementation
        result = alltoall_pooled(embeddings, batch_sizes, dim_sums)
        output = result.wait()
        # Using custom implementation
        custom_comm = MyCustomAll2All()
        result = alltoall_pooled(embeddings, batch_sizes, dim_sums,
                                 all2all_single_comm=custom_comm)
        ```

    .. warning::
        `alltoall_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if group.size() <= 1:
        return NoWait(a2a_pooled_embs_tensor)

    a2ai = All2AllPooledInfo(
        batch_size_per_rank=batch_size_per_rank,
        dim_sum_per_rank=dim_sum_per_rank,
        dim_sum_per_rank_tensor=dim_sum_per_rank_tensor,
        cumsum_dim_sum_per_rank_tensor=cumsum_dim_sum_per_rank_tensor,
        codecs=codecs,
        recat_dim_sum_per_rank_tensor=recat_dim_sum_per_rank_tensor,
        recat_cumsum_dim_sum_per_rank_tensor=recat_cumsum_dim_sum_per_rank_tensor,
    )

    # Keep the experimental dependency out of non-TPU import and execution paths.
    if getattr(torch, "tpu", None) is not None and (
        torch.accelerator.current_accelerator(check_available=True)
        == torch.device("tpu")
    ):
        from torchrec.experimental.torch_tpu.uneven_all_to_all import (
            maybe_all2all_pooled_uneven_tpu,
        )

        def even_all_to_all(input_tensor: Tensor, split_size: int) -> Tensor:
            split_sizes = [split_size] * group.size()
            return AllToAllSingle.apply(
                input_tensor,
                split_sizes,
                split_sizes,
                pg_name(group),
                group.size(),
                get_gradient_division(),
            )

        uneven = maybe_all2all_pooled_uneven_tpu(
            group,
            a2a_pooled_embs_tensor,
            batch_size_per_rank,
            dim_sum_per_rank,
            has_codecs=codecs is not None,
            even_all_to_all=even_all_to_all,
        )
        if uneven is not None:
            return NoWait(uneven)

    if get_use_sync_collectives():
        return NoWait(all2all_pooled_sync(group, a2ai, a2a_pooled_embs_tensor))

    myreq = Request(group, device=a2a_pooled_embs_tensor.device)
    if all_to_all_single_comm is None:
        all_to_all_single_comm = DefaultAll2AllSingle()
    All2All_Pooled_Req.apply(
        group,
        myreq,
        a2ai,
        a2a_pooled_embs_tensor,
        all_to_all_single_comm,
    )
    return myreq


def pg_name(pg: dist.ProcessGroup) -> str:
    # pyrefly: ignore[implicit-import]
    return dist._functional_collectives._resolve_group_name(pg, "")


@torch.compiler.assume_constant_result
def is_fake_pg(pg: dist.ProcessGroup) -> bool:
    """
    Whether ``pg`` uses the ``fake`` backend on every device. Mixed backends
    such as ``"cpu:fake,cuda:nccl2"`` are not fake.

    Raises ``ValueError`` if ``pg`` is not registered with torch.distributed.
    """
    backend = dist.get_backend(pg)
    return all(entry.rsplit(":", 1)[-1] == "fake" for entry in backend.split(","))


def all2all_pooled_sync(
    pg: dist.ProcessGroup, a2ai: All2AllPooledInfo, input_embeddings: Tensor
) -> Tensor:
    my_rank = pg.rank()

    (B_global, D_local_sum) = input_embeddings.shape

    dim_sum_per_rank = a2ai.dim_sum_per_rank
    batch_size_per_rank = a2ai.batch_size_per_rank
    B_local = batch_size_per_rank[my_rank]

    assert B_global == sum(batch_size_per_rank)

    if a2ai.codecs is not None:
        codecs = none_throws(a2ai.codecs)
        qcomm_ctx = codecs.forward.create_context()
        # Compute padding for quantized comm (e.g. FP8 rowwise, MX4) to ensure
        # dim_sum per rank is aligned to the quantization group size.
        # Mirrors the padding logic in All2All_Pooled_Req.forward().
        padded_D_local_sum, padding_size = codecs.forward.padded_size(
            input_embeddings, dim_sum_per_rank, my_rank, qcomm_ctx
        )

        if padding_size == 0:
            sharded_input_embeddings = input_embeddings.view(-1)
        else:
            sharded_input_embeddings = torch.nn.functional.pad(
                input_embeddings, (0, padding_size)
            )
        sharded_input_embeddings = codecs.forward.encode(
            sharded_input_embeddings,
            qcomm_ctx,
        )
        padded_dim_sum_per_rank = (
            qcomm_ctx.padded_dim_sum_per_rank
            if qcomm_ctx is not None and qcomm_ctx.padded_dim_sum_per_rank is not None
            else dim_sum_per_rank
        )
        output_split_sizes = [
            codecs.forward.calc_quantized_size(
                B_local * D_rank_sum,
                qcomm_ctx,
            )
            for D_rank_sum in padded_dim_sum_per_rank
        ]
        input_split_sizes = [
            codecs.forward.calc_quantized_size(
                padded_D_local_sum * B_rank,
                qcomm_ctx,
            )
            for B_rank in batch_size_per_rank
        ]
    else:
        sharded_input_embeddings = input_embeddings.view(-1)
        output_split_sizes = [B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank]
        input_split_sizes = [D_local_sum * B_rank for B_rank in batch_size_per_rank]
        qcomm_ctx = None

    with record_function("## all2all_pooled ##"):
        sharded_output_embeddings = AllToAllSingle.apply(
            sharded_input_embeddings,
            output_split_sizes,
            input_split_sizes,
            pg_name(pg),
            pg.size(),
            get_gradient_division(),
        )

    if a2ai.codecs is not None:
        codecs = none_throws(a2ai.codecs)
        sharded_output_embeddings = codecs.forward.decode(
            sharded_output_embeddings,
            qcomm_ctx,
        )

    padded_dim_sum_per_rank = (
        qcomm_ctx.padded_dim_sum_per_rank
        if qcomm_ctx is not None and qcomm_ctx.padded_dim_sum_per_rank is not None
        else dim_sum_per_rank
    )

    if is_torchdynamo_compiling():
        # Default implementation fails on backward with unbacked symints in full Ads model compilation.
        # This is workaround to do desired split-cat under tracing in custom_op.
        # TODO: Remove when pt2 symbolic shapes default split - cat can be compiled forward and backward with unbacked symints
        if qcomm_ctx is not None and qcomm_ctx.padded_dim_sum_per_rank is not None:
            # When FP8 rowwise padding is applied, the output tensor contains
            # padded data. Remove padding per-rank before the split-cat op.
            outputs_by_rank = sharded_output_embeddings.split(
                [B_local * D_rank_sum for D_rank_sum in padded_dim_sum_per_rank]
            )
            sharded_output_embeddings = torch.cat(
                [
                    output.view(B_local, -1)[:, :dim_sum].reshape(-1)
                    for output, dim_sum in zip(outputs_by_rank, dim_sum_per_rank)
                ]
            )
        return torch.ops.torchrec._split_1d_cat_2d(
            sharded_output_embeddings, B_local, dim_sum_per_rank
        )

    outputs_by_rank = sharded_output_embeddings.split(
        [B_local * D_rank_sum for D_rank_sum in padded_dim_sum_per_rank]
    )
    if qcomm_ctx is not None and qcomm_ctx.padded_dim_sum_per_rank is not None:
        # Remove padding added for quantization alignment.
        # Mirrors the unpadding logic in All2All_Pooled_Wait.forward().
        outputs_by_rank = [
            output.view(B_local, -1)[:, :dim_sum]
            for output, dim_sum in zip(outputs_by_rank, dim_sum_per_rank)
        ]
    return torch.cat(
        [
            output.view(B_local, dim)
            for output, dim in zip(outputs_by_rank, dim_sum_per_rank)
        ],
        dim=1,
    )


def variable_batch_alltoall_pooled(
    a2a_pooled_embs_tensor: Tensor,
    batch_size_per_rank_per_feature: List[List[int]],
    batch_size_per_feature_pre_a2a: List[int],
    emb_dim_per_rank_per_feature: List[List[int]],
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
    all_to_all_single_comm: Optional[All2AllSingle] = None,
) -> Awaitable[Tensor]:

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(a2a_pooled_embs_tensor)

    a2ai = VariableBatchAll2AllPooledInfo(
        batch_size_per_rank_per_feature=batch_size_per_rank_per_feature,
        batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
        emb_dim_per_rank_per_feature=emb_dim_per_rank_per_feature,
        codecs=codecs,
    )

    if get_use_sync_collectives():
        return NoWait(
            variable_batch_all2all_pooled_sync(group, a2ai, a2a_pooled_embs_tensor)
        )

    myreq = Request(group, device=a2a_pooled_embs_tensor.device)
    if all_to_all_single_comm is None:
        all_to_all_single_comm = DefaultAll2AllSingle()
    Variable_Batch_All2All_Pooled_Req.apply(
        group, myreq, a2ai, a2a_pooled_embs_tensor, all_to_all_single_comm
    )
    return myreq


def variable_batch_all2all_pooled_sync(
    pg: dist.ProcessGroup,
    a2ai: VariableBatchAll2AllPooledInfo,
    input_embeddings: Tensor,
) -> Tensor:
    my_rank = pg.rank()

    # get input splits
    world_size = dist.get_world_size(pg)
    input_split_sizes = [0 for _ in range(world_size)]
    if a2ai.batch_size_per_rank_per_feature:
        for i in range(world_size):
            curr_size = 0
            for batch_size, emb_dim in zip(
                a2ai.batch_size_per_rank_per_feature[i],
                a2ai.emb_dim_per_rank_per_feature[my_rank],
            ):
                curr_size += batch_size * emb_dim
            input_split_sizes[i] = curr_size
    a2ai.input_splits = input_split_sizes

    # get output splits
    output_split_sizes = [0 for _ in range(world_size)]
    ind = 0
    for i in range(world_size):
        curr_size = 0
        for emb_dim in a2ai.emb_dim_per_rank_per_feature[i]:
            curr_size += a2ai.batch_size_per_feature_pre_a2a[ind] * emb_dim
            ind += 1
        output_split_sizes[i] = curr_size
    a2ai.output_splits = output_split_sizes

    sharded_input_embeddings = input_embeddings.view(-1)
    qcomm_ctx = None

    if a2ai.codecs is not None:
        codecs = none_throws(a2ai.codecs)
        qcomm_ctx = codecs.forward.create_context()
        padded_input_split_sizes = _qcomm_padded_splits(
            codecs.forward, qcomm_ctx, input_split_sizes
        )
        if padded_input_split_sizes is not None:
            sharded_input_embeddings = _pad_splits(
                sharded_input_embeddings, input_split_sizes, padded_input_split_sizes
            )
            input_split_sizes = padded_input_split_sizes
        padded_output_split_sizes = _qcomm_padded_splits(
            codecs.forward, qcomm_ctx, output_split_sizes
        )
        if padded_output_split_sizes is not None:
            output_split_sizes = padded_output_split_sizes

        sharded_input_embeddings = codecs.forward.encode(
            sharded_input_embeddings,
            qcomm_ctx,
        )
        output_split_sizes = [
            codecs.forward.calc_quantized_size(
                split,
                qcomm_ctx,
            )
            for split in output_split_sizes
        ]
        input_split_sizes = [
            codecs.forward.calc_quantized_size(
                split,
                qcomm_ctx,
            )
            for split in input_split_sizes
        ]

    with record_function("## variable_batch_all2all_pooled ##"):
        if is_fake_pg(pg):
            sharded_output_embeddings = torch.empty(
                sum(output_split_sizes),
                device=sharded_input_embeddings.device,
                dtype=sharded_input_embeddings.dtype,
            )
            s0 = sharded_output_embeddings.size(0)
            # Bad assumption that our rank GE than other
            torch._check(s0 <= sharded_input_embeddings.size(0))
            sharded_output_embeddings.copy_(sharded_input_embeddings[:s0])
        else:
            sharded_output_embeddings = AllToAllSingle.apply(
                sharded_input_embeddings,
                output_split_sizes,
                input_split_sizes,
                pg_name(pg),
                pg.size(),
                get_gradient_division(),
            )

    if a2ai.codecs is not None:
        codecs = none_throws(a2ai.codecs)
        sharded_output_embeddings = codecs.forward.decode(
            sharded_output_embeddings,
            qcomm_ctx,
        )
        splits = none_throws(a2ai.output_splits)
        padded_splits = _qcomm_padded_splits(codecs.forward, qcomm_ctx, splits)
        if padded_splits is not None:
            sharded_output_embeddings = _strip_splits(
                sharded_output_embeddings, padded_splits, splits
            )
    return sharded_output_embeddings


def alltoall_sequence(
    # (T, B, L_i * D) flattened
    a2a_sequence_embs_tensor: Tensor,
    forward_recat_tensor: Tensor,
    backward_recat_tensor: Tensor,
    lengths_after_sparse_data_all2all: Tensor,
    input_splits: List[int],
    output_splits: List[int],
    variable_batch_size: bool = False,
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
    precomputed_send: Optional["SeqA2APrecomputedSend"] = None,
) -> Awaitable[Tensor]:
    """
    Performs AlltoAll operation for sequence embeddings. Each process splits the input
    tensor based on the world size, and then scatters the split list to all processes in
    the group. Then concatenates the received tensors from all processes in the group
    and returns a single output tensor.

    NOTE:
        AlltoAll operator for Sequence embedding tensors.
        Does not support mixed dimensions.

    Args:
        a2a_sequence_embs_tensor (Tensor): input embeddings.
        forward_recat_tensor (Tensor): recat tensor for forward.
        backward_recat_tensor (Tensor): recat tensor for backward.
        lengths_after_sparse_data_all2all (Tensor): lengths of sparse features after
            AlltoAll.
        input_splits (List[int]): input splits.
        output_splits (List[int]): output splits.
        variable_batch_size (bool): whether variable batch size is enabled.
        group (Optional[dist.ProcessGroup]): the process group to work on. If None, the
            default process group will be used.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.
        precomputed_send (Optional[SeqA2APrecomputedSend]): forward permute and
            encode of ``a2a_sequence_embs_tensor`` already run by
            `precompute_seq_a2a_send`; the caller must have checked it matches.

    Returns:
        Awaitable[List[Tensor]]: async work handle (`Awaitable`), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `alltoall_sequence` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(a2a_sequence_embs_tensor)

    a2ai = All2AllSequenceInfo(
        embedding_dim=a2a_sequence_embs_tensor.shape[1],
        lengths_after_sparse_data_all2all=lengths_after_sparse_data_all2all,
        forward_recat_tensor=forward_recat_tensor,
        backward_recat_tensor=backward_recat_tensor,
        input_splits=input_splits,
        output_splits=output_splits,
        variable_batch_size=variable_batch_size,
        codecs=codecs,
        precomputed_send=precomputed_send,
    )
    # sequence of embeddings, bags are definitely non-uniform

    if get_use_sync_collectives():
        return NoWait(all2all_sequence_sync(group, a2ai, a2a_sequence_embs_tensor))

    myreq = Request(group, device=a2a_sequence_embs_tensor.device)
    All2All_Seq_Req.apply(group, myreq, a2ai, a2a_sequence_embs_tensor)
    return myreq


def all2all_sequence_sync(
    pg: dist.ProcessGroup,
    a2ai: All2AllSequenceInfo,
    sharded_input_embeddings: Tensor,
) -> Tensor:
    world_size = pg.size()
    D = a2ai.embedding_dim
    forward_recat_tensor = a2ai.forward_recat_tensor
    variable_batch_size = a2ai.variable_batch_size
    lengths_after_sparse_data_all2all = a2ai.lengths_after_sparse_data_all2all * D
    input_splits = [i * D for i in a2ai.output_splits]
    output_splits = [i * D for i in a2ai.input_splits]

    a2ai.input_splits = input_splits
    a2ai.output_splits = output_splits

    local_T = lengths_after_sparse_data_all2all.shape[0]
    if local_T > 0:
        with record_function("## all2all_sequence_permute ##"):
            if not variable_batch_size:
                (
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_input_embeddings,
                    _,
                ) = _permute_2D_sparse_data(
                    forward_recat_tensor,
                    lengths_after_sparse_data_all2all.view(local_T * world_size, -1),
                    sharded_input_embeddings.view(-1),
                    None,
                    sharded_input_embeddings.numel(),
                )
            else:
                (
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_input_embeddings,
                    _,
                ) = torch.ops.fbgemm.permute_1D_sparse_data(
                    forward_recat_tensor,
                    lengths_after_sparse_data_all2all.view(-1),
                    sharded_input_embeddings.view(-1),
                    None,
                    sharded_input_embeddings.numel(),
                )
    else:
        # Variable is not used in sync mode, left for conformity with async path
        permuted_lengths_after_sparse_data_all2all = None  # noqa: F841

    if a2ai.codecs is not None:
        codecs = none_throws(a2ai.codecs)
        qcomm_ctx = codecs.forward.create_context()
        sharded_input_embeddings = a2ai.codecs.forward.encode(
            sharded_input_embeddings, qcomm_ctx
        )
        output_splits = [
            a2ai.codecs.forward.calc_quantized_size(x, qcomm_ctx) for x in output_splits
        ]
        input_splits = [
            a2ai.codecs.forward.calc_quantized_size(x, qcomm_ctx) for x in input_splits
        ]
    else:
        qcomm_ctx = None

    with record_function("## all2all_sequence ##"):
        sharded_output_embeddings = AllToAllSingle.apply(
            sharded_input_embeddings,
            output_splits,
            input_splits,
            pg_name(pg),
            pg.size(),
            get_gradient_division(),
        )

    if a2ai.codecs is not None:
        codecs = none_throws(a2ai.codecs)
        sharded_output_embeddings = codecs.forward.decode(
            sharded_output_embeddings, qcomm_ctx
        )
    return sharded_output_embeddings.view(-1, D)


def reduce_scatter_pooled(
    inputs: List[Tensor],
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
) -> Awaitable[Tensor]:
    """
    Performs reduce-scatter operation for a pooled embeddings tensor split into world
    size number of chunks. The result of the reduce operation gets scattered to all
    processes in the group.

    Args:
        inputs (List[Tensor]): list of tensors to scatter, one per rank.
        group (Optional[dist.ProcessGroup]): the process group to work on. If None, the
            default process group will be used.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Returns:
        Awaitable[Tensor]: async work handle (Awaitable), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `reduce_scatter_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if group.size() <= 1:
        return NoWait(inputs[group.rank()])

    rsi = ReduceScatterInfo(
        input_sizes=[tensor.size() for tensor in inputs], codecs=codecs
    )

    if get_use_sync_collectives():
        return NoWait(reduce_scatter_sync(group, rsi, *inputs))

    myreq = Request(group, device=inputs[0].device)
    ReduceScatter_Req.apply(group, myreq, rsi, *inputs)
    return myreq


def reduce_scatter_sync(
    pg: dist.ProcessGroup,
    rsi: ReduceScatterInfo,
    *inputs: Any,
) -> Tensor:
    if rsi.codecs is not None:
        # pyrefly: ignore[bad-assignment]
        inputs = [rsi.codecs.forward.encode(input) for input in inputs]

    with record_function("## reduce_scatter ##"):
        output = torch.ops.torchrec.reduce_scatter_tensor(
            torch.cat(inputs),
            reduceOp="sum",
            group_size=pg.size(),
            group_name=pg_name(pg),
            gradient_division=get_gradient_division(),
        )
    if rsi.codecs is not None:
        output = rsi.codecs.forward.decode(output)
    return output


def reduce_scatter_base_pooled(
    input: Tensor,
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
) -> Awaitable[Tensor]:
    """
    Reduces then scatters a flattened pooled embeddings tensor to all processes in a
    group.
    Input tensor is of size `output_tensor_size * world_size`.

    Args:
        input (Tensor): flattened tensor to scatter.
        group (Optional[dist.ProcessGroup]): the process group to work on. If None, the
            default process group will be used.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Returns:
        Awaitable[Tensor]: async work handle (Awaitable), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `reduce_scatter_base_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(input)

    rsi = ReduceScatterBaseInfo(input_sizes=input.size(), codecs=codecs)

    if get_use_sync_collectives():
        return NoWait(reduce_scatter_base_sync(group, rsi, input))

    myreq = Request(group, device=input.device)
    ReduceScatterBase_Req.apply(group, myreq, rsi, input)
    return myreq


def reduce_scatter_base_sync(
    pg: dist.ProcessGroup,
    rsi: ReduceScatterBaseInfo,
    inputs: Tensor,
) -> Tensor:
    my_size = pg.size()
    assert inputs.size(0) % my_size == 0
    if rsi.codecs is not None:
        inputs = rsi.codecs.forward.encode(inputs)

    with record_function("## reduce_scatter_base ##"):
        output = torch.ops.torchrec.reduce_scatter_tensor(
            inputs,
            reduceOp="sum",
            group_size=pg.size(),
            group_name=pg_name(pg),
            gradient_division=get_gradient_division(),
        )
    if rsi.codecs is not None:
        output = rsi.codecs.forward.decode(output)
    return output


def all_gather_base_pooled(
    input: Tensor,
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
) -> Awaitable[Tensor]:
    """
    All-gathers tensors from all processes in a group to form a flattened pooled
    embeddings tensor.
    Input tensor is of size `output_tensor_size / world_size`.

    Args:
        input (Tensor): tensor to gather.
        group (Optional[dist.ProcessGroup]): the process group to work on. If None, the
            default process group will be used.

    Returns:
        Awaitable[Tensor]: async work handle (Awaitable), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `all_gather_base_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    agi = AllGatherBaseInfo(input_size=input.size(), codecs=codecs)
    if dist.get_world_size(group) <= 1:
        return NoWait(input)

    if get_use_sync_collectives():
        return NoWait(all_gather_base_sync(group, agi, input))

    myreq = Request(group, device=input.device)
    AllGatherBase_Req.apply(group, myreq, agi, input)
    return myreq


def all_gather_base_sync(
    pg: dist.ProcessGroup,
    agi: AllGatherBaseInfo,
    input: Tensor,
) -> Tensor:
    if agi.codecs is not None:
        input = agi.codecs.forward.encode(input)

    with record_function("## all_gather_base ##"):
        outputs = torch.ops.torchrec.all_gather_into_tensor(
            input,
            gather_dim=0,
            group_name=pg_name(pg),
            group_size=pg.size(),
            gradient_division=get_gradient_division(),
        )
    if agi.codecs is not None:
        outputs = agi.codecs.forward.decode(outputs)
    return outputs


def reduce_scatter_v_pooled(
    input: Tensor,
    input_splits: List[int],
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
) -> Awaitable[Tensor]:
    """
    Performs reduce-scatter-v operation for a pooled embeddings tensor split unevenly
    into world size number of chunks. The result of the reduce operation gets scattered
    to all processes in the group according to `input_splits`.

    Args:
        input (Tensor): tensor to scatter.
        input_splits (List[int]): input splits.
        group (Optional[dist.ProcessGroup]): the process group to work on. If None, the
            default process group will be used.

    Returns:
        Awaitable[Tensor]: async work handle (Awaitable), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `reduce_scatter_v_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    if dist.get_world_size(group) <= 1:
        return NoWait(input)

    input_size = list(input.size())
    input_sizes = [
        [ip_split if d == 0 else input_size[d] for d in range(len(input_size))]
        for ip_split in input_splits
    ]

    equal_splits = False
    if not is_torchdynamo_compiling():
        # We can not check during tracing equality of splits -> fallback on general
        equal_splits = all(ip_split == input_splits[0] for ip_split in input_splits)

    rsvi = ReduceScatterVInfo(
        input_sizes=input_sizes,
        input_splits=input_splits,
        equal_splits=equal_splits,
        total_input_size=input_size,
        codecs=codecs,
    )

    if get_use_sync_collectives():
        return NoWait(reduce_scatter_v_sync(group, rsvi, input))

    myreq = Request(group, device=input.device)
    ReduceScatterV_Req.apply(group, myreq, rsvi, input)
    return myreq


def reduce_scatter_v_sync(
    pg: dist.ProcessGroup,
    rsi: ReduceScatterVInfo,
    input: Tensor,
) -> Tensor:
    world_size = pg.size()
    rank = pg.rank()

    if rsi.codecs is not None:
        input = rsi.codecs.forward.encode(input)

    if rsi.equal_splits:
        with record_function("## reduce_scatter_v ##"):
            output = torch.ops.torchrec.reduce_scatter_tensor(
                input,
                reduceOp="sum",
                group_size=pg.size(),
                group_name=pg_name(pg),
                gradient_division=get_gradient_division(),
            )
    else:
        with record_function("## reduce_scatter_v (AllToAllSingle) ##"):
            input_splits = rsi.input_splits
            output_splits = [rsi.input_splits[rank]] * world_size
            # TODO(ivankobzarev): Replace with _functional_collectives.reduce_scatter_v when it is added
            a2a_output = AllToAllSingle.apply(
                input,
                output_splits,
                input_splits,
                pg_name(pg),
                pg.size(),
                get_gradient_division(),
            )
            output = torch.sum(
                torch.stack(torch.split(a2a_output, output_splits)), dim=0
            )

    if rsi.codecs is not None:
        output = rsi.codecs.forward.decode(output)

    return output


def reduce_scatter_v_per_feature_pooled(
    input: Tensor,
    batch_size_per_rank_per_feature: List[List[int]],
    embedding_dims: List[int],
    group: Optional[dist.ProcessGroup] = None,
    codecs: Optional[QuantizedCommCodecs] = None,
) -> Awaitable[Tensor]:
    """
    Performs reduce-scatter-v operation for a 1-d pooled embeddings tensor of variable
    batch size per feature split unevenly into world size number of chunks. The result
    of the reduce operation gets scattered to all processes in the group.

    Args:
        input (Tensor): tensors to scatter, one per rank.
        batch_size_per_rank_per_feature (List[List[int]]): batch size per rank per
            feature used to determine input splits.
        embedding_dims (List[int]): embedding dimensions per feature used to determine
            input splits.
        group (Optional[dist.ProcessGroup]): The process group to work on. If None, the
            default process group will be used.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Returns:
        Awaitable[Tensor]: async work handle (Awaitable), which can be `wait()` later to get the resulting tensor.

    .. warning::
        `reduce_scatter_v_per_feature_pooled` is experimental and subject to change.
    """

    if group is None:
        group = dist.distributed_c10d._get_default_group()

    world_size = group.size()
    if world_size <= 1:
        return NoWait(input)

    input_splits = [0 for _ in range(world_size)]
    if batch_size_per_rank_per_feature:
        for rank in range(world_size):
            rank_splits = 0
            for batch_size, emb_dim in zip(
                batch_size_per_rank_per_feature[rank], embedding_dims
            ):
                rank_splits += batch_size * emb_dim
            input_splits[rank] = rank_splits
    input_sizes = [[s] for s in input_splits]

    rsvi = ReduceScatterVInfo(
        input_sizes=input_sizes,
        input_splits=input_splits,
        equal_splits=False,
        total_input_size=list(input.size()),
        codecs=codecs,
    )

    if get_use_sync_collectives():
        return NoWait(reduce_scatter_v_sync(group, rsvi, input))

    myreq = Request(group, device=input.device)
    ReduceScatterV_Req.apply(group, myreq, rsvi, input)
    return myreq


# The fbgemm kernels only have float and half dispatch arms.
_FBGEMM_RECAT_DTYPES: Tuple[torch.dtype, ...] = (torch.float32, torch.float16)


# TODO: improve performance of _recat_pooled_embedding_grad_out, see T87591139
def _recat_pooled_embedding_grad_out(
    grad_output: Tensor,
    num_features_per_rank: List[int],
    dim_sum_per_rank_tensor: Optional[Tensor] = None,
    cumsum_dim_sum_per_rank_tensor: Optional[Tensor] = None,
) -> Tensor:
    """
    Regroups a ``[B_local, sum(num_features_per_rank)]`` gradient, whose columns are
    ordered by rank, into the flat rank-major send buffer the backward all-to-all
    expects: ``[rank0 block][rank1 block]...``, each block ``[B_local, D_rank]``
    flattened row-major.

    ``dim_sum_per_rank_tensor`` and ``cumsum_dim_sum_per_rank_tensor`` select the
    single-kernel fbgemm path; both must be int64 and the cumsum must be
    *exclusive* (``[0, d0, d0 + d1, ...]``), which is what the kernel indexes as
    each rank's starting column. Passing an inclusive cumsum silently shifts every
    rank's stripe by one slot.
    """
    if USE_FBGEMM_RECAT and grad_output.dtype in _FBGEMM_RECAT_DTYPES:
        if (
            grad_output.is_cuda
            and dim_sum_per_rank_tensor is not None
            and cumsum_dim_sum_per_rank_tensor is not None
            and dim_sum_per_rank_tensor.dtype == torch.int64
            and cumsum_dim_sum_per_rank_tensor.dtype == torch.int64
        ):
            torch._C._log_api_usage_once(
                "torchrec.distributed.comm_ops.recat_pooled_embedding_grad_out."
                "recat_embedding_grad_output_mixed_D_batch"
            )
            return torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
                grad_output,
                dim_sum_per_rank_tensor,
                cumsum_dim_sum_per_rank_tensor,
            )
        torch._C._log_api_usage_once(
            "torchrec.distributed.comm_ops.recat_pooled_embedding_grad_out."
            "recat_embedding_grad_output_mixed_D"
        )
        return torch.ops.fbgemm.recat_embedding_grad_output_mixed_D(
            grad_output, num_features_per_rank
        )
    grad_outputs_by_rank = grad_output.split(num_features_per_rank, dim=1)
    return torch.cat(
        [
            grad_output_by_rank.contiguous().view(-1)
            for grad_output_by_rank in grad_outputs_by_rank
        ],
        dim=0,
    )


def _recat_seq_embedding(
    input_embeddings: Tensor,
    split_sizes: List[int],
    T_local: int,
    my_size: int,
    forward: bool,
) -> Tensor:
    seq_embeddings_by_rank = input_embeddings.split(split_sizes)
    if forward:
        return torch.cat(
            [
                seq_embeddings_by_rank[t * my_size + i]
                # .contiguous().view(-1)
                for i in range(my_size)
                for t in range(T_local)
            ],
            dim=0,
        )
    else:
        return torch.cat(
            [
                seq_embeddings_by_rank[i * T_local + t]
                # .contiguous()
                # .view(-1)
                for t in range(T_local)
                for i in range(my_size)
            ],
            dim=0,
        )


class All2All_Pooled_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: All2AllPooledInfo,
        input_embeddings: Tensor,
        all_to_all_single_comm: All2AllSingle,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)
        (B_global, D_local_sum) = input_embeddings.shape

        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank
        B_local = batch_size_per_rank[my_rank]

        assert B_global == sum(batch_size_per_rank)

        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            qcomm_ctx = codecs.forward.create_context()
            padded_D_local_sum, padding_size = codecs.forward.padded_size(
                input_embeddings, dim_sum_per_rank, my_rank, qcomm_ctx
            )

            if padding_size == 0:
                sharded_input_embeddings = input_embeddings.view(-1)
            else:
                sharded_input_embeddings = torch.nn.functional.pad(
                    input_embeddings, (0, padding_size)
                )
            sharded_input_embeddings = codecs.forward.encode(
                sharded_input_embeddings,
                qcomm_ctx,
            )
            padded_dim_sum_per_rank = (
                qcomm_ctx.padded_dim_sum_per_rank
                if qcomm_ctx is not None
                and qcomm_ctx.padded_dim_sum_per_rank is not None
                else dim_sum_per_rank
            )
            output_split_sizes = [
                codecs.forward.calc_quantized_size(
                    B_local * D_rank_sum,
                    qcomm_ctx,
                )
                for D_rank_sum in padded_dim_sum_per_rank
            ]
            input_split_sizes = [
                codecs.forward.calc_quantized_size(
                    padded_D_local_sum * B_rank,
                    qcomm_ctx,
                )
                for B_rank in batch_size_per_rank
            ]
        else:
            sharded_input_embeddings = input_embeddings.view(-1)
            output_split_sizes = [
                B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank
            ]
            input_split_sizes = [D_local_sum * B_rank for B_rank in batch_size_per_rank]
            qcomm_ctx = None

        sharded_output_embeddings = all_to_all_single_comm.allocate(
            (sum(output_split_sizes),),
            dtype=sharded_input_embeddings.dtype,
            device=sharded_input_embeddings.device,
        )

        sharded_input_embeddings_registered = all_to_all_single_comm.allocate(
            sharded_input_embeddings.shape,
            dtype=sharded_input_embeddings.dtype,
            device=sharded_input_embeddings.device,
        )

        sharded_input_embeddings_registered.copy_(sharded_input_embeddings)

        with record_function("## All2All_Pooled_fwd ##"):
            req = all_to_all_single_comm(
                output_tensor=sharded_output_embeddings,
                input_tensor=sharded_input_embeddings_registered,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = sharded_output_embeddings
        myreq.qcomm_ctx = qcomm_ctx
        myreq.a2ai = a2ai
        myreq.wait_function = All2All_Pooled_Wait
        ctx.myreq = myreq
        ctx.pg = pg
        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused) -> Tuple[None, None, None, Tensor, None]:
        pg = ctx.pg
        my_rank = dist.get_rank(pg)
        myreq = ctx.myreq
        a2ai = myreq.a2ai
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        grad_output = myreq.tensor
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank
        D_local_sum = dim_sum_per_rank[my_rank]
        B_global = sum(batch_size_per_rank)
        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            grad_input = codecs.backward.decode(grad_output, myreq.qcomm_ctx)
            splits = [D_local_sum * B_rank for B_rank in batch_size_per_rank]
            padded_splits = _qcomm_padded_splits(
                codecs.backward, myreq.qcomm_ctx, splits
            )
            if padded_splits is not None:
                grad_input = _strip_splits(grad_input, padded_splits, splits)
            grad_input = grad_input.view(B_global, D_local_sum)
        else:
            grad_input = grad_output.view(B_global, D_local_sum)
        if GRADIENT_DIVISION:
            grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        myreq.dummy_tensor = None
        return (None, None, None, grad_input, None)


class All2All_Pooled_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: Tensor,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        assert myreq.req is not None
        myreq.req.wait()
        sharded_output_embeddings = myreq.tensor
        myreq.req = None
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq
        # pyrefly: ignore[missing-attribute]
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        # pyrefly: ignore[missing-attribute]
        batch_size_per_rank = a2ai.batch_size_per_rank
        B_local = batch_size_per_rank[my_rank]

        # pyrefly: ignore[missing-attribute]
        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            # pyrefly: ignore[missing-attribute]
            sharded_output_embeddings = codecs.forward.decode(
                sharded_output_embeddings,
                myreq.qcomm_ctx,
            )

        padded_dim_sum_per_rank = (
            myreq.qcomm_ctx.padded_dim_sum_per_rank
            if myreq.qcomm_ctx is not None
            and myreq.qcomm_ctx.padded_dim_sum_per_rank is not None
            else dim_sum_per_rank
        )
        # pyrefly: ignore[missing-attribute]
        outputs_by_rank = sharded_output_embeddings.split(
            [B_local * D_rank_sum for D_rank_sum in padded_dim_sum_per_rank]
        )
        final_dim_sum_per_rank = padded_dim_sum_per_rank
        if (
            myreq.qcomm_ctx is not None
            and myreq.qcomm_ctx.padded_dim_sum_per_rank is not None
        ):
            outputs_by_rank = [
                output.view(B_local, -1)[:, :dim_sum]
                for output, dim_sum in zip(outputs_by_rank, dim_sum_per_rank)
            ]
            final_dim_sum_per_rank = dim_sum_per_rank

        result = torch.cat(
            [
                output.view(B_local, dim)
                for output, dim in zip(outputs_by_rank, final_dim_sum_per_rank)
            ],
            dim=1,
        )
        return result

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        a2ai = ctx.a2ai
        pg = ctx.pg
        my_rank = dist.get_rank(pg)
        dim_sum_per_rank = a2ai.dim_sum_per_rank
        batch_size_per_rank = a2ai.batch_size_per_rank

        D_local_sum = dim_sum_per_rank[my_rank]
        (B_local, D_global_sum) = grad_output.shape
        assert sum(dim_sum_per_rank) == D_global_sum

        sharded_grad_output = _recat_pooled_embedding_grad_out(
            grad_output.contiguous(),
            dim_sum_per_rank,
            a2ai.recat_dim_sum_per_rank_tensor,
            a2ai.recat_cumsum_dim_sum_per_rank_tensor,
        )

        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            qcomm_ctx = codecs.backward.create_context()
            input_split_sizes = [
                B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank
            ]
            output_split_sizes = [
                D_local_sum * B_rank for B_rank in batch_size_per_rank
            ]
            padded_input_split_sizes = _qcomm_padded_splits(
                codecs.backward, qcomm_ctx, input_split_sizes
            )
            if padded_input_split_sizes is not None:
                sharded_grad_output = _pad_splits(
                    sharded_grad_output, input_split_sizes, padded_input_split_sizes
                )
                input_split_sizes = padded_input_split_sizes
            padded_output_split_sizes = _qcomm_padded_splits(
                codecs.backward, qcomm_ctx, output_split_sizes
            )
            if padded_output_split_sizes is not None:
                output_split_sizes = padded_output_split_sizes
            sharded_grad_output = codecs.backward.encode(
                sharded_grad_output,
                qcomm_ctx,
            )
            input_split_sizes = [
                codecs.backward.calc_quantized_size(split, qcomm_ctx)
                for split in input_split_sizes
            ]
            output_split_sizes = [
                codecs.backward.calc_quantized_size(split, qcomm_ctx)
                for split in output_split_sizes
            ]
        else:
            qcomm_ctx = None
            input_split_sizes = [
                B_local * D_rank_sum for D_rank_sum in dim_sum_per_rank
            ]
            output_split_sizes = [
                D_local_sum * B_rank for B_rank in batch_size_per_rank
            ]

        sharded_grad_input = torch.empty(
            sum(output_split_sizes),
            device=sharded_grad_output.device,
            dtype=sharded_grad_output.dtype,
        )
        with record_function("## All2All_Pooled_bwd ##"):
            req = dist.all_to_all_single(
                output=sharded_grad_input,
                input=sharded_grad_output,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = sharded_grad_input
        myreq.qcomm_ctx = qcomm_ctx

        return (None, None, myreq.dummy_tensor)


class Variable_Batch_All2All_Pooled_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: VariableBatchAll2AllPooledInfo,
        input_embeddings: Tensor,
        all_to_all_single_comm: All2AllSingle,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)

        # get input splits
        world_size = dist.get_world_size(pg)
        input_split_sizes = [0 for _ in range(world_size)]
        if a2ai.batch_size_per_rank_per_feature:
            for i in range(world_size):
                curr_size = 0
                for batch_size, emb_dim in zip(
                    a2ai.batch_size_per_rank_per_feature[i],
                    a2ai.emb_dim_per_rank_per_feature[my_rank],
                ):
                    curr_size += batch_size * emb_dim
                input_split_sizes[i] = curr_size
        a2ai.input_splits = input_split_sizes

        # get output splits
        output_split_sizes = [0 for _ in range(world_size)]
        ind = 0
        for i in range(world_size):
            curr_size = 0
            for emb_dim in a2ai.emb_dim_per_rank_per_feature[i]:
                curr_size += a2ai.batch_size_per_feature_pre_a2a[ind] * emb_dim
                ind += 1
            output_split_sizes[i] = curr_size
        a2ai.output_splits = output_split_sizes

        sharded_input_embeddings = input_embeddings.view(-1)
        qcomm_ctx = None

        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            qcomm_ctx = codecs.forward.create_context()
            padded_input_split_sizes = _qcomm_padded_splits(
                codecs.forward, qcomm_ctx, input_split_sizes
            )
            if padded_input_split_sizes is not None:
                sharded_input_embeddings = _pad_splits(
                    sharded_input_embeddings,
                    input_split_sizes,
                    padded_input_split_sizes,
                )
                input_split_sizes = padded_input_split_sizes
            padded_output_split_sizes = _qcomm_padded_splits(
                codecs.forward, qcomm_ctx, output_split_sizes
            )
            if padded_output_split_sizes is not None:
                output_split_sizes = padded_output_split_sizes

            sharded_input_embeddings = codecs.forward.encode(
                sharded_input_embeddings,
                qcomm_ctx,
            )
            output_split_sizes = [
                codecs.forward.calc_quantized_size(
                    split,
                    qcomm_ctx,
                )
                for split in output_split_sizes
            ]
            input_split_sizes = [
                codecs.forward.calc_quantized_size(
                    split,
                    qcomm_ctx,
                )
                for split in input_split_sizes
            ]

        sharded_output_embeddings = all_to_all_single_comm.allocate(
            (sum(output_split_sizes),),
            dtype=sharded_input_embeddings.dtype,
            device=sharded_input_embeddings.device,
        )

        sharded_input_embeddings_registered = all_to_all_single_comm.allocate(
            sharded_input_embeddings.shape,
            dtype=sharded_input_embeddings.dtype,
            device=sharded_input_embeddings.device,
        )

        sharded_input_embeddings_registered.copy_(sharded_input_embeddings)

        with record_function("## Variable_Batch_All2All_Pooled_fwd ##"):
            req = all_to_all_single_comm(
                output_tensor=sharded_output_embeddings,
                input_tensor=sharded_input_embeddings_registered,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = sharded_output_embeddings
        myreq.qcomm_ctx = qcomm_ctx
        myreq.a2ai = a2ai
        myreq.wait_function = Variable_Batch_All2All_Pooled_Wait
        ctx.myreq = myreq
        ctx.pg = pg
        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused) -> Tuple[None, None, None, Tensor, None]:
        myreq = ctx.myreq
        a2ai = myreq.a2ai
        assert myreq.req is not None
        myreq.req.wait()
        if isinstance(myreq.req, dist.Work):
            myreq.req.wait()

        myreq.req = None
        grad_output = myreq.tensor

        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            grad_input = codecs.backward.decode(grad_output, myreq.qcomm_ctx)
            splits = none_throws(a2ai.input_splits)
            padded_splits = _qcomm_padded_splits(
                codecs.backward, myreq.qcomm_ctx, splits
            )
            if padded_splits is not None:
                grad_input = _strip_splits(grad_input, padded_splits, splits)
        else:
            grad_input = grad_output
        if GRADIENT_DIVISION:
            grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        myreq.dummy_tensor = None
        return (None, None, None, grad_input, None)


class Variable_Batch_All2All_Pooled_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: Tensor,
    ) -> Tensor:
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        assert myreq.req is not None
        if isinstance(myreq.req, dist.Work):
            myreq.req.wait()
        sharded_output_embeddings = myreq.tensor
        myreq.req = None
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq

        # pyrefly: ignore[missing-attribute]
        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            # pyrefly: ignore[missing-attribute]
            sharded_output_embeddings = codecs.forward.decode(
                sharded_output_embeddings,
                myreq.qcomm_ctx,
            )
            # pyrefly: ignore[missing-attribute]
            splits = none_throws(a2ai.output_splits)
            padded_splits = _qcomm_padded_splits(
                # pyrefly: ignore[missing-attribute]
                codecs.forward,
                myreq.qcomm_ctx,
                splits,
            )
            if padded_splits is not None:
                sharded_output_embeddings = _strip_splits(
                    sharded_output_embeddings, padded_splits, splits
                )
        # the return result is a 1-d tensor, like: f_0_s_0, f_0_s1, ..., f_n_s_0, f_n_s_k
        # f_0, f_1, ... , f_n are ordered by features on each rank
        # pyrefly: ignore[bad-return]
        return sharded_output_embeddings

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        a2ai = ctx.a2ai
        pg = ctx.pg

        assert a2ai.input_splits is not None
        assert a2ai.output_splits is not None
        input_split_sizes = a2ai.output_splits
        output_split_sizes = a2ai.input_splits

        sharded_grad_output = grad_output.contiguous()
        qcomm_ctx = None

        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            qcomm_ctx = codecs.backward.create_context()
            padded_input_split_sizes = _qcomm_padded_splits(
                codecs.backward, qcomm_ctx, input_split_sizes
            )
            if padded_input_split_sizes is not None:
                sharded_grad_output = _pad_splits(
                    sharded_grad_output, input_split_sizes, padded_input_split_sizes
                )
                input_split_sizes = padded_input_split_sizes
            padded_output_split_sizes = _qcomm_padded_splits(
                codecs.backward, qcomm_ctx, output_split_sizes
            )
            if padded_output_split_sizes is not None:
                output_split_sizes = padded_output_split_sizes
            sharded_grad_output = codecs.backward.encode(
                sharded_grad_output,
                qcomm_ctx,
            )
            input_split_sizes = [
                codecs.backward.calc_quantized_size(
                    split,
                    qcomm_ctx,
                )
                for split in input_split_sizes
            ]
            output_split_sizes = [
                codecs.backward.calc_quantized_size(
                    split,
                    qcomm_ctx,
                )
                for split in output_split_sizes
            ]

        sharded_grad_input = torch.empty(
            sum(output_split_sizes),
            device=sharded_grad_output.device,
            dtype=sharded_grad_output.dtype,
        )
        with record_function("## Variable_Batch_All2All_Pooled_bwd ##"):
            req = dist.all_to_all_single(
                output=sharded_grad_input,
                input=sharded_grad_output,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = sharded_grad_input
        myreq.qcomm_ctx = qcomm_ctx

        return (None, None, myreq.dummy_tensor)


def _seq_a2a_permute_and_encode(
    sharded_input_embeddings: Tensor,
    lengths_after_sparse_data_all2all: Tensor,
    forward_recat_tensor: Optional[Tensor],
    variable_batch_size: bool,
    world_size: int,
    codecs: Optional[QuantizedCommCodecs],
) -> Tuple[Optional[Tensor], Tensor, Any]:
    local_T = lengths_after_sparse_data_all2all.shape[0]
    permuted_lengths_after_sparse_data_all2all = None
    if local_T > 0:
        with record_function("## All2All_Seq_fwd_permute ##"):
            if not variable_batch_size:
                (
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_input_embeddings,
                    _,
                ) = _permute_2D_sparse_data(
                    forward_recat_tensor,
                    lengths_after_sparse_data_all2all.view(local_T * world_size, -1),
                    sharded_input_embeddings.view(-1),
                    None,
                    sharded_input_embeddings.numel(),
                )
            else:
                (
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_input_embeddings,
                    _,
                ) = torch.ops.fbgemm.permute_1D_sparse_data(
                    forward_recat_tensor,
                    lengths_after_sparse_data_all2all.view(-1),
                    sharded_input_embeddings.view(-1),
                    None,
                    sharded_input_embeddings.numel(),
                )

    qcomm_ctx = None
    if codecs is not None:
        qcomm_ctx = codecs.forward.create_context()
        sharded_input_embeddings = codecs.forward.encode(
            sharded_input_embeddings, qcomm_ctx
        )
    return (
        permuted_lengths_after_sparse_data_all2all,
        sharded_input_embeddings,
        qcomm_ctx,
    )


@dataclass
class SeqA2APrecomputedSend:
    """Forward permute and encode of an `alltoall_sequence` input, run ahead.

    Attributes:
        local_embs (Tensor): the input it was computed from; holding it keeps its
            storage from being reused, so a matching data_ptr means the same rows.
        lengths (Tensor): lengths after the sparse data all-to-all it used.
        forward_recat_tensor (Tensor): recat tensor it used.
        recat_source (Tensor): tensor ``forward_recat_tensor`` was derived from:
            the sparse features recat it inverted, or ``forward_recat_tensor``.
        codecs (Optional[QuantizedCommCodecs]): codecs it used.
        permuted_lengths (Optional[Tensor]): permuted lengths (times embedding dim).
        send_tensor (Tensor): permuted and encoded all-to-all input.
        qcomm_ctx (Any): quantization context of the encode.
        ready (torch.Event): recorded on the producing stream after the encode.
    """

    local_embs: Tensor
    lengths: Tensor
    forward_recat_tensor: Tensor
    recat_source: Tensor
    codecs: Optional[QuantizedCommCodecs]
    permuted_lengths: Optional[Tensor]
    send_tensor: Tensor
    qcomm_ctx: Any
    ready: torch.Event

    def mismatch(
        self,
        local_embs: Tensor,
        lengths: Tensor,
        recat_source: Tensor,
        codecs: Optional[QuantizedCommCodecs],
    ) -> Optional[str]:
        """Returns why an all-to-all of these inputs cannot use this send, or None.

        The recat is compared by source tensor identity, not by the derived
        forward recat: the forward pass rebuilds that with `invert_permute` on
        every call, which is a deterministic function of the same, never
        mutated, source tensor.
        """
        if not (
            local_embs.data_ptr() == self.local_embs.data_ptr()
            and local_embs.shape == self.local_embs.shape
            and local_embs.stride() == self.local_embs.stride()
            and local_embs.dtype == self.local_embs.dtype
        ):
            return "rows differ"
        if not (
            lengths.data_ptr() == self.lengths.data_ptr()
            and lengths.shape == self.lengths.shape
            and lengths.stride() == self.lengths.stride()
        ):
            return "lengths differ"
        if recat_source is not self.recat_source:
            return "recat differs"
        if codecs is not self.codecs:
            return "codecs differ"
        return None


def precompute_seq_a2a_send(
    local_embs: Tensor,
    lengths: Tensor,
    forward_recat_tensor: Tensor,
    world_size: int,
    codecs: Optional[QuantizedCommCodecs],
    recat_source: Optional[Tensor] = None,
) -> SeqA2APrecomputedSend:
    """Runs the non-variable-batch `All2All_Seq_Req` permute and encode on the
    current stream, for a later all-to-all of the same ``local_embs``.
    ``recat_source`` is the tensor ``forward_recat_tensor`` was derived from."""
    permuted_lengths, send_tensor, qcomm_ctx = _seq_a2a_permute_and_encode(
        local_embs,
        lengths * local_embs.shape[1],
        forward_recat_tensor,
        False,
        world_size,
        codecs,
    )
    return SeqA2APrecomputedSend(
        local_embs=local_embs,
        lengths=lengths,
        forward_recat_tensor=forward_recat_tensor,
        recat_source=(forward_recat_tensor if recat_source is None else recat_source),
        codecs=codecs,
        permuted_lengths=permuted_lengths,
        send_tensor=send_tensor,
        qcomm_ctx=qcomm_ctx,
        ready=torch.get_device_module(local_embs.device)
        .current_stream()
        .record_event(),
    )


# Env TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD=1: with MX4 forward qcomms,
# All2All_Seq_Req.forward encodes the rows first and permutes the packed payload
# words. fbgemm's MX4 encode maps each group of mx_group_size values to
# mx_group_size // 2 + 1 contiguous bytes and shares nothing across groups, so
# when D % mx_group_size == 0 every recat segment is a whole number of groups
# and the payload is bitwise identical to permuting the rows first.
# TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_CHECK=1 also builds the old payload on the
# first few calls, compares it, and falls back to the old path for good on a
# mismatch.
_SEQ_A2A_PERMUTE_Q_CHECK_CALLS = 3
_seq_a2a_permute_q_checks = 0
_seq_a2a_permute_q_disabled = False
_seq_a2a_permute_q_logged: set[Tuple[int, Any, Optional[int]]] = set()


def _is_fbgemm_mx4_codec(codec: Any) -> bool:
    cls = type(codec)
    return (
        cls.__module__ == "fbgemm_gpu.quantize_comm"
        and cls.__name__ == "QuantizedCommCodec"
        and getattr(codec, "_comm_precision", None) == SparseType.MX4
    )


def _mx4_payload_row_words(codec: Any, qcomm_ctx: Any, D: int) -> Optional[int]:
    """int64 words per row of D values in the MX4 payload of `codec`, or None
    when permuting the payload in place of the rows is not exact or not aligned.
    """
    if not _is_fbgemm_mx4_codec(codec):
        return None
    rounding_mode = getattr(qcomm_ctx, "rounding_mode", None)
    # 3 is RoundingMode.stochastic: random bits indexed by group position.
    if rounding_mode is not None and int(rounding_mode) == 3:
        return None
    group_size = getattr(qcomm_ctx, "mx_group_size", 0)
    if group_size <= 0 or group_size % 2 != 0 or D % group_size != 0:
        return None
    row_bytes = D // 2 + D // group_size
    # 16-byte rows keep every segment on fbgemm's uint4 copy path.
    if row_bytes % 16 != 0:
        return None
    return row_bytes // 8


def _seq_a2a_fwd_permute_quantized(
    a2ai: All2AllSequenceInfo,
    sharded_input_embeddings: Tensor,
    lengths_after_sparse_data_all2all: Tensor,
    world_size: int,
) -> Optional[Tuple[Tensor, Tensor, Any]]:
    """TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD path of All2All_Seq_Req.forward.
    `lengths_after_sparse_data_all2all` is in elements (lengths * D).

    Returns (permuted lengths in elements, uint8 payload, qcomm ctx), or None
    for the old path. The self-check syncs (torch.equal) on each of the first
    _SEQ_A2A_PERMUTE_Q_CHECK_CALLS calls only.
    """
    global _seq_a2a_permute_q_checks, _seq_a2a_permute_q_disabled
    codecs = a2ai.codecs
    forward_recat_tensor = a2ai.forward_recat_tensor
    if (
        _seq_a2a_permute_q_disabled
        or codecs is None
        or forward_recat_tensor is None
        or a2ai.variable_batch_size
        or is_torchdynamo_compiling()
        or sharded_input_embeddings.numel() == 0
        or not sharded_input_embeddings.is_contiguous()
    ):
        return None
    D = a2ai.embedding_dim
    qcomm_ctx = codecs.forward.create_context()
    words = _mx4_payload_row_words(codecs.forward, qcomm_ctx, D)
    log_key = (D, getattr(qcomm_ctx, "mx_group_size", None), words)
    if log_key not in _seq_a2a_permute_q_logged:
        _seq_a2a_permute_q_logged.add(log_key)
        logger.info(
            "All2All_Seq_Req forward permutes the quantized payload: %s "
            "(D=%d mx_group_size=%s int64 words/row=%s)",
            words is not None,
            *log_key,
        )
    if words is None:
        return None
    local_T = lengths_after_sparse_data_all2all.shape[0]
    payload = _seq_a2a_codec_encode(
        codecs.forward,
        sharded_input_embeddings.view(-1),
        qcomm_ctx,
        "fwd_enc",
        "the payload-permute forward (TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD)",
    )
    permuted_lengths_words, permuted_words, _ = _permute_2D_sparse_data(
        forward_recat_tensor,
        (a2ai.lengths_after_sparse_data_all2all * words).view(local_T * world_size, -1),
        payload.view(torch.int64),
        None,
        payload.numel() // 8,
    )
    # Element units: the backward permutes its gradient with these lengths.
    permuted_lengths = permuted_lengths_words // words * D
    permuted_payload = permuted_words.view(torch.uint8)
    if (
        _seq_a2a_permute_q_checks < _SEQ_A2A_PERMUTE_Q_CHECK_CALLS
        and os.environ.get("TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_CHECK", "0") == "1"
        and not (payload.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        _seq_a2a_permute_q_checks += 1
        ref_lengths, ref_rows, _ = _permute_2D_sparse_data(
            forward_recat_tensor,
            lengths_after_sparse_data_all2all.view(local_T * world_size, -1),
            sharded_input_embeddings.view(-1),
            None,
            sharded_input_embeddings.numel(),
        )
        ref_payload = codecs.forward.encode(ref_rows, qcomm_ctx)
        ok = torch.equal(ref_lengths, permuted_lengths) and torch.equal(
            ref_payload, permuted_payload
        )
        logger.warning(
            "All2All_Seq_Req quantized-payload permute self-check %d/%d: "
            "rows=%d D=%d words/row=%d ok=%s",
            _seq_a2a_permute_q_checks,
            _SEQ_A2A_PERMUTE_Q_CHECK_CALLS,
            sharded_input_embeddings.numel() // D,
            D,
            words,
            ok,
        )
        if not ok:
            logger.error(
                "All2All_Seq_Req quantized-payload permute self-check FAILED; "
                "disabling TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD for the rest "
                "of the process"
            )
            _seq_a2a_permute_q_disabled = True
            return ref_lengths, ref_payload, qcomm_ctx
    return permuted_lengths, permuted_payload, qcomm_ctx


# Env TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD=1: with MX4 forward qcomms,
# All2All_Seq_Req.forward quantizes the rows straight into forward-recat order
# instead of permuting the rows (or the payload) and quantizing separately. With
# D % mx_group_size == 0 every row is a whole number of MX4 groups, each group is
# quantized on its own, and packed row j is the quantized input row row_idx[j],
# where row_idx is the forward recat applied to arange(rows); the kernel runs
# fbgemm's quantize arithmetic over the same work split, so the payload is
# bitwise identical to the old path's.
# TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD_CHECK=1 also builds the old payload on the
# first few calls, compares it, and falls back to the old path for good on a
# mismatch.
_SEQ_A2A_ROWMAP_FWD_CHECK_CALLS = 3
_seq_a2a_rowmap_fwd_checks = 0
_seq_a2a_rowmap_fwd_disabled = False
_seq_a2a_rowmap_fwd_logged: set[Tuple[int, int, int]] = set()


def _seq_a2a_fwd_quantize_rowmap(
    a2ai: All2AllSequenceInfo,
    sharded_input_embeddings: Tensor,
    lengths_after_sparse_data_all2all: Tensor,
    world_size: int,
) -> Optional[Tuple[Tensor, Tensor, Any]]:
    """TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD path of All2All_Seq_Req.forward.
    `lengths_after_sparse_data_all2all` is in elements (lengths * D).

    Returns (permuted lengths in elements, uint8 payload, qcomm ctx), or None
    for the old path. The self-check syncs (torch.equal) on each of the first
    _SEQ_A2A_ROWMAP_FWD_CHECK_CALLS calls only.
    """
    global _seq_a2a_rowmap_fwd_checks, _seq_a2a_rowmap_fwd_disabled
    codecs = a2ai.codecs
    forward_recat_tensor = a2ai.forward_recat_tensor
    if (
        _seq_a2a_rowmap_fwd_disabled
        or codecs is None
        or forward_recat_tensor is None
        or a2ai.variable_batch_size
        or is_torchdynamo_compiling()
        or sharded_input_embeddings.numel() == 0
        or not sharded_input_embeddings.is_contiguous()
        or sharded_input_embeddings.dtype not in (torch.float32, torch.bfloat16)
        or not _is_fbgemm_mx4_codec(codecs.forward)
        or getattr(codecs.forward, "_loss_scale", None) is not None
    ):
        return None
    codec = codecs.forward
    D = a2ai.embedding_dim
    qcomm_ctx = codec.create_context()
    group_size = getattr(qcomm_ctx, "mx_group_size", 0)
    rounding_mode = getattr(qcomm_ctx, "rounding_mode", None)
    # fp32_to_mx4 quantizes with RoundingMode.even when the context has none.
    rounding = 2 if rounding_mode is None else int(rounding_mode)
    # The kernel's tl.arange extents need a power-of-two group size, and each of
    # its steps of 64 groups (mx4_rowmap_quantize.MX4_ROWMAP_GROUP_LOAD) must
    # span at most two rows; 3 is RoundingMode.stochastic, whose random bits are
    # indexed by group position.
    exact = (
        group_size >= 2
        and group_size & (group_size - 1) == 0
        and D % group_size == 0
        and 64 * group_size <= D
        and rounding in (0, 1, 2, 4)
    )
    row_bytes = D // 2 + D // group_size if exact else 0
    log_key = (D, group_size, row_bytes)
    if log_key not in _seq_a2a_rowmap_fwd_logged:
        _seq_a2a_rowmap_fwd_logged.add(log_key)
        logger.info(
            "All2All_Seq_Req forward quantizes the MX4 rows through a row map: "
            "%s (D=%d mx_group_size=%d bytes/row=%d)",
            exact,
            *log_key,
        )
    if not exact or sharded_input_embeddings.numel() % D != 0:
        return None
    rows = sharded_input_embeddings.numel() // D
    local_T = lengths_after_sparse_data_all2all.shape[0]
    permuted_lengths_rows, row_idx, _ = _permute_2D_sparse_data(
        forward_recat_tensor,
        a2ai.lengths_after_sparse_data_all2all.view(local_T * world_size, -1),
        torch.arange(rows, dtype=torch.int64, device=sharded_input_embeddings.device),
        None,
        rows,
    )
    # Element units: the backward permutes its gradient with these lengths.
    permuted_lengths = permuted_lengths_rows * D
    if sharded_input_embeddings.is_cuda:
        payload = _seq_a2a_rowmap_quantize(
            sharded_input_embeddings, row_idx, D, group_size, rounding
        )
    else:
        # Off GPU, gathering the rows ahead of the codec's own encode is the
        # same identity.
        payload = codec.encode(
            sharded_input_embeddings.view(rows, D).index_select(0, row_idx).view(-1),
            qcomm_ctx,
        )
    if (
        _seq_a2a_rowmap_fwd_checks < _SEQ_A2A_ROWMAP_FWD_CHECK_CALLS
        and os.environ.get("TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD_CHECK", "0") == "1"
        and not (payload.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        _seq_a2a_rowmap_fwd_checks += 1
        ref_lengths, ref_rows, _ = _permute_2D_sparse_data(
            forward_recat_tensor,
            lengths_after_sparse_data_all2all.view(local_T * world_size, -1),
            sharded_input_embeddings.view(-1),
            None,
            sharded_input_embeddings.numel(),
        )
        ref_payload = codec.encode(ref_rows, qcomm_ctx)
        ok = torch.equal(ref_lengths, permuted_lengths) and torch.equal(
            ref_payload, payload
        )
        logger.warning(
            "All2All_Seq_Req row-map MX4 quantize self-check %d/%d: "
            "rows=%d D=%d mx_group_size=%d ok=%s",
            _seq_a2a_rowmap_fwd_checks,
            _SEQ_A2A_ROWMAP_FWD_CHECK_CALLS,
            rows,
            D,
            group_size,
            ok,
        )
        if not ok:
            logger.error(
                "All2All_Seq_Req row-map MX4 quantize self-check FAILED; "
                "disabling TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD for the rest of "
                "the process"
            )
            _seq_a2a_rowmap_fwd_disabled = True
            return ref_lengths, ref_payload, qcomm_ctx
    return permuted_lengths, payload, qcomm_ctx


# Env TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD=1: with MX4 backward qcomms,
# All2All_Seq_Req.backward dequantizes the received payload straight into
# backward-recat order instead of decoding it and then permuting the decoded
# rows. With D % mx_group_size == 0 no MX4 group spans two rows, so output row j
# is the dequantized payload row row_idx[j], where row_idx is the backward recat
# applied to arange(rows), and the result is bitwise identical to the old path.
# TORCHREC_SEQ_A2A_FUSED_ROWMAP_CHECK=1 also runs the old decode and permute on
# the first few calls, compares the raw bits, and falls back to the old path for
# good on a mismatch.
_SEQ_A2A_ROWMAP_CHECK_CALLS = 3
_seq_a2a_rowmap_checks = 0
_seq_a2a_rowmap_disabled = False
_seq_a2a_rowmap_logged: set[Tuple[int, int, int]] = set()


def _seq_a2a_bwd_dequant_rowmap(
    a2ai: All2AllSequenceInfo,
    payload: Tensor,
    qcomm_ctx: Any,
) -> Optional[Tensor]:
    """TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD path of All2All_Seq_Req.backward.

    Returns the flat dequantized gradient in backward-recat order, not yet
    divided by the world size, or None for the old path. The self-check syncs
    (torch.equal) on each of the first _SEQ_A2A_ROWMAP_CHECK_CALLS calls only.
    """
    global _seq_a2a_rowmap_checks, _seq_a2a_rowmap_disabled
    codec = none_throws(a2ai.codecs).backward
    backward_recat_tensor = a2ai.backward_recat_tensor
    permuted_lengths = a2ai.permuted_lengths_after_sparse_data_all2all
    if (
        _seq_a2a_rowmap_disabled
        or backward_recat_tensor is None
        or permuted_lengths is None
        or a2ai.variable_batch_size
        or is_torchdynamo_compiling()
        or payload.numel() == 0
        or payload.dtype != torch.uint8
        or not payload.is_contiguous()
        or not _is_fbgemm_mx4_codec(codec)
        or getattr(codec, "_loss_scale", None) is not None
    ):
        return None
    D = a2ai.embedding_dim
    group_size = getattr(qcomm_ctx, "mx_group_size", 0)
    output_dtype = getattr(codec, "_output_dtype", None)
    out_dtype = torch.float32 if output_dtype is None else output_dtype.as_dtype()
    # The kernel's tl.arange extents need a power-of-two group size.
    exact = (
        group_size >= 2
        and group_size & (group_size - 1) == 0
        and D % group_size == 0
        and out_dtype in (torch.float32, torch.bfloat16)
    )
    row_bytes = D // 2 + D // group_size if exact else 0
    log_key = (D, group_size, row_bytes)
    if log_key not in _seq_a2a_rowmap_logged:
        _seq_a2a_rowmap_logged.add(log_key)
        logger.info(
            "All2All_Seq_Req backward dequantizes the MX4 gradient through a "
            "row map: %s (D=%d mx_group_size=%d bytes/row=%d)",
            exact,
            *log_key,
        )
    if not exact or payload.numel() % row_bytes != 0:
        return None
    rows = payload.numel() // row_bytes
    _, row_idx, _ = _permute_2D_sparse_data(
        backward_recat_tensor,
        permuted_lengths // D,
        torch.arange(rows, dtype=torch.int64, device=payload.device),
        None,
        rows,
    )
    if payload.is_cuda:
        grad = _seq_a2a_rowmap_dequantize(payload, row_idx, D, group_size, out_dtype)
    else:
        # Off GPU, gathering whole packed rows ahead of the codec's own decode
        # is the same identity.
        grad = codec.decode(
            payload.view(rows, row_bytes).index_select(0, row_idx).view(-1),
            qcomm_ctx,
        )
    if (
        _seq_a2a_rowmap_checks < _SEQ_A2A_ROWMAP_CHECK_CALLS
        and os.environ.get("TORCHREC_SEQ_A2A_FUSED_ROWMAP_CHECK", "0") == "1"
        and not (payload.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        _seq_a2a_rowmap_checks += 1
        decoded = codec.decode(payload, qcomm_ctx)
        _, ref, _ = _permute_2D_sparse_data(
            backward_recat_tensor,
            permuted_lengths,
            decoded,
            None,
            decoded.numel(),
        )
        # Raw bits, so that NaNs compare equal to themselves.
        bits = torch.int16 if ref.element_size() == 2 else torch.int32
        ok = ref.dtype == grad.dtype and torch.equal(ref.view(bits), grad.view(bits))
        logger.warning(
            "All2All_Seq_Req row-map MX4 dequant self-check %d/%d: "
            "rows=%d D=%d mx_group_size=%d ok=%s",
            _seq_a2a_rowmap_checks,
            _SEQ_A2A_ROWMAP_CHECK_CALLS,
            rows,
            D,
            group_size,
            ok,
        )
        if not ok:
            logger.error(
                "All2All_Seq_Req row-map MX4 dequant self-check FAILED; "
                "disabling TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD for the rest of "
                "the process"
            )
            _seq_a2a_rowmap_disabled = True
            return ref
    return grad


# Env TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE=1: with MX4 backward qcomms,
# All2All_Seq_Req_Wait.backward encodes its [rows, D] gradient with fbgemm's
# MX4 quantize kernel launched once per row chunk of less than 2 GiB, which
# keeps the AMD buffer-ops specialization of the input, and with the groups per
# program rounded to whole 64-group steps of the kernel
# (mx4_rowmap_quantize.quantize_mx4_rows_chunked). With D % mx_group_size == 0
# every chunk is a whole number of MX4 groups, each group is quantized on its
# own into mx_group_size // 2 + 1 bytes at a fixed offset, and a rounding mode
# other than stochastic draws no random bits, so the payload is bitwise
# identical to the codec's encode. An exception in the chunked encode, such as a
# compile or launch failure, logs and falls back to the old path for good.
# TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE_CHECK=1 also runs the codec's encode on
# the first few calls for each number of chunks, compares the payloads, and
# falls back to the old path for good on a mismatch.
_SEQ_A2A_BWD_CHUNKED_CHECK_CALLS = 3
_seq_a2a_bwd_chunked_checks: dict[int, int] = {}
_seq_a2a_bwd_chunked_disabled = False
_seq_a2a_bwd_chunked_logged: set[Tuple[int, int, int]] = set()


def _seq_a2a_bwd_chunked_encode(
    a2ai: All2AllSequenceInfo,
    grad: Tensor,
    qcomm_ctx: Any,
) -> Optional[Tensor]:
    """TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE path of
    All2All_Seq_Req_Wait.backward.

    Returns the flat uint8 MX4 payload of `grad`, or None for the old path. The
    self-check syncs (torch.equal) on each of the first
    _SEQ_A2A_BWD_CHUNKED_CHECK_CALLS calls for each number of chunks only.
    """
    global _seq_a2a_bwd_chunked_disabled
    codec = none_throws(a2ai.codecs).backward
    if (
        _seq_a2a_bwd_chunked_disabled
        or is_torchdynamo_compiling()
        or grad.numel() == 0
        or not grad.is_contiguous()
        or grad.dtype not in (torch.float32, torch.bfloat16)
        or not _is_fbgemm_mx4_codec(codec)
        or getattr(codec, "_loss_scale", None) is not None
    ):
        return None
    D = a2ai.embedding_dim
    group_size = getattr(qcomm_ctx, "mx_group_size", 0)
    rounding_mode = getattr(qcomm_ctx, "rounding_mode", None)
    # fp32_to_mx4 quantizes with RoundingMode.even when the context has none.
    rounding = 2 if rounding_mode is None else int(rounding_mode)
    # The kernel's tl.arange extents need a power-of-two group size; 3 is
    # RoundingMode.stochastic, whose random bits follow the work split.
    exact = (
        group_size >= 2
        and group_size & (group_size - 1) == 0
        and D > 0
        and D % group_size == 0
        and grad.numel() % D == 0
        and rounding in (0, 1, 2, 4)
    )
    rows = grad.numel() // D if D > 0 else 0
    plan = None
    if exact:
        from torchrec.distributed.mx4_rowmap_quantize import (
            MX4_BUFFER_OPS_MAX_BYTES,
            mx4_rows_chunk_plan,
        )

        plan = mx4_rows_chunk_plan(
            rows, D, grad.element_size(), group_size, MX4_BUFFER_OPS_MAX_BYTES
        )
    num_chunks = plan[1] if plan is not None else 0
    log_key = (D, group_size, num_chunks)
    if log_key not in _seq_a2a_bwd_chunked_logged:
        _seq_a2a_bwd_chunked_logged.add(log_key)
        logger.info(
            "All2All_Seq_Req_Wait backward encodes the MX4 gradient in row "
            "chunks: %s (D=%d mx_group_size=%d chunks=%d groups/program=%d)",
            plan is not None,
            *log_key,
            plan[2] if plan is not None else 0,
        )
    if plan is None:
        return None
    rows_per_chunk, _, groups_per_program = plan
    try:
        if grad.is_cuda:
            payload = _seq_a2a_rows_quantize(
                grad.view(rows, D),
                group_size,
                rounding_mode,
                rows_per_chunk,
                groups_per_program,
            )
        else:
            # Off GPU, the codec's own encode of each row chunk, concatenated,
            # is the same identity.
            payload = torch.cat(
                [
                    codec.encode(chunk, qcomm_ctx).view(-1)
                    for chunk in grad.view(rows, D).split(rows_per_chunk)
                ]
            )
    except Exception:
        logger.exception(
            "All2All_Seq_Req_Wait chunked MX4 encode failed; disabling "
            "TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE for the rest of the process"
        )
        _seq_a2a_bwd_chunked_disabled = True
        return None
    checks = _seq_a2a_bwd_chunked_checks.get(num_chunks, 0)
    if (
        checks < _SEQ_A2A_BWD_CHUNKED_CHECK_CALLS
        and os.environ.get("TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE_CHECK", "0") == "1"
        and not (payload.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        _seq_a2a_bwd_chunked_checks[num_chunks] = checks + 1
        ref = codec.encode(grad, qcomm_ctx).view(-1)
        ok = ref.dtype == payload.dtype and torch.equal(ref, payload)
        logger.warning(
            "All2All_Seq_Req_Wait chunked MX4 encode self-check %d/%d: "
            "rows=%d D=%d mx_group_size=%d chunks=%d groups/program=%d ok=%s",
            checks + 1,
            _SEQ_A2A_BWD_CHUNKED_CHECK_CALLS,
            rows,
            D,
            group_size,
            num_chunks,
            groups_per_program,
            ok,
        )
        if not ok:
            logger.error(
                "All2All_Seq_Req_Wait chunked MX4 encode self-check FAILED; "
                "disabling TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE for the rest of "
                "the process"
            )
            _seq_a2a_bwd_chunked_disabled = True
            return ref
    return payload


# Env TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP=1 (read per call by
# SequenceEmbeddingsAwaitable, which then hands its bijective row-wise
# unbucketize permute `perm` to the Request): with MX4 forward qcomms,
# All2All_Seq_Req_Wait.forward gathers the received packed rows by perm and then
# decodes them, instead of decoding every row and then gathering the decoded
# rows. Its backward then encodes the gradient in the order it arrives and
# gathers the packed rows by the inverse of perm, instead of gathering the
# gradient rows (dist_data._UnbucketizePermute.backward) and then encoding them.
# fbgemm's MX4 codec turns each group of mx_group_size values into
# mx_group_size // 2 + 1 bytes and back on its own, and a rounding mode other
# than stochastic draws no random bits, so with D % mx_group_size == 0 packed
# row r is exactly row r and both directions are bitwise identical to the old
# path, while each gather copies D // 2 + D // mx_group_size bytes per row
# instead of D values. The backward follows its forward, not the flag.
# TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP_CHECK=1 also runs the old path on the
# first few calls in each direction, compares the raw bits, and falls back to
# the old path for good on a mismatch.
_SEQ_A2A_UNBUCKETIZE_CHECK_CALLS = 3
_seq_a2a_unbucketize_fwd_checks = 0
_seq_a2a_unbucketize_bwd_checks = 0
_seq_a2a_unbucketize_fwd_disabled = False
_seq_a2a_unbucketize_bwd_disabled = False
_seq_a2a_unbucketize_logged: set[Tuple[str, int, int, int]] = set()


def _mx4_packed_row_bytes(codec: Any, qcomm_ctx: Any, D: int) -> int:
    """Bytes per row of D values in the MX4 payload of `codec`, or 0 when a
    packed row does not hold exactly one row."""
    if (
        not _is_fbgemm_mx4_codec(codec)
        or getattr(codec, "_loss_scale", None) is not None
    ):
        return 0
    group_size = getattr(qcomm_ctx, "mx_group_size", 0)
    if group_size < 2 or group_size % 2 != 0 or D <= 0 or D % group_size != 0:
        return 0
    return D // 2 + D // group_size


def _gather_packed_rows(payload: Tensor, row_bytes: int, index: Tensor) -> Tensor:
    """Packed rows `index` of the flat uint8 `payload` of row_bytes-byte rows,
    as a flat uint8 tensor; the rows are copied as the widest integers that
    tile them."""
    rows = payload.view(-1, row_bytes)
    for dtype, size in ((torch.int64, 8), (torch.int32, 4), (torch.int16, 2)):
        if row_bytes % size == 0 and payload.storage_offset() % size == 0:
            rows = rows.view(dtype)
            break
    return rows.index_select(0, index).view(torch.uint8).view(-1)


def _seq_a2a_unbucketize_log(direction: str, D: int, group_size: int, ok: bool) -> None:
    log_key = (direction, D, group_size, int(ok))
    if log_key not in _seq_a2a_unbucketize_logged:
        _seq_a2a_unbucketize_logged.add(log_key)
        logger.info(
            "All2All_Seq_Req_Wait %s applies the unbucketize permute to the "
            "packed MX4 rows: %s (D=%d mx_group_size=%d)",
            direction,
            ok,
            D,
            group_size,
        )


def _seq_a2a_fwd_decode_unbucketize(
    a2ai: All2AllSequenceInfo,
    payload: Tensor,
    qcomm_ctx: Any,
    row_map: Tensor,
) -> Optional[Tensor]:
    """TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP path of All2All_Seq_Req_Wait.forward.

    Returns decode(payload).view(-1, D).index_select(0, row_map) as a [rows, D]
    tensor that is not a view, or None for the old path. The self-check syncs
    (torch.equal) on each of the first _SEQ_A2A_UNBUCKETIZE_CHECK_CALLS calls
    only.
    """
    global _seq_a2a_unbucketize_fwd_checks, _seq_a2a_unbucketize_fwd_disabled
    codecs = a2ai.codecs
    if (
        _seq_a2a_unbucketize_fwd_disabled
        or codecs is None
        or is_torchdynamo_compiling()
        or payload.dtype != torch.uint8
        or payload.dim() != 1
        or not payload.is_contiguous()
        or row_map.dim() != 1
        or row_map.numel() == 0
        or row_map.dtype not in (torch.int32, torch.int64)
        or row_map.device != payload.device
    ):
        return None
    codec = codecs.forward
    D = a2ai.embedding_dim
    row_bytes = _mx4_packed_row_bytes(codec, qcomm_ctx, D)
    # 16-byte rows keep the packed-row gather on aten's vectorized gather.
    exact = row_bytes > 0 and row_bytes % 16 == 0
    _seq_a2a_unbucketize_log(
        "forward", D, getattr(qcomm_ctx, "mx_group_size", 0), exact
    )
    rows = row_map.numel()
    if not exact or payload.numel() != rows * row_bytes:
        return None
    out = _seq_a2a_fwd_decode(
        codec, D, _gather_packed_rows(payload, row_bytes, row_map), qcomm_ctx
    )
    # detach() drops the view relation to the decode output, so in-place ops on
    # the result stay allowed, as on the old path's index_select output.
    out = out.view(rows, D).detach()
    if (
        _seq_a2a_unbucketize_fwd_checks < _SEQ_A2A_UNBUCKETIZE_CHECK_CALLS
        and os.environ.get("TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP_CHECK", "0") == "1"
        and not (payload.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        _seq_a2a_unbucketize_fwd_checks += 1
        ref = codec.decode(payload, qcomm_ctx).view(-1, D).index_select(0, row_map)
        # Raw bits, so that NaNs compare equal to themselves.
        bits = torch.int16 if ref.element_size() == 2 else torch.int32
        ok = (
            ref.dtype == out.dtype
            and ref.shape == out.shape
            and torch.equal(ref.view(bits), out.view(bits))
        )
        logger.warning(
            "All2All_Seq_Req_Wait unbucketize-order MX4 decode self-check %d/%d: "
            "rows=%d D=%d ok=%s",
            _seq_a2a_unbucketize_fwd_checks,
            _SEQ_A2A_UNBUCKETIZE_CHECK_CALLS,
            rows,
            D,
            ok,
        )
        if not ok:
            logger.error(
                "All2All_Seq_Req_Wait unbucketize-order MX4 decode self-check "
                "FAILED; disabling TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP for the "
                "rest of the process"
            )
            _seq_a2a_unbucketize_fwd_disabled = True
            return ref
    return out


def _seq_a2a_unbucketize_grad(grad: Tensor, row_map: Tensor) -> Tensor:
    """The gradient of the rows before the unbucketize permute, as
    dist_data._UnbucketizePermute.backward computes it: out[i] = src[row_map[i]],
    so grad_src[r] = grad[inv[r]] with inv[row_map[i]] = i."""
    inv = torch.ops.fbgemm.invert_permute(row_map).clamp_(0, row_map.numel() - 1)
    return grad.index_select(0, inv)


def _seq_a2a_bwd_unbucketize_packed(
    a2ai: All2AllSequenceInfo,
    grad: Tensor,
    row_map: Tensor,
    qcomm_ctx: Any,
) -> bool:
    """Whether All2All_Seq_Req_Wait.backward may encode `grad`, the gradient of
    the unbucketized rows, as it is and then gather the packed rows."""
    codecs = a2ai.codecs
    if (
        _seq_a2a_unbucketize_bwd_disabled
        or codecs is None
        or is_torchdynamo_compiling()
        or not grad.is_contiguous()
        or grad.dtype not in (torch.float32, torch.bfloat16)
    ):
        return False
    D = a2ai.embedding_dim
    rounding_mode = getattr(qcomm_ctx, "rounding_mode", None)
    # 3 is RoundingMode.stochastic, whose random bits follow the position.
    exact = _mx4_packed_row_bytes(codecs.backward, qcomm_ctx, D) > 0 and (
        rounding_mode is None or int(rounding_mode) != 3
    )
    _seq_a2a_unbucketize_log(
        "backward", D, getattr(qcomm_ctx, "mx_group_size", 0), exact
    )
    return exact and grad.numel() == row_map.numel() * D


def _seq_a2a_bwd_unbucketize_payload(
    a2ai: All2AllSequenceInfo,
    grad: Tensor,
    payload: Tensor,
    row_map: Tensor,
    qcomm_ctx: Any,
) -> Tensor:
    """TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP path of
    All2All_Seq_Req_Wait.backward, after _seq_a2a_bwd_unbucketize_packed.

    `payload` is the MX4 payload of `grad`, the gradient of the unbucketized
    rows. Returns the flat payload of the gradient of the rows before the
    unbucketize permute: packed row r is packed row inv[r] of `payload`. The
    self-check syncs (torch.equal) on each of the first
    _SEQ_A2A_UNBUCKETIZE_CHECK_CALLS calls only.
    """
    global _seq_a2a_unbucketize_bwd_checks, _seq_a2a_unbucketize_bwd_disabled
    codec = none_throws(a2ai.codecs).backward
    rows = row_map.numel()
    row_bytes = _mx4_packed_row_bytes(codec, qcomm_ctx, a2ai.embedding_dim)
    if (
        row_bytes == 0
        or payload.dtype != torch.uint8
        or payload.numel() != rows * row_bytes
        or not payload.is_contiguous()
    ):
        # Not an MX4 payload of whole rows: encode in the old row order.
        grad = _seq_a2a_unbucketize_grad(grad, row_map)
        return codec.encode(grad, qcomm_ctx).view(-1)
    inv = torch.ops.fbgemm.invert_permute(row_map).clamp_(0, rows - 1)
    out = _gather_packed_rows(payload, row_bytes, inv)
    if (
        _seq_a2a_unbucketize_bwd_checks < _SEQ_A2A_UNBUCKETIZE_CHECK_CALLS
        and os.environ.get("TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP_CHECK", "0") == "1"
        and not (out.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        _seq_a2a_unbucketize_bwd_checks += 1
        ref = codec.encode(grad.index_select(0, inv), qcomm_ctx).view(-1)
        ok = ref.dtype == out.dtype and torch.equal(ref, out)
        logger.warning(
            "All2All_Seq_Req_Wait unbucketize-order MX4 encode self-check %d/%d: "
            "rows=%d D=%d ok=%s",
            _seq_a2a_unbucketize_bwd_checks,
            _SEQ_A2A_UNBUCKETIZE_CHECK_CALLS,
            rows,
            a2ai.embedding_dim,
            ok,
        )
        if not ok:
            logger.error(
                "All2All_Seq_Req_Wait unbucketize-order MX4 encode self-check "
                "FAILED; encoding in the old row order for the rest of the "
                "process"
            )
            _seq_a2a_unbucketize_bwd_disabled = True
            return ref
    return out


# Env TORCHREC_SEQ_A2A_MX4_V2=1, or a comma list of the scopes below (read
# per call): with fbgemm MX4 qcomms on ROCm GPU tensors, the sequence
# all-to-all runs its MX4 encodes and decodes on the row-tiled kernels of
# mx4_rows_v2 instead of fbgemm's and the row-map ones:
# - fwd_enc: All2All_Seq_Req.forward's row-map encode
#   (TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD);
# - bwd_enc: All2All_Seq_Req_Wait.backward's row-chunked encode
#   (TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE), as one launch;
# - fwd_dec: All2All_Seq_Req_Wait.forward's decode;
# - bwd_dec: All2All_Seq_Req.backward's row-map decode
#   (TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD).
# The kernels run fbgemm's per-group and per-value arithmetic, so the payload
# bytes and decoded bits are the old kernels' (see mx4_rows_v2). A scope that
# is asked for but not taken logs a warning once per reason.
# TORCHREC_SEQ_A2A_MX4_V2_CFG=block_r:num_warps, or a comma list of
# scope=block_r:num_warps, sets the tiles (default 4:4);
# TORCHREC_SEQ_A2A_MX4_V2_DIV=1 makes the encode divide by the group scale.
# TORCHREC_SEQ_A2A_MX4_V2_CHECK=1 also runs the old kernel on the first few
# calls of each scope, and once per scope and config an edge input through both
# kernels, and compares the bits on the rank alone. On a mismatch the call
# returns the old kernel's result and logs an error; an encode scope then
# divides by the scale, and a second mismatch (the first for a decode scope)
# turns the scope off for the rest of the process. A kernel that fails to
# launch turns its scope off too. Every path writes the old kernel's bytes, so
# none of these rank-local choices changes what a rank sends.
_MX4V2_SCOPES: Tuple[str, ...] = ("fwd_enc", "bwd_enc", "fwd_dec", "bwd_dec")
_MX4V2_CHECK_CALLS = 3
_MX4V2_DEFAULT_CFG: Tuple[int, int] = (4, 4)
_MX4V2_EDGE_ROWS = 64
_MX4V2_EDGE_SPECIALS: List[float] = [
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    -float("nan"),
    6.0e37,
    -3.0e38,
    3.3895e38,
    2.0**-126,
    2.0**-127,
    -(2.0**-133),
    1.0e-40,
    1.75,
    7.5,
    -6.0,
]


@dataclass
class _Mx4V2State:
    checks: int = 0
    mismatches: int = 0
    off: bool = False
    edge_checked: set[Tuple[str, bool]] = field(default_factory=set)
    logged: set[Tuple[str, int, int, bool]] = field(default_factory=set)


_mx4v2_state: dict[str, _Mx4V2State] = {s: _Mx4V2State() for s in _MX4V2_SCOPES}
_mx4v2_warned: set[Tuple[Any, ...]] = set()
_mx4v2_edge_cache: dict[Tuple[Any, ...], Tensor] = {}


def _mx4v2_warn_once(key: Tuple[Any, ...], msg: str, *args: object) -> None:
    if key not in _mx4v2_warned:
        _mx4v2_warned.add(key)
        logger.warning(msg, *args)


def _mx4v2_requested(scope: str) -> bool:
    """Whether TORCHREC_SEQ_A2A_MX4_V2 asks for `scope`."""
    flag = os.environ.get("TORCHREC_SEQ_A2A_MX4_V2", "1").strip()
    if flag in ("", "0"):
        return False
    if flag == "1":
        return True
    scopes = {s.strip() for s in flag.split(",")}
    unknown = sorted(scopes - set(_MX4V2_SCOPES))
    if unknown:
        _mx4v2_warn_once(
            ("flag", flag),
            "TORCHREC_SEQ_A2A_MX4_V2=%s names unknown scopes %s (known: %s); "
            "they are ignored",
            flag,
            unknown,
            ",".join(_MX4V2_SCOPES),
        )
    return scope in scopes


def _mx4v2_cfg(scope: str) -> Tuple[int, int]:
    """(block_r, num_warps) of `scope` from TORCHREC_SEQ_A2A_MX4_V2_CFG."""
    spec = os.environ.get("TORCHREC_SEQ_A2A_MX4_V2_CFG", "").strip()
    if not spec:
        return _MX4V2_DEFAULT_CFG
    value = spec
    if "=" in spec:
        entries = dict(e.split("=", 1) for e in spec.split(",") if "=" in e)
        value = entries.get(scope, "")
        if not value:
            return _MX4V2_DEFAULT_CFG
    parts = value.split(":")
    cfg = (
        (int(parts[0]), int(parts[1]))
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
        else (0, 0)
    )
    if cfg[0] not in (1, 2, 4, 8, 16) or cfg[1] not in (1, 2, 4, 8):
        _mx4v2_warn_once(
            ("cfg", spec),
            "TORCHREC_SEQ_A2A_MX4_V2_CFG=%s is not block_r:num_warps with "
            "block_r in 1,2,4,8,16 and num_warps in 1,2,4,8; using %d:%d",
            spec,
            *_MX4V2_DEFAULT_CFG,
        )
        return _MX4V2_DEFAULT_CFG
    return cfg


def _mx4v2_decline(scope: str, reason: str) -> None:
    _mx4v2_warn_once(
        ("decline", scope, reason),
        "TORCHREC_SEQ_A2A_MX4_V2 %s declined (once per reason): %s; using the "
        "old kernel",
        scope,
        reason,
    )


def _mx4v2_active(scope: str, D: int, group_size: int) -> bool:
    """Whether `scope` is asked for and may run the row-tiled kernels for
    rows of D values in groups of group_size; warns once per decline."""
    if not _mx4v2_requested(scope):
        return False
    from torchrec.distributed.mx4_rows_v2 import mx4v2_supported

    state = _mx4v2_state[scope]
    if state.off:
        _mx4v2_decline(scope, "turned off after a failed self-check or launch")
    elif is_torchdynamo_compiling() or torch.jit.is_scripting():
        _mx4v2_decline(scope, "compiling or scripting")
    elif torch.is_inference_mode_enabled():
        _mx4v2_decline(scope, "inference mode")
    elif torch.version.hip is None:
        # The kernels state the gfx9 shift semantics of fbgemm's kernels.
        _mx4v2_decline(scope, "not a ROCm build")
    elif not mx4v2_supported(D, group_size):
        _mx4v2_decline(
            scope, f"D={D} mx_group_size={group_size} is not a supported shape"
        )
    else:
        return True
    return False


def _mx4v2_codec_path(scope: str, what: str) -> None:
    """Warns once when `scope` is asked for but `what` runs the codec."""
    if _mx4v2_requested(scope):
        _mx4v2_decline(scope, f"{what} runs the codec")


def _mx4v2_same_bits(out: Tensor, ref: Tensor) -> bool:
    # Raw bits, so that NaNs compare equal to themselves.
    if out.dtype != ref.dtype or out.shape != ref.shape:
        return False
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[out.element_size()]
    return torch.equal(out.view(bits), ref.view(bits))


def _mx4v2_run(
    scope: str,
    key: str,
    rows: int,
    run: Callable[[bool, int, int], Tensor],
    ref: Callable[[], Tensor],
    edge: Callable[[bool, int, int], Tuple[Tensor, Tensor]],
) -> Tensor:
    """Runs scope `scope` for a call of `rows` rows with config `key`:
    run(mul_rcp, block_r, num_warps) is the row-tiled kernel, ref() the old
    kernel and edge(...) the pair of both kernels' outputs for an edge input.
    Returns the old kernel's result on a failed launch or self-check."""
    state = _mx4v2_state[scope]
    mul_rcp = (
        scope.endswith("_enc")
        and state.mismatches == 0
        and os.environ.get("TORCHREC_SEQ_A2A_MX4_V2_DIV", "0") != "1"
    )
    block_r, num_warps = _mx4v2_cfg(scope)
    log_key = (key, block_r, num_warps, mul_rcp)
    if log_key not in state.logged:
        state.logged.add(log_key)
        logger.info(
            "TORCHREC_SEQ_A2A_MX4_V2 %s on: row-tiled MX4 kernel for %s "
            "(block_r=%d num_warps=%d mul_rcp=%s)",
            scope,
            key,
            block_r,
            num_warps,
            mul_rcp,
        )
    try:
        with record_function(f"## All2All_Seq_mx4v2_{scope} ##"):
            out = run(mul_rcp, block_r, num_warps)
    except Exception:
        logger.exception(
            "TORCHREC_SEQ_A2A_MX4_V2 %s kernel failed for %s; turning the "
            "scope off for the rest of the process",
            scope,
            key,
        )
        state.off = True
        return ref()
    if (
        state.checks >= _MX4V2_CHECK_CALLS
        or os.environ.get("TORCHREC_SEQ_A2A_MX4_V2_CHECK", "0") != "1"
        or (out.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        return out
    return _mx4v2_check(
        scope, key, rows, out, ref(), lambda: edge(mul_rcp, block_r, num_warps), mul_rcp
    )


def _mx4v2_check(
    scope: str,
    key: str,
    rows: int,
    out: Tensor,
    ref: Tensor,
    edge: Callable[[], Tuple[Tensor, Tensor]],
    mul_rcp: bool,
) -> Tensor:
    """TORCHREC_SEQ_A2A_MX4_V2_CHECK of a call of `scope`: compares `out`
    with the old kernel's `ref`, and once per config the edge outputs. Syncs
    (torch.equal) on the rank alone."""
    state = _mx4v2_state[scope]
    state.checks += 1
    ok = _mx4v2_same_bits(out, ref)
    edge_ok = True
    if (key, mul_rcp) not in state.edge_checked:
        state.edge_checked.add((key, mul_rcp))
        try:
            edge_ok = _mx4v2_same_bits(*edge())
        except Exception:
            logger.exception("TORCHREC_SEQ_A2A_MX4_V2 %s edge self-check failed", scope)
            edge_ok = False
    logger.warning(
        "TORCHREC_SEQ_A2A_MX4_V2 %s self-check %d/%d: %s rows=%d mul_rcp=%s "
        "ok=%s edge_ok=%s",
        scope,
        state.checks,
        _MX4V2_CHECK_CALLS,
        key,
        rows,
        mul_rcp,
        ok,
        edge_ok,
    )
    if ok and edge_ok:
        return out
    state.mismatches += 1
    if mul_rcp:
        # Check the division form from scratch.
        state.checks = 0
        action = "dividing by the group scale from now on"
    else:
        state.off = True
        action = "turning the scope off for the rest of the process"
    logger.error(
        "TORCHREC_SEQ_A2A_MX4_V2 %s self-check FAILED for %s (ok=%s "
        "edge_ok=%s); this call uses the old kernel's result, %s",
        scope,
        key,
        ok,
        edge_ok,
        action,
    )
    return ref


def _mx4v2_edge_input(D: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """A fixed [_MX4V2_EDGE_ROWS, D] encode input of edge groups: zeros,
    signed zeros, infinities, NaNs of both signs, all-NaN and all-inf groups,
    values near the format's max, values so far below their group max that
    the mantissa shift amount passes 31, subnormal groups and ratios, values
    on the E2M1 rounding boundaries, and random bit patterns. NaNs are quiet:
    arithmetic never makes signaling ones, and the group max of a bf16
    signaling NaN depends on how the compiler orders the max."""
    cache_key = ("enc", D, dtype, device)
    if cache_key in _mx4v2_edge_cache:
        return _mx4v2_edge_cache[cache_key]
    gen = torch.Generator().manual_seed(0x4D58)
    x = torch.randn(_MX4V2_EDGE_ROWS, D, generator=gen) * torch.exp2(
        torch.randint(-40, 40, (_MX4V2_EDGE_ROWS, 1), generator=gen).float()
    )
    # Every 8th value of rows 0-15 cycles through the specials, so that each
    # one meets ordinary values in groups of any size.
    specials = torch.tensor(_MX4V2_EDGE_SPECIALS)
    head = x[:16].view(-1)[::8]
    head.copy_(specials.repeat(head.numel() // specials.numel() + 1)[: head.numel()])
    # Rows 16-23: groups of 8 of 4.0 and +-2^-8 .. 2^-31.
    tiny = x[16:24].view(-1, 8)
    k = torch.arange(tiny.shape[0] * 7)
    tiny[:, 0] = 4.0
    tiny[:, 1:] = (torch.exp2(-(8 + k % 24).float()) * (1 - 2 * (k % 2))).view(-1, 7)
    x[24] = 0.0
    x[25] = float("nan")
    x[26] = float("inf")
    x[27] = float("-inf")
    x[28] = torch.exp2(torch.randint(-149, -120, (D,), generator=gen).float())
    x[29] = torch.where(torch.arange(D) % 3 == 0, -1.0, 1.0) * 2.0**-127
    x[30] = torch.exp2(torch.randint(100, 128, (D,), generator=gen).float())
    x[31] = torch.tensor([7.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]).repeat(
        D // 8
    ) * torch.exp2((torch.arange(D) // 8 % 16 - 8).float())
    x = x.to(dtype)
    # Rows 56-63: random bit patterns, NaNs quieted.
    itype, quiet = (
        (torch.int16, 0x40) if dtype == torch.bfloat16 else (torch.int32, 0x400000)
    )
    width = 8 * x.element_size()
    bits = torch.randint(
        -(2 ** (width - 1)), 2 ** (width - 1), (8, D), generator=gen, dtype=torch.int64
    ).to(itype)
    nan = bits.view(dtype).isnan()
    x[-8:] = torch.where(nan, bits | quiet, bits).view(dtype)
    out = x.to(device)
    _mx4v2_edge_cache[cache_key] = out
    return out


def _mx4v2_edge_row_idx(device: torch.device) -> Tensor:
    # A fixed permutation of the edge rows (37 is coprime with 64).
    return (torch.arange(_MX4V2_EDGE_ROWS) * 37 % _MX4V2_EDGE_ROWS).to(device)


def _seq_a2a_rowmap_quantize(
    a: Tensor, row_idx: Tensor, D: int, group_size: int, rounding: int
) -> Tensor:
    """The GPU encode of _seq_a2a_fwd_quantize_rowmap: quantize_mx4_rowmap,
    or the row-tiled encode for TORCHREC_SEQ_A2A_MX4_V2 fwd_enc."""
    from torchrec.distributed.mx4_rowmap_quantize import quantize_mx4_rowmap

    if not _mx4v2_active("fwd_enc", D, group_size):
        return quantize_mx4_rowmap(a, row_idx, D, group_size, rounding)
    from torchrec.distributed.mx4_rows_v2 import mx4_encode_rows

    def edge(mul_rcp: bool, block_r: int, num_warps: int) -> Tuple[Tensor, Tensor]:
        x = _mx4v2_edge_input(D, a.dtype, a.device)
        idx = _mx4v2_edge_row_idx(a.device)
        return (
            mx4_encode_rows(
                x, idx, D, group_size, rounding, block_r, num_warps, mul_rcp
            ),
            quantize_mx4_rowmap(x, idx, D, group_size, rounding),
        )

    return _mx4v2_run(
        "fwd_enc",
        f"D={D} mx_group_size={group_size} rounding={rounding} dtype={a.dtype} "
        "row_map=True",
        row_idx.numel(),
        lambda mul_rcp, block_r, num_warps: mx4_encode_rows(
            a, row_idx, D, group_size, rounding, block_r, num_warps, mul_rcp
        ),
        lambda: quantize_mx4_rowmap(a, row_idx, D, group_size, rounding),
        edge,
    )


def _seq_a2a_rows_quantize(
    grad: Tensor,
    group_size: int,
    rounding_mode: Optional[int],
    rows_per_chunk: int,
    groups_per_program: int,
) -> Tensor:
    """The GPU encode of _seq_a2a_bwd_chunked_encode for a [rows, D] `grad`:
    quantize_mx4_rows_chunked, or one row-tiled launch for
    TORCHREC_SEQ_A2A_MX4_V2 bwd_enc."""
    from torchrec.distributed.mx4_rowmap_quantize import (
        mx4_rows_chunk_plan,
        quantize_mx4_rows_chunked,
    )

    D = grad.shape[1]
    if not _mx4v2_active("bwd_enc", D, group_size):
        return quantize_mx4_rows_chunked(
            grad, group_size, rounding_mode, rows_per_chunk, groups_per_program
        )
    from torchrec.distributed.mx4_rows_v2 import mx4_encode_rows

    # fp32_to_mx4 quantizes with RoundingMode.even when the context has none.
    rounding = 2 if rounding_mode is None else int(rounding_mode)

    def edge(mul_rcp: bool, block_r: int, num_warps: int) -> Tuple[Tensor, Tensor]:
        x = _mx4v2_edge_input(D, grad.dtype, grad.device)
        plan = none_throws(
            mx4_rows_chunk_plan(x.shape[0], D, x.element_size(), group_size)
        )
        return (
            mx4_encode_rows(
                x, None, D, group_size, rounding, block_r, num_warps, mul_rcp
            ),
            quantize_mx4_rows_chunked(x, group_size, rounding, plan[0], plan[2]),
        )

    return _mx4v2_run(
        "bwd_enc",
        f"D={D} mx_group_size={group_size} rounding={rounding} dtype={grad.dtype} "
        "row_map=False",
        grad.shape[0],
        lambda mul_rcp, block_r, num_warps: mx4_encode_rows(
            grad, None, D, group_size, rounding, block_r, num_warps, mul_rcp
        ),
        lambda: quantize_mx4_rows_chunked(
            grad, group_size, rounding_mode, rows_per_chunk, groups_per_program
        ),
        edge,
    )


def _seq_a2a_codec_encode(
    codec: Any, x: Tensor, qcomm_ctx: Any, scope: str, what: str
) -> Tensor:
    """codec.encode(x, qcomm_ctx) on a path that TORCHREC_SEQ_A2A_MX4_V2
    `scope` does not take."""
    _mx4v2_codec_path(scope, what)
    return codec.encode(x, qcomm_ctx)


def _mx4v2_edge_payload(
    D: int, group_size: int, device: torch.device
) -> Tuple[Tensor, int]:
    """A fixed MX4 payload of D-value rows, and its number of rows, in which
    every byte value occurs as a nibble byte and every exponent byte occurs
    at least twice."""
    G = D // group_size
    rows = max(_MX4V2_EDGE_ROWS, 512 // G)
    cache_key = ("dec", D, group_size, device)
    if cache_key not in _mx4v2_edge_cache:
        gen = torch.Generator().manual_seed(0x4D59)
        nibbles = torch.randint(0, 256, (rows * G, group_size // 2), generator=gen)
        nibbles.view(-1)[:1024] = torch.arange(256).repeat(4)
        exps = (torch.arange(rows * G) % 256).view(-1, 1)
        packed = torch.cat([nibbles, exps], dim=1).to(torch.uint8).view(-1)
        _mx4v2_edge_cache[cache_key] = packed.to(device)
    return _mx4v2_edge_cache[cache_key], rows


def _mx4v2_fwd_decode_reason(
    codec: Any, D: int, payload: Tensor, qcomm_ctx: Any
) -> str:
    """Why TORCHREC_SEQ_A2A_MX4_V2 fwd_dec declines `payload`, or ""."""
    row_bytes = _mx4_packed_row_bytes(codec, qcomm_ctx, D)
    if row_bytes == 0:
        return "not an fbgemm MX4 codec without a loss scale"
    output_dtype = getattr(codec, "_output_dtype", None)
    if output_dtype not in (None, SparseType.FP32, SparseType.BF16):
        return f"output dtype {output_dtype} is not FP32 or BF16"
    if not payload.is_cuda:
        return "the payload is not on GPU"
    if payload.dtype != torch.uint8:
        return "the payload is not uint8"
    if payload.dim() != 1 or not payload.is_contiguous():
        return "the payload is not contiguous and 1-D"
    if payload.numel() % row_bytes != 0:
        return "the payload is not whole packed rows"
    return ""


def _seq_a2a_fwd_decode(codec: Any, D: int, payload: Tensor, qcomm_ctx: Any) -> Tensor:
    """The forward decode of All2All_Seq_Req_Wait: codec.decode, or the
    row-tiled decode for TORCHREC_SEQ_A2A_MX4_V2 fwd_dec."""
    group_size = getattr(qcomm_ctx, "mx_group_size", 0)
    if not _mx4v2_active("fwd_dec", D, group_size):
        return codec.decode(payload, qcomm_ctx)
    reason = _mx4v2_fwd_decode_reason(codec, D, payload, qcomm_ctx)
    if reason:
        _mx4v2_decline("fwd_dec", reason)
        return codec.decode(payload, qcomm_ctx)
    from torchrec.distributed.mx4_rows_v2 import mx4_decode_rows

    rows = payload.numel() // (D // 2 + D // group_size)
    output_dtype = getattr(codec, "_output_dtype", None)
    out_dtype = torch.float32 if output_dtype is None else output_dtype.as_dtype()

    def edge(mul_rcp: bool, block_r: int, num_warps: int) -> Tuple[Tensor, Tensor]:
        x, n = _mx4v2_edge_payload(D, group_size, payload.device)
        return (
            mx4_decode_rows(x, None, n, D, group_size, out_dtype, block_r, num_warps),
            codec.decode(x, qcomm_ctx),
        )

    return _mx4v2_run(
        "fwd_dec",
        f"D={D} mx_group_size={group_size} out={out_dtype} row_map=False",
        rows,
        lambda mul_rcp, block_r, num_warps: mx4_decode_rows(
            payload, None, rows, D, group_size, out_dtype, block_r, num_warps
        ),
        lambda: codec.decode(payload, qcomm_ctx),
        edge,
    )


def _seq_a2a_rowmap_dequantize(
    payload: Tensor,
    row_idx: Tensor,
    D: int,
    group_size: int,
    out_dtype: torch.dtype,
) -> Tensor:
    """The GPU decode of _seq_a2a_bwd_dequant_rowmap: dequantize_mx4_rowmap,
    or the row-tiled decode for TORCHREC_SEQ_A2A_MX4_V2 bwd_dec."""
    from torchrec.distributed.mx4_rowmap_dequant import dequantize_mx4_rowmap

    if not _mx4v2_active("bwd_dec", D, group_size):
        return dequantize_mx4_rowmap(payload, row_idx, D, group_size, out_dtype)
    from torchrec.distributed.mx4_rows_v2 import mx4_decode_rows

    rows = row_idx.numel()

    def edge(mul_rcp: bool, block_r: int, num_warps: int) -> Tuple[Tensor, Tensor]:
        x, n = _mx4v2_edge_payload(D, group_size, payload.device)
        idx = (torch.arange(n) * 37 % n).to(payload.device)
        return (
            mx4_decode_rows(x, idx, n, D, group_size, out_dtype, block_r, num_warps),
            dequantize_mx4_rowmap(x, idx, D, group_size, out_dtype),
        )

    return _mx4v2_run(
        "bwd_dec",
        f"D={D} mx_group_size={group_size} out={out_dtype} row_map=True",
        rows,
        lambda mul_rcp, block_r, num_warps: mx4_decode_rows(
            payload, row_idx, rows, D, group_size, out_dtype, block_r, num_warps
        ),
        lambda: dequantize_mx4_rowmap(payload, row_idx, D, group_size, out_dtype),
        edge,
    )


def _seq_a2a_codec_decode(
    codec: Any, payload: Tensor, qcomm_ctx: Any, scope: str, what: str
) -> Tensor:
    """codec.decode(payload, qcomm_ctx) on a path that
    TORCHREC_SEQ_A2A_MX4_V2 `scope` does not take."""
    _mx4v2_codec_path(scope, what)
    return codec.decode(payload, qcomm_ctx)


class All2All_Seq_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: All2AllSequenceInfo,
        sharded_input_embeddings: Tensor,
    ) -> Tensor:
        world_size = dist.get_world_size(pg)
        my_rank = dist.get_rank(pg)
        D = a2ai.embedding_dim
        forward_recat_tensor = a2ai.forward_recat_tensor
        variable_batch_size = a2ai.variable_batch_size
        lengths_after_sparse_data_all2all = a2ai.lengths_after_sparse_data_all2all * D
        input_splits = [i * D for i in a2ai.output_splits]
        output_splits = [i * D for i in a2ai.input_splits]

        a2ai.input_splits = input_splits
        a2ai.output_splits = output_splits

        send = (
            a2ai.precomputed_send
            if os.environ.get(SIDE_STREAM_OUTPUT_ENCODE_ENV, "0") == "1"
            else None
        )
        a2ai.precomputed_send = None
        if send is not None:
            # Computed on another stream by `precompute_seq_a2a_send`.
            current_stream = torch.get_device_module(
                send.send_tensor.device
            ).current_stream()
            current_stream.wait_event(send.ready)
            send.send_tensor.record_stream(current_stream)
            if send.permuted_lengths is not None:
                send.permuted_lengths.record_stream(current_stream)
            permuted_lengths_after_sparse_data_all2all = send.permuted_lengths
            sharded_input_embeddings = send.send_tensor
            qcomm_ctx = send.qcomm_ctx
        else:
            local_T = lengths_after_sparse_data_all2all.shape[0]
            permuted_quantized = None
            if (
                local_T > 0
                and os.environ.get("TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD", "1") == "1"
            ):
                with record_function("## All2All_Seq_fwd_quantize_rowmap ##"):
                    permuted_quantized = _seq_a2a_fwd_quantize_rowmap(
                        a2ai,
                        sharded_input_embeddings,
                        lengths_after_sparse_data_all2all,
                        world_size,
                    )
            if (
                permuted_quantized is None
                and local_T > 0
                and os.environ.get("TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD", "1") == "1"
            ):
                with record_function("## All2All_Seq_fwd_permute_quantized ##"):
                    permuted_quantized = _seq_a2a_fwd_permute_quantized(
                        a2ai,
                        sharded_input_embeddings,
                        lengths_after_sparse_data_all2all,
                        world_size,
                    )
            if permuted_quantized is not None:
                (
                    permuted_lengths_after_sparse_data_all2all,
                    sharded_input_embeddings,
                    _,
                ) = permuted_quantized
            elif local_T > 0:
                with record_function("## All2All_Seq_fwd_permute ##"):
                    if not variable_batch_size:
                        (
                            permuted_lengths_after_sparse_data_all2all,
                            sharded_input_embeddings,
                            _,
                        ) = _permute_2D_sparse_data(
                            forward_recat_tensor,
                            lengths_after_sparse_data_all2all.view(
                                local_T * world_size, -1
                            ),
                            sharded_input_embeddings.view(-1),
                            None,
                            sharded_input_embeddings.numel(),
                        )
                    else:
                        (
                            permuted_lengths_after_sparse_data_all2all,
                            sharded_input_embeddings,
                            _,
                        ) = torch.ops.fbgemm.permute_1D_sparse_data(
                            forward_recat_tensor,
                            lengths_after_sparse_data_all2all.view(-1),
                            sharded_input_embeddings.view(-1),
                            None,
                            sharded_input_embeddings.numel(),
                        )
            else:
                permuted_lengths_after_sparse_data_all2all = None

            qcomm_ctx = None
            if a2ai.codecs is not None:
                codecs = none_throws(a2ai.codecs)
                if permuted_quantized is not None:
                    qcomm_ctx = permuted_quantized[2]
                else:
                    qcomm_ctx = codecs.forward.create_context()
                    sharded_input_embeddings = _seq_a2a_codec_encode(
                        codecs.forward,
                        sharded_input_embeddings,
                        qcomm_ctx,
                        "fwd_enc",
                        "All2All_Seq_Req.forward without the row-map encode "
                        "(TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD)",
                    )

        if a2ai.codecs is not None:
            output_splits = [
                a2ai.codecs.forward.calc_quantized_size(x, qcomm_ctx)
                for x in output_splits
            ]
            input_splits = [
                a2ai.codecs.forward.calc_quantized_size(x, qcomm_ctx)
                for x in input_splits
            ]

        sharded_output_embeddings = torch.empty(
            sum(output_splits),
            dtype=sharded_input_embeddings.dtype,
            device=sharded_input_embeddings.device,
        )

        with record_function("## All2All_Seq_fwd ##"):
            req = dist.all_to_all_single(
                output=sharded_output_embeddings,
                input=sharded_input_embeddings,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=pg,
                async_op=True,
            )
        a2ai.permuted_lengths_after_sparse_data_all2all = (
            permuted_lengths_after_sparse_data_all2all
        )
        myreq.req = req
        myreq.tensor = sharded_output_embeddings
        myreq.a2ai = a2ai
        myreq.wait_function = All2All_Seq_Req_Wait
        ctx.myreq = myreq
        myreq.qcomm_ctx = qcomm_ctx
        ctx.pg = pg
        ctx.my_rank = my_rank
        ctx.world_size = world_size
        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused) -> Tuple[None, None, None, Tensor]:
        myreq = ctx.myreq
        a2ai = myreq.a2ai
        D = a2ai.embedding_dim
        variable_batch_size = a2ai.variable_batch_size
        backward_recat_tensor = a2ai.backward_recat_tensor
        permuted_lengths_after_sparse_data_all2all = (
            a2ai.permuted_lengths_after_sparse_data_all2all
        )
        assert myreq.req is not None
        myreq.req.wait()
        sharded_grad_input = myreq.tensor
        if _seq_a2a_bwd_record_stream():
            # The a2a buffers were allocated on the stream that launched it, and
            # wait() only orders this stream after the collective. Without this,
            # freeing them lets the allocator reuse or unmap (expandable segments)
            # memory the collective and the decode below may still be using.
            current_stream = torch.get_device_module(
                sharded_grad_input.device
            ).current_stream()
            sharded_grad_input.record_stream(current_stream)
            if myreq.send_tensor is not None:
                myreq.send_tensor.record_stream(current_stream)
        rowmap_grad_input = None
        if (
            a2ai.codecs is not None
            and permuted_lengths_after_sparse_data_all2all is not None
            and os.environ.get("TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD", "1") == "1"
        ):
            with record_function("## All2All_Seq_bwd_dequant_rowmap ##"):
                rowmap_grad_input = _seq_a2a_bwd_dequant_rowmap(
                    a2ai, sharded_grad_input, myreq.qcomm_ctx
                )
        if rowmap_grad_input is not None:
            sharded_grad_input = rowmap_grad_input
        elif a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            sharded_grad_input = _seq_a2a_codec_decode(
                codecs.backward,
                sharded_grad_input,
                myreq.qcomm_ctx,
                "bwd_dec",
                "All2All_Seq_Req.backward without the row-map decode "
                "(TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD)",
            )
        myreq.req = None
        myreq.tensor = None
        myreq.send_tensor = None
        myreq.dummy_tensor = None

        if (
            rowmap_grad_input is None
            and permuted_lengths_after_sparse_data_all2all is not None
        ):
            with record_function("## All2All_Seq_bwd_permute ##"):
                if not variable_batch_size:
                    _, sharded_grad_input, _ = _permute_2D_sparse_data(
                        backward_recat_tensor,
                        permuted_lengths_after_sparse_data_all2all,
                        sharded_grad_input,
                        None,
                        sharded_grad_input.numel(),
                    )
                else:
                    _, sharded_grad_input, _ = torch.ops.fbgemm.permute_1D_sparse_data(
                        backward_recat_tensor,
                        permuted_lengths_after_sparse_data_all2all,
                        sharded_grad_input,
                        None,
                        sharded_grad_input.numel(),
                    )
        if GRADIENT_DIVISION:
            sharded_grad_input.div_(dist.get_world_size(ctx.pg))
        return (None, None, None, sharded_grad_input.view(-1, D))


class All2All_Seq_Req_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: torch.Tensor,
    ) -> Tensor:
        a2ai = myreq.a2ai
        # pyrefly: ignore[missing-attribute]
        D = a2ai.embedding_dim
        ctx.a2ai = a2ai
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        sharded_output_embeddings = myreq.tensor
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq
        row_map = myreq.output_row_map
        myreq.output_row_map = None
        # pyrefly: ignore[missing-attribute]
        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            if row_map is not None:
                with record_function("## All2All_Seq_fwd_decode_unbucketize ##"):
                    unbucketized = _seq_a2a_fwd_decode_unbucketize(
                        # pyrefly: ignore[bad-argument-type]
                        a2ai,
                        # pyrefly: ignore[bad-argument-type]
                        sharded_output_embeddings,
                        myreq.qcomm_ctx,
                        row_map,
                    )
                if unbucketized is not None:
                    ctx.save_for_backward(row_map)
                    ctx.row_map_on = True
                    myreq.row_map_applied = True
                    return unbucketized
            seq_info = cast(All2AllSequenceInfo, a2ai)
            if seq_info.decode_row_permute is not None:
                sharded_output_embeddings = _select_encoded_rows(
                    none_throws(sharded_output_embeddings),
                    seq_info.decode_row_permute,
                    none_throws(seq_info.codecs).forward.calc_quantized_size(
                        D, myreq.qcomm_ctx
                    ),
                )
            sharded_output_embeddings = _seq_a2a_fwd_decode(
                # pyrefly: ignore [missing-attribute]
                codecs.forward,
                D,
                # pyrefly: ignore[bad-argument-type]
                sharded_output_embeddings,
                myreq.qcomm_ctx,
            )
        # pyrefly: ignore[missing-attribute]
        return sharded_output_embeddings.view(-1, D)

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, sharded_grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        a2ai = ctx.a2ai
        pg = ctx.pg
        input_splits = a2ai.output_splits
        output_splits = a2ai.input_splits
        if a2ai.decode_row_permute is not None:
            raise RuntimeError(
                "a row permute folded into the sequence all-to-all decode has no "
                "backward"
            )

        if a2ai.codecs is not None:
            codecs = none_throws(a2ai.codecs)
            qcomm_ctx = codecs.backward.create_context()
            # Set only by a forward that applied the unbucketize permute, which
            # needs MX4 codecs (TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP).
            row_map = (
                ctx.saved_tensors[0] if getattr(ctx, "row_map_on", False) else None
            )
            if row_map is not None and not _seq_a2a_bwd_unbucketize_packed(
                a2ai, sharded_grad_output, row_map, qcomm_ctx
            ):
                with record_function("## All2All_Seq_bwd_unbucketize ##"):
                    sharded_grad_output = _seq_a2a_unbucketize_grad(
                        sharded_grad_output, row_map
                    )
                row_map = None
            unbucketized_grad = sharded_grad_output
            chunked_payload = None
            if os.environ.get("TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE", "1") == "1":
                with record_function("## All2All_Seq_bwd_chunked_encode ##"):
                    chunked_payload = _seq_a2a_bwd_chunked_encode(
                        a2ai, sharded_grad_output, qcomm_ctx
                    )
            if chunked_payload is not None:
                sharded_grad_output = chunked_payload
            else:
                sharded_grad_output = _seq_a2a_codec_encode(
                    codecs.backward,
                    sharded_grad_output,
                    qcomm_ctx,
                    "bwd_enc",
                    "All2All_Seq_Req_Wait.backward without the row-chunked "
                    "encode (TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE)",
                )
            if row_map is not None:
                with record_function("## All2All_Seq_bwd_unbucketize_payload ##"):
                    sharded_grad_output = _seq_a2a_bwd_unbucketize_payload(
                        a2ai, unbucketized_grad, sharded_grad_output, row_map, qcomm_ctx
                    )
            output_splits = [
                a2ai.codecs.backward.calc_quantized_size(x, qcomm_ctx)
                for x in output_splits
            ]
            input_splits = [
                a2ai.codecs.backward.calc_quantized_size(x, qcomm_ctx)
                for x in input_splits
            ]
        else:
            qcomm_ctx = None

        sharded_grad_input = torch.empty(
            sum(output_splits),
            device=sharded_grad_output.device,
            dtype=sharded_grad_output.dtype,
        )
        with record_function("## All2All_Seq_bwd ##"):
            req = dist.all_to_all_single(
                output=sharded_grad_input,
                input=sharded_grad_output.view(-1),
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = sharded_grad_input
        if _seq_a2a_bwd_record_stream():
            myreq.send_tensor = sharded_grad_output
        myreq.qcomm_ctx = qcomm_ctx

        return (None, None, myreq.dummy_tensor)


def _select_encoded_rows(
    encoded: Tensor, row_permute: Tensor, row_numel: int
) -> Tensor:
    rows = encoded.view(-1, row_numel)
    row_bytes = row_numel * encoded.element_size()
    # Gathering rows of wider elements is several times faster than byte rows.
    for dtype in (torch.int64, torch.int32, torch.int16):
        if row_bytes % dtype.itemsize == 0:
            rows = rows.view(dtype)
            break
    return torch.index_select(rows, 0, row_permute).view(encoded.dtype).view(-1)


def fold_row_permute_into_seq_decode(
    tensor_awaitable: Awaitable[Tensor], row_permute: Tensor
) -> bool:
    """Makes a pending `alltoall_sequence` select ``row_permute`` rows before decoding.

    MX4 groups decode independently, so when each embedding row is a whole number
    of groups, selecting encoded rows and then decoding is bit-identical to decoding
    and then selecting, and the gather moves about 1/7 of the bytes. Only applies
    with ``TORCHREC_SEQ_A2A_PERMUTE_BEFORE_DECODE=1`` and when nothing can
    backpropagate through the request, since the folded select has no backward.

    Args:
        tensor_awaitable (Awaitable[Tensor]): awaitable returned by
            `alltoall_sequence`.
        row_permute (Tensor): row indices to select from the received embeddings.

    Returns:
        bool: whether the select was folded; if not, the caller must apply it.
    """
    if os.environ.get(SEQ_PERMUTE_BEFORE_DECODE_ENV, "0") != "1":
        return False
    if (
        not isinstance(tensor_awaitable, Request)
        or tensor_awaitable.wait_function is not All2All_Seq_Req_Wait
        or tensor_awaitable.dummy_tensor.requires_grad
    ):
        return False
    a2ai = tensor_awaitable.a2ai
    qcomm_ctx = tensor_awaitable.qcomm_ctx
    if not isinstance(a2ai, All2AllSequenceInfo) or a2ai.codecs is None:
        return False
    codec = a2ai.codecs.forward
    if (
        getattr(codec, "_comm_precision", None) != SparseType.MX4
        or getattr(codec, "_loss_scale", None) is not None
        or qcomm_ctx is None
        or a2ai.embedding_dim % qcomm_ctx.mx_group_size != 0
    ):
        return False
    a2ai.decode_row_permute = row_permute
    return True


class All2Allv_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        a2ai: All2AllVInfo,
        inputs: List[Tensor],
    ) -> Tensor:
        input_split_sizes = [m * sum(a2ai.D_local_list) for m in a2ai.B_local_list]
        output_split_sizes = [a2ai.B_local * e for e in a2ai.dims_sum_per_rank]
        input = torch.cat(inputs, dim=1).view([-1])
        if a2ai.codecs is not None:
            input = a2ai.codecs.forward.encode(input)

        output = input.new_empty(sum(output_split_sizes))
        with record_function("## All2Allv_fwd ##"):
            req = dist.all_to_all_single(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                group=pg,
                async_op=True,
            )

        myreq.req = req
        myreq.tensor = output
        myreq.wait_function = All2Allv_Wait
        a2ai.input_split_sizes = input_split_sizes
        a2ai.output_split_sizes = output_split_sizes
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        ctx.myreq = myreq
        ctx.tensor = output
        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        a2ai = ctx.a2ai
        myreq = ctx.myreq
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        if a2ai.codecs is not None:
            grad_input = a2ai.codecs.backward.decode(grad_input)

        grad_inputs = grad_input.view([a2ai.B_global, -1]).split(
            a2ai.D_local_list, dim=1
        )
        grad_inputs = [gin.contiguous() for gin in grad_inputs]
        myreq.tensor = None
        myreq.dummy_tensor = None
        return (None, None, None, *grad_inputs)


class All2Allv_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: torch.Tensor,
    ) -> Tuple[Tensor]:
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        output = myreq.tensor
        myreq.tensor = None
        ctx.pg = pg
        ctx.myreq = myreq

        # pyrefly: ignore[missing-attribute]
        if a2ai.codecs is not None:
            # pyrefly: ignore[missing-attribute]
            output = a2ai.codecs.forward.decode(output)
        outputs = tuple(
            [
                # pyrefly: ignore[missing-attribute]
                out.view([a2ai.B_local, -1])
                # pyrefly: ignore[missing-attribute]
                for out in output.split(a2ai.output_split_sizes)
            ]
        )
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs) -> Tuple[None, None, Tensor]:
        pg = ctx.pg
        myreq = ctx.myreq
        a2ai = ctx.a2ai

        if a2ai.codecs is not None:
            grad_outputs = a2ai.codecs.backward.encode(grad_outputs)

        grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
        grad_output = torch.cat(grad_outputs)
        grad_input = grad_output.new_empty([a2ai.B_global * sum(a2ai.D_local_list)])
        with record_function("## All2Allv_bwd ##"):
            req = dist.all_to_all_single(
                grad_input,
                grad_output,
                a2ai.input_split_sizes,
                a2ai.output_split_sizes,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_input
        return (None, None, myreq.dummy_tensor)


class ReduceScatter_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        rsi: ReduceScatterInfo,
        *inputs: Any,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)

        if rsi.codecs is not None:
            # pyrefly: ignore[bad-assignment]
            inputs = [rsi.codecs.forward.encode(input) for input in inputs]

        output = inputs[my_rank].new_empty(
            inputs[my_rank].size(),
            dtype=inputs[my_rank].dtype,
            device=inputs[my_rank].device,
        )
        with record_function("## ReduceScatter_fwd ##"):
            req = dist.reduce_scatter(
                output,
                list(inputs),
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = output
        myreq.wait_function = ReduceScatter_Wait
        myreq.rsi = rsi
        ctx.myreq = myreq
        ctx.pg = pg
        ctx.tensor = output
        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused: Tensor) -> Tuple[Optional[Tensor], ...]:
        myreq = ctx.myreq
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        grad_inputs = list(myreq.tensor)
        rsi = myreq.rsi
        if rsi.codecs is not None:
            grad_inputs = [
                rsi.codecs.backward.decode(grad_input) for grad_input in grad_inputs
            ]
        # Make it equivalent to running on a single rank.
        if GRADIENT_DIVISION:
            for grad_input in grad_inputs:
                grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        myreq.dummy_tensor
        return (None, None, None, *grad_inputs)


class ReduceScatter_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: Tensor,
    ) -> Tensor:
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        output = myreq.tensor
        myreq.tensor = None
        ctx.myreq = myreq
        ctx.pg = pg

        rsi = myreq.rsi
        # pyrefly: ignore[missing-attribute]
        if rsi.codecs is not None:
            # pyrefly: ignore[missing-attribute]
            output = rsi.codecs.forward.decode(output)
        # pyrefly: ignore[bad-return]
        return output

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        rsi = myreq.rsi
        if rsi.codecs is not None:
            grad_output = rsi.codecs.backward.encode(grad_output)

        grad_inputs = [
            grad_output.new_empty(
                in_size,
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            for in_size in rsi.input_sizes
        ]

        with record_function("## ReduceScatter_bwd (all_gather) ##"):
            req = dist.all_gather(
                grad_inputs,
                grad_output.contiguous(),
                group=ctx.pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_inputs
        return (None, None, myreq.dummy_tensor)


class ReduceScatterBase_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        rsi: ReduceScatterBaseInfo,
        inputs: Tensor,
    ) -> Tensor:
        my_size = dist.get_world_size(pg)
        assert inputs.size(0) % my_size == 0
        if rsi.codecs is not None:
            inputs = rsi.codecs.forward.encode(inputs)
        output = inputs.new_empty((inputs.size(0) // my_size, inputs.size(1)))
        with record_function("## ReduceScatterBase_fwd (tensor) ##"):
            req = dist.reduce_scatter_tensor(
                output,
                inputs,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = output
        myreq.wait_function = ReduceScatterBase_Wait
        myreq.rsi = rsi
        myreq.tensor = output
        ctx.myreq = myreq
        ctx.pg = pg

        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused: Tensor) -> Tuple[Optional[Tensor], ...]:
        myreq = ctx.myreq
        myreq.req.wait()
        myreq.req = None
        grad_inputs = myreq.tensor
        rsi = myreq.rsi
        if rsi.codecs is not None:
            grad_inputs = rsi.codecs.backward.decode(grad_inputs)
        # Make it equivalent to running on a single rank.
        if GRADIENT_DIVISION:
            grad_inputs.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        myreq.dummy_tensor = None
        return (None, None, None, grad_inputs)


class ReduceScatterBase_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_Tensor: Tensor,
    ) -> Tensor:
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        output = myreq.tensor
        myreq.tensor = None
        ctx.myreq = myreq
        ctx.pg = pg
        rsi = myreq.rsi

        # pyrefly: ignore[missing-attribute]
        if rsi.codecs is not None:
            # pyrefly: ignore[missing-attribute]
            output = rsi.codecs.forward.decode(output)
        # pyrefly: ignore[bad-return]
        return output

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        rsi = myreq.rsi

        if rsi.codecs is not None:
            grad_output = rsi.codecs.backward.encode(grad_output)
        grad_inputs = grad_output.new_empty(rsi.input_sizes)
        with record_function("## ReduceScatterBase_bwd (all_gather) ##"):
            req = dist.all_gather_into_tensor(
                grad_inputs,
                grad_output.contiguous(),
                group=ctx.pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_inputs
        return (None, None, myreq.dummy_tensor)


class AllGatherBase_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        agi: AllGatherBaseInfo,
        input: Tensor,
    ) -> Tensor:
        my_size = dist.get_world_size(pg)

        if agi.codecs is not None:
            input = agi.codecs.forward.encode(input)

        outputs = input.new_empty((input.size(0) * my_size, input.size(1)))
        with record_function("## AllGatherBase_fwd (into_tensor) ##"):
            req = dist.all_gather_into_tensor(
                outputs,
                input,
                group=pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = outputs
        myreq.wait_function = AllGatherBase_Wait
        myreq.agi = agi
        ctx.myreq = myreq
        ctx.pg = pg
        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused: Tensor) -> Tuple[Optional[Tensor], ...]:
        myreq = ctx.myreq
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        agi = myreq.agi
        grad_input = myreq.tensor
        if agi.codecs is not None:
            grad_input = agi.codecs.backward.decode(grad_input)

        # Make it equivalent to running on a single rank.
        if GRADIENT_DIVISION:
            grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        myreq.dummy_tensor = None
        return (None, None, None, grad_input)


class AllGatherBase_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: Tensor,
    ) -> Tensor:
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        outputs = myreq.tensor
        myreq.tensor = None
        ctx.myreq = myreq
        ctx.pg = pg

        agi = myreq.agi
        # pyrefly: ignore[missing-attribute]
        if agi.codecs is not None:
            # pyrefly: ignore[missing-attribute]
            outputs = agi.codecs.forward.decode(outputs)
        # pyrefly: ignore[bad-return]
        return outputs

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_outputs: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        agi = myreq.agi

        if agi.codecs is not None:
            grad_outputs = agi.codecs.backward.encode(grad_outputs)
        grad_input = grad_outputs.new_empty(agi.input_size)
        with record_function("## AllGatherBase_bw (reduce_scatter_tensor) ##"):
            req = dist.reduce_scatter_tensor(
                grad_input,
                grad_outputs.contiguous(),
                group=ctx.pg,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_input

        return (None, None, myreq.dummy_tensor)


class ReduceScatterV_Req(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        rsi: ReduceScatterVInfo,
        input: Tensor,
    ) -> Tensor:
        my_rank = dist.get_rank(pg)

        if rsi.codecs is not None:
            input = rsi.codecs.forward.encode(input)

        output = input.new_empty(rsi.input_sizes[my_rank])

        # Use dist.reduce_scatter_tensor when a vector reduce-scatter is not needed
        # else use dist.reduce_scatter which internally supports vector reduce-scatter
        if rsi.equal_splits:
            with record_function("## ReduceScatterV_fwd (tensor) ##"):
                req = dist.reduce_scatter_tensor(
                    output,
                    input,
                    group=pg,
                    async_op=True,
                )
        else:
            with record_function("## ReduceScatterV_fwd ##"):
                req = dist.reduce_scatter(
                    output,
                    list(torch.split(input, rsi.input_splits)),
                    group=pg,
                    async_op=True,
                )

        myreq.req = req
        myreq.tensor = output
        myreq.wait_function = ReduceScatterV_Wait
        myreq.rsi = rsi
        ctx.myreq = myreq
        ctx.pg = pg

        return myreq.dummy_tensor

    @staticmethod
    def backward(ctx, *unused: Tensor) -> Tuple[Optional[Tensor], ...]:
        myreq = ctx.myreq
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        rsi = myreq.rsi
        if rsi.codecs is not None:
            grad_input = rsi.codecs.backward.decode(grad_input)
        # Make it equivalent to running on a single rank.
        if GRADIENT_DIVISION:
            grad_input.div_(dist.get_world_size(ctx.pg))
        myreq.tensor = None
        myreq.dummy_tensor = None
        return (None, None, None, grad_input)


class ReduceScatterV_Wait(Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        pg: dist.ProcessGroup,
        myreq: Request[Tensor],
        *dummy_tensor: Tensor,
    ) -> Tensor:
        assert myreq.req is not None
        myreq.req.wait()
        myreq.req = None
        # pyrefly: ignore[bad-assignment]
        output: torch.Tensor = myreq.tensor
        myreq.tensor = None

        ctx.myreq = myreq
        ctx.pg = pg

        rsi = myreq.rsi
        # pyrefly: ignore[missing-attribute]
        if rsi.codecs is not None:
            # pyrefly: ignore[missing-attribute]
            output = rsi.codecs.forward.decode(output)

        return output

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_output: Tensor) -> Tuple[None, None, Tensor]:
        myreq = ctx.myreq
        rsi = myreq.rsi
        if rsi.codecs is not None:
            grad_output = rsi.codecs.backward.encode(grad_output)
        grad_input = grad_output.new_empty(rsi.total_input_size)

        if rsi.equal_splits:
            with record_function("## ReduceScatterV_bwd (all_gather) ##"):
                req = dist.all_gather_into_tensor(
                    grad_input,
                    grad_output.contiguous(),
                    group=ctx.pg,
                    async_op=True,
                )
        else:
            with record_function("## ReduceScatterV_bwd (all_gather_v) ##"):
                req = dist.all_gather(
                    list(torch.split(grad_input, rsi.input_splits)),
                    grad_output.contiguous(),
                    group=ctx.pg,
                    async_op=True,
                )
        myreq.req = req
        myreq.tensor = grad_input
        return (None, None, myreq.dummy_tensor)


class AllToAllSingle(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(
        ctx,
        input: Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        group_name: str,
        group_size: int,
        gradient_division: bool,
    ) -> Tensor:
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        ctx.group_name = group_name
        ctx.group_size = group_size
        ctx.gradient_division = gradient_division
        return torch.distributed._functional_collectives.all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            # pyrefly: ignore[bad-argument-type]
            group_name,
        )

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad):
        grad = torch.distributed._functional_collectives.all_to_all_single(
            grad,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            ctx.group_name,
        )
        if ctx.gradient_division:
            grad.div_(ctx.group_size)

        return grad, None, None, None, None, None


# torchrec::reduce_scatter_tensor
@torch.library.custom_op("torchrec::reduce_scatter_tensor", mutates_args=())
def reduce_scatter_tensor(
    input: Tensor,
    reduceOp: str,
    group_size: int,
    group_name: str,
    gradient_division: bool,
) -> Tensor:
    out = torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduceOp,
        group_size,
        group_name,
    )
    return torch.ops._c10d_functional.wait_tensor(out)


@torch.library.register_fake("torchrec::reduce_scatter_tensor")
def reduce_scatter_tensor_fake(
    input: Tensor,
    reduceOp: str,
    group_size: int,
    group_name: str,
    gradient_division: bool,
) -> Tensor:
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduceOp,
        group_size,
        group_name,
    )


def reduce_scatter_tensor_setup_context(ctx, inputs, output) -> None:
    _, _, group_size, group_name, gradient_division = inputs
    ctx.group_size = group_size
    ctx.group_name = group_name
    ctx.gradient_division = gradient_division


def reduce_scatter_tensor_backward(ctx, grad):
    # TODO(ivankobzarev): Support codecs(quantization) on backward
    out = torch.ops._c10d_functional.all_gather_into_tensor(
        grad,
        ctx.group_size,
        ctx.group_name,
    )
    grad = torch.ops._c10d_functional.wait_tensor(out)
    if ctx.gradient_division:
        grad.div_(ctx.group_size)

    return grad, None, None, None, None


torch.library.register_autograd(
    "torchrec::reduce_scatter_tensor",
    reduce_scatter_tensor_backward,
    setup_context=reduce_scatter_tensor_setup_context,
)


# torchrec::all_gather_into_tensor
@torch.library.custom_op("torchrec::all_gather_into_tensor", mutates_args=())
def all_gather_into_tensor(
    shard: Tensor,
    gather_dim: int,
    group_size: int,
    group_name: str,
    gradient_division: bool,
) -> Tensor:
    out = torch.ops._c10d_functional.all_gather_into_tensor(
        shard, group_size, group_name
    )
    return torch.ops._c10d_functional.wait_tensor(out)


@torch.library.register_fake("torchrec::all_gather_into_tensor")
def all_gather_into_tensor_fake(
    shard: Tensor,
    gather_dim: int,
    group_size: int,
    group_name: str,
    gradient_division: bool,
) -> Tensor:
    return torch.ops._c10d_functional.all_gather_into_tensor(
        shard, group_size, group_name
    )


def all_gather_into_tensor_setup_context(ctx, inputs, output) -> None:
    _, gather_dim, group_size, group_name, gradient_division = inputs
    ctx.group_size = group_size
    ctx.group_name = group_name
    ctx.gradient_division = gradient_division


def all_gather_into_tensor_backward(ctx, grad):
    # TODO(ivankobzarev): Support codecs(quantization) on backward
    out = torch.ops._c10d_functional.reduce_scatter_tensor(
        grad,
        "sum",
        ctx.group_size,
        ctx.group_name,
    )
    grad = torch.ops._c10d_functional.wait_tensor(out)
    if ctx.gradient_division:
        grad.div_(ctx.group_size)

    return grad, None, None, None, None


torch.library.register_autograd(
    "torchrec::all_gather_into_tensor",
    all_gather_into_tensor_backward,
    setup_context=all_gather_into_tensor_setup_context,
)


@torch.library.custom_op("torchrec::_split_1d_cat_2d", mutates_args=())
def _split_1d_cat_2d_impl(t: torch.Tensor, dim0: int, dim1s: List[int]) -> torch.Tensor:
    torch._check_is_size(dim0)
    [torch._check_is_size(dim1) for dim1 in dim1s]
    splits: List[torch.Tensor] = t.split([dim0 * dim1 for dim1 in dim1s])
    return torch.cat(
        [s.reshape(dim0, dim1) for s, dim1 in zip(splits, dim1s)],
        dim=1,
    )


@torch.library.register_fake("torchrec::_split_1d_cat_2d")
def _split_1d_cat_2d_impl_abstract(
    t: torch.Tensor, dim0: int, dim1s: List[int]
) -> torch.Tensor:
    return t.new_empty([dim0, sum(dim1s)])


@torch.library.custom_op("torchrec::_split_1d_cat_2d_backward_impl", mutates_args=())
def _split_1d_cat_2d_backward_impl(
    grad: torch.Tensor, dim1s: List[int]
) -> torch.Tensor:
    splits = grad.split(dim1s, dim=1)
    return torch.cat([s.reshape(-1) for s in splits], dim=0)


@torch.library.register_fake("torchrec::_split_1d_cat_2d_backward_impl")
def _split_1d_cat_2d_backward_impl_fake(
    grad: torch.Tensor, dim1s: List[int]
) -> torch.Tensor:
    return grad.new_empty([grad.numel()])


def _split_1d_cat_2d_backward(ctx, grad):
    ret = torch.ops.torchrec._split_1d_cat_2d_backward_impl(grad, ctx.dim1s)
    return ret, None, None


def _split_1d_cat_2d_setup_context(ctx, inputs, output):
    (x, dim0, dim1s) = inputs
    ctx.dim1s = dim1s


torch.library.register_autograd(
    "torchrec::_split_1d_cat_2d",
    _split_1d_cat_2d_backward,
    setup_context=_split_1d_cat_2d_setup_context,
)
