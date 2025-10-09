#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.profiler import record_function
from torchrec.distributed.comm_ops import (
    all_gather_base_pooled,
    alltoall_pooled,
    alltoall_sequence,
    reduce_scatter_base_pooled,
    reduce_scatter_v_per_feature_pooled,
    reduce_scatter_v_pooled,
    variable_batch_alltoall_pooled,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.global_settings import get_propogate_device
from torchrec.distributed.types import Awaitable, QuantizedCommCodecs, rank_device
from torchrec.fx.utils import fx_marker
from torchrec.pt2.checks import is_torchdynamo_compiling
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
except (OSError, RuntimeError):
    pass

# OSS
try:
    pass
except ImportError:
    pass


logger: logging.Logger = logging.getLogger()


def _get_recat(
    local_split: int,
    num_splits: int,
    stagger: int = 1,
    device: Optional[torch.device] = None,
    batch_size_per_rank: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    """
    Calculates relevant recat indices required to reorder AlltoAll collective.

    Args:
        local_split (int): number of features in local split.
        num_splits (int): number of splits (typically WORLD_SIZE).
        stagger (int): secondary reordering, (typically 1, but
            `WORLD_SIZE/LOCAL_WORLD_SIZE` for TWRW).
        device (Optional[torch.device]): device on which buffer will be allocated.
        batch_size_per_rank (Optional[List[int]]): batch size per rank, needed for
            variable batch size.

    Returns:
        Optional[torch.Tensor]: recat tensor, None if local rank is empty.

    Example::

        _recat(2, 4, 1)
            # [0, 2, 4, 6, 1, 3, 5, 7]
        _recat(2, 4, 2)
            # [0, 4, 2, 6, 1, 5, 3, 7]
        _recat(0, 4, 2)
            # None
    """

    with record_function("## all2all_data:recat_permute_gen ##"):
        if local_split == 0:
            return None

        recat: List[int] = []

        feature_order: List[int] = [
            x + num_splits // stagger * y
            for x in range(num_splits // stagger)
            for y in range(stagger)
        ]

        for i in range(local_split):
            for j in feature_order:  # range(num_splits):
                recat.append(i + j * local_split)

        vb_per_rank_condition: bool = False
        if not is_torchdynamo_compiling():
            vb_per_rank_condition = batch_size_per_rank is not None and any(
                bs != batch_size_per_rank[0] for bs in batch_size_per_rank
            )

        # variable batch per rank
        if vb_per_rank_condition:
            batch_size_per_feature = list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, local_split)
                    # pyre-ignore
                    for x in batch_size_per_rank
                )
            )
            permuted_batch_size_per_feature = [batch_size_per_feature[r] for r in recat]
            input_offset = [0] + list(itertools.accumulate(batch_size_per_feature))
            output_offset = [0] + list(
                itertools.accumulate(permuted_batch_size_per_feature)
            )
            recat_tensor = torch.tensor(
                recat,
                device=device,
                dtype=torch.int32,
            )
            input_offset_tensor = torch.tensor(
                input_offset,
                device=device,
                dtype=torch.int32,
            )
            output_offset_tensor = torch.tensor(
                output_offset,
                device=device,
                dtype=torch.int32,
            )
            recat = torch.ops.fbgemm.expand_into_jagged_permute(
                recat_tensor,
                input_offset_tensor,
                output_offset_tensor,
                output_offset[-1],
            )
            return recat
        else:
            return torch.tensor(recat, device=device, dtype=torch.int32)


class _MergePooledEmbeddingsModuleImpl(torch.nn.Module):
    """
    Does the merge_pooled_embeddings operation. Separate module necessary for lowering.

    Args:
        device (torch.device): device for fbgemm.merge_pooled_embeddings
    """

    current_device: torch.device

    def __init__(
        self,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.current_device = device

    # the method will be used by inference application to update the
    # device information
    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self.current_device = torch.device(device_str)

    def forward(self, tensors: List[torch.Tensor], cat_dim: int) -> torch.Tensor:
        """
        Here we assume input tensors are:
        [TBE_output_0, ..., TBE_output_(n-1)]
        """
        B = tensors[0].size(1 - cat_dim)
        return torch.ops.fbgemm.merge_pooled_embeddings(
            tensors,
            B,
            self.current_device,
            cat_dim,
        )


class MergePooledEmbeddingsModule(torch.nn.Module):
    """
    This module is used for merge_pooled_embedding_optimization.
    _MergePooledEmbeddingsModuleImpl provides the `set_device` API
    to set device at model loading time.

    Args:
        device (torch.device): device for fbgemm.merge_pooled_embeddings
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.impl = _MergePooledEmbeddingsModuleImpl(device)

    # This method can be used by an inference runtime to update the
    # device information for this module.
    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self.impl.set_device(device_str)

    def forward(self, tensors: List[torch.Tensor], cat_dim: int) -> torch.Tensor:
        """
        Calls _MergePooledEmbeddingsModuleImpl with tensors and cat_dim.

        Args:
            tensors (List[torch.Tensor]): list of embedding tensors.
            cat_dim (int): which dimension you would like to concatenate on.

        Returns:
            torch.Tensor: merged embeddings.
        """
        merged_embeddings = self.impl(tensors, cat_dim)
        return merged_embeddings


class SplitsAllToAllAwaitable(Awaitable[List[List[int]]]):
    """
    Awaitable for splits AlltoAll.

    Args:
        input_tensors (List[torch.Tensor]): tensor of splits to redistribute.
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
    """

    def __init__(
        self,
        input_tensors: List[torch.Tensor],
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.num_workers: int = pg.size()

        if is_torchdynamo_compiling():
            # TODO(ivankobzarev) Remove this dynamo condition once dynamo functional collectives remapping does not emit copy_
            # https://github.com/pytorch/pytorch/issues/122788
            with record_function("## all2all_data:kjt splits ##"):
                input_tensor = torch.stack(input_tensors, dim=1).flatten()
                if pg._get_backend_name() == "custom":
                    self._output_tensor = torch.empty(
                        [self.num_workers * len(input_tensors)],
                        device=input_tensors[0].device,
                        dtype=input_tensors[0].dtype,
                    )

                    self._output_tensor = input_tensor[
                        : input_tensor.size(0) // 2
                    ].repeat(2)
                else:
                    self._output_tensor = (
                        dist._functional_collectives.all_to_all_single(
                            input_tensor,
                            output_split_sizes=None,
                            input_split_sizes=None,
                            group=pg,
                        )
                    )
            # To avoid hasattr in _wait_impl to check self._splits_awaitable
            # pyre-ignore
            self._splits_awaitable = None
        else:
            with record_function("## all2all_data:kjt splits ##"):
                self._output_tensor: torch.Tensor = torch.empty(
                    [self.num_workers * len(input_tensors)],
                    device=input_tensors[0].device,
                    dtype=input_tensors[0].dtype,
                )
                input_tensor = torch.stack(input_tensors, dim=1).flatten()
                self._splits_awaitable: dist.Work = dist.all_to_all_single(
                    output=self._output_tensor,
                    input=input_tensor,
                    group=pg,
                    async_op=not is_torchdynamo_compiling(),
                )

    def _wait_impl(self) -> List[List[int]]:
        # Can not use is_torchdynamo_compiling(), as every such condition should be independent for compilation with graph breaks.
        if isinstance(self._splits_awaitable, dist.Work):
            self._splits_awaitable.wait()

        ret = self._output_tensor.view(self.num_workers, -1).T.tolist()

        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            for i in range(len(ret)):
                for j in range(len(ret[i])):
                    torch._check_is_size(ret[i][j])

        return ret


class KJTAllToAllTensorsAwaitable(Awaitable[KeyedJaggedTensor]):
    """
    Awaitable for KJT tensors AlltoAll.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        input (KeyedJaggedTensor): input KJT.
        splits (List[int]): list of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        input_splits (List[List[int]]): input splits (number of values each rank will
            get) for each tensor in AlltoAll.
        output_splits (List[List[int]]): output splits (number of values per rank in
            output) for each tensor in AlltoAll.
        input_tensors (List[torch.Tensor]): provided KJT tensors (ie. lengths, values)
            to redistribute according to splits.
        labels (List[str]): labels for each provided tensor.
        keys (List[str]): KJT keys after AlltoAll.
        device (torch.device): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor.
        stride_per_rank (Optional[List[int]]): stride per rank in the non variable
            batch per feature case.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: KeyedJaggedTensor,
        splits: List[int],
        input_splits: List[List[int]],
        output_splits: List[List[int]],
        input_tensors: List[torch.Tensor],
        labels: List[str],
        keys: List[str],
        device: torch.device,
        stagger: int,
        stride_per_rank: Optional[List[int]],
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._pg: dist.ProcessGroup = pg
        self._device: torch.device = device
        self._input = input
        self._splits = splits
        self._input_splits_list = input_splits
        self._output_splits_list = output_splits
        self._input_splits: Dict[str, List[int]] = dict(zip(labels, input_splits))
        self._output_splits: Dict[str, List[int]] = dict(zip(labels, output_splits))
        self._keys = keys
        self._stagger = stagger
        self._stride_per_rank = stride_per_rank
        self._recat: Optional[torch.Tensor] = _get_recat(
            local_split=splits[pg.rank()],
            num_splits=len(splits),
            stagger=stagger,
            device=device,
            batch_size_per_rank=self._stride_per_rank,
        )
        if self._workers == 1:
            return

        self._output_tensors: List[torch.Tensor] = []
        self._awaitables: List[dist.Work] = []
        self._world_size: int = self._pg.size()
        rank = dist.get_rank(self._pg)

        for input_split, output_split, input_tensor, label in zip(
            input_splits,
            output_splits,
            input_tensors,
            labels,
        ):
            if is_torchdynamo_compiling():
                # TODO(ivankobzarev) Remove this dynamo condition once dynamo functional collectives remapping does not emit copy_
                # https://github.com/pytorch/pytorch/issues/122788
                with record_function(f"## all2all_data:kjt {label} ##"):
                    if self._pg._get_backend_name() == "custom":
                        if input_tensor.dim() == 2:
                            output_size = [sum(output_split), input_tensor.size(1)]
                        else:
                            output_size = [sum(output_split)]
                        output_tensor = torch.empty(
                            output_size,
                            device=self._device,
                            dtype=input_tensor.dtype,
                        )
                        _l = sum(output_split[:rank])
                        _r = _l + output_split[rank]
                        torch._check(_r < input_tensor.size(0))
                        torch._check(_l < input_tensor.size(0))
                        torch._check(_l <= _r)
                        torch._check(2 * (_r - _l) == output_tensor.size(0))
                        output_tensor.copy_(
                            input_tensor[_l:_r].repeat(self._world_size)
                        )
                    else:
                        output_tensor = dist._functional_collectives.all_to_all_single(
                            input_tensor,
                            output_split,
                            input_split,
                            pg,
                        )
                    self._output_tensors.append(output_tensor)
            else:
                if input_tensor.dim() == 2:
                    output_size = [sum(output_split), input_tensor.size(1)]
                else:
                    output_size = [sum(output_split)]
                output_tensor = torch.empty(
                    output_size, device=self._device, dtype=input_tensor.dtype
                )
                with record_function(f"## all2all_data:kjt {label} ##"):
                    awaitable = dist.all_to_all_single(
                        output=output_tensor,
                        input=input_tensor,
                        output_split_sizes=output_split,
                        input_split_sizes=input_split,
                        group=self._pg,
                        async_op=not is_torchdynamo_compiling(),
                    )

                self._output_tensors.append(output_tensor)
                self._awaitables.append(awaitable)

    def _wait_impl(self) -> KeyedJaggedTensor:
        """
        Overwrites wait function as we don't handle callbacks here.

        Returns:
            KeyedJaggedTensor: Synced KJT after AlltoAll.
        """

        if self._workers == 1:
            self._input.sync()
            return self._input

        if not is_torchdynamo_compiling():
            for awaitable in self._awaitables:
                awaitable.wait()

        return type(self._input).dist_init(
            keys=self._keys,
            tensors=self._output_tensors,
            variable_stride_per_key=self._input.variable_stride_per_key(),
            num_workers=self._workers,
            recat=self._recat,
            stride_per_rank=self._stride_per_rank,
            stagger=self._stagger,
        )


class KJTAllToAllSplitsAwaitable(Awaitable[KJTAllToAllTensorsAwaitable]):
    """
    Awaitable for KJT tensors splits AlltoAll.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        input (KeyedJaggedTensor): input KJT.
        splits (List[int]): list of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        tensor_splits (Dict[str, List[int]]): tensor splits provided by input KJT.
        input_tensors (List[torch.Tensor]): provided KJT tensors (ie. lengths, values)
            to redistribute according to splits.
        keys (List[str]): KJT keys after AlltoAll.
        device (torch.device): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: KeyedJaggedTensor,
        splits: List[int],
        labels: List[str],
        tensor_splits: List[List[int]],
        input_tensors: List[torch.Tensor],
        keys: List[str],
        device: torch.device,
        stagger: int,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._pg: dist.ProcessGroup = pg
        self._device: torch.device = device
        self._input = input
        self._splits = splits
        self._labels = labels
        self._input_splits = tensor_splits
        self._input_tensors = input_tensors
        self._keys = keys
        self._stagger = stagger
        self._output_splits: List[List[int]] = self._input_splits
        self._stride_per_rank: Optional[List[int]] = (
            None
            if self._input.variable_stride_per_key()
            else [self._input.stride()] * self._workers
        )
        if self._workers == 1:
            return

        input_tensors = [
            torch.tensor(splits, device=device) for splits in self._input_splits
        ]
        if not self._input.variable_stride_per_key():
            input_tensors.append(
                torch.tensor([input.stride()] * self._workers, device=device)
            )

        self._splits_awaitable = SplitsAllToAllAwaitable(
            input_tensors,
            self._pg,
        )

    def _wait_impl(self) -> KJTAllToAllTensorsAwaitable:
        """
        Overwrites wait function as we don't handle callbacks here.

        Returns:
            KJTAllToAllTensorsAwaitable.
        """

        if self._workers > 1:
            output_list = self._splits_awaitable.wait()
            if self._input.variable_stride_per_key():
                self._output_splits = output_list
            else:
                self._output_splits = output_list[:-1]
                self._stride_per_rank = output_list[-1]

            if is_torchdynamo_compiling():
                rank: int = self._pg.rank()
                for i in range(len(self._output_splits)):
                    for j in range(len(self._output_splits[i])):
                        torch._check_is_size(self._output_splits[i][j])
                    torch._check(
                        self._output_splits[i][rank] == self._input_splits[i][rank]
                    )
                if self._stride_per_rank is not None:
                    # pyre-ignore
                    for i in range(len(self._stride_per_rank)):
                        # pyre-ignore
                        torch._check_is_size(self._stride_per_rank[i])

        return KJTAllToAllTensorsAwaitable(
            pg=self._pg,
            input=self._input,
            splits=self._splits,
            input_splits=self._input_splits,
            output_splits=self._output_splits,
            input_tensors=self._input_tensors,
            labels=self._labels,
            keys=self._keys,
            device=self._device,
            stagger=self._stagger,
            stride_per_rank=self._stride_per_rank,
        )


class KJEAllToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to a `ProcessGroup` according to splits.

    Implementation utilizes AlltoAll collective as part of torch.distributed.

    The input provides the necessary tensors, embedding weights and input splits to distribute.
    The first collective call in `KJTAllToAllSplitsAwaitable` will transmit output
    splits (to allocate correct space for tensors) and batch size per rank. The
    following collective calls in `KJTAllToAllTensorsAwaitable` will transmit the actual
    tensors asynchronously.
    This module is used for embedding updates wherein input KJT weights are updated into the embedding tables.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        splits (List[int]): List of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        stagger (int): stagger value to apply to recat tensor, see `_get_recat` function
            for more detail.

    Example::

        keys=['A','B','C']
        splits=[2,1]
        kjeA2A = KJEAllToAll(pg, splits)
        awaitable = kjeA2A(rank0_input)

        # where:
        # rank0_input is KeyedJaggedTensor holding

        #         0           1           2
        # 'A'    [A.V0]       None        [A.V1, A.V2]
        # 'B'    None         [B.V0]      [B.V1]
        # 'C'    [C.V0]       [C.V1]      None

        # rank1_input is KeyedJaggedTensor holding

        #         0           1           2
        # 'A'     [A.V3]      [A.V4]      None
        # 'B'     None        [B.V2]      [B.V3, B.V4]
        # 'C'     [C.V2]      [C.V3]      None

        Output is None since this is write operation but still awaitable for synchronization
        awaitable.wait()

        # where input after the distribution is :
        # rank0

        #         0           1           2           3           4           5
        # 'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None
        # 'B'     None        [B.V0]    [B.V1]        None        [B.V2]      [B.V3, B.V4]

        # rank1
        #         0           1           2           3           4           5
        # 'C'     [C.V0]      [C.V1]      None        [C.V2]      [C.V3]      None
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        splits: List[int],
        stagger: int = 1,
    ) -> None:
        super().__init__()
        torch._check(len(splits) == pg.size())
        self._pg: dist.ProcessGroup = pg
        self._splits = splits
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))
        self._stagger = stagger

    def forward(
        self, input: KeyedJaggedTensor
    ) -> Awaitable[KJTAllToAllTensorsAwaitable]:
        """
        Sends input to relevant `ProcessGroup` ranks.

        The first wait will get the output splits for the provided tensors and issue
        tensors AlltoAll. The second wait will wait for the update.

        Args:
            input (KeyedJaggedTensor): `KeyedJaggedTensor` of values and weights to distribute.

        Returns:
            Awaitable[KJTAllToAllTensorsAwaitable]: awaitable of a `KJTAllToAllTensorsAwaitable`.
        """

        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]

            return KJTAllToAllSplitsAwaitable(
                pg=self._pg,
                input=input,
                splits=self._splits,
                labels=input.dist_labels(),
                tensor_splits=input.dist_splits(self._splits),
                input_tensors=input.dist_tensors(),
                keys=local_keys,
                device=input.device(),
                stagger=self._stagger,
            )


class KJTAllToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to a `ProcessGroup` according to splits.

    Implementation utilizes AlltoAll collective as part of torch.distributed.

    The input provides the necessary tensors and input splits to distribute.
    The first collective call in `KJTAllToAllSplitsAwaitable` will transmit output
    splits (to allocate correct space for tensors) and batch size per rank. The
    following collective calls in `KJTAllToAllTensorsAwaitable` will transmit the actual
    tensors asynchronously.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        splits (List[int]): List of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        stagger (int): stagger value to apply to recat tensor, see `_get_recat` function
            for more detail.

    Example::

        keys=['A','B','C']
        splits=[2,1]
        kjtA2A = KJTAllToAll(pg, splits)
        awaitable = kjtA2A(rank0_input)

        # where:
        # rank0_input is KeyedJaggedTensor holding

        #         0           1           2
        # 'A'    [A.V0]       None        [A.V1, A.V2]
        # 'B'    None         [B.V0]      [B.V1]
        # 'C'    [C.V0]       [C.V1]      None

        # rank1_input is KeyedJaggedTensor holding

        #         0           1           2
        # 'A'     [A.V3]      [A.V4]      None
        # 'B'     None        [B.V2]      [B.V3, B.V4]
        # 'C'     [C.V2]      [C.V3]      None

        rank0_output = awaitable.wait()

        # where:
        # rank0_output is KeyedJaggedTensor holding

        #         0           1           2           3           4           5
        # 'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None
        # 'B'     None        [B.V0]    [B.V1]        None        [B.V2]      [B.V3, B.V4]

        # rank1_output is KeyedJaggedTensor holding
        #         0           1           2           3           4           5
        # 'C'     [C.V0]      [C.V1]      None        [C.V2]      [C.V3]      None
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        splits: List[int],
        stagger: int = 1,
    ) -> None:
        super().__init__()
        torch._check(len(splits) == pg.size())
        self._pg: dist.ProcessGroup = pg
        self._splits = splits
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))
        self._stagger = stagger

    def forward(
        self, input: KeyedJaggedTensor
    ) -> Awaitable[KJTAllToAllTensorsAwaitable]:
        """
        Sends input to relevant `ProcessGroup` ranks.

        The first wait will get the output splits for the provided tensors and issue
        tensors AlltoAll. The second wait will get the tensors.

        Args:
            input (KeyedJaggedTensor): `KeyedJaggedTensor` of values to distribute.

        Returns:
            Awaitable[KJTAllToAllTensorsAwaitable]: awaitable of a `KJTAllToAllTensorsAwaitable`.
        """

        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]

            return KJTAllToAllSplitsAwaitable(
                pg=self._pg,
                input=input,
                splits=self._splits,
                labels=input.dist_labels(),
                tensor_splits=input.dist_splits(self._splits),
                input_tensors=input.dist_tensors(),
                keys=local_keys,
                device=input.device(),
                stagger=self._stagger,
            )


class KJTOneToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to all devices.

    Implementation utilizes OnetoAll function, which essentially P2P copies the feature
    to the devices.

    Args:
        splits (List[int]): lengths of features to split the `KeyJaggedTensor` features
            into before copying them.
        world_size (int): number of devices in the topology.
        device (torch.device): the device on which the KJTs will be allocated.
    """

    def __init__(
        self,
        splits: List[int],
        world_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._splits = splits
        self._world_size = world_size

        if get_propogate_device():
            self._device_type: str = (
                "cpu" if device is None else device.type
            )  # TODO: replace hardcoded cpu with DEFAULT_DEVICE_TYPE in torchrec.distributed.types when torch package issue resolved
        else:
            # BUG: device will default to cuda if cpu specified
            self._device_type: str = (
                device.type
                if device is not None and device.type in {"meta", "cuda", "mtia"}
                else "cuda"
            )
        assert self._world_size == len(splits)

    def forward(self, kjt: KeyedJaggedTensor) -> KJTList:
        """
        Splits features first and then sends the slices to the corresponding devices.

        Args:
            kjt (KeyedJaggedTensor): the input features.

        Returns:
            Awaitable[List[KeyedJaggedTensor]]: awaitable of `KeyedJaggedTensor` splits.
        """
        fx_marker("KJT_ONE_TO_ALL_FORWARD_BEGIN", kjt)
        kjts: List[KeyedJaggedTensor] = kjt.split(self._splits)
        dist_kjts = [
            (
                kjts[rank]
                if self._device_type == "meta"
                else kjts[rank].to(
                    rank_device(self._device_type, rank),
                    non_blocking=True,
                )
            )
            for rank in range(self._world_size)
        ]
        ret = KJTList(dist_kjts)
        fx_marker("KJT_ONE_TO_ALL_FORWARD_END", kjt)
        return ret


class PooledEmbeddingsAwaitable(Awaitable[torch.Tensor]):
    """
    Awaitable for pooled embeddings after collective operation.

    Args:
        tensor_awaitable (Awaitable[torch.Tensor]): awaitable of concatenated tensors
            from all the processes in the group after collective.
    """

    def __init__(
        self,
        tensor_awaitable: Awaitable[torch.Tensor],
    ) -> None:
        super().__init__()
        self._tensor_awaitable = tensor_awaitable

    def _wait_impl(self) -> torch.Tensor:
        """
        Syncs pooled embeddings after collective operation.

        Returns:
            torch.Tensor: synced pooled embeddings.
        """

        ret = self._tensor_awaitable.wait()
        return ret

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks


class PooledEmbeddingsAllToAll(nn.Module):
    # TODO: potentially refactor to take KT instead of torch.Tensor: D29174501
    """
    Shards batches and collects keys of tensor with a `ProcessGroup` according to
    `dim_sum_per_rank`.

    Implementation utilizes `alltoall_pooled` operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
        callbacks (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): callback
            functions.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Example::

        dim_sum_per_rank = [2, 1]
        a2a = PooledEmbeddingsAllToAll(pg, dim_sum_per_rank, device)

        t0 = torch.rand((6, 2))
        t1 = torch.rand((6, 1))
        rank0_output = a2a(t0).wait()
        rank1_output = a2a(t1).wait()
        print(rank0_output.size())
            # torch.Size([3, 3])
        print(rank1_output.size())
            # torch.Size([3, 3])
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._callbacks: List[Callable[[torch.Tensor], torch.Tensor]] = []
        if callbacks is not None:
            self._callbacks = callbacks
        self._dim_sum_per_rank = dim_sum_per_rank
        self._codecs = codecs
        self.register_buffer(
            "_dim_sum_per_rank_tensor",
            torch.tensor(dim_sum_per_rank, device=device, dtype=torch.int),
        )
        cumsum_dim_sum_per_rank = list(itertools.accumulate(dim_sum_per_rank))
        self.register_buffer(
            "_cumsum_dim_sum_per_rank_tensor",
            torch.tensor(cumsum_dim_sum_per_rank, device=device, dtype=torch.int),
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        batch_size_per_rank: Optional[List[int]] = None,
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs AlltoAll pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            batch_size_per_rank (Optional[List[int]]): batch size per rank, to support
                variable batch size.

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings.
        """

        if not batch_size_per_rank:
            B_global = local_embs.size(0)
            assert (
                B_global % self._pg.size() == 0
            ), f"num of ranks {self._pg.size()} doesn't divide global batch size {B_global}"
            B_local = B_global // self._pg.size()
            batch_size_per_rank = [B_local] * self._pg.size()
        tensor_awaitable = alltoall_pooled(
            a2a_pooled_embs_tensor=local_embs,
            batch_size_per_rank=batch_size_per_rank,
            dim_sum_per_rank=self._dim_sum_per_rank,
            dim_sum_per_rank_tensor=self._dim_sum_per_rank_tensor,
            cumsum_dim_sum_per_rank_tensor=self._cumsum_dim_sum_per_rank_tensor,
            group=self._pg,
            codecs=self._codecs,
        )

        pooled_embedding_awaitable = PooledEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
        )
        pooled_embedding_awaitable.callbacks.extend(self._callbacks)

        return pooled_embedding_awaitable

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks


class VariableBatchPooledEmbeddingsAllToAll(nn.Module):
    """
    Shards batches and collects keys of tensor with a `ProcessGroup` according to
    `dim_sum_per_rank`.

    Implementation utilizes `variable_batch_alltoall_pooled` operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        emb_dim_per_rank_per_feature (List[List[int]]): embedding dimensions per rank
            per feature.
        device (Optional[torch.device]): device on which buffers will be allocated.
        callbacks (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): callback
            functions.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Example::

        kjt_split = [1, 2]

        # the kjt_split informs the number of features owned by each rank, here t0 owns f0 and
        # t1 owns f1 and f2.

        emb_dim_per_rank_per_feature = [[2], [3, 3]]
        a2a = VariableBatchPooledEmbeddingsAllToAll(
            pg, emb_dim_per_rank_per_feature, device
        )

        t0 = torch.rand(6) # 2 * (2 + 1)
        t1 = torch.rand(24) # 3 * (1 + 2) + 3 * (3 + 2)

        # t0 and t1 are the flattened send buffers of pooled embedding outputs produced on the
        # ranks that own the features, computed as embedding_dim * (sum of variable batch sizes
        # for that feature across all source ranks), summed over the features owned by that destination rank.

        #        r0_batch_size   r1_batch_size
        #  f_0:              2               1
        -----------------------------------------
        #  f_1:              1               2
        #  f_2:              3               2

        # batch_size_per_rank_per_feature tensor is specified from the perspective of the sending rank
        # outer_index = destination rank, inner vector = features ownwed by the sending rank (in emb_dim_per_rank_per_feature order)

        r0_batch_size_per_rank_per_feature = [[2], [1]]
        r1_batch_size_per_rank_per_feature = [[1, 3], [2, 2]]

        #        r0 wants   r1 wants
        # f0:        2          1
        # f1:        1          2
        # f2:        3          2
        # which informs the per_feature_pre_a2a vectors

        r0_batch_size_per_feature_pre_a2a = [2, 1, 3]
        r1_batch_size_per_feature_pre_a2a = [1, 2, 2]

        # r0 should recieve f0: 2 (from r0), f1: 1 (from r1), f2: 3 (from r1)
        # r1 should recieve f0: 1 (from r0), f1: 2 (from r1), f2: 2 (from r1)

        rank0_output = a2a(
            t0, r0_batch_size_per_rank_per_feature, r0_batch_size_per_feature_pre_a2a
        ).wait()
        rank1_output = a2a(
            t1, r1_batch_size_per_rank_per_feature, r1_batch_size_per_feature_pre_a2a
        ).wait()

        # input splits:
        #   r0: [2*2, 1*2]
        #   r1: [1*3 + 3*3, 2*3 + 2*3]

        # output splits:
        #   r0: [2*2, 1*3 + 3*3]
        #   r1: [1*2, 2*3 + 2*3]

        print(rank0_output.size())
            # torch.Size([16])
            # 2*2 + 1*3 + 3*3
        print(rank1_output.size())
            # torch.Size([14])
            # 1*2 + 2*3 + 2*3
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        emb_dim_per_rank_per_feature: List[List[int]],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once("torchrec.distributed.vbe")
        self._pg = pg
        self._emb_dim_per_rank_per_feature = emb_dim_per_rank_per_feature
        self._callbacks: List[Callable[[torch.Tensor], torch.Tensor]] = []
        if callbacks is not None:
            self._callbacks = callbacks
        self._codecs = codecs

    def forward(
        self,
        local_embs: torch.Tensor,
        batch_size_per_rank_per_feature: List[List[int]],
        batch_size_per_feature_pre_a2a: List[int],
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs AlltoAll pooled operation with variable batch size per feature on a
        pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            batch_size_per_rank_per_feature (List[List[int]]): batch size per rank per
                feature, post a2a. Used to get the input splits.
            batch_size_per_feature_pre_a2a (List[int]): local batch size before
                scattering, used to get the output splits.
                Ordered by rank_0 feature, rank_1 feature, ...

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings.
        """

        tensor_awaitable = variable_batch_alltoall_pooled(
            a2a_pooled_embs_tensor=local_embs,
            batch_size_per_rank_per_feature=batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
            emb_dim_per_rank_per_feature=self._emb_dim_per_rank_per_feature,
            group=self._pg,
            codecs=self._codecs,
        )

        pooled_embedding_awaitable = PooledEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
        )
        pooled_embedding_awaitable.callbacks.extend(self._callbacks)

        return pooled_embedding_awaitable

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks


class EmbeddingsAllToOneReduce(nn.Module):
    """
    Merges the pooled embedding tensor on each device into single tensor.

    Args:
        device (torch.device): device on which buffer will be allocated.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._device = device
        self._world_size = world_size

    # This method can be used by an inference runtime to update the
    # device information for this module.
    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self._device = torch.device(device_str)

    def forward(
        self,
        tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Performs AlltoOne operation with Reduce on pooled embeddings tensors.

        Args:
            tensors (List[torch.Tensor]): list of embedding tensors.

        Returns:
            Awaitable[torch.Tensor]: awaitable of the reduced embeddings.
        """
        assert len(tensors) == self._world_size
        return torch.ops.fbgemm.sum_reduce_to_one(
            tensors,
            self._device,
        )


class EmbeddingsAllToOne(nn.Module):
    """
    Merges the pooled/sequence embedding tensor on each device into single tensor.

    Args:
        device (torch.device): device on which buffer will be allocated.
        world_size (int): number of devices in the topology.
        cat_dim (int): which dimension you would like to concatenate on.
            For pooled embedding it is 1; for sequence embedding it is 0.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        cat_dim: int,
    ) -> None:
        super().__init__()
        self.merge_pooled_embeddings = MergePooledEmbeddingsModule(device)
        self._world_size = world_size
        self._cat_dim = cat_dim
        self._device = device

    # This method can be used by an inference runtime to update the
    # device information for this module.
    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self._device = torch.device(device_str)
        self.merge_pooled_embeddings.set_device(device_str)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Performs AlltoOne operation on pooled/sequence embeddings tensors.

        Args:
            tensors (List[torch.Tensor]): list of embedding tensors.

        Returns:
            Awaitable[torch.Tensor]: awaitable of the merged embeddings.
        """
        assert len(tensors) <= self._world_size

        is_target_device_cpu: bool = self._device.type == "cpu"

        if self._world_size == 1:
            merge = tensors[0]
        else:
            merge = self.merge_pooled_embeddings(
                tensors,
                self._cat_dim,
            )

        return merge if not is_target_device_cpu else merge.to(self._device)


class SeqEmbeddingsAllToOne(nn.Module):
    """
    Merges the pooled/sequence embedding tensor on each device into single tensor.

    Args:
        device (torch.device): device on which buffer will be allocated
        world_size (int): number of devices in the topology.
        cat_dim (int): which dimension you like to concate on.
            For pooled embedding it is 1; for sequence embedding it is 0.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._device = device
        self._world_size = world_size

    # This method can be used by an inference runtime to update the
    # device information for this module.
    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self._device = torch.device(device_str)

    def forward(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Performs AlltoOne operation on pooled embeddings tensors.

        Args:
            tensors (List[torch.Tensor]): list of pooled embedding tensors.

        Returns:
            Awaitable[torch.Tensor]: awaitable of the merged pooled embeddings.
        """

        assert len(tensors) == self._world_size
        return torch.ops.fbgemm.all_to_one_device(
            tensors,
            self._device,
        )


class PooledEmbeddingsReduceScatter(nn.Module):
    """
    The module class that wraps reduce-scatter communication primitives for pooled
    embedding communication in row-wise and twrw sharding.

    For pooled embeddings, we have a local model-parallel output tensor with a layout of
    `[num_buckets x batch_size, dimension]`. We need to sum over `num_buckets` dimension
    across batches. We split the tensor along the first dimension into unequal chunks
    (tensor slices of different buckets) according to `input_splits` and reduce them
    into the output tensor and scatter the results for corresponding ranks.

    The class returns the async `Awaitable` handle for pooled embeddings tensor.
    The `reduce-scatter-v` operation is only available for NCCL backend.

    Args:
        pg (dist.ProcessGroup): the process group that the reduce-scatter communication
            happens within.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

     Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        input = torch.randn(2 * 2, 2)
        input_splits = [1,3]
        m = PooledEmbeddingsReduceScatter(pg)
        output = m(input, input_splits)
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._codecs = codecs

    def forward(
        self, local_embs: torch.Tensor, input_splits: Optional[List[int]] = None
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs reduce scatter operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of shape
                `[num_buckets * batch_size, dimension]`.
            input_splits (Optional[List[int]]): list of splits for `local_embs` dim 0.

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings of tensor of shape [batch_size, dimension].
        """

        # Dynamo can not trace through data dependent condition: len(set(input_splits)) > 1
        if is_torchdynamo_compiling():
            if input_splits is not None:
                tensor_awaitable = reduce_scatter_v_pooled(
                    local_embs, input_splits, self._pg, codecs=self._codecs
                )
            else:
                tensor_awaitable = reduce_scatter_base_pooled(
                    local_embs, self._pg, codecs=self._codecs
                )
        else:
            if input_splits and len(set(input_splits)) > 1:
                tensor_awaitable = reduce_scatter_v_pooled(
                    local_embs, input_splits, self._pg, codecs=self._codecs
                )
            else:
                tensor_awaitable = reduce_scatter_base_pooled(
                    local_embs, self._pg, codecs=self._codecs
                )
        return PooledEmbeddingsAwaitable(tensor_awaitable=tensor_awaitable)


class VariableBatchPooledEmbeddingsReduceScatter(nn.Module):
    """
    The module class that wraps reduce-scatter communication primitives for pooled
    embedding communication of variable batch in rw and twrw sharding.

    For variable batch per feature pooled embeddings, we have a local model-parallel
    output tensor with a 1d layout of the total sum of batch sizes per rank per feature
    multiplied by corresponding embedding dim `[batch_size_r0_f0 * emb_dim_f0 + ...)]`.
    We split the tensor into unequal chunks by rank according to
    `batch_size_per_rank_per_feature` and corresponding `embedding_dims` and reduce them
    into the output tensor and scatter the results for corresponding ranks.

    The class returns the async `Awaitable` handle for pooled embeddings tensor.
    The `reduce-scatter-v` operation is only available for NCCL backend.

    Args:
        pg (dist.ProcessGroup): the process group that the reduce-scatter communication
            happens within.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

     Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        input = torch.randn(52)
        batch_size_per_rank_per_feature = [[1, 3], [2, 2]]
        embedding_dims = [4, 8]
        m = VariableBatchPooledEmbeddingsReduceScatter(pg)
        output = m(input, batch_size_per_rank_per_feature)
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._codecs = codecs

    def forward(
        self,
        local_embs: torch.Tensor,
        batch_size_per_rank_per_feature: List[List[int]],
        embedding_dims: List[int],
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs reduce scatter operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of shape
                `[num_buckets * batch_size, dimension]`.
            batch_size_per_rank_per_feature (List[List[int]]): batch size per rank per
                feature used to determine input splits.
            embedding_dims (List[int]): embedding dimensions per feature used to
                determine input splits.

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings of tensor of shape [batch_size, dimension].
        """

        tensor_awaitable = reduce_scatter_v_per_feature_pooled(
            input=local_embs,
            batch_size_per_rank_per_feature=batch_size_per_rank_per_feature,
            embedding_dims=embedding_dims,
            group=self._pg,
            codecs=self._codecs,
        )
        return PooledEmbeddingsAwaitable(tensor_awaitable=tensor_awaitable)


class PooledEmbeddingsAllGather(nn.Module):
    """
    The module class that wraps the all-gather communication primitive for pooled
    embedding communication.

    Provided a local input tensor with a layout of `[batch_size, dimension]`, we want to
    gather input tensors from all ranks into a flattened output tensor.

    The class returns the async `Awaitable` handle for pooled embeddings tensor.
    The all-gather is only available for NCCL backend.

    Args:
        pg (dist.ProcessGroup): the process group that the all-gather communication
            happens within.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        input = torch.randn(2, 2)
        m = PooledEmbeddingsAllGather(pg)
        output = m(input)
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._codecs = codecs

    def forward(self, local_emb: torch.Tensor) -> PooledEmbeddingsAwaitable:
        """
        Performs reduce scatter operation on pooled embeddings tensor.

        Args:
            local_emb (torch.Tensor): tensor of shape
                `[num_buckets x batch_size, dimension]`.

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings of tensor of shape [batch_size, dimension].
        """

        tensor_awaitable = all_gather_base_pooled(
            local_emb, self._pg, codecs=self._codecs
        )
        return PooledEmbeddingsAwaitable(tensor_awaitable=tensor_awaitable)


class SequenceEmbeddingsAwaitable(Awaitable[torch.Tensor]):
    """
    Awaitable for sequence embeddings after collective operation.

    Args:
        tensor_awaitable (Awaitable[torch.Tensor]): awaitable of concatenated tensors
            from all the processes in the group after collective.
        unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute order of
            KJT bucketize (for row-wise sharding only).
        embedding_dim (int): embedding dimension.
    """

    def __init__(
        self,
        tensor_awaitable: Awaitable[torch.Tensor],
        unbucketize_permute_tensor: Optional[torch.Tensor],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._tensor_awaitable = tensor_awaitable

        if unbucketize_permute_tensor is not None:
            self.callbacks.append(
                lambda ret: torch.index_select(
                    ret.view(-1, embedding_dim),
                    0,
                    unbucketize_permute_tensor,
                )
            )

    def _wait_impl(self) -> torch.Tensor:
        """
        Syncs sequence embeddings after collective operation.

        Returns:
            torch.Tensor: synced sequence embeddings.
        """

        ret = self._tensor_awaitable.wait()
        return ret


class SequenceEmbeddingsAllToAll(nn.Module):
    """
    Redistributes sequence embedding to a `ProcessGroup` according to splits.

    Args:
        pg (dist.ProcessGroup): the process group that the AlltoAll communication
            happens within.
        features_per_rank (List[int]): list of number of features per rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
        codecs (Optional[QuantizedCommCodecs]): quantized communication codecs.

    Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        features_per_rank = [4, 4]
        m = SequenceEmbeddingsAllToAll(pg, features_per_rank)
        local_embs = torch.rand((6, 2))
        sharding_ctx: SequenceShardingContext
        output = m(
            local_embs=local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            unbucketize_permute_tensor=None,
        )
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._local_split: int = features_per_rank[self._pg.rank()]
        self._num_splits: int = self._pg.size()

        forward_recat = []
        for j in range(self._num_splits):
            for i in range(self._local_split):
                forward_recat.append(j + i * self._num_splits)
        self.register_buffer(
            "_forward_recat_tensor",
            torch.tensor(forward_recat, device=device, dtype=torch.int),
        )
        backward_recat = []
        for i in range(self._local_split):
            for j in range(self._num_splits):
                backward_recat.append(i + j * self._local_split)
        self.register_buffer(
            "_backward_recat_tensor",
            torch.tensor(backward_recat, device=device, dtype=torch.int),
        )
        self._codecs = codecs

    def forward(
        self,
        local_embs: torch.Tensor,
        lengths: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        unbucketize_permute_tensor: Optional[torch.Tensor] = None,
        batch_size_per_rank: Optional[List[int]] = None,
        sparse_features_recat: Optional[torch.Tensor] = None,
    ) -> SequenceEmbeddingsAwaitable:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): input embeddings tensor.
            lengths (torch.Tensor): lengths of sparse features after AlltoAll.
            input_splits (List[int]): input splits of AlltoAll.
            output_splits (List[int]): output splits of AlltoAll.
            unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute
                order of the KJT bucketize (for row-wise sharding only).
            batch_size_per_rank: (Optional[List[int]]): batch size per rank.
            sparse_features_recat (Optional[torch.Tensor]): recat tensor used for sparse
                feature input dist. Must be provided if using variable batch size.

        Returns:
            SequenceEmbeddingsAwaitable: awaitable of sequence embeddings.
        """

        variable_batch_size = (
            batch_size_per_rank is not None and len(set(batch_size_per_rank)) > 1
        )

        if sparse_features_recat is not None:
            forward_recat_tensor = torch.ops.fbgemm.invert_permute(
                sparse_features_recat
            )
            backward_recat_tensor = sparse_features_recat
        else:
            forward_recat_tensor = self._forward_recat_tensor
            backward_recat_tensor = self._backward_recat_tensor

        tensor_awaitable = alltoall_sequence(
            a2a_sequence_embs_tensor=local_embs,
            forward_recat_tensor=forward_recat_tensor,
            backward_recat_tensor=backward_recat_tensor,
            lengths_after_sparse_data_all2all=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            variable_batch_size=variable_batch_size,
            group=self._pg,
            codecs=self._codecs,
        )
        return SequenceEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
            unbucketize_permute_tensor=unbucketize_permute_tensor,
            embedding_dim=local_embs.shape[1],
        )


class JaggedTensorAllToAll(Awaitable[JaggedTensor]):
    """
    Redistributes `JaggedTensor` to a `ProcessGroup` along the batch dimension according
    to the number of items to send and receive. The number of items to send
    must be known ahead of time on each rank. This is currently used for sharded
    KeyedJaggedTensorPool, after distributing the number of IDs to lookup or update on
    each rank.

    Implementation utilizes AlltoAll collective as part of torch.distributed.

    Args:
        jt (JaggedTensor): JaggedTensor to distribute.
        num_items_to_send (int): Number of items to send.
        num_items_to_receive (int): Number of items to receive from all other ranks.
            This must be known ahead of time on each rank, usually via another AlltoAll.
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
    """

    def __init__(
        self,
        jt: JaggedTensor,
        num_items_to_send: torch.Tensor,
        num_items_to_receive: torch.Tensor,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()

        self._dist_lengths: torch.Tensor = torch.empty(
            sum(num_items_to_receive),
            device=jt.lengths().device,
            dtype=jt.lengths().dtype,
        )

        dist.all_to_all_single(
            self._dist_lengths,
            jt.lengths(),
            output_split_sizes=num_items_to_receive.tolist(),
            input_split_sizes=num_items_to_send.tolist(),
            group=pg,
            async_op=False,
        )

        # below will calculate chunks sums e.g.
        # num_batches_to_receive = [2,2]
        # lengths = [2,3,1,1]
        # output_splits = [5,2]
        dist_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            num_items_to_receive
        )
        value_output_splits = torch.ops.fbgemm.segment_sum_csr(
            1,
            dist_id_offsets,
            self._dist_lengths,
        ).tolist()

        self._dist_values: torch.Tensor = torch.empty(
            sum(value_output_splits),
            dtype=jt.values().dtype,
            device=jt.values().device,
        )

        #  same as above, calculate chunk sums
        id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_items_to_send)
        value_input_splits = torch.ops.fbgemm.segment_sum_csr(
            1,
            id_offsets,
            jt.lengths(),
        ).tolist()

        self._dist_values_req: dist.Work = dist.all_to_all_single(
            self._dist_values,
            jt.values(),
            output_split_sizes=value_output_splits,
            input_split_sizes=value_input_splits,
            group=pg,
            async_op=True,
        )

        self._dist_weights: Optional[torch.Tensor] = None
        self._dist_weights_req: Optional[dist.Work] = None
        if jt.weights_or_none() is not None:
            self._dist_weights = torch.empty(
                sum(value_output_splits),
                dtype=jt.weights().dtype,
                device=jt.weights().device,
            )

            self._dist_weights_req = dist.all_to_all_single(
                self._dist_weights,
                jt.weights(),
                output_split_sizes=value_output_splits,
                input_split_sizes=value_input_splits,
                group=pg,
                async_op=True,
            )

    def _wait_impl(self) -> JaggedTensor:
        self._dist_values_req.wait()
        if self._dist_weights_req is not None:
            self._dist_weights_req.wait()

        return JaggedTensor(
            values=self._dist_values,
            lengths=self._dist_lengths,
            weights=self._dist_weights,
        )


class TensorAllToAllValuesAwaitable(Awaitable[torch.Tensor]):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: torch.Tensor,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._device: torch.device = device
        self._input = input

        self._dist_values: torch.Tensor
        if self._workers == 1:
            self._dist_values = input
            return
        else:
            if input.dim() > 1:
                self._dist_values = torch.empty(
                    (sum(output_splits), input.shape[1]),
                    device=self._device,
                    dtype=input.dtype,
                )
            else:
                self._dist_values = torch.empty(
                    sum(output_splits), device=self._device, dtype=input.dtype
                )

        with record_function("## all2all_data:ids ##"):
            self._values_awaitable: dist.Work = dist.all_to_all_single(
                output=self._dist_values,
                input=input,
                output_split_sizes=output_splits.tolist(),
                input_split_sizes=input_splits.tolist(),
                group=pg,
                async_op=True,
            )

    def _wait_impl(self) -> torch.Tensor:
        if self._workers > 1:
            self._values_awaitable.wait()
        return self._dist_values


class TensorAllToAllSplitsAwaitable(Awaitable[TensorAllToAllValuesAwaitable]):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: torch.Tensor,
        splits: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._pg: dist.ProcessGroup = pg
        self._device: torch.device = device
        self._input = input
        self._input_splits = splits

        self._output_splits: torch.Tensor
        if self._workers == 1:
            self._output_splits = splits
            return
        else:
            self._output_splits = torch.empty(
                [self._workers],
                device=device,
                dtype=torch.int,
            )

        with record_function("## all2all_data:ids splits ##"):
            self._num_ids_awaitable: dist.Work = dist.all_to_all_single(
                output=self._output_splits,
                input=splits,
                group=pg,
                async_op=True,
            )

    def _wait_impl(self) -> TensorAllToAllValuesAwaitable:
        if self._workers > 1:
            self._num_ids_awaitable.wait()

        return TensorAllToAllValuesAwaitable(
            pg=self._pg,
            input=self._input,
            input_splits=self._input_splits,
            output_splits=self._output_splits,
            device=self._device,
        )


class TensorValuesAllToAll(nn.Module):
    """
    Redistributes torch.Tensor to a `ProcessGroup` according to input and output splits.

    Implementation utilizes AlltoAll collective as part of torch.distributed.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.

    Example::
        tensor_vals_A2A = TensorValuesAllToAll(pg)
        input_splits = torch.Tensor([1,2]) on rank0 and torch.Tensor([1,1]) on rank1
        output_splits = torch.Tensor([1,1]) on rank0 and torch.Tensor([2,1]) on rank1
        awaitable = tensor_vals_A2A(rank0_input, input_splits, output_splits)

        where:
        rank0_input is 3 x 3 torch.Tensor holding
        [
            [V1, V2, V3],
            [V4, V5, V6],
            [V7, V8, V9],
        ]

        rank1_input is 2 x 3 torch.Tensor holding
        [
            [V10, V11, V12],
            [V13, V14, V15],
        ]

        rank0_output = awaitable.wait()

        # where:
        # rank0_output is torch.Tensor holding
        [
            [V1, V2, V3],
            [V10, V11, V12],
        ]

        # rank1_output is torch.Tensor holding
        [
            [V1, V2, V3],
            [V4, V5, V6],
            [V7, V8, V9],
        ]
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self._pg: dist.ProcessGroup = pg

    def forward(
        self,
        input: torch.Tensor,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
    ) -> TensorAllToAllValuesAwaitable:
        """
        Sends tensor to relevant `ProcessGroup` ranks.

        Args:
            input (torch.Tensor): `torch.Tensor` of values to distribute.
            input_splits (torch.Tensor): tensor containing number of rows
                to be sent to each rank.  len(input_splits) must equal self._pg.size()
            output_splits (torch.Tensor): tensor containing number of rows
            to be received from each rank. len(output_splits) must equal self._pg.size()

        Returns: `TensorAllToAllValuesAwaitable`
        """
        with torch.no_grad():
            return TensorAllToAllValuesAwaitable(
                pg=self._pg,
                input=input,
                input_splits=input_splits,
                output_splits=output_splits,
                device=input.device,
            )


class TensorAllToAll(nn.Module):
    """
    Redistributes a 1D tensor to a `ProcessGroup` according to splits.

    Implementation utilizes AlltoAll collective as part of torch.distributed.

    The first collective call in `TensorAllToAllSplitsAwaitable` will transmit
    splits to allocate correct space for the tensor values. The following collective
    calls in `TensorAllToAllValuesAwaitable` will transmit the actual
    tensor values asynchronously.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.

    Example::
        tensor_A2A = TensorAllToAll(pg)
        splits = torch.Tensor([1,1]) on rank0 and rank1
        awaitable = tensor_A2A(rank0_input, splits)

        where:
        rank0_input is torch.Tensor holding
        [
            [V1, V2, V3],
            [V4, V5, V6],
        ]

        rank1_input is torch.Tensor holding
        [
            [V7, V8, V9],
            [V10, V11, V12],
        ]

        rank0_output = awaitable.wait().wait()

        # where:
        rank0_input is torch.Tensor holding
        [
            [V1, V2, V3],
            [V7, V8, V9],
        ]

        rank1_input is torch.Tensor holding
        [
            [V4, V5, V6],
            [V10, V11, V12],
        ]
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self._pg: dist.ProcessGroup = pg

    def forward(
        self,
        input: torch.Tensor,
        splits: torch.Tensor,
    ) -> TensorAllToAllSplitsAwaitable:
        """
        Sends tensor to relevant `ProcessGroup` ranks.

        The first wait will get the splits for the provided tensors and issue
        tensors AlltoAll. The second wait will get the tensors.

        Args:
            input (torch.Tensor): `torch.Tensor` of values to distribute.

        Returns:
            Awaitable[TensorAllToAllValuesAwaitable]: awaitable of a `TensorAllToAllValuesAwaitable`.
        """
        with torch.no_grad():
            temp = TensorAllToAllSplitsAwaitable(
                pg=self._pg,
                input=input,
                splits=splits,
                device=input.device,
            )
            return temp
