#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import copy
import logging
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
from torch import nn
from torch.distributed._tensor import DTensor
from torchrec.distributed.embedding_types import (
    DTensorMetadata,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torchrec.distributed.utils import none_throws
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


class BaseEmbedding(abc.ABC, nn.Module):
    """
    Abstract base class for grouped `nn.Embedding` and `nn.EmbeddingBag`
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor):
        Returns:
            torch.Tensor: sparse gradient parameter names.
        """
        pass

    @property
    @abc.abstractmethod
    def config(self) -> GroupedEmbeddingConfig:
        pass


def create_virtual_table_local_metadata(
    local_metadata: ShardMetadata,
    param: Union[torch.Tensor, PartiallyMaterializedTensor],
    my_rank: int,
    offset: Optional[int] = None,
    weight_count_per_rank: Optional[List[int]] = None,
) -> None:
    if offset is None:
        offset = (
            my_rank
            if weight_count_per_rank is None
            else sum(weight_count_per_rank[:my_rank])
        )
    local_metadata.shard_sizes = list(param.size())  # pyre-ignore[6]
    local_metadata.shard_offsets = [
        offset if dim == 0 else 0 for dim in range(len(param.size()))  # pyre-ignore[6]
    ]


def create_virtual_table_global_metadata(
    metadata: ShardedTensorMetadata,
    my_rank: int,
    param: Union[torch.Tensor, PartiallyMaterializedTensor],
    weight_count_per_rank: Optional[List[int]],
    use_param_size_as_rows: bool,
) -> None:
    # update tensor properties from local tensor properties, this should be universal for all ranks
    metadata.tensor_properties.dtype = param.dtype
    metadata.tensor_properties.requires_grad = param.requires_grad

    offset = 0
    # Manually craft metadata according to the weight_count_per_rank to set correct
    # metadata for cp loading for non-PMT case, where recalc won't be triggered.
    # Otherwise it will only set correct size on current rank and
    # virtual PMT will trigger recalc for the correct global size/offset.
    # NOTE this currently only works for row-wise sharding
    for rank, shard_metadata in enumerate(metadata.shards_metadata):
        if use_param_size_as_rows:  # respect the param size and treat it as rows
            curr_rank_rows = param.size()[0]  # pyre-ignore[16]
        else:
            curr_rank_rows = (
                weight_count_per_rank[rank] if weight_count_per_rank is not None else 1
            )
        if rank < my_rank:
            shard_metadata.shard_sizes = [
                curr_rank_rows if dim == 0 else param.size(dim)
                for dim in range(len(param.size()))  # pyre-ignore[6]
            ]
            shard_metadata.shard_offsets = [
                offset if dim == 0 else 0
                for dim in range(len(param.size()))  # pyre-ignore[6]
            ]
        elif rank == my_rank:
            curr_rank_rows = param.size()[0]  # pyre-ignore[16]
            create_virtual_table_local_metadata(shard_metadata, param, my_rank, offset)
        else:
            shard_metadata.shard_sizes = [
                curr_rank_rows if dim == 0 else param.size(dim)
                for dim in range(len(param.size()))  # pyre-ignore[6]
            ]
            shard_metadata.shard_offsets = [
                offset if dim == 0 else 0
                for dim in range(len(param.size()))  # pyre-ignore[6]
            ]
        offset += curr_rank_rows

    metadata.size = torch.Size(
        [  # pyre-ignore[6]
            offset if dim == 0 else param.size(dim)
            for dim in range(len(param.size()))  # pyre-ignore[6]
        ]
    )


def create_virtual_sharded_tensors(
    embedding_tables: List[ShardedEmbeddingTable],
    params: Union[List[torch.Tensor], List[PartiallyMaterializedTensor]],
    pg: Optional[dist.ProcessGroup] = None,
    prefix: str = "",
    table_name_to_weight_count_per_rank: Optional[Dict[str, List[int]]] = None,
    use_param_size_as_rows: bool = False,
) -> List[ShardedTensor]:
    """
    Create virtual sharded tensors for the given embedding tables and parameters.
    This is used to create sharded tensor for virtual table, where the table rows changes dynamically
    everytime we call ec/ebc state_dict, we will create a ShardedTensor and recalculate the metadata by
    all_gather, however since we don't want to put all_gather implicitly inside state_dict, we need to
    create a ShardedTensor temporarily and return it to the caller side to do the metadata recalculation work

    Here for the local shard tensor, we just fake zero sizes on the remote shards' metadata
    for this shard's ShardedTensor
    """
    key_to_local_shards: Dict[str, List[Shard]] = defaultdict(list)
    key_to_global_metadata: Dict[str, ShardedTensorMetadata] = {}

    def get_key_from_embedding_table(embedding_table: ShardedEmbeddingTable) -> str:
        return prefix + f"{embedding_table.name}"

    def get_weight_count_per_rank(table_name: str) -> Optional[List[int]]:
        return (
            table_name_to_weight_count_per_rank.get(table_name, None)
            if table_name_to_weight_count_per_rank
            and table_name in table_name_to_weight_count_per_rank.keys()
            else None
        )

    my_rank = dist.get_rank()
    for embedding_table, param in zip(embedding_tables, params):
        key = get_key_from_embedding_table(embedding_table)
        assert embedding_table.use_virtual_table

        assert embedding_table.global_metadata is not None
        global_metadata = copy.deepcopy(embedding_table.global_metadata)
        weight_count_per_rank = get_weight_count_per_rank(embedding_table.name)
        create_virtual_table_global_metadata(
            global_metadata,
            my_rank,
            param,
            weight_count_per_rank,
            use_param_size_as_rows,
        )
        key_to_global_metadata[key] = global_metadata

        local_metadata = copy.deepcopy(global_metadata.shards_metadata[my_rank])
        local_metadata.placement = none_throws(
            none_throws(
                embedding_table.local_metadata,
                f"local_metadata is None for emb_table: {embedding_table.name}",
            ).placement,
            f"placement is None for local_metadata of emb table: {embedding_table.name}",
        )

        key_to_local_shards[key].append(Shard(param, local_metadata))  # pyre-ignore

    result: List[ShardedTensor] = []
    if pg is not None:
        for key in key_to_local_shards:
            global_metadata = key_to_global_metadata[key]
            result.append(
                ShardedTensor._init_from_local_shards_and_global_metadata(
                    local_shards=key_to_local_shards[key],
                    sharded_tensor_metadata=global_metadata,
                    process_group=pg,
                )
            )
    return result


def get_state_dict(
    embedding_tables: List[ShardedEmbeddingTable],
    params: Union[
        nn.ModuleList,
        List[Union[nn.Module, torch.Tensor]],
        List[torch.Tensor],
        List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
        List[PartiallyMaterializedTensor],
    ],
    pg: Optional[dist.ProcessGroup] = None,
    destination: Optional[Dict[str, Any]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    if destination is None:
        destination = OrderedDict()
        # pyre-ignore [16]
        destination._metadata = OrderedDict()
    """
    It is possible for there to be multiple shards from a table on a single rank.
    We accumulate them in key_to_local_shards. Repeat shards should have identical
    global ShardedTensorMetadata.
    """
    key_to_local_shards: Dict[str, List[Shard]] = defaultdict(list)
    key_to_global_metadata: Dict[str, ShardedTensorMetadata] = {}
    key_to_dtensor_metadata: Dict[str, DTensorMetadata] = {}
    # pyre-ignore[33]
    key_to_local_tensor_shards: Dict[str, List[Any]] = defaultdict(list)

    # validate on the function input for kv zch cases
    use_virtual_size = None
    for emb_table in embedding_tables:
        if use_virtual_size is None:
            use_virtual_size = emb_table.use_virtual_table
        assert use_virtual_size == emb_table.use_virtual_table

    def get_key_from_embedding_table(embedding_table: ShardedEmbeddingTable) -> str:
        return prefix + f"{embedding_table.name}.weight"

    for embedding_table, param in zip(embedding_tables, params):
        weights_key = get_key_from_embedding_table(embedding_table)
        is_quant = embedding_table.compute_kernel in [
            EmbeddingComputeKernel.QUANT,
            EmbeddingComputeKernel.QUANT_UVM,
            EmbeddingComputeKernel.QUANT_UVM_CACHING,
        ]
        qscale = None
        qbias = None
        if is_quant:
            # For QUANT* param is Tuple[torch.Tensor, Optional[torch.Tensor]] where first argument is
            # the weight table, the second is optional quantization extra information, depending on
            # quantization type. e.g. for fbgemm rowwise quantization this is scale and shift for each row.
            assert isinstance(param, tuple)
            qscale = param[1]
            qbias = param[2]
            param = param[0]

        if not embedding_table.use_virtual_table:
            assert embedding_table.local_rows == param.size(  # pyre-ignore[16]
                0
            ), f"{embedding_table.local_rows=}, {param.size(0)=}, {param.shape=}"  # pyre-ignore[16]

        if qscale is not None:
            assert embedding_table.local_cols == param.size(1)  # pyre-ignore[16]

        if embedding_table.dtensor_metadata is not None and pg is not None:
            # DTensor path
            key_to_dtensor_metadata[weights_key] = embedding_table.dtensor_metadata
            key_to_local_tensor_shards[weights_key].append(
                [
                    param,
                    embedding_table.local_metadata.shard_offsets,  # pyre-ignore[16]
                ]
            )
        elif embedding_table.global_metadata is not None and pg is not None:
            # set additional field of sharded tensor based on local tensor properties
            embedding_table.global_metadata.tensor_properties.dtype = (
                param.dtype  # pyre-ignore[16]
            )
            embedding_table.global_metadata.tensor_properties.requires_grad = (
                param.requires_grad  # pyre-ignore[16]
            )
            local_metadata = embedding_table.local_metadata
            glb_metadata = embedding_table.global_metadata
            if use_virtual_size:
                assert local_metadata is not None and glb_metadata is not None
                local_metadata = copy.deepcopy(embedding_table.local_metadata)
                local_metadata.shard_sizes = list(param.size())  # pyre-ignore
                local_metadata.shard_offsets = [0, 0]

                glb_metadata = copy.deepcopy(embedding_table.global_metadata)
                glb_metadata.size = param.size()  # pyre-ignore
                my_rank = dist.get_rank()
                for rank, shards_metadata in enumerate(
                    # pyre-ignore
                    glb_metadata.shards_metadata
                ):
                    if rank < my_rank:
                        shards_metadata.shard_offsets = [0, 0]
                        shards_metadata.shard_sizes = [0, 0]
                    elif rank == my_rank:
                        shards_metadata.shard_offsets = local_metadata.shard_offsets
                        shards_metadata.shard_sizes = local_metadata.shard_sizes
                    else:
                        shards_metadata.shard_offsets = [
                            local_metadata.shard_sizes[0],
                            0,
                        ]
                        shards_metadata.shard_sizes = [0, 0]

            key_to_global_metadata[weights_key] = glb_metadata  # pyre-ignore

            # for kv zch cases, we use virtual space, the logic will be the same as non-kv zch cases
            key_to_local_shards[weights_key].append(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Module, Tensor]`.
                # pyre-fixme[6]: For 2nd argument expected `ShardMetadata` but got
                #  `Optional[ShardMetadata]`.
                Shard(param, local_metadata)
            )

        else:
            destination[weights_key] = param
            if qscale is not None:
                destination[f"{weights_key}_qscale"] = qscale
            if qbias is not None:
                destination[f"{weights_key}_qbias"] = qbias

    if pg is not None:
        # Populate the remaining destinations that have a global metadata
        for key in key_to_local_shards:
            global_metadata = key_to_global_metadata[key]
            destination[key] = (
                ShardedTensor._init_from_local_shards_and_global_metadata(
                    local_shards=key_to_local_shards[key],
                    sharded_tensor_metadata=global_metadata,
                    process_group=pg,
                )
            )
        # DTensor path
        for key in key_to_local_tensor_shards:
            dtensor_metadata = key_to_dtensor_metadata[key]
            destination[key] = DTensor.from_local(
                local_tensor=LocalShardsWrapper(
                    local_shards=[
                        tensor_shards[0]
                        for tensor_shards in key_to_local_tensor_shards[key]
                    ],
                    local_offsets=[
                        tensor_shards[1]
                        for tensor_shards in key_to_local_tensor_shards[key]
                    ],
                ),
                device_mesh=dtensor_metadata.mesh,
                placements=dtensor_metadata.placements,
                shape=torch.Size(dtensor_metadata.size),  # pyre-ignore[6]
                stride=dtensor_metadata.stride,
                run_check=False,
            )
    return destination
