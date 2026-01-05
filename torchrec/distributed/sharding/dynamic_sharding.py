#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ModuleShardingPlan,
    ParameterSharding,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)


class CommStrategy(Enum):
    STATE_INTER_NODE_COMM = "state_inter_node_comm"
    STATE_INTRA_NODE_COMM = "state_intra_node_comm"
    OPT_INTER_NODE_COMM = "opt_inter_node_comm"
    OPT_INTRA_NODE_COMM = "opt_intra_node_comm"


OrderedShardNamesWithSizes = List[Tuple[str, List[int]]]
"""
A type alias to represent an ordered shard name and the corresponding shard_size
in dim 0 & 1 that were sent to the current rank.
This is a flattened and pruned nested list, which orders the shards names and
sizes in the following priority:
1. Rank order
2. Table order
3. Shard order

<table_x, shard_y> in below examples represent the 2d tensor correlated to a
certain table `x`, allocated to rank `z`. The `y` here denotes the order of shards
in the module attributes such as state_dict, sharding_plan, etc..

`z` != `y` numerically, but the order of shards is based on the order of ranks allocated

Example 1 NOTE: the ordering by rank:
Rank 0 sends table_0, shard_0 to Rank 1.
Rank 2 sends table_1, shard_0 to Rank 1.
Rank 2 sends table_1, shard_1 to Rank 0
Rank 3 sends table_0, shard_1 to Rank 0

NOTE: table_1 comes first due to its source rank being 'first'
On Rank 0:output_tensor = [
    <table_1, shard_0>, # from rank 2
    <table_0, shard_1>  # from rank 3
]

On Rank 1: output_tensor = [
    <table_0, shard_0>, # from rank 0
    <table_1, shard_0>  # from rank 2
]

Example 2: NOTE: ordered by table when ranks are the same
Rank 0 sends table_1 to Rank 1
Rank 0 sends table_2 to Rank 1

output_tensor = [
    <table_0, shard_y>,
    <table_1, shard_y>
]

Example 3: NOTE: ordered by shard if table and rank are the same
Rank 0 sends table_1, shard_0 to Rank 1
Rank 0 sends table_1, shard_1 to Rank 1

Rank 1: output_tensor = [
    <table_0, shard_0>,
    <table_1, shard_1>
]
"""

ShardToRankMapping = Dict[int, List[Tuple[int, int, int, List[int]]]]
"""
ShardToRankMapping is a type alias for a dictionary that maps source ranks to a list of shard metadata tuples.
Each key in the dictionary is an integer representing a source rank.
The value associated with each key is a list of tuples, where each tuple contains metadata about a shard.
The structure of each tuple is as follows:
1. First Element (int): Represents the position of the shard metadata in the original module sharding plan.
   This indicates the order or index of the shard within the sharding plan.
2. Second Element (int): Represents the position of the shard tensor in the original state dictionary.
   This helps in identifying the specific shard tensor within the state dictionary.
3. Third Element (int): Represents the destination rank.
   This indicates the rank to which the shard is being redistributed.
4. Fourth Element (List[int]): A list representing the shard size.
   This provides the dimensions or size of the shard being handled.

   E.g [0,(1,0,2,[10,4])] means that the source rank 0 has a shard with size (10,4) and it's  in the first position of the
    source modulesharding plan. and the data can be accessed through rank=1, first local tensor and this is being sent to rank 2,
    the new shard size is (10,4)
"""
MAX_TAG_HASH_VALUE: int = 0x7FFFFFFF


@dataclass
class CommP2PMetadata:
    shard_name: str
    offsets: List[int]
    comm_op: dist.P2POp
    requires_copy: bool
    is_optimizer_state: bool


def _generate_shard_allocation_metadata(
    source_params: ParameterSharding,
    destination_params: ParameterSharding,
    rank: int,
    world_size: int,
) -> ShardToRankMapping:
    """
    Generates metadata mapping for shard allocation between source and destination sharding parameters.

    This function computes how shards from the source sharding plan should be allocated to the destination
    sharding plan, producing a mapping from source ranks to lists of tuples describing the shard transfer.
    Each tuple contains:
        - The index of the shard in the source sharding plan.
        - The local shard index for the source rank.
        - The destination rank to which the shard is being sent.
        - The offsets (start and end) in the shard's dimension for the transfer.

    Args:
        source_params (ParameterSharding): The sharding parameters of the source (current) sharding plan.
        destination_params (ParameterSharding): The sharding parameters of the destination (new) sharding plan.
        rank (int): The current process rank.
        world_size (int): The total number of ranks in the distributed setup.

    Returns:
        ShardToRankMapping: A dictionary mapping source ranks to lists of shard transfer metadata tuples.
    """
    shard_to_rank_mapping: ShardToRankMapping = {}
    src_rank_index = 0
    dst_rank_index = 0
    curr_source_offset = 0
    curr_dst_offset = 0
    local_shard_indices = [0 for _ in range(world_size)]

    assert source_params.ranks is not None
    assert destination_params.ranks is not None
    assert source_params.sharding_spec is not None
    assert destination_params.sharding_spec is not None

    # Initialize dictionary keys for all source ranks
    # Pyre-ignore
    for rank in source_params.ranks:
        shard_to_rank_mapping[rank] = []

    # Pyre-ignore
    while src_rank_index < len(source_params.ranks) and dst_rank_index < len(
        destination_params.ranks  # Pyre-ignore
    ):
        # Pyre-ignore
        src_shard_size = source_params.sharding_spec.shards[src_rank_index].shard_sizes
        dst_shard_size = destination_params.sharding_spec.shards[
            dst_rank_index
        ].shard_sizes

        shard_dim = min(
            src_shard_size[1] - curr_source_offset, dst_shard_size[1] - curr_dst_offset
        )

        next_source_offset = curr_source_offset + shard_dim
        next_dst_offset = curr_dst_offset + shard_dim

        # Greedy way of allocating shards to ranks
        # Pyre-ignore
        shard_to_rank_mapping[source_params.ranks[src_rank_index]].append(
            (
                src_rank_index,
                local_shard_indices[source_params.ranks[src_rank_index]],
                destination_params.ranks[dst_rank_index],
                [curr_source_offset, next_source_offset],
            )
        )
        curr_source_offset = next_source_offset
        curr_dst_offset = next_dst_offset

        if next_source_offset >= src_shard_size[1]:
            local_shard_indices[source_params.ranks[src_rank_index]] += 1
            src_rank_index += 1
            curr_source_offset = 0

        if next_dst_offset >= dst_shard_size[1]:
            dst_rank_index += 1
            curr_dst_offset = 0
    return shard_to_rank_mapping


def _is_intra_comm(src_rank: int, dst_rank: int, local_world_size: int) -> bool:
    return (src_rank // local_world_size) == (dst_rank // local_world_size)


def _generate_tag(
    src_rank: int,
    shard_id: int,
    local_shard_id: int,
    dst_rank: int,
    shard_name: str,
    is_optimizer_state: bool = False,
) -> int:
    """
    Generates a unique tag for point-to-point (P2P) communication operations.

    This function creates a unique tag for each P2P communication operation based on the provided parameters.
    The tag is generated using a hash-based approach to ensure uniqueness within a 32-bit signed integer range.
    """
    return (
        hash(
            (
                src_rank,
                shard_id,
                local_shard_id,
                dst_rank,
                shard_name,
                is_optimizer_state,
            )
        )
        & MAX_TAG_HASH_VALUE
    )


def _prepare_shard_distribution_comm_ops(
    rank: int,
    shard_name: str,
    shard_to_rank_mapping: ShardToRankMapping,
    current_state_dict: Dict[str, ShardedTensor],
    new_state_dict: Dict[str, ShardedTensor],
    comm_ops: Dict[CommStrategy, List[CommP2PMetadata]],
    extend_shard_name: Callable[[str], str] = lambda x: x,
    current_opt_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
    new_opt_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
    has_optimizer: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Prepares the point-to-point (P2P) communication operations required to redistribute sharded tensors
    (and optionally optimizer state) between source and destination ranks during resharding.

    This function processes the shard-to-rank mapping and, for each shard that needs to be sent or received
    by the current rank, creates the appropriate send/receive operations (isend/irecv) for both model state
    and optimizer state. It organizes these operations into the provided comm_ops dictionary, grouped by
    communication strategy (inter-node/intra-node, model/optimizer).

    Args:
        rank (int): The current process rank.
        shard_name (str): The name of the shard/table being processed.
        shard_to_rank_mapping (ShardToRankMapping): Mapping from source ranks to lists of shard transfer metadata.
        current_state_dict (Dict[str, ShardedTensor]): State dict containing current sharded tensors.
        new_state_dict (Dict[str, ShardedTensor]): State dict for the new sharded tensors after redistribution.
        comm_ops (Dict[CommStrategy, List[CommP2PMetadata]]): Dictionary to be populated with communication operations.
        extend_shard_name (Callable[[str], str], optional): Function to extend shard names to full names in state dicts.
        current_opt_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]], optional): Current optimizer state dict.
        new_opt_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]], optional): New optimizer state dict to update.
        has_optimizer (bool, optional): Whether optimizer state communication is required.
        process_group (Optional[dist.ProcessGroup], optional): The process group for distributed communication.

    Returns:
        None
    """
    has_local_optimizer = (
        current_opt_state is not None
    )  # local optimizer represents wether sharding table has optimizer state locally.

    receiving_shards_metadata: List[Tuple[int, int, int, int, List[int]]] = []
    extended_shard_name: str = extend_shard_name(shard_name)

    # Process each shard mapping from source to destination
    for src_rank, dsts in shard_to_rank_mapping.items():

        for shard_id, local_shard_id, dst_rank, split_offsets in dsts:

            # Get shard metadata

            sharded_t = current_state_dict[extended_shard_name]

            shard_metadata = sharded_t.metadata().shards_metadata[shard_id]
            shard_size = shard_metadata.shard_sizes

            shard_offsets = shard_metadata.shard_offsets

            assert split_offsets[0] >= 0
            assert split_offsets[1] <= shard_size[1]
            # Update the shard size with new size
            shard_size = [shard_size[0], split_offsets[1] - split_offsets[0], shard_id]

            intra = _is_intra_comm(src_rank, dst_rank, 8)
            # Create unique tag for P2P communication (32-bit limit for PyTorch)

            tag = _generate_tag(
                src_rank, shard_id, local_shard_id, dst_rank, shard_name, False
            )
            tag_opt = _generate_tag(
                src_rank, shard_id, local_shard_id, dst_rank, shard_name, True
            )
            # Process data being sent from current rank
            if src_rank == rank:
                if has_local_optimizer:
                    momentun_name = _tmp_momentum_extender(shard_name)
                    # Pyre-ignore
                    local_optimizer_shards = current_opt_state["state"][
                        extended_shard_name
                    ][momentun_name].local_shards()

                    local_optimizer_tensor = local_optimizer_shards[
                        local_shard_id
                    ].tensor

                    if len(local_optimizer_tensor.size()) > 1:  # 2D Optimizer Tensor
                        local_optimizer_tensor = local_optimizer_tensor[
                            :, split_offsets[0] : split_offsets[1]
                        ]

                    comm_op = dist.P2POp(
                        dist.isend,
                        local_optimizer_tensor,
                        dst_rank,
                        process_group,
                        tag_opt,
                    )
                    comm_ops_obj = CommP2PMetadata(
                        shard_name,
                        [split_offsets[0], split_offsets[1]],
                        comm_op,
                        False,
                        True,
                    )
                    if intra:
                        comm_ops[CommStrategy.OPT_INTRA_NODE_COMM].append(comm_ops_obj)
                    else:
                        comm_ops[CommStrategy.OPT_INTER_NODE_COMM].append(comm_ops_obj)

                local_shards = sharded_t.local_shards()

                local_tensor = local_shards[local_shard_id].tensor

                dst_t = local_tensor[:, split_offsets[0] : split_offsets[1]]
                if not dst_t.is_contiguous():
                    dst_t = dst_t.contiguous()

                comm_op_state = dist.P2POp(
                    dist.isend, dst_t, dst_rank, process_group, tag
                )
                comm_ops_obj = CommP2PMetadata(
                    shard_name,
                    [split_offsets[0], split_offsets[1]],
                    comm_op_state,
                    False,
                    False,
                )

                if intra:
                    comm_ops[CommStrategy.STATE_INTRA_NODE_COMM].append(comm_ops_obj)
                else:
                    comm_ops[CommStrategy.STATE_INTER_NODE_COMM].append(comm_ops_obj)

            # Process data being received at current rank
            if dst_rank == rank:
                receiving_shards_metadata.append(
                    (shard_offsets[1], src_rank, tag, tag_opt, shard_size)
                )

    if len(receiving_shards_metadata) > 0:

        receiving_shards_metadata.sort(key=lambda x: x[0])
        dst_t = new_state_dict[extended_shard_name]
        local_shards_dst = dst_t.local_shards()

        # TODO: assume only one tensor is available, but min_partiton property can have multiple tensors
        local_tensor_dst = local_shards_dst[0].tensor

        tensor_col_offset = 0
        for _, src_rank, tag, tag_opt, shard_size in receiving_shards_metadata:
            intra = _is_intra_comm(src_rank, rank, 8)
            end_col_offset = tensor_col_offset + shard_size[1]
            receving_tensor_view = local_tensor_dst[:, tensor_col_offset:end_col_offset]
            copy = False
            if not receving_tensor_view.is_contiguous():
                receving_tensor_view = receving_tensor_view.contiguous()
                copy = True

            comm_op_state_new = dist.P2POp(
                dist.irecv, receving_tensor_view, src_rank, process_group, tag
            )
            comm_ops_obj = CommP2PMetadata(
                shard_name,
                [tensor_col_offset, end_col_offset],
                comm_op_state_new,
                copy,
                False,
            )

            if intra:
                comm_ops[CommStrategy.STATE_INTRA_NODE_COMM].append(comm_ops_obj)
            else:
                comm_ops[CommStrategy.STATE_INTER_NODE_COMM].append(comm_ops_obj)
            if has_optimizer:
                momentum_name = _tmp_momentum_extender(shard_name)
                new_opt_state_state = new_opt_state["state"]
                sharded_t_opt = new_opt_state_state[extended_shard_name][momentum_name]
                local_tensor_opt_dst = sharded_t_opt._local_shards[0].tensor
                receving_tensor_view_opt = local_tensor_opt_dst
                if len(local_tensor_opt_dst.size()) > 1:
                    # TODO: changing col_dims require separate handling of tensor,for row_wise ada grad,
                    # because we need to merge incoming weights
                    receving_tensor_view_opt = local_tensor_opt_dst[
                        :, tensor_col_offset:end_col_offset
                    ]
                copy = False
                if not receving_tensor_view_opt.is_contiguous():
                    receving_tensor_view_opt = receving_tensor_view_opt.contiguous()
                    copy = True
                comm_op_opt_receiv = dist.P2POp(
                    dist.irecv,
                    receving_tensor_view_opt,
                    src_rank,
                    process_group,
                    tag_opt,
                )
                comm_ops_obj = CommP2PMetadata(
                    shard_name,
                    [tensor_col_offset, end_col_offset],
                    comm_op_opt_receiv,
                    copy,
                    True,
                )
                if intra:
                    comm_ops[CommStrategy.OPT_INTRA_NODE_COMM].append(comm_ops_obj)
                else:
                    comm_ops[CommStrategy.OPT_INTER_NODE_COMM].append(comm_ops_obj)
            tensor_col_offset = end_col_offset


def _update_state_post_resharding(
    old_state: Dict[str, ShardedTensor],
    new_state: Dict[str, ShardedTensor],
    changed_sharding_params: Dict[str, ParameterSharding],
    extend_shard_name: Callable[[str], str] = lambda x: x,
    comm_metadata: Optional[CommP2PMetadata] = None,
) -> None:
    """
    Updates the model state dictionary after resharding communication.

    This function handles two scenarios:
    1. If `comm_metadata` is provided, it updates a specific slice of the local shard tensor in `new_state`
       with the received data (`updated_tensor`) for the given shard and offsets.
    2. If `comm_metadata` is not provided, it iterates over all shards in `new_state` and copies the local
       shard tensors and metadata from `old_state` for shards that have not changed sharding parameters.

    Args:
        old_state (Dict[str, ShardedTensor]): Previous model state dictionary containing sharded tensors.
        new_state (Dict[str, ShardedTensor]): New model state dictionary to be updated.
        changed_sharding_params (Dict[str, ParameterSharding]): Dictionary of parameters whose sharding has changed.
        extend_shard_name (Callable[[str], str], optional): Function to extend shard names to full names in state dicts.
        comm_metadata (Optional[CommP2PMetadata], optional): Communication metadata for a specific shard update.

    Returns:
        None
    """
    if comm_metadata:
        shard_name = comm_metadata.shard_name
        offsets = comm_metadata.offsets
        updated_tensor = comm_metadata.comm_op.tensor
        extended_shard_name = extend_shard_name(shard_name)
        sharded_t = new_state[extended_shard_name]
        local_shards = sharded_t._local_shards
        assert len(sharded_t._local_shards) == 1
        local_t = sharded_t._local_shards[0].tensor
        local_t_slice = local_t[:, offsets[0] : offsets[1]]
        local_t_slice.copy_(updated_tensor, non_blocking=True)
    else:
        for extended_shard_name, _ in new_state.items():
            shard_name = _extract_shard_name(extended_shard_name)
            if (
                old_state is not None
                and extended_shard_name in old_state
                and shard_name not in changed_sharding_params.keys()
            ):

                sharded_t = new_state[extended_shard_name]
                sharded_t_old = old_state[extended_shard_name]

                local_shards = sharded_t._local_shards
                for i, shard in enumerate(local_shards):
                    shard.tensor.copy_(
                        sharded_t_old._local_shards[i].tensor, non_blocking=True
                    )
                    shard.metadata = sharded_t_old._local_shards[i].metadata


def _update_optimizer_state_post_resharding(
    old_opt_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]],
    new_opt_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]],
    changed_sharding_params: Dict[str, ParameterSharding],
    extend_shard_name: Callable[[str], str] = lambda x: x,
    comm_metadata: Optional[CommP2PMetadata] = None,
) -> None:
    """
    Updates the optimizer state dictionary after resharding communication.

    This function handles two scenarios:
    1. If `comm_metadata` is provided, it updates a specific slice of the local shard tensor in `new_opt_state`
       with the received data (`updated_tensor`) for the given shard and offsets.
    2. If `comm_metadata` is not provided, it iterates over all shards in `new_opt_state` and copies the local
       shard tensors and metadata from `old_opt_state` for shards that have not changed sharding parameters.

    Args:
        old_opt_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]]): Previous optimizer state dictionary.
        new_opt_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]]): New optimizer state dictionary to be updated.
        changed_sharding_params (Dict[str, ParameterSharding]): Dictionary of parameters whose sharding has changed.
        extend_shard_name (Callable[[str], str], optional): Function to extend shard names to full names in state dicts.
        comm_metadata (Optional[CommP2PMetadata], optional): Communication metadata for a specific shard update.

    Returns:
        None
    """
    new_opt_state_state = new_opt_state["state"] if new_opt_state else None
    old_opt_state_state = old_opt_state["state"] if old_opt_state else None

    if new_opt_state_state is None or len(new_opt_state_state) == 0:
        return

    if comm_metadata:
        shard_name = comm_metadata.shard_name
        offsets = comm_metadata.offsets
        updated_tensor = comm_metadata.comm_op.tensor
        momentum_name = _tmp_momentum_extender(shard_name)
        extended_shard_name = extend_shard_name(shard_name)
        item = new_opt_state_state[extended_shard_name]
        sharded_t = item[momentum_name]
        assert len(sharded_t._local_shards) == 1
        local_t = sharded_t._local_shards[0].tensor
        if len(local_t.size()) == 1:
            # TODO:  need a way to  merge the optimizer
            # Only invokes in row_wise adagrad, col dimesional changes
            local_t.copy_(updated_tensor, non_blocking=True)

        else:
            local_t_slice = local_t[:, offsets[0] : offsets[1]]
            local_t_slice.copy_(updated_tensor, non_blocking=True)

    else:
        for extended_shard_name, _ in new_opt_state_state.items():
            shard_name = _extract_shard_name(extended_shard_name)
            momentum_name = _tmp_momentum_extender(shard_name)
            if (
                old_opt_state_state is not None
                and extended_shard_name in old_opt_state_state
                and shard_name not in changed_sharding_params.keys()
            ):

                sharded_t = new_opt_state_state[extended_shard_name][momentum_name]
                sharded_t_old = old_opt_state_state[extended_shard_name][momentum_name]
                local_shards = sharded_t._local_shards
                for i, shard in enumerate(local_shards):
                    shard.tensor.copy_(
                        sharded_t_old._local_shards[i].tensor, non_blocking=True
                    )
                    shard.metadata = sharded_t_old._local_shards[i].metadata


def _aggregate_response_list(
    works: List[dist.Work], ops_list: List[CommP2PMetadata]
) -> List[Tuple[dist.Work, CommP2PMetadata]]:
    """
    Aggregates the results of asynchronous communication operations with their corresponding metadata.

    This function matches the list of distributed work objects (`works`) with the list of communication
    operation metadata (`ops_list`). If there is a one-to-one correspondence (no batching), it zips the two lists.
    If batching occurred (i.e., a single work object corresponds to multiple operations), it pairs each operation
    in `ops_list` with the first work object in `works`.

    Args:
        works (List[dist.Work]): List of distributed work objects returned by communication operations.
        ops_list (List[CommP2PMetadata]): List of communication operation metadata.

    Returns:
        List[Tuple[dist.Work, CommP2PMetadata]]: List of tuples pairing each work object with its corresponding metadata.
    """
    res: List[Tuple[dist.Work, CommP2PMetadata]] = []
    if len(works) == len(ops_list):
        # 1:1 mapping (no batching occurred)
        res.extend(list(zip(works, ops_list)))
    else:
        # Batching occurred - all operations share the same work object
        for entry in ops_list:
            res.append((works[0], entry))
    return res


"""
Utils for Optimizer State accessing
"""


def _tmp_momentum_extender(name: str) -> str:
    return name + ".momentum1"


def _extract_shard_name(name: str) -> str:
    return name.split(".")[-2]


def update_state_dictionaries(
    reqs: List[Tuple[dist.Work, CommP2PMetadata]],
    old_optimizer_state: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    new_optimizer_state: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    old_state: Dict[str, Any],
    new_state: Dict[str, Any],
    changed_sharding_params: Dict[str, ParameterSharding],
    extend_shard_name: Callable[[str], str] = lambda x: x,
    update_local: bool = False,
) -> None:
    """
    Updates the state dictionaries for model and optimizer after resharding communication operations.

    This function waits for corresponding asynchronous communication operations (such as isend/irecv) to complete,
    and then updates the local state dictionaries (model and optimizer) with the received data.
    It handles both optimizer state and model state updates, depending on the communication metadata.
    If `update_local` is True, it also updates local shards that were not changed during communication.

    Args:
        reqs (List[Tuple[dist.Work, CommP2PMetadata]]): List of tuples containing waitable communication work objects
            and associated metadata for each communication operation.
        old_optimizer_state (Optional[Dict[str, Dict[str, Dict[str, Any]]]]): Previous optimizer state dictionary.
        new_optimizer_state (Optional[Dict[str, Dict[str, Dict[str, Any]]]]): New optimizer state dictionary to be updated.
        old_state (Dict[str, Any]): Previous model state dictionary.
        new_state (Dict[str, Any]): New model state dictionary to be updated.
        changed_sharding_params (Dict[str, ParameterSharding]): Dictionary of sharding parameters that have changed.
        extend_shard_name (Callable[[str], str], optional): Function to extend shard names to full names in state dicts.
        update_local (bool, optional): If True, also update local shards that were not changed during communication.

    Returns:
        None
    """
    for waitable, comm_obj in reqs:
        waitable.wait()
        if comm_obj.requires_copy:
            if comm_obj.is_optimizer_state:
                _update_optimizer_state_post_resharding(
                    old_opt_state=old_optimizer_state,
                    new_opt_state=new_optimizer_state,
                    changed_sharding_params=changed_sharding_params,
                    extend_shard_name=extend_shard_name,
                    comm_metadata=comm_obj,
                )
            else:
                _update_state_post_resharding(
                    old_state=old_state,
                    new_state=new_state,
                    changed_sharding_params=changed_sharding_params,
                    extend_shard_name=extend_shard_name,
                    comm_metadata=comm_obj,
                )
    if update_local:
        if old_optimizer_state:
            _update_optimizer_state_post_resharding(
                old_opt_state=old_optimizer_state,
                new_opt_state=new_optimizer_state,
                changed_sharding_params=changed_sharding_params,
                extend_shard_name=extend_shard_name,
            )

        _update_state_post_resharding(
            old_state=old_state,
            new_state=new_state,
            changed_sharding_params=changed_sharding_params,
            extend_shard_name=extend_shard_name,
        )


def move_sharded_tensors_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively traverse a state dictionary and move all local shard tensors to CPU.
    This helps reduce GPU memory usage by keeping tensors in CPU memory until needed.

    Args:
        state_dict: The state dictionary to traverse (can be model or optimizer state dict)

    Returns:
        The modified state dictionary with tensors moved to CPU
    """

    # Pyre-ignore
    def _process_item(item: Any) -> Any:
        if isinstance(item, ShardedTensor):
            # For ShardedTensor, move all local shards to CPU
            for shard in item.local_shards():
                if shard.tensor.device.type == "cuda":
                    shard.tensor = shard.tensor.cpu()
            return item
        elif isinstance(item, dict):
            # Recursively process dictionaries
            return {k: _process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            # Recursively process lists
            return [_process_item(v) for v in item]
        elif isinstance(item, tuple):
            # Recursively process tuples
            return tuple(_process_item(v) for v in item)
        else:
            # Return other types unchanged
            return item

    processed_dict = _process_item(state_dict)
    torch.cuda.empty_cache()
    return processed_dict


def prepare_comm_ops(
    module_sharding_plan: EmbeddingModuleShardingPlan,
    current_state_dict: Dict[str, ShardedTensor],
    new_state_dict: Dict[str, ShardedTensor],
    changed_sharding_params: Dict[str, ParameterSharding],
    shard_name: str,
    env: ShardingEnv,
    extend_shard_name: Callable[[str], str] = lambda x: x,
    current_opt_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
    new_opt_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
    has_optimizer: bool = False,
) -> Dict[CommStrategy, List[CommP2PMetadata]]:
    """
    Prepares the communication operations required for resharding a specific embedding table or parameter.

    This function determines the necessary point-to-point (P2P) communication operations (send/receive)
    to redistribute sharded tensors (and optionally optimizer state) according to a new sharding plan.
    It supports both inter-node and intra-node communication, and handles both model and optimizer state.

    Args:
        module_sharding_plan (EmbeddingModuleShardingPlan): The current sharding plan for the module.
        current_state_dict (Dict[str, ShardedTensor]): The current state dictionary containing sharded tensors.
        new_state_dict (Dict[str, ShardedTensor]): The new state dictionary to be updated after resharding.
        changed_sharding_params (Dict[str, ParameterSharding]): Dictionary of parameters whose sharding has changed.
        shard_name (str): The name of the shard/table to be resharded.
        env (ShardingEnv): The sharding environment containing world size, rank, and process group.
        extend_shard_name (Callable[[str], str], optional): Function to extend shard names to full names in state dicts.
        current_opt_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]], optional): Current optimizer state dict.
        new_opt_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]], optional): New optimizer state dict to update.
        has_optimizer (bool, optional): Whether optimizer state communication is required.

    Returns:
        Dict[CommStrategy, List[CommP2PMetadata]]: A dictionary mapping communication strategies to lists of
        communication metadata, describing the P2P operations to be performed for resharding.
    """
    comm_dict = {
        CommStrategy.STATE_INTER_NODE_COMM: [],
        CommStrategy.STATE_INTRA_NODE_COMM: [],
        CommStrategy.OPT_INTER_NODE_COMM: [],
        CommStrategy.OPT_INTRA_NODE_COMM: [],
    }

    if env.output_dtensor:
        raise RuntimeError("We do not yet support DTensor for resharding yet")

    world_size = env.world_size
    rank = dist.get_rank()
    pg = env.process_group

    param = changed_sharding_params[shard_name]

    assert param.ranks is not None

    src_params = module_sharding_plan[shard_name]

    assert (
        param.sharding_type == ShardingType.COLUMN_WISE.value
        or param.sharding_type == ShardingType.TABLE_WISE.value
    )

    shard_mapping = _generate_shard_allocation_metadata(
        source_params=src_params,
        destination_params=param,
        rank=rank,
        world_size=world_size,
    )

    _prepare_shard_distribution_comm_ops(
        rank=rank,
        shard_name=shard_name,
        shard_to_rank_mapping=shard_mapping,
        current_state_dict=current_state_dict,
        new_state_dict=new_state_dict,
        comm_ops=comm_dict,
        current_opt_state=current_opt_state,
        new_opt_state=new_opt_state,
        extend_shard_name=extend_shard_name,
        has_optimizer=has_optimizer,
        process_group=pg,
    )

    return comm_dict


def transfer_data(
    comms_op: Dict[CommStrategy, List[CommP2PMetadata]]
) -> List[Tuple[dist.Work, CommP2PMetadata]]:
    """
    Executes batched point-to-point communication operations for resharding.

    This function processes the communication operations grouped by strategy (inter/intra node, state/optimizer)
    and launches batched isend/irecv operations for each group using torch.distributed. It then aggregates
    the resulting work objects with their corresponding communication metadata for later synchronization and
    post-processing.

    Args:
        comms_op (Dict[CommStrategy, List[CommP2PMetadata]]): Dictionary mapping communication strategies to lists
            of communication metadata describing the P2P operations to be performed.

    Returns:
        List[Tuple[dist.Work, CommP2PMetadata]]: List of tuples pairing each distributed work object with its
        corresponding communication metadata, to be used for waiting and updating state dictionaries.
    """

    reqs: List[Tuple[dist.Work, CommP2PMetadata]] = []
    for _, ops in comms_op.items():
        if len(ops) > 0:
            works = dist.batch_isend_irecv([entry.comm_op for entry in ops])
            reqs.extend(_aggregate_response_list(works, ops))
    return reqs


def update_module_sharding_plan(
    module: ShardedModule[Any, Any, Any, Any],  # pyre-ignore
    changed_sharding_params: Dict[str, ParameterSharding],
    sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]],
) -> None:
    """
    Updates the module's sharding plan and associated sharding information after a change in sharding parameters.

    This function modifies the `module_sharding_plan` attribute of the given `module` by updating it with the
    new sharding parameters provided in `changed_sharding_params`. It also updates the corresponding
    `EmbeddingShardingInfo` objects in `sharding_type_to_sharding_infos` to reflect the new parameter sharding.

    Args:
        module (ShardedModule): The sharded module whose sharding plan is to be updated. Must have a `module_sharding_plan` attribute.
        changed_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping parameter/table names to their new sharding parameters.
        sharding_type_to_sharding_infos (Dict[str, List[EmbeddingShardingInfo]]): A mapping from sharding type to a list of
            `EmbeddingShardingInfo` objects, which are updated to reflect the new sharding parameters.

    Raises:
        RuntimeError: If the module does not have a `module_sharding_plan` attribute.

    Returns:
        None
    """
    if not hasattr(module, "module_sharding_plan"):
        raise RuntimeError("Module does not have a module_sharding_plan attribute")

    module_sharding_plan = module.module_sharding_plan
    for name, param in changed_sharding_params.items():
        # pyre-ignore
        module_sharding_plan[name] = param
        # TODO: Support detecting old sharding type when sharding type is changing
        for sharding_info in sharding_type_to_sharding_infos[param.sharding_type]:
            if sharding_info.embedding_config.name == name:
                sharding_info.param_sharding = param


# Utils
def output_sharding_plan_delta_single(
    old_plan: EmbeddingModuleShardingPlan,
    new_plan: EmbeddingModuleShardingPlan,
    return_data_volume: bool = False,
) -> Tuple[float, EmbeddingModuleShardingPlan]:
    """
    Compute and return a new sharding plan that is the delta
    between new and old embedding module plans. Assumes that the old and new plan
    have the same number of parameters/tables.

    This is useful for Dynamic Sharding since Resharding API takes in only the
    ParameterSharding or shards that needs to be moved. Takes in EmbeddingModuleShardingPlan.
    """
    assert len(old_plan) == len(new_plan)
    diff = EmbeddingModuleShardingPlan(
        {
            k: copy.deepcopy(v)
            for k, v in new_plan.items()
            if v.ranks != old_plan[k].ranks
        }
    )
    data_volume: float = 0
    if return_data_volume:
        for _, v in diff.items():
            # Pyre-ignore
            for shard in v.sharding_spec.shards:
                data_volume += (
                    shard.shard_sizes[0] * shard.shard_sizes[1] * 4 / (1024 * 1024)
                )  # Asumming float datatype

    return (data_volume, diff)


def output_sharding_plans_delta(
    old_plan: Dict[str, EmbeddingModuleShardingPlan],
    new_plan: Dict[str, EmbeddingModuleShardingPlan],
    return_data_volume: bool = False,
) -> Dict[str, Tuple[float, EmbeddingModuleShardingPlan]]:
    """
    Compute and return a new sharding plan that is the delta
    between new and old embedding module plans. Assumes that the old and new plan
    have the same number of parameters/tables.

    This is useful for Dynamic Sharding since Resharding API takes in only the
    ParameterSharding or shards that needs to be moved. Takes in a Dict
    which is the format of DMP sharding plans.
    """
    delta_plans: Dict[str, Tuple[float, EmbeddingModuleShardingPlan]] = {}
    for key, plan in old_plan.items():
        assert key in new_plan

        delta_plans[key] = output_sharding_plan_delta_single(
            plan, new_plan[key], return_data_volume
        )
    return delta_plans
