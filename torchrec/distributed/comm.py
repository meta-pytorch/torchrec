#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.types import ShardingEnv2D

logger: logging.Logger = logging.getLogger(__name__)

# Global, only should be accessed via intra_and_cross_node_pg()
_INTRA_PG: Optional[dist.ProcessGroup] = None
_CROSS_PG: Optional[dist.ProcessGroup] = None

# For 2D parallel
_INTRA_PG_2D: Optional[dist.ProcessGroup] = None
_CROSS_PG_2D: Optional[dist.ProcessGroup] = None
_NODE_GROUP_SIZE_2D: Optional[int] = None


def _env2int(env_list: List[str], default: int = -1) -> int:
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def get_local_size(world_size: Optional[int] = None) -> int:
    if world_size is None:
        world_size = dist.get_world_size()
    """
    Gets the local world size (see https://pytorch.org/docs/stable/elastic/run.html)
    This is usually the size of workers on each node, or nproc_per_node
    """
    local_size = _env2int(
        [
            "LOCAL_WORLD_SIZE",
            "MPI_LOCALNRANKS",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "MV2_COMM_WORLD_LOCAL_SIZE",
        ],
        8,
    )

    if local_size == -1 or world_size % local_size != 0:
        logging.warning(
            "Could not determine LOCAL_WORLD_SIZE from environment, falling back to WORLD_SIZE."
        )
        local_size = world_size
    return local_size


def get_node_group_size(world_size: Optional[int] = None) -> int:
    """
    Get the local world size accounting for 2D environment, if not set, we fallback to global environment
    """
    if _NODE_GROUP_SIZE_2D is None:
        return get_local_size(world_size)
    return _NODE_GROUP_SIZE_2D


def get_local_rank(world_size: Optional[int] = None, rank: Optional[int] = None) -> int:
    """
    Gets the local rank of the local processes (see https://pytorch.org/docs/stable/elastic/run.html)
    This is usually the rank of the worker on its node
    """
    my_local_rank = _env2int(
        [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
        ],
        -1,
    )
    local_size = get_local_size(world_size)

    if my_local_rank == -1 or my_local_rank >= local_size:
        logging.warning(
            "Could not determine LOCAL_RANK from environment, falling back to GLOBAL_RANK % LOCAL_SIZE."
        )
        if rank is None:
            rank = dist.get_rank()
        my_local_rank = rank % local_size
    return my_local_rank


def get_group_rank(world_size: Optional[int] = None, rank: Optional[int] = None) -> int:
    """
    Gets the group rank of the worker group. Also available with GROUP_RANK environment varible
    A number between 0 and get_num_groups() (See https://pytorch.org/docs/stable/elastic/run.html)
    """
    if rank is None:
        rank = dist.get_rank()
    return rank // get_local_size(world_size)


def get_num_groups(world_size: Optional[int] = None) -> int:
    """
    Gets the number of worker groups.
    Usually equivalent to max_nnodes (See https://pytorch.org/docs/stable/elastic/run.html)
    """
    if world_size is None:
        world_size = dist.get_world_size()
    return world_size // get_local_size(world_size)


def get_topology_domain_multiple() -> Optional[int]:
    """The number of host/server/node per pod/domain."""
    topology_domain_multiple = _env2int(
        [
            "TOPOLOGY_DOMAIN_MULTIPLE",
        ],
        -1,
    )
    if topology_domain_multiple == -1:
        return None
    return topology_domain_multiple


def get_topology_group_world_size(world_size: Optional[int] = None) -> int:
    """
    Gets topology group world size, total number of processes linked within a topology group

    This is the largest number of processes linked together by high-bandwidth communication.
    If it isn't specified, it falls back to LOCAL_WORLD_SIZE
    """
    topology_domain_multiple = get_topology_domain_multiple()
    local_world_size = get_local_size(world_size)

    if topology_domain_multiple is None:
        logger.warning(
            "Could not determine TOPOLOGY_DOMAIN_MULTIPLE from environment,"
            " utilizing LOCAL_WORLD_SIZE instead."
        )
        return local_world_size

    # Total number of processes/gpu in domain = topology_domain_mult * number_gpu_per_domain
    numb_proc_per_pod = topology_domain_multiple * local_world_size
    world_size = dist.get_world_size() if world_size is None else world_size
    if world_size % numb_proc_per_pod != 0:
        raise ValueError(
            f"World size {world_size} is not a multiple of the topology group: {numb_proc_per_pod}"
        )
    return numb_proc_per_pod


def intra_and_cross_node_pg(
    device: Optional[torch.device] = None,
    backend: Optional[str] = None,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """
    Creates sub process groups (intra and cross node)

    e.g. world_size = 12 need to split into groups of size `local_size = 6`
    process groups = [0, 1, 2, ... 11]

    intra-group:
        [0] -> [0, 1, .., 5]
        [1] -> [0, 1, .., 5]
        ...
        [6] -> [6, ..., 11]
        ...
        [11] -> [6, ..., 11]
    cross-group:
        [0] -> [[0, 6]]
        [1] -> [[1, 7]]
        ...
        [5] -> [[5, 11]]
    """
    if device is not None and device.type == "meta":
        return None, None

    global _INTRA_PG  # intra node process group
    global _CROSS_PG  # cross node process group

    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    local_size = get_topology_group_world_size(world_size)
    # TODO: Alireza look into incorporating topology group WS in
    #       get_group_rank, get_num_groups, get_local_rank
    my_group_rank = my_rank // local_size
    group_count = world_size // local_size
    my_local_rank = my_rank % local_size  # Not the same as the actual local_rank

    if backend is None:
        backend = dist.get_backend()

    logger.info(
        f"[{my_rank}] my_local_rank = {my_local_rank}, local_size = {local_size},"
        f"my_group_rank = {my_group_rank}, group_count = {group_count}, backend = {backend}"
    )
    if _INTRA_PG is None:
        for group_rank in range(group_count):
            peers = [group_rank * local_size + r for r in range(local_size)]
            curr_intra_group_pg = dist.new_group(backend=backend, ranks=peers)
            if my_group_rank == group_rank:
                logger.warning(
                    "[Connection] intra_group: [%d] -> [%s]" % (my_rank, peers)
                )
                _INTRA_PG = curr_intra_group_pg
        assert _INTRA_PG is not None
        dist.barrier()

    if _CROSS_PG is None:
        for l_rank in range(local_size):
            peers = [l_rank + g * local_size for g in range(group_count)]
            curr_cross_group_pg = dist.new_group(backend=backend, ranks=peers)
            if l_rank == my_local_rank:
                logger.warning(
                    "[Connection] cross_group: [%d] -> [%s]" % (my_rank, peers)
                )
                _CROSS_PG = curr_cross_group_pg
        assert _CROSS_PG is not None
        dist.barrier()

    return _INTRA_PG, _CROSS_PG


def intra_and_cross_node_pg_2D(
    env: ShardingEnv2D,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """
    Creates sub process groups (intra and cross node) under 2D parallelism scheme
    The concept of "intra" and "cross" node is lost under a 2D parallelism scheme
    due to the ranks that exist under a sharding group do not have gurantee of the typical
    node topology. And as such there are no guarantees of "intra" group exploiting intra node bandwidth.

    NOTE:
    These process groups are created for sharding schemes (ie: GRID) that were designed to exploit
    intra node bandwidth for optimized comms. There will be future work to redesign the comms for GRID
    sharding to be optimized under a 2D setup.

    Example::
    Here is what "intra" and "cross" groups look like in a 2D environment,
    Sharding Groups:
        Group 0: [0, 2, 4, 6]
        Group 1: [1, 3, 5, 7]
    devices_per_node = 2:
    "intra" groups for each sharding group,
        Group 0: [0, 2], [4, 6]
        Group 1: [1, 3], [5, 7]
    "cross" groups for each sharding group,
        Group 0: [0, 4], [2, 6]
        Group 1: [1, 5], [3, 7]

    We can see as this scales to real world topologies how the "intra" and "cross" node ideas in a traditional
    sense are not applicable here.
    """
    if device is not None and device.type == "meta":
        return None, None

    global _INTRA_PG_2D
    global _CROSS_PG_2D
    global _NODE_GROUP_SIZE_2D

    backend = dist.get_backend(env.sharding_pg)
    my_rank = dist.get_rank()

    sharding_group_size = dist.get_world_size(
        env.sharding_pg
    )  # Local replica group world size
    world_size = dist.get_world_size()  # Global world size
    step = world_size // sharding_group_size
    devices_per_node = (
        env.node_group_size if env.node_group_size else get_local_size(world_size)
    )
    _NODE_GROUP_SIZE_2D = devices_per_node

    assert (
        sharding_group_size % devices_per_node == 0
    ), f"node group size is not divisible by sharding group size, {devices_per_node=}, {sharding_group_size=}"
    intra_pg_groups: List[List[List[int]]] = [[] for _ in range(step)]

    if _INTRA_PG_2D is None:
        for group_rank in range(step):
            if env.use_inter_host_allreduce:
                # for inter host all reduce, we change the sharding group calculation to be continuous
                ranks = group_rank * sharding_group_size
                sharding_pg_peers = list(range(ranks, ranks + sharding_group_size))
            else:
                sharding_pg_peers = [
                    step * r + group_rank for r in range(sharding_group_size)
                ]
            for group in range(len(sharding_pg_peers) // devices_per_node):
                intra_pg_peers = sharding_pg_peers[
                    group * devices_per_node : (group + 1) * devices_per_node
                ]
                intra_pg_groups[group_rank].append(intra_pg_peers)
                curr_intra_pg = dist.new_group(
                    backend=backend,
                    ranks=intra_pg_peers,
                    group_desc="sharding_intra_pg",
                )
                if my_rank in intra_pg_peers:
                    logger.warning(
                        f"[Connection] 2D rank {my_rank} -> intra_pg_peers {intra_pg_peers}"
                    )
                    _INTRA_PG_2D = curr_intra_pg
        assert _INTRA_PG_2D is not None, "INTRA_PG_2D is not initialized!"
        dist.barrier()

    if _CROSS_PG_2D is None:
        for group_rank in range(step):
            intra_pg_group = intra_pg_groups[group_rank]
            for cross_group_rank in range(devices_per_node):
                cross_pg_peers = [
                    intra_pg_group[j][cross_group_rank]
                    for j in range(len(intra_pg_group))
                ]
                curr_cross_pg = dist.new_group(
                    backend=backend,
                    ranks=cross_pg_peers,
                    group_desc="sharding_cross_pg",
                )
                if my_rank in cross_pg_peers:
                    logger.warning(
                        f"[Connection] 2D rank {my_rank} -> cross_pg_peers {cross_pg_peers}"
                    )
                    _CROSS_PG_2D = curr_cross_pg
        assert _CROSS_PG_2D is not None, "CROSS_PG_2D is not initialized!"
        dist.barrier()

    return _INTRA_PG_2D, _CROSS_PG_2D
