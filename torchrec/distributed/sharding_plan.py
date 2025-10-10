#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import math
import warnings
from typing import Callable, cast, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import distributed as dist, nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fp_embeddingbag import (
    FeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.mc_embedding import ManagedCollisionEmbeddingCollectionSharder
from torchrec.distributed.mc_embeddingbag import (
    ManagedCollisionEmbeddingBagCollectionSharder,
)
from torchrec.distributed.mc_modules import InferManagedCollisionCollectionSharder
from torchrec.distributed.planner.constants import MIN_CW_DIM
from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollectionSharder,
    QuantManagedCollisionEmbeddingCollectionSharder,
)
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import get_device_type, none_throws


def get_default_sharders() -> List[ModuleSharder[nn.Module]]:
    return [
        cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], FeatureProcessedEmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder()),
        cast(ModuleSharder[nn.Module], FusedEmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], QuantEmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], QuantEmbeddingCollectionSharder()),
        cast(ModuleSharder[nn.Module], ManagedCollisionEmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], ManagedCollisionEmbeddingCollectionSharder()),
        cast(
            ModuleSharder[nn.Module],
            QuantManagedCollisionEmbeddingCollectionSharder(
                QuantEmbeddingCollectionSharder(),
                InferManagedCollisionCollectionSharder(),
            ),
        ),
    ]


def get_module_to_default_sharders() -> Dict[Type[nn.Module], ModuleSharder[nn.Module]]:
    return {sharder.module_type: sharder for sharder in get_default_sharders()}


def placement(
    compute_device: str,
    rank: int,
    local_size: int,
) -> str:
    param_device = compute_device
    if compute_device in {"cuda", "mtia"}:
        param_device = torch.device(compute_device, rank % local_size)
    return f"rank:{rank}/{param_device}"


# TODO: Consolidate placement and placement_helper into one function.
def placement_helper(device_type: str, index: int = 0, rank: int = 0) -> str:
    if device_type == "cpu":
        return f"rank:0/{device_type}"  # cpu only use rank 0

    result = f"rank:{rank}/{device_type}:{index}"
    return result


def calculate_shard_sizes_and_offsets(
    tensor: torch.Tensor,
    world_size: int,
    local_world_size: int,
    sharding_type: str,
    col_wise_shard_dim: Optional[int] = None,
    device_memory_sizes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Calculates sizes and offsets for tensor sharded according to provided sharding type.

    Args:
        tensor (torch.Tensor): tensor to be sharded.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        sharding_type (str): provided ShardingType value.
        col_wise_shard_dim (Optional[int]): dimension for column wise sharding split.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: shard sizes, represented as a list of the dimensions of the sharded tensor on each device, and shard offsets, represented as a list of coordinates of placement on each device.

    Raises:
        ValueError: If `sharding_type` is not a valid ShardingType.
    """

    (rows, columns) = tensor.shape

    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return [[rows, columns]] * world_size, [[0, 0]] * world_size
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return [[rows, columns]], [[0, 0]]
    elif sharding_type == ShardingType.ROW_WISE.value:
        return (
            _calculate_rw_shard_sizes_and_offsets(rows, world_size, columns)
            if not device_memory_sizes
            else _calculate_uneven_rw_shard_sizes_and_offsets(
                rows, world_size, columns, device_memory_sizes
            )
        )
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_rw_shard_sizes_and_offsets(rows, local_world_size, columns)
    elif (
        sharding_type == ShardingType.COLUMN_WISE.value
        or sharding_type == ShardingType.TABLE_COLUMN_WISE.value
    ):
        return _calculate_cw_shard_sizes_and_offsets(columns, rows, col_wise_shard_dim)
    elif sharding_type == ShardingType.GRID_SHARD.value:
        return _calculate_grid_shard_sizes_and_offsets(
            rows, local_world_size, columns, col_wise_shard_dim
        )

    raise ValueError(
        f"Unrecognized or unsupported sharding type provided: {sharding_type}"
    )


def _calculate_grid_shard_sizes_and_offsets(
    hash_size: int,
    num_device: int,
    columns: int,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Similar to row-wise case, but also splits columns into blocks of size `col_wise_shard_dim`.
    """
    row_shard_sizes, row_shard_offsets = _calculate_rw_shard_sizes_and_offsets(
        hash_size, num_device, columns
    )
    block_size = _get_block_size_for_cw_shard(columns, col_wise_shard_dim)
    num_col_wise_nodes, _residual = divmod(columns, block_size)
    shard_sizes: List[List[int]] = []
    shard_offsets: List[List[int]] = []

    for node in range(num_col_wise_nodes):
        for row_shard_size, row_shard_offset in zip(row_shard_sizes, row_shard_offsets):
            shard_sizes.append([row_shard_size[0], block_size])
            shard_offsets.append([row_shard_offset[0], block_size * node])
    return shard_sizes, shard_offsets


def _calculate_rw_shard_sizes_and_offsets(
    hash_size: int, num_devices: int, columns: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Sets prefix of shard_sizes to be `math.ceil(hash_size/num_devices)`.

    For example if hash_size = 10, num_devices = 4, we will allocate the rows as 3,3,3,1
    (rather than 3,3,2,2).
    This is due to implementation in RW sharding that sets block_size_lists to be ceil.
    The balanced way is harder to support on GPU.
    For more details see https://fb.quip.com/xbgbAchCTOL0

    Also consider the example of hash_size = 5, num_devices = 4. The expected rows per
    rank is [2,2,1,0].
    """

    block_size: int = math.ceil(hash_size / num_devices)
    last_rank: int = hash_size // block_size
    last_block_size: int = hash_size - block_size * last_rank
    shard_sizes: List[List[int]] = []

    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = block_size
        elif rank == last_rank:
            local_row: int = last_block_size
        else:
            local_row: int = 0
        shard_sizes.append([local_row, columns])
    shard_offsets = [[0, 0]]

    for i in range(num_devices - 1):
        shard_offsets.append([shard_sizes[i][0] + shard_offsets[i][0], 0])

    return shard_sizes, shard_offsets


def _calculate_uneven_rw_shard_sizes_and_offsets(
    hash_size: int, num_devices: int, columns: int, device_memory_sizes: List[int]
) -> Tuple[List[List[int]], List[List[int]]]:
    assert num_devices == len(device_memory_sizes), "must provide all the memory size"
    total_size = sum(device_memory_sizes)
    shard_sizes: List[List[int]] = []
    last_rank = num_devices - 1

    processed_total_rows = 0

    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = int(hash_size * (device_memory_sizes[rank] / total_size))
            processed_total_rows += local_row
        elif rank == last_rank:
            local_row: int = hash_size - processed_total_rows
        else:
            local_row: int = 0
        shard_sizes.append([local_row, columns])
    shard_offsets = [[0, 0]]

    for i in range(num_devices - 1):
        shard_offsets.append([shard_sizes[i][0] + shard_offsets[i][0], 0])

    return shard_sizes, shard_offsets


def _find_base_dim(lower_bound: int, dim: int) -> int:
    for i in range(lower_bound, dim):
        if dim % i == 0 and i % 4 == 0:
            return i
    return dim


def _get_block_size_for_cw_shard(
    columns: int, column_wise_shard_dim: Optional[int]
) -> int:
    block_size: int = min(
        (
            _find_base_dim(column_wise_shard_dim, columns)
            if column_wise_shard_dim
            else _find_base_dim(MIN_CW_DIM, columns)
        ),
        columns,
    )

    if columns % block_size != 0:
        warnings.warn(
            f"Dim of {columns} cannot be evenly divided with column wise shard"
            "dim {column_wise_shard_dim}, overriding block_size to embedding_dim={columns}",
            UserWarning,
            stacklevel=2,
        )
        block_size = columns
    return block_size


def _calculate_cw_shard_sizes_and_offsets(
    columns: int,
    rows: int,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    block_size = _get_block_size_for_cw_shard(columns, col_wise_shard_dim)

    num_col_wise_shards, _residual = divmod(columns, block_size)

    shard_sizes: List[List[int]] = [[rows, block_size]] * num_col_wise_shards
    shard_offsets: List[List[int]] = [
        [0, block_size * rank] for rank in range(num_col_wise_shards)
    ]
    return shard_sizes, shard_offsets


def _get_parameter_size_offsets(
    param: torch.nn.Parameter,
    sharding_type: ShardingType,
    local_size: int,
    world_size: int,
    col_wise_shard_dim: Optional[int] = None,
) -> List[Tuple[List[int], List[int]]]:
    (
        shard_sizes,
        shard_offsets,
    ) = calculate_shard_sizes_and_offsets(
        tensor=none_throws(param),
        world_size=world_size,
        local_world_size=local_size,
        sharding_type=sharding_type.value,
        col_wise_shard_dim=col_wise_shard_dim,
    )
    return list(zip(shard_sizes, shard_offsets))


def _get_compute_kernel(
    sharder: ModuleSharder[nn.Module],
    param: nn.Parameter,
    sharding_type: str,
    device_type: str,
) -> str:
    # TODO add placement support for compute_kernel
    compute_kernels = [EmbeddingComputeKernel.DENSE.value]
    if sharding_type != ShardingType.DATA_PARALLEL.value:
        compute_kernels += [
            EmbeddingComputeKernel.FUSED.value,
        ]
    if device_type in {"cuda"}:
        compute_kernels += [
            EmbeddingComputeKernel.FUSED_UVM.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        ]

    if sharding_type == ShardingType.DATA_PARALLEL.value:
        if EmbeddingComputeKernel.DENSE.value in compute_kernels:
            return EmbeddingComputeKernel.DENSE.value
        elif EmbeddingComputeKernel.QUANT.value in compute_kernels:
            return EmbeddingComputeKernel.QUANT.value
    else:
        if (
            hasattr(param, "_in_backward_optimizers")
            and EmbeddingComputeKernel.FUSED.value in compute_kernels
        ):
            return EmbeddingComputeKernel.FUSED.value
        elif EmbeddingComputeKernel.DENSE.value in compute_kernels:
            return EmbeddingComputeKernel.DENSE.value
        elif EmbeddingComputeKernel.QUANT.value in compute_kernels:
            return EmbeddingComputeKernel.QUANT.value

    raise ValueError(
        f"Could not find compute kernel for sharding_type={sharding_type} in {compute_kernels}"
    )


def _get_parameter_sharding(
    param: nn.Parameter,
    sharding_type: str,
    size_offset_ranks: List[Tuple[List[int], List[int], int]],
    local_size: int,
    device_type: str,
    sharder: ModuleSharder[nn.Module],
    placements: Optional[List[str]] = None,
    compute_kernel: Optional[str] = None,
) -> ParameterSharding:
    return ParameterSharding(
        sharding_spec=(
            None
            if sharding_type == ShardingType.DATA_PARALLEL.value
            else EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=size,
                        shard_offsets=offset,
                        placement=(
                            placement(
                                device_type,
                                none_throws(rank),
                                none_throws(local_size),
                            )
                            if not device_placement
                            else device_placement
                        ),
                    )
                    for (size, offset, rank), device_placement in zip(
                        size_offset_ranks,
                        placements if placements else [None] * len(size_offset_ranks),
                    )
                ]
            )
        ),
        sharding_type=sharding_type,
        compute_kernel=(
            compute_kernel
            if compute_kernel
            else _get_compute_kernel(sharder, param, sharding_type, device_type)
        ),
        ranks=[rank for (_, _, rank) in size_offset_ranks],
    )


ParameterShardingGenerator = Callable[
    [
        nn.Parameter,
        int,
        int,
        str,
        ModuleSharder[nn.Module],
    ],
    ParameterSharding,
]


def get_sharding_constructor_from_type(
    sharding_type: ShardingType,
) -> Callable[..., ParameterShardingGenerator]:
    sharding_type_to_constructor = {
        ShardingType.TABLE_WISE: table_wise,
        ShardingType.ROW_WISE: row_wise,
        ShardingType.COLUMN_WISE: column_wise,
        ShardingType.TABLE_ROW_WISE: table_row_wise,
        ShardingType.GRID_SHARD: grid_shard,
        ShardingType.DATA_PARALLEL: data_parallel,
    }
    return sharding_type_to_constructor[sharding_type]


def data_parallel() -> ParameterShardingGenerator:
    """
    Returns a generator of ParameterShardingPlan for `ShardingType::DATA_PARALLEL` for construct_module_sharding_plan.

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_0": data_parallel(),
            },
        )
    """

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.DATA_PARALLEL,
            local_size,
            world_size,
        )
        size_offset_ranks = []

        assert len(size_and_offsets) == world_size
        for (size, offset), rank in zip(size_and_offsets, range(world_size)):
            size_offset_ranks.append((size, offset, rank))

        return _get_parameter_sharding(
            param,
            ShardingType.DATA_PARALLEL.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def table_wise(
    rank: int,
    device: Optional[str] = None,
    compute_kernel: Optional[str] = None,
) -> ParameterShardingGenerator:
    """
    Returns a generator of ParameterShardingPlan for `ShardingType::TABLE_WISE` for construct_module_sharding_plan.

    Args:
    rank (int): rank to place table when doing table wise
    device (Optional[str]): device to place table when doing table_wise sharding
    compute_kernel (Optional[str]): embedding compute kernel to use for the table

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_0": table_wise(rank=0),
            },
        )
    """

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.TABLE_WISE,
            local_size,
            world_size,
        )
        assert len(size_and_offsets) == 1
        (size, offset) = size_and_offsets[0]
        size_offset_ranks = [(size, offset, rank)]

        return _get_parameter_sharding(
            param,
            ShardingType.TABLE_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
            placements=([placement_helper(device, rank, rank)] if device else None),
            compute_kernel=compute_kernel,
        )

    return _parameter_sharding_generator


def row_wise(
    sizes_placement: Optional[Tuple[List[int], Union[str, List[str]]]] = None,
    compute_kernel: Optional[str] = None,
) -> ParameterShardingGenerator:
    """
    Returns a generator of ParameterShardingPlan for `ShardingType::ROW_WISE` for construct_module_sharding_plan.

    Args:
    sizes_placement (Optional[Tuple[List[int], str]]): Only use it in inference for uneven shardinglist of tuples of (sizes, placement); sizes is the row size list

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_1": row_wise(),
                "table_2": row_wise([10, 5, 0, 3], "cpu")
            },
        )
    """

    if sizes_placement is not None and isinstance(sizes_placement[1], list):
        assert len(sizes_placement[0]) == len(
            sizes_placement[1]
        ), "sizes_placement and device per placement (in case of sharding "
        "across HBM and CPU host) must have the same length"

    compute_kernel = (
        EmbeddingComputeKernel.QUANT.value if sizes_placement else compute_kernel
    )

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        if sizes_placement is None:
            size_and_offsets = _get_parameter_size_offsets(
                param,
                ShardingType.ROW_WISE,
                local_size,
                world_size,
            )
            assert len(size_and_offsets) <= world_size
            size_offset_ranks = []
            for (size, offset), rank in zip(size_and_offsets, range(world_size)):
                size_offset_ranks.append((size, offset, rank))
        else:
            size_offset_ranks = []
            sizes = sizes_placement[0]
            (rows, cols) = param.shape
            cur_offset = 0
            prev_offset = 0
            for rank, size in enumerate(sizes):
                per_rank_row = size
                cur_offset += per_rank_row
                cur_offset = min(cur_offset, rows)
                per_rank_row = cur_offset - prev_offset
                size_offset_ranks.append(([per_rank_row, cols], [prev_offset, 0], rank))
                prev_offset = cur_offset

            if cur_offset < rows:
                raise ValueError(
                    f"Cannot fit tensor of {rows, cols} into sizes_ranks_placements = {sizes_placement}"
                )

        index: int = 0
        placements: List[str] = []
        if sizes_placement is not None:
            device_type = ""
            for i in range(len(sizes_placement[0])):
                if isinstance(sizes_placement[1], list):
                    device_type = sizes_placement[1][i]
                    placements.append(placement_helper(device_type, index, i))
                else:
                    device_type = str(sizes_placement[1])
                    placements.append(placement_helper(device_type, index, i))

                if device_type == "cuda":
                    index += 1

        return _get_parameter_sharding(
            param,
            ShardingType.ROW_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
            placements=placements if sizes_placement else None,
            compute_kernel=compute_kernel,
        )

    return _parameter_sharding_generator


def column_wise(
    ranks: Optional[List[int]] = None,
    size_per_rank: Optional[List[int]] = None,
    compute_kernel: Optional[str] = None,
    device_types: Optional[List[str]] = None,
) -> ParameterShardingGenerator:
    """
    Returns a generator of ParameterShardingPlan for `ShardingType::COLUMN_WISE` for construct_module_sharding_plan.
    Table will be sharded column-wise across specified ranks (and can reuse ranks).

    Args:
        ranks (Optional[List[int]]): Ranks to place columns. Required if size_per_rank is None.
        size_per_rank (Optional[List[int]]): List specifying the number of columns per rank.
            If provided, the columns will be distributed according to these sizes.
        device_types (Optional[List[str]]): List of device types (e.g., "cpu", "cuda", "mtia") for each shard.
            Used to specify different device placements for different shards.

    Returns:
        ParameterShardingGenerator: A function that generates parameter sharding configuration.

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_3": column_wise(ranks=[0,1,2]),
            },
        )
    """

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        """
        Internal function that generates the parameter sharding configuration.

        Args:
            param: The parameter tensor to be sharded.
            local_size: Number of devices in the local process group.
            world_size: Total number of devices across all process groups.
            device_type: Type of device (e.g., "cuda", "cpu", "mtia").
            sharder: The module sharder instance.

        Returns:
            ParameterSharding: The sharding configuration for the parameter.

        Raises:
            ValueError: If the parameter cannot be evenly divided across ranks or
                if the specified sizes cannot fit the tensor.
        """
        if size_per_rank is None:
            assert ranks is not None

            if param.shape[1] % len(ranks) != 0:
                raise ValueError(
                    f"column dim of {param.shape[1]} cannot be evenly divided across {ranks}"
                )
            shard_dim = param.shape[1] // len(ranks)
            size_and_offsets = _get_parameter_size_offsets(
                param,
                ShardingType.COLUMN_WISE,
                local_size,
                world_size,
                col_wise_shard_dim=shard_dim,
            )

            size_offset_ranks = []
            for (size, offset), rank in zip(size_and_offsets, ranks):
                size_offset_ranks.append((size, offset, rank))
        else:
            size_offset_ranks = []
            (rows, cols) = param.shape
            cur_offset = 0
            prev_offset = 0
            for rank, cur_size in enumerate(size_per_rank):
                cur_offset += cur_size
                cur_offset = min(cur_offset, cols)
                cur_cols = cur_offset - prev_offset
                size_offset_ranks.append(([rows, cur_cols], [0, prev_offset], rank))
                prev_offset = cur_offset

            if cur_offset < cols:
                raise ValueError(
                    f"Cannot fit tensor of {rows, cols} into sizes_ranks_placements = {size_per_rank}"
                )

        placements: List[str] = []
        if device_types is not None:
            assert len(device_types) == len(
                size_offset_ranks
            ), "device_types must be the same length as ranks"
            index: int = 0
            for device in device_types:
                placements.append(placement_helper(device, index, index))
                if device != "cpu":
                    index += 1

        return _get_parameter_sharding(
            param,
            ShardingType.COLUMN_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
            placements=placements if placements else None,
            compute_kernel=compute_kernel,
        )

    return _parameter_sharding_generator


def table_row_wise(
    host_index: int,
) -> ParameterShardingGenerator:
    """
    Returns a generator of ParameterShardingPlan for `ShardingType::TABLE_ROW_WISE` for construct_module_sharding_plan.

    Args:
    host_index (int): index of host (node) to do row wise

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_4": table_row_wise(host_index=2),
            },
        )
    """

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.TABLE_ROW_WISE,
            local_size,
            world_size,
        )

        size_offset_ranks = []
        assert len(size_and_offsets) <= local_size
        for (size, offset), rank in zip(size_and_offsets, range(local_size)):
            rank_offset = host_index * local_size
            size_offset_ranks.append((size, offset, rank_offset + rank))

        return _get_parameter_sharding(
            param,
            ShardingType.TABLE_ROW_WISE.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def grid_shard(
    host_indexes: List[int],
) -> ParameterShardingGenerator:
    """
    Returns a generator of ParameterShardingPlan for `ShardingType::GRID_SHARD` for construct_module_sharding_plan.

    Args:
    host_indexes (List[int]): index of hosts (nodes) to do row wise

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_4": grid_shard(host_indexes=[1,2]),
            },
        )
    """

    def _parameter_sharding_generator(
        param: nn.Parameter,
        local_size: int,
        world_size: int,
        device_type: str,
        sharder: ModuleSharder[nn.Module],
    ) -> ParameterSharding:
        size_and_offsets = _get_parameter_size_offsets(
            param,
            ShardingType.GRID_SHARD,
            local_size,
            world_size,
        )
        size_offset_ranks = []
        for host_count, host_index in enumerate(host_indexes):
            for rank in range(local_size):
                (size, offset) = size_and_offsets[host_count * local_size + rank]
                rank_offset = host_index * local_size
                size_offset_ranks.append((size, offset, rank_offset + rank))

        return _get_parameter_sharding(
            param,
            ShardingType.GRID_SHARD.value,
            size_offset_ranks,
            local_size,
            device_type,
            sharder,
        )

    return _parameter_sharding_generator


def apply_to_all(
    module: nn.Module,
    parameter_sharding_generator: ParameterShardingGenerator,
    sharder: Optional[ModuleSharder[nn.Module]] = None,
) -> Dict[str, ParameterShardingGenerator]:
    """
    Convenience function to apply a sharding scheme generator for all modules in construct_module_sharding_plan.

    Example::

        ebc = EmbeddingBagCollection(...)
        sharder = EmbeddingBagCollectionSharder()
        plan = construct_parameter_sharding_plan(
            ebc,
            apply_to_all(ebc, row_wise(), sharder),
        )
    """
    if sharder is None:
        sharder = get_module_to_default_sharders().get(type(module), None)
    else:
        assert isinstance(
            module, sharder.module_type
        ), f"Incorrect sharder for module type {type(module)}"

    assert (
        sharder is not None
    ), f"Could not find a valid sharder type for {type(module)}"

    shardable_parameters = sharder.shardable_parameters(module)
    return {
        param_name: parameter_sharding_generator for param_name in shardable_parameters
    }


def construct_module_sharding_plan(
    module: nn.Module,
    per_param_sharding: Dict[str, ParameterShardingGenerator],
    sharder: Optional[ModuleSharder[nn.Module]] = None,
    local_size: Optional[int] = None,
    world_size: Optional[int] = None,
    device_type: Optional[str] = None,
) -> EmbeddingModuleShardingPlan:
    """
    Helper function to create module sharding plans (EmbeddingModuleShardingPlan) for an module
    Args:
        module (nn.Module): module to create plan for.
        per_param_sharding: Dict[str, Callable[[nn.Parameter, int, int, str], ParameterSharding]]: A mapping of parameter names to a generator function
        that takes in [parameter, local_size, world_size, device_type] and returns a ParameterSharding. We recommend using one of the predefined generator functions
        e.g. table_wise_sharding, row_wise_sharding, etc,
        sharder: Optional[ModuleSharder[nn.Module]]: Sharder that we are creating a plan for. If set to none, we will try to derive it from the module. We recommend setting this to None.
        local_size: Optional[int] = None: process group local size
        world_size: Optional[int] = None: process_group world_size
        device_type: str : Torch device type,

    Example::

        ebc = EmbeddingBagCollection(...)
        plan = construct_module_sharding_plan(
            ebc,
            {
                "table_0": data_parallel(),
                "table_1": row_wise(),
                "table_2": column_wise(),
                "table_3": column_wise(ranks=[0,1,2]),
                "table_4": table_row_wise(host_index=2),
            },
        )
    """
    if device_type is None:
        device_type = get_device_type()
    if sharder is None:
        sharder = get_module_to_default_sharders().get(type(module), None)
    assert (
        sharder is not None
    ), f"Could not find a valid sharder type for {type(module)}"

    assert isinstance(
        module, sharder.module_type
    ), f"Incorrect sharder {type(sharder)} for module type {type(module)}"
    shardable_parameters = sharder.shardable_parameters(module)
    assert shardable_parameters.keys() == per_param_sharding.keys(), (
        "per_param_sharding_config doesn't match the shardable parameters of the module,"
        f"got {list(shardable_parameters.keys())} != {list(per_param_sharding.keys())}"
    )

    local_size = local_size or get_local_size()
    world_size = world_size or dist.get_world_size()

    per_parameter_sharding = EmbeddingModuleShardingPlan()
    for table_name, sharding_plan_generator in per_param_sharding.items():
        param = shardable_parameters[table_name]
        per_parameter_sharding[table_name] = sharding_plan_generator(
            param, local_size, world_size, device_type, sharder
        )

    return per_parameter_sharding
