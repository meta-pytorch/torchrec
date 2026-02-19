#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import operator
from functools import reduce
from typing import Any, cast, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torchrec.distributed.planner.constants import NUM_POOLINGS
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Perf,
    ShardingOption,
    Storage,
)
from torchrec.distributed.types import CommOp, ModuleSharder, ShardingType


def sharder_name(t: Type[Any]) -> str:
    return t.__module__ + "." + t.__name__


def is_prefetch_pipelined(
    sharding_option: ShardingOption, sharder: ModuleSharder[nn.Module]
) -> bool:
    prefetch_pipeline = (
        sharding_option.cache_params.prefetch_pipeline
        if sharding_option.cache_params
        else None
    )
    # TODO: remove after deprecating fused_params in sharder
    if not prefetch_pipeline:
        prefetch_pipeline = (
            sharder.fused_params.get("prefetch_pipeline", False)  # pyre-ignore[16]
            if hasattr(sharder, "fused_params") and sharder.fused_params
            else False
        )
    return prefetch_pipeline


def extract_comm_data_type_size(
    sharder: ModuleSharder[nn.Module], sharding_option: ShardingOption
) -> Tuple[float, float, float, float]:
    table_data_type_size = sharding_option.tensor.element_size()

    fwd_a2a_comm_data_type_size = table_data_type_size
    bwd_a2a_comm_data_type_size = table_data_type_size
    fwd_sr_comm_data_type_size = table_data_type_size
    bwd_sr_comm_data_type_size = table_data_type_size

    if sharder.qcomm_codecs_registry is not None:
        qcomm_codecs_registry = sharder.qcomm_codecs_registry
        if (
            sharding_option.is_pooled
            and CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name in qcomm_codecs_registry
        ):
            codecs = sharder.qcomm_codecs_registry[
                CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name
            ]
            fwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.forward.quantized_dtype
            ).element_size()
            bwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.backward.quantized_dtype
            ).element_size()

        if (
            not sharding_option.is_pooled
            and CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name in qcomm_codecs_registry
        ):
            codecs = qcomm_codecs_registry[CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name]
            fwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.forward.quantized_dtype
            ).element_size()
            bwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.backward.quantized_dtype
            ).element_size()

        if (
            sharding_option.is_pooled
            and CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name in qcomm_codecs_registry
        ):
            codecs = qcomm_codecs_registry[CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name]
            fwd_sr_comm_data_type_size = torch.tensor(
                [], dtype=codecs.forward.quantized_dtype
            ).element_size()
            bwd_sr_comm_data_type_size = torch.tensor(
                [], dtype=codecs.backward.quantized_dtype
            ).element_size()

    return (
        fwd_a2a_comm_data_type_size,
        bwd_a2a_comm_data_type_size,
        fwd_sr_comm_data_type_size,
        bwd_sr_comm_data_type_size,
    )


def get_num_poolings(
    constraints: Optional[Dict[str, ParameterConstraints]], so: ShardingOption
) -> List[float]:
    # first priority is given for sharding_option.num_poolings,
    # otherwise Manifold planner configs will be overwritten by parameter constraints
    # default path will use constraints
    if so.num_poolings is not None:
        num_poolings = so.num_poolings
        if len(so.input_lengths) == len(num_poolings):
            return num_poolings

    # Second priority: use constraint-based num_poolings
    if constraints and constraints.get(so.name) and constraints[so.name].num_poolings:
        return cast(List[float], constraints[so.name].num_poolings)

    # Fallback: use default NUM_POOLINGS constant
    return [NUM_POOLINGS] * len(so.input_lengths)


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes / (1024 * 1024 * 1024))


def bytes_to_mb(num_bytes: Union[float, int]) -> float:
    return float(num_bytes / (1024 * 1024))


def gb_to_bytes(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)


def mb_to_bytes(mb: float) -> int:
    return int(mb * 1024 * 1024)


def prod(iterable: Iterable[int]) -> int:
    return reduce(operator.mul, iterable, 1)


def placement(
    compute_device: str,
    rank: int,
    local_size: int,
) -> str:
    """
    Returns placement, formatted as string
    """

    param_device = compute_device
    if compute_device in {"cuda", "mtia"}:
        param_device = torch.device(compute_device, rank % local_size)
    return f"rank:{rank}/{param_device}"


def storage_repr_in_gb(storage: Optional[Storage]) -> str:
    if storage is None:
        return ""
    return (
        f"Storage(hbm = {round(bytes_to_gb(storage.hbm), 3)} GB, "
        f"ddr = {round(bytes_to_gb(storage.ddr), 3)} GB)"
    )


def reset_shard_rank(proposal: List[ShardingOption]) -> None:
    for sharding_option in proposal:
        for shard in sharding_option.shards:
            shard.rank = None


def _find_imbalance_tables(
    sharding_options: List[ShardingOption], target_imbalance: str = "perf"
) -> List[ShardingOption]:
    """
    Find the tables that are causing the imbalance, and return their names.
    """
    rank_to_target_stats: Dict[int, float] = {}

    # populate rank_to_target_stats
    for sharding_option in sharding_options:
        for shard in sharding_option.shards:
            rank = cast(int, shard.rank)
            if rank not in rank_to_target_stats:
                rank_to_target_stats[rank] = 0

            if target_imbalance == "perf":
                rank_to_target_stats[rank] += cast(Perf, shard.perf).total
            elif target_imbalance == "hbm":
                rank_to_target_stats[rank] += cast(Storage, shard.storage).hbm
            else:
                raise ValueError(f"Unknown target imbalance {target_imbalance}")

    if len(rank_to_target_stats.values()) <= 1:
        # world_size is 1
        return []

    max_value = max(rank_to_target_stats.values())
    max_value_ranks = {
        rank for rank, value in rank_to_target_stats.items() if value == max_value
    }

    # find tables
    tables_in_max_value_ranks: List[ShardingOption] = []
    for sharding_option in sharding_options:
        sharding_option_ranks = [shard.rank for shard in sharding_option.shards]
        if set(
            sharding_option_ranks
        ) >= max_value_ranks and sharding_option.sharding_type not in [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.ROW_WISE.value,
        ]:
            tables_in_max_value_ranks.append(sharding_option)

    if target_imbalance == "perf":
        # sort tables by total perf from largest to smallest
        tables_in_max_value_ranks.sort(
            key=lambda sharding_option: sharding_option.shards[0].perf.total,
            reverse=True,
        )
    elif target_imbalance == "hbm":
        # sort tables by hbm from largest to smallest
        tables_in_max_value_ranks.sort(
            key=lambda sharding_option: sharding_option.shards[0].storage.hbm,
            reverse=True,
        )
    else:
        raise ValueError(f"Unknown target imbalance {target_imbalance}")

    return tables_in_max_value_ranks


class BinarySearchPredicate:
    """Generates values of X between A & B to invoke on an external predicate F(X) to
    discover the largest X for which F(X) is true. Uses binary search to minimize the
    number of invocations of F. Assumes F is a step function, i.e. if F(X) is false,
    there is no point trying F(X+1)."""

    def __init__(self, A: int, B: int, tolerance: int) -> None:
        """A = lower boundary (inclusive)
        B = upper boundary (inclusive)
        tolerance = stop search early if remaining search range is less than tolerance
        """
        self.left = A
        self.right = B
        self.tolerance = tolerance
        self.first = True

    def next(self, prior_result: bool) -> Optional[int]:
        """next() returns the next value to probe, given the result of the prior probe.
        The first time next() is invoked the prior_result is ignored. Returns None if
        entire range explored or threshold reached."""
        if self.right - self.left < self.tolerance:
            return None

        mid = self._mid()
        if self.first:
            self.first = False
            return mid

        if prior_result:
            self.left = mid + 1
        else:
            self.right = mid - 1
        if self.right - self.left < self.tolerance:
            return None

        return self._mid()

    def _mid(self) -> int:
        return self.left + ((self.right - self.left) // 2)


class LuusJaakolaSearch:
    """Implements a clamped variant of Luus Jaakola search.

    See https://en.wikipedia.org/wiki/Luus-Jaakola.
    """

    def __init__(
        self,
        A: float,
        B: float,
        max_iterations: int,
        seed: int = 42,
        left_cost: Optional[float] = None,
    ) -> None:
        self.left = A
        self.right = B
        self.iteration = -1
        self.max_iterations = max_iterations

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        self.x: float = self.uniform(self.left, self.right)
        self.fx: float = 0.0
        self.y: float = math.nan
        self.fleft: Optional[float] = left_cost
        self.fright: Optional[float] = None
        self.d: float = self.right - self.left

    def shrink_right(self, B: float) -> None:
        "Shrink right boundary given [B,infinity) -> infinity"
        self.right = B
        self.fright = math.inf
        self.d = self.right - self.left
        self.x = self.clamp(self.x)

    def clamp(self, x: float) -> float:
        "Clamp x into range [left, right]"
        if x < self.left:
            return self.left
        if x > self.right:
            return self.right
        return x

    def uniform(self, A: float, B: float) -> float:
        "Return a random uniform position in range [A,B]."
        u = torch.rand(1, generator=self.gen, device="cpu").item()
        return A + (B - A) * u

    def next(self, fy: float) -> Optional[float]:
        """Return the next probe point 'y' to evaluate, given the previous result.

        The first time around fy is ignored. Subsequent invocations should provide the
        result of evaluating the function being minimized, i.e. f(y).

        Returns None when the maximum number of iterations has been reached.
        """
        self.iteration += 1
        if self.iteration == 0:
            return self.x
        elif self.iteration == 1:
            self.fx = fy
        elif self.iteration == self.max_iterations:
            return None
        elif fy <= self.fx:
            self.x = self.y
            self.fx = fy
            self.d = 0.95 * self.d

        if self.y == self.left:
            self.fleft = fy
        elif self.y == self.right:
            self.fright = fy

        while True:
            a = self.uniform(-self.d, self.d)
            y = self.clamp(self.x + a)
            # Unlike standard Luus-Jaakola, we don't want to explore outside of our bounds.
            # Clamping can cause us to explore the boundary multiple times, so we
            # remember if we already know the boundary cost and request a new sample if
            # we do.
            if y == self.left and self.fleft is not None:
                continue
            if y == self.right and self.fright is not None:
                continue
            self.y = y
            return self.y

    def best(self) -> Tuple[float, float]:
        "Return the best position so far, and its associated cost."
        return self.x, self.fx
