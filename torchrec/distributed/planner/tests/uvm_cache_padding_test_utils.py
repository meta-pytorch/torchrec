#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Iterator, List
from unittest.mock import patch

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    Partitioner,
    PlanLoader,
    Proposer,
    Shard,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.types import CacheParams, DataType, ShardingType
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TWUVMCachingSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED_UVM_CACHING.value]


class RWUVMCachingSharder(TWUVMCachingSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]


class CWUVMCachingSharder(TWUVMCachingSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.COLUMN_WISE.value]


class TWUVMCachingAndUVMSharder(TWUVMCachingSharder):
    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            EmbeddingComputeKernel.FUSED_UVM.value,
        ]


def enable_uvm_cache_planner_knobs(test: unittest.TestCase) -> None:
    """Turn on the knobs that gate UVM cache precision and padding estimates."""
    for target in (
        "torchrec.distributed.planner.planners._uvm_cache_dimension_padding_jk_enabled",
        "torchrec.distributed.planner.shard_estimators._cache_precision_hbm_estimate_enabled",
    ):
        patcher = patch(target, return_value=True)
        patcher.start()
        test.addCleanup(patcher.stop)


@contextmanager
def uvm_cache_padding_knob(enabled: bool) -> Iterator[None]:
    """Set the UVM cache dimension padding knob for the planner calls inside."""
    with patch(
        "torchrec.distributed.planner.planners._uvm_cache_dimension_padding_jk_enabled",
        return_value=enabled,
    ):
        yield


class FixedRankPartitioner(Partitioner):
    def __init__(self, ranks_by_table: dict[str, list[int]]) -> None:
        self._ranks_by_table = ranks_by_table

    def partition(
        self,
        proposal: List[ShardingOption],
        storage_constraint: Topology,
    ) -> List[ShardingOption]:
        for option in proposal:
            ranks = self._ranks_by_table[option.name]
            if len(ranks) != len(option.shards):
                raise AssertionError(
                    f"Expected {len(option.shards)} ranks for {option.name}, "
                    f"got {len(ranks)}"
                )
            for shard, rank in zip(option.shards, ranks):
                shard.rank = rank
        return proposal


class KernelSequenceProposer(Proposer):
    def __init__(self, selections: list[list[tuple[str, str]]]) -> None:
        self._selections = selections
        self._proposal_index = 0
        self.proposals: list[list[ShardingOption]] = []
        self.partitionable_feedback: list[bool] = []

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Enumerator | None = None,
    ) -> None:
        options = {
            (option.name, option.compute_kernel): option for option in search_space
        }
        self.proposals = [
            [options[key] for key in selection] for selection in self._selections
        ]
        self._proposal_index = 0
        self.partitionable_feedback = []

    def propose(self) -> List[ShardingOption] | None:
        if self._proposal_index >= len(self.proposals):
            return None
        proposal = self.proposals[self._proposal_index]
        self._proposal_index += 1
        return proposal

    def feedback(
        self,
        partitionable: bool,
        plan: List[ShardingOption] | None = None,
        perf_rating: float | None = None,
        storage_constraint: Topology | None = None,
    ) -> None:
        self.partitionable_feedback.append(partitionable)


def _mixed_dimension_tables(
    dimensions: list[tuple[str, int]],
    *,
    data_type: DataType = DataType.FP32,
) -> list[EmbeddingBagConfig]:
    return [
        EmbeddingBagConfig(
            num_embeddings=1000,
            embedding_dim=embedding_dim,
            name=name,
            feature_names=[f"feature_{name}"],
            data_type=data_type,
        )
        for name, embedding_dim in dimensions
    ]


def _uvm_cache_constraints(
    tables: list[EmbeddingBagConfig],
    sharding_type: str,
    *,
    compute_kernels: list[str] | None = None,
    min_partitions: dict[str, int] | None = None,
    cache_precision: DataType | None = None,
) -> dict[str, ParameterConstraints]:
    kernels = compute_kernels or [EmbeddingComputeKernel.FUSED_UVM_CACHING.value]
    partitions = min_partitions or {}
    return {
        table.name: ParameterConstraints(
            sharding_types=[sharding_type],
            compute_kernels=kernels,
            min_partition=partitions.get(table.name),
            cache_params=CacheParams(
                load_factor=0.5,
                precision=cache_precision,
            ),
        )
        for table in tables
    }


def _loaded_options_without_cache_metadata(
    selected_options: list[ShardingOption],
) -> dict[int, ShardingOption]:
    loaded_options = {}
    for option in selected_options:
        loaded_shards = []
        for shard in option.shards:
            storage = none_throws(shard.storage)
            loaded_shards.append(
                Shard(
                    size=list(shard.size),
                    offset=list(shard.offset),
                    storage=Storage(
                        hbm=storage.hbm,
                        ddr=storage.ddr,
                        ssd=storage.ssd,
                    ),
                    perf=shard.perf,
                    rank=shard.rank,
                )
            )
        loaded_options[option.storage_hash()] = ShardingOption(
            name=option.name,
            tensor=option.tensor,
            module=option.module,
            input_lengths=option.input_lengths,
            batch_size=option.batch_size,
            compute_kernel=option.compute_kernel,
            sharding_type=option.sharding_type,
            partition_by=option.partition_by,
            shards=loaded_shards,
            cache_params=option.cache_params,
            enforce_hbm=option.enforce_hbm,
            feature_names=option.feature_names,
        )
    return loaded_options


def _total_hbm(options: list[ShardingOption]) -> int:
    return sum(
        none_throws(shard.storage).hbm for option in options for shard in option.shards
    )


def _hbm_by_table(options: list[ShardingOption]) -> dict[str, list[int]]:
    return {
        option.name: [none_throws(shard.storage).hbm for shard in option.shards]
        for option in options
    }


ShardKey = tuple[str, tuple[int, ...], tuple[int, ...], int | None]


def _shard_key(table_name: str, shard: Shard) -> ShardKey:
    return table_name, tuple(shard.size), tuple(shard.offset), shard.rank


def _hbm_by_shard(options: list[ShardingOption]) -> dict[ShardKey, int]:
    return {
        _shard_key(option.name, shard): none_throws(shard.storage).hbm
        for option in options
        for shard in option.shards
    }


def _hbm_delta_by_shard(
    padded: list[ShardingOption],
    unpadded: list[ShardingOption],
) -> dict[ShardKey, int]:
    padded_hbm = _hbm_by_shard(padded)
    unpadded_hbm = _hbm_by_shard(unpadded)
    if padded_hbm.keys() != unpadded_hbm.keys():
        raise AssertionError("Padded and unpadded plans have different shard layouts")
    return {key: value - unpadded_hbm[key] for key, value in padded_hbm.items()}


class MockPlanLoader(PlanLoader):
    def __init__(
        self,
        loaded_sharding_options: dict[int, ShardingOption] | None = None,
        context_hash: str | None = None,
        plan_id: str = "test_plan_123",
    ) -> None:
        self._loaded_sharding_options = loaded_sharding_options
        self._context_hash = context_hash
        self._plan_id = plan_id

    def load(self) -> dict[int, ShardingOption] | None:
        return self._loaded_sharding_options

    def plan_context_hash(self) -> str | None:
        return self._context_hash

    def get_plan_id(self) -> str:
        return self._plan_id
