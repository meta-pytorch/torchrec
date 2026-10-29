#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
End-to-end test for the KVZCH virtual-table enrichment + EmbeddingCollection
write API round trip, against the IN_MEMORY_TEST_ONLY enrichment backend.

In KVZCH+enrichment mode, EmbeddingCollection.write() is really a request to
populate the cache from the enrichment provider:
- KJT.values() = hashed cache keys
- KJT.weights() = unhashed object IDs, bit-cast to float32 in pairs (each int64
  becomes 2 float32 columns), shape [N, 2]

Calling .write() dispatches an async fetch to the enrichment provider with the
unhashed IDs; the returned embeddings land in the cache keyed by the hashed
IDs. A subsequent forward() with the hashed IDs returns those embeddings.

The IN_MEMORY_TEST_ONLY backend lives at
fbcode/deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/fake_enrichment.h
and returns embedding[id, slot] = float(id) + slot * 1e-3.

The unit test in test_embedding_update.py covers EmbeddingCollection.write()
against a plain DRAM_VIRTUAL_TABLE (no enrichment). This file extends coverage
to the enrichment-enabled path.
"""

import os
import time
import unittest
from typing import Any, cast, List, Optional
from unittest import mock

import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.ssd.ssd_config import (
    EnrichmentPolicy,
    EnrichmentResponseFormat,
    EnrichmentType,
    KVZCHTBEConfig,
)
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.batched_embedding_kernel import (
    _PinnedHostCacheTrimmer,
    ZeroCollisionEmbeddingEnrichmentCache,
    ZeroCollisionKeyValueEmbedding,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    EmbeddingCollectionSharder,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingConfig,
    TimestampBasedEvictionPolicy,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# Embedding dim small enough for fast tests; large enough that the deterministic
# formula gives distinguishable per-slot values within fp16/bf16 precision.
EMBEDDING_DIM = 8
# At 8 dims the fake rows span fewer fp16 ulps than INT4 has levels, so the
# serving round trip leaves them unchanged; 128 dims (the CU cache width) moves them.
SERVING_QDQ_DIM = 128
NUM_EMBEDDINGS = 8000
TOTAL_NUM_BUCKETS = 20
FEATURE_NAME = "feature_0"
PROVIDER_NAME = "test_in_memory_test_only"
CLIENT_ID = "test_kvzch_enrichment_e2e"
_DIRECT_BACKEND_GATHER_POLICY_FIELDS = ("direct_backend_gather",)
_OWNER_LOCAL_CACHE_POLICY_FIELDS = (
    "direct_backend_gather",
    "owner_cache_size_mb",
    "owner_cache_read_populate",
    "owner_cache_read_populate_max_lookups",
)
_SERVING_QDQ_POLICY_FIELDS = ("serving_qdq_bit_rate",)


def _expected_fake_value(id_value: int, slot: int) -> float:
    """Mirror of fake_enrichment::fakeEmbeddingValue in C++."""
    return float(id_value) + slot * 1e-3


def _expected_fake_row(id_value: int, dim: int) -> torch.Tensor:
    return torch.tensor(
        [_expected_fake_value(id_value, j) for j in range(dim)],
        dtype=torch.float32,
    )


class TestECModel(nn.Module):
    def __init__(self, tables: List[EmbeddingConfig], device: torch.device) -> None:
        super().__init__()
        self.ec = EmbeddingCollection(tables=tables, device=device)

    def forward(self, features: KeyedJaggedTensor) -> dict:  # pyre-ignore[3]
        return self.ec(features)


def _build_input_kjt(ids: List[int], device: torch.device) -> KeyedJaggedTensor:
    return KeyedJaggedTensor.from_lengths_sync(
        keys=[FEATURE_NAME],
        values=torch.tensor(ids, dtype=torch.int64, device=device),
        lengths=torch.tensor([1] * len(ids), dtype=torch.int32, device=device),
    )


def _read_rows(
    sharded_model: DistributedModelParallel, kjt: KeyedJaggedTensor
) -> torch.Tensor:
    """Run the model forward and flatten the per-feature output to [N, dim]."""
    return _read_rows_by_feature(sharded_model, kjt, [FEATURE_NAME])[0]


def _read_rows_by_feature(
    sharded_model: DistributedModelParallel,
    kjt: KeyedJaggedTensor,
    feature_names: list[str],
) -> list[torch.Tensor]:
    """Run forward and flatten each requested feature output to [N, dim]."""
    out = sharded_model(kjt)
    rows = []
    for feature_name in feature_names:
        values = out[feature_name].to_dense()
        rows.append(torch.cat(list(values)) if isinstance(values, list) else values)
    return rows


def _wait_for_cache_fill(
    sharded_model: DistributedModelParallel,
    kjt: KeyedJaggedTensor,
    num_ids: int,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
) -> torch.Tensor:
    """Poll-read until every row is non-zero or timeout.

    The enrichment fetch dispatched by .write() runs on a background executor
    in the dram_kv backend, so we cannot synchronously wait for it. Polling
    via repeated forward() is how production training observes the fill.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        rows = _read_rows(sharded_model, kjt)
        nonzero_per_row = rows.float().abs().sum(dim=1) > 1e-6
        if nonzero_per_row.sum().item() == num_ids:
            return rows
        if time.monotonic() > deadline:
            raise AssertionError(
                f"Enrichment cache did not fill within {timeout_s}s; "
                f"{int(nonzero_per_row.sum().item())}/{num_ids} rows non-zero"
            )
        time.sleep(poll_interval_s)


def _wait_for_multi_feature_cache_fill(
    sharded_model: DistributedModelParallel,
    kjt: KeyedJaggedTensor,
    feature_names: list[str],
    timeout_s: float = 30.0,
) -> list[torch.Tensor]:
    """Poll until every row in every lookup group has been enriched."""
    deadline = time.monotonic() + timeout_s
    while True:
        rows_by_feature = _read_rows_by_feature(sharded_model, kjt, feature_names)
        if all(
            bool(torch.all(rows.float().abs().sum(dim=1) > 1e-6).item())
            for rows in rows_by_feature
        ):
            return rows_by_feature
        if time.monotonic() > deadline:
            populated = [
                int((rows.float().abs().sum(dim=1) > 1e-6).sum().item())
                for rows in rows_by_feature
            ]
            raise AssertionError(
                f"Enrichment cache did not fill within {timeout_s}s; "
                f"populated rows by feature: {populated}"
            )
        time.sleep(0.5)


def _build_write_kjt(
    hashed_ids: List[int], unhashed_ids: List[int], device: torch.device
) -> KeyedJaggedTensor:
    """Build the KJT consumed by EC.write() in KVZCH+enrichment mode.

    Mirrors minimal_viable_ai/core/utils/kvzch_utils.build_embedding_cache_write_kjt:
    weights are the unhashed IDs bit-cast to float32 pairs, shape [N, 2].
    """
    assert len(hashed_ids) == len(unhashed_ids)
    # Citrine C3: construct test inputs directly on their target CUDA device.
    unhashed_tensor = torch.tensor(unhashed_ids, dtype=torch.int64, device=device)
    # Lossless reinterpret_cast: each int64 -> two float32 columns.
    weights = unhashed_tensor.view(torch.float32).reshape(-1, 2)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=[FEATURE_NAME],
        values=torch.tensor(hashed_ids, dtype=torch.int64, device=device),
        lengths=torch.tensor([1] * len(hashed_ids), dtype=torch.int32, device=device),
        weights=weights,
    )


def _build_multi_feature_kjt(
    ids_by_feature: list[list[int]],
    device: torch.device,
    unhashed_ids_by_feature: Optional[list[list[int]]] = None,
) -> KeyedJaggedTensor:
    """Build a feature-major KJT for multiple singleton lookup groups."""
    feature_names = [f"feature_{index}" for index in range(len(ids_by_feature))]
    flattened_ids = [value for ids in ids_by_feature for value in ids]
    lengths = torch.ones(len(flattened_ids), dtype=torch.int32, device=device)
    weights = None
    if unhashed_ids_by_feature is not None:
        assert len(unhashed_ids_by_feature) == len(ids_by_feature)
        flattened_unhashed_ids = [
            value for ids in unhashed_ids_by_feature for value in ids
        ]
        assert len(flattened_unhashed_ids) == len(flattened_ids)
        # Citrine C3: construct test inputs directly on their target CUDA device.
        weights = torch.tensor(
            flattened_unhashed_ids, dtype=torch.int64, device=device
        ).view(torch.float32)
        weights = weights.reshape(-1, 2)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=feature_names,
        values=torch.tensor(flattened_ids, dtype=torch.int64, device=device),
        lengths=lengths,
        weights=weights,
    )


def _build_sharded_model(
    ctx: MultiProcessContext,
    world_size: int,
    local_size: Optional[int],
    direct_backend_gather: bool = False,
    owner_local_cache: bool = False,
    owner_local_cache_read_populate: bool = False,
    owner_local_cache_read_populate_max_lookups: Optional[int] = None,
    prefetch_pipeline: bool = False,
    num_lookup_groups: int = 1,
    serving_qdq_bit_rate: int = 0,
    embedding_dim: int = EMBEDDING_DIM,
    data_type: DataType = DataType.FP32,
) -> DistributedModelParallel:
    """Build row-wise virtual tables backed by the test enrichment provider."""
    tables = [
        EmbeddingConfig(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=embedding_dim,
            name=f"table_{index}",
            data_type=data_type,
            feature_names=[f"feature_{index}"],
            total_num_buckets=TOTAL_NUM_BUCKETS,
            use_virtual_table=True,
            enable_embedding_update=True,
            virtual_table_eviction_policy=TimestampBasedEvictionPolicy(
                training_id_eviction_trigger_count=10_000_000,
                eviction_ttl_mins=60,
            ),
        )
        for index in range(num_lookup_groups)
    ]
    model = TestECModel(tables=tables, device=ctx.device)
    enrichment_policy = EnrichmentPolicy(
        enrichment_type=EnrichmentType.IN_MEMORY_TEST_ONLY,
        provider_name=PROVIDER_NAME,
        client_id=CLIENT_ID,
        enrichment_dim=embedding_dim,
        response_format=EnrichmentResponseFormat.THRIFT_FLOAT,
    )
    if serving_qdq_bit_rate:
        enrichment_policy = enrichment_policy._replace(
            serving_qdq_bit_rate=serving_qdq_bit_rate
        )
    if direct_backend_gather or owner_local_cache:
        enrichment_policy = enrichment_policy._replace(direct_backend_gather=True)
    if owner_local_cache:
        enrichment_policy = enrichment_policy._replace(owner_cache_size_mb=1)
    if owner_local_cache_read_populate:
        enrichment_policy = enrichment_policy._replace(owner_cache_read_populate=True)
    if owner_local_cache_read_populate_max_lookups is not None:
        enrichment_policy = enrichment_policy._replace(
            owner_cache_read_populate_max_lookups=(
                owner_local_cache_read_populate_max_lookups
            )
        )
    fused_params = {
        "embedding_cache_mode": True,
        "l2_cache_size": 1,
        "prefetch_pipeline": prefetch_pipeline,
        "use_one_tbe_per_table": num_lookup_groups > 1,
        "output_dtype": (
            SparseType.BF16
            if direct_backend_gather or owner_local_cache
            else SparseType.FP32
        ),
        "kvzch_tbe_config": KVZCHTBEConfig(enrichment_policy=enrichment_policy),
    }
    sharder = EmbeddingCollectionSharder(fused_params=fused_params)
    kvzch_sharding_plan = construct_module_sharding_plan(
        model.ec,
        per_param_sharding={
            table.name: row_wise(
                compute_kernel=EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value
            )
            for table in tables
        },
        local_size=local_size,
        world_size=world_size,
        device_type=ctx.device.type,
        # pyrefly: ignore[bad-argument-type]
        sharder=sharder,
    )
    set_propogate_device(True)
    pg = ctx.pg
    assert pg is not None
    return DistributedModelParallel(
        model,
        env=ShardingEnv.from_process_group(pg),
        plan=ShardingPlan({"ec": kvzch_sharding_plan}),
        # pyrefly: ignore[bad-argument-type]
        sharders=[sharder],
        device=ctx.device,
    )


def _run_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
    direct_backend_gather: bool = False,
    owner_local_cache: bool = False,
    owner_local_cache_read_populate: bool = False,
    owner_local_cache_read_populate_max_lookups: Optional[int] = None,
) -> None:
    """Verify enrichment writes and subsequent reads through the selected path."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        sharded_model = _build_sharded_model(
            ctx,
            world_size,
            local_size,
            direct_backend_gather=direct_backend_gather,
            owner_local_cache=owner_local_cache,
            owner_local_cache_read_populate=owner_local_cache_read_populate,
            owner_local_cache_read_populate_max_lookups=(
                owner_local_cache_read_populate_max_lookups
            ),
        )

        hashed_ids = [101, 202, 303, 404]
        unhashed_ids = [1010, 2020, 3030, 4040]
        read_kjt = _build_input_kjt(hashed_ids, ctx.device)

        cold_rows = _read_rows(sharded_model, read_kjt)
        cold_sum = cold_rows.float().abs().sum().item()
        assert (
            cold_sum < 1e-6
        ), f"Expected cold read to return zeros; got nonzero sum {cold_sum}"

        write_kjt = _build_write_kjt(hashed_ids, unhashed_ids, ctx.device)
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(write_kjt)
        torch.cuda.synchronize()

        filled_rows = _wait_for_cache_fill(sharded_model, read_kjt, len(hashed_ids))
        if owner_local_cache and not owner_local_cache_read_populate:
            # The next write cadence publishes the prior completed Laser rows.
            sharded_model.write(write_kjt)
            torch.cuda.synchronize()
        if direct_backend_gather or owner_local_cache:
            assert filled_rows.dtype == torch.bfloat16

        expected = torch.stack(
            [_expected_fake_row(uid, EMBEDDING_DIM) for uid in unhashed_ids]
        )
        torch.testing.assert_close(
            filled_rows.float().cpu(),
            expected,
            rtol=1e-2,
            atol=1e-2,
        )
        if owner_local_cache:
            cached_rows = _read_rows(sharded_model, read_kjt)
            torch.testing.assert_close(
                cached_rows.float().cpu(),
                expected,
                rtol=1e-2,
                atol=1e-2,
            )
            owner_cache = _owner_local_cache(sharded_model)
            hits, queries = owner_cache.last_hit_counts()
            if rank == 0:
                assert queries > 0
            if owner_local_cache_read_populate_max_lookups is None:
                assert hits == queries
            else:
                assert hits == 0
                assert isinstance(owner_cache._read_populate_lookup_count, torch.Tensor)
                assert owner_cache._read_populate_lookup_count.device == ctx.device


def _run_direct_backend_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    _run_round_trip(
        rank,
        world_size,
        backend,
        local_size,
        direct_backend_gather=True,
    )


def _run_direct_backend_prefetch_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        sharded_model = _build_sharded_model(
            ctx,
            world_size,
            local_size,
            direct_backend_gather=True,
            prefetch_pipeline=True,
            num_lookup_groups=2,
        )

        feature_names = ["feature_0", "feature_1"]
        hashed_ids_by_feature = [
            [101, 202, 303, 404],
            [501, 602, 703, 804],
        ]
        unhashed_ids_by_feature = [
            [1010, 2020, 3030, 4040],
            [5010, 6020, 7030, 8040],
        ]
        read_kjt = _build_multi_feature_kjt(hashed_ids_by_feature, ctx.device)
        write_kjt = _build_multi_feature_kjt(
            hashed_ids_by_feature,
            ctx.device,
            unhashed_ids_by_feature,
        )
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(write_kjt)
        torch.cuda.synchronize()
        filled_rows = _wait_for_multi_feature_cache_fill(
            sharded_model, read_kjt, feature_names
        )
        expected_rows = [
            torch.stack(
                [_expected_fake_row(uid, EMBEDDING_DIM) for uid in unhashed_ids]
            )
            for unhashed_ids in unhashed_ids_by_feature
        ]
        for rows, expected in zip(filled_rows, expected_rows):
            torch.testing.assert_close(
                rows.float().cpu(), expected, rtol=1e-2, atol=1e-2
            )

        sharded_ec = cast(Any, sharded_model.module.ec)
        grouped_lookup = sharded_ec._lookups[0]
        enrichment_caches = [
            emb_module
            for emb_module in grouped_lookup._emb_modules
            if getattr(emb_module, "direct_backend_gather_enabled", False)
        ]
        assert len(enrichment_caches) == 2
        tbes = [enrichment_cache.emb_module for enrichment_cache in enrichment_caches]
        assert all(tbe.prefetch_pipeline for tbe in tbes)
        initial_steps = [tbe.step for tbe in tbes]
        generic_queue_sizes = [
            (len(tbe.timesteps_prefetched), len(tbe.ssd_prefetch_data)) for tbe in tbes
        ]
        locking_counters = [tbe.lxu_cache_locking_counter.clone() for tbe in tbes]
        assert all(counter.numel() > 0 for counter in locking_counters)

        reversed_read_kjt = _build_multi_feature_kjt(
            [list(reversed(ids)) for ids in hashed_ids_by_feature], ctx.device
        )
        reversed_expected_rows = [
            torch.flip(expected, dims=(0,)) for expected in expected_rows
        ]
        read_batches = [read_kjt, reversed_read_kjt]
        expected_batches = [expected_rows, reversed_expected_rows]
        prefetch_stream = torch.cuda.Stream(device=ctx.device)
        consumer_stream = torch.cuda.Stream(device=ctx.device)
        queued_outputs: list[list[torch.Tensor]] = []
        queued_expected: list[list[torch.Tensor]] = []
        for cycle in range(4):
            batch_index = cycle % len(read_batches)
            with torch.cuda.stream(prefetch_stream):
                grouped_lookup.prefetch(
                    read_batches[batch_index], forward_stream=consumer_stream
                )
            assert all(
                cache.emb_module._prefetched_direct_backend_output is not None
                for cache in enrichment_caches
            )
            with torch.cuda.stream(consumer_stream):
                consumer_stream.wait_stream(prefetch_stream)
                torch.cuda._sleep(50_000_000)
                queued_outputs.append(
                    _read_rows_by_feature(
                        sharded_model, read_batches[batch_index], feature_names
                    )
                )
            assert all(
                cache.emb_module._prefetched_direct_backend_output is None
                for cache in enrichment_caches
            )
            queued_expected.append(expected_batches[batch_index])

        consumer_done = torch.cuda.Event()
        consumer_done.record(consumer_stream)
        consumer_done.synchronize()
        for actual_by_feature, expected_by_feature in zip(
            queued_outputs, queued_expected
        ):
            for actual, expected in zip(actual_by_feature, expected_by_feature):
                torch.testing.assert_close(
                    actual.float().cpu(), expected, rtol=1e-2, atol=1e-2
                )
        assert generic_queue_sizes == [
            (len(tbe.timesteps_prefetched), len(tbe.ssd_prefetch_data)) for tbe in tbes
        ]
        for tbe, locking_counter in zip(tbes, locking_counters):
            torch.testing.assert_close(
                tbe.lxu_cache_locking_counter,
                locking_counter,
            )
        actual_steps = [tbe.step for tbe in tbes]
        expected_steps = [step + 4 for step in initial_steps]
        assert actual_steps == expected_steps, (initial_steps, actual_steps)

        # A prefetched result belongs only to the exact values storage prepared
        # for it. Leaving a pipeline must not make the next plain forward return
        # rows (or a row count) from the pipeline's look-ahead batch.
        fallback_read_kjt = _build_multi_feature_kjt(
            [ids[:2] for ids in hashed_ids_by_feature], ctx.device
        )
        fallback_expected = [expected[:2] for expected in expected_rows]
        before_stale_fallback = [tbe.step for tbe in tbes]
        with torch.cuda.stream(prefetch_stream):
            grouped_lookup.prefetch(read_kjt, forward_stream=consumer_stream)
        with torch.cuda.stream(consumer_stream):
            consumer_stream.wait_stream(prefetch_stream)
            fallback_rows = _read_rows_by_feature(
                sharded_model, fallback_read_kjt, feature_names
            )
            fallback_done = torch.cuda.Event()
            fallback_done.record()
        fallback_done.synchronize()
        for actual, expected in zip(fallback_rows, fallback_expected):
            torch.testing.assert_close(
                actual.float().cpu(), expected, rtol=1e-2, atol=1e-2
            )
        assert [tbe.step for tbe in tbes] == [
            step + 2 for step in before_stale_fallback
        ]

        # A rebuilt pipeline may prefetch before consuming the previous
        # pipeline's look-ahead. The second prefetch replaces that stale result
        # and is then consumed normally.
        before_replacement = [tbe.step for tbe in tbes]
        with torch.cuda.stream(prefetch_stream):
            grouped_lookup.prefetch(read_kjt, forward_stream=consumer_stream)
            grouped_lookup.prefetch(fallback_read_kjt, forward_stream=consumer_stream)
        with torch.cuda.stream(consumer_stream):
            consumer_stream.wait_stream(prefetch_stream)
            replacement_rows = _read_rows_by_feature(
                sharded_model, fallback_read_kjt, feature_names
            )
            replacement_done = torch.cuda.Event()
            replacement_done.record()
        replacement_done.synchronize()
        for actual, expected in zip(replacement_rows, fallback_expected):
            torch.testing.assert_close(
                actual.float().cpu(), expected, rtol=1e-2, atol=1e-2
            )
        assert [tbe.step for tbe in tbes] == [step + 2 for step in before_replacement]


def _run_owner_local_cache_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    _run_round_trip(
        rank,
        world_size,
        backend,
        local_size,
        owner_local_cache=True,
    )


def _run_owner_local_cache_cross_stream_writes(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        sharded_model = _build_sharded_model(
            ctx,
            world_size,
            local_size,
            owner_local_cache=True,
        )
        enrichment_cache = _enrichment_cache(sharded_model)
        owner_cache = _owner_local_cache(sharded_model)

        slot_count = owner_cache.cache_state.numel()
        used_slots: set[int] = set()
        hashed_ids: list[int] = []
        candidate = 100
        while len(hashed_ids) < 16:
            slot = int(torch.ops.fbgemm.lxu_cache_slot(candidate, slot_count))
            if slot not in used_slots:
                used_slots.add(slot)
                hashed_ids.append(candidate)
            candidate += 1
        hashed_batches = [hashed_ids[offset : offset + 4] for offset in range(0, 16, 4)]
        unhashed_batches = [
            [hashed_id * 10 for hashed_id in batch] for batch in hashed_batches
        ]
        read_kjts = [_build_input_kjt(batch, ctx.device) for batch in hashed_batches]
        write_kjts = [
            _build_write_kjt(hashed, unhashed, ctx.device)
            for hashed, unhashed in zip(hashed_batches, unhashed_batches)
        ]

        # Populate the host backend first; this setup synchronization is outside
        # the cross-stream lifetime stress sequence below.
        for read_kjt, write_kjt, unhashed in zip(
            read_kjts, write_kjts, unhashed_batches
        ):
            # pyrefly: ignore[missing-attribute]
            sharded_model.write(write_kjt)
            torch.cuda.synchronize()
            rows = _wait_for_cache_fill(sharded_model, read_kjt, len(unhashed))
            expected = torch.stack(
                [_expected_fake_row(uid, EMBEDDING_DIM) for uid in unhashed]
            )
            torch.testing.assert_close(
                rows.float().cpu(), expected, rtol=1e-2, atol=1e-2
            )

        enrichment_cache.emb_module.reset_enrichment_cache()

        # Prime one pending query, then perform three updates on a distinct
        # stream. There is intentionally no per-write synchronization: each
        # update consumes the prior query's IDs while the next query can reuse
        # allocator storage on the enrichment stream.
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(write_kjts[0])
        update_stream = torch.cuda.Stream(device=ctx.device)
        update_stream.wait_stream(torch.cuda.current_stream())
        update_done = torch.cuda.Event()
        with torch.cuda.stream(update_stream):
            for write_kjt in write_kjts[1:]:
                # pyrefly: ignore[missing-attribute]
                sharded_model.write(write_kjt)
            update_done.record()
        update_done.synchronize()
        enrichment_cache.emb_module.enrichment_query_stream.synchronize()

        # The first three batches were consumed by later write cadences; all
        # must now be owner-cache hits with the correct corresponding rows.
        for read_kjt, unhashed in zip(read_kjts[:-1], unhashed_batches[:-1]):
            rows = _read_rows(sharded_model, read_kjt)
            expected = torch.stack(
                [_expected_fake_row(uid, EMBEDDING_DIM) for uid in unhashed]
            )
            torch.testing.assert_close(
                rows.float().cpu(), expected, rtol=1e-2, atol=1e-2
            )
            hits, queries = owner_cache.last_hit_counts()
            assert hits == queries == len(unhashed)


def _run_owner_local_cache_cross_stream_collision(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """Keep a prefetched hit stable while another stream replaces its slot."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        sharded_model = _build_sharded_model(
            ctx,
            world_size,
            local_size,
            owner_local_cache=True,
            prefetch_pipeline=True,
        )
        owner_cache = _owner_local_cache(sharded_model)

        first_key_by_slot: dict[int, int] = {}
        hit_key = -1
        replacement_key = -1
        cache_sets = owner_cache.cache_state.size(0)
        for key in range(NUM_EMBEDDINGS):
            slot = int(torch.ops.fbgemm.lxu_cache_slot(key, cache_sets))
            if slot in first_key_by_slot:
                hit_key = first_key_by_slot[slot]
                replacement_key = key
                break
            first_key_by_slot[slot] = key
        assert hit_key >= 0
        assert replacement_key >= 0

        hit_unhashed_id = hit_key + 100_000
        replacement_unhashed_id = replacement_key + 200_000
        expected_hit = _expected_fake_row(hit_unhashed_id, EMBEDDING_DIM)
        expected_replacement = _expected_fake_row(
            replacement_unhashed_id, EMBEDDING_DIM
        )
        owner_cache.populate_from_write(
            torch.tensor([hit_key], dtype=torch.int64, device=ctx.device),
            expected_hit.to(device=ctx.device, dtype=owner_cache.cache_weights.dtype)[
                None
            ],
        )
        torch.cuda.synchronize()

        replacement_read_kjt = _build_input_kjt([replacement_key], ctx.device)
        replacement_write_kjt = _build_write_kjt(
            [replacement_key], [replacement_unhashed_id], ctx.device
        )
        # Prime one pending write-fill for the colliding replacement key.
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(replacement_write_kjt)
        torch.cuda.synchronize()
        _wait_for_cache_fill(sharded_model, replacement_read_kjt, 1)

        miss_ids = [
            key
            for key in range(NUM_EMBEDDINGS)
            if key not in (hit_key, replacement_key)
        ]
        prefetched_kjt = _build_input_kjt([hit_key, *miss_ids], ctx.device)
        sharded_ec = cast(Any, sharded_model.module.ec)
        grouped_lookup = sharded_ec._lookups[0]
        prefetch_stream = torch.cuda.Stream(device=ctx.device)
        update_stream = torch.cuda.Stream(device=ctx.device)
        consumer_stream = torch.cuda.Stream(device=ctx.device)
        source_stream = torch.cuda.current_stream()
        step_before_prefetch = _enrichment_cache(sharded_model).emb_module.step
        prefetch_stream.wait_stream(source_stream)
        update_stream.wait_stream(source_stream)

        with torch.cuda.stream(prefetch_stream):
            # Keep lookup pending while update is enqueued so the write-side
            # wait on prefetch_complete_event is exercised deterministically.
            torch.cuda._sleep(50_000_000)
            grouped_lookup.prefetch(
                prefetched_kjt,
                forward_stream=consumer_stream,
            )
        with torch.cuda.stream(update_stream):
            # This consumes the pending replacement fill on a stream distinct
            # from the prefetch. There is intentionally no synchronization in
            # between; TBE cache events provide the required ordering.
            # pyrefly: ignore[missing-attribute]
            sharded_model.write(replacement_write_kjt)
            update_done = torch.cuda.Event()
            update_done.record()
        with torch.cuda.stream(consumer_stream):
            consumer_stream.wait_stream(prefetch_stream)
            prefetched_rows = _read_rows(sharded_model, prefetched_kjt)
            consumer_done = torch.cuda.Event()
            consumer_done.record()

        update_done.synchronize()
        consumer_done.synchronize()
        assert (
            _enrichment_cache(sharded_model).emb_module.step == step_before_prefetch + 1
        )
        torch.testing.assert_close(
            prefetched_rows[0].float().cpu(),
            expected_hit,
            rtol=1e-2,
            atol=1e-2,
        )
        replacement_rows = _read_rows(sharded_model, replacement_read_kjt)
        torch.testing.assert_close(
            replacement_rows[0].float().cpu(),
            expected_replacement,
            rtol=1e-2,
            atol=1e-2,
        )

        # Mirror the ordering: enqueue a delayed pending insert first, then a
        # lookup. Synchronizing the lookup consumer must also complete the
        # source-stream event that the insert waits on, proving the lookup
        # observed the newly recorded insert-complete event.
        mirror_step = _enrichment_cache(sharded_model).emb_module.step
        mirror_update_done = torch.cuda.Event()
        with torch.cuda.stream(update_stream):
            torch.cuda._sleep(200_000_000)
            # pyrefly: ignore[missing-attribute]
            sharded_model.write(replacement_write_kjt)
            mirror_update_done.record()
        with torch.cuda.stream(prefetch_stream):
            grouped_lookup.prefetch(
                replacement_read_kjt,
                forward_stream=consumer_stream,
            )
        with torch.cuda.stream(consumer_stream):
            consumer_stream.wait_stream(prefetch_stream)
            mirror_rows = _read_rows(sharded_model, replacement_read_kjt)
            mirror_consumer_done = torch.cuda.Event()
            mirror_consumer_done.record()
        mirror_consumer_done.synchronize()
        assert mirror_update_done.query()
        assert _enrichment_cache(sharded_model).emb_module.step == mirror_step + 1
        torch.testing.assert_close(
            mirror_rows[0].float().cpu(),
            expected_replacement,
            rtol=1e-2,
            atol=1e-2,
        )


def _run_owner_local_cache_read_populate_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    _run_round_trip(
        rank,
        world_size,
        backend,
        local_size,
        owner_local_cache=True,
        owner_local_cache_read_populate=True,
    )


def _run_owner_local_cache_bounded_read_populate_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    _run_round_trip(
        rank,
        world_size,
        backend,
        local_size,
        owner_local_cache=True,
        owner_local_cache_read_populate=True,
        owner_local_cache_read_populate_max_lookups=1,
    )


def _publish_round_trip(rows: torch.Tensor) -> torch.Tensor:
    """What serving decodes after the rowwise INT4 publish of FP16 `rows`."""
    packed = torch.ops.quantized.embedding_bag_4bit_prepack(rows)
    return torch.ops.quantized.embedding_bag_4bit_unpack(packed).to(torch.float16)


def _serving_fixed_point(rows: torch.Tensor) -> torch.Tensor:
    for _ in range(4):
        decoded = _publish_round_trip(rows)
        if torch.equal(decoded, rows):
            break
        rows = decoded
    return rows


def _run_serving_qdq_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
    serving_qdq_bit_rate: int = 4,
) -> None:
    """Enriched rows, and rows restored around the enrichment path, are stored as
    the values serving decodes from the publish; with the knob off, as fetched."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_model = _build_sharded_model(
            ctx,
            world_size,
            local_size,
            serving_qdq_bit_rate=serving_qdq_bit_rate,
            embedding_dim=SERVING_QDQ_DIM,
            data_type=DataType.FP16,
        )
        hashed_ids = [101, 202, 303, 404]
        unhashed_ids = [1, 2, 3, 7]
        read_kjt = _build_input_kjt(hashed_ids, ctx.device)
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(_build_write_kjt(hashed_ids, unhashed_ids, ctx.device))
        torch.cuda.synchronize()

        stored = _wait_for_cache_fill(sharded_model, read_kjt, len(hashed_ids)).cpu()

        fetched = torch.stack(
            [_expected_fake_row(uid, SERVING_QDQ_DIM) for uid in unhashed_ids]
        ).to(torch.float16)
        expected = _serving_fixed_point(fetched)
        assert not torch.equal(expected, fetched)
        if not serving_qdq_bit_rate:
            torch.testing.assert_close(stored, fetched.float(), rtol=0, atol=0)
            return
        torch.testing.assert_close(stored, expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(
            _publish_round_trip(stored.half()).float(), stored, rtol=0, atol=0
        )

        # Rows written straight into the backend, as a checkpoint restore does,
        # become serving values once apply_state_dict() runs.
        emb_module = _enrichment_cache(sharded_model).emb_module
        restored_ids = [505, 606]
        raw = torch.stack(
            [_expected_fake_row(uid, SERVING_QDQ_DIM) for uid in (1, 3)]
        ).to(torch.float16)
        backend_rows = torch.zeros(
            len(restored_ids), emb_module.cache_row_dim, dtype=torch.float16
        )
        backend_rows[:, :SERVING_QDQ_DIM] = raw
        emb_module.ssd_db.set(
            torch.tensor(restored_ids, dtype=torch.int64),
            backend_rows,
            torch.tensor([len(restored_ids)], dtype=torch.int64),
        )
        emb_module.apply_state_dict()
        restored = _read_rows(
            sharded_model, _build_input_kjt(restored_ids, ctx.device)
        ).cpu()
        torch.testing.assert_close(
            restored, _serving_fixed_point(raw).float(), rtol=0, atol=0
        )
        assert emb_module.ssd_db.serving_qdq_existing_rows() == 0


def _enrichment_cache(sharded_model: DistributedModelParallel) -> Any:
    sharded_ec = cast(Any, sharded_model.module.ec)
    for lookup in sharded_ec._lookups:
        for embedding_module in lookup._emb_modules:
            if hasattr(embedding_module, "direct_backend_gather_enabled"):
                return embedding_module
    raise AssertionError("enrichment cache was not created")


def _owner_local_cache(sharded_model: DistributedModelParallel) -> Any:
    owner_cache = _enrichment_cache(sharded_model).emb_module._direct_enrichment_cache
    if owner_cache is not None:
        return owner_cache
    raise AssertionError("owner-local cache was not created")


def _run_reduced_lookup_equivalence(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_model = _build_sharded_model(ctx, world_size, local_size)
        unique_hashed_ids = [101, 202, 303, 404]
        unique_unhashed_ids = [1010, 2020, 3030, 4040]
        unique_kjt = _build_input_kjt(unique_hashed_ids, ctx.device)
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(
            _build_write_kjt(unique_hashed_ids, unique_unhashed_ids, ctx.device)
        )
        torch.cuda.synchronize()
        _wait_for_cache_fill(sharded_model, unique_kjt, len(unique_hashed_ids))

        duplicated_ids = torch.tensor(
            [303, 101, 303, 202, 101, 404],
            dtype=torch.int64,
            device=ctx.device,
        )
        full_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[FEATURE_NAME],
            values=duplicated_ids,
            lengths=torch.ones(
                duplicated_ids.numel(), dtype=torch.int32, device=ctx.device
            ),
        )
        full_rows = sharded_model(full_kjt)[FEATURE_NAME].values()

        _, inverse, counts = torch.unique(
            duplicated_ids,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        _, sorted_inverse = torch.sort(inverse, stable=True)
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
        first_occurrence = sorted_inverse[starts]
        reduced_ids = duplicated_ids.index_select(0, first_occurrence)
        reduced_lengths = torch.zeros_like(duplicated_ids, dtype=torch.int32)
        reduced_lengths[: reduced_ids.numel()] = 1
        reduced_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[FEATURE_NAME],
            values=reduced_ids,
            lengths=reduced_lengths,
        )
        reduced_rows = sharded_model(reduced_kjt)[FEATURE_NAME].values()

        torch.testing.assert_close(
            reduced_rows,
            full_rows.index_select(0, first_occurrence),
        )


class TestKvzchEnrichmentE2E(MultiProcessTestBase):
    """End-to-end coverage for KVZCH enrichment reads, writes, and caching."""

    def _require_enrichment_policy_fields(self, fields: tuple[str, ...]) -> None:
        missing_fields = [
            field for field in fields if field not in EnrichmentPolicy._fields
        ]
        if missing_fields:
            self.skipTest(
                "The installed FBGEMM EnrichmentPolicy does not support "
                f"{', '.join(missing_fields)}"
            )

    def _require_test_runtime(self, world_size: int) -> None:
        if torch.cuda.device_count() < world_size:
            self.skipTest(
                f"Not enough GPUs, this test requires at least {world_size} GPUs"
            )
        try:
            _ = torch.classes.fbgemm.FeatureEvictConfig
        except RuntimeError:
            self.skipTest(
                "The installed FBGEMM does not register fbgemm::FeatureEvictConfig"
            )

    def test_enrichment_round_trip_disabled_in_oss_compatibility(self) -> None:
        world_size = 2
        self._require_test_runtime(world_size)
        self._run_multi_process_test(
            callable=_run_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_direct_backend_gather_preserves_enrichment_round_trip(self) -> None:
        world_size = 2
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_DIRECT_BACKEND_GATHER_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_direct_backend_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_direct_backend_gather_prefetches_the_forward_result(self) -> None:
        world_size = 1
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_DIRECT_BACKEND_GATHER_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_direct_backend_prefetch_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_owner_local_direct_cache_populates_from_completed_writes(self) -> None:
        world_size = 2
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_OWNER_LOCAL_CACHE_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_owner_local_cache_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_owner_local_write_warming_preserves_cross_stream_ids(self) -> None:
        world_size = 1
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_OWNER_LOCAL_CACHE_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_owner_local_cache_cross_stream_writes,
            world_size=world_size,
            backend="nccl",
        )

    def test_owner_local_write_cannot_replace_an_inflight_prefetched_hit(
        self,
    ) -> None:
        world_size = 1
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_OWNER_LOCAL_CACHE_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_owner_local_cache_cross_stream_collision,
            world_size=world_size,
            backend="nccl",
        )

    def test_owner_local_direct_cache_populates_from_reads(self) -> None:
        world_size = 2
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_OWNER_LOCAL_CACHE_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_owner_local_cache_read_populate_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_owner_local_direct_cache_bounds_read_population(self) -> None:
        world_size = 2
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_OWNER_LOCAL_CACHE_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_owner_local_cache_bounded_read_populate_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_reduced_lookup_preserves_row_wise_output_order(self) -> None:
        world_size = 2
        self._require_test_runtime(world_size)
        self._run_multi_process_test(
            callable=_run_reduced_lookup_equivalence,
            world_size=world_size,
            backend="nccl",
        )

    def test_serving_qdq_stores_enriched_rows_as_served(self) -> None:
        world_size = 1
        self._require_test_runtime(world_size)
        self._require_enrichment_policy_fields(_SERVING_QDQ_POLICY_FIELDS)
        self._run_multi_process_test(
            callable=_run_serving_qdq_round_trip,
            world_size=world_size,
            backend="nccl",
        )

    def test_serving_qdq_off_stores_enriched_rows_as_fetched(self) -> None:
        world_size = 1
        self._require_test_runtime(world_size)
        self._run_multi_process_test(
            callable=_run_serving_qdq_round_trip,
            world_size=world_size,
            backend="nccl",
            serving_qdq_bit_rate=0,
        )


class _DirectPrefetchBackend(nn.Module):
    direct_backend_gather_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.pending: Optional[torch.Tensor] = None
        self.prefetch_calls = 0
        self.indices_dtype: Optional[torch.dtype] = None
        self.offsets_dtype: Optional[torch.dtype] = None

    def prefetch(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.prefetch_calls += 1
        self.indices_dtype, self.offsets_dtype = indices.dtype, offsets.dtype
        self.pending = torch.stack((indices.float(), indices.float() + 1), dim=1)

    def get_prefetched_direct_backend_output(self) -> Optional[torch.Tensor]:
        return self.pending

    def clear_prefetched_direct_backend_output(self) -> None:
        self.pending = None

    def reset_enrichment_cache(self) -> None:
        raise AssertionError("Dropping a prefetch must not reset the owner cache")

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rows = self.pending
        self.pending = None
        if rows is None:
            rows = torch.stack((indices.float(), indices.float() + 1), dim=1)
        return rows.reshape(-1)


class DirectPrefetchWrapperTest(unittest.TestCase):
    def _cache(
        self,
    ) -> tuple[ZeroCollisionEmbeddingEnrichmentCache, _DirectPrefetchBackend]:
        backend = _DirectPrefetchBackend()

        def initialize_backend(
            cache: ZeroCollisionEmbeddingEnrichmentCache, *args: Any, **kwargs: Any
        ) -> None:
            nn.Module.__init__(cache)
            cache._emb_module = cast(Any, backend)
            cache._optim = mock.Mock()

        with mock.patch.object(
            ZeroCollisionKeyValueEmbedding, "__init__", initialize_backend
        ):
            cache = ZeroCollisionEmbeddingEnrichmentCache(
                config=mock.Mock(spec=GroupedEmbeddingConfig),
                device=torch.device("cpu"),
            )
        return cache, backend

    @staticmethod
    def _features() -> KeyedJaggedTensor:
        return KeyedJaggedTensor.from_lengths_sync(
            keys=[FEATURE_NAME],
            values=torch.tensor([3, 1], dtype=torch.int32),
            lengths=torch.ones(2, dtype=torch.int32),
        )

    def test_default_forward_starts_no_sampler_or_allocator_trim(self) -> None:
        from torchrec.distributed import batched_embedding_kernel as kernel

        with (
            mock.patch.dict(
                os.environ,
                {
                    "TORCHREC_KVZCH_HOST_MEM_LOG_SECS": "0",
                    "TORCHREC_KVZCH_PINNED_TRIM_MIN_IDLE_GB": "0",
                },
            ),
            mock.patch.object(kernel, "_host_mem_logger_started", False),
            mock.patch.object(kernel, "_pinned_trimmer_initialized", False),
            mock.patch.object(kernel, "_pinned_trimmer", None),
            mock.patch.object(kernel.threading, "Thread") as thread,
            mock.patch.object(torch._C, "_host_emptyCache", create=True) as trim,
        ):
            cache, backend = self._cache()
            rows = cache(self._features())
        torch.testing.assert_close(rows, torch.tensor([3.0, 4.0, 1.0, 2.0]))
        self.assertEqual(backend.prefetch_calls, 0)
        thread.assert_not_called()
        trim.assert_not_called()

    def test_public_prefetch_is_consumed_by_forward_with_long_indices(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TORCHREC_KVZCH_HOST_MEM_LOG_SECS": "0",
                "TORCHREC_KVZCH_PINNED_TRIM_MIN_IDLE_GB": "0",
            },
        ):
            cache, backend = self._cache()
            features = self._features()
            prefetched = cache.prefetch_direct_backend_gather(features)
            self.assertIs(cache.get_prefetched_direct_backend_output(), prefetched)
            rows = cache(features)
        self.assertEqual(backend.indices_dtype, torch.int64)
        self.assertEqual(backend.offsets_dtype, torch.int64)
        torch.testing.assert_close(rows, prefetched.reshape(-1))
        self.assertIsNone(cache.get_prefetched_direct_backend_output())

    def test_public_clear_drops_ssd_state_without_resetting_owner_cache(self) -> None:
        with mock.patch.dict(os.environ, {"TORCHREC_KVZCH_HOST_MEM_LOG_SECS": "0"}):
            cache, _ = self._cache()
            cache.prefetch_direct_backend_gather(self._features())
            cache.clear_prefetched_direct_backend_output()
        self.assertIsNone(cache.get_prefetched_direct_backend_output())


def _host_stats(held: int, active: int) -> dict[str, int]:
    return {"allocated_bytes.current": held, "active_bytes.current": active}


class TestPinnedHostCacheTrimmer(unittest.TestCase):
    def test_off_unless_the_env_is_set(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_PinnedHostCacheTrimmer.from_env())

    def test_trims_only_when_idle_reaches_the_threshold(self) -> None:
        trimmer = _PinnedHostCacheTrimmer(
            min_idle_gb=1, every_n_calls=2, min_interval_s=0
        )
        stats = [_host_stats(3 << 30, (2 << 30) + 1)]
        with (
            mock.patch.object(
                torch.cuda, "host_memory_stats", side_effect=lambda: stats[0]
            ),
            mock.patch.object(torch._C, "_host_emptyCache", create=True) as empty,
        ):
            self.assertFalse(trimmer.maybe_trim())  # call 1: not a check step
            self.assertFalse(trimmer.maybe_trim())  # call 2: idle < 1 GiB
            stats[0] = _host_stats(3 << 30, 1 << 30)
            self.assertFalse(trimmer.maybe_trim())
            self.assertTrue(trimmer.maybe_trim())  # call 4: idle == 2 GiB
        empty.assert_called_once()

    def test_respects_the_min_interval(self) -> None:
        trimmer = _PinnedHostCacheTrimmer(
            min_idle_gb=1, every_n_calls=1, min_interval_s=3600
        )
        with (
            mock.patch.object(
                torch.cuda, "host_memory_stats", return_value=_host_stats(8 << 30, 0)
            ),
            mock.patch.object(torch._C, "_host_emptyCache", create=True) as empty,
        ):
            self.assertTrue(trimmer.maybe_trim())
            self.assertFalse(trimmer.maybe_trim())
        empty.assert_called_once()

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
    def test_releases_idle_pinned_blocks_and_keeps_live_ones(self) -> None:
        mb = 1 << 20
        device = torch.accelerator.current_accelerator(True)
        sink = torch.empty(64 * mb, dtype=torch.uint8, device=device)
        live = torch.full((8 * mb,), 7, dtype=torch.uint8, pin_memory=True)
        # A queue that deepens (stall) and drains leaves its high-water cached.
        queue = [
            torch.empty(n * mb, dtype=torch.uint8, pin_memory=True)
            for n in (20, 40, 33, 60, 17, 48)
        ]
        for t in queue:
            sink[: t.numel()].copy_(t, non_blocking=True)
        del queue, t
        torch.cuda.synchronize()
        before = torch.cuda.host_memory_stats()
        trimmer = _PinnedHostCacheTrimmer(
            min_idle_gb=0, every_n_calls=1, min_interval_s=0
        )
        self.assertTrue(trimmer.maybe_trim())
        after = torch.cuda.host_memory_stats()
        self.assertLess(
            after["allocated_bytes.current"], before["allocated_bytes.current"]
        )
        self.assertEqual(
            after["allocated_bytes.current"], after["active_bytes.current"]
        )
        self.assertTrue(live.is_pinned())
        self.assertTrue(bool((live == 7).all()))
