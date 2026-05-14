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

import time
import unittest
from typing import List, Optional

import torch
import torch.nn as nn
from fbgemm_gpu.tbe.ssd.ssd_config import (
    EnrichmentPolicy,
    EnrichmentResponseFormat,
    EnrichmentType,
    KVZCHTBEConfig,
)
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
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
    EmbeddingConfig,
    TimestampBasedEvictionPolicy,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# Embedding dim small enough for fast tests; large enough that the deterministic
# formula gives distinguishable per-slot values within fp16/bf16 precision.
EMBEDDING_DIM = 8
NUM_EMBEDDINGS = 8000
TOTAL_NUM_BUCKETS = 20
FEATURE_NAME = "feature_0"
TABLE_NAME = "table_0"
PROVIDER_NAME = "test_in_memory_test_only"
CLIENT_ID = "test_kvzch_enrichment_e2e"


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
    out = sharded_model(kjt)
    values = out[FEATURE_NAME].to_dense()
    return torch.cat(list(values)) if isinstance(values, list) else values


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


def _build_write_kjt(
    hashed_ids: List[int], unhashed_ids: List[int], device: torch.device
) -> KeyedJaggedTensor:
    """Build the KJT consumed by EC.write() in KVZCH+enrichment mode.

    Mirrors minimal_viable_ai/core/utils/kvzch_utils.build_embedding_cache_write_kjt:
    weights are the unhashed IDs bit-cast to float32 pairs, shape [N, 2].
    """
    assert len(hashed_ids) == len(unhashed_ids)
    unhashed_tensor = torch.tensor(unhashed_ids, dtype=torch.int64)
    # Lossless reinterpret_cast: each int64 -> two float32 columns.
    weights = unhashed_tensor.view(torch.float32).reshape(-1, 2)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=[FEATURE_NAME],
        values=torch.tensor(hashed_ids, dtype=torch.int64),
        lengths=torch.tensor([1] * len(hashed_ids), dtype=torch.int32),
        weights=weights,
    ).to(device)


def _build_sharded_model(
    ctx: MultiProcessContext, world_size: int, local_size: Optional[int]
) -> DistributedModelParallel:
    """Construct a sharded EC with one virtual KVZCH table backed by the
    IN_MEMORY_TEST_ONLY enrichment provider."""
    tables = [
        EmbeddingConfig(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=EMBEDDING_DIM,
            name=TABLE_NAME,
            feature_names=[FEATURE_NAME],
            total_num_buckets=TOTAL_NUM_BUCKETS,
            use_virtual_table=True,
            enable_embedding_update=True,
            virtual_table_eviction_policy=TimestampBasedEvictionPolicy(
                training_id_eviction_trigger_count=10_000_000,
                eviction_ttl_mins=60,
            ),
        ),
    ]
    model = TestECModel(tables=tables, device=ctx.device)

    fused_params = {
        "embedding_cache_mode": True,
        "l2_cache_size": 1,  # GB, plenty for the test
        "kvzch_tbe_config": KVZCHTBEConfig(
            enrichment_policy=EnrichmentPolicy(
                enrichment_type=EnrichmentType.IN_MEMORY_TEST_ONLY,
                provider_name=PROVIDER_NAME,
                client_id=CLIENT_ID,
                enrichment_dim=EMBEDDING_DIM,
                response_format=EnrichmentResponseFormat.THRIFT_FLOAT,
            ),
        ),
    }
    sharder = EmbeddingCollectionSharder(fused_params=fused_params)
    sharding_plan = construct_module_sharding_plan(
        model.ec,
        per_param_sharding={
            TABLE_NAME: row_wise(
                compute_kernel=EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value
            ),
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
    sharded_model = DistributedModelParallel(
        model,
        env=ShardingEnv.from_process_group(pg),
        plan=ShardingPlan({"ec": sharding_plan}),
        # pyrefly: ignore[bad-argument-type]
        sharders=[sharder],
        device=ctx.device,
    )
    return sharded_model


def _run_round_trip(
    rank: int,
    world_size: int,
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """Body of the multi-process test, called per rank.

    1. Cold read with the hashed IDs - expect zeros (cache miss, no enrichment
       has been triggered yet).
    2. Call EC.write() with values=hashed, weights=encoded(unhashed) - this
       triggers the async fake enrichment fetch.
    3. Poll-read until the cache fills - assert each row equals the
       deterministic fake formula evaluated at its unhashed ID. This proves
       the write API plumbing landed an enrichment-driven fill in the cache.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        sharded_model = _build_sharded_model(ctx, world_size, local_size)

        # (hashed, unhashed) pairs. The fake formula operates on the unhashed
        # ID; the cache stores the result keyed by the hashed ID.
        hashed_ids = [101, 202, 303, 404]
        unhashed_ids = [1010, 2020, 3030, 4040]
        read_kjt = _build_input_kjt(hashed_ids, ctx.device)

        # Phase 1 - cold read returns zeros.
        cold_rows = _read_rows(sharded_model, read_kjt)
        self_check_zero = cold_rows.float().abs().sum().item()
        assert (
            self_check_zero < 1e-6
        ), f"Expected cold read to return zeros; got nonzero sum {self_check_zero}"

        # Phase 2 - write triggers the async fake fetch.
        write_kjt = _build_write_kjt(hashed_ids, unhashed_ids, ctx.device)
        # pyrefly: ignore[missing-attribute]
        sharded_model.write(write_kjt)
        torch.cuda.synchronize()

        # Phase 3 - poll until the cache fills, then assert per-row values.
        filled_rows = _wait_for_cache_fill(sharded_model, read_kjt, len(hashed_ids))

        expected = torch.stack(
            [_expected_fake_row(uid, EMBEDDING_DIM) for uid in unhashed_ids]
        )
        torch.testing.assert_close(
            filled_rows.float().cpu(),
            expected,
            rtol=1e-2,
            atol=1e-2,
        )


class TestKvzchEnrichmentE2E(MultiProcessTestBase):
    """End-to-end test of the KVZCH enrichment read+write round trip.

    Disabled in OSS: depends on Meta-internal fbgemm fb/ enrichment headers.
    """

    def _gpu_check(self, world_size: int) -> None:
        if torch.cuda.device_count() < world_size:
            self.skipTest(
                f"Not enough GPUs, this test requires at least {world_size} GPUs"
            )

    def test_enrichment_round_trip_disabled_in_oss_compatibility(self) -> None:
        WORLD_SIZE = 2
        self._gpu_check(WORLD_SIZE)
        self._run_multi_process_test(
            callable=_run_round_trip,
            world_size=WORLD_SIZE,
            backend="nccl",
        )


if __name__ == "__main__":
    unittest.main()
