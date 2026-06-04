#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import copy
import unittest
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.pec_embedding import (
    BackwardPartitionContext,
    ForwardPartitionContext,
    PECEmbeddingCollectionContext,
    PECEmbeddingCollectionSharder,
    ShardedPECEmbeddingCollection,
)
from torchrec.distributed.shard import shard_modules
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.pec_embedding_modules import PECEmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

EMBEDDING_TABLES: List[EmbeddingConfig] = [
    EmbeddingConfig(
        name="table_0",
        feature_names=["feature_0"],
        embedding_dim=8,
        num_embeddings=16,
    ),
    EmbeddingConfig(
        name="table_1",
        feature_names=["feature_1"],
        embedding_dim=8,
        num_embeddings=8,
    ),
]


# =============================================================================
# Unit tests (no distributed setup)
# =============================================================================


class PECEmbeddingCollectionTest(unittest.TestCase):
    """Unit tests for PECEmbeddingCollection (unsharded) — no distributed setup."""

    def test_forward(self) -> None:
        ec = EmbeddingCollection(tables=EMBEDDING_TABLES, device=torch.device("cpu"))
        pec = PECEmbeddingCollection(ec)

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 1, 2, 3, 4, 5]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        )
        out: Dict[str, JaggedTensor] = pec(kjt)

        self.assertIn("feature_0", out)
        self.assertIn("feature_1", out)
        self.assertEqual(out["feature_0"].values().shape[1], 8)
        self.assertEqual(out["feature_1"].values().shape[1], 8)


class PECEmbeddingCollectionSharderTest(unittest.TestCase):
    """Unit tests for PECEmbeddingCollectionSharder — no distributed setup."""

    def test_sharding_types_rw_only(self) -> None:
        sharder = PECEmbeddingCollectionSharder()
        types = sharder.sharding_types(compute_device_type="cuda")
        self.assertEqual(types, [ShardingType.ROW_WISE.value])

    def test_shardable_parameters(self) -> None:
        ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES[:1], device=torch.device("cpu")
        )
        pec = PECEmbeddingCollection(ec)

        sharder = PECEmbeddingCollectionSharder()
        params = sharder.shardable_parameters(pec)

        self.assertIn("table_0", params)
        self.assertEqual(params["table_0"].shape, (16, 8))


# =============================================================================
# Model & sharding helpers
# =============================================================================


class PECSparseArch(nn.Module):
    """Simple model wrapping PECEmbeddingCollection for sharded testing."""

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
    ) -> None:
        super().__init__()
        self._pec_ec: PECEmbeddingCollection = PECEmbeddingCollection(
            EmbeddingCollection(tables=tables, device=device),
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, Dict[str, JaggedTensor]]:
        ec_out = self._pec_ec(kjt)
        pred = torch.cat(
            [ec_out[key].values() for key in ["feature_0", "feature_1"]],
            dim=0,
        )
        loss = pred.mean()
        return loss, ec_out


def _shard_pec(
    tables: List[EmbeddingConfig],
    ctx: MultiProcessContext,
    sharder: ModuleSharder[nn.Module],
    local_size: Optional[int] = None,
) -> PECSparseArch:
    sparse_arch = PECSparseArch(tables, torch.device("meta"))

    module_sharding_plan = construct_module_sharding_plan(
        sparse_arch._pec_ec,
        per_param_sharding={
            table.name: row_wise(compute_kernel="fused") for table in tables
        },
        local_size=local_size,
        world_size=ctx.world_size,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        sharder=sharder,
    )

    return shard_modules(  # pyre-ignore[7]
        module=copy.deepcopy(sparse_arch),
        plan=ShardingPlan({"_pec_ec": module_sharding_plan}),
        env=ShardingEnv.from_process_group(ctx.pg),
        sharders=[sharder],
        device=ctx.device,
    )


@dataclass
class _PECForwardResult:
    """Result of _pec_forward for one batch."""

    # Pipeline contexts (passed to backward)
    pec_ctx: PECEmbeddingCollectionContext
    fwd_ctx: ForwardPartitionContext
    bwd_ctx: BackwardPartitionContext

    # Intermediate: partitioned embeddings before merge
    ol_awaitables: List[Awaitable[torch.Tensor]]
    nol_awaitables: List[Awaitable[torch.Tensor]]

    # Final: merged output (populated when should_merge_partitions=True)
    jt_dict: Dict[str, JaggedTensor]

    # Original input (for verification against ref_ec)
    original_values: torch.Tensor
    original_lengths: torch.Tensor
    device: torch.device


@dataclass
class _PECState:
    """PEC state across batches."""

    sharded_pec: ShardedPECEmbeddingCollection
    pec_ctx: PECEmbeddingCollectionContext
    dist_input: KJTList
    fwd_ctx: ForwardPartitionContext
    batches: List[KeyedJaggedTensor]
    device: torch.device


VerifyFn = Optional[
    Callable[[_PECForwardResult, EmbeddingCollection | None, int, int], None]
]


def _pec_pre_forward(
    tables: List[EmbeddingConfig],
    ctx: MultiProcessContext,
    sharder: ModuleSharder[nn.Module],
    batches: List[KeyedJaggedTensor],
    ref_ec: EmbeddingCollection | None = None,
    local_size: Optional[int] = None,
) -> _PECState:
    """Shard PEC, copy weights, input_dist + overlap_dist for batch 0."""
    sharded_sparse_arch = _shard_pec(tables, ctx, sharder, local_size)
    sharded_pec = sharded_sparse_arch._pec_ec
    assert isinstance(sharded_pec, ShardedPECEmbeddingCollection)
    assert len(sharded_pec._overlap_handlers) > 0

    if ref_ec is not None:
        copy_state_dict(
            sharded_pec._embedding_collection.state_dict(),
            ref_ec.state_dict(),
        )

    kjt_0 = batches[0].to(ctx.device)
    pec_ctx = sharded_pec.create_context()
    dist_input = sharded_pec.input_dist(pec_ctx, kjt_0).wait().wait()

    results_0 = sharded_pec.overlap_dist(ctx=pec_ctx, dist_input=dist_input)
    fwd_ctx, _ = results_0[0].wait()
    assert fwd_ctx is not None

    return _PECState(
        sharded_pec=sharded_pec,
        pec_ctx=pec_ctx,
        dist_input=dist_input,
        fwd_ctx=fwd_ctx,
        batches=batches,
        device=ctx.device,
    )


def _pec_forward(
    state: _PECState,
    batch_idx: int,
    rank: int = 0,
    ref_ec: EmbeddingCollection | None = None,
    verify_fn: VerifyFn = None,
    should_merge_partitions: bool = True,
) -> _PECForwardResult:
    """PEC forward: lookahead overlap_dist + compute + optional merge + verify.

    Args:
        verify_fn: called with (result, ref_ec, rank, batch_idx) after
            merge (or directly if should_merge_partitions=False). Use
            partial to bind expected data.
        should_merge_partitions: if True, run merge_partitioned_embeddings
            before verify_fn. If False, skip merge and call verify_fn
            directly.
    """
    sharded_pec = state.sharded_pec
    batches = state.batches
    pec_ctx = state.pec_ctx
    dist_input = state.dist_input
    fwd_ctx = state.fwd_ctx

    original_values = batches[batch_idx].values()
    original_lengths = batches[batch_idx].lengths()

    # Lookahead: input_dist + overlap_dist for next batch
    if batch_idx + 1 < len(batches):
        next_kjt = batches[batch_idx + 1].to(state.device)
        next_pec_ctx = sharded_pec.create_context()
        next_dist_input = sharded_pec.input_dist(next_pec_ctx, next_kjt).wait().wait()
        results_next = sharded_pec.overlap_dist(
            ctx=next_pec_ctx,
            dist_input=next_dist_input,
            prev_ctx=pec_ctx,
            prev_dist_input=dist_input,
        )
    else:
        results_next = sharded_pec.overlap_dist(
            ctx=None,
            dist_input=None,
            prev_ctx=pec_ctx,
            prev_dist_input=dist_input,
        )
        next_pec_ctx = None
        next_dist_input = None

    fwd_next, bwd_cur = results_next[0].wait()
    assert bwd_cur is not None

    # Compute OL/NOL
    ol_awaitables = sharded_pec.compute_and_output_dist_in_partition(
        pec_ctx,
        fwd_ctx.ol_features,
        fwd_ctx.splits,
        is_overlapped=True,
    )
    nol_awaitables = sharded_pec.compute_and_output_dist_in_partition(
        pec_ctx,
        fwd_ctx.nol_features,
        fwd_ctx.splits,
        is_overlapped=False,
    )

    result = _PECForwardResult(
        pec_ctx=pec_ctx,
        fwd_ctx=fwd_ctx,
        bwd_ctx=bwd_cur,
        ol_awaitables=ol_awaitables,
        nol_awaitables=nol_awaitables,
        original_values=original_values,
        original_lengths=original_lengths,
        jt_dict={},
        device=state.device,
    )

    if should_merge_partitions:
        result.jt_dict = sharded_pec.merge_partitioned_embeddings(
            pec_ctx,
            ol_awaitables,
            nol_awaitables,
            [fwd_ctx],
            [bwd_cur],
        ).wait()

    if verify_fn is not None:
        verify_fn(result, ref_ec, rank, batch_idx)

    # Advance state
    if fwd_next is not None:
        state.fwd_ctx = fwd_next

    if next_pec_ctx is not None:
        assert next_dist_input is not None
        state.pec_ctx = next_pec_ctx
        state.dist_input = next_dist_input

    return result


def _pec_backward(
    state: _PECState,
    result: _PECForwardResult,
) -> None:
    """PEC backward: loss.backward(), grad_dist."""
    pec_pred = torch.cat(
        [result.jt_dict[k].values() for k in result.jt_dict],
        dim=0,
    )
    pec_pred.mean().backward()

    ol_grad_aw = state.sharded_pec.grad_dist(
        result.pec_ctx,
        KJTList([result.bwd_ctx.ol_features]),
        is_overlapped=True,
    )
    ol_grad_aw[0].wait()

    nol_grad_aw = state.sharded_pec.grad_dist(
        result.pec_ctx,
        KJTList([result.bwd_ctx.nol_features]),
        is_overlapped=False,
    )
    nol_grad_aw[0].wait()


def _verify_partition_embeddings(
    embs: torch.Tensor,
    expected_specs: List[Tuple[str, int]],
    ref_ec: EmbeddingCollection,
    device: torch.device,
) -> None:
    assert embs.shape[0] == len(expected_specs)
    if len(expected_specs) > 0:
        expected_tensor = torch.stack(
            [
                # pyre-ignore[16]: embeddings[tname] is nn.Embedding
                ref_ec.embeddings[tname].weight[oid]
                for tname, oid in expected_specs
            ]
        ).to(device)
        torch.testing.assert_close(embs, expected_tensor)


def _verify_fwd_permutes(
    result: _PECForwardResult,
    ref_ec: EmbeddingCollection | None,
    rank: int,
    batch_idx: int,
    expected_per_rank: List[List[List[int]]],
) -> None:
    assert result.fwd_ctx is not None
    assert result.fwd_ctx.permute.tolist() == expected_per_rank[rank][batch_idx]


def _verify_bwd_permutes(
    result: _PECForwardResult,
    ref_ec: EmbeddingCollection | None,
    rank: int,
    batch_idx: int,
    expected_ol_per_rank: List[List[List[int]]],
    expected_nol_per_rank: List[List[List[int]]],
) -> None:
    assert result.bwd_ctx.ol_permute.tolist() == expected_ol_per_rank[rank][batch_idx]
    assert result.bwd_ctx.nol_permute.tolist() == expected_nol_per_rank[rank][batch_idx]


def _verify_ol_nol_partitions(
    result: _PECForwardResult,
    ref_ec: EmbeddingCollection | None,
    rank: int,
    batch_idx: int,
    expected_ol_per_rank: Dict[int, List[List[Tuple[str, int]]]],
    expected_nol_per_rank: Dict[int, List[List[Tuple[str, int]]]],
) -> None:
    assert ref_ec is not None
    _verify_partition_embeddings(
        result.ol_awaitables[0].wait(),
        expected_ol_per_rank[rank][batch_idx],
        ref_ec,
        result.device,
    )
    _verify_partition_embeddings(
        result.nol_awaitables[0].wait(),
        expected_nol_per_rank[rank][batch_idx],
        ref_ec,
        result.device,
    )


def _verify_merged_output(
    result: _PECForwardResult,
    ref_ec: EmbeddingCollection | None,
    rank: int,
    batch_idx: int,
    tables: List[EmbeddingConfig],
    batches_per_rank: List[List[KeyedJaggedTensor]],
) -> None:
    """Verifies merged PEC output matches ref EC for the given batch."""
    assert ref_ec is not None
    kjt_input = batches_per_rank[rank][batch_idx].to(result.device)

    feature_to_table = {}
    for table in tables:
        for fname in table.feature_names:
            feature_to_table[fname] = table.name

    assert set(result.jt_dict.keys()) == set(feature_to_table.keys())

    stride = kjt_input.stride()
    offset = 0
    for feat_idx, fname in enumerate(kjt_input.keys()):
        tname = feature_to_table[fname]
        feat_lengths = result.original_lengths[
            feat_idx * stride : (feat_idx + 1) * stride
        ]
        num_values = feat_lengths.sum().item()
        feat_ids = result.original_values[offset : offset + num_values]

        actual_jt = result.jt_dict[fname]
        torch.testing.assert_close(actual_jt.lengths().cpu(), feat_lengths)
        if num_values > 0:
            expected_embs = torch.stack(
                [
                    # pyre-ignore[16]
                    ref_ec.embeddings[tname].weight[vid.item()]
                    for vid in feat_ids
                ]
            ).to(result.device)
            torch.testing.assert_close(actual_jt.values(), expected_embs)
        offset += num_values


def _verify_weights_match(
    tables: List[EmbeddingConfig],
    sharded_pec: ShardedPECEmbeddingCollection,
    ref_ec: EmbeddingCollection,
    device: torch.device,
    atol: float = 1e-5,
    rtol: float = 1.3e-6,
) -> None:
    """Forward all indices through both models and compare outputs."""
    verify_kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=[t.feature_names[0] for t in tables],
        values=torch.cat([torch.arange(t.num_embeddings) for t in tables]),
        lengths=torch.LongTensor([t.num_embeddings for t in tables]),
    )

    with torch.no_grad():
        ref_out = ref_ec(verify_kjt)
        pec_out = sharded_pec(verify_kjt.to(device))

    for fname in ref_out:
        torch.testing.assert_close(
            pec_out[fname].values(),
            ref_out[fname].values().to(device),
            atol=atol,
            rtol=rtol,
        )


def _test_sharding(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _shard_pec(tables, ctx, sharder, local_size)

        assert isinstance(sharded_sparse_arch._pec_ec, ShardedPECEmbeddingCollection)
        assert isinstance(
            sharded_sparse_arch._pec_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )

        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        for _ in range(2):
            loss, _ = sharded_sparse_arch(kjt_input)
            loss.backward()


def _test_pec_forward_stages(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[List[KeyedJaggedTensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    ref_ec: EmbeddingCollection | None = None,
    verify_fn: VerifyFn = None,
    should_merge_partitions: bool = True,
    local_size: Optional[int] = None,
) -> None:
    """Runs PEC forward pipeline, calling verify_fn per batch."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        batches = kjt_input_per_rank[rank]
        state = _pec_pre_forward(
            tables,
            ctx,
            sharder,
            batches,
            ref_ec,
            local_size,
        )

        for i in range(len(batches)):
            _pec_forward(
                state,
                i,
                rank=rank,
                ref_ec=ref_ec,
                verify_fn=verify_fn,
                should_merge_partitions=should_merge_partitions,
            )


def _test_pec_grad_update(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[List[KeyedJaggedTensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    ref_ec: EmbeddingCollection,
    learning_rate: float = 0.01,
    local_size: Optional[int] = None,
) -> None:
    """Tests full train loop and verifies weights match ref_ec.

    Each step: ref_ec processes all ranks' batches (matching AllToAll
    redistribution), PEC runs the normal pipeline. After all batches,
    a full-index KJT lookup verifies the weights converged identically.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        batches = kjt_input_per_rank[rank]
        state = _pec_pre_forward(tables, ctx, sharder, batches, ref_ec, local_size)

        ref_optimizer = torch.optim.SGD(
            ref_ec.parameters(), lr=learning_rate, foreach=True
        )

        for i in range(len(batches)):
            ref_optimizer.zero_grad()
            ref_out = ref_ec(kjt_input_per_rank[rank][i])
            torch.cat(
                [ref_out[k].values() for k in ref_out],
                dim=0,
            ).mean().backward()
            ref_optimizer.step()

            result = _pec_forward(state, i)
            _pec_backward(state, result)

        # Relax tolerances: the sharded PEC uses a fused GPU optimizer kernel
        # while ref_ec uses standard CPU SGD, causing expected numerical
        # divergence in accumulated gradients. This matches the tolerance
        # convention used by other torchrec sharded-vs-unsharded weight
        # comparisons (e.g., test_embedding_update, test_pt2_multiprocess).
        _verify_weights_match(
            tables, state.sharded_pec, ref_ec, ctx.device, atol=1e-3, rtol=1e-3
        )


# =============================================================================
# Test data
# =============================================================================

# Cross-shard data for uneven tables (table_0: 16 rows, table_1: 8 rows).
# With world_size=2 RW sharding:
#   table_0: block_size=8, shard 0 rows 0-7, shard 1 rows 8-15
#   table_1: block_size=4, shard 0 rows 0-3, shard 1 rows 4-7
# stride=2 (batch_size=2 per rank), 3 batches per rank.
CROSS_SHARD_KJT_INPUT_PER_RANK = [
    [
        # Rank 0 batch 0
        # f0: [0,8,2] → shard0:[0,2], shard1:[8]
        # f1: [1,5,3] → shard0:[1,3], shard1:[5]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 8, 2, 1, 5, 3]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        # Rank 0 batch 1 (overlaps with batch 0: f0: 0, f1: 1)
        # f0: [0,9,3] → shard0:[0,3], shard1:[9]
        # f1: [1,6,2] → shard0:[1,2], shard1:[6]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 9, 3, 1, 6, 2]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        # Rank 0 batch 2 (overlaps with batch 1: f0: 0,3, f1: 2)
        # f0: [0,10,3] → shard0:[0,3], shard1:[10]
        # f1: [2,4,0] → shard0:[2,0], shard1:[4]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 10, 3, 2, 4, 0]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
    [
        # Rank 1 batch 0
        # f0: [1,10,5] → shard0:[1,5], shard1:[10]
        # f1: [4,2,7] → shard0:[2], shard1:[4,7]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 10, 5, 4, 2, 7]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        # Rank 1 batch 1 (overlaps with batch 0: f0: 1, f1: 4)
        # f0: [1,11,7] → shard0:[1,7], shard1:[11]
        # f1: [4,3,6] → shard0:[3], shard1:[4,6]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 11, 7, 4, 3, 6]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        # Rank 1 batch 2 (overlaps with batch 1: f0: 1,7, f1: 4,6)
        # f0: [1,12,7] → shard0:[1,7], shard1:[12]
        # f1: [4,0,6] → shard0:[0], shard1:[4,6]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 12, 7, 4, 0, 6]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
]


# =============================================================================
# Multi-process tests
# =============================================================================


@skip_if_asan_class
class ShardedPECEmbeddingCollectionTest(MultiProcessTestBase):
    """Multi-process tests for ShardedPECEmbeddingCollection."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharding_rw(self) -> None:
        WORLD_SIZE = 2

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=[
                batches[0] for batches in CROSS_SHARD_KJT_INPUT_PER_RANK
            ],
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_overlap_dist_forward_permute(self) -> None:
        """Verifies exact forward_permute values from overlap_dist."""
        WORLD_SIZE = 2

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            should_merge_partitions=False,
            verify_fn=partial(
                _verify_fwd_permutes,
                expected_per_rank=[
                    [[0, 4, 1, 2, 5, 3], [0, 4, 3, 1, 5, 2], [0, 5, 1, 2, 3, 4]],
                    [[0, 3, 1, 4, 2, 5], [0, 4, 3, 2, 1, 5], [0, 5, 1, 2, 4, 3]],
                ],
            ),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_overlap_dist_backward_permutes(self) -> None:
        """Verifies backward ol/nol permutes from overlap_dist."""
        WORLD_SIZE = 2

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            should_merge_partitions=False,
            verify_fn=partial(
                _verify_bwd_permutes,
                expected_ol_per_rank=[
                    [[0, 3, 5], [0, 2, 5, 4], [0, 2, 3, 5, 1, 4]],
                    [[0, 4, 3], [0, 2, 3, 5], [0, 2, 4, 1, 3, 5]],
                ],
                expected_nol_per_rank=[
                    [[2, 1, 4], [3, 1], []],
                    [[2, 1, 5], [4, 1], []],
                ],
            ),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_compute_and_output_dist_in_partition(self) -> None:
        """Tests OL/NOL embeddings match expected weight rows."""
        WORLD_SIZE = 2
        T0 = "table_0"
        T1 = "table_1"

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            ref_ec=ref_ec,
            should_merge_partitions=False,
            verify_fn=partial(
                _verify_ol_nol_partitions,
                expected_ol_per_rank={
                    0: [
                        [],
                        [(T0, 0), (T1, 1), (T1, 2)],
                        [(T0, 0), (T0, 3), (T1, 2), (T1, 4)],
                    ],
                    1: [
                        [],
                        [(T0, 1), (T1, 3), (T1, 4)],
                        [(T0, 1), (T0, 7), (T1, 4), (T1, 6)],
                    ],
                },
                expected_nol_per_rank={
                    0: [
                        [(T0, 0), (T0, 2), (T1, 1), (T1, 3), (T0, 8), (T1, 5)],
                        [(T0, 3), (T0, 9), (T1, 6)],
                        [(T1, 0), (T0, 10)],
                    ],
                    1: [
                        [(T0, 1), (T0, 5), (T1, 2), (T0, 10), (T1, 4), (T1, 7)],
                        [(T0, 7), (T0, 11), (T1, 6)],
                        [(T1, 0), (T0, 12)],
                    ],
                },
            ),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_merge_partitioned_embeddings(self) -> None:
        """Tests merged output matches expected embeddings from ref EC."""
        WORLD_SIZE = 2

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            ref_ec=ref_ec,
            verify_fn=partial(
                _verify_merged_output,
                tables=EMBEDDING_TABLES,
                batches_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            ),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_single_shard_features(self) -> None:
        """Tests overlap_dist when one shard receives no features.

        All values in shard 1's range (table_0: 8-15, table_1: 4-7),
        so shard 0 gets nothing after input_dist. Verifies mask_dist,
        splits, permutes, and merge handle empty KJTs on the empty shard.
        """
        WORLD_SIZE = 2

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        # table_0 shard1: [8,15], table_1 shard1: [4,7]
        shard1_only_input = [
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([8, 9, 4, 5]),
                    lengths=torch.LongTensor([2, 2]),
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([8, 10, 5, 6]),
                    lengths=torch.LongTensor([2, 2]),
                ),
            ],
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([12, 13, 6, 7]),
                    lengths=torch.LongTensor([2, 2]),
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([12, 14, 7, 4]),
                    lengths=torch.LongTensor([2, 2]),
                ),
            ],
        ]
        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=shard1_only_input,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            ref_ec=ref_ec,
            verify_fn=partial(
                _verify_merged_output,
                tables=EMBEDDING_TABLES,
                batches_per_rank=shard1_only_input,
            ),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_grad_update(self) -> None:
        """Tests full pipeline: overlap_dist → compute → merge → backward → grad apply.

        Uses 3 batches to exercise the complete pipeline ordering where
        batch i's backward data comes from batch i+1's overlap_dist.
        After all batches, verifies that sharded PEC weights match an
        unsharded ref EC trained with the same data and optimizer.
        """
        WORLD_SIZE = 2
        LEARNING_RATE = 0.01

        torch.manual_seed(42)
        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        self._run_multi_process_test(
            callable=_test_pec_grad_update,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(
                fused_params={"learning_rate": LEARNING_RATE},
            ),
            backend="nccl",
            ref_ec=ref_ec,
            learning_rate=LEARNING_RATE,
        )
