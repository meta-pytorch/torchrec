#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, NamedTuple

import torch
import torch.nn as nn
from pyre_extensions import none_throws
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.embeddingbag import (
    _create_mean_pooling_divisor,
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    data_parallel,
    row_wise,
    table_row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.test_utils import skip_if_asan_class

ROWS, DIM, BATCH = 64, 8, 16
SUM, MEAN = PoolingType.SUM, PoolingType.MEAN


class Table(NamedTuple):
    name: str
    features: list[str]
    sharding: str
    pooling: PoolingType


class Case(NamedTuple):
    tables: list[Table]
    weighted: bool = False
    variable_batch: bool = False


# The mean-pooling divisor only runs when the EBC has an RW or TWRW MEAN table.
CASES: dict[str, Case] = {
    "all_sharding_types": Case(
        [
            Table("tw_sum", ["f_tw_sum"], "tw", SUM),
            Table("tw_mean", ["f_tw_mean"], "tw", MEAN),
            Table("cw_sum", ["f_cw_sum"], "cw", SUM),
            Table("dp_sum", ["f_dp_sum"], "dp", SUM),
            Table("rw_sum", ["f_rw_sum"], "rw", SUM),
            Table("twrw_sum", ["f_twrw_sum"], "twrw", SUM),
            Table("rw_mean", ["f_rw_mean"], "rw", MEAN),
            Table("twrw_mean", ["f_twrw_mean"], "twrw", MEAN),
        ]
    ),
    "tw_sum_twrw_mean": Case(
        [Table("tw_sum", ["f_a"], "tw", SUM), Table("twrw_mean", ["f_b"], "twrw", MEAN)]
    ),
    "dp_sum_rw_mean": Case(
        [Table("dp_sum", ["f_a"], "dp", SUM), Table("rw_mean", ["f_b"], "rw", MEAN)]
    ),
    # A feature shared by two tables needs a divisor per embedding, not per feature.
    "shared_rw_sum_rw_mean": Case(
        [
            Table("rw_sum", ["f_shared", "f_a"], "rw", SUM),
            Table("rw_mean", ["f_shared", "f_b"], "rw", MEAN),
        ]
    ),
    "shared_twrw_mean_tw_sum": Case(
        [
            Table("twrw_mean", ["f_shared"], "twrw", MEAN),
            Table("tw_sum", ["f_shared", "f_a"], "tw", SUM),
        ]
    ),
    "shared_rw_mean_dp_sum": Case(
        [
            Table("rw_mean", ["f_shared"], "rw", MEAN),
            Table("dp_sum", ["f_shared"], "dp", SUM),
        ]
    ),
    # Weighted: the RW MEAN divisor is the summed weights; TW SUM is a weighted sum.
    "weighted_tw_sum_rw_mean": Case(
        [Table("tw_sum", ["f_a"], "tw", SUM), Table("rw_mean", ["f_b"], "rw", MEAN)],
        weighted=True,
    ),
    "variable_batch_mixed": Case(
        [
            Table("tw_sum", ["f_a", "f_shared"], "tw", SUM),
            Table("rw_mean", ["f_shared", "f_b"], "rw", MEAN),
        ],
        variable_batch=True,
    ),
    # Controls that were already correct before the fix.
    "rw_twrw_only": Case(
        [
            Table("rw_sum", ["f_a"], "rw", SUM),
            Table("twrw_sum", ["f_b"], "twrw", SUM),
            Table("twrw_mean", ["f_c"], "twrw", MEAN),
        ]
    ),
    "no_row_sharded_mean": Case(
        [
            Table("tw_sum", ["f_a"], "tw", SUM),
            Table("tw_mean", ["f_b"], "tw", MEAN),
            Table("cw_sum", ["f_c"], "cw", SUM),
            Table("dp_sum", ["f_d"], "dp", SUM),
        ]
    ),
}


class _Wrapper(nn.Module):
    def __init__(self, ebc: EmbeddingBagCollection) -> None:
        super().__init__()
        self.ebc = ebc

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        return self.ebc(features)


def _features(case: Case, rank: int) -> tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    """Returns the input KJT and the same input with every key at the full batch."""
    keys = list(dict.fromkeys(f for t in case.tables for f in t.features))
    gen = torch.Generator().manual_seed(rank)
    if not case.variable_batch:
        lengths = torch.randint(0, 4, (len(keys) * BATCH,), generator=gen)
        values = torch.randint(0, ROWS, (int(lengths.sum()),), generator=gen)
        weights = torch.rand(values.shape, generator=gen) if case.weighted else None
        kjt = KeyedJaggedTensor(
            keys=keys, values=values, lengths=lengths, weights=weights, stride=BATCH
        )
        return kjt, kjt

    strides = [int(torch.randint(1, BATCH, (1,), generator=gen)) for _ in keys]
    inverse = torch.stack(
        [torch.randint(0, s, (BATCH,), generator=gen) for s in strides]
    )
    lengths = [torch.randint(0, 4, (s,), generator=gen) for s in strides]
    values = [torch.randint(0, ROWS, (int(ln.sum()),), generator=gen) for ln in lengths]
    vbe = KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(values),
        lengths=torch.cat(lengths),
        stride_per_key_per_rank=[[s] for s in strides],
        inverse_indices=(keys, inverse),
    )
    full_lengths, full_values = [], []
    for ln, vals, inv in zip(lengths, values, inverse):
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), ln.cumsum(0)])
        for row in inv.tolist():
            full_lengths.append(ln[row])
            full_values.append(vals[offsets[row] : offsets[row + 1]])
    full = KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(full_values),
        lengths=torch.stack(full_lengths),
        stride=BATCH,
    )
    return vbe, full


def _weighted_reference(
    tables: list[EmbeddingBagConfig],
    state: dict[str, torch.Tensor],
    features: KeyedJaggedTensor,
) -> dict[str, torch.Tensor]:
    # nn.EmbeddingBag has no weighted MEAN, so pool directly from the weights.
    jagged = features.to_dict()
    out = {}
    for config, names in zip(tables, get_embedding_names_by_table(tables)):
        weight = state[f"embedding_bags.{config.name}.weight"]
        for feature, name in zip(config.feature_names, names):
            jt = jagged[feature]
            rows = []
            for ids, w in zip(jt.to_dense(), jt.to_dense_weights() or []):
                pooled = (weight[ids] * w.unsqueeze(1)).sum(0)
                if config.pooling == MEAN and w.numel() > 0:
                    pooled = pooled / w.sum()
                rows.append(pooled)
            out[name] = torch.stack(rows)
    return out


def _compare_sharded_to_unsharded(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
) -> list[str]:
    sharder = cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder())
    env = ShardingEnv.from_process_group(none_throws(ctx.pg))
    generators = {
        "tw": lambda: table_wise(rank=world_size - 1),
        "rw": lambda: row_wise(),
        "twrw": lambda: table_row_wise(host_index=0),
        "cw": lambda: column_wise(ranks=list(range(world_size))),
        "dp": lambda: data_parallel(),
    }
    failures: list[str] = []
    for case_name, case in CASES.items():
        tables = [
            EmbeddingBagConfig(
                name=t.name,
                feature_names=t.features,
                embedding_dim=DIM,
                num_embeddings=ROWS,
                pooling=t.pooling,
            )
            for t in case.tables
        ]
        unsharded = EmbeddingBagCollection(
            tables=tables, is_weighted=case.weighted, device=ctx.device
        )
        weight_gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for weight in unsharded.state_dict().values():
                weight.copy_(torch.rand(weight.shape, generator=weight_gen))

        model = _Wrapper(
            EmbeddingBagCollection(
                tables, is_weighted=case.weighted, device=torch.device("meta")
            )
        )
        plan = construct_module_sharding_plan(
            model.ebc,
            per_param_sharding={t.name: generators[t.sharding]() for t in case.tables},
            local_size=world_size,
            world_size=world_size,
            device_type="cuda",
            sharder=sharder,
        )
        sharded = DistributedModelParallel(
            model,
            env=env,
            device=ctx.device,
            plan=ShardingPlan({"ebc": plan}),
            sharders=[sharder],
        )
        with torch.no_grad():
            for key, weight in sharded.state_dict().items():
                full = unsharded.state_dict()[key.removeprefix("ebc.")]
                if isinstance(weight, ShardedTensor):
                    for shard in weight.local_shards():
                        r, c = shard.metadata.shard_offsets
                        h, w = shard.metadata.shard_sizes
                        shard.tensor.copy_(full[r : r + h, c : c + w])
                else:
                    weight.copy_(full)

        features, full_features = _features(case, rank)
        actual = sharded(features.to(ctx.device)).to_dict()
        if case.weighted:
            expected = _weighted_reference(
                tables, unsharded.state_dict(), full_features.to(ctx.device)
            )
        else:
            expected = unsharded(full_features.to(ctx.device)).to_dict()
        for name in expected:
            # rtol covers the eps=1e-6 added to every divisor.
            if not torch.allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6):
                max_abs = (actual[name] - expected[name]).abs().max().item()
                failures.append(f"{case_name}/{name}: max abs diff {max_abs:.3e}")
    return failures


class MeanPoolingDivisorUnitTest(unittest.TestCase):
    def test_row_sharded_only_divisor_unchanged(self) -> None:
        """With only RW / TWRW tables, the per-embedding divisor is bitwise equal
        to the original one (RW / TWRW SUM features reset to 1)."""
        features = ["a", "b", "c", "d"]
        is_mean = [False, True, False, True]
        dims = [8, 4, 16, 8]
        sum_features = [f for f, m in zip(features, is_mean) if not m]
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(0, 5, (len(features) * BATCH,), generator=gen)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        dim_per_key = torch.tensor(dims)
        for weights in (None, torch.rand(int(lengths.sum()), generator=gen)):
            divisor = _create_mean_pooling_divisor(
                lengths=lengths,
                keys=features,
                offsets=offsets,
                stride=BATCH,
                stride_per_key=[BATCH] * len(features),
                dim_per_key=dim_per_key,
                pooling_type_to_rs_features={
                    SUM.value: sum_features,
                    MEAN.value: [f for f in features if f not in sum_features],
                },
                embedding_names=features,
                embedding_dims=dims,
                variable_batch_per_feature=False,
                kjt_inverse_order=torch.arange(len(features)),
                kjt_key_indices={f: i for i, f in enumerate(features)},
                kt_key_ordering=torch.arange(len(features)),
                weights=weights,
                rs_mean_embedding_mask=torch.tensor(is_mean),
            )
            original = (
                lengths.clone()
                if weights is None
                else torch.ops.fbgemm.segment_sum_csr(1, offsets.int(), weights)
            )
            for i, feature in enumerate(features):
                if feature in sum_features:
                    original[i * BATCH : (i + 1) * BATCH] = 1
            original = original.reshape(-1, BATCH).T
            expected = torch.repeat_interleave(original, dim_per_key, dim=1) + 1e-6
            self.assertTrue(torch.equal(divisor, expected))


@skip_if_asan_class
class MeanPoolingDivisorTest(unittest.TestCase):
    """Sharded EBC must match the unsharded EBC for every embedding when RW or
    TWRW MEAN tables share the EBC with other sharding types or features."""

    maxDiff = None

    def _run(self, world_size: int) -> None:
        results = run_local_multi_process_func(
            _compare_sharded_to_unsharded, world_size=world_size, backend="nccl"
        )
        for rank, failures in enumerate(results):
            self.assertEqual(failures, [], f"rank {rank}")

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires a GPU")
    def test_single_rank(self) -> None:
        self._run(world_size=1)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_two_ranks(self) -> None:
        self._run(world_size=2)
