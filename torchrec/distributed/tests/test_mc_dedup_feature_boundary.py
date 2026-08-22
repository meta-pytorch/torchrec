#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.mc_embedding import (
    ManagedCollisionEmbeddingCollectionSharder,
    ShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import (
    _restore_dedup_feature_boundary,
    ManagedCollisionCollectionContext,
    ManagedCollisionCollectionSharder,
    ShardedManagedCollisionCollection,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

_HASH_SIZE = 100_000
_STRIDE = 2  # batch size

WORLD_SIZE = 2
_ZCH_SIZE = 128
_TABLE = "table_0"
_WRITE_FEATURE = "media"
_READ_ONLY_FEATURE = "media_readonly"
_NUM_WRITE_IDS = 4
_NUM_READ_ONLY_IDS = 40


def _dedup(
    values: torch.Tensor, lengths: torch.Tensor, num_features: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs the dedup op with the buffers ``_create_dedup_indices`` builds for a single
    table binding ``num_features`` features: one shared hash range, one feature group.
    """
    device = values.device
    hash_offsets = torch.tensor(
        [0] * num_features + [_HASH_SIZE], dtype=torch.int64, device=device
    )
    feature_offsets = torch.tensor(
        [0] * num_features + [num_features], dtype=torch.int64, device=device
    )
    lengths_, _, unique_indices, reverse_indices = (
        torch.ops.fbgemm.jagged_unique_indices(
            hash_offsets,
            feature_offsets,
            torch.ops.fbgemm.asynchronous_complete_cumsum(lengths),
            values,
        )
    )
    return lengths_, unique_indices, reverse_indices


def _per_feature(lengths: torch.Tensor, num_features: int) -> List[int]:
    return lengths.view(num_features, -1).sum(dim=1).tolist()


def _lengths(counts: List[int], device: torch.device) -> torch.Tensor:
    """Per-feature totals -> a feature major KJT lengths tensor of stride _STRIDE."""
    return torch.tensor(
        [
            count // _STRIDE + (1 if b < count % _STRIDE else 0)
            for count in counts
            for b in range(_STRIDE)
        ],
        dtype=torch.int64,
        device=device,
    )


class _DedupOnlyCollection:
    """
    Carries just the state ``ShardedManagedCollisionCollection._dedup_indices`` reads
    off ``self``, so the method can be exercised without standing up a sharded module.
    """

    def __init__(
        self, num_features: int, device: torch.device, restore: bool = True
    ) -> None:
        self._buffers: Dict[str, torch.Tensor] = {
            "_dedup_hash_offsets_0": torch.tensor(
                [0] * num_features + [_HASH_SIZE], dtype=torch.int64, device=device
            ),
            "_dedup_feature_offsets_0": torch.tensor(
                [0] * num_features + [num_features], dtype=torch.int64, device=device
            ),
        }
        self._dedup_needs_boundary_restore: List[bool] = [restore and num_features > 1]

    def get_buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]


@unittest.skipIf(
    not torch.cuda.is_available(), "fbgemm.jagged_unique_indices is CUDA only"
)
class McDedupFeatureBoundaryTest(unittest.TestCase):
    """
    A managed collision table that binds more than one feature gives all of its
    features one shared hash range, so ``fbgemm.jagged_unique_indices`` dedups them as
    a single group and cannot say which feature a surviving row came from. These tests
    cover the boundary that ``_restore_dedup_feature_boundary`` puts back.

    Shapes mirror a ranking model where a HASH_ZCH table binds a short write feature
    of candidate ids and a long read-only feature of user history.
    """

    def setUp(self) -> None:
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.ids: torch.Tensor = torch.randperm(_HASH_SIZE, device=self.device)

    def _run_dedup_indices(
        self, kjt: KeyedJaggedTensor, restore: bool
    ) -> Tuple[KeyedJaggedTensor, ManagedCollisionCollectionContext]:
        ctx = ManagedCollisionCollectionContext(sharding_contexts=[])
        dedup_kjt = ShardedManagedCollisionCollection._dedup_indices(
            cast(
                ShardedManagedCollisionCollection,
                _DedupOnlyCollection(len(kjt.keys()), self.device, restore=restore),
            ),
            ctx,
            [kjt],
        )[0]
        return dedup_kjt, ctx

    def test_dedup_op_loses_feature_boundary(self) -> None:
        """
        Pins the upstream behavior being corrected: the op returns the group's unique
        count spread evenly over the group's slots rather than the real per-feature
        counts. If this fails, the op has learned per-feature attribution and
        ``_restore_dedup_feature_boundary`` can be dropped.
        """
        write_len, read_len = 33, 1280
        values = self.ids[: write_len + read_len]
        lengths, unique_indices, _ = _dedup(
            values, _lengths([write_len, read_len], self.device), 2
        )

        self.assertEqual(unique_indices.numel(), write_len + read_len)
        self.assertEqual(_per_feature(lengths, 2), [657, 656])

    def test_restore_feature_boundary(self) -> None:
        write_len, read_len = 33, 1280
        values = self.ids[: write_len + read_len]
        input_lengths = _lengths([write_len, read_len], self.device)
        _, unique_indices, reverse_indices = _dedup(values, input_lengths, 2)

        lengths, unique_indices, reverse_indices = _restore_dedup_feature_boundary(
            input_lengths, 2, unique_indices, reverse_indices
        )

        self.assertEqual(_per_feature(lengths, 2), [write_len, read_len])
        # the write feature's segment holds exactly the ids it sent, nothing borrowed
        # from the read-only feature
        torch.testing.assert_close(
            unique_indices[:write_len].sort().values, values[:write_len].sort().values
        )
        torch.testing.assert_close(unique_indices[reverse_indices], values)

        # the 2D weights of the write path are gathered through reverse_indices, so
        # they have to survive the regrouping too
        weights = torch.rand(values.numel(), 4, device=self.device)
        dedup_weights = torch.empty(
            unique_indices.numel(), 4, dtype=weights.dtype, device=self.device
        )
        dedup_weights[reverse_indices] = weights
        torch.testing.assert_close(dedup_weights[reverse_indices], weights)

    def test_restore_feature_boundary_with_duplicate_and_shared_ids(self) -> None:
        """
        Dedup still collapses repeats, including ids sent by both features. A shared id
        is attributed to the first feature that carried it, which is the write feature,
        so a candidate that is also in history is still inserted rather than gated.
        """
        write_len, hist_len, shared_len = 33, 600, 5
        write_ids = self.ids[:write_len]
        hist_ids = self.ids[write_len : write_len + hist_len]
        read_values = torch.cat([hist_ids, hist_ids, write_ids[:shared_len]])
        values = torch.cat([write_ids, read_values])
        input_lengths = _lengths([write_len, read_values.numel()], self.device)
        _, unique_indices, reverse_indices = _dedup(values, input_lengths, 2)

        lengths, unique_indices, reverse_indices = _restore_dedup_feature_boundary(
            input_lengths, 2, unique_indices, reverse_indices
        )

        self.assertEqual(unique_indices.numel(), write_len + hist_len)
        self.assertEqual(_per_feature(lengths, 2), [write_len, hist_len])
        torch.testing.assert_close(
            unique_indices[:write_len].sort().values, write_ids.sort().values
        )
        torch.testing.assert_close(unique_indices[reverse_indices], values)

    def test_dedup_indices_keeps_feature_boundary(self) -> None:
        """End to end through ``_dedup_indices``, covering the wiring and the gate."""
        write_len, read_len = 33, 1280
        kjt = KeyedJaggedTensor(
            keys=[_WRITE_FEATURE, _READ_ONLY_FEATURE],
            values=self.ids[: write_len + read_len],
            lengths=_lengths([write_len, read_len], self.device),
        )

        dedup_kjt, ctx = self._run_dedup_indices(kjt, restore=True)

        self.assertEqual(dedup_kjt.length_per_key(), [write_len, read_len])
        torch.testing.assert_close(
            dedup_kjt[_WRITE_FEATURE].values().sort().values,
            kjt[_WRITE_FEATURE].values().sort().values,
        )
        torch.testing.assert_close(
            dedup_kjt.values()[ctx.reverse_indices[0]], kjt.values()
        )

    def test_dedup_indices_skips_restore_when_disabled(self) -> None:
        """
        With restore_dedup_feature_boundary off, _dedup_indices leaves the op's even
        split alone. Pins that the knob actually gates the work.
        """
        write_len, read_len = 33, 1280
        kjt = KeyedJaggedTensor(
            keys=[_WRITE_FEATURE, _READ_ONLY_FEATURE],
            values=self.ids[: write_len + read_len],
            lengths=_lengths([write_len, read_len], self.device),
        )

        dedup_kjt, _ = self._run_dedup_indices(kjt, restore=False)

        self.assertEqual(dedup_kjt.length_per_key(), [657, 656])

    def test_restore_feature_boundary_with_three_features(self) -> None:
        counts = [11, 517, 42]
        values = self.ids[: sum(counts)]
        input_lengths = _lengths(counts, self.device)
        _, unique_indices, reverse_indices = _dedup(values, input_lengths, 3)

        lengths, unique_indices, reverse_indices = _restore_dedup_feature_boundary(
            input_lengths, 3, unique_indices, reverse_indices
        )

        self.assertEqual(_per_feature(lengths, 3), counts)
        torch.testing.assert_close(unique_indices[reverse_indices], values)


class _ReadOnlyFeatureSparseArch(nn.Module):
    """
    One HASH_ZCH table bound to a write feature and a read-only feature, the binding
    that loses its per-feature boundary under index dedup.
    """

    def __init__(self, table: EmbeddingConfig, device: torch.device) -> None:
        super().__init__()
        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(tables=[table], device=device),
                ManagedCollisionCollection(
                    managed_collision_modules={
                        table.name: HashZchManagedCollisionModule(
                            zch_size=table.num_embeddings,
                            device=device,
                            total_num_buckets=WORLD_SIZE,
                            input_hash_size=0,
                            enable_per_feature_lookups=True,
                            read_only_suffix="_readonly",
                        )
                    },
                    embedding_configs=[table],
                ),
            )
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ec_out, _ = self._mc_ec(kjt)
        return torch.cat([ec_out[key].values() for key in kjt.keys()]).sum()


def _test_read_only_feature_is_never_inserted(
    rank: int,
    world_size: int,
    table: EmbeddingConfig,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Build the real module, run a lookup, then read the ZCH identity buffer.

    Only the candidates the write feature sent may be inserted. Without the boundary
    restore the even split moves roughly half the read-only history ids under the
    write feature, which inserts them and is what filled the table up in production.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sparse_arch = _ReadOnlyFeatureSparseArch(table, torch.device("meta"))
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            # pyrefly: ignore[bad-argument-type]
            [sparse_arch._mc_ec._embedding_collection.embeddings[_TABLE].weight],
            {"lr": 0.01},
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._mc_ec,
            per_param_sharding={_TABLE: row_wise(num_buckets=world_size)},
            local_size=ctx.local_size,
            world_size=world_size,
            device_type="cuda",
            sharder=sharder,
        )
        sharded_sparse_arch = _shard_modules(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        # this runs in a worker process, so there is no TestCase instance to use;
        # a bare one still reports failures as test failures rather than as
        # AssertionError from library code
        tc = unittest.TestCase()

        mc_ec = sharded_sparse_arch._mc_ec
        tc.assertIsInstance(mc_ec, ShardedManagedCollisionEmbeddingCollection)
        mcc = cast(
            ShardedManagedCollisionEmbeddingCollection, mc_ec
        )._managed_collision_collection
        tc.assertTrue(mcc._use_index_dedup, "this test needs index dedup enabled")
        tc.assertTrue(
            mcc._restore_dedup_feature_boundary,
            "this test needs the boundary restore enabled",
        )

        # a short write feature next to a long read-only one: that length gap is what
        # the even split smears across the feature boundary
        write_ids = [1000 + rank * 100 + i for i in range(_NUM_WRITE_IDS)]
        read_only_ids = [5000 + rank * 1000 + i for i in range(_NUM_READ_ONLY_IDS)]
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[_WRITE_FEATURE, _READ_ONLY_FEATURE],
            values=torch.tensor(write_ids + read_only_ids, dtype=torch.int64),
            lengths=torch.tensor(
                [len(write_ids), len(read_only_ids)], dtype=torch.int64
            ),
        ).to(ctx.device)

        sharded_sparse_arch(kjt)

        identities = mcc._managed_collision_modules[_TABLE]._hash_zch_identities
        tc.assertIsInstance(identities, torch.Tensor)
        occupied = (cast(torch.Tensor, identities) != -1).sum()
        dist.all_reduce(occupied, group=ctx.pg)
        expected = world_size * _NUM_WRITE_IDS
        tc.assertEqual(
            int(occupied),
            expected,
            f"{int(occupied)} slots written, expected {expected} (the candidates only)",
        )


@skip_if_asan_class
class McDedupReadOnlyFeatureTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() < WORLD_SIZE,
        f"needs {WORLD_SIZE} GPUs",
    )
    def test_read_only_feature_is_never_inserted(self) -> None:
        self._run_multi_process_test(
            callable=_test_read_only_feature_is_never_inserted,
            world_size=WORLD_SIZE,
            table=EmbeddingConfig(
                name=_TABLE,
                num_embeddings=_ZCH_SIZE,
                embedding_dim=8,
                feature_names=[_WRITE_FEATURE, _READ_ONLY_FEATURE],
            ),
            sharder=ManagedCollisionEmbeddingCollectionSharder(
                ec_sharder=EmbeddingCollectionSharder(use_index_dedup=True),
                mc_sharder=ManagedCollisionCollectionSharder(
                    restore_dedup_feature_boundary=True
                ),
            ),
            backend="nccl",
        )
