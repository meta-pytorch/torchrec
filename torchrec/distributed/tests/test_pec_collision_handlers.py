#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from typing import List

import torch
import torch.distributed as dist
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.pec_collision_handlers import (
    BooleanOverlapChecker,
    create_overlap_handler,
    OverlapMasks,
    RWOverlapHandler,
    split_kjt_by_values_mask,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.pec_embedding_modules import OverlappingCheckerType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port


def _get_single_rank_pg() -> dist.ProcessGroup:
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return dist.group.WORLD  # pyre-ignore[7]


def tearDownModule() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_grouped_configs(
    tables: List[EmbeddingConfig],
    local_rows: int,
) -> List[GroupedEmbeddingConfig]:
    """Create GroupedEmbeddingConfig with ShardedEmbeddingTables for testing."""
    emb_tables = []
    for table in tables:
        shard_table = ShardedEmbeddingTable(
            name=table.name,
            embedding_dim=table.embedding_dim,
            num_embeddings=table.num_embeddings,
            feature_names=table.feature_names,
            data_type=table.data_type,
            has_feature_processor=False,
            local_rows=local_rows,
            local_cols=table.embedding_dim,
            compute_kernel=None,  # pyre-ignore[6]
            local_metadata=None,
            global_metadata=None,
            weight_init_max=None,
            weight_init_min=None,
        )
        emb_tables.append(shard_table)
    return [
        GroupedEmbeddingConfig(
            data_type=tables[0].data_type,
            pooling=None,  # pyre-ignore[6]
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=EmbeddingComputeKernel.DENSE,
            embedding_tables=emb_tables,
        )
    ]


def _make_table_name_to_config(
    tables: List[EmbeddingConfig],
) -> dict[str, EmbeddingConfig]:
    return {t.name: t for t in tables}


class BooleanOverlapCheckerTest(unittest.TestCase):
    """Unit tests for BooleanOverlapChecker."""

    def setUp(self) -> None:
        self.checker = BooleanOverlapChecker(torch.device("cpu"), mask_size=10)

    def test_no_overlap(self) -> None:
        prev = torch.tensor([0, 1, 2], dtype=torch.long)
        current = torch.tensor([3, 4, 5], dtype=torch.long)

        fwd_mask, bwd_mask = self.checker.check(current, prev)

        self.assertTrue(torch.equal(fwd_mask, torch.tensor([False, False, False])))
        self.assertTrue(torch.equal(bwd_mask, torch.tensor([False, False, False])))

    def test_full_overlap(self) -> None:
        prev = torch.tensor([1, 3, 5], dtype=torch.long)
        current = torch.tensor([1, 3, 5], dtype=torch.long)

        fwd_mask, bwd_mask = self.checker.check(current, prev)

        self.assertTrue(torch.equal(fwd_mask, torch.tensor([True, True, True])))
        self.assertTrue(torch.equal(bwd_mask, torch.tensor([True, True, True])))

    def test_partial_overlap(self) -> None:
        prev = torch.tensor([1, 3, 5], dtype=torch.long)
        current = torch.tensor([1, 3, 7], dtype=torch.long)

        fwd_mask, bwd_mask = self.checker.check(current, prev)

        self.assertTrue(torch.equal(fwd_mask, torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(bwd_mask, torch.tensor([True, True, False])))

    def test_asymmetric_overlap(self) -> None:
        prev = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        current = torch.tensor([1, 4], dtype=torch.long)

        fwd_mask, bwd_mask = self.checker.check(current, prev)

        self.assertTrue(torch.equal(fwd_mask, torch.tensor([True, False])))
        self.assertTrue(
            torch.equal(bwd_mask, torch.tensor([False, True, False, False]))
        )


class SplitFeaturesTest(unittest.TestCase):
    """Unit tests for the standalone split_kjt_by_values_mask function."""

    def test_all_nonoverlapped(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f0", "f1"],
            values=torch.tensor([10, 20, 30, 40]),
            lengths=torch.tensor([2, 0, 1, 1]),
        )
        mask = torch.tensor([False, False, False, False])

        ol, nol = split_kjt_by_values_mask(kjt, mask)

        self.assertEqual(ol.values().numel(), 0)
        self.assertTrue(torch.equal(ol.lengths(), torch.tensor([0, 0, 0, 0])))
        self.assertEqual(ol.keys(), ["f0", "f1"])
        self.assertTrue(torch.equal(nol.values(), torch.tensor([10, 20, 30, 40])))
        self.assertTrue(torch.equal(nol.lengths(), torch.tensor([2, 0, 1, 1])))

    def test_all_overlapped(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f0", "f1"],
            values=torch.tensor([10, 20, 30, 40]),
            lengths=torch.tensor([2, 0, 1, 1]),
        )
        mask = torch.tensor([True, True, True, True])

        ol, nol = split_kjt_by_values_mask(kjt, mask)

        self.assertTrue(torch.equal(ol.values(), torch.tensor([10, 20, 30, 40])))
        self.assertTrue(torch.equal(ol.lengths(), torch.tensor([2, 0, 1, 1])))
        self.assertEqual(nol.values().numel(), 0)
        self.assertTrue(torch.equal(nol.lengths(), torch.tensor([0, 0, 0, 0])))
        self.assertEqual(nol.keys(), ["f0", "f1"])

    def test_partial_overlap(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f0", "f1"],
            values=torch.tensor([10, 20, 30, 40, 50, 60]),
            lengths=torch.tensor([2, 1, 1, 2]),
        )
        mask = torch.tensor([False, True, False, False, True, False])

        ol, nol = split_kjt_by_values_mask(kjt, mask)

        self.assertTrue(torch.equal(ol.values(), torch.tensor([20, 50])))
        self.assertTrue(torch.equal(ol.lengths(), torch.tensor([1, 0, 0, 1])))
        self.assertTrue(torch.equal(nol.values(), torch.tensor([10, 30, 40, 60])))
        self.assertTrue(torch.equal(nol.lengths(), torch.tensor([1, 1, 1, 1])))

    def test_with_weights(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f0"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([3]),
            weights=torch.tensor([0.1, 0.2, 0.3]),
        )
        mask = torch.tensor([False, True, False])

        ol, nol = split_kjt_by_values_mask(kjt, mask)

        self.assertTrue(torch.equal(ol.values(), torch.tensor([2])))
        self.assertTrue(torch.allclose(ol.weights(), torch.tensor([0.2])))
        self.assertTrue(torch.equal(nol.values(), torch.tensor([1, 3])))
        self.assertTrue(torch.allclose(nol.weights(), torch.tensor([0.1, 0.3])))

    def test_preserves_keys(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_a", "feature_b"],
            values=torch.tensor([1, 2, 3, 4]),
            lengths=torch.tensor([2, 2]),
        )
        mask = torch.tensor([True, False, False, True])

        ol, nol = split_kjt_by_values_mask(kjt, mask)

        self.assertEqual(ol.keys(), ["feature_a", "feature_b"])
        self.assertEqual(nol.keys(), ["feature_a", "feature_b"])


class RWOverlapHandlerTest(unittest.TestCase):
    """Unit tests for RWOverlapHandler."""

    def setUp(self) -> None:
        self.tables = [
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
                num_embeddings=16,
            ),
        ]
        self.handler = self._create_handler(self.tables)

    def _create_handler(
        self,
        tables: List[EmbeddingConfig],
        local_rows: int = 8,
    ) -> RWOverlapHandler:
        return create_overlap_handler(  # pyre-ignore[7]
            sharding_type="row_wise",
            device=torch.device("cpu"),
            grouped_emb_configs=_make_grouped_configs(tables, local_rows),
            table_name_to_config=_make_table_name_to_config(tables),
            process_group=_get_single_rank_pg(),
            checker_type=OverlappingCheckerType.BOOLEAN,
        )

    def test_remap_kjt_values(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3]),
            lengths=torch.tensor([2, 2]),
        )

        remapped = self.handler.remap_kjt_values(kjt)

        self.assertTrue(torch.equal(remapped, torch.tensor([0, 1, 10, 11])))

    def test_first_batch_no_overlap(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3]),
            lengths=torch.tensor([2, 2]),
        )
        remapped = self.handler.remap_kjt_values(kjt)

        result = self.handler.detect_overlap(remapped, prev_remapped=None)

        self.assertIsInstance(result, OverlapMasks)
        assert result.forward_overlap_mask is not None
        self.assertTrue(
            torch.equal(result.forward_overlap_mask, torch.zeros(4, dtype=torch.bool))
        )
        self.assertIsNone(result.backward_overlap_mask)

    def test_last_batch_all_ol(self) -> None:
        """Last batch: current=None → no forward mask, all-True backward."""
        prev_remapped = torch.tensor([0, 1, 10, 11], dtype=torch.long)

        result = self.handler.detect_overlap(
            current_remapped=None, prev_remapped=prev_remapped
        )

        self.assertIsNone(result.forward_overlap_mask)
        assert result.backward_overlap_mask is not None
        self.assertTrue(result.backward_overlap_mask.all())
        self.assertEqual(result.backward_overlap_mask.numel(), 4)

    def test_second_batch_with_overlap(self) -> None:
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3]),
            lengths=torch.tensor([2, 2]),
        )
        remapped1 = self.handler.remap_kjt_values(kjt1)

        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([1, 4, 3, 5]),
            lengths=torch.tensor([2, 2]),
        )
        remapped2 = self.handler.remap_kjt_values(kjt2)

        result = self.handler.detect_overlap(remapped2, prev_remapped=remapped1)

        assert result.forward_overlap_mask is not None
        self.assertTrue(
            torch.equal(
                result.forward_overlap_mask,
                torch.tensor([True, False, True, False]),
            )
        )
        assert result.backward_overlap_mask is not None
        self.assertTrue(
            torch.equal(
                result.backward_overlap_mask,
                torch.tensor([False, True, False, True]),
            )
        )

    def test_no_overlap_between_batches(self) -> None:
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3]),
            lengths=torch.tensor([2, 2]),
        )
        remapped1 = self.handler.remap_kjt_values(kjt1)

        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([4, 5, 6, 7]),
            lengths=torch.tensor([2, 2]),
        )
        remapped2 = self.handler.remap_kjt_values(kjt2)

        result = self.handler.detect_overlap(remapped2, prev_remapped=remapped1)

        assert result.forward_overlap_mask is not None
        self.assertFalse(result.forward_overlap_mask.any())
        assert result.backward_overlap_mask is not None
        self.assertFalse(result.backward_overlap_mask.any())

    def test_full_overlap_between_batches(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3]),
            lengths=torch.tensor([2, 2]),
        )
        remapped = self.handler.remap_kjt_values(kjt)

        result = self.handler.detect_overlap(remapped, prev_remapped=remapped)

        assert result.forward_overlap_mask is not None
        self.assertTrue(result.forward_overlap_mask.all())
        assert result.backward_overlap_mask is not None
        self.assertTrue(result.backward_overlap_mask.all())

    def test_variable_length_features(self) -> None:
        """Features with different lengths per batch element."""
        handler = self._create_handler(self.tables, local_rows=16)
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([3, 1, 1, 3]),
        )
        remapped1 = handler.remap_kjt_values(kjt1)

        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([2, 8, 5, 9]),
            lengths=torch.tensor([2, 2]),
        )
        remapped2 = handler.remap_kjt_values(kjt2)

        result = handler.detect_overlap(remapped2, prev_remapped=remapped1)

        assert result.forward_overlap_mask is not None
        self.assertTrue(
            torch.equal(
                result.forward_overlap_mask,
                torch.tensor([True, False, True, False]),
            )
        )
        assert result.backward_overlap_mask is not None
        # prev values [0,1,2,3,4,5,6,7] remapped to [0,1,2,3,20,21,22,23]
        # current values [2,8,5,9] remapped to [2,8,21,25]
        # backward: prev[i] in current? → [F,F,T,F, F,T,F,F]
        self.assertTrue(
            torch.equal(
                result.backward_overlap_mask,
                torch.tensor([False, False, True, False, False, True, False, False]),
            )
        )

    def test_duplicate_values_in_batch(self) -> None:
        """Duplicate values within a batch should still detect overlap correctly."""
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 0, 2, 2]),
            lengths=torch.tensor([2, 2]),
        )
        remapped1 = self.handler.remap_kjt_values(kjt1)

        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 3, 2]),
            lengths=torch.tensor([2, 2]),
        )
        remapped2 = self.handler.remap_kjt_values(kjt2)

        result = self.handler.detect_overlap(remapped2, prev_remapped=remapped1)

        assert result.forward_overlap_mask is not None
        # current [0,1,3,2] remapped [0,1,11,10]: 0 in prev, 1 not, 11 not, 10 in prev
        self.assertTrue(
            torch.equal(
                result.forward_overlap_mask,
                torch.tensor([True, False, False, True]),
            )
        )
        assert result.backward_overlap_mask is not None
        # prev [0,0,2,2] remapped [0,0,10,10]: 0 in current, 0 in current, 10 in current, 10 in current
        self.assertTrue(result.backward_overlap_mask.all())

    def test_empty_batch(self) -> None:
        """Empty KJT (all lengths=0) should produce empty masks."""
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([], dtype=torch.long),
            lengths=torch.tensor([0, 0]),
        )
        remapped = self.handler.remap_kjt_values(kjt)

        result = self.handler.detect_overlap(remapped, prev_remapped=None)

        assert result.forward_overlap_mask is not None
        self.assertEqual(result.forward_overlap_mask.numel(), 0)
        self.assertIsNone(result.backward_overlap_mask)

    def test_single_table(self) -> None:
        """Single table — offset is always 0."""
        handler = self._create_handler([self.tables[0]])
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.tensor([0, 1, 2]),
            lengths=torch.tensor([3]),
        )
        remapped1 = handler.remap_kjt_values(kjt1)

        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.tensor([1, 3]),
            lengths=torch.tensor([2]),
        )
        remapped2 = handler.remap_kjt_values(kjt2)

        result = handler.detect_overlap(remapped2, prev_remapped=remapped1)

        assert result.forward_overlap_mask is not None
        self.assertTrue(torch.equal(remapped1, torch.tensor([0, 1, 2])))
        self.assertTrue(
            torch.equal(result.forward_overlap_mask, torch.tensor([True, False]))
        )

    def test_multiple_features_per_table(self) -> None:
        """Two features sharing one table get the same offset."""
        tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_a", "feature_b"],
                embedding_dim=8,
                num_embeddings=16,
            ),
        ]
        handler = self._create_handler(tables)
        kjt1 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_a", "feature_b"],
            values=torch.tensor([0, 1, 2, 3]),
            lengths=torch.tensor([2, 2]),
        )
        remapped1 = handler.remap_kjt_values(kjt1)

        kjt2 = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_a", "feature_b"],
            values=torch.tensor([2, 5]),
            lengths=torch.tensor([1, 1]),
        )
        remapped2 = handler.remap_kjt_values(kjt2)

        result = handler.detect_overlap(remapped2, prev_remapped=remapped1)

        assert result.forward_overlap_mask is not None
        self.assertEqual(handler._feature_to_table_offset["feature_a"], 0)
        self.assertEqual(handler._feature_to_table_offset["feature_b"], 0)
        self.assertTrue(
            torch.equal(result.forward_overlap_mask, torch.tensor([True, False]))
        )


class CreateOverlapHandlerTest(unittest.TestCase):
    """Unit tests for the create_overlap_handler factory function."""

    def test_row_wise_returns_rw_handler(self) -> None:
        tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
        ]

        handler = create_overlap_handler(
            sharding_type="row_wise",
            device=torch.device("cpu"),
            grouped_emb_configs=_make_grouped_configs(tables, local_rows=8),
            table_name_to_config=_make_table_name_to_config(tables),
            process_group=_get_single_rank_pg(),
            checker_type=OverlappingCheckerType.BOOLEAN,
        )

        self.assertIsInstance(handler, RWOverlapHandler)

    def test_unsupported_sharding_type_raises(self) -> None:
        tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
        ]

        with self.assertRaises(ValueError):
            create_overlap_handler(
                sharding_type="column_wise",
                device=torch.device("cpu"),
                grouped_emb_configs=_make_grouped_configs(tables, local_rows=8),
                table_name_to_config=_make_table_name_to_config(tables),
                process_group=_get_single_rank_pg(),
                checker_type=OverlappingCheckerType.BOOLEAN,
            )
