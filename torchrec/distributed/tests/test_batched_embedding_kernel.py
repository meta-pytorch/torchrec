#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import types
import unittest
from typing import Any, cast, List, Tuple
from unittest.mock import patch

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.ssd.ssd_config import KVZCHTBEConfig
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensorMetadata
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torchrec.distributed.batched_embedding_kernel import (
    _eviction_coordination_kwargs,
    _gen_named_parameters_by_table_dense,
    _gen_named_parameters_by_table_fused,
    _get_grid_shard_cw_count,
    _resolve_eviction_check_interval_batch,
    EmbeddingFusedOptimizer,
    KeyValueEmbeddingFusedOptimizer,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.sharding_plan import _calculate_grid_shard_sizes_and_offsets
from torchrec.modules.embedding_configs import DataType, PoolingType


def _grid_shards(
    row_sizes: List[int], num_cw_blocks: int, block_size: int
) -> List[ShardMetadata]:
    """Lay out `row_sizes` row shards within each of `num_cw_blocks` column blocks."""
    shards = []
    for block in range(num_cw_blocks):
        row_offset = 0
        for row_size in row_sizes:
            shards.append(
                ShardMetadata(
                    shard_sizes=[row_size, block_size],
                    shard_offsets=[row_offset, block * block_size],
                    placement=f"rank:{len(shards)}/cuda:0",
                )
            )
            row_offset += row_size
    return shards


class GridShardCwCountTest(unittest.TestCase):
    def test_counts_column_blocks_not_row_shards(self) -> None:
        # A pod of 8 ranks: 2 column blocks x 8 row shards of a 300M x 256 table.
        shards = _grid_shards([37_500_000] * 8, num_cw_blocks=2, block_size=128)
        self.assertEqual(len(shards), 16)
        self.assertEqual(_get_grid_shard_cw_count(shards), 2)

    def test_independent_of_row_shard_count(self) -> None:
        # Same table, same column split, a narrower pod: the answer must not move.
        shards = _grid_shards([150_000_000] * 2, num_cw_blocks=2, block_size=128)
        self.assertEqual(_get_grid_shard_cw_count(shards), 2)

    def test_uneven_rows_with_empty_trailing_shard(self) -> None:
        shards = _grid_shards([4, 4, 2, 0], num_cw_blocks=4, block_size=16)
        self.assertEqual(_get_grid_shard_cw_count(shards), 4)

    def test_single_column_block(self) -> None:
        shards = _grid_shards([8, 8], num_cw_blocks=1, block_size=64)
        self.assertEqual(_get_grid_shard_cw_count(shards), 1)


class GridShardOptimizerMetadataTest(unittest.TestCase):
    """Covers the sizing that raised 'Total volume of shards does not match tensor volume'."""

    def _rowwise_metadata(
        self, rows_per_shard: int, shards_per_block: int, num_cw_blocks: int
    ) -> Tuple[ShardedTensorMetadata, int]:
        # Rows are derived so the row shards tile the table exactly, which is what
        # the planner emits and what the volume check demands.
        rows = rows_per_shard * shards_per_block
        shards = _grid_shards(
            [rows_per_shard] * shards_per_block,
            num_cw_blocks=num_cw_blocks,
            block_size=128,
        )
        table_md = ShardedTensorMetadata(
            shards_metadata=shards,
            size=torch.Size([rows, 128 * num_cw_blocks]),
            tensor_properties=TensorProperties(dtype=torch.float32),
        )
        # Unbound: the method never touches `self`.
        _, rowwise_md = (
            KeyValueEmbeddingFusedOptimizer.get_optimizer_rowwise_shard_metadata_and_global_metadata(
                cast(KeyValueEmbeddingFusedOptimizer, None),
                table_md,
                torch.empty(0),
                1,
                True,
            )
        )
        return rowwise_md, rows

    def test_size_covers_shards_exactly(self) -> None:
        # The runtime invariant checked by check_tensor(): declared size must equal
        # the summed shard volume, or ShardedTensor construction raises.
        md, rows = self._rowwise_metadata(
            rows_per_shard=37, shards_per_block=8, num_cw_blocks=2
        )
        self.assertEqual(md.size[0], sum(s.shard_sizes[0] for s in md.shards_metadata))
        self.assertEqual(md.size, torch.Size([rows * 2]))

    def test_size_independent_of_row_shards_per_block(self) -> None:
        # Same table width and column split, a narrower pod: size must not move.
        md, rows = self._rowwise_metadata(
            rows_per_shard=148, shards_per_block=2, num_cw_blocks=2
        )
        self.assertEqual(md.size[0], sum(s.shard_sizes[0] for s in md.shards_metadata))
        self.assertEqual(md.size, torch.Size([rows * 2]))


class GridShardPlannerLayoutTest(unittest.TestCase):
    """Drives the planner's own shard generator, so a layout change breaks these."""

    # 4 hosts x 2 GPUs per host. The planner spreads row shards over the whole pod
    # (Topology.intra_group_size), so per-host GPU count is not the row-shard count.
    HOSTS_PER_POD = 4
    GPUS_PER_HOST = 2
    POD_SIZE = HOSTS_PER_POD * GPUS_PER_HOST

    def _planner_shards(
        self, rows: int, columns: int, col_wise_shard_dim: int
    ) -> List[ShardMetadata]:
        shard_sizes, shard_offsets = _calculate_grid_shard_sizes_and_offsets(
            rows, self.POD_SIZE, columns, col_wise_shard_dim
        )
        return [
            ShardMetadata(
                shard_sizes=sizes,
                shard_offsets=offsets,
                placement=f"rank:{rank}/cuda:0",
            )
            for rank, (sizes, offsets) in enumerate(zip(shard_sizes, shard_offsets))
        ]

    def test_cw_count_on_multi_host_pod(self) -> None:
        shards = self._planner_shards(
            rows=300_000_000, columns=256, col_wise_shard_dim=128
        )
        self.assertEqual(len(shards), 2 * self.POD_SIZE)
        self.assertEqual(_get_grid_shard_cw_count(shards), 2)
        # Must not track the per-host GPU count, which would give 8 here.
        self.assertNotEqual(
            _get_grid_shard_cw_count(shards), len(shards) // self.GPUS_PER_HOST
        )

    def test_optimizer_size_covers_shards_on_multi_host_pod(self) -> None:
        rows, columns = 300_000_000, 256
        shards = self._planner_shards(rows, columns, col_wise_shard_dim=128)
        table_md = ShardedTensorMetadata(
            shards_metadata=shards,
            size=torch.Size([rows, columns]),
            tensor_properties=TensorProperties(dtype=torch.float32),
        )
        _, rowwise_md = (
            KeyValueEmbeddingFusedOptimizer.get_optimizer_rowwise_shard_metadata_and_global_metadata(
                cast(KeyValueEmbeddingFusedOptimizer, None),
                table_md,
                torch.empty(0),
                1,
                True,
            )
        )
        self.assertEqual(
            rowwise_md.size[0],
            sum(s.shard_sizes[0] for s in rowwise_md.shards_metadata),
        )
        self.assertEqual(rowwise_md.size, torch.Size([rows * 2]))


class GridShardParameterSliceTest(unittest.TestCase):
    def _config(
        self, row_sizes: List[int], kernel: EmbeddingComputeKernel
    ) -> GroupedEmbeddingConfig:
        tables = []
        for name, sizes in (("table", row_sizes), ("neighbor", [9])):
            shards = []
            row_offset = 0
            for rows in sizes:
                shards.append(
                    ShardMetadata(
                        shard_sizes=[rows, 32],
                        shard_offsets=[row_offset, 0],
                        placement="rank:0/cpu",
                    )
                )
                row_offset += rows
            metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size([sum(sizes), 32]),
                tensor_properties=TensorProperties(dtype=torch.float32),
            )
            for shard in shards:
                tables.append(
                    ShardedEmbeddingTable(
                        name=name,
                        num_embeddings=sum(sizes),
                        embedding_dim=32,
                        feature_names=[name],
                        local_rows=shard.shard_sizes[0],
                        local_cols=32,
                        local_metadata=shard,
                        global_metadata=metadata,
                        compute_kernel=kernel,
                    )
                )
        return GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=kernel,
            embedding_tables=tables,
        )

    def _check_slices(
        self,
        row_sizes: List[int],
        fused: bool,
        precision: SparseType = SparseType.FP32,
        row_slicing_enabled: bool = True,
    ) -> None:
        all_rows = row_sizes + [9]
        counts = {"table": len(row_sizes), "neighbor": 1}
        if fused:
            fused_module = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (rows, 32, EmbeddingLocation.HOST, ComputeDevice.CPU)
                    for rows in all_rows
                ],
                optimizer=EmbOptimType.EXACT_SGD,
                weights_precision=precision,
                device=torch.device("cpu"),
            )
            native_weights = fused_module.split_embedding_weights()
            config = self._config(row_sizes, EmbeddingComputeKernel.FUSED)
            with patch(
                "torch._utils_internal.justknobs_check",
                return_value=row_slicing_enabled,
            ):
                parameters = dict(
                    _gen_named_parameters_by_table_fused(fused_module, counts, config)
                )
            for weight in parameters.values():
                optimizers = cast(
                    List[EmbeddingFusedOptimizer],
                    vars(weight)["_in_backward_optimizers"],
                )
                self.assertEqual(len(optimizers), 1)
                self.assertIs(optimizers[0].params[""], weight)
                self.assertIs(optimizers[0]._emb_module, fused_module)
        else:
            dense_module = DenseTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[(rows, 32) for rows in all_rows],
                use_cpu=True,
            )
            native_weights = dense_module.split_embedding_weights()
            config = self._config(row_sizes, EmbeddingComputeKernel.DENSE)
            with patch(
                "torch._utils_internal.justknobs_check",
                return_value=row_slicing_enabled,
            ):
                parameters = dict(
                    _gen_named_parameters_by_table_dense(dense_module, counts, config)
                )
        with torch.no_grad():
            for index, weight in enumerate(native_weights):
                weight.fill_(index + 1)
        saved_table = torch.cat(native_weights[:-1]).clone()
        saved_neighbor = native_weights[-1].clone()
        torch.testing.assert_close(parameters["table"], saved_table)
        torch.testing.assert_close(parameters["neighbor"], saved_neighbor)
        self.assertEqual(parameters["table"].shape[0], sum(row_sizes))

        model = torch.nn.Module()
        for name, weight in parameters.items():
            model.register_parameter(name, weight)
        with torch.no_grad():
            parameters["table"].fill_(42)
        torch.testing.assert_close(native_weights[-1], saved_neighbor)
        model.load_state_dict({"table": saved_table, "neighbor": saved_neighbor})
        torch.testing.assert_close(torch.cat(native_weights[:-1]), saved_table)
        torch.testing.assert_close(native_weights[-1], saved_neighbor)

    def test_dense_slices_and_checkpoint_load(self) -> None:
        for rows in ([7, 1], [7, 7, 3], [7, 7], [7]):
            with self.subTest(rows=rows):
                self._check_slices(rows, fused=False)

    def test_fused_slices_and_checkpoint_load(self) -> None:
        for rows in ([7, 1], [7, 7, 3], [7, 7], [7]):
            with self.subTest(rows=rows):
                self._check_slices(rows, fused=True)

    def test_fused_int8_slices_include_quantization_bytes(self) -> None:
        self._check_slices([7, 1], fused=True, precision=SparseType.INT8)

    def test_uniform_slices_with_row_slicing_disabled(self) -> None:
        for fused in (False, True):
            for rows in ([7, 7], [7]):
                with self.subTest(fused=fused, rows=rows):
                    self._check_slices(rows, fused=fused, row_slicing_enabled=False)


# OSS CI pairs TorchRec with the fbgemm_gpu nightly, which can predate these
# KVZCHTBEConfig fields.
_needs_eviction_coordination_fields = unittest.skipIf(
    "eviction_check_interval_batch" not in KVZCHTBEConfig._fields,
    "fbgemm_gpu predates the coordinated eviction fields of KVZCHTBEConfig",
)


class EvictionCheckIntervalResolutionTest(unittest.TestCase):
    """`eviction_free_mem_check_interval_batch` is honoured only for free_mem."""

    @staticmethod
    def _config(**kwargs: Any) -> KVZCHTBEConfig:
        return KVZCHTBEConfig(**kwargs)

    @_needs_eviction_coordination_fields
    def test_canonical_field_wins_over_the_deprecated_alias(self) -> None:
        resolved = _resolve_eviction_check_interval_batch(
            self._config(
                kvzch_eviction_trigger_mode=5,
                eviction_check_interval_batch=7,
                eviction_free_mem_check_interval_batch=1000,
            )
        )
        self.assertEqual(resolved, 7)

    def test_alias_is_honoured_for_free_mem(self) -> None:
        resolved = _resolve_eviction_check_interval_batch(
            self._config(
                kvzch_eviction_trigger_mode=5,
                eviction_free_mem_check_interval_batch=1000,
            )
        )
        self.assertEqual(resolved, 1000)

    def test_unset_alias_does_not_pin_a_cadence(self) -> None:
        # The alias defaults to None, so a config that never mentions it must
        # resolve to "take the per-mode default" without warning about a field
        # nobody set.
        resolved = _resolve_eviction_check_interval_batch(
            self._config(kvzch_eviction_trigger_mode=2)
        )
        self.assertIsNone(resolved)

    def test_alias_is_ignored_for_the_budgeted_modes(self) -> None:
        # Configs that set the alias to 1000 under mode 2 never had it read --
        # those tables were checked on every prefetch -- so honouring it now
        # would be a 1000x coarsening.
        # None means "take the per-mode default in training.py".
        for mode in (1, 2, 4):
            with self.subTest(trigger_mode=mode):
                resolved = _resolve_eviction_check_interval_batch(
                    self._config(
                        kvzch_eviction_trigger_mode=mode,
                        eviction_free_mem_check_interval_batch=1000,
                    )
                )
                self.assertIsNone(resolved)


class EvictionCoordinationKwargsTest(unittest.TestCase):
    """OSS TorchRec can run against an fbgemm_gpu without these fields."""

    _NEW_FIELDS: Tuple[str, ...] = (
        "eviction_check_interval_batch",
        "eviction_insufficient_backoff_batch",
        "eviction_free_mem_floor_gb",
    )

    @_needs_eviction_coordination_fields
    def test_set_fields_are_passed(self) -> None:
        kwargs = _eviction_coordination_kwargs(
            KVZCHTBEConfig(
                kvzch_eviction_trigger_mode=4,
                eviction_check_interval_batch=7,
                eviction_free_mem_floor_gb=0,
            ),
            self._NEW_FIELDS,
        )
        self.assertEqual(
            kwargs,
            {"eviction_check_interval_batch": 7, "eviction_free_mem_floor_gb": 0},
        )

    def test_config_without_the_fields_passes_nothing(self) -> None:
        old_config = types.SimpleNamespace(
            kvzch_eviction_trigger_mode=4, eviction_free_mem_check_interval_batch=None
        )
        self.assertEqual(
            _eviction_coordination_kwargs(old_config, self._NEW_FIELDS), {}
        )

    def test_old_eviction_policy_gets_the_deprecated_cadence(self) -> None:
        old_config = types.SimpleNamespace(
            kvzch_eviction_trigger_mode=5, eviction_free_mem_check_interval_batch=1000
        )
        self.assertEqual(
            _eviction_coordination_kwargs(old_config, ("eviction_trigger_mode",)),
            {"eviction_free_mem_check_interval_batch": 1000},
        )
