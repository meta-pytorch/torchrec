#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import io
import unittest
from collections import defaultdict
from typing import Any, cast, Dict, List

import torch
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import KVEmbeddingInference
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
    FUSED_PARAM_TBE_ROW_ALIGNMENT,
)
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.quant_embedding_kernel import _get_grouped_int4_group_size
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    ParameterShardingGenerator,
    row_wise,
    table_wise,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.inference.modules import quantize_inference_model
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)

# Open-source CI installs a released fbgemm_gpu, which can predate grouped INT4
# KV rows; the tests that build KV modules need it.
skip_without_grouped_int4_fbgemm = unittest.skipUnless(
    "grouped_int4_group_size"
    in inspect.signature(KVEmbeddingInference.__init__).parameters,
    "fbgemm_gpu KVEmbeddingInference predates grouped_int4_group_size",
)

NUM_EMBEDDINGS = 64
EMBEDDING_DIM = 128
GROUP_SIZE = 8
# dim / 2 code bytes followed by dim / GROUP_SIZE float8_e4m3fn scale bytes.
GROUPED_ROW_BYTES = 80
WORLD_SIZE = 2
G8_TABLE = "g8_table"
ROWWISE_TABLE = "rowwise_table"
# The fused params of TGIF's default DI sharders (tgif/types/types.py).
DI_FUSED_PARAMS: Dict[str, Any] = {
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: True,
    FUSED_PARAM_REGISTER_TBE_BOOL: True,
    FUSED_PARAM_TBE_ROW_ALIGNMENT: 1,
}
TW_PLAN: Dict[str, ParameterShardingGenerator] = {
    name: table_wise(
        rank=0, device="cpu", compute_kernel=EmbeddingComputeKernel.QUANT.value
    )
    for name in (G8_TABLE, ROWWISE_TABLE)
}
RW_PLAN: Dict[str, ParameterShardingGenerator] = {
    name: row_wise(([NUM_EMBEDDINGS // WORLD_SIZE] * WORLD_SIZE, "cpu"))
    for name in (G8_TABLE, ROWWISE_TABLE)
}


def _table(
    name: str, grouped_int4_group_size: int, use_virtual_table: bool = True
) -> EmbeddingConfig:
    return EmbeddingConfig(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        name=name,
        feature_names=[f"{name}_feature"],
        use_virtual_table=use_virtual_table,
        enable_embedding_update=True,
        grouped_int4_group_size=grouped_int4_group_size,
    )


class _ECModel(torch.nn.Module):
    def __init__(self, tables: List[EmbeddingConfig]) -> None:
        super().__init__()
        self.ec = EmbeddingCollection(tables=tables, device=torch.device("meta"))


def _quantize_int4(model: torch.nn.Module, table_names: List[str]) -> torch.nn.Module:
    return quantize_inference_model(
        model,
        per_table_weight_dtype={name: torch.quint4x2 for name in table_names},
        quantization_dtype=torch.quint4x2,
    )


def _quant_ec(quant_model: torch.nn.Module) -> QuantEmbeddingCollection:
    return cast(QuantEmbeddingCollection, quant_model.get_submodule("ec"))


def _di_shard(
    quant_model: torch.nn.Module,
    per_param_sharding: Dict[str, ParameterShardingGenerator],
) -> torch.nn.Module:
    sharder = cast(
        ModuleSharder[torch.nn.Module],
        QuantEmbeddingCollectionSharder(fused_params=dict(DI_FUSED_PARAMS)),
    )
    module_plan = construct_module_sharding_plan(
        _quant_ec(quant_model),
        per_param_sharding=per_param_sharding,
        sharder=sharder,
        local_size=WORLD_SIZE,
        world_size=WORLD_SIZE,
        device_type="cpu",
    )
    return _shard_modules(
        module=quant_model,
        sharders=[sharder],
        device=torch.device("meta"),
        plan=ShardingPlan(plan={"ec": module_plan}),
        env=ShardingEnv.from_local(world_size=WORLD_SIZE, rank=0),
    )


def _kv_modules(model: torch.nn.Module) -> List[KVEmbeddingInference]:
    return [m for m in model.modules() if isinstance(m, KVEmbeddingInference)]


def _kv_modules_by_table(
    model: torch.nn.Module,
) -> Dict[str, List[KVEmbeddingInference]]:
    kv_modules: Dict[str, List[KVEmbeddingInference]] = defaultdict(list)
    for kv in _kv_modules(model):
        table_names = [spec[0] for spec in kv.embedding_specs]
        assert len(table_names) == 1, f"tables share a TBE: {table_names}"
        kv_modules[table_names[0]].append(kv)
    return kv_modules


def _disable_random_init(kv_embedding_cache: Any) -> int:
    return int(kv_embedding_cache.serialize()[2].item())


class InferKVGroupedInt4Test(unittest.TestCase):
    def test_group_size_is_keyword_only_and_defaults_to_rowwise(self) -> None:
        parameter = inspect.signature(EmbeddingBagConfig).parameters[
            "grouped_int4_group_size"
        ]

        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(EmbeddingConfig(8, 4).grouped_int4_group_size, 0)

    def _assert_served_modules(
        self, sharded_model: torch.nn.Module, num_shards: int
    ) -> List[KVEmbeddingInference]:
        kv_modules = _kv_modules_by_table(sharded_model)

        self.assertEqual(len(kv_modules[G8_TABLE]), num_shards)
        self.assertEqual(len(kv_modules[ROWWISE_TABLE]), num_shards)
        for kv in kv_modules[G8_TABLE]:
            self.assertEqual(kv.grouped_int4_group_size, GROUP_SIZE)
            self.assertEqual(kv.grouped_int4_row_bytes, GROUPED_ROW_BYTES)
            self.assertTrue(kv.use_grouped_int4_rows)
            self.assertTrue(kv.embedding_cache_mode)
            self.assertEqual(_disable_random_init(kv.kv_embedding_cache), 1)
        for kv in kv_modules[ROWWISE_TABLE]:
            self.assertEqual(kv.grouped_int4_group_size, 0)
            self.assertFalse(kv.use_grouped_int4_rows)
            self.assertTrue(kv.embedding_cache_mode)
            self.assertEqual(_disable_random_init(kv.kv_embedding_cache), 1)
        return kv_modules[G8_TABLE]

    def _assert_scripted_round_trip(self, kv: KVEmbeddingInference) -> None:
        archive = io.BytesIO()
        torch.jit.save(torch.jit.script(kv), archive)
        archive.seek(0)
        loaded = torch.jit.load(archive)

        self.assertEqual(loaded.grouped_int4_group_size, GROUP_SIZE)
        self.assertEqual(loaded.grouped_int4_dim, EMBEDDING_DIM)
        self.assertEqual(loaded.grouped_int4_row_bytes, GROUPED_ROW_BYTES)
        self.assertTrue(loaded.embedding_cache_mode)
        self.assertEqual(_disable_random_init(loaded.kv_embedding_cache), 1)
        self.assertIn("GroupedInt4E4M3ToHalf", loaded.forward.code)

    @skip_without_grouped_int4_fbgemm
    def test_di_table_wise_builds_grouped_kv_module(self) -> None:
        tables = [_table(G8_TABLE, GROUP_SIZE), _table(ROWWISE_TABLE, 0)]
        quant_model = _quantize_int4(_ECModel(tables), [G8_TABLE, ROWWISE_TABLE])

        sharded_model = _di_shard(quant_model, TW_PLAN)

        g8_modules = self._assert_served_modules(sharded_model, num_shards=1)
        self._assert_scripted_round_trip(g8_modules[0])

    @skip_without_grouped_int4_fbgemm
    def test_di_row_wise_builds_grouped_kv_module(self) -> None:
        tables = [_table(G8_TABLE, GROUP_SIZE), _table(ROWWISE_TABLE, 0)]
        quant_model = _quantize_int4(_ECModel(tables), [G8_TABLE, ROWWISE_TABLE])

        sharded_model = _di_shard(quant_model, RW_PLAN)

        g8_modules = self._assert_served_modules(sharded_model, num_shards=WORLD_SIZE)
        self.assertEqual(
            sorted(kv.table_sharding_offset for kv in g8_modules),
            [[0], [NUM_EMBEDDINGS // WORLD_SIZE]],
        )
        self._assert_scripted_round_trip(g8_modules[0])

    @skip_without_grouped_int4_fbgemm
    def test_di_sharding_without_group_size_keeps_one_rowwise_tbe(self) -> None:
        tables = [_table(G8_TABLE, 0), _table(ROWWISE_TABLE, 0)]
        quant_model = _quantize_int4(_ECModel(tables), [G8_TABLE, ROWWISE_TABLE])

        sharded_model = _di_shard(quant_model, TW_PLAN)

        kv_modules = _kv_modules(sharded_model)
        self.assertEqual(len(kv_modules), 1)
        self.assertEqual(
            [spec[0] for spec in kv_modules[0].embedding_specs],
            [G8_TABLE, ROWWISE_TABLE],
        )
        self.assertEqual(kv_modules[0].grouped_int4_group_size, 0)
        scripted = cast(torch.jit.ScriptModule, torch.jit.script(kv_modules[0]))
        self.assertNotIn("GroupedInt4E4M3ToHalf", scripted.code)

    @skip_without_grouped_int4_fbgemm
    def test_unsharded_quant_ec_builds_grouped_kv_module(self) -> None:
        tables = [_table(G8_TABLE, GROUP_SIZE), _table(ROWWISE_TABLE, 0)]

        quant_model = _quantize_int4(_ECModel(tables), [G8_TABLE, ROWWISE_TABLE])

        kv_modules = _kv_modules_by_table(quant_model)
        self.assertEqual(kv_modules[G8_TABLE][0].grouped_int4_group_size, GROUP_SIZE)
        self.assertTrue(kv_modules[G8_TABLE][0].embedding_cache_mode)
        self.assertEqual(kv_modules[ROWWISE_TABLE][0].grouped_int4_group_size, 0)

    @skip_without_grouped_int4_fbgemm
    def test_unsharded_quant_ec_without_group_size_keeps_one_rowwise_tbe(
        self,
    ) -> None:
        tables = [_table(G8_TABLE, 0), _table(ROWWISE_TABLE, 0)]

        quant_model = _quantize_int4(_ECModel(tables), [G8_TABLE, ROWWISE_TABLE])

        kv_modules = _kv_modules(quant_model)
        self.assertEqual(len(kv_modules), 1)
        self.assertEqual(kv_modules[0].grouped_int4_group_size, 0)

    def test_group_size_on_non_virtual_table_fails_unsharded(self) -> None:
        tables = [_table(G8_TABLE, GROUP_SIZE, use_virtual_table=False)]

        with self.assertRaisesRegex(ValueError, "KVEmbeddingInference"):
            _quantize_int4(_ECModel(tables), [G8_TABLE])

    def test_group_size_on_non_virtual_table_fails_sharded(self) -> None:
        tables = [
            _table(G8_TABLE, 0, use_virtual_table=False),
            _table(ROWWISE_TABLE, 0, use_virtual_table=False),
        ]
        quant_model = _quantize_int4(_ECModel(tables), [G8_TABLE, ROWWISE_TABLE])
        g8_config = _quant_ec(quant_model).embedding_configs()[0]
        g8_config.grouped_int4_group_size = GROUP_SIZE

        with self.assertRaisesRegex(ValueError, "KVEmbeddingInference"):
            _di_shard(quant_model, TW_PLAN)

    def test_group_size_on_pooled_table_fails(self) -> None:
        model = torch.nn.Sequential(
            EmbeddingBagCollection(
                tables=[
                    EmbeddingBagConfig(
                        num_embeddings=NUM_EMBEDDINGS,
                        embedding_dim=EMBEDDING_DIM,
                        name=G8_TABLE,
                        feature_names=["f"],
                        grouped_int4_group_size=GROUP_SIZE,
                    )
                ],
                device=torch.device("meta"),
            )
        )

        with self.assertRaisesRegex(ValueError, "EmbeddingCollection"):
            _quantize_int4(model, [G8_TABLE])

    def test_mixed_group_sizes_cannot_share_a_tbe(self) -> None:
        config = GroupedEmbeddingConfig(
            data_type=DataType.INT4,
            pooling=PoolingType.NONE,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=EmbeddingComputeKernel.QUANT,
            embedding_tables=[
                ShardedEmbeddingTable(
                    num_embeddings=NUM_EMBEDDINGS,
                    embedding_dim=EMBEDDING_DIM,
                    name=name,
                    grouped_int4_group_size=group_size,
                )
                for name, group_size in ((G8_TABLE, GROUP_SIZE), (ROWWISE_TABLE, 0))
            ],
        )

        with self.assertRaisesRegex(ValueError, "must share grouped_int4_group_size"):
            _get_grouped_int4_group_size(config)
