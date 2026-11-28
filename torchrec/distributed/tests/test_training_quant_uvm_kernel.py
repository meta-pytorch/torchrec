#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, cast, Dict, List
from unittest.mock import MagicMock, patch

import torch
import torchrec.distributed.embedding_lookup as el
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    EmbeddingBagSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.fp_embeddingbag import (
    FeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_BOUNDS_CHECK_MODE,
    FUSED_PARAM_IS_DEVICE_RO,
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
)
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _config(
    data_types: List[DataType], fused_params: Dict[str, Any]
) -> GroupedEmbeddingConfig:
    tables = [
        ShardedEmbeddingTable(
            num_embeddings=10,
            embedding_dim=64,
            name=f"t{i}",
            feature_names=[f"f{i}"],
            data_type=data_type,
            local_rows=10,
            local_cols=64,
            compute_kernel=EmbeddingComputeKernel.QUANT_UVM,
        )
        for i, data_type in enumerate(data_types)
    ]
    return GroupedEmbeddingConfig(
        data_type=data_types[0],
        pooling=PoolingType.SUM,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=EmbeddingComputeKernel.QUANT_UVM,
        embedding_tables=tables,
        fused_params=fused_params,
    )


class TrainingQuantUvmKernelTest(unittest.TestCase):
    def test_offered_only_column_and_table_wise(self) -> None:
        for sharder in (
            EmbeddingBagCollectionSharder(),
            FeatureProcessedEmbeddingBagCollectionSharder(),
        ):
            for sharding_type in ShardingType:
                with self.subTest(sharder=type(sharder), sharding_type=sharding_type):
                    kernels = sharder.compute_kernels(sharding_type.value, "cuda")
                    self.assertEqual(
                        EmbeddingComputeKernel.QUANT_UVM.value in kernels,
                        sharding_type
                        in (ShardingType.COLUMN_WISE, ShardingType.TABLE_WISE),
                    )

    def test_offered_only_on_cuda_and_cpu(self) -> None:
        sharder = EmbeddingBagCollectionSharder()
        for device in ("cuda", "cpu", "tpu"):
            with self.subTest(device=device):
                self.assertEqual(
                    EmbeddingComputeKernel.QUANT_UVM.value
                    in sharder.compute_kernels(ShardingType.TABLE_WISE.value, device),
                    device in ("cuda", "cpu"),
                )

    def test_not_offered_by_sharders_that_cannot_serve_it(self) -> None:
        # Their modules neither skip a quant_uvm table's torch state nor gate on it.
        for sharder in (EmbeddingCollectionSharder(), EmbeddingBagSharder()):
            for sharding_type in (ShardingType.COLUMN_WISE, ShardingType.TABLE_WISE):
                with self.subTest(sharder=type(sharder), sharding_type=sharding_type):
                    self.assertNotIn(
                        EmbeddingComputeKernel.QUANT_UVM.value,
                        sharder.compute_kernels(sharding_type.value, "cuda"),
                    )

    def test_builds_each_table_at_its_planned_dtype(self) -> None:
        kernel = el._training_quant_uvm_kernel(
            config=_config([DataType.INT4, DataType.INT8], {}),
            pg=None,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(kernel, el.TrainingQuantBatchedEmbeddingBag)
        self.assertEqual(
            [spec[3] for spec in kernel.emb_module.embedding_specs],
            [SparseType.INT4, SparseType.INT8],
        )

    def test_the_lookup_builds_it_for_the_group_group_tables_files_it_under(
        self,
    ) -> None:
        # group_tables files a quant_uvm table's group under QUANT
        # (_get_compute_kernel_type), so that is what the lookup sees.
        for group_kernel in (
            EmbeddingComputeKernel.QUANT,
            EmbeddingComputeKernel.QUANT_UVM,
        ):
            config = _config([DataType.INT8], {})
            config.compute_kernel = group_kernel
            with self.subTest(group_kernel=group_kernel):
                kernel = el.GroupedPooledEmbeddingsLookup._create_embedding_kernel(
                    cast(Any, None),
                    config,
                    device=torch.device("cpu"),
                    pg=None,
                    sharding_type=None,
                    env=None,
                )
                self.assertIsInstance(kernel, el.TrainingQuantBatchedEmbeddingBag)

    def test_only_a_quant_uvm_table_is_sharded_at_its_planned_dtype(self) -> None:
        for kernel, expected in (
            (EmbeddingComputeKernel.QUANT_UVM.value, DataType.INT8),
            (EmbeddingComputeKernel.FUSED.value, DataType.FP16),
        ):
            info = EmbeddingShardingInfo(
                embedding_config=cast(Any, SimpleNamespace(data_type=DataType.FP16)),
                param_sharding=cast(
                    Any,
                    SimpleNamespace(
                        compute_kernel=kernel, quantized_weight_dtype=DataType.INT8
                    ),
                ),
                param=torch.empty(0),
            )
            with self.subTest(kernel=kernel):
                self.assertEqual(info.sharded_data_type(), expected)

    def test_refuses_a_quant_table_that_is_not_quant_uvm(self) -> None:
        config = _config([DataType.INT8], {})
        config.embedding_tables[0].compute_kernel = EmbeddingComputeKernel.QUANT
        with self.assertRaisesRegex(ValueError, "only quant_uvm"):
            el._training_quant_uvm_kernel(
                config=config, pg=None, device=torch.device("cpu")
            )

    def test_refuses_a_float_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantized_weight_dtype"):
            el._training_quant_uvm_kernel(
                config=_config([DataType.FP16], {}),
                pg=None,
                device=torch.device("cpu"),
            )

    def test_passes_only_what_the_inference_tbe_takes(self) -> None:
        training_fused_params: Dict[str, Any] = {
            "learning_rate": 0.01,  # optimizer: dropped
            "optimizer": "exact_row_wise_adagrad",  # optimizer: dropped
            "row_alignment": 4,  # set by the kernel itself: dropped
            "output_dtype": SparseType.FP16,  # kept: matches the FUSED siblings
            FUSED_PARAM_IS_DEVICE_RO: True,  # read by the kernel itself: kept
        }
        with patch.object(el, "TrainingQuantBatchedEmbeddingBag") as qbeb:
            el._training_quant_uvm_kernel(
                config=_config([DataType.INT8], training_fused_params),
                pg=None,
                device=torch.device("cpu"),
            )
        passed = cast(MagicMock, qbeb).call_args.kwargs["fused_params"]
        self.assertEqual(
            passed,
            {
                "output_dtype": SparseType.FP16,
                FUSED_PARAM_IS_DEVICE_RO: True,
                FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: True,
            },
        )

    def test_passes_the_plans_bounds_check_mode_under_its_torchrec_key(self) -> None:
        mode = object()  # passed through untouched
        with patch.object(el, "TrainingQuantBatchedEmbeddingBag") as qbeb:
            el._training_quant_uvm_kernel(
                config=_config([DataType.INT8], {"bounds_check_mode": mode}),
                pg=None,
                device=torch.device("cpu"),
            )
        passed = cast(MagicMock, qbeb).call_args.kwargs["fused_params"]
        self.assertIs(passed[FUSED_PARAM_BOUNDS_CHECK_MODE], mode)
        self.assertNotIn("bounds_check_mode", passed)

    def test_output_dtype_defaults_to_fp32(self) -> None:
        with patch.object(el, "TrainingQuantBatchedEmbeddingBag") as qbeb:
            el._training_quant_uvm_kernel(
                config=_config([DataType.INT8], {}),
                pg=None,
                device=torch.device("cpu"),
            )
        passed = cast(MagicMock, qbeb).call_args.kwargs["fused_params"]
        self.assertEqual(passed["output_dtype"], SparseType.FP32)


class QuantUvmServingGateTest(unittest.TestCase):
    """The gate's state machine, on a bare module built from a plan. The 2-rank
    behaviour (a rank holding no quant_uvm shard refuses too) is in
    ShardedQuantUvmTest."""

    def _module(self, kernels: List[str]) -> ShardedEmbeddingBagCollection:
        module = ShardedEmbeddingBagCollection.__new__(ShardedEmbeddingBagCollection)
        torch.nn.Module.__init__(module)
        module._module_fqn = "ebc"
        plan = {
            f"t{i}": cast(Any, SimpleNamespace(compute_kernel=kernel))
            for i, kernel in enumerate(kernels)
        }
        module._init_quant_uvm_gate(plan)
        return module

    def _with_quant_uvm(self) -> ShardedEmbeddingBagCollection:
        return self._module(
            [EmbeddingComputeKernel.FUSED.value, EmbeddingComputeKernel.QUANT_UVM.value]
        )

    def test_closed_from_construction_until_servable(self) -> None:
        module = self._with_quant_uvm()
        with self.assertRaisesRegex(RuntimeError, "not filled"):
            module._check_quant_uvm_servable()
        module.set_quant_uvm_servable(True)
        module._check_quant_uvm_servable()
        module.set_quant_uvm_servable(False)
        with self.assertRaisesRegex(RuntimeError, "not filled"):
            module._check_quant_uvm_servable()

    def test_every_load_closes_it_until_refilled(self) -> None:
        # A load changes the checkpoint; the quant_uvm rows still hold the previous one.
        module = self._with_quant_uvm()
        module.set_quant_uvm_servable(True)
        module.load_state_dict({})
        with self.assertRaisesRegex(RuntimeError, "not filled"):
            module._check_quant_uvm_servable()
        module.set_quant_uvm_servable(True)
        module._check_quant_uvm_servable()

    def test_a_module_without_quant_uvm_tables_is_unaffected(self) -> None:
        module = self._module([EmbeddingComputeKernel.FUSED.value])
        module.load_state_dict({})
        module._check_quant_uvm_servable()
        with self.assertRaisesRegex(ValueError, "no quant_uvm table"):
            module.set_quant_uvm_servable(True)

    def test_compute_rechecks_the_gate(self) -> None:
        module = self._with_quant_uvm()
        module._lookups = []
        with self.assertRaisesRegex(RuntimeError, "not filled"):
            module.compute(cast(Any, None), cast(Any, []))

    def test_input_dist_refuses_while_closed(self) -> None:
        module = self._with_quant_uvm()
        features = MagicMock()
        features.variable_stride_per_key.return_value = False
        with self.assertRaisesRegex(RuntimeError, "not filled"):
            module.input_dist(cast(Any, SimpleNamespace()), features)


# Two keys over 3 ranks, uneven per-key bag counts, so padding runs: key a has 3 bags
# (one on rank 0, two on rank 2), key b has 2 (ranks 0 and 1).
_STRIDES = [[1, 0, 2], [1, 1, 0]]
_DIMS = [2, 3]


def _fake_pool_batch() -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=["a", "b"],
        values=torch.arange(1, 7),
        lengths=torch.tensor([1, 2, 0, 2, 1]),
        stride_per_key_per_rank=_STRIDES,
    )


class VbePoolTest(unittest.TestCase):
    """`_vbe_pool` with a fake pool whose every row encodes its bag (100 * key + the
    sum of the bag's values), so each block's position is checkable."""

    def setUp(self) -> None:
        self.uniform_batches: List[KeyedJaggedTensor] = []

    def _pool(self, uniform: KeyedJaggedTensor) -> torch.Tensor:
        self.uniform_batches.append(uniform)
        columns = []
        for k, key in enumerate(uniform.keys()):
            jt = uniform[key]
            bounds = jt.offsets().tolist()
            sums = [float(jt.values()[a:b].sum()) for a, b in zip(bounds, bounds[1:])]
            columns.append(torch.tensor([[100.0 * k + s] * _DIMS[k] for s in sums]))
        return torch.cat(columns, dim=1)

    def _want(self) -> torch.Tensor:
        a = [[1.0] * 2, [5.0] * 2, [0.0] * 2]  # key a's bags: {1}, {2, 3}, {}
        b = [[109.0] * 3, [106.0] * 3]  # key b's bags: {4, 5}, {6}
        # rank 0: a0, b0 · rank 1: b1 · rank 2: a1, a2
        return torch.tensor(a[0] + b[0] + b[1] + a[1] + a[2])

    def test_pads_to_a_uniform_batch_and_cuts_blocks_rank_major(self) -> None:
        got = el._vbe_pool(self._pool, _fake_pool_batch(), _DIMS, torch.float32)
        (uniform,) = self.uniform_batches
        self.assertFalse(uniform.variable_stride_per_key())
        self.assertEqual(uniform.stride(), 3)
        self.assertEqual(uniform.lengths().tolist(), [1, 2, 0, 2, 1, 0])
        torch.testing.assert_close(got, self._want())

    def test_writes_each_block_at_its_offset_in_a_preallocated_output(self) -> None:
        sizes = [[2, 3], [0, 3], [4, 0]]  # stride * dim per [rank][key]
        offsets: List[List[int]] = []
        at = 1
        for rank_sizes in sizes:
            offsets.append([])
            for size in rank_sizes:
                offsets[-1].append(at)
                at += size + 1  # one sentinel slot after every block
        buffer = torch.full((at,), -7.0)
        out = el._vbe_pool(
            self._pool,
            _fake_pool_batch(),
            _DIMS,
            torch.float32,
            buffer,
            torch.tensor(offsets),
        )
        self.assertIs(out, buffer)
        want = self._want().tolist()
        for rank in range(3):
            for key in range(2):
                start, size = offsets[rank][key], sizes[rank][key]
                self.assertEqual(out[start : start + size].tolist(), want[:size])
                want = want[size:]
        self.assertEqual(int((out == -7.0).sum()), at - sum(map(sum, sizes)))

    def test_refuses_offsets_of_the_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "vbe_output_offsets"):
            el._vbe_pool(
                self._pool,
                _fake_pool_batch(),
                _DIMS,
                torch.float32,
                torch.zeros(20),
                torch.tensor([[0, 5], [10, 15]]),  # 2 ranks, the batch has 3
            )

    def test_a_batch_with_no_bags_is_not_pooled_and_keeps_the_output_dtype(
        self,
    ) -> None:
        # Every rank must emit one dtype into the pooled all-to-all, so a rank whose
        # batch holds no bag for these keys still answers in the TBE's output dtype.
        empty = KeyedJaggedTensor(
            keys=["a", "b"],
            values=torch.zeros(0, dtype=torch.long),
            lengths=torch.zeros(0, dtype=torch.long),
            stride_per_key_per_rank=[[0, 0, 0], [0, 0, 0]],
        )
        got = el._vbe_pool(self._pool, empty, _DIMS, torch.float16)
        self.assertEqual(self.uniform_batches, [])
        self.assertEqual((got.dtype, got.numel()), (torch.float16, 0))


class TrainingQuantBatchedEmbeddingBagVbeTest(unittest.TestCase):
    """The real kernel (IntNBit TBE, CPU) on a VBE batch, against rows written into
    it and dequantized independently."""

    def setUp(self) -> None:
        self.kernel = el._training_quant_uvm_kernel(
            config=_config([DataType.INT8, DataType.INT8], {}),
            pg=None,
            device=torch.device("cpu"),
        )
        generator = torch.Generator().manual_seed(0)
        self.reference: List[torch.Tensor] = []
        for table, (weight, scale, shift) in zip(
            self.kernel.config.embedding_tables, self.kernel.split_embedding_weights()
        ):
            assert scale is not None and shift is not None
            rows, cols = table.local_rows, table.local_cols
            codes = torch.randint(
                0, 256, (rows, cols), dtype=torch.uint8, generator=generator
            )
            s = (torch.rand(rows, 1, generator=generator) * 0.01 + 0.001).half()
            b = (torch.rand(rows, 1, generator=generator) - 0.5).half()
            with torch.no_grad():
                weight[:, :cols].copy_(codes)
                scale.copy_(s)
                shift.copy_(b)
            self.reference.append(codes.float() * s.float() + b.float())
        # 2 ranks: f0 has 1 bag on rank 0 and 2 on rank 1, f1 has 2 on rank 0.
        self.strides = [[1, 2], [2, 0]]
        self.lengths = [[2, 0, 1], [1, 3]]
        self.batch = KeyedJaggedTensor(
            keys=["f0", "f1"],
            values=torch.randint(0, 10, (7,), generator=generator),
            lengths=torch.tensor(sum(self.lengths, [])),
            stride_per_key_per_rank=self.strides,
        )

    def _blocks(self) -> List[List[torch.Tensor]]:
        """[rank][key] -> the key's bags on that rank, pooled from the reference."""
        pooled = [
            torch.nn.functional.embedding_bag(
                self.batch[key].values(),
                self.reference[k],
                self.batch[key].offsets()[:-1],
                mode="sum",
            )
            for k, key in enumerate(self.batch.keys())
        ]
        return [
            [
                pooled[k][sum(self.strides[k][:rank]) :][: self.strides[k][rank]]
                for k in range(2)
            ]
            for rank in range(2)
        ]

    def test_serves_a_variable_batch(self) -> None:
        with torch.no_grad():
            got = self.kernel(self.batch)
        want = torch.cat([b.reshape(-1) for blocks in self._blocks() for b in blocks])
        torch.testing.assert_close(got, want)

    def test_fills_a_preallocated_vbe_output(self) -> None:
        blocks = self._blocks()
        offsets = [[0, 70], [200, 400]]  # [rank][key]; gaps stay untouched
        buffer = torch.full((500,), -7.0)
        with torch.no_grad():
            out = self.kernel(
                self.batch,
                vbe_output=buffer,
                vbe_output_offsets=torch.tensor(offsets),
            )
        self.assertIs(out, buffer)
        written = 0
        for rank in range(2):
            for key in range(2):
                block = blocks[rank][key].reshape(-1)
                start = offsets[rank][key]
                torch.testing.assert_close(out[start : start + block.numel()], block)
                written += block.numel()
        self.assertEqual(int((out == -7.0).sum()), 500 - written)

    def test_its_lookup_neither_saves_nor_loads_it(self) -> None:
        lookup = el.GroupedPooledEmbeddingsLookup.__new__(
            el.GroupedPooledEmbeddingsLookup
        )
        torch.nn.Module.__init__(lookup)
        lookup._emb_modules = torch.nn.ModuleList([self.kernel])
        state = lookup.state_dict()
        self.assertEqual(list(state), [])
        result = lookup.load_state_dict(OrderedDict(state))
        self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))

    def test_refuses_a_preallocated_output_with_a_uniform_batch(self) -> None:
        uniform = KeyedJaggedTensor(
            keys=["f0", "f1"],
            values=torch.tensor([1, 2]),
            lengths=torch.tensor([1, 0, 0, 1]),
        )
        with self.assertRaisesRegex(ValueError, "non-VBE batch"):
            self.kernel(
                uniform,
                vbe_output=torch.zeros(256),
                vbe_output_offsets=torch.tensor([[0, 128]]),
            )


_ROWS = 20
_DIM = 16
_BATCH = 3
# name -> (training data_type, compute kernel, planned quantized dtype or None)
_TABLES: Dict[str, Any] = {
    "t_f32": (DataType.FP32, EmbeddingComputeKernel.FUSED, None),
    "t_f16": (DataType.FP16, EmbeddingComputeKernel.FUSED, None),
    "t_int8": (DataType.FP16, EmbeddingComputeKernel.QUANT_UVM, DataType.INT8),
    "t_int4": (DataType.FP16, EmbeddingComputeKernel.QUANT_UVM, DataType.INT4),
}


def _rows(name: str) -> Dict[str, torch.Tensor]:
    """The same rows on every rank: codes, scale and shift for a quant_uvm table
    (one scale and shift per row, written into every column-wise shard), the float
    weight for a FUSED one; "reference" is what each row should pool as."""
    index = list(_TABLES).index(name)
    data_type, kernel, quantized = _TABLES[name]
    generator = torch.Generator().manual_seed(index)
    if kernel == EmbeddingComputeKernel.FUSED:
        weight = torch.rand(_ROWS, _DIM, generator=generator) - 0.5
        if data_type == DataType.FP16:
            weight = weight.half().float()
        return {"reference": weight}
    top = 16 if quantized == DataType.INT4 else 256
    codes = torch.randint(0, top, (_ROWS, _DIM), dtype=torch.uint8, generator=generator)
    scale = (torch.rand(_ROWS, 1, generator=generator) * 0.01 + 0.001).half()
    shift = (torch.rand(_ROWS, 1, generator=generator) - 0.5).half()
    reference = codes.float() * scale.float() + shift.float()
    return {"codes": codes, "scale": scale, "shift": shift, "reference": reference}


def _pack(codes: torch.Tensor, quantized: DataType) -> torch.Tensor:
    if quantized == DataType.INT8:
        return codes
    # INT4: two codes a byte, the even column in the low nibble (fbgemm's layout)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def _tbes(module: ShardedEmbeddingBagCollection) -> List[Any]:
    return [
        tbe
        for lookup in cast(List[Any], module._lookups)
        for tbe in lookup._emb_modules
    ]


def _fill(module: ShardedEmbeddingBagCollection, quant: bool) -> None:
    """Write each local shard of every quant_uvm (or every FUSED) table from `_rows`."""
    for tbe in _tbes(module):
        if isinstance(tbe, el.TrainingQuantBatchedEmbeddingBag) != quant:
            continue
        for table, split in zip(
            tbe.config.embedding_tables, tbe.split_embedding_weights()
        ):
            assert table.local_metadata is not None
            col = table.local_metadata.shard_offsets[1]
            rows = _rows(table.name)
            with torch.no_grad():
                if not quant:
                    split.copy_(rows["reference"][:, col : col + table.local_cols])
                    continue
                weight, scale, shift = split
                codes = rows["codes"][:, col : col + table.local_cols]
                packed = _pack(codes, _TABLES[table.name][2])
                assert weight.shape == packed.shape, (weight.shape, packed.shape)
                weight.copy_(packed)
                scale.copy_(rows["scale"])
                shift.copy_(rows["shift"])


def _quant_rows(module: ShardedEmbeddingBagCollection) -> List[torch.Tensor]:
    return [
        part.clone()
        for tbe in _tbes(module)
        if isinstance(tbe, el.TrainingQuantBatchedEmbeddingBag)
        for split in tbe.split_embedding_weights()
        for part in split
        if part is not None
    ]


def _batch(rank: int) -> KeyedJaggedTensor:
    generator = torch.Generator().manual_seed(10 + rank)
    lengths = torch.randint(1, 4, (len(_TABLES), _BATCH), generator=generator)
    lengths[:, 1] = 0  # every key's second bag is empty: it must pool to 0
    values = torch.randint(0, _ROWS, (int(lengths.sum()),), generator=generator)
    return KeyedJaggedTensor(
        keys=[f"f_{name}" for name in _TABLES],
        values=values,
        lengths=lengths.reshape(-1),
    )


def _serve_sharded_quant_uvm(rank: int, world_size: int, layout: str) -> None:
    with MultiProcessContext(rank, world_size, "gloo") as ctx:
        assert ctx.pg is not None
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name=name,
                    num_embeddings=_ROWS,
                    embedding_dim=_DIM,
                    feature_names=[f"f_{name}"],
                    data_type=data_type,
                )
                for name, (data_type, _, _) in _TABLES.items()
            ],
            device=torch.device("meta"),
        )
        sharder = EmbeddingBagCollectionSharder()
        per_table = {}
        for name, (_, kernel, _) in _TABLES.items():
            if layout == "column_wise":
                per_table[name] = column_wise(ranks=[0, 1], compute_kernel=kernel.value)
            else:  # quant_uvm tables on rank 0 only, so rank 1 holds none
                quant = kernel == EmbeddingComputeKernel.QUANT_UVM
                per_table[name] = table_wise(
                    rank=0 if quant else 1, compute_kernel=kernel.value
                )
        plan = construct_module_sharding_plan(
            ebc,
            per_table,
            sharder=cast(ModuleSharder[torch.nn.Module], sharder),
            local_size=world_size,
            world_size=world_size,
            device_type="cpu",
        )
        for name, (_, _, quantized) in _TABLES.items():
            if quantized is not None:
                plan[name].quantized_weight_dtype = quantized
        module = cast(
            ShardedEmbeddingBagCollection,
            sharder.shard(
                ebc, plan, ShardingEnv.from_process_group(ctx.pg), torch.device("cpu")
            ),
        )
        holds_quant = bool(_quant_rows(module))
        assert holds_quant == (layout == "column_wise" or rank == 0), (rank, layout)

        # Only the FUSED tables have torch state.
        assert set(module.state_dict()) == {
            "embedding_bags.t_f32.weight",
            "embedding_bags.t_f16.weight",
        }, (rank, sorted(module.state_dict()))

        # Closed from construction on every rank, the rank holding no quant_uvm
        # shard too, before any collective.
        try:
            module(_batch(rank))
        except RuntimeError as e:
            assert "not filled" in str(e), (rank, str(e))
        else:
            raise AssertionError(f"rank {rank}: forward served before servable")

        # reset_parameters leaves the quant_uvm rows alone.
        _fill(module, quant=True)
        filled = _quant_rows(module)
        module.reset_parameters()
        after = _quant_rows(module)
        assert len(after) == len(filled) and all(
            torch.equal(a, b) for a, b in zip(after, filled)
        ), rank
        _fill(module, quant=False)

        # A train pipeline runs input_dist ahead: closing the gate after it still
        # refuses compute_and_output_dist on every rank.
        module.set_quant_uvm_servable(True)
        context = module.create_context()
        dist_input = module.input_dist(context, _batch(rank)).wait().wait()
        module.set_quant_uvm_servable(False)
        try:
            module.compute_and_output_dist(context, dist_input)
        except RuntimeError as e:
            assert "not filled" in str(e), (rank, str(e))
        else:
            raise AssertionError(f"rank {rank}: compute served after the gate closed")

        module.set_quant_uvm_servable(True)
        batch = _batch(rank)
        with torch.no_grad():
            out = module(batch).wait()
        for name in _TABLES:
            key = f"f_{name}"
            jt = batch[key]
            want = torch.nn.functional.embedding_bag(
                jt.values(), _rows(name)["reference"], jt.offsets()[:-1], mode="sum"
            )
            assert out[key].dtype == torch.float32, (rank, key, out[key].dtype)
            torch.testing.assert_close(out[key], want, msg=f"rank {rank} {key}")
            assert not out[key][1].any(), (rank, key)

        # A later load closes the gate again on every rank, the rank holding no
        # quant_uvm shard too: the quant_uvm rows still hold the previous checkpoint.
        module.load_state_dict(module.state_dict())
        try:
            module(_batch(rank))
        except RuntimeError as e:
            assert "not filled" in str(e), (rank, str(e))
        else:
            raise AssertionError(f"rank {rank}: forward served after a later load")


class ShardedQuantUvmTest(MultiProcessTestBase):
    """A real 2-rank gloo ShardedEmbeddingBagCollection with FUSED FP32/FP16 and
    quant_uvm INT8/INT4 tables: no torch state for quant_uvm, the gate on every rank,
    reset_parameters, pooled rows equal to the dequantized reference, and a later
    load closing the gate again."""

    def test_table_wise_with_a_rank_holding_no_quant_uvm_shard(self) -> None:
        self._run_multi_process_test(
            callable=_serve_sharded_quant_uvm, world_size=2, layout="table_wise"
        )

    def test_column_wise(self) -> None:
        self._run_multi_process_test(
            callable=_serve_sharded_quant_uvm, world_size=2, layout="column_wise"
        )
