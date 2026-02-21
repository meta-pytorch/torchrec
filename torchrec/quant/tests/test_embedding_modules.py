#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest
from dataclasses import replace
from typing import Dict, List, Optional, Type

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torchrec import inference as trec_infer
from torchrec.distributed.quant_embedding_kernel import _unwrap_kjt, _unwrap_kjt_for_cpu
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
    QuantConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    _fx_trec_unwrap_kjt,
    _get_batching_hinted_output,
    _get_unflattened_lengths,
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
    MODULE_ATTR_USE_UNFLATTENED_LENGTHS_FOR_BATCHING,
    quant_prep_enable_quant_state_dict_split_scale_bias,
)
from torchrec.sparse.jagged_tensor import (
    ComputeKJTToJTDict,
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)

logger: logging.Logger = logging.getLogger(__name__)


def load_required_dram_kv_embedding_libraries() -> bool:
    try:
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:dram_kv_embedding_inference"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to load dram_kv_embedding libraries, skipping test: {e}")
        return False


class EmbeddingBagCollectionTest(unittest.TestCase):
    def _asserting_same_embeddings(
        self,
        pooled_embeddings_1: KeyedTensor,
        pooled_embeddings_2: KeyedTensor,
        atol: float = 1e-08,
    ) -> None:
        self.assertEqual(
            set(pooled_embeddings_1.keys()), set(pooled_embeddings_2.keys())
        )
        for key in pooled_embeddings_1.keys():
            self.assertEqual(
                pooled_embeddings_1[key].shape, pooled_embeddings_2[key].shape
            )
            self.assertTrue(
                torch.allclose(
                    pooled_embeddings_1[key].cpu().float(),
                    pooled_embeddings_2[key].cpu().float(),
                    atol=atol,
                )
            )

    def _test_ebc(
        self,
        tables: List[EmbeddingBagConfig],
        features: KeyedJaggedTensor,
        quant_type: torch.dtype = torch.qint8,
        output_type: torch.dtype = torch.float,
        quant_state_dict_split_scale_bias: bool = False,
        per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
    ) -> None:
        ebc = EmbeddingBagCollection(tables=tables)
        if quant_state_dict_split_scale_bias:
            quant_prep_enable_quant_state_dict_split_scale_bias(ebc)

        embeddings = ebc(features)

        # test forward
        if not per_table_weight_dtype:
            # pyrefly: ignore[bad-argument-type]
            ebc.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.PlaceholderObserver.with_args(
                    dtype=output_type
                ),
                weight=torch.quantization.PlaceholderObserver.with_args(
                    dtype=quant_type
                ),
            )
        else:
            # pyrefly: ignore[bad-argument-type]
            ebc.qconfig = QuantConfig(
                activation=torch.quantization.PlaceholderObserver.with_args(
                    dtype=output_type
                ),
                weight=torch.quantization.PlaceholderObserver.with_args(
                    dtype=quant_type
                ),
                per_table_weight_dtype=per_table_weight_dtype,
            )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        quantized_embeddings = qebc(features)

        self.assertEqual(quantized_embeddings.values().dtype, output_type)

        self._asserting_same_embeddings(embeddings, quantized_embeddings, atol=0.1)

        # test state dict
        state_dict = ebc.state_dict()
        quantized_state_dict = qebc.state_dict()
        self.assertTrue(
            set(state_dict.keys()).issubset(set(quantized_state_dict.keys()))
        )

    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        permute_order=st.booleans(),
        quant_state_dict_split_scale_bias=st.booleans(),
        per_table_weight_dtype=st.sampled_from(
            [
                {"t1": torch.quint4x2, "t2": torch.qint8},
                {"t1": torch.qint8, "t2": torch.quint4x2},
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_ebc(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
        permute_order: bool,
        quant_state_dict_split_scale_bias: bool,
        per_table_weight_dtype: Dict[str, torch.dtype],
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb1_mean_config = replace(
            eb1_config,
            name="t1_mean",
            pooling=PoolingType.MEAN,
            embedding_dim=32,
        )
        eb2_config = replace(eb1_config, name="t2", feature_names=["f2"])
        features = (
            KeyedJaggedTensor(
                keys=["f1", "f2"],
                values=torch.as_tensor([0, 2, 1, 3]),
                lengths=torch.as_tensor([1, 1, 2, 0]),
            )
            if not permute_order
            else KeyedJaggedTensor(
                keys=["f2", "f1"],
                values=torch.as_tensor([1, 3, 0, 2]),
                lengths=torch.as_tensor([2, 0, 1, 1]),
            )
        )
        # The key for grouping tables is (pooling, data_type).  Test having a different
        # key value in the middle.
        self._test_ebc(
            [eb1_config, eb1_mean_config, eb2_config],
            features,
            quant_type,
            output_type,
            quant_state_dict_split_scale_bias,
        )

        self._test_ebc(
            [eb1_config, eb1_mean_config, eb2_config],
            features,
            quant_type,
            output_type,
            quant_state_dict_split_scale_bias,
            per_table_weight_dtype,
        )

    def test_create_on_meta_device_without_providing_weights(self) -> None:
        emb_bag = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
        )
        QuantEmbeddingBagCollection(
            [emb_bag], is_weighted=False, device=torch.device("meta")
        )
        emb = EmbeddingConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
        )
        QuantEmbeddingCollection([emb], device=torch.device("meta"))

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb_config], features)

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        features = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config], features)

    def test_multiple_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f3"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.as_tensor([0, 1, 2]),
            lengths=torch.as_tensor([1, 1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config], features)

    @unittest.skipIf(
        not load_required_dram_kv_embedding_libraries(),
        "Skip when required libraries are not available",
    )
    def test_multiple_kernels_per_ebc_table(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self, m: torch.nn.Module) -> None:
                super().__init__()
                self.m = m

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            use_virtual_table=True,
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=16, num_embeddings=10, feature_names=["f3"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config, eb3_config])
        model = TestModule(ebc)
        qebc = trec_infer.modules.quantize_embeddings(
            model,
            dtype=torch.int8,
            inplace=True,
            per_table_weight_dtype={"t1": torch.float16},
        )
        self.assertTrue(isinstance(qebc.m, QuantEmbeddingBagCollection))
        # feature name should be consistent with the order of grouped embeddings
        self.assertEqual(qebc.m._feature_names, ["f1", "f3", "f2"])

        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.as_tensor([0, 1, 2]),
            lengths=torch.as_tensor([1, 1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config, eb3_config], features)

    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_save_load_state_dict(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        tables = [eb1_config, eb2_config]

        ebc = EmbeddingBagCollection(tables=tables)

        # test forward
        # pyrefly: ignore[bad-argument-type]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        qebc_state_dict = qebc.state_dict()

        ebc_2 = EmbeddingBagCollection(tables=tables)
        # pyrefly: ignore[bad-argument-type]
        ebc_2.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc_2 = QuantEmbeddingBagCollection.from_float(ebc_2)

        qebc_2.load_state_dict(qebc_state_dict)
        qebc_2_state_dict = qebc_2.state_dict()

        for key in qebc_state_dict:
            torch.testing.assert_close(qebc_state_dict[key], qebc_2_state_dict[key])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )

        embeddings = qebc(features)
        embeddings_2 = qebc_2(features)
        self._asserting_same_embeddings(embeddings, embeddings_2)

    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_composability(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
    ) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self, ebc: EmbeddingBagCollection) -> None:
                super().__init__()
                self.ebc = ebc
                self.over_arch = torch.nn.Linear(
                    16,
                    1,
                )

            def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
                ebc_output = self.ebc.forward(kjt).to_dict()
                sparse_features = []
                for key in kjt.keys():
                    sparse_features.append(ebc_output[key])
                sparse_features = torch.cat(sparse_features, dim=0)
                return self.over_arch(sparse_features)

        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        tables = [eb1_config, eb2_config]

        ebc = EmbeddingBagCollection(tables=tables)
        # test forward
        # pyrefly: ignore[bad-argument-type]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        test_model = TestModel(ebc)

        before_quant_state_dict = test_model.state_dict()
        # pyrefly: ignore[bad-assignment]
        test_model.ebc = QuantEmbeddingBagCollection.from_float(ebc)

        state_dict = test_model.state_dict()
        self.assertTrue(
            set(before_quant_state_dict.keys()).issubset(set(state_dict.keys()))
        )
        test_model.load_state_dict(state_dict)

    def test_trace_and_script(self) -> None:
        data_type = DataType.FP16
        quant_type = torch.half
        output_type = torch.half

        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        # pyrefly: ignore[bad-argument-type]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)

        from torchrec.fx import symbolic_trace

        gm = symbolic_trace(qebc, leaf_modules=[ComputeKJTToJTDict.__name__])

        non_placeholder_nodes = [
            node for node in gm.graph.nodes if node.op != "placeholder"
        ]
        self.assertTrue(
            len(non_placeholder_nodes) > 0, "Graph must have non-placeholder nodes"
        )
        self.assertEqual(
            non_placeholder_nodes[0].op,
            "call_function",
            f"First non-placeholder node must be call_function, got {non_placeholder_nodes[0].op} instead",
        )
        self.assertEqual(
            non_placeholder_nodes[0].name,
            "_get_kjt_keys",
            f"First non-placeholder node must be '_get_kjt_keys', got {non_placeholder_nodes[0].name} instead",
        )

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )

        original_out = qebc(features)
        traced_out = gm(features)

        scripted_module = torch.jit.script(gm)
        scripted_out = scripted_module(features)

        self.assertEqual(original_out.keys(), traced_out.keys())
        torch.testing.assert_close(original_out.values(), traced_out.values())
        self.assertEqual(original_out.offset_per_key(), traced_out.offset_per_key())

        self.assertEqual(original_out.keys(), scripted_out.keys())
        torch.testing.assert_close(original_out.values(), scripted_out.values())
        self.assertEqual(original_out.offset_per_key(), scripted_out.offset_per_key())


class EmbeddingCollectionTest(unittest.TestCase):
    def _comp_ec_output(
        self,
        embeddings: Dict[str, JaggedTensor],
        transformed_graph_embeddings: Dict[str, JaggedTensor],
        atol: int = 1,
    ) -> None:
        self.assertEqual(embeddings.keys(), transformed_graph_embeddings.keys())
        for key in embeddings.keys():
            self.assertEqual(
                embeddings[key].values().size(),
                transformed_graph_embeddings[key].values().size(),
            )
            self.assertTrue(
                torch.allclose(
                    embeddings[key].values().cpu().float(),
                    transformed_graph_embeddings[key].values().cpu().float(),
                    atol=atol,
                )
            )

    def _test_ec(
        self,
        tables: List[EmbeddingConfig],
        features: KeyedJaggedTensor,
        quant_type: torch.dtype = torch.qint8,
        output_type: torch.dtype = torch.float,
        quant_state_dict_split_scale_bias: bool = False,
    ) -> None:
        ec = EmbeddingCollection(tables=tables)
        if quant_state_dict_split_scale_bias:
            quant_prep_enable_quant_state_dict_split_scale_bias(ec)

        embeddings = ec(features)

        # test forward
        # pyrefly: ignore[bad-argument-type]
        ec.qconfig = QuantConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
            per_table_weight_dtype={
                x.name: torch.quint4x2 if x.data_type == DataType.INT4 else torch.qint8
                for x in ec._embedding_configs
            },
        )

        qec = QuantEmbeddingCollection.from_float(ec)
        quantized_embeddings = qec(features)
        self.assertEqual(
            list(quantized_embeddings.values())[0].values().dtype, output_type
        )
        self._comp_ec_output(embeddings, quantized_embeddings)

        # test state dict
        state_dict = ec.state_dict()
        quantized_state_dict = ec.state_dict()
        self.assertEqual(state_dict.keys(), quantized_state_dict.keys())

    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.INT8,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        quant_state_dict_split_scale_bias=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_ec(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
        quant_state_dict_split_scale_bias: bool,
    ) -> None:
        eb1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1", "f2"],
            data_type=data_type,
        )
        eb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f3", "f4"],
            data_type=data_type,
        )
        eb3_config = EmbeddingConfig(
            name="t3",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f5", "f6"],
            data_type=DataType.INT4,
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3", "f4", "f5", "f6"],
            values=torch.as_tensor(
                [
                    5,
                    1,
                    0,
                    0,
                    4,
                    3,
                    4,
                    9,
                    2,
                    2,
                    3,
                    3,
                    1,
                    5,
                    0,
                    7,
                    5,
                    0,
                    9,
                    9,
                    3,
                    5,
                    6,
                    6,
                    9,
                    3,
                    7,
                    8,
                    7,
                    7,
                    9,
                    1,
                    2,
                    6,
                    7,
                    6,
                    1,
                    8,
                    3,
                    8,
                    1,
                    9,
                    7,
                    7,
                    9,
                    1,
                    2,
                    6,
                    7,
                    6,
                    1,
                    8,
                    3,
                    8,
                    1,
                    9,
                ]
            ),
            lengths=torch.as_tensor([9, 12, 9, 12, 5, 9]),
        )
        self._test_ec(
            tables=[eb3_config, eb1_config, eb2_config],
            features=features,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb_config], features)

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        features = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb1_config, eb2_config], features)

    def test_different_quantization_dtype_per_ec_table(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self, m: torch.nn.Module) -> None:
                super().__init__()
                self.m = m

        eb1_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        ec = EmbeddingCollection(tables=[eb1_config, eb2_config])
        model = TestModule(ec)
        qconfig_spec_keys: List[Type[torch.nn.Module]] = [EmbeddingCollection]
        quant_mapping: Dict[Type[torch.nn.Module], Type[torch.nn.Module]] = {
            EmbeddingCollection: QuantEmbeddingCollection
        }
        trec_infer.modules.quantize_embeddings(
            model,
            dtype=torch.int8,
            additional_qconfig_spec_keys=qconfig_spec_keys,
            additional_mapping=quant_mapping,
            inplace=True,
            per_table_weight_dtype={"t1": torch.float16},
        )
        # pyrefly: ignore[not-callable]
        configs = model.m.embedding_configs()
        self.assertEqual(len(configs), 2)
        self.assertNotEqual(configs[0].name, configs[1].name)
        for config in configs:
            if config.name == "t1":
                self.assertEqual(config.data_type, DataType.FP16)
            else:
                self.assertEqual(config.name, "t2")
                self.assertEqual(config.data_type, DataType.INT8)

    @unittest.skipIf(
        not load_required_dram_kv_embedding_libraries(),
        "Skip when required libraries are not available",
    )
    def test_multiple_kernels_per_ec_table(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self, m: torch.nn.Module) -> None:
                super().__init__()
                self.m = m

        eb1_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            use_virtual_table=True,
        )
        eb3_config = EmbeddingConfig(
            name="t3",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f3"],
        )
        ec = EmbeddingCollection(tables=[eb1_config, eb2_config, eb3_config])
        model = TestModule(ec)
        qconfig_spec_keys: List[Type[torch.nn.Module]] = [EmbeddingCollection]
        quant_mapping: Dict[Type[torch.nn.Module], Type[torch.nn.Module]] = {
            EmbeddingCollection: QuantEmbeddingCollection
        }
        qec = trec_infer.modules.quantize_embeddings(
            model,
            dtype=torch.int8,
            additional_qconfig_spec_keys=qconfig_spec_keys,
            additional_mapping=quant_mapping,
            inplace=True,
            per_table_weight_dtype={
                "t1": torch.float16,
                "t2": torch.float16,
                "t3": torch.float16,
            },
        )
        self.assertTrue(isinstance(qec.m, QuantEmbeddingCollection))
        # feature name should be consistent with the order of grouped embeddings
        self.assertEqual(qec.m._feature_names, ["f1", "f3", "f2"])

        # pyrefly: ignore[not-callable]
        configs = model.m.embedding_configs()
        self.assertEqual(len(configs), 3)
        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.as_tensor(
                [
                    5,
                    1,
                    0,
                    0,
                    4,
                    3,
                    4,
                    9,
                    2,
                    2,
                    3,
                    3,
                    1,
                    5,
                    0,
                    7,
                    5,
                    0,
                    9,
                    9,
                    3,
                    5,
                    6,
                    6,
                    9,
                    3,
                    7,
                    8,
                    7,
                    7,
                ]
            ),
            lengths=torch.as_tensor([9, 12, 9]),
        )
        self._test_ec(tables=[eb3_config, eb1_config, eb2_config], features=features)

    def test_different_quantization_dtype_per_ebc_table(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self, m: torch.nn.Module) -> None:
                super().__init__()
                self.m = m

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = TestModule(ebc)
        trec_infer.modules.quantize_embeddings(
            model,
            dtype=torch.int8,
            inplace=True,
            per_table_weight_dtype={"t1": torch.float16},
        )
        # pyrefly: ignore[not-callable]
        configs = model.m.embedding_bag_configs()
        self.assertEqual(len(configs), 2)
        self.assertNotEqual(configs[0].name, configs[1].name)
        for config in configs:
            if config.name == "t1":
                self.assertEqual(config.data_type, DataType.FP16)
            else:
                self.assertEqual(config.name, "t2")
                self.assertEqual(config.data_type, DataType.INT8)

    def test_trace_and_script(self) -> None:
        data_type = DataType.FP16
        quant_type = torch.half
        output_type = torch.half

        ec1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1", "f2"],
            data_type=data_type,
        )
        ec2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f3", "f4"],
            data_type=data_type,
        )

        ec = EmbeddingCollection(tables=[ec1_config, ec2_config])
        # pyrefly: ignore[bad-argument-type]
        ec.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qec = QuantEmbeddingCollection.from_float(ec)

        from torchrec.fx import symbolic_trace

        gm = symbolic_trace(qec)

        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3", "f4"],
            values=torch.as_tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.as_tensor([1, 2, 3, 2]),
        )

        original_out = qec(features)
        traced_out = gm(features)

        scripted_module = torch.jit.script(gm)
        scripted_out = scripted_module(features)
        self._comp_ec_output(original_out, traced_out, atol=0)
        self._comp_ec_output(original_out, scripted_out, atol=0)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs available",
    )
    @given(
        offsets_dtype=st.sampled_from(
            [
                torch.int32,
                torch.int64,
            ]
        ),
        indices_dtype=st.sampled_from(
            [
                torch.int32,
                torch.int64,
            ]
        ),
        device=st.sampled_from(
            [
                torch.device("cpu"),
                torch.device("cuda"),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_fx_unwrap_unsharded_vs_sharded_in_sync(
        self,
        offsets_dtype: torch.dtype,
        indices_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3", "f4"],
            values=torch.tensor(
                [0, 1, 2, 3, 4, 5, 6, 7], dtype=indices_dtype, device=device
            ),
            offsets=torch.tensor([0, 2, 5, 7, 8], dtype=offsets_dtype, device=device),
        )

        indices, offsets = _fx_trec_unwrap_kjt(features)
        self.assertEqual(indices.dtype, offsets.dtype)
        if device.type == "cpu":
            sharded_indices, sharded_offsets, _ = _unwrap_kjt_for_cpu(
                features, weighted=False
            )
            self.assertEqual(sharded_indices.dtype, indices_dtype)
        else:  # cuda
            sharded_indices, sharded_offsets, _ = _unwrap_kjt(features)
            self.assertEqual(sharded_indices.dtype, torch.int32)  # only option!

        self.assertEqual(indices.dtype, sharded_indices.dtype)
        self.assertEqual(offsets.dtype, sharded_offsets.dtype)

    def test_using_flattened_or_unflattened_length_rebatching(self) -> None:
        data_type = DataType.FP16
        quant_type = torch.half
        output_type = torch.half

        ec1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1", "f2"],
            data_type=data_type,
        )
        ec2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f3", "f4"],
            data_type=data_type,
        )

        ec = EmbeddingCollection(tables=[ec1_config, ec2_config])
        # pyrefly: ignore[bad-argument-type]
        ec.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qec = QuantEmbeddingCollection.from_float(ec)

        import copy

        from torchrec.fx import symbolic_trace

        # test using flattened lengths for rebatching (default)

        gm = symbolic_trace(copy.deepcopy(qec))

        found_get_unflattened_lengths_func = False

        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.name == _get_unflattened_lengths.__name__
            ):
                found_get_unflattened_lengths_func = True
                for user in node.users:
                    if (
                        user.op == "call_function"
                        and user.name == _get_batching_hinted_output.__name__
                    ):
                        self.assertTrue(
                            False,
                            "Should not call _get_batching_hinted_output after _get_unflattened_lengths",
                        )

        self.assertTrue(
            found_get_unflattened_lengths_func,
            "_get_unflattened_lengths must exist in the graph",
        )

        # test using unflattened lengths for rebatching

        setattr(qec, MODULE_ATTR_USE_UNFLATTENED_LENGTHS_FOR_BATCHING, True)

        gm = symbolic_trace(qec)

        found_get_unflattened_lengths_func = False
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.name == _get_unflattened_lengths.__name__
            ):
                found_get_unflattened_lengths_func = True
                found_get_batching_hinted_output_func = False
                for user in node.users:
                    if (
                        user.op == "call_function"
                        and user.name == _get_batching_hinted_output.__name__
                    ):
                        found_get_batching_hinted_output_func = True
                self.assertTrue(
                    found_get_batching_hinted_output_func,
                    "Should call _get_batching_hinted_output after _get_unflattened_lengths",
                )

        self.assertTrue(
            found_get_unflattened_lengths_func,
            "_get_unflattened_lengths must exist in the graph",
        )
