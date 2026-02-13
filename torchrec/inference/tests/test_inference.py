#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

import unittest
from argparse import Namespace
from typing import Any, cast, Dict, List

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec import PoolingType
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed.fused_params import (
    FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP,
    FUSED_PARAM_REGISTER_TBE_BOOL,
)
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestOverArchRegroupModule,
    TestSparseNN,
)
from torchrec.distributed.types import ModuleSharder

from torchrec.inference.dlrm_predict import (
    create_training_batch,
    DLRMModelConfig,
    DLRMPredictFactory,
)
from torchrec.inference.modules import (
    assign_weights_to_tbe,
    DEFAULT_FUSED_PARAMS,
    DEFAULT_SHARDERS,
    get_table_to_weights_from_tbe,
    quantize_inference_model,
    set_pruning_data,
    shard_quant_model,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class InferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

    def test_dlrm_inference_package(self) -> None:
        args = Namespace()
        args.batch_size = 10
        args.num_embedding_features = 26
        args.num_dense_features = len(DEFAULT_INT_NAMES)
        args.dense_arch_layer_sizes = "512,256,64"
        args.over_arch_layer_sizes = "512,512,256,1"
        args.sparse_feature_names = ",".join(DEFAULT_CAT_NAMES)
        args.num_embeddings = 100_000
        args.num_embeddings_per_feature = ",".join(
            [str(args.num_embeddings)] * args.num_embedding_features
        )

        batch = create_training_batch(args)

        model_config = DLRMModelConfig(
            dense_arch_layer_sizes=list(
                map(int, args.dense_arch_layer_sizes.split(","))
            ),
            dense_in_features=args.num_dense_features,
            embedding_dim=64,
            id_list_features_keys=args.sparse_feature_names.split(","),
            num_embeddings_per_feature=list(
                map(int, args.num_embeddings_per_feature.split(","))
            ),
            num_embeddings=args.num_embeddings,
            over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
            sample_input=batch,
        )

        # Create torchscript model for inference
        DLRMPredictFactory(model_config).create_predict_module(
            world_size=1, device="cpu"
        )

    def test_regroup_module_inference(self) -> None:
        set_propogate_device(True)
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=torch.device("cpu"),
            sparse_device=torch.device("cpu"),
            over_arch_clazz=TestOverArchRegroupModule,
        )

        model.eval()
        _, local_batch = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )

        with torch.inference_mode():
            output = model(local_batch[0])

            # Quantize the model and collect quantized weights
            quantized_model = quantize_inference_model(model)
            quantized_output = quantized_model(local_batch[0])
            table_to_weight = get_table_to_weights_from_tbe(quantized_model)

            # Shard the model, all weights are initialized back to 0, so have to reassign weights
            sharded_quant_model, _ = shard_quant_model(
                quantized_model,
                world_size=2,
                compute_device="cpu",
                sharding_device="cpu",
            )
            assign_weights_to_tbe(quantized_model, table_to_weight)

            sharded_quant_output = sharded_quant_model(local_batch[0])

            self.assertTrue(torch.allclose(output, quantized_output, atol=1e-4))
            self.assertTrue(torch.allclose(output, sharded_quant_output, atol=1e-4))

    def test_set_pruning_data(self) -> None:
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=torch.device("cpu"),
            sparse_device=torch.device("cpu"),
            over_arch_clazz=TestOverArchRegroupModule,
        )

        pruning_dict = {}

        for table in self.tables:
            pruning_dict[table.name] = table.num_embeddings - 1

        set_pruning_data(model, pruning_dict)
        quantized_model = quantize_inference_model(model)

        # Check EBC configs and TBE for correct shapes
        for module in quantized_model.modules():
            if isinstance(module, EmbeddingBagCollection):
                for config in module.embedding_bag_configs():
                    if config.name in pruning_dict:
                        self.assertEqual(
                            config.num_embeddings_post_pruning,
                            pruning_dict[config.name],
                        )
            elif module.__class__.__name__ == "IntNBitTableBatchedEmbeddingBagsCodegen":
                # pyrefly: ignore[bad-argument-type]
                for i, spec in enumerate(module.embedding_specs):
                    if spec[0] in pruning_dict:
                        self.assertEqual(
                            # pyrefly: ignore[not-callable]
                            module.split_embedding_weights()[i][0].size(0),
                            pruning_dict[spec[0]],
                        )
                        self.assertEqual(
                            spec[1],
                            pruning_dict[spec[0]],
                        )

    def test_quantize_per_table_dtype(self) -> None:
        max_feature_lengths = {}

        # First two tables as FPEBC
        max_feature_lengths[self.tables[0].name] = 100
        max_feature_lengths[self.tables[1].name] = 100

        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=torch.device("cpu"),
            sparse_device=torch.device("cpu"),
            over_arch_clazz=TestOverArchRegroupModule,
            max_feature_lengths=max_feature_lengths,
        )

        per_table_dtype = {}

        for table in self.tables + self.weighted_tables:
            # quint4x2 different than int8, which is default
            per_table_dtype[table.name] = torch.quint4x2

        quantized_model = quantize_inference_model(
            model, per_table_weight_dtype=per_table_dtype
        )

        num_tbes = 0
        # Check EBC configs and TBE for correct shapes
        for module in quantized_model.modules():
            if module.__class__.__name__ == "IntNBitTableBatchedEmbeddingBagsCodegen":
                num_tbes += 1
                # pyrefly: ignore[bad-argument-type]
                for i, spec in enumerate(module.embedding_specs):
                    self.assertEqual(spec[3], SparseType.INT4)

        # 3 TBES (1 FPEBC, 2 EBCs (1 weighted, 1 unweighted))

        self.assertEqual(num_tbes, 3)

    def test_sharded_quantized_tbe_count(self) -> None:
        set_propogate_device(True)

        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=torch.device("cpu"),
            sparse_device=torch.device("cpu"),
            over_arch_clazz=TestOverArchRegroupModule,
        )

        per_table_weight_dtypes = {}

        for table in self.tables + self.weighted_tables:
            # quint4x2 different than int8, which is default
            per_table_weight_dtypes[table.name] = (
                torch.quint4x2 if table.name == "table_0" else torch.quint8
            )

        model.eval()
        _, local_batch = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )

        # with torch.inference_mode(): # TODO: Why does inference mode fail when using different quant data types
        output = model(local_batch[0])

        # Quantize the model and collect quantized weights
        quantized_model = quantize_inference_model(
            model, per_table_weight_dtype=per_table_weight_dtypes
        )
        quantized_output = quantized_model(local_batch[0])
        table_to_weight = get_table_to_weights_from_tbe(quantized_model)

        # Shard the model, all weights are initialized back to 0, so have to reassign weights
        sharded_quant_model, _ = shard_quant_model(
            quantized_model,
            world_size=1,
            compute_device="cpu",
            sharding_device="cpu",
        )
        assign_weights_to_tbe(quantized_model, table_to_weight)
        sharded_quant_output = sharded_quant_model(local_batch[0])

        # When world_size = 1, we should have 1 TBE per sharded, quantized ebc
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(len(sharded_quant_model.sparse.ebc.tbes) == 1)
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(len(sharded_quant_model.sparse.weighted_ebc.tbes) == 1)

        # Check the weights are close
        self.assertTrue(torch.allclose(output, quantized_output, atol=1e-3))
        self.assertTrue(torch.allclose(output, sharded_quant_output, atol=1e-3))

        # Check the sizes are correct
        expected_num_embeddings = {}

        for table in self.tables:
            expected_num_embeddings[table.name] = table.num_embeddings

        for module in quantized_model.modules():
            if module.__class__.__name__ == "IntNBitTableBatchedEmbeddingBagsCodegen":
                # pyrefly: ignore[bad-argument-type]
                for i, spec in enumerate(module.embedding_specs):
                    if spec[0] in expected_num_embeddings:
                        # We only expect the first table to be quantized to int4 due to test set up
                        if spec[0] == "table_0":
                            self.assertEqual(spec[3], SparseType.INT4)
                        else:
                            self.assertEqual(spec[3], SparseType.INT8)

                        # Check sizes are equal
                        self.assertEqual(
                            # pyrefly: ignore[not-callable]
                            module.split_embedding_weights()[i][0].size(0),
                            expected_num_embeddings[spec[0]],
                        )
                        self.assertEqual(
                            spec[1],
                            expected_num_embeddings[spec[0]],
                        )

    def test_sharded_quantized_lengths_to_tbe(self) -> None:
        set_propogate_device(True)

        fused_params: Dict[str, Any] = {FUSED_PARAM_LENGTHS_TO_OFFSETS_LOOKUP: True}
        sharders: List[ModuleSharder[torch.nn.Module]] = [
            cast(
                ModuleSharder[torch.nn.Module],
                QuantEmbeddingBagCollectionSharder(fused_params=fused_params),
            ),
        ]

        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=torch.device("cpu"),
            sparse_device=torch.device("cpu"),
            over_arch_clazz=TestOverArchRegroupModule,
        )

        model.eval()
        _, local_batch = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )

        # with torch.inference_mode(): # TODO: Why does inference mode fail when using different quant data types
        output = model(local_batch[0])

        # Quantize the model and collect quantized weights
        quantized_model = quantize_inference_model(model)
        quantized_output = quantized_model(local_batch[0])
        table_to_weight = get_table_to_weights_from_tbe(quantized_model)

        # Shard the model, all weights are initialized back to 0, so have to reassign weights
        sharded_quant_model, _ = shard_quant_model(
            quantized_model,
            world_size=1,
            compute_device="cpu",
            sharding_device="cpu",
            sharders=sharders,
        )
        assign_weights_to_tbe(quantized_model, table_to_weight)
        sharded_quant_output = sharded_quant_model(local_batch[0])

        # When world_size = 1, we should have 1 TBE per sharded, quantized ebc
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(len(sharded_quant_model.sparse.ebc.tbes) == 1)
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(len(sharded_quant_model.sparse.weighted_ebc.tbes) == 1)

        # Check the weights are close
        self.assertTrue(torch.allclose(output, quantized_output, atol=1e-3))
        self.assertTrue(torch.allclose(output, sharded_quant_output, atol=1e-3))

    def test_quantized_tbe_count_different_pooling(self) -> None:
        set_propogate_device(True)

        self.tables[0].pooling = PoolingType.MEAN
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=torch.device("cpu"),
            sparse_device=torch.device("cpu"),
            over_arch_clazz=TestOverArchRegroupModule,
        )

        model.eval()
        _, local_batch = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )

        model(local_batch[0])

        # Quantize the model and collect quantized weights
        quantized_model = quantize_inference_model(model)
        # We should have 2 TBEs for unweighted ebc as the 2 tables here have different pooling types
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(len(quantized_model.sparse.ebc.tbes) == 2)
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(len(quantized_model.sparse.weighted_ebc.tbes) == 1)
        # Changing this back
        self.tables[0].pooling = PoolingType.SUM

    def test_fused_params_overwrite(self) -> None:
        orig_value = DEFAULT_FUSED_PARAMS[FUSED_PARAM_REGISTER_TBE_BOOL]

        sharders = DEFAULT_SHARDERS
        ebc_sharder = sharders[0]
        # pyrefly: ignore[missing-attribute]
        ebc_fused_params = ebc_sharder.fused_params
        ebc_fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = -1

        ec_sharder = sharders[1]
        # pyrefly: ignore[missing-attribute]
        ec_fused_params = ec_sharder.fused_params

        # Make sure that overwrite of ebc_fused_params is not reflected in ec_fused_params
        self.assertEqual(ec_fused_params[FUSED_PARAM_REGISTER_TBE_BOOL], orig_value)

        # change it back to the original value because it modifies the global variable
        # otherwise it will affect other tests
        ebc_fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = orig_value
