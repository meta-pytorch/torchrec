#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest
from typing import cast

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ModuleSharder
from torchrec.distributed.infer_utils import (
    get_all_torchrec_modules,
    get_tbe_specs_from_sharded_module,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    ShardedQuantEmbeddingBagCollection,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    table_wise,
)
from torchrec.distributed.test_utils.infer_utils import (
    quantize,
    TestModelInfo,
    TestQuantEBCSharder,
    TestQuantECSharder,
    TorchTypesModelInputWrapper,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingEnv, ShardingPlan, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)


class UtilsTest(unittest.TestCase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_get_tbe_specs_from_sqebc(self) -> None:
        device = torch.device("cuda:0")

        num_features = 3

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 20,
                embedding_dim=(i + 1) * 10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        model = torch.nn.Sequential(
            EmbeddingBagCollection(
                tables=tables,
                device=device,
            )
        )
        model.training = False

        quant_model = quantize(
            model,
            inplace=True,
            output_type=torch.float,
            quant_state_dict_split_scale_bias=True,
        )

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.TABLE_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[f"table_{i}" for i in range(num_features)],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            quant_model[0],
            per_param_sharding={
                "table_0": table_wise(rank=1),
                "table_1": table_wise(rank=0),
                "table_2": table_wise(rank=0),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=2,
            world_size=2,
        )

        plan = ShardingPlan(plan={"": module_plan})

        sharded_model = _shard_modules(
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            module=quant_model[0],
            sharders=[sharder],
            device=device,
            plan=plan,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        specs = get_tbe_specs_from_sharded_module(sharded_model)

        expected_specs = [
            ("table_1", 40, 20, "int8", "EmbeddingLocation.DEVICE"),
            ("table_2", 60, 30, "int8", "EmbeddingLocation.DEVICE"),
            ("table_0", 20, 10, "int8", "EmbeddingLocation.DEVICE"),
        ]

        self.assertEqual(specs, expected_specs)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_get_tbe_specs_from_sqec(self) -> None:
        device = torch.device("cuda:0")

        num_features = 3

        tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 20,
                embedding_dim=10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        model = torch.nn.Sequential(
            EmbeddingCollection(
                tables=tables,
                device=device,
            )
        )
        model.training = False

        quant_model = quantize(
            model,
            inplace=True,
            output_type=torch.float,
            quant_state_dict_split_scale_bias=True,
        )

        sharder = TestQuantECSharder(
            sharding_type=ShardingType.TABLE_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[f"table_{i}" for i in range(num_features)],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            quant_model[0],
            per_param_sharding={
                "table_0": table_wise(rank=1),
                "table_1": table_wise(rank=0),
                "table_2": table_wise(rank=0),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=2,
            world_size=2,
        )

        plan = ShardingPlan(plan={"": module_plan})

        sharded_model = _shard_modules(
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            module=quant_model[0],
            sharders=[sharder],
            device=device,
            plan=plan,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        specs = get_tbe_specs_from_sharded_module(sharded_model)

        expected_specs = [
            ("table_1", 40, 10, "int8", "EmbeddingLocation.DEVICE"),
            ("table_2", 60, 10, "int8", "EmbeddingLocation.DEVICE"),
            ("table_0", 20, 10, "int8", "EmbeddingLocation.DEVICE"),
        ]

        self.assertEqual(specs, expected_specs)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_get_all_torchrec_modules_for_single_module(self) -> None:
        device = torch.device("cuda:0")

        num_features = 2

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 20,
                embedding_dim=(i + 1) * 10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        model = torch.nn.Sequential(
            EmbeddingBagCollection(
                tables=tables,
                device=device,
            )
        )
        model.training = False

        all_trec_mdoules = get_all_torchrec_modules(model)

        quant_model = quantize(
            model,
            inplace=True,
            output_type=torch.float,
            quant_state_dict_split_scale_bias=True,
        )

        all_trec_mdoules = get_all_torchrec_modules(quant_model)

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.TABLE_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[f"table_{i}" for i in range(num_features)],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            quant_model[0],
            per_param_sharding={
                "table_0": table_wise(rank=0),
                "table_1": table_wise(rank=1),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=2,
            world_size=2,
        )

        plan = ShardingPlan(plan={"": module_plan})

        sharded_model = _shard_modules(
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            module=quant_model[0],
            sharders=[sharder],
            device=device,
            plan=plan,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        all_trec_mdoules = get_all_torchrec_modules(sharded_model)
        self.assertDictEqual(all_trec_mdoules, {"": sharded_model})

        all_trec_modules = get_all_torchrec_modules(
            sharded_model, [QuantEmbeddingBagCollection]
        )
        self.assertEqual(all_trec_modules, {})
        self.assertDictEqual

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_get_all_torchrec_modules_for_test_SparseNN_model(self) -> None:
        local_device = torch.device("cuda:0")
        model_info = TestModelInfo(
            sparse_device=local_device,
            dense_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=2,
        )

        model_info.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=512,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(model_info.num_features)
        ]
        model_info.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(model_info.num_weighted_features)
        ]
        model_info.model = TorchTypesModelInputWrapper(
            TestSparseNN(
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
                num_float_features=model_info.num_float_features,
                dense_device=model_info.dense_device,
                sparse_device=model_info.sparse_device,
            )
        )

        model_info.model.training = False
        model_info.quant_model = quantize(
            model_info.model,
            inplace=True,
            quant_state_dict_split_scale_bias=True,
        )

        model_info.sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in model_info.tables],
                ),
            ),
        ]

        sharded_model = _shard_modules(
            module=model_info.quant_model,
            sharders=model_info.sharders,
            device=model_info.sparse_device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        all_trec_mdoules = get_all_torchrec_modules(sharded_model)

        expected_all_trec_modules = {
            "_module.sparse.ebc": sharded_model._module.sparse.ebc,
            "_module.sparse.weighted_ebc": sharded_model._module.sparse.weighted_ebc,
        }

        self.assertDictEqual(
            all_trec_mdoules,
            expected_all_trec_modules,
        )

        all_trec_mdoules = get_all_torchrec_modules(
            sharded_model, [ShardedQuantEmbeddingBagCollection]
        )

        self.assertDictEqual(
            all_trec_mdoules,
            {
                "_module.sparse.ebc": sharded_model._module.sparse.ebc,
            },
        )
