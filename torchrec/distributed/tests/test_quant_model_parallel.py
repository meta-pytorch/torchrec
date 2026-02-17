#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict, Optional, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ModuleSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import DEFAULT_PERF_ESTIMATOR
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.estimator import EmbeddingPerfEstimatorFactory
from torchrec.distributed.planner.shard_estimators import EmbeddingStorageEstimator
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.infer_utils import quantize, TestQuantEBCSharder
from torchrec.distributed.test_utils.test_model import (
    _get_default_rtol_and_atol,
    ModelInput,
    TestSparseNN,
)
from torchrec.distributed.types import ShardedModule, ShardingEnv, ShardingType
from torchrec.distributed.utils import copy_to_device
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.types import CopyMixIn


class CopyModule(nn.Module, CopyMixIn):
    def __init__(self) -> None:
        super().__init__()
        self.tensor: torch.Tensor = torch.empty((10), device="cpu")

    def copy(self, device: torch.device) -> nn.Module:
        self.tensor = self.tensor.to(device)
        return self


class NoCopyModule(nn.Module, CopyMixIn):
    def __init__(self) -> None:
        super().__init__()
        self.tensor: torch.Tensor = torch.empty((10), device="cpu")

    def copy(self, device: torch.device) -> nn.Module:
        return self


class QuantModelParallelModelCopyTest(unittest.TestCase):
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

    def _buffer_param_check(
        self,
        module: nn.Module,
        module_copy: nn.Module,
        device: torch.device,
        device_copy: torch.device,
        recurse: bool = True,
    ) -> None:
        # check all buffer/param under the module is value-identical
        # but device-different with the copied module.
        for (name, buffer), (name_copy, buffer_copy) in zip(
            list(module.named_buffers(recurse=recurse))
            + list(module.named_parameters(recurse=recurse)),
            list(module_copy.named_buffers(recurse=recurse))
            + list(module_copy.named_parameters(recurse=recurse)),
        ):
            self.assertEqual(name, name_copy)
            actual, expected = buffer.detach().cpu(), buffer_copy.detach().cpu()
            rtol, atol = _get_default_rtol_and_atol(actual, expected)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            self.assertEqual(buffer.detach().device, device)
            self.assertEqual(buffer_copy.detach().device, device_copy)

    def _recursive_device_check(
        self,
        module: nn.Module,
        module_copy: nn.Module,
        device: torch.device,
        device_copy: torch.device,
    ) -> None:
        if isinstance(module, ShardedModule):
            # sparse part parameter needs to be the same reference
            # TBE ops's parameter is accessed via buffer.
            for name_buffer, name_buffer_copy in zip(
                module.named_buffers(),
                module_copy.named_buffers(),
            ):
                name, buffer = name_buffer
                name_copy, buffer_copy = name_buffer_copy
                self.assertEqual(name, name_copy)
                # compare tensor storage reference
                self.assertTrue(buffer.detach().is_set_to(buffer_copy.detach()))
            # don't go into named_children of ShardedModule
            return
        self._buffer_param_check(
            module, module_copy, device, device_copy, recurse=False
        )
        for name_child, name_child_copy in zip(
            module.named_children(), module_copy.named_children()
        ):
            name, child = name_child
            name_copy, child_copy = name_child_copy
            if not any(
                [isinstance(submodule, ShardedModule) for submodule in child.modules()]
            ):
                # other part parameter/buffer needs to be
                # identical in value and different in device
                self._buffer_param_check(child, child_copy, device, device_copy)
            else:
                self._recursive_device_check(child, child_copy, device, device_copy)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    @given(
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_quant_pred(
        self, output_type: torch.dtype, sharding_type_qsplitscalebias: Tuple[str, bool]
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")
        device_1 = torch.device("cuda:1")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = quantize(
            model,
            inplace=True,
            output_type=output_type,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
        dmp = DistributedModelParallel(
            quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantEBCSharder(
                        sharding_type=sharding_type,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
            init_data_parallel=False,
        )
        dmp_1 = dmp.copy(device_1)
        self._recursive_device_check(dmp.module, dmp_1.module, device, device_1)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    @given(
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_quant_pred_state_dict(
        self, output_type: torch.dtype, sharding_type_qsplitscalebias: Tuple[str, bool]
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")

        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = quantize(
            model,
            inplace=True,
            output_type=output_type,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
        model.training = False

        dmp = DistributedModelParallel(
            quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantEBCSharder(
                        sharding_type=sharding_type,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
            init_data_parallel=False,
        )

        dmp_copy = DistributedModelParallel(
            quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantEBCSharder(
                        sharding_type=sharding_type,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
            init_data_parallel=False,
        )

        _, local_batch = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )

        # pyrefly: ignore[bad-argument-type]
        dmp_copy.load_state_dict(dmp.state_dict())
        torch.testing.assert_close(
            dmp(local_batch[0].to(device)).cpu(),
            dmp_copy(local_batch[0].to(device)).cpu(),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    @given(
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_quant_pred_shard(
        self, output_type: torch.dtype, sharding_type_qsplitscalebias: Tuple[str, bool]
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")
        device_1 = torch.device("cuda:1")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = quantize(
            model,
            inplace=True,
            output_type=output_type,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

        sharded_model = _shard_modules(
            module=quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantEBCSharder(
                        sharding_type=sharding_type,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        sharded_model_copy = copy_to_device(
            module=sharded_model, current_device=device, to_device=device_1
        )

        self._recursive_device_check(
            sharded_model, sharded_model_copy, device, device_1
        )

        _, local_batch = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )

        torch.testing.assert_close(
            sharded_model(local_batch[0].to(device)).cpu(),
            sharded_model_copy(local_batch[0].to(device_1)).cpu(),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_copy_mixin(self) -> None:
        device = torch.device("cuda:0")
        device_1 = torch.device("cuda:1")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        model.copy_module = CopyModule()
        model.no_copy_module = NoCopyModule()
        quant_model = quantize(model, inplace=True)
        dmp = DistributedModelParallel(
            quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantEBCSharder(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=None,
            env=ShardingEnv.from_local(world_size=2, rank=0),
            init_data_parallel=False,
        )

        dmp_1 = dmp.copy(device_1)
        # pyrefly: ignore[missing-attribute]
        self.assertEqual(dmp_1.module.copy_module.tensor.device, device_1)
        # pyrefly: ignore[missing-attribute]
        self.assertEqual(dmp_1.module.no_copy_module.tensor.device, torch.device("cpu"))


class QuantModelParallelModelSharderTest(unittest.TestCase):
    def setUp(self) -> None:
        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100000,
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

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs available",
    )
    @given(
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
        per_table_weight_dtypes=st.sampled_from(
            [
                None,
                {"table_0": torch.quint4x2, "table_1": torch.quint8},
                {
                    "table_0": torch.quint4x2,
                    "table_1": torch.quint8,
                    "table_3": torch.quint4x2,
                    "table_4": torch.int8,
                },
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_shard_one_ebc_cuda(
        self,
        sharding_type_qsplitscalebias: Tuple[str, bool],
        per_table_weight_dtypes: Optional[Dict[str, torch.dtype]],
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")
        sparse_device = torch.device("meta")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=sparse_device,
        )
        quant_model = quantize(
            model,
            inplace=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
            per_table_weight_dtypes=per_table_weight_dtypes,
        )
        sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in self.tables],
                ),
            )
        ]
        topology = Topology(world_size=1, compute_device="cuda")
        plan = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=10,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=1,
                estimator=[
                    EmbeddingPerfEstimatorFactory.create(
                        DEFAULT_PERF_ESTIMATOR, topology=topology, is_inference=True
                    ),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
            # pyrefly: ignore[bad-argument-type, missing-argument]
        ).plan(quant_model, sharders)

        sharding_device_type = "cuda"
        dmp = DistributedModelParallel(
            quant_model,
            plan=plan,
            device=torch.device(sharding_device_type),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            init_data_parallel=False,
            init_parameters=False,
        )
        self.assertTrue(
            # flake8: noqa:C419
            all(
                param.device.type == sharding_device_type
                #  attribute `ebc`.
                # pyrefly: ignore[missing-attribute]
                for param in dmp.module.sparse.ebc.buffers()
            )
        )
        self.assertTrue(
            # flake8: noqa:C419
            all(
                param.device.type == sparse_device.type
                #  attribute `weighted_ebc`.
                # pyrefly: ignore[missing-attribute]
                for param in dmp.module.sparse.weighted_ebc.buffers()
            )
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs available",
    )
    @given(
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_shard_one_ebc_meta(
        self, sharding_type_qsplitscalebias: Tuple[str, bool]
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = quantize(
            model,
            inplace=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
        sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in self.tables],
                ),
            )
        ]
        topology = Topology(world_size=1, compute_device="cuda")
        plan = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=10,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=1,
                estimator=[
                    EmbeddingPerfEstimatorFactory.create(
                        DEFAULT_PERF_ESTIMATOR, topology=topology, is_inference=True
                    ),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
            # pyrefly: ignore[bad-argument-type, missing-argument]
        ).plan(quant_model, sharders)

        dmp = DistributedModelParallel(
            quant_model,
            plan=plan,
            device=torch.device("cuda"),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            init_data_parallel=False,
            init_parameters=False,
        )

        self.assertTrue(
            # pyrefly: ignore[missing-attribute]
            all(param.device == device for param in dmp.module.sparse.ebc.buffers())
        )
        self.assertTrue(
            # flake8: noqa:C419
            all(
                param.device == torch.device("meta")
                #  attribute `weighted_ebc`.
                # pyrefly: ignore[missing-attribute]
                for param in dmp.module.sparse.weighted_ebc.buffers()
            )
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs available",
    )
    @given(
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_shard_all_ebcs(
        self, sharding_type_qsplitscalebias: Tuple[str, bool]
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")
        sparse_device = torch.device("meta")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=sparse_device,
        )
        quant_model = quantize(
            model,
            inplace=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
        sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                ),
            )
        ]
        topology = Topology(world_size=1, compute_device="cuda")
        plan = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=10,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=1,
                estimator=[
                    EmbeddingPerfEstimatorFactory.create(
                        DEFAULT_PERF_ESTIMATOR, topology=topology, is_inference=True
                    ),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
            # pyrefly: ignore[bad-argument-type, missing-argument]
        ).plan(quant_model, sharders)

        dmp = DistributedModelParallel(
            quant_model,
            plan=plan,
            device=torch.device("cuda"),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            init_data_parallel=False,
            init_parameters=True,
        )

        self.assertTrue(
            all(
                param.device.type == device.type
                #  attribute `ebc`.
                # pyrefly: ignore[missing-attribute]
                for param in dmp.module.sparse.ebc.buffers()
            )
        )
        # DMP init_parameters == True by default reinits meta parameters on sharding device
        self.assertTrue(
            all(
                param.device.type == device.type
                #  attribute `weighted_ebc`.
                # pyrefly: ignore[missing-attribute]
                for param in dmp.module.sparse.weighted_ebc.buffers()
            )
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs available",
    )
    @given(
        sharding_type_qsplitscalebias=st.sampled_from(
            [
                (ShardingType.TABLE_WISE.value, False),
                (ShardingType.TABLE_WISE.value, True),
                (ShardingType.ROW_WISE.value, True),
                (ShardingType.COLUMN_WISE.value, True),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_sharder_bad_param_config(
        self, sharding_type_qsplitscalebias: Tuple[str, bool]
    ) -> None:
        (
            sharding_type,
            quant_state_dict_split_scale_bias,
        ) = sharding_type_qsplitscalebias

        device = torch.device("cuda:0")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = quantize(
            model,
            inplace=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
        sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[
                        table.name for table in self.tables[:-1]
                    ],  # partial list of shardable params
                ),
            )
        ]
        topology = Topology(world_size=1, compute_device="cuda")
        with self.assertRaises(AssertionError):
            EmbeddingShardingPlanner(
                topology=topology,
                batch_size=10,
                enumerator=EmbeddingEnumerator(
                    topology=topology,
                    batch_size=1,
                    estimator=[
                        EmbeddingPerfEstimatorFactory.create(
                            DEFAULT_PERF_ESTIMATOR, topology=topology, is_inference=True
                        ),
                        EmbeddingStorageEstimator(topology=topology),
                    ],
                ),
                # pyrefly: ignore[bad-argument-type, missing-argument]
            ).plan(quant_model, sharders)
