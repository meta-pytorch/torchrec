#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest
from dataclasses import dataclass
from enum import Enum
from typing import cast, List, Tuple

import torch
from torch.distributed import ProcessGroup
from torchrec import EmbeddingCollection, EmbeddingConfig
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    KJTList,
    ListOfKJTList,
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import DEFAULT_PERF_ESTIMATOR
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.estimator import EmbeddingPerfEstimatorFactory
from torchrec.distributed.planner.shard_estimators import EmbeddingStorageEstimator
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    KJTInputWrapper,
    model_input_to_forward_args,
    model_input_to_forward_args_kjt,
    prep_inputs,
    quantize,
    TestModelInfo,
    TestQuantEBCSharder,
    TestQuantECSharder,
    TorchTypesModelInputWrapper,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import Awaitable, ShardingEnv
from torchrec.fx.tracer import Tracer as TorchrecFxTracer
from torchrec.fx.utils import fake_range
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class FxJitTestType(Enum):
    CREATE_ONLY = 0
    FX = 1
    FX_JIT = 2


@dataclass
class Context:
    process_group: ProcessGroup


class ModelTraceScriptTest(unittest.TestCase):
    def _set_up_qebc(
        self, sharding_type: str, quant_state_dict_split_scale_bias: bool
    ) -> TestModelInfo:
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
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

        model_info.sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                    shardable_params=[table.name for table in model_info.tables],
                ),
            ),
            cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder()),
        ]

        return model_info

    def _set_up_qec(
        self, sharding_type: str, quant_state_dict_split_scale_bias: bool
    ) -> TestModelInfo:
        local_device = torch.device("cuda:0")
        model_info = TestModelInfo(
            sparse_device=local_device,
            dense_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
        )
        model_info.tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(model_info.num_features)
        ]

        model_info.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=model_info.tables,
                    device=model_info.sparse_device,
                )
            )
        )

        model_info.model.training = False
        model_info.quant_model = quantize(
            model_info.model,
            inplace=True,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

        model_info.sharders = [
            cast(
                ModuleSharder[torch.nn.Module],
                TestQuantECSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.QUANT.value,
                ),
            )
        ]

        return model_info

    def shard_modules_QEBC(
        self,
        world_size: int,
        sharding_type: str,
        quant_state_dict_split_scale_bias: bool,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qebc(sharding_type, quant_state_dict_split_scale_bias)
        sharded_model = _shard_modules(
            module=model_info.quant_model,
            sharders=model_info.sharders,
            device=model_info.sparse_device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
        )

        inputs = prep_inputs(model_info, world_size, long_indices=False)

        return (
            model_info.quant_model,
            sharded_model,
            [
                model_input_to_forward_args(inp.to(model_info.sparse_device))
                for inp in inputs
            ],
        )

    def shard_modules_QEC(
        self,
        world_size: int,
        sharding_type: str,
        quant_state_dict_split_scale_bias: bool,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qec(sharding_type, quant_state_dict_split_scale_bias)
        sharded_model = _shard_modules(
            module=model_info.quant_model,
            sharders=model_info.sharders,
            device=model_info.sparse_device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
        )

        inputs = prep_inputs(model_info, world_size, long_indices=False)

        return (
            model_info.quant_model,
            sharded_model,
            [
                model_input_to_forward_args_kjt(inp.to(model_info.sparse_device))
                for inp in inputs
            ],
        )

    def DMP_QEBC(
        self,
        world_size: int,
        sharding_type: str,
        quant_state_dict_split_scale_bias: bool,
        unwrap_dmp: bool,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qebc(sharding_type, quant_state_dict_split_scale_bias)
        topology = Topology(world_size=world_size, compute_device="cuda")
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
        ).plan(model_info.quant_model, model_info.sharders)

        dmp = DistributedModelParallel(
            model_info.quant_model,
            plan=plan,
            sharders=model_info.sharders,
            device=model_info.sparse_device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
            init_data_parallel=False,
        )

        dmp = dmp.copy(model_info.sparse_device)

        inputs = prep_inputs(model_info, world_size, long_indices=False)

        m = dmp.module if unwrap_dmp else dmp
        return (
            model_info.quant_model,
            m,
            [
                model_input_to_forward_args(inp.to(model_info.sparse_device))
                for inp in inputs
            ],
        )

    def DMP_QEC(
        self,
        world_size: int,
        sharding_type: str,
        quant_state_dict_split_scale_bias: bool,
        sharding_enabled: bool,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, List[Tuple]]:
        model_info = self._set_up_qec(sharding_type, quant_state_dict_split_scale_bias)

        if sharding_enabled:
            topology = Topology(world_size=world_size, compute_device="cuda")
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
            ).plan(model_info.quant_model, model_info.sharders)
            m = DistributedModelParallel(
                model_info.quant_model,
                plan=plan,
                sharders=model_info.sharders,
                device=model_info.sparse_device,
                env=ShardingEnv.from_local(world_size=world_size, rank=0),
                init_data_parallel=False,
            )
            model_info.model = m.module

        inputs = prep_inputs(model_info, world_size, long_indices=False)

        return (
            model_info.quant_model,
            model_info.model,
            [
                model_input_to_forward_args_kjt(inp.to(model_info.sparse_device))
                for inp in inputs
            ],
        )

    def _models_with_inputs(
        self,
        *args,
        **kwargs,
    ) -> List[Tuple[torch.nn.Module, torch.nn.Module, List[Tuple], FxJitTestType]]:
        return [
            (*fn(*args, **kwargs), test_type)
            for fn, test_type in [
                (
                    lambda world_size: self.DMP_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=False,
                        unwrap_dmp=True,  # preferred usage is to provide fx trace with unwrapped dmp
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=False,
                        unwrap_dmp=False,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.ROW_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                        unwrap_dmp=False,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=False,
                        sharding_enabled=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                        sharding_enabled=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.DMP_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=False,
                        sharding_enabled=False,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=False,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.ROW_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEBC(
                        world_size=world_size,
                        sharding_type=ShardingType.COLUMN_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=False,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.TABLE_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.ROW_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
                (
                    lambda world_size: self.shard_modules_QEC(
                        world_size=world_size,
                        sharding_type=ShardingType.COLUMN_WISE.value,
                        quant_state_dict_split_scale_bias=True,
                    ),
                    FxJitTestType.FX_JIT,
                ),
            ]
        ]

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_fxtrace_jitscript(self) -> None:
        for non_sharded_model, model, inputs, test_type in self._models_with_inputs(
            world_size=2,
        ):
            # We need more than one input to verify correctness of tracing and scripting using input different from what was used for tracing
            assert len(inputs) > 1

            # Run model first time to go through lazy initialized blocks before tracing
            # Targeting only inference for this time
            non_sharded_model(*inputs[0])
            eager_output = model(*inputs[0])

            if test_type == FxJitTestType.CREATE_ONLY:
                continue

            tracer = TorchrecFxTracer()
            graph = tracer.trace(model)

            gm = torch.fx.GraphModule(tracer.root, graph)

            if test_type == FxJitTestType.FX_JIT:
                gm_script = torch.jit.script(gm)
                gm_script_output = gm_script(*inputs[0])

                if isinstance(eager_output, Awaitable):
                    eager_output = eager_output.wait()

                assert_close(eager_output, gm_script_output)

                for inp in inputs[1:]:
                    eager_output = model(*inp)
                    script_output = gm_script(*inp)
                    assert_close(eager_output, script_output)

    def test_jitscript(self) -> None:
        # Check main types to be torch jit scriptable
        for clz in [
            JaggedTensor,
            KeyedJaggedTensor,
            KeyedTensor,
            KJTList,
            ListOfKJTList,
        ]:
            # Using torch.jit._script._recursive_compile_class instead of torch.jit.script
            # As classes later is more restrictive, checking no inheritance
            # (e.g. Multistreamable which we so far do not need in jit script) etc.
            # We need those classes not as it is, but as composable blocks in model.
            # _recursive_compile_class for that is enough
            torch.jit._script._recursive_compile_class(clz, fake_range())
        torch.jit.script(KeyedJaggedTensor.from_jt_dict)

    def test_jitscript_kjt(self) -> None:
        def kjt_split(segments: List[int]) -> List[KeyedJaggedTensor]:
            kjt = KeyedJaggedTensor(
                keys=["a", "b", "c"],
                values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                lengths=torch.tensor([2, 0, 1, 1, 1, 2]),
            )
            return kjt.split(segments)

        sm = torch.jit.script(kjt_split)
        sm([1, 0, 2, 0])
