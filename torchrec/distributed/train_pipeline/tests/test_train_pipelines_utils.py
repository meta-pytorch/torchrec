#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import enum
import unittest
from typing import Tuple, Union
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestNegSamplingModule,
    TestSparseNN,
)
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.runtime_forwards import PipelinedForward
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.tracing import CallArgs, PipelinedPostproc
from torchrec.distributed.train_pipeline.utils import _rewrite_model
from torchrec.distributed.types import ShardingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class ModelType(enum.Enum):
    VANILLA = "vanilla"
    SHARDED = "sharded"
    PIPELINED = "pipelined"


@torch.fx.wrap
def enrich_hstu_features(
    kjt: KeyedJaggedTensor, hstu_factor: float
) -> KeyedJaggedTensor:
    if kjt._weights is not None:
        kjt._weights *= hstu_factor
    return kjt


class TrainPipelineUtilsTest(TrainPipelineSparseDistTestBase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_rewrite_model(self) -> None:
        sharding_type = ShardingType.TABLE_WISE.value
        kernel_type = EmbeddingComputeKernel.FUSED.value
        fused_params = {}

        extra_input = ModelInput.generate(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            batch_size=10,
            world_size=1,
            num_float_features=10,
            randomize_indices=False,
        )[0].to(self.device)

        postproc_module = TestNegSamplingModule(
            extra_input=extra_input,
        )
        model = self._setup_model(postproc_module=postproc_module)

        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, fused_params
        )

        # Try to rewrite model without ignored_postproc_modules defined, EBC forwards not overwritten to PipelinedForward due to KJT modification
        _rewrite_model(
            model=sharded_model,
            batch=None,
            context=TrainPipelineContext(),
            dist_stream=None,
        )
        self.assertNotIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.ebc.forward,
            PipelinedForward,
        )
        self.assertNotIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.weighted_ebc.forward,
            PipelinedForward,
        )

        # Now provide postproc module explicitly
        _rewrite_model(
            model=sharded_model,
            batch=None,
            context=TrainPipelineContext(),
            dist_stream=None,
            pipeline_postproc=True,
        )

        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `sparse`.
        self.assertIsInstance(sharded_model.module.sparse.ebc.forward, PipelinedForward)
        self.assertIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.weighted_ebc.forward,
            PipelinedForward,
        )
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `postproc_module`.
            sharded_model.module.postproc_module,
        )
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.weighted_ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `postproc_module`.
            sharded_model.module.postproc_module,
        )
        state_dict = sharded_model.state_dict()
        missing_keys, unexpected_keys = sharded_model.load_state_dict(state_dict)
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])

    def test_pipelined_postproc_state_dict(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.tensor(1.0))

            def forward(self, x):
                return x

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.test_module = TestModule()

            def forward(self, x):
                return self.test_module(x)

        model = TestModel()

        rewritten_model = copy.deepcopy(model)
        # pyre-ignore[8]
        rewritten_model.test_module = PipelinedPostproc(
            postproc_module=rewritten_model.test_module,
            fqn="test_module",
            args=CallArgs(args=[], kwargs={}),
            context=TrainPipelineContext(),
            default_stream=MagicMock(),
            dist_stream=MagicMock(),
        )
        # self-check - we want the state dict be the same between vanilla model and "rewritten model"
        self.assertDictEqual(model.state_dict(), rewritten_model.state_dict())
        state_dict = rewritten_model.state_dict()
        self.assertEqual(list(state_dict.keys()), ["test_module.weight"])

    def _create_model_for_snapshot_test(
        self, source_model_type: ModelType
    ) -> torch.nn.Module:
        if source_model_type == ModelType.VANILLA:
            extra_input = ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=10,
                world_size=1,
                num_float_features=10,
                randomize_indices=False,
            )[0].to(self.device)

            postproc_module = TestNegSamplingModule(
                extra_input=extra_input,
            )
            model = self._setup_model(postproc_module=postproc_module)
            model.to_empty(device=self.device)
            return model
        elif source_model_type == ModelType.SHARDED:
            model = self._create_model_for_snapshot_test(ModelType.VANILLA)
            sharded_model, optim = self._generate_sharded_model_and_optimizer(
                model,
                ShardingType.TABLE_WISE.value,
                EmbeddingComputeKernel.FUSED.value,
                {},
            )
            return sharded_model
        elif source_model_type == ModelType.PIPELINED:
            model = self._create_model_for_snapshot_test(ModelType.SHARDED)
            _rewrite_model(
                model=model,
                batch=None,
                context=TrainPipelineContext(),
                dist_stream=None,
                pipeline_postproc=True,
            )
            return model
        else:
            raise ValueError(f"Unknown model type {source_model_type}")

    def _test_restore_from_snapshot(
        self, source_model_type: ModelType, recipient_model_type: ModelType
    ) -> None:
        source_model = self._create_model_for_snapshot_test(source_model_type)
        recipient_model = self._create_model_for_snapshot_test(recipient_model_type)

        # self-check - we want the state dict be the same between source and recipient
        # although this is not strictly necessary
        # Asserting only on keys since the asserting on entire state dict fails with
        # "Boolean value of Tensor with more than one value is ambiguous" (not sure why)
        self.assertEqual(
            source_model.state_dict().keys(), recipient_model.state_dict().keys()
        )

        state_dict = source_model.state_dict()
        self.assertTrue(
            f"postproc_module.{TestNegSamplingModule.TEST_BUFFER_NAME}"
            in state_dict.keys()
        )

        missing_keys, unexpected_keys = recipient_model.load_state_dict(state_dict)
        # if both are empty, restoring the state dict was successful
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_restore_from_snapshot(self) -> None:
        # makeshift parameterized test - to avoid introducing new dependencies
        variants = [
            # Self-consistency checks - model should be able to load it's own state
            (ModelType.VANILLA, ModelType.VANILLA),
            (ModelType.SHARDED, ModelType.SHARDED),
            (ModelType.PIPELINED, ModelType.PIPELINED),
            # Production case - saved from pipelined, restored to sharded
            (ModelType.PIPELINED, ModelType.SHARDED),
            # Nice-to-haves:
            (ModelType.SHARDED, ModelType.PIPELINED),
            (ModelType.VANILLA, ModelType.PIPELINED),
            (ModelType.VANILLA, ModelType.SHARDED),
            # Won't work - restoring sharded/pipelined into vanilla fails with
            # "'Parameter' object has no attribute 'local_shards'"
            # ... which is totally expected, as vanilla model is not sharded
            # (ModelType.SHARDED, ModelType.VANILLA),
            # (ModelType.PIPELINED, ModelType.VANILLA),
        ]
        for source_model_type, recipient_model_type in variants:
            self._test_restore_from_snapshot(source_model_type, recipient_model_type)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_rewrite_model_with_fx_wrap(self) -> None:
        sharding_type = ShardingType.TABLE_WISE.value
        kernel_type = EmbeddingComputeKernel.FUSED.value
        fused_params = {}

        class TestPostProcModule(torch.nn.Module):
            def __init__(self, f: float):
                super().__init__()
                self.f = f

            def forward(self, x: KeyedJaggedTensor) -> KeyedJaggedTensor:
                return enrich_hstu_features(x, self.f)

        postproc_module = TestPostProcModule(0.3)

        class TestModel(TestSparseNN):
            use_postproc_module: bool = False

            def forward(
                self,
                input: ModelInput,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                if (type(self)).use_postproc_module:
                    input = self.postproc_module(input)
                else:
                    input = enrich_hstu_features(input, 0.3)
                return self.dense_forward(input, self.sparse_forward(input))

        model = TestModel(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
            postproc_module=postproc_module,
        )

        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, fused_params
        )

        # Try to rewrite model using a function for postproc
        # EBC forwards not overwritten to PipelinedForward due to KJT modification
        self.assertFalse(model.use_postproc_module)
        _rewrite_model(
            model=sharded_model,
            batch=None,
            context=TrainPipelineContext(),
            dist_stream=None,
            pipeline_postproc=True,
        )
        self.assertNotIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.ebc.forward,
            PipelinedForward,
        )
        self.assertNotIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.weighted_ebc.forward,
            PipelinedForward,
        )

        # Now use postproc module
        TestModel.use_postproc_module = True
        self.assertTrue(model.use_postproc_module)
        _rewrite_model(
            model=sharded_model,
            batch=None,
            context=TrainPipelineContext(),
            dist_stream=None,
            pipeline_postproc=True,
        )

        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `sparse`.
        self.assertIsInstance(sharded_model.module.sparse.ebc.forward, PipelinedForward)
        self.assertIsInstance(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.weighted_ebc.forward,
            PipelinedForward,
        )
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `postproc_module`.
            sharded_model.module.postproc_module,
        )
        self.assertEqual(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            sharded_model.module.sparse.weighted_ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `postproc_module`.
            sharded_model.module.postproc_module,
        )
        state_dict = sharded_model.state_dict()
        missing_keys, unexpected_keys = sharded_model.load_state_dict(state_dict)
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])
