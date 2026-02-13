#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.fx._symbolic_trace import is_fx_tracing
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestModelWithPreproc,
    TestModelWithPreprocCollectionArgs,
    TestNegSamplingModule,
    TestPositionWeightedPreprocModule,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.tracing import (
    GetAttrArgInfoStep,
    GetItemArgInfoStep,
    NoopArgInfoStep,
    PostprocArgInfoStep,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.train_pipeline.types import CallArgs
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class RewriteModelWithPostProcTest(unittest.TestCase):
    pass


class TrainPipelinePostprocTest(TrainPipelineSparseDistTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.num_batches = 10
        self.batch_size = 32
        self.sharding_type = ShardingType.TABLE_WISE.value
        self.kernel_type = EmbeddingComputeKernel.FUSED.value
        self.fused_params = {}

        self.extra_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="extra_table_" + str(i),
                feature_names=["extra_feature_" + str(i)],
            )
            for i in range(3)
        ]
        self.extra_weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="extra_weighted_table_" + str(i),
                feature_names=["extra_weighted_feature_" + str(i)],
            )
            for i in range(3)
        ]

    def _assert_output_equal(
        self,
        model: nn.Module,
        sharding_type: str,
        max_feature_lengths: Optional[List[int]] = None,
    ) -> Tuple[nn.Module, TrainPipelineSparseDist[ModelInput, torch.Tensor]]:
        data = self._generate_data(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            max_feature_lengths=max_feature_lengths,
        )
        dataloader = iter(data)

        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, self.kernel_type, self.fused_params
        )

        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, sharding_type, self.kernel_type, self.fused_params
        )
        copy_state_dict(
            sharded_model.state_dict(), sharded_model_pipelined.state_dict()
        )

        pipeline = self.pipeline_class(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            pipeline_postproc=True,
        )

        for i in range(self.num_batches):
            batch = data[i]
            # Forward + backward w/o pipelining
            batch = batch.to(self.device)
            optim.zero_grad()
            loss, pred = sharded_model(batch)
            loss.backward()
            optim.step()

            pred_pipelined = pipeline.progress(dataloader)
            self.assertTrue(torch.equal(pred, pred_pipelined))

        return sharded_model_pipelined, pipeline

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_modules_share_postproc(self) -> None:
        """
        Setup: postproc module takes in input batch and returns modified
        input batch. EBC and weighted EBC inside model sparse arch subsequently
        uses this modified KJT.

        Test case where single postproc module is shared by multiple sharded modules
        and output of postproc module needs to be transformed in the SAME way
        """
        extra_input = ModelInput.generate(
            tables=self.extra_tables,
            weighted_tables=self.extra_weighted_tables,
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=10,
            randomize_indices=False,
        )[0].to(self.device)

        postproc_module = TestNegSamplingModule(
            extra_input=extra_input,
        )
        model = self._setup_model(postproc_module=postproc_module)

        _, pipeline = self._assert_output_equal(
            model,
            self.sharding_type,
        )

        # Check that both EC and EBC pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 2)
        self.assertEqual(len(pipeline._pipelined_postprocs), 1)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_postproc_not_shared_with_arg_transform(self) -> None:
        """
        Test case where arguments to postproc module is some non-modifying
        transformation of the input batch (no nested postproc modules) AND
        arguments to multiple sharded modules can be derived from the output
        of different postproc modules (i.e. postproc modules not shared).
        """
        model = TestModelWithPreproc(
            tables=self.tables[:-1],  # ignore last table as postproc will remove
            weighted_tables=self.weighted_tables[:-1],  # ignore last table
            device=self.device,
        )

        pipelined_model, pipeline = self._assert_output_equal(
            model,
            self.sharding_type,
        )

        # Check that both EBC and weighted EBC pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 2)

        pipelined_ebc = pipeline._pipelined_modules[0]
        pipelined_weighted_ebc = pipeline._pipelined_modules[1]

        # Check pipelined args
        self.assertEqual(len(pipelined_ebc.forward._args.args), 1)
        self.assertEqual(len(pipelined_ebc.forward._args.kwargs), 0)
        self.assertEqual(
            pipelined_ebc.forward._args.args[0].steps,
            [
                PostprocArgInfoStep(pipelined_model.module.postproc_nonweighted),
                GetItemArgInfoStep(0),
            ],
        )
        self.assertEqual(len(pipelined_weighted_ebc.forward._args.args), 1)
        self.assertEqual(len(pipelined_weighted_ebc.forward._args.kwargs), 0)
        self.assertEqual(
            pipelined_weighted_ebc.forward._args.args[0].steps,
            [
                PostprocArgInfoStep(pipelined_model.module.postproc_weighted),
                GetItemArgInfoStep(0),
            ],
        )

        # postproc args
        self.assertEqual(len(pipeline._pipelined_postprocs), 2)
        # postprocs can be added in any order, so we can't assert on exact steps structures
        # TODO: find way not to inspect private parts
        postproc1_args: CallArgs = pipeline._pipelined_postprocs[0]._args
        self.assertEqual(len(postproc1_args.args), 1)
        self.assertEqual(len(postproc1_args.kwargs), 0)
        self.assertEqual(len(postproc1_args.args[0].steps), 2)
        self.assertEqual(postproc1_args.args[0].steps[0], NoopArgInfoStep())
        self.assertIsInstance(postproc1_args.args[0].steps[1], GetAttrArgInfoStep)

        postproc2_args: CallArgs = pipeline._pipelined_postprocs[1]._args
        self.assertEqual(len(postproc2_args.args), 1)
        self.assertEqual(len(postproc2_args.kwargs), 0)
        self.assertEqual(len(postproc2_args.args[0].steps), 2)
        self.assertEqual(postproc2_args.args[0].steps[0], NoopArgInfoStep())
        self.assertIsInstance(postproc2_args.args[0].steps[1], GetAttrArgInfoStep)

        get_arg_infos = {
            postproc._args.args[0].steps[1].attr_name
            for postproc in pipeline._pipelined_postprocs
        }
        self.assertEqual(get_arg_infos, {"idlist_features", "idscore_features"})

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_postproc_recursive(self) -> None:
        """
        Test recursive case where multiple arguments to postproc module is derived
        from output of another postproc module. For example,

        out_a, out_b, out_c = postproc_1(input)
        out_d = postproc_2(out_a, out_b)
        # do something with out_c
        out = ebc(out_d)
        """
        extra_input = ModelInput.generate(
            tables=self.extra_tables[:-1],
            weighted_tables=self.extra_weighted_tables[:-1],
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=10,
            randomize_indices=False,
        )[0].to(self.device)

        postproc_module = TestNegSamplingModule(
            extra_input=extra_input,
        )

        model = TestModelWithPreproc(
            tables=self.tables[:-1],
            weighted_tables=self.weighted_tables[:-1],
            device=self.device,
            postproc_module=postproc_module,
        )

        pipelined_model, pipeline = self._assert_output_equal(model, self.sharding_type)

        # Check that both EBC and weighted EBC pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 2)

        pipelined_ebc = pipeline._pipelined_modules[0]
        pipelined_weighted_ebc = pipeline._pipelined_modules[1]

        # Check pipelined args
        self.assertEqual(len(pipelined_ebc.forward._args.args), 1)
        self.assertEqual(len(pipelined_ebc.forward._args.kwargs), 0)
        self.assertEqual(
            pipelined_ebc.forward._args.args[0].steps,
            [
                PostprocArgInfoStep(pipelined_model.module.postproc_nonweighted),
                GetItemArgInfoStep(0),
            ],
        )
        self.assertEqual(len(pipelined_weighted_ebc.forward._args.args), 1)
        self.assertEqual(len(pipelined_weighted_ebc.forward._args.kwargs), 0)
        self.assertEqual(
            pipelined_weighted_ebc.forward._args.args[0].steps,
            [
                PostprocArgInfoStep(pipelined_model.module.postproc_weighted),
                GetItemArgInfoStep(0),
            ],
        )

        # postproc args
        self.assertEqual(len(pipeline._pipelined_postprocs), 3)

        parent_postproc_mod = pipelined_model.module._postproc_module

        for postproc_mod in pipeline._pipelined_postprocs:
            #  `postproc_nonweighted`.
            if postproc_mod == pipelined_model.module.postproc_nonweighted:
                self.assertEqual(len(postproc_mod._args.args), 1)
                self.assertEqual(len(postproc_mod._args.kwargs), 0)
                self.assertEqual(
                    postproc_mod._args.args[0].steps,
                    [
                        PostprocArgInfoStep(parent_postproc_mod),
                        GetAttrArgInfoStep("idlist_features"),
                    ],
                )

            #  `postproc_weighted`.
            elif postproc_mod == pipelined_model.module.postproc_weighted:
                self.assertEqual(len(postproc_mod._args.args), 1)
                self.assertEqual(len(postproc_mod._args.kwargs), 0)
                self.assertEqual(
                    postproc_mod._args.args[0].steps,
                    [
                        PostprocArgInfoStep(parent_postproc_mod),
                        GetAttrArgInfoStep("idscore_features"),
                    ],
                )
            elif postproc_mod == parent_postproc_mod:
                self.assertEqual(len(postproc_mod._args.args), 1)
                self.assertEqual(len(postproc_mod._args.kwargs), 0)
                self.assertEqual(postproc_mod._args.args[0].steps, [NoopArgInfoStep()])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_invalid_postproc_inputs_has_trainable_params(self) -> None:
        """
        Test case where postproc module sits in front of sharded module but this cannot be
        safely pipelined as it contains trainable params in its child modules
        """
        max_feature_lengths = {
            "feature_0": 10,
            "feature_1": 10,
            "feature_2": 10,
            "feature_3": 10,
        }

        postproc_module = TestPositionWeightedPreprocModule(
            max_feature_lengths=max_feature_lengths,
            device=self.device,
        )

        model = self._setup_model(postproc_module=postproc_module)

        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, self.sharding_type, self.kernel_type, self.fused_params
        )

        pipeline = self.pipeline_class(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            pipeline_postproc=True,
        )

        data = self._generate_data(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            max_feature_lengths=list(max_feature_lengths.values()),
        )
        dataloader = iter(data)

        pipeline.progress(dataloader)

        # Check that no modules are pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 0)
        self.assertEqual(len(pipeline._pipelined_postprocs), 0)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_invalid_postproc_trainable_params_recursive(
        self,
    ) -> None:
        max_feature_lengths = {
            "feature_0": 10,
            "feature_1": 10,
            "feature_2": 10,
            "feature_3": 10,
        }

        postproc_module = TestPositionWeightedPreprocModule(
            max_feature_lengths=max_feature_lengths,
            device=self.device,
        )

        model = TestModelWithPreproc(
            tables=self.tables[:-1],
            weighted_tables=self.weighted_tables[:-1],
            device=self.device,
            postproc_module=postproc_module,
        )

        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, self.sharding_type, self.kernel_type, self.fused_params
        )

        pipeline = self.pipeline_class(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            pipeline_postproc=True,
        )

        data = self._generate_data(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
            max_feature_lengths=list(max_feature_lengths.values()),
        )
        dataloader = iter(data)
        pipeline.progress(dataloader)

        # Check that no modules are pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 0)
        self.assertEqual(len(pipeline._pipelined_postprocs), 0)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_invalid_postproc_inputs_modify_kjt_recursive(self) -> None:
        """
        Test case where postproc module cannot be pipelined because at least one of args
        is derived from output of another postproc module whose arg(s) cannot be derived
        from input batch (i.e. it has modifying transformations)
        """
        model = TestModelWithPreproc(
            tables=self.tables[:-1],
            weighted_tables=self.weighted_tables[:-1],
            device=self.device,
            postproc_module=None,
            run_postproc_inline=True,  # run postproc inline, outside a module
        )

        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, self.sharding_type, self.kernel_type, self.fused_params
        )

        pipeline = self.pipeline_class(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            pipeline_postproc=True,
        )

        data = self._generate_data(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
        )
        dataloader = iter(data)
        pipeline.progress(dataloader)

        # Check that only weighted EBC is pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 1)
        self.assertEqual(len(pipeline._pipelined_postprocs), 1)
        self.assertEqual(pipeline._pipelined_modules[0]._is_weighted, True)
        self.assertEqual(
            pipeline._pipelined_postprocs[0],
            #  `postproc_weighted`.
            sharded_model_pipelined.module.postproc_weighted,
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_postproc_fwd_values_cached(self) -> None:
        """
        Test to check that during model forward, the postproc module pipelined uses the
        saved result from previous iteration(s) and doesn't perform duplicate work
        check that fqns for ALL postproc modules are populated in the right train pipeline
        context.
        """
        extra_input = ModelInput.generate(
            tables=self.extra_tables[:-1],
            weighted_tables=self.extra_weighted_tables[:-1],
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=10,
            randomize_indices=False,
        )[0].to(self.device)

        postproc_module = TestNegSamplingModule(
            extra_input=extra_input,
        )

        model = TestModelWithPreproc(
            tables=self.tables[:-1],
            weighted_tables=self.weighted_tables[:-1],
            device=self.device,
            postproc_module=postproc_module,
        )

        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, self.sharding_type, self.kernel_type, self.fused_params
        )

        pipeline = self.pipeline_class(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            pipeline_postproc=True,
        )

        data = self._generate_data(
            num_batches=self.num_batches,
            batch_size=self.batch_size,
        )
        dataloader = iter(data)

        pipeline.progress(dataloader)

        # This was second context that was appended
        current_context = pipeline.contexts[0]
        cached_results = current_context.postproc_fwd_results
        self.assertEqual(
            list(cached_results.keys()),
            ["_postproc_module", "postproc_nonweighted", "postproc_weighted"],
        )

        # next context cached results should be empty
        next_context = pipeline.contexts[1]
        next_cached_results = next_context.postproc_fwd_results
        self.assertEqual(len(next_cached_results), 0)

        # After progress, next_context should be populated
        pipeline.progress(dataloader)
        self.assertEqual(
            list(next_cached_results.keys()),
            ["_postproc_module", "postproc_nonweighted", "postproc_weighted"],
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_nested_postproc(self) -> None:
        """
        If postproc module is nested, we should still be able to pipeline it
        """
        extra_input = ModelInput.generate(
            tables=self.extra_tables,
            weighted_tables=self.extra_weighted_tables,
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=10,
            randomize_indices=False,
        )[0].to(self.device)

        postproc_module = TestNegSamplingModule(
            extra_input=extra_input,
        )
        model = self._setup_model(postproc_module=postproc_module)

        class ParentModule(nn.Module):
            def __init__(
                self,
                nested_model: nn.Module,
            ) -> None:
                super().__init__()
                self.nested_model = nested_model

            def forward(
                self,
                input: ModelInput,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                return self.nested_model(input)

        model = ParentModule(model)

        pipelined_model, pipeline = self._assert_output_equal(
            model,
            self.sharding_type,
        )

        # Check that both EC and EBC pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 2)
        self.assertEqual(len(pipeline._pipelined_postprocs), 1)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_pipeline_postproc_with_collection_args(self) -> None:
        """
        Exercises scenario when postproc module has an argument that is a list or dict
        with some elements being:
            * static scalars
            * static tensors (e.g. torch.ones())
            * tensors derived from input batch (e.g. input.idlist_features["feature_0"])
            * tensors derived from input batch and other postproc module (e.g. other_postproc(input.idlist_features["feature_0"]))
        """
        test_runner = self

        class PostprocOuter(nn.Module):
            def __init__(
                self,
            ) -> None:
                super().__init__()

            def forward(
                self,
                model_input: ModelInput,
            ) -> torch.Tensor:
                return model_input.float_features * 0.1

        class PostprocInner(nn.Module):
            def __init__(
                self,
            ) -> None:
                super().__init__()

            def forward(
                self,
                model_input: ModelInput,
                input_list: List[Union[torch.Tensor, int]],
                input_dict: Dict[str, Union[torch.Tensor, int]],
            ) -> ModelInput:
                if not is_fx_tracing():
                    for idx, value in enumerate(input_list):
                        if isinstance(value, torch.fx.Node):
                            test_runner.fail(
                                f"input_list[{idx}] was a fx.Node: {value}"
                            )
                        model_input.float_features += value

                    for key, value in input_dict.items():
                        if isinstance(value, torch.fx.Node):
                            test_runner.fail(
                                f"input_dict[{key}] was a fx.Node: {value}"
                            )
                        model_input.float_features += value

                return model_input

        model = TestModelWithPreprocCollectionArgs(
            tables=self.tables[:-1],  # ignore last table as postproc will remove
            weighted_tables=self.weighted_tables[:-1],  # ignore last table
            device=self.device,
            postproc_module_outer=PostprocOuter(),
            postproc_module_nested=PostprocInner(),
        )

        pipelined_model, pipeline = self._assert_output_equal(
            model,
            self.sharding_type,
        )

        # both EC end EBC are pipelined
        self.assertEqual(len(pipeline._pipelined_modules), 2)
        # both outer and nested postproces are pipelined
        self.assertEqual(len(pipeline._pipelined_postprocs), 4)
