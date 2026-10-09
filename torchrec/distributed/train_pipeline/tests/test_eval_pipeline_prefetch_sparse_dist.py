#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for EvalPipelinePrefetchSparseDist.

Covers the eval-only prefetch pipeline:
- queue/context bookkeeping through fill_pipeline and progress
- tail draining (every batch must be evaluated -- eval metrics depend on it)
- no gradients are produced
- output parity against non-pipelined eval
"""

import threading
import unittest
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import patch

import torch
from hypothesis import given, settings, strategies as st
from torch import nn
from torch.optim import Optimizer
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.experimental_pipelines import (
    EvalPipelinePrefetchSparseDist,
    EvalPipelineStage,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    PrefetchTrainPipelineContext,
)
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import DataType


class EvalPipelinePrefetchTestBase(TrainPipelineSparseDistTestBase):
    """Shared setup for EvalPipelinePrefetchSparseDist tests."""

    DEFAULT_FUSED_PARAMS: Dict[str, Any] = {
        "cache_load_factor": 0.5,
        "cache_precision": DataType.FP32,
        "stochastic_rounding": False,
        "prefetch_pipeline": True,
    }

    def setUp(self) -> None:
        feature_gate_patcher = patch(
            "torch.ops.fbgemm.check_feature_gate_key", return_value=False
        )
        feature_gate_patcher.start()
        self.addCleanup(feature_gate_patcher.stop)
        with patch(
            "torch.distributed.distributed_c10d."
            "set_pytorch_distributed_envs_from_justknobs"
        ):
            super().setUp()

    def _create_pipeline(
        self,
        num_batches: int = 5,
        batch_size: int = 32,
        fused_params: Optional[Dict[str, Any]] = None,
        sharding_type: str = ShardingType.TABLE_WISE.value,
        kernel_type: str = EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        stage_hooks: Optional[Dict[str, str]] = None,
        async_prefetch: bool = False,
    ) -> Tuple[
        EvalPipelinePrefetchSparseDist,
        Iterator[ModelInput],
        nn.Module,
        Optimizer,
    ]:
        """
        Creates an eval prefetch pipeline with all necessary components.

        Returns:
            Tuple of (pipeline, dataloader, sharded_model, optimizer)
        """
        self._set_table_weights_precision(DataType.FP32)
        data = self._generate_data(num_batches=num_batches, batch_size=batch_size)
        dataloader = iter(data)

        params = fused_params if fused_params is not None else self.DEFAULT_FUSED_PARAMS

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, params
        )
        # NOTE: TestSparseNN returns a bare prediction tensor (not the
        # `(losses, output)` tuple every pipeline expects) once `.eval()` is
        # called, so it stays in training mode here. The pipeline suppresses
        # grad itself, which is what these tests assert.

        pipeline = EvalPipelinePrefetchSparseDist(
            model=sharded_model,
            optimizer=optim,
            device=self.device,
            stage_hooks=stage_hooks,
            async_prefetch=async_prefetch,
        )

        return pipeline, dataloader, sharded_model, optim

    def _hook_target(
        self, pipeline: EvalPipelinePrefetchSparseDist, fqn: str
    ) -> nn.Module:
        """Resolve a hook FQN the way ``_try_hook_stages`` does."""
        model = pipeline._model
        if isinstance(model, DistributedModelParallel):
            model = model.module
        return model.get_submodule(fqn)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Not enough GPUs, this test requires at least one GPU",
)
class EvalPipelinePrefetchTest(EvalPipelinePrefetchTestBase):
    """Tests for EvalPipelinePrefetchSparseDist API and behavior."""

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_fill_pipeline_initializes_batches_and_contexts(self) -> None:
        """fill_pipeline primes exactly 2 batches with prefetch contexts."""
        pipeline, dataloader, _, _ = self._create_pipeline()

        pipeline.fill_pipeline(dataloader)

        self.assertEqual(len(pipeline.batches), 2)
        self.assertEqual(len(pipeline.contexts), 2)
        self.assertIsInstance(pipeline.contexts[0], PrefetchTrainPipelineContext)
        self.assertIsInstance(pipeline.contexts[1], PrefetchTrainPipelineContext)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_fill_pipeline_is_idempotent(self) -> None:
        """Repeated fill_pipeline calls do not enqueue extra batches."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=10)

        pipeline.fill_pipeline(dataloader)
        initial_batch_count = len(pipeline.batches)
        pipeline.fill_pipeline(dataloader)

        self.assertEqual(len(pipeline.batches), initial_batch_count)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_prefetch_stream_initialization(self) -> None:
        """The prefetch and default streams are created on a CUDA device."""
        pipeline, _, _, _ = self._create_pipeline()

        self.assertIsNotNone(pipeline._prefetch_stream)
        self.assertIsNotNone(pipeline._default_stream)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_batch_queue_maintains_sync_with_context_queue(self) -> None:
        """Batch and context queues stay the same length across iterations."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=10)

        for _ in range(7):
            _ = pipeline.progress(dataloader)
            self.assertLessEqual(len(pipeline.batches), 3)
            self.assertEqual(
                len(pipeline.batches),
                len(pipeline.contexts),
                "Batch and context queues must have the same length",
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_progress_raises_stop_iteration_when_empty(self) -> None:
        """progress raises StopIteration on an empty dataloader."""
        pipeline, _, _, _ = self._create_pipeline(num_batches=0)
        empty_dataloader: Iterator[ModelInput] = iter([])

        with self.assertRaises(StopIteration):
            pipeline.progress(empty_dataloader)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_drains_every_batch(self) -> None:
        """
        Eval must yield one output per batch, including the ones left in flight
        when the dataloader ends -- dropping them would corrupt eval metrics.
        """
        num_batches = 7
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=num_batches)

        outputs = []
        try:
            for _ in range(num_batches + 5):
                outputs.append(pipeline.progress(dataloader))
        except StopIteration:
            pass

        self.assertEqual(len(outputs), num_batches)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_single_batch_execution(self) -> None:
        """A dataloader with one batch still produces one output."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=1)

        output = pipeline.progress(dataloader)

        self.assertIsNotNone(output)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_output_does_not_require_grad(self) -> None:
        """
        The forward runs under no_grad, so no autograd graph is built -- even
        though the module itself is still in training mode.
        """
        pipeline, dataloader, sharded_model, _ = self._create_pipeline(num_batches=4)

        output = pipeline.progress(dataloader)

        self.assertFalse(output.requires_grad)
        for param in sharded_model.parameters():
            self.assertIsNone(param.grad)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_stage_installs_forward_hook(self) -> None:
        """hook_stage relocates a stage onto the named module's forward."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=6)
        pipeline.hook_stage(EvalPipelineStage.WAIT_SPARSE_DATA_DIST, "dense")
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "dense")

        self.assertTrue(pipeline._stage_is_hooked(EvalPipelineStage.PREFETCH))
        self.assertFalse(
            pipeline._stage_is_hooked(EvalPipelineStage.START_SPARSE_DATA_DIST)
        )

        pipeline.fill_pipeline(dataloader)
        self.assertEqual(len(pipeline._stage_hook_handles), 2)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_detach_removes_stage_hooks(self) -> None:
        """
        detach() must take the stage hooks off the model it hands back.

        A surviving hook fires on a standalone forward of the detached model and
        runs batch i+1's sparse work against whatever contexts the pipeline still
        holds. It also closes over the pipeline, so the model would pin the
        in-flight batches detach() deliberately keeps.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6,
            stage_hooks={"wait_sparse_data_dist": "dense", "prefetch": "dense"},
        )
        pipeline.progress(dataloader)

        dense = self._hook_target(pipeline, "dense")
        self.assertEqual(len(pipeline._stage_hook_handles), 2)
        hooks_before = len(dense._forward_hooks)

        pipeline.detach()

        self.assertEqual(pipeline._stage_hook_handles, [])
        self.assertEqual(len(dense._forward_hooks), hooks_before - 2)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_attach_reinstalls_stage_hooks(self) -> None:
        """
        attach() must put back what detach() removed, and the stages must fire.

        fill_pipeline is the usual install site but returns early while batches
        are in flight -- the exact state detach() preserves -- so without an
        explicit reinstall the hooked stages would be skipped in both their
        default slot and their hook, and never run at all.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=8,
            stage_hooks={"wait_sparse_data_dist": "dense", "prefetch": "dense"},
        )
        pipeline.progress(dataloader)

        model = pipeline.detach()
        self.assertEqual(pipeline._stage_hook_handles, [])

        pipeline.attach(model)
        self.assertEqual(len(pipeline._stage_hook_handles), 2)

        pipeline.progress(dataloader)
        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.WAIT_SPARSE_DATA_DIST])
        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.PREFETCH])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_stage_hooks_ctor_arg_matches_hook_stage(self) -> None:
        """The config-driven ctor arg is equivalent to calling hook_stage()."""
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6,
            stage_hooks={"wait_sparse_data_dist": "dense", "prefetch": "dense"},
        )

        self.assertTrue(pipeline._stage_is_hooked(EvalPipelineStage.PREFETCH))
        self.assertTrue(
            pipeline._stage_is_hooked(EvalPipelineStage.WAIT_SPARSE_DATA_DIST)
        )

        pipeline.progress(dataloader)
        pipeline.progress(dataloader)

        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.PREFETCH])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_stage_hooks_install_in_chain_order(self) -> None:
        """
        Hooks on one module fire in dependency order, not caller order.

        PyTorch fires forward hooks in registration order, so installing in dict
        order would let a yaml's key order decide whether the chain holds --
        listing prefetch first would trip the prerequisite check.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6,
            stage_hooks={
                "prefetch": "dense",
                "wait_sparse_data_dist": "dense",
            },
        )

        pipeline.progress(dataloader)
        pipeline.progress(dataloader)

        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.PREFETCH])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_stage_hooks_ctor_arg_rejects_unknown_stage(self) -> None:
        """A typo'd stage name fails at construction, naming the valid stages."""
        with self.assertRaisesRegex(ValueError, "Unknown pipeline stage"):
            self._create_pipeline(num_batches=2, stage_hooks={"prefetchh": "dense"})

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_stage_runs_during_forward(self) -> None:
        """The hooked stage fires from inside the dense forward, not after it."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=6)
        pipeline.hook_stage(EvalPipelineStage.WAIT_SPARSE_DATA_DIST, "dense")
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "dense")

        pipeline.progress(dataloader)
        pipeline.progress(dataloader)

        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.PREFETCH])
        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.WAIT_SPARSE_DATA_DIST])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_stage_rejected_after_pipeline_started(self) -> None:
        """Hooks are installed during model surgery, so they cannot be added later."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=4)
        pipeline.progress(dataloader)

        with self.assertRaisesRegex(RuntimeError, "before the first progress"):
            pipeline.hook_stage(EvalPipelineStage.PREFETCH, "dense")

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_stage_unknown_fqn_raises(self) -> None:
        """A bad FQN fails loudly at install time rather than silently no-op'ing."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=4)
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "no_such_module")

        with self.assertRaises(AttributeError):
            pipeline.fill_pipeline(dataloader)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_on_pipelined_sparse_module_raises(self) -> None:
        """
        A pipelined sparse module is a valid FQN but an invalid hook site.

        Its forward is rewritten to return an awaitable, and driving the next
        batch's sparse work from inside it would mutate the cache and contexts
        the current batch is still using.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=4)
        pipeline.hook_stage(EvalPipelineStage.WAIT_SPARSE_DATA_DIST, "sparse.ebc")
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "sparse.ebc")

        with self.assertRaisesRegex(ValueError, "pipelined sparse module"):
            pipeline.fill_pipeline(dataloader)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_firing_before_sparse_modules_raises(self) -> None:
        """
        A hook that fires while batch i's sparse modules are still pending raises.

        After fill_pipeline the prefetch has populated one
        module_input_post_prefetch entry per pipelined module and no forward has
        popped them, which is exactly the state a too-early hook site would
        observe. TestSparseNN has no module that runs before its sparse arch, so
        the check is driven directly rather than through a contrived model.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=4)
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "dense")
        pipeline.fill_pipeline(dataloader)

        context_0 = pipeline.contexts[0]
        assert isinstance(context_0, PrefetchTrainPipelineContext)
        self.assertTrue(
            context_0.module_input_post_prefetch,
            "fill_pipeline should leave batch 0's prefetch unconsumed",
        )
        # Satisfy the prerequisite check so the too-early check is what runs.
        pipeline._stage_ran[EvalPipelineStage.WAIT_SPARSE_DATA_DIST] = True

        with self.assertRaisesRegex(RuntimeError, "runs before the sparse modules"):
            pipeline._run_stage(EvalPipelineStage.PREFETCH)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_on_sparse_arch_parent_is_allowed(self) -> None:
        """
        Hooking the module that *contains* the sparse modules is legal.

        `sparse` wraps the pipelined EBCs, so its forward hook fires after they
        have run -- the check must not reject it.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=4)
        pipeline.hook_stage(EvalPipelineStage.WAIT_SPARSE_DATA_DIST, "sparse")
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "sparse")

        pipeline.progress(dataloader)

        self.assertTrue(pipeline._stage_ran[EvalPipelineStage.PREFETCH])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_stage_out_of_order_raises(self) -> None:
        """
        Hooking PREFETCH but not its WAIT_SPARSE_DATA_DIST producer makes the
        prefetch run first, which must fail loudly rather than read stale state.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=6)
        pipeline.hook_stage(EvalPipelineStage.PREFETCH, "dense")

        with self.assertRaisesRegex(RuntimeError, "prerequisite"):
            pipeline.progress(dataloader)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_hook_stage_preserves_output(self) -> None:
        """Relocating stages must not change predictions."""
        self._assert_matches_non_pipelined(
            cache_precision=DataType.FP32,
            load_factor=0.5,
            sharding_type=ShardingType.TABLE_WISE.value,
            hooked_stages=[
                EvalPipelineStage.WAIT_SPARSE_DATA_DIST,
                EvalPipelineStage.PREFETCH,
            ],
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_prefetch_allocates_worker_only_when_enabled(self) -> None:
        """The worker is opt-in; the default path stays single-threaded."""
        pipeline, _, _, _ = self._create_pipeline(num_batches=4)
        self.assertIsNone(pipeline._prefetch_executor)

        async_pipeline, _, _, _ = self._create_pipeline(
            num_batches=4, async_prefetch=True
        )
        self.assertIsNotNone(async_pipeline._prefetch_executor)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_prefetch_runs_off_the_main_thread(self) -> None:
        """
        The point of the flag is that the stage leaves the forward's thread.
        ``fill_pipeline`` still prefetches inline, so only the steady-state
        calls are expected off-thread.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6, async_prefetch=True
        )
        original = pipeline._prefetch
        idents: List[int] = []

        def _record(context: PrefetchTrainPipelineContext) -> None:
            idents.append(threading.get_ident())
            original(context)

        pipeline._prefetch = _record

        for _ in range(3):
            pipeline.progress(dataloader)

        self.assertTrue(idents)
        self.assertTrue(
            any(ident != threading.get_ident() for ident in idents),
            "no prefetch ran on the worker thread",
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_prefetch_is_joined_before_the_forward(self) -> None:
        """
        The forward reads the cache lines the worker fills, so no prefetch may
        still be in flight once it starts.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6, async_prefetch=True
        )
        original_fwd = pipeline._model_fwd
        pending_at_forward: List[bool] = []

        def _record(batch: Optional[ModelInput]) -> Tuple[torch.Tensor, Any]:
            pending_at_forward.append(pipeline._prefetch_future is not None)
            return original_fwd(batch)

        pipeline._model_fwd = _record

        for _ in range(3):
            pipeline.progress(dataloader)

        self.assertEqual(pending_at_forward, [False, False, False])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_prefetch_propagates_worker_failure(self) -> None:
        """
        A worker that dies must surface on the main thread at the join, not
        strand the pipeline with a silently unprefetched batch.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6, async_prefetch=True
        )
        # Prime inline so the failure below can only come from the worker.
        pipeline.fill_pipeline(dataloader)

        def _boom(context: PrefetchTrainPipelineContext) -> None:
            raise RuntimeError("prefetch worker exploded")

        pipeline._prefetch = _boom

        pipeline.progress(dataloader)
        with self.assertRaisesRegex(RuntimeError, "prefetch worker exploded"):
            pipeline.progress(dataloader)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_detach_joins_async_prefetch(self) -> None:
        """
        ``detach`` hands the model back to a caller who may run it standalone,
        so no worker may still be mutating the embedding cache behind it.
        """
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=6, async_prefetch=True
        )
        pipeline.progress(dataloader)
        self.assertIsNotNone(pipeline._prefetch_future)

        pipeline.detach()

        self.assertIsNone(pipeline._prefetch_future)
        self.assertIsNone(pipeline._prefetch_executor)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_attach_restarts_async_prefetch_executor(self) -> None:
        """An async pipeline can resume after detach without leaking its worker."""
        pipeline, _, _, _ = self._create_pipeline(num_batches=6, async_prefetch=True)

        model = pipeline.detach()
        self.assertIsNone(pipeline._prefetch_executor)

        pipeline.attach(model)

        self.assertIsNotNone(pipeline._prefetch_executor)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_prefetch_drains_every_batch(self) -> None:
        """Backgrounding the stage must not lose the in-flight tail batches."""
        num_batches = 7
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=num_batches, async_prefetch=True
        )

        outputs = []
        try:
            for _ in range(num_batches + 5):
                outputs.append(pipeline.progress(dataloader))
        except StopIteration:
            pass

        self.assertEqual(len(outputs), num_batches)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_async_prefetch_correctness(self) -> None:
        """
        Moving the stage to a worker must not change the predictions, in the
        default slot or hooked mid-forward.
        """
        for hooked_stages in (
            None,
            [
                EvalPipelineStage.WAIT_SPARSE_DATA_DIST,
                EvalPipelineStage.PREFETCH,
            ],
        ):
            with self.subTest(hooked_stages=hooked_stages):
                self._assert_matches_non_pipelined(
                    cache_precision=DataType.FP32,
                    load_factor=0.5,
                    sharding_type=ShardingType.TABLE_WISE.value,
                    hooked_stages=hooked_stages,
                    async_prefetch=True,
                )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @settings(max_examples=6, deadline=None)
    @given(
        cache_precision=st.sampled_from([DataType.FP16, DataType.FP32]),
        load_factor=st.sampled_from([0.2, 0.6]),
        sharding_type=st.sampled_from(
            [ShardingType.TABLE_WISE.value, ShardingType.ROW_WISE.value]
        ),
    )
    def test_eval_prefetch_pipeline_correctness(
        self,
        cache_precision: DataType,
        load_factor: float,
        sharding_type: str,
    ) -> None:
        """
        Pipelined eval produces the same predictions as non-pipelined eval with
        the FUSED_UVM_CACHING kernel, across the precision/sharding matrix.
        """
        self._assert_matches_non_pipelined(
            cache_precision=cache_precision,
            load_factor=load_factor,
            sharding_type=sharding_type,
        )

    def _assert_matches_non_pipelined(
        self,
        cache_precision: DataType,
        load_factor: float,
        sharding_type: str,
        hooked_stages: Optional[List[EvalPipelineStage]] = None,
        async_prefetch: bool = False,
    ) -> None:
        """
        Runs the same batches through a non-pipelined model and the pipeline and
        asserts the predictions match. Weights never change during eval, so any
        divergence is a pipelining bug rather than drift.

        ``hooked_stages`` relocates those stages onto the dense arch's forward
        hook, so the same parity check covers the reordered schedule.
        """
        self._set_table_weights_precision(DataType.FP32)
        data = self._generate_data(num_batches=12, batch_size=32)
        dataloader = iter(data)

        fused_params = {
            "cache_load_factor": load_factor,
            "cache_precision": cache_precision,
            "stochastic_rounding": False,
        }
        fused_params_pipelined = {**fused_params, "prefetch_pipeline": True}

        model = self._setup_model()
        sharded_model, _ = self._generate_sharded_model_and_optimizer(
            model,
            sharding_type,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            fused_params,
        )
        sharded_model_pipelined, optim_pipelined = (
            self._generate_sharded_model_and_optimizer(
                model,
                sharding_type,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                fused_params_pipelined,
            )
        )
        copy_state_dict(
            sharded_model.state_dict(), sharded_model_pipelined.state_dict()
        )

        pipeline = EvalPipelinePrefetchSparseDist(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            async_prefetch=async_prefetch,
        )
        for stage in hooked_stages or []:
            pipeline.hook_stage(stage, "dense")

        for batch in data:
            batch = batch.to(self.device)
            with torch.no_grad():
                _, pred = sharded_model(batch)

            pred_pipeline = pipeline.progress(dataloader)

            torch.testing.assert_close(pred, pred_pipeline)
