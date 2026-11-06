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
from typing import Any, cast, List, NamedTuple, Optional, Protocol, Tuple, Union
from unittest.mock import MagicMock, patch

import torch
from torch._dynamo import is_dynamo_supported
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
)
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestNegSamplingModule,
    TestSparseNN,
)
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    _GAOptimizerWrapper,
    GradientAccumulationConfig,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    EmbeddingTrainPipelineContext,
    PrefetchEmbeddingTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import (
    EmbeddingPipelinedForward,
    InSyncPrefetchEmbeddingPipelinedForward,
    PipelinedForward,
)
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.tracing import CallArgs, PipelinedPostproc
from torchrec.distributed.train_pipeline.utils import (
    _is_data_loading_retriable,
    _rewrite_model,
    _start_embedding_lookup,
    async_optimizer_unsupported_reason,
    DataLoadingThread,
    ddp_managed_param_ids,
    find_ddp_modules,
    start_embedding_lookup_post_prefetch,
    use_context_for_postprocs,
)
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ShardedModule,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.clipping import GradientClippingOptimizer
from torchrec.optim.fused import EmptyFusedOptimizer
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.warmup import WarmupOptimizer, WarmupStage
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable

_UTILS_LOGGER_NAME: str = "torchrec.distributed.train_pipeline.utils"


class ModelType(enum.Enum):
    VANILLA = "vanilla"
    SHARDED = "sharded"
    PIPELINED = "pipelined"


class _HasSparseModule(Protocol):
    sparse: torch.nn.Module


@torch.fx.wrap
def enrich_hstu_features(
    kjt: KeyedJaggedTensor, hstu_factor: float
) -> KeyedJaggedTensor:
    if kjt._weights is not None:
        kjt._weights *= hstu_factor
    return kjt


@torch.fx.has_side_effect
@torch.fx.wrap
def _hidden_ebc_lookup(ebc: torch.nn.Module, kjt: KeyedJaggedTensor) -> KeyedTensor:
    """
    Performs an embedding lookup from inside an fx-wrapped function.

    fx does not trace into a wrapped function: it emits a single `call_function`
    node that carries `ebc` as a constant argument. `_rewrite_model` only scans
    `call_module` nodes, so a sharded module invoked this way is invisible to it.
    """
    return ebc(kjt)


class TrainPipelineUtilsTest(TrainPipelineSparseDistTestBase):
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
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.forward,
            PipelinedForward,
        )
        self.assertNotIsInstance(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
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

        # pyrefly: ignore[missing-attribute]
        self.assertIsInstance(sharded_model.module.sparse.ebc.forward, PipelinedForward)
        self.assertIsInstance(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.weighted_ebc.forward,
            PipelinedForward,
        )
        self.assertEqual(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            #  `postproc_module`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.postproc_module,
        )
        self.assertEqual(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.weighted_ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            #  `postproc_module`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.postproc_module,
        )
        state_dict = sharded_model.state_dict()
        missing_keys, unexpected_keys = sharded_model.load_state_dict(state_dict)
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_rewrite_model_compiled_with_error_on_nested_fx_trace(self) -> None:
        """Test that _rewrite_model can trace a compiled model even when
        error_on_nested_fx_trace is True, thanks to the config.patch override.

        The compiled module must contain ShardedModules so the FX tracer traces
        *into* it (non-leaf), triggering dynamo's eval frame hook which checks
        the error_on_nested_fx_trace config."""
        sharding_type = ShardingType.TABLE_WISE.value
        kernel_type = EmbeddingComputeKernel.FUSED.value
        fused_params = {}

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, fused_params
        )

        # Compile the sparse sub-module which contains ShardedModules (ebc,
        # weighted_ebc).  Because it contains ShardedModules the FX tracer will
        # treat it as non-leaf and trace into it, invoking the OptimizedModule's
        # __call__ which goes through dynamo's eval frame and checks
        # error_on_nested_fx_trace.
        inner = cast(_HasSparseModule, sharded_model.module)
        compiled_sparse = cast(
            torch.nn.Module,
            torch.compile(inner.sparse, backend="eager", fullgraph=False),
        )
        inner.sparse = compiled_sparse

        # Set error_on_nested_fx_trace to True globally — without the fix in
        # _rewrite_model this would cause tracing to raise an error.
        original_value = torch._dynamo.config.error_on_nested_fx_trace
        torch._dynamo.config.error_on_nested_fx_trace = True
        try:
            pipelined_forwards, _, original_forwards, _, _ = _rewrite_model(
                model=sharded_model,
                batch=None,
                context=TrainPipelineContext(),
                dist_stream=None,
            )

            # Verify that sharded modules were successfully pipelined
            self.assertGreater(len(pipelined_forwards), 0)
            for mod in pipelined_forwards:
                self.assertIsInstance(mod.forward, PipelinedForward)
        finally:
            torch._dynamo.config.error_on_nested_fx_trace = original_value

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
        # pyrefly: ignore[bad-assignment]
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
                    # pyrefly: ignore[not-callable]
                    input = self.postproc_module(input)
                else:
                    # pyrefly: ignore[bad-argument-type, bad-assignment]
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
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.forward,
            PipelinedForward,
        )
        self.assertNotIsInstance(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
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

        # pyrefly: ignore[missing-attribute]
        self.assertIsInstance(sharded_model.module.sparse.ebc.forward, PipelinedForward)
        self.assertIsInstance(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.weighted_ebc.forward,
            PipelinedForward,
        )
        self.assertEqual(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            #  `postproc_module`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.postproc_module,
        )
        self.assertEqual(
            #  `sparse`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.weighted_ebc.forward._args.args[0]
            .steps[0]
            .postproc_module,
            #  `postproc_module`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.postproc_module,
        )
        state_dict = sharded_model.state_dict()
        missing_keys, unexpected_keys = sharded_model.load_state_dict(state_dict)
        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_rewrite_model_sharded_module_hidden_in_fx_wrap(self) -> None:
        """A sharded module called from inside an fx-wrapped function is emitted
        as a `call_function` node holding the module as a constant argument, not
        as a `call_module` node. `_rewrite_model` only rewrites `call_module`
        nodes, so such a module is neither pipelined nor reported as
        non-pipelined -- it is silently skipped.

        ARM A calls the EBC through an fx-wrapped function (the production bug).
        ARM B calls the exact same EBC directly with the same KJT (the fix).
        """
        sharding_type = ShardingType.TABLE_WISE.value
        kernel_type = EmbeddingComputeKernel.FUSED.value

        class TestModel(TestSparseNN):
            call_ebc_directly: bool = False

            def forward(
                self,
                input: ModelInput,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                kjt = cast(KeyedJaggedTensor, input.idlist_features)
                # pyre-ignore[29]: `sparse` is a Module attribute, not a Tensor
                # pyrefly: ignore[missing-attribute]
                ebc = cast(torch.nn.Module, self.sparse.ebc)
                if (type(self)).call_ebc_directly:
                    ebc_out = ebc(kjt)
                else:
                    ebc_out = _hidden_ebc_lookup(ebc, kjt)
                # pyrefly: ignore[bad-argument-type]
                return self.dense_forward(input, ebc_out)

        model = TestModel(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
        )
        sharded_model, _optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, {}
        )
        # pyrefly: ignore[missing-attribute]
        ebc: torch.nn.Module = sharded_model.module.sparse.ebc

        # ARM A: lookup hidden behind @torch.fx.wrap -- not pipelined, silently.
        TestModel.call_ebc_directly = False
        with self.assertLogs(_UTILS_LOGGER_NAME, level="INFO") as arm_a_logs:
            pipelined_a, _, _, _, non_pipelined_a = _rewrite_model(
                model=sharded_model,
                batch=None,
                context=TrainPipelineContext(),
                dist_stream=None,
                pipeline_postproc=True,
            )
        arm_a_output = "\n".join(arm_a_logs.output)
        # guards the assertNotIn checks below from passing vacuously
        self.assertIn("pipeline_postproc is enabled", arm_a_output)
        self.assertNotIsInstance(
            ebc.forward,
            PipelinedForward,
            "EBC hidden inside an fx-wrapped function must not be pipelined",
        )
        self.assertEqual(pipelined_a, [])
        # The bug is silent: the module shows up in neither of the two warnings
        # that _rewrite_model emits for modules it refuses to pipeline.
        self.assertNotIn("sparse.ebc", non_pipelined_a)
        self.assertNotIn("will NOT be pipelined", arm_a_output)
        self.assertNotIn("Sharded modules were not pipelined", arm_a_output)
        self.assertNotIn("sparse.ebc", arm_a_output)

        # ARM B: same model, same KJT, module called directly -- pipelined.
        TestModel.call_ebc_directly = True
        with self.assertLogs(_UTILS_LOGGER_NAME, level="INFO") as arm_b_logs:
            pipelined_b, _, _, _, non_pipelined_b = _rewrite_model(
                model=sharded_model,
                batch=None,
                context=TrainPipelineContext(),
                dist_stream=None,
                pipeline_postproc=True,
            )
        arm_b_output = "\n".join(arm_b_logs.output)
        diagnostics = (
            f"non_pipelined_sharded_modules={non_pipelined_b}, logs:\n{arm_b_output}"
        )
        self.assertIsInstance(ebc.forward, PipelinedForward, diagnostics)
        self.assertIn(ebc, pipelined_b, diagnostics)
        self.assertNotIn("sparse.ebc", non_pipelined_b, diagnostics)
        self.assertIn("Module 'sparse.ebc' will be pipelined", arm_b_output)


# ~50ms on a modern GPU: long enough that a consumer on an unordered stream
# reliably reads the input-dist output before it has been written.
_INPUT_DIST_DELAY_CYCLES = 100_000_000

# Fills the KJT before the input dist writes it, so an embedding lookup that runs
# too early reads this instead of the real values.
_PRE_INPUT_DIST_SENTINEL = -1


@enum.unique
class _StreamCase(enum.Enum):
    """Call shapes `_start_embedding_lookup` sees across its callers.

    Named for where the embedding lookup runs relative to the source (data-dist)
    and target streams.
    """

    LOOKUP_ON_NEW = enum.auto()
    LOOKUP_ON_SOURCE = enum.auto()
    LOOKUP_ON_TARGET = enum.auto()
    SOURCE_UNSET = enum.auto()
    NO_STREAMS = enum.auto()


class _DelayedInputDistAwaitable(Awaitable[KeyedJaggedTensor]):
    """Input-dist awaitable that simulates a slow input dist racing the lookup.

    `wait()` enqueues a long GPU delay and then the write on whatever stream is
    current, which is the source (data-dist) stream, and never synchronizes the
    host. An embedding lookup running on an unordered stream therefore reads the
    KJT before it has been written -- the race `_start_embedding_lookup` is
    responsible for preventing.
    """

    def __init__(self, kjt: KeyedJaggedTensor, values: torch.Tensor) -> None:
        super().__init__()
        self._kjt = kjt
        self._values = values

    def _wait_impl(self) -> KeyedJaggedTensor:
        torch.cuda._sleep(_INPUT_DIST_DELAY_CYCLES)
        self._kjt.values().copy_(self._values)
        return self._kjt


class _RecordingModuleContext(Multistreamable):
    """Module context that remembers every stream it was recorded on.

    Makes the `record_stream` calls that register the KJT and the module context
    for caching-allocator lifetime tracking observable. `_start_embedding_lookup`
    records both on the same streams, and a real KJT forwards `record_stream` to
    its tensors without leaving a trace, so the streams collected here stand for
    both calls.
    """

    def __init__(self) -> None:
        self.recorded_streams: List[torch.Stream] = []

    def record_stream(self, stream: torch.Stream) -> None:
        self.recorded_streams.append(stream)


class _StreamProbeModule:
    """Test double for a pipelined `ShardedModule`, replacing the TBE lookup.

    Implements only the surface `_start_embedding_lookup` touches: a `forward`
    carrying the module name, and `compute_and_output_dist`. Instead of an
    embedding lookup it copies the KJT it is handed, on whatever stream is
    current, so the values the embedding-lookup stream actually read can be
    asserted on directly rather than inferred from embedding outputs.
    """

    def __init__(self, observed: torch.Tensor) -> None:
        self.observed = observed
        # Assigned after construction, since the forward refers back to us.
        self.forward: Optional[EmbeddingPipelinedForward] = None

    def compute_and_output_dist(
        self, ctx: Multistreamable, kjt: KeyedJaggedTensor
    ) -> torch.Tensor:
        self.observed.copy_(kjt.values())
        return self.observed


def _pipeline_module(
    name: str, context: EmbeddingTrainPipelineContext, observed: torch.Tensor
) -> _StreamProbeModule:
    """Builds a probe module carrying the forward `_rewrite_model` would install.

    `_rewrite_model` replaces a `ShardedModule`'s bound `forward` with an
    `EmbeddingPipelinedForward`, which is where `_start_embedding_lookup` reads
    the module name from, so the forward has to be attached after construction.
    """
    module = _StreamProbeModule(observed)
    module.forward = EmbeddingPipelinedForward(
        name=name,
        args=CallArgs(args=[], kwargs={}),
        module=cast(ShardedModule, module),
        context=context,
    )
    return module


class _StreamSetup(NamedTuple):
    """Resolved streams for one `_start_embedding_lookup` call shape.

    Attributes:
        source_stream: the `source_stream` argument.
        target_stream: the `target_stream` argument.
        lookup_stream: stream entered before the call, mirroring the caller's
            `stream_context(emb_lookup_stream)`. `None` leaves the default stream
            current, which is what the callers without a stream context do.
        expected_recorded: streams the module context must be recorded on.
    """

    source_stream: Optional[torch.cuda.Stream]
    target_stream: Optional[torch.cuda.Stream]
    lookup_stream: Optional[torch.cuda.Stream]
    expected_recorded: List[torch.cuda.Stream]


def _build_stream_setup(case: _StreamCase, device: torch.device) -> _StreamSetup:
    """Builds the streams for one call shape of `_start_embedding_lookup`."""
    default_stream = torch.cuda.current_stream(device)
    source_stream = torch.cuda.Stream(device=device)
    match case:
        case _StreamCase.NO_STREAMS:
            # No caller reaches this today: target_stream is always
            # `current_stream()`, which is non-None even on CPU. Pins the
            # helper's guard for the case where no device stream is resolvable.
            return _StreamSetup(None, None, None, [])
        case _StreamCase.SOURCE_UNSET:
            # No data-dist stream, as on a CPU device, so the current stream
            # comes from target_stream.
            return _StreamSetup(None, default_stream, None, [default_stream])
        case _StreamCase.LOOKUP_ON_TARGET:
            # emb_lookup_stream="current", and the TrainPipelineSemiSync call shape.
            return _StreamSetup(source_stream, default_stream, None, [default_stream])
        case _StreamCase.LOOKUP_ON_SOURCE:
            # emb_lookup_stream="data_dist", the TrainPipelineFusedSparseDist
            # default. The lookup runs on the stream that allocated the KJT, so
            # only the target stream needs recording.
            return _StreamSetup(
                source_stream,
                default_stream,
                source_stream,
                [default_stream],
            )
        case _StreamCase.LOOKUP_ON_NEW:
            # emb_lookup_stream="new": source, target and lookup are all distinct.
            lookup_stream = torch.cuda.Stream(device=device)
            return _StreamSetup(
                source_stream,
                default_stream,
                lookup_stream,
                [lookup_stream, default_stream],
            )


class StartEmbeddingLookupTest(unittest.TestCase):
    """Tests the stream contract of `_start_embedding_lookup`.

    Steps the helper performs:
      1. Waits for the input-dist output on `source_stream`.
      2. Runs `compute_and_output_dist` on the current stream, which callers set
         to their embedding-lookup stream.

    Expected behavior when those two streams differ:
      - The current stream waits on an event recorded on `source_stream`. CUDA
        orders nothing between two streams on its own, so without that edge the
        lookup reads input-dist output that has not been written yet.
      - The KJT and module context are recorded on the current stream, so the
        caching allocator cannot reuse their memory while the lookup reads it.
      - A stream serving two of these roles at once is recorded only once.
    """

    def _run_start_embedding_lookup(
        self, setup: _StreamSetup, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, _RecordingModuleContext]:
        """Runs `_start_embedding_lookup` once for the given stream setup.

        Returns:
            The values the input dist writes, the values the embedding lookup
            actually read, and the module context that recorded the streams.
        """
        name = "sharded_ebc"
        num_values = 16
        expected = torch.arange(num_values, dtype=torch.int64, device=device)
        kjt = KeyedJaggedTensor(
            keys=["feature_0"],
            values=torch.full_like(expected, _PRE_INPUT_DIST_SENTINEL),
            lengths=torch.ones(num_values, dtype=torch.int64, device=device),
        )
        observed = torch.full_like(expected, _PRE_INPUT_DIST_SENTINEL)
        module_context = _RecordingModuleContext()
        context = EmbeddingTrainPipelineContext(
            input_dist_tensors_requests={
                name: _DelayedInputDistAwaitable(kjt, expected)
            },
            module_contexts={name: module_context},
        )
        # The fixtures above are filled on the default stream but read below on
        # source_stream and the lookup stream. PyTorch creates streams with
        # cudaStreamNonBlocking, so they are not implicitly ordered against the
        # default stream; drain it so a stale fixture read cannot masquerade as
        # the race under test.
        torch.cuda.synchronize(device)

        with torch.cuda.stream(setup.lookup_stream):
            _start_embedding_lookup(
                module=cast(ShardedModule, _pipeline_module(name, context, observed)),
                context=context,
                source_stream=setup.source_stream,
                target_stream=setup.target_stream,
                stream_context=torch.cuda.stream,
            )
        # `observed` is written on the lookup stream; wait for it before reading.
        torch.cuda.synchronize(device)
        return expected, observed, module_context

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @patch("torch._utils_internal.justknobs_check", return_value=True)
    def test_stream_combinations(self, _mock_justknobs_check: MagicMock) -> None:
        """
        Tests every stream combination `_start_embedding_lookup` is called with,
        asserting for each that the lookup read the fully written input-dist
        output and that the module context recorded exactly the expected streams.
        """
        device = torch.device("cuda:0")
        for case in _StreamCase:
            with self.subTest(streams=case.name):
                setup = _build_stream_setup(case, device)
                expected, observed, module_context = self._run_start_embedding_lookup(
                    setup, device
                )
                torch.testing.assert_close(observed, expected)
                self.assertCountEqual(
                    module_context.recorded_streams, setup.expected_recorded
                )


class _PostPrefetchProbeModule:
    """Test double for a pipelined `ShardedModule`, replacing the TBE lookup.

    Implements only the surface `start_embedding_lookup_post_prefetch` touches: a
    `forward` carrying the module name, and a `compute_and_output_dist` that records
    what it was handed instead of running a lookup.
    """

    def __init__(self) -> None:
        self.awaitable = MagicMock()
        self.calls: List[Tuple[object, object]] = []
        # Assigned after construction, since the forward refers back to us.
        self.forward: Optional[InSyncPrefetchEmbeddingPipelinedForward] = None

    def compute_and_output_dist(self, ctx: object, data: object) -> MagicMock:
        self.calls.append((ctx, data))
        return self.awaitable


class StartEmbeddingLookupPostPrefetchTest(unittest.TestCase):
    """Tests the handoff contract of `start_embedding_lookup_post_prefetch`.

    This is the lookup stage of `PrefetchTrainPipelineFusedSparseDist`. Unlike
    `_start_embedding_lookup` it has no input-dist awaitable to wait on and no
    cross-stream race to order, because the prefetch stage already resolved the
    input dist and parked its output in the post-prefetch dicts.
    """

    _NAME = "sharded_ebc"

    def _build(
        self,
    ) -> Tuple[
        _PostPrefetchProbeModule,
        PrefetchEmbeddingTrainPipelineContext,
        _RecordingModuleContext,
        _RecordingModuleContext,
    ]:
        """Builds a probe module and the context the prefetch stage would leave it."""
        data = _RecordingModuleContext()
        module_context = _RecordingModuleContext()
        context = PrefetchEmbeddingTrainPipelineContext(
            module_input_post_prefetch={self._NAME: data},
            module_contexts_post_prefetch={self._NAME: module_context},
        )
        module = _PostPrefetchProbeModule()
        module.forward = InSyncPrefetchEmbeddingPipelinedForward(
            name=self._NAME,
            args=CallArgs(args=[], kwargs={}),
            module=cast(ShardedModule, module),
            context=context,
        )
        return module, context, data, module_context

    def test_consumes_post_prefetch_handoff(self) -> None:
        """The lookup runs on the prefetched input and leaves the a2a for the forward.

        Both post-prefetch entries are popped: the pipeline reuses the same context
        object across iterations, so an entry left behind would be handed to a later
        batch's lookup.
        """
        module, context, data, module_context = self._build()

        start_embedding_lookup_post_prefetch(
            module=cast(ShardedModule, module),
            context=context,
        )

        self.assertEqual(module.calls, [(module_context, data)])
        self.assertEqual(context.embedding_a2a_requests, {self._NAME: module.awaitable})
        self.assertEqual(context.module_input_post_prefetch, {})
        self.assertEqual(context.module_contexts_post_prefetch, {})


class _RetryableDataError(Exception):
    is_retryable: bool = True


class _NonRetryableDataError(Exception):
    is_retryable: bool = False


class DataLoadingExceptionTest(unittest.TestCase):
    def test_is_data_loading_retriable_direct(self) -> None:
        self.assertTrue(
            _is_data_loading_retriable(_RetryableDataError("transient network error"))
        )
        self.assertFalse(
            _is_data_loading_retriable(_NonRetryableDataError("permission denied"))
        )

    def test_is_data_loading_retriable_via_cause(self) -> None:
        cause = _RetryableDataError("transient network error")
        wrapper = OSError("data loading failed")
        wrapper.__cause__ = cause
        self.assertTrue(_is_data_loading_retriable(wrapper))

    def test_is_data_loading_retriable_no_attribute(self) -> None:
        self.assertFalse(_is_data_loading_retriable(RuntimeError("plain")))

    def test_data_loading_thread_non_retriable_exception(self) -> None:
        error = _NonRetryableDataError("permission denied")
        thread = DataLoadingThread(
            device=torch.device("cpu"),
            dataloader_iter=iter([]),
            to_device_non_blocking=False,
        )
        thread._exception = error
        thread._buffer_filled_event.set()

        with self.assertRaises(_NonRetryableDataError) as ctx:
            thread.get_next_batch(none_throws=True)
        self.assertIs(ctx.exception, error)
        self.assertIsNone(thread._exception)


class UseContextForPostprocsTest(unittest.TestCase):
    def _make_postprocs(self, context: TrainPipelineContext) -> List[PipelinedPostproc]:
        return [
            PipelinedPostproc(
                postproc_module=torch.nn.Identity(),
                fqn=f"postproc_{i}",
                args=CallArgs(args=[], kwargs={}),
                context=context,
                default_stream=None,
                dist_stream=None,
            )
            for i in range(2)
        ]

    def test_restores_context_on_normal_exit(self) -> None:
        original = TrainPipelineContext(index=0)
        next_batch = TrainPipelineContext(index=1)
        postprocs = self._make_postprocs(original)

        with use_context_for_postprocs(postprocs, next_batch):
            for postproc in postprocs:
                self.assertIs(postproc.get_context(), next_batch)

        for postproc in postprocs:
            self.assertIs(postproc.get_context(), original)

    def test_restores_context_when_body_raises(self) -> None:
        original = TrainPipelineContext(index=0)
        next_batch = TrainPipelineContext(index=1)
        postprocs = self._make_postprocs(original)

        with self.assertRaises(RuntimeError):
            with use_context_for_postprocs(postprocs, next_batch):
                raise RuntimeError("_start_data_dist failed")

        for postproc in postprocs:
            self.assertIs(postproc.get_context(), original)


class _FindDDPModulesTestModel(torch.nn.Module):
    def __init__(self, table: EmbeddingBagConfig, device: torch.device) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(tables=[table], device=device)
        self.dense = torch.nn.Linear(1, 1, device=device)


class FindDDPModulesTest(TrainPipelineSparseDistTestBase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_finds_dp_sharded_ebc_lookup(self) -> None:
        device = self.device
        table = self.tables[0]
        model = _FindDDPModulesTestModel(table, device)
        sharder = cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
        module_plan = construct_module_sharding_plan(
            model.ebc,
            per_param_sharding={table.name: data_parallel()},
            local_size=1,
            world_size=1,
            device_type=device.type,
        )
        dmp = DistributedModelParallel(
            module=model,
            plan=ShardingPlan({"ebc": module_plan}),
            env=ShardingEnv.from_process_group(self.pg),
            sharders=[sharder],
            device=device,
        )

        ddp_wrapper = dmp._dmp_wrapped_module
        self.assertIsInstance(ddp_wrapper, DistributedDataParallel)
        sharded_ebc = next(
            module
            for module in dmp.modules()
            if isinstance(module, ShardedEmbeddingBagCollection)
        )
        lookup_ddp = sharded_ebc._lookups[0]
        self.assertIsInstance(lookup_ddp, DistributedDataParallel)

        ddp_modules = find_ddp_modules(dmp)
        self.assertEqual(2, len(ddp_modules))
        self.assertIs(ddp_modules[0], ddp_wrapper)
        self.assertIs(ddp_modules[1], lookup_ddp)


class _DDPManagedParamIdsTestModel(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.inner = torch.nn.Linear(2, 2, device=device)
        self.frozen = torch.nn.Parameter(
            torch.zeros(2, device=device), requires_grad=False
        )


class DDPManagedParamIdsTest(TrainPipelineSparseDistTestBase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_skips_ignored_and_frozen_params(self) -> None:
        model = _DDPManagedParamIdsTestModel(self.device)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model, ["inner.bias"]
        )
        ddp = DistributedDataParallel(model, process_group=self.pg)

        self.assertEqual(ddp_managed_param_ids([ddp]), {id(model.inner.weight)})


def _sgd(params: dict[str, torch.Tensor]) -> KeyedOptimizerWrapper:
    return KeyedOptimizerWrapper(
        params,
        # Citrine C2: use the multi-tensor optimizer implementation.
        lambda p: torch.optim.SGD(p, lr=0.1, foreach=True),
    )


class AsyncOptimizerUnsupportedReasonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.dense: KeyedOptimizerWrapper = _sgd({"w": self.weight})

    def _fused_and_dense(self) -> CombinedOptimizer:
        return CombinedOptimizer(
            [("sparse", EmptyFusedOptimizer()), ("dense", self.dense)]
        )

    def test_fused_and_ddp_params_are_supported(self) -> None:
        self.assertIsNone(
            async_optimizer_unsupported_reason(
                self._fused_and_dense(), {id(self.weight)}
            )
        )

    def test_param_outside_ddp_is_unsupported(self) -> None:
        reason = async_optimizer_unsupported_reason(self._fused_and_dense(), set())

        assert reason is not None
        self.assertIn("dense.w", reason)
        self.assertIn("DDP", reason)

    def test_slice_of_ddp_param_is_supported(self) -> None:
        # Sharded embedding modules expose per-table views of a batched weight; DDP
        # reduces the batched weight, and each view's .grad is a slice of its grad.
        batched_weight = torch.nn.Parameter(torch.zeros(8))
        table = TableBatchedEmbeddingSlice(batched_weight, 0, 4, 2, 2)
        optimizer = CombinedOptimizer([("dense", _sgd({"table": table}))])

        self.assertIsNone(
            async_optimizer_unsupported_reason(optimizer, {id(batched_weight)})
        )

    def test_warmup_wrapped_fused_is_supported(self) -> None:
        warmup = WarmupOptimizer(
            CombinedOptimizer([("f", EmptyFusedOptimizer())]),
            [WarmupStage(max_iters=10, value=0.1)],
            lr=0.1,
        )
        optimizer = CombinedOptimizer([("sparse", warmup), ("dense", self.dense)])

        self.assertIsNone(
            async_optimizer_unsupported_reason(optimizer, {id(self.weight)})
        )

    def test_warmup_wrapped_dense_is_supported(self) -> None:
        # MVAI wraps a dense group that has an LR warmup schedule this way. The
        # placeholder param WarmupOptimizer adds to self.dense.params is never stepped.
        warmup = WarmupOptimizer(
            self.dense,
            [WarmupStage(max_iters=10, value=0.1)],
            lr=0.1,
            param_name="dense.__warmup",
        )
        optimizer = CombinedOptimizer(
            [("sparse", EmptyFusedOptimizer()), ("dense", warmup)]
        )

        self.assertIsNone(
            async_optimizer_unsupported_reason(optimizer, {id(self.weight)})
        )

    def test_top_level_optimizer_wrapper_is_checked(self) -> None:
        optimizer = GradientClippingOptimizer(self._fused_and_dense())

        self.assertIsNone(
            async_optimizer_unsupported_reason(optimizer, {id(self.weight)})
        )
        self.assertIsNotNone(async_optimizer_unsupported_reason(optimizer, set()))

    def test_frozen_param_outside_ddp_is_supported(self) -> None:
        frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        optimizer = CombinedOptimizer(
            [("dense", _sgd({"w": self.weight, "frozen": frozen}))]
        )

        self.assertIsNone(
            async_optimizer_unsupported_reason(optimizer, {id(self.weight)})
        )

    def test_plain_optimizer_is_checked(self) -> None:
        # Citrine C2: use the multi-tensor optimizer implementation.
        optimizer = torch.optim.SGD([self.weight], lr=0.1, foreach=True)

        self.assertIsNone(
            async_optimizer_unsupported_reason(optimizer, {id(self.weight)})
        )
        self.assertIsNotNone(async_optimizer_unsupported_reason(optimizer, set()))

    def test_step_hooks_are_unsupported(self) -> None:
        optimizer = self._fused_and_dense()
        optimizer.register_step_pre_hook(lambda *_: None)

        reason = async_optimizer_unsupported_reason(optimizer, {id(self.weight)})

        assert reason is not None
        self.assertIn("hook", reason)

    def test_step_hooks_on_children_are_unsupported(self) -> None:
        hooked_combined = self._fused_and_dense()
        hooked_combined.register_step_pre_hook(lambda *_: None)

        def hooked_sgd(params: Any) -> torch.optim.Optimizer:
            # Citrine C2: use the multi-tensor optimizer implementation.
            optimizer = torch.optim.SGD(params, lr=0.1, foreach=True)
            optimizer.register_step_post_hook(lambda *_: None)
            return optimizer

        hooked_leaf = KeyedOptimizerWrapper({"w": self.weight}, hooked_sgd)

        for name, optimizer in [
            ("child", CombinedOptimizer([("child", hooked_combined)])),
            ("wrapped", GradientClippingOptimizer(hooked_combined)),
            ("leaf", CombinedOptimizer([("dense", hooked_leaf)])),
        ]:
            with self.subTest(name):
                reason = async_optimizer_unsupported_reason(
                    optimizer, {id(self.weight)}
                )

                assert reason is not None
                self.assertIn("hook", reason)

    def test_gradient_accumulation_wrapper_is_unsupported(self) -> None:
        wrapper = _GAOptimizerWrapper(
            CombinedOptimizer([("dense", self.dense)]), GradientAccumulationConfig()
        )

        reason = async_optimizer_unsupported_reason(wrapper, {id(self.weight)})

        assert reason is not None
        self.assertIn("_GAOptimizerWrapper", reason)

    def test_only_fused_is_unsupported(self) -> None:
        optimizer = CombinedOptimizer([("sparse", EmptyFusedOptimizer())])

        reason = async_optimizer_unsupported_reason(optimizer, set())

        assert reason is not None
        self.assertIn("nothing", reason)
