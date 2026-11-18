#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch._functorch.aot_autograd import make_boxed_func
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch.distributed.pipelining._backward import (
    stage_backward_input,
    stage_backward_weight,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torchrec.distributed.maglev.module import StructuredActivationsLayout
from torchrec.distributed.maglev.split_backward import (
    Compiler,
    eager_compiler,
    inductor_compiler,
    MAGLEV_SPLIT_BACKEND_NAME,
    register_maglev_split_backend,
    SplitBackwardBackend,
)
from torchrec.distributed.maglev.stage import (
    HandoffPGMode,
    MaglevProcessGroups,
    StageWrapper,
)
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import (
    MaglevTestActivations,
    MaglevTestLayer,
    MaglevTestModel,
)

_NUM_LAYERS = 2
_NUM_MICROBATCHES = 2
_MATMULS = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)


class _MatmulCounter(TorchDispatchMode):
    """Counts the matmuls dispatched inside a region.

    Every Linear backward is one matmul for its input gradient and one for its
    weight gradient, so the count is a direct measure of how much of a dense
    backward a region actually ran.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if func in _MATMULS:
            self.count += 1
        return func(*args, **(kwargs or {}))


class _RecordingCompiler:
    """Records every graph a :class:`SplitBackwardBackend` hands its compiler."""

    def __init__(self, compiler: Compiler = eager_compiler) -> None:
        self.graphs: List[torch.fx.GraphModule] = []
        self._compiler = compiler

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
    ) -> Callable[..., Any]:
        self.graphs.append(gm)
        return self._compiler(gm, example_inputs)


class _CountingPartitioner(CustomPartitionerFn):
    """A ``custom_partitioner_fn`` that counts its calls and partitions by min-cut."""

    def __init__(self) -> None:
        self.calls = 0

    def uuid(self) -> Any:
        return "counting-partitioner"

    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        self.calls += 1
        return min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)


def _model() -> MaglevTestModel:
    layout = StructuredActivationsLayout[MaglevTestActivations]()
    layers = []
    for layer_index in range(_NUM_LAYERS):
        torch.manual_seed(100 + layer_index)
        layers.append(
            MaglevTestLayer(
                tables=_tables(layer_index),
                layer_dim=6,
                is_first=layer_index == 0,
                activation_layout=layout,
                num_float_features=4,
                device=torch.device("cpu"),
            )
        )
    return MaglevTestModel(layers, layout)


def _tables(layer_index: int) -> List[Any]:
    return EmbeddingTablesConfig(
        num_unweighted_features=2,
        num_weighted_features=0,
        embedding_feature_dim=4,
        base_row_size=16,
    ).generate_tables(name_prefix=f"l{layer_index}_")[0]


def _inputs() -> List[ModelInput]:
    inputs = []
    for layer_index in range(_NUM_LAYERS):
        torch.manual_seed(500 + layer_index)
        random.seed(500 + layer_index)
        inputs.append(
            ModelInput.generate(
                batch_size=4,
                tables=_tables(layer_index),
                weighted_tables=[],
                num_float_features=4,
                device=torch.device("cpu"),
            )
        )
    return inputs


def _compile_dense_halves(model: MaglevTestModel, backend: Any) -> None:
    """What ``maglev_pp.runtime.compile_dense_halves`` does to a real model."""
    for layer in model.modules():
        if isinstance(layer, MaglevTestLayer):
            layer.dense = cast(
                torch.nn.Module, torch.compile(layer.require_dense(), backend=backend)
            )


def _counting_aot_backend(backward_calls: List[int]) -> Callable[..., Any]:
    """``aot_eager``, plus a count of how often the compiled backward runs."""

    def forward_compiler(gm: torch.fx.GraphModule, _inputs: Any) -> Any:
        return make_boxed_func(gm)

    def backward_compiler(gm: torch.fx.GraphModule, _inputs: Any) -> Any:
        def run(*args: Any) -> Any:
            backward_calls.append(1)
            return gm(*args)

        return make_boxed_func(run)

    return aot_autograd(fw_compiler=forward_compiler, bw_compiler=backward_compiler)


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
def _run_split_backward(
    model: MaglevTestModel, _get_rank: Any, _driver: Any
) -> Tuple[int, int]:
    """One single-stage Rail pass; return the matmuls run by every I, then W."""
    process_group = MagicMock()
    stage = StageWrapper(
        model,
        layers_per_stage=[_NUM_LAYERS],
        stage_size=1,
        process_groups=MaglevProcessGroups(
            stage_ranks=((0,),),
            stage_pg=process_group,
            handoff_pgs=(process_group, process_group),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
            handoff_pg_mode=HandoffPGMode.SHARED,
        ),
        enable_rail=True,
    )
    state = stage.sparse_forward_global(_inputs(), num_microbatches=_NUM_MICROBATCHES)
    input_matmuls = weight_matmuls = 0
    for microbatch in range(_NUM_MICROBATCHES):
        stage.compute_dense_forward_micro(
            state.dense_inputs[microbatch], (), microbatch, state.seams_for(microbatch)
        )
        with _MatmulCounter() as counter:
            stage.dense_backward_act_micro()
        input_matmuls += counter.count
        with _MatmulCounter() as counter:
            stage.dense_backward_weight_micro()
        weight_matmuls += counter.count
    stage.sparse_backward_global(state)
    return input_matmuls, weight_matmuls


class _AliasingRegion(torch.nn.Module):
    """Hands its input on reshaped next to a projection: one output aliases it."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return value.view(-1, 2, 2), self.proj(value)


class _IndexOutputRegion(torch.nn.Module):
    """A projection next to an integer index map, which takes no gradient."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.proj(value)
        return projected, projected.argmax(dim=-1)


class _NoGradOutputsRegion(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return value.detach() * 2, value > 0


def _run_region_halves(
    region: torch.nn.Module, weights: List[torch.nn.Parameter], value: torch.Tensor
) -> Tuple[int, int, Optional[torch.Tensor]]:
    """I then W on one region, through the helpers ``StageWrapper`` calls.

    Returns:
        The matmuls I runs, the matmuls W runs, and the input's gradient.
    """
    value = value.detach().requires_grad_()
    outputs = [output for output in region(value) if output.requires_grad]
    with _MatmulCounter() as counter:
        _, param_groups = stage_backward_input(
            stage_outputs_or_loss=[output.clone() for output in outputs],
            output_grads=[torch.ones_like(output) for output in outputs],
            input_values=[value],
            weights=iter(weights),
        )
    input_matmuls = counter.count
    with _MatmulCounter() as counter:
        stage_backward_weight(iter(weights), param_groups)
    return input_matmuls, counter.count, value.grad


def _assert_grads_close(actual: torch.nn.Module, expected: torch.nn.Module) -> None:
    expected_grads = {
        name: parameter.grad for name, parameter in expected.named_parameters()
    }
    for name, parameter in actual.named_parameters():
        name = name.replace("._orig_mod", "")
        assert parameter.grad is not None, f"no gradient for {name}"
        torch.testing.assert_close(parameter.grad, expected_grads[name], msg=name)


# Dynamo skips torchrec/distributed by default, which would leave the test
# model's dense halves silently eager and every assertion here vacuous.
@torch._dynamo.config.patch(skip_torchrec=False)
class SplitBackwardCompileTest(unittest.TestCase):
    def setUp(self) -> None:
        torch._dynamo.reset()
        counters.clear()

    def _assert_captured_whole(self, num_regions: int) -> None:
        """Every compiled region was traced in one graph, so none of it ran eager."""
        self.assertEqual(dict(counters["graph_break"]), {})
        self.assertEqual(counters["stats"]["unique_graphs"], num_regions)

    def test_torch_compile_reruns_whole_backward_in_each_half(self) -> None:
        """The PT2 problem: under torch.compile, I and W each run the whole backward.

        Eager splits for free -- each aten backward node computes only the
        outputs the current graph task consumes. AOTAutograd wraps the compiled
        backward in one ``autograd.Function``, which cannot see that, so ``I``
        runs every gradient and discards the weight ones, and ``W`` re-enters
        the same node and runs every gradient again.
        """
        eager_input, eager_weight = _run_split_backward(_model())

        backward_calls: List[int] = []
        compiled = _model()
        _compile_dense_halves(compiled, _counting_aot_backend(backward_calls))
        compiled_input, compiled_weight = _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertGreater(eager_input, 0)
        self.assertGreater(eager_weight, 0)
        # Each compiled dense half's backward runs once in I and again in W.
        self.assertEqual(len(backward_calls), 2 * _NUM_LAYERS * _NUM_MICROBATCHES)
        self.assertEqual(compiled_input, eager_input + eager_weight)
        self.assertEqual(compiled_weight, eager_input + eager_weight)

    def test_split_backend_runs_only_its_own_half(self) -> None:
        """The fix: with the dI/dW graphs compiled apart, each half matches eager."""
        eager = _model()
        eager_input, eager_weight = _run_split_backward(eager)

        compiler = _RecordingCompiler()
        compiled = _model()
        _compile_dense_halves(compiled, SplitBackwardBackend(compiler))
        compiled_input, compiled_weight = _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        # Forward, input half, weight half per region; never the uncut backward.
        self.assertEqual(len(compiler.graphs), 3 * _NUM_LAYERS)
        self.assertEqual(compiled_input, eager_input)
        self.assertEqual(compiled_weight, eager_weight)
        _assert_grads_close(compiled, eager)

    def test_split_backend_whole_backward_matches_eager(self) -> None:
        """A plain backward wants both halves, which is the monolithic schedules."""
        eager = _model()
        eager_loss, _ = eager(_inputs())
        eager_loss.backward()

        compiled = _model()
        _compile_dense_halves(compiled, SplitBackwardBackend(eager_compiler))
        compiled_loss, _ = compiled(_inputs())
        compiled_loss.backward()

        self._assert_captured_whole(_NUM_LAYERS)
        torch.testing.assert_close(compiled_loss, eager_loss)
        _assert_grads_close(compiled, eager)

    def test_split_backend_without_input_grads_runs_whole_backward(self) -> None:
        """A region with no differentiable input has no I; its backward stays whole."""
        torch.manual_seed(0)
        eager = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        compiled = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        compiled.load_state_dict(eager.state_dict())
        value = torch.randn(5, 4)

        compiler = _RecordingCompiler()
        eager(value).sum().backward()
        torch.compile(compiled, backend=SplitBackwardBackend(compiler))(
            value
        ).sum().backward()

        self._assert_captured_whole(1)
        # Forward and the uncut backward.
        self.assertEqual(len(compiler.graphs), 2)
        _assert_grads_close(compiled, eager)

    def test_split_backend_with_inductor_matches_eager(self) -> None:
        """The default compiler: each graph through Inductor, numerics unchanged."""
        eager = _model()
        _run_split_backward(eager)

        compiler = _RecordingCompiler(inductor_compiler)
        compiled = _model()
        _compile_dense_halves(compiled, SplitBackwardBackend(compiler))
        _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertEqual(len(compiler.graphs), 3 * _NUM_LAYERS)
        _assert_grads_close(compiled, eager)

    def test_split_backend_partitions_with_the_custom_partitioner(self) -> None:
        """A ``custom_partitioner_fn`` partitions each region; each half still runs alone."""
        eager = _model()
        eager_input, eager_weight = _run_split_backward(eager)

        partitioner = _CountingPartitioner()
        compiler = _RecordingCompiler()
        with torch._inductor.config.patch(
            custom_partitioner_fn=partitioner, force_disable_caches=True
        ):
            compiled = _model()
            _compile_dense_halves(compiled, SplitBackwardBackend(compiler))
            compiled_input, compiled_weight = _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertEqual(partitioner.calls, _NUM_LAYERS)
        self.assertEqual(len(compiler.graphs), 3 * _NUM_LAYERS)
        self.assertEqual(compiled_input, eager_input)
        self.assertEqual(compiled_weight, eager_weight)
        _assert_grads_close(compiled, eager)

    def test_registered_backend_splits_by_name(self) -> None:
        """``torch.compile(backend=<registered name>)`` runs the split backward."""
        name = "maglev_split_backward_eager_test"
        register_maglev_split_backend(name=name, compiler=eager_compiler)
        eager = _model()
        eager_input, eager_weight = _run_split_backward(eager)

        compiled = _model()
        _compile_dense_halves(compiled, name)
        compiled_input, compiled_weight = _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertEqual(compiled_input, eager_input)
        self.assertEqual(compiled_weight, eager_weight)
        _assert_grads_close(compiled, eager)

    def test_registered_backend_compiles_under_torch_compile_options(self) -> None:
        """Every extracted graph compiles under the options ``torch.compile`` passes."""
        wanted = not torch._inductor.config.max_autotune_gemm
        seen: List[bool] = []

        def recording(
            gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]
        ) -> Callable[..., Any]:
            seen.append(torch._inductor.config.max_autotune_gemm)
            return eager_compiler(gm, example_inputs)

        name = "maglev_split_backward_options_test"
        register_maglev_split_backend(name=name, compiler=recording)
        eager = _model()
        _run_split_backward(eager)

        compiled = _model()
        for layer in compiled.modules():
            if isinstance(layer, MaglevTestLayer):
                layer.dense = cast(
                    torch.nn.Module,
                    torch.compile(
                        layer.require_dense(),
                        backend=name,
                        options={"max_autotune_gemm": wanted},
                    ),
                )
        _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        # Each layer's forward, input half and weight half.
        self.assertEqual(seen, [wanted] * (3 * _NUM_LAYERS))
        _assert_grads_close(compiled, eager)

    def test_registers_the_inductor_split_backend_under_its_default_name(
        self,
    ) -> None:
        register_maglev_split_backend()

        self.assertIn(MAGLEV_SPLIT_BACKEND_NAME, torch._dynamo.list_backends())

    def test_split_backend_copies_an_output_that_aliases_an_input(self) -> None:
        """A region that hands an input on as a view still splits: the view is
        returned as a copy, which ``aot_export_joint_simple`` accepts."""
        value = torch.randn(6, 4)
        eager = _AliasingRegion()
        eager_input, eager_weight, eager_grad = _run_region_halves(
            eager, list(eager.parameters()), value
        )

        compiler = _RecordingCompiler()
        region = _AliasingRegion()
        compiled_input, compiled_weight, compiled_grad = _run_region_halves(
            cast(
                torch.nn.Module,
                torch.compile(region, backend=SplitBackwardBackend(compiler)),
            ),
            list(region.parameters()),
            value,
        )

        self._assert_captured_whole(1)
        # Forward, input half, weight half: split, not refused and run eager.
        self.assertEqual(len(compiler.graphs), 3)
        self.assertEqual((compiled_input, compiled_weight), (eager_input, eager_weight))
        torch.testing.assert_close(compiled_grad, eager_grad)
        _assert_grads_close(region, eager)

    def test_split_backend_compiles_outputs_without_grad_apart(self) -> None:
        """An output that takes no gradient comes from a separate inference graph,
        and the differentiable ones still split."""
        value = torch.randn(6, 4)
        eager = _IndexOutputRegion()
        eager_input, eager_weight, eager_grad = _run_region_halves(
            eager, list(eager.parameters()), value
        )

        compiler = _RecordingCompiler()
        region = _IndexOutputRegion()
        compiled = cast(
            torch.nn.Module,
            torch.compile(region, backend=SplitBackwardBackend(compiler)),
        )
        compiled_input, compiled_weight, compiled_grad = _run_region_halves(
            compiled, list(region.parameters()), value
        )

        self._assert_captured_whole(1)
        # The split's forward, input half and weight half, and the inference graph.
        self.assertEqual(len(compiler.graphs), 4)
        self.assertEqual((compiled_input, compiled_weight), (eager_input, eager_weight))
        torch.testing.assert_close(compiled_grad, eager_grad)
        _assert_grads_close(region, eager)
        self.assertTrue(torch.equal(compiled(value)[1], eager(value)[1]))

    def test_split_backend_compiles_a_region_with_no_differentiable_output_as_is(
        self,
    ) -> None:
        """Differentiable inputs but no differentiable output: nothing to split."""
        compiler = _RecordingCompiler()
        region = torch.compile(
            _NoGradOutputsRegion(), backend=SplitBackwardBackend(compiler)
        )
        value = torch.randn(6, 4, requires_grad=True)
        doubled, positive = region(value)

        self._assert_captured_whole(1)
        self.assertEqual(len(compiler.graphs), 1)
        torch.testing.assert_close(doubled, value.detach() * 2)
        self.assertTrue(torch.equal(positive, value > 0))
