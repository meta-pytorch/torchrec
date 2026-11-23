#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Sequence, Tuple
from unittest.mock import patch

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch._functorch.aot_autograd import make_boxed_func
from torch.utils._python_dispatch import TorchDispatchMode
from torchrec.distributed.maglev.data_parallel import (
    _accumulate_weight_grads,
    PassReduction,
)
from torchrec.distributed.maglev.input_dist import split_dense_inputs
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    MaglevLayer,
    StructuredActivationsLayout,
)
from torchrec.distributed.maglev.split_backward import (
    Compiler,
    eager_compiler,
    inductor_compiler,
    SplitBackwardBackend,
)
from torchrec.distributed.maglev.stage import StageWrapper
from torchrec.distributed.maglev.tests.utils import (
    maglev_test_inputs,
    maglev_test_model,
    mock_groups,
)
from torchrec.distributed.test_utils.model_input import ModelInput
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


def _model() -> MaglevTestModel:
    return maglev_test_model(_NUM_LAYERS)


def _inputs() -> List[ModelInput]:
    return maglev_test_inputs(_NUM_LAYERS)


def _compile_dense_nets(model: MaglevTestModel, backend: Any) -> None:
    """Compile every layer's dense net in place with ``backend``."""
    for layer in model.modules():
        if isinstance(layer, MaglevLayer):
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
def _stage(model: MaglevTestModel, _get_rank: Any, _driver: Any) -> StageWrapper:
    """A single-stage StageWrapper over the whole model."""
    return StageWrapper(
        model,
        layers_per_stage=[len(model)],
        stage_size=1,
        process_groups=mock_groups(),
    )


class _Schedule:
    """Stands in for a schedule: the stage only tells one from another."""


class _AccumulatingReduction(PassReduction):
    """No data parallelism, but ``W`` runs each weight's accumulator, as the DDP
    reduction's does."""

    def weight_backward(
        self,
        graph_parameters: Callable[[], Iterable[torch.nn.Parameter]],
        param_groups: List[Dict[str, Any]],
    ) -> None:
        _accumulate_weight_grads(param_groups)


def _run_split_backward(
    model: MaglevTestModel,
    inputs: Optional[List[ModelInput]] = None,
    reduction: Optional[PassReduction] = None,
) -> Tuple[int, int]:
    """One single-stage split pass; return the matmuls run by every I, then W.
    ``reduction`` is handed to the stage as a schedule would hand its own."""
    stage = _stage(model)
    if reduction is not None:
        stage.configure_pass(
            _Schedule(), loss_scale=1.0, allow_unreached=True, reduction=reduction
        )
    state = stage.sparse_forward_global(
        _inputs() if inputs is None else inputs, num_microbatches=_NUM_MICROBATCHES
    )
    input_matmuls = weight_matmuls = 0
    for microbatch in range(_NUM_MICROBATCHES):
        stage._compute_dense_forward_micro(
            state.dense_inputs[microbatch], (), microbatch, state.seams[microbatch]
        )
        with _MatmulCounter() as counter:
            stage.dense_backward_act_micro()
        input_matmuls += counter.count
        with _MatmulCounter() as counter:
            stage.dense_backward_weight_micro()
        weight_matmuls += counter.count
    stage.sparse_backward_global(state)
    return input_matmuls, weight_matmuls


def _run_whole_backward(model: MaglevTestModel) -> int:
    """One single-stage pass of whole backwards; return the matmuls they run."""
    stage = _stage(model)
    state = stage.sparse_forward_global(_inputs(), num_microbatches=_NUM_MICROBATCHES)
    matmuls = 0
    for microbatch in range(_NUM_MICROBATCHES):
        stage._compute_dense_forward_micro(
            state.dense_inputs[microbatch], (), microbatch, state.seams[microbatch]
        )
        with _MatmulCounter() as counter:
            stage.dense_backward_micro()
        matmuls += counter.count
    stage.sparse_backward_global(state)
    return matmuls


def _count_accumulations(model: MaglevTestModel) -> Dict[int, int]:
    """Count each dense weight's accumulator runs: the hook DDP's reducer uses."""
    counts: Dict[int, int] = {}

    def hook(parameter: torch.Tensor) -> None:
        counts[id(parameter)] += 1

    for layer in model:
        for parameter in cast(MaglevLayer, layer).dense_parameters():
            counts[id(parameter)] = 0
            parameter.register_post_accumulate_grad_hook(hook)
    return counts


class _MixedSeamSparse(torch.nn.Module):
    """One projection as an FP32 and a BF16 seam, plus a whole-pass table."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        self.table = torch.nn.Parameter(torch.arange(10.0).view(5, 2) / 10)

    def forward(self, layer_input: ModelInput) -> Tuple[torch.Tensor, ...]:
        value = layer_input.float_features[:, :2] @ self.weight
        return value, value.to(torch.bfloat16), self.table * 2


class _MixedSeamDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.proj = torch.nn.Linear(2, 2)

    def forward(
        self,
        sparse_output: Tuple[torch.Tensor, ...],
        layer_input: ModelInput,
        in_activations: Activations = (),
    ) -> Activations:
        exact, low, table = sparse_output
        rows = (layer_input.float_features[:, 2] * 100).long() % table.shape[0]
        return (self.proj(exact + low.float() + table.index_select(0, rows)),)


class _MixedSeamLayer(MaglevLayer):
    """Per-dtype seams and a shared seam, the forms a tuple seam can take."""

    def __init__(self) -> None:
        super().__init__(sparse=_MixedSeamSparse(), dense=_MixedSeamDense())

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2]), torch.float32),)

    def split_dense_input(
        self, layer_input: ModelInput, batch_size: int, num_microbatches: int
    ) -> List[Any]:
        return split_dense_inputs(layer_input, batch_size, num_microbatches)

    def shared_sparse_outputs(self) -> Tuple[int, ...]:
        return (2,)


def _mixed_seam_model() -> MaglevTestModel:
    return MaglevTestModel(
        [cast(MaglevTestLayer, _MixedSeamLayer())],
        StructuredActivationsLayout[MaglevTestActivations](),
    )


def _assert_grads_close(actual: torch.nn.Module, expected: torch.nn.Module) -> None:
    expected_grads = {
        name: parameter.grad for name, parameter in expected.named_parameters()
    }
    for name, parameter in actual.named_parameters():
        name = name.replace("._orig_mod", "")
        assert parameter.grad is not None, f"no gradient for {name}"
        torch.testing.assert_close(parameter.grad, expected_grads[name], msg=name)


# Dynamo skips torchrec/distributed by default, which would leave the test
# model's dense nets silently eager and every assertion here vacuous.
@torch._dynamo.config.patch(skip_torchrec=False)
class SplitBackwardCompileTest(unittest.TestCase):
    def setUp(self) -> None:
        torch._dynamo.reset()
        counters.clear()

    def _assert_captured_whole(self, num_regions: int) -> None:
        """Every compiled region was traced in one graph, so none of it ran eager."""
        self.assertEqual(dict(counters["graph_break"]), {})
        self.assertEqual(counters["stats"]["unique_graphs"], num_regions)

    def test_torch_compile_reruns_whole_backward_in_each_part(self) -> None:
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
        _compile_dense_nets(compiled, _counting_aot_backend(backward_calls))
        compiled_input, compiled_weight = _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertGreater(eager_input, 0)
        self.assertGreater(eager_weight, 0)
        # Each compiled dense net's backward runs once in I and again in W.
        self.assertEqual(len(backward_calls), 2 * _NUM_LAYERS * _NUM_MICROBATCHES)
        self.assertEqual(compiled_input, eager_input + eager_weight)
        self.assertEqual(compiled_weight, eager_input + eager_weight)

    def test_split_backend_runs_only_its_own_part(self) -> None:
        """The fix: with the dI/dW graphs compiled apart, each part matches eager."""
        eager = _model()
        eager_input, eager_weight = _run_split_backward(eager)

        compiler = _RecordingCompiler()
        compiled = _model()
        _compile_dense_nets(compiled, SplitBackwardBackend(compiler))
        compiled_input, compiled_weight = _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        # Forward, input-gradient part, weight-gradient part per region; never
        # the uncut backward.
        self.assertEqual(len(compiler.graphs), 3 * _NUM_LAYERS)
        self.assertEqual(compiled_input, eager_input)
        self.assertEqual(compiled_weight, eager_weight)
        _assert_grads_close(compiled, eager)

    def test_split_backend_whole_backward_matches_eager(self) -> None:
        """A plain backward, as a whole-backward schedule runs, wants both parts."""
        eager = _model()
        eager_loss, _ = eager(_inputs())
        eager_loss.backward()

        compiled = _model()
        _compile_dense_nets(compiled, SplitBackwardBackend(eager_compiler))
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
        _compile_dense_nets(compiled, SplitBackwardBackend(compiler))
        _run_split_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertEqual(len(compiler.graphs), 3 * _NUM_LAYERS)
        _assert_grads_close(compiled, eager)

    def test_split_backend_whole_backward_through_the_stage(self) -> None:
        """A stage's whole backward, as MaglevRail1F1B runs it: one call runs
        the dI graph, then the dW graph, and repeats no matmul."""
        eager = _model()
        eager_matmuls = _run_whole_backward(eager)
        eager_input, eager_weight = _run_split_backward(_model())

        compiler = _RecordingCompiler()
        compiled = _model()
        _compile_dense_nets(compiled, SplitBackwardBackend(compiler))
        compiled_matmuls = _run_whole_backward(compiled)

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertEqual(len(compiler.graphs), 3 * _NUM_LAYERS)
        self.assertEqual(eager_matmuls, eager_input + eager_weight)
        self.assertEqual(compiled_matmuls, eager_matmuls)
        _assert_grads_close(compiled, eager)

    def test_split_backend_keeps_the_split_when_w_runs_the_accumulators(
        self,
    ) -> None:
        """DDP's weight-gradient part runs each weight's accumulator, so its reducer sees
        the gradient; the compiled ``W`` still runs only the dW graph."""
        eager = _model()
        eager_input, eager_weight = _run_split_backward(eager)

        compiled = _model()
        _compile_dense_nets(compiled, SplitBackwardBackend(_RecordingCompiler()))
        accumulations = _count_accumulations(compiled)
        compiled_input, compiled_weight = _run_split_backward(
            compiled, reduction=_AccumulatingReduction()
        )

        self._assert_captured_whole(_NUM_LAYERS)
        self.assertEqual(compiled_input, eager_input)
        self.assertEqual(compiled_weight, eager_weight)
        self.assertEqual(set(accumulations.values()), {_NUM_MICROBATCHES})
        _assert_grads_close(compiled, eager)

    def test_split_backend_with_tuple_and_shared_seams(self) -> None:
        """Per-dtype seams and a shared whole-pass seam reach the compiled dense
        net as plain inputs, and the split still matches eager."""
        inputs = _inputs()[:1]
        eager = _mixed_seam_model()
        eager_input, eager_weight = _run_split_backward(eager, inputs)

        compiler = _RecordingCompiler()
        compiled = _mixed_seam_model()
        _compile_dense_nets(compiled, SplitBackwardBackend(compiler))
        compiled_input, compiled_weight = _run_split_backward(compiled, inputs)

        self._assert_captured_whole(1)
        self.assertEqual(len(compiler.graphs), 3)
        self.assertGreater(eager_input, 0)
        self.assertGreater(eager_weight, 0)
        self.assertEqual((compiled_input, compiled_weight), (eager_input, eager_weight))
        _assert_grads_close(compiled, eager)
