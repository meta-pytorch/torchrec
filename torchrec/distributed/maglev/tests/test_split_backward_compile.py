#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch._functorch.aot_autograd import make_boxed_func
from torch.utils._python_dispatch import TorchDispatchMode
from torchrec.distributed.maglev.module import StructuredActivationsLayout
from torchrec.distributed.maglev.split_backward import (
    Compiler,
    eager_compiler,
    inductor_compiler,
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
