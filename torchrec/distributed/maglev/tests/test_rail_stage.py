#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import unittest
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.pipelining._backward import (
    stage_backward_input,
    stage_backward_weight,
)
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.checks import (
    reject_parameters_outside_layers,
    reject_parameters_outside_nets,
)
from torchrec.distributed.maglev.data_parallel import (
    _accumulate_weight_grads,
    only_data_parallel_wraps,
)
from torchrec.distributed.maglev.input_dist import split_dense_inputs
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    MaglevLayer,
    MaglevModuleList,
    StructuredActivations,
    StructuredActivationsLayout,
)
from torchrec.distributed.maglev.pipeline import MaglevRail1F1B, MaglevRailZB
from torchrec.distributed.maglev.stage import StageWrapper
from torchrec.distributed.maglev.tests.utils import (
    maglev_test_inputs,
    maglev_test_model,
    mock_groups,
    run_split_dense,
)
from torchrec.distributed.test_utils.test_model import (
    MaglevTestActivations,
    MaglevTestLayer,
    MaglevTestModel,
)


@dataclass(frozen=True)
class _WrappedTensor:
    tensor: torch.Tensor


@dataclass(frozen=True)
class _TestStructuredActivations(StructuredActivations):
    tensor: torch.Tensor
    pair: Tuple[torch.Tensor, torch.Tensor]
    sequence: List[torch.Tensor]
    mapping: Dict[str, torch.Tensor]
    numbers: List[int]
    counts: Dict[str, int]
    version: int
    label: str
    wrapped: _WrappedTensor

    def _pack_wrapped(self, value: _WrappedTensor, metadata: str) -> Activations:
        if metadata != "wrapped":
            raise ValueError("unexpected wrapped metadata")
        return (value.tensor,)

    @classmethod
    def _unpack_wrapped(
        cls,
        activations: Activations,
        metadata: str,
    ) -> Tuple[_WrappedTensor, Activations]:
        if metadata != "wrapped" or not activations:
            raise ValueError("cannot unpack wrapped tensor")
        return _WrappedTensor(activations[0]), activations[1:]


@dataclass(frozen=True)
class _TensorStructuredActivations(StructuredActivations):
    tensor: torch.Tensor


class _SplitSpyLayer(MaglevLayer):
    """A minimal layer whose nets and splitter can be made to misbehave."""

    def __init__(
        self,
        sparse: Optional[torch.nn.Module] = None,
        split_count: Optional[int] = None,
    ) -> None:
        super().__init__(
            sparse=sparse if sparse is not None else _GoodSparse(),
            dense=_PassThroughDense(),
        )
        self._split_count = split_count
        self._spec: ActivationSpec = ActivationSpec(torch.Size([-1, 2]), torch.float32)

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (self._spec,)

    def split_dense_input(
        self, layer_input: Any, batch_size: int, num_microbatches: int
    ) -> List[Any]:
        count = self._split_count or num_microbatches
        return [layer_input] * count


class _GoodSparse(torch.nn.Module):
    def forward(self, layer_input: Any) -> torch.Tensor:
        return torch.ones(layer_input.float_features.shape[0], 2)


class _PassThroughDense(torch.nn.Module):
    def forward(
        self,
        sparse_output: Any,
        layer_input: Any,
        in_activations: Activations = (),
    ) -> Activations:
        return (sparse_output,)


class _TupleSeamSparse(torch.nn.Module):
    """One projection, returned as an FP32 seam and a BF16 seam."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))

    def forward(self, layer_input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        value = layer_input.float_features[:, :2] @ self.weight
        return value, value.to(torch.bfloat16)


class _TupleSeamDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        sparse_output: Any,
        layer_input: Any,
        in_activations: Activations = (),
    ) -> Activations:
        exact, low = sparse_output
        return ((exact + low.float()) * self.scale,)


class _TupleSeamLayer(MaglevLayer):
    def __init__(self) -> None:
        super().__init__(sparse=_TupleSeamSparse(), dense=_TupleSeamDense())

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2]), torch.float32),)

    def split_dense_input(
        self, layer_input: Any, batch_size: int, num_microbatches: int
    ) -> List[Any]:
        return split_dense_inputs(layer_input, batch_size, num_microbatches)


class _SharedSeamSparse(torch.nn.Module):
    """A batch-major projection and a whole-pass table that any row may read."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
        self.table = torch.nn.Parameter(torch.arange(10.0).view(5, 2) / 10)

    def forward(self, layer_input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        value = layer_input.float_features[:, :2] @ self.weight
        return value, self.table * 2


class _SharedSeamDense(torch.nn.Module):
    def forward(
        self,
        sparse_output: Any,
        layer_input: Any,
        in_activations: Activations = (),
    ) -> Activations:
        value, table = sparse_output
        rows = (layer_input.float_features[:, 2] * 100).long() % table.shape[0]
        return (value + table.index_select(0, rows),)


class _SharedSeamLayer(MaglevLayer):
    def __init__(self) -> None:
        super().__init__(sparse=_SharedSeamSparse(), dense=_SharedSeamDense())

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2]), torch.float32),)

    def split_dense_input(
        self, layer_input: Any, batch_size: int, num_microbatches: int
    ) -> List[Any]:
        return split_dense_inputs(layer_input, batch_size, num_microbatches)

    def shared_sparse_outputs(self) -> Tuple[int, ...]:
        return (1,)


class RailStageTest(unittest.TestCase):
    """Single-process checks of the authoring API and the stage, on mock
    process groups."""

    def _model(self, num_layers: int = 4) -> MaglevTestModel:
        return maglev_test_model(num_layers)

    def _inputs(self, num_layers: int = 4, batch_size: int = 4) -> List[Any]:
        return list(maglev_test_inputs(num_layers, batch_size))

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_take_inputs_sends_one_carrier_per_stage(
        self,
        _get_rank: Any,
        input_dist_driver: Any,
    ) -> None:
        process_groups = mock_groups(2)
        stage = StageWrapper(
            self._model(num_layers=2),
            layers_per_stage=[1, 1],
            stage_size=1,
            process_groups=process_groups,
        )
        raw_batch = self._inputs(num_layers=2, batch_size=4)
        received = [raw_batch[0]]
        send_sets: List[List[List[Any]]] = []

        def take(next_send: Any, count: int) -> List[List[Any]]:
            send_sets.append(next_send())
            return [received] * count

        input_dist_driver.return_value.take.side_effect = take

        (global_inputs,) = stage.take_inputs(iter([raw_batch]), 1)

        self.assertIs(global_inputs, received)
        self.assertEqual(input_dist_driver.return_value.take.call_args.args[1], 1)
        (send_set,) = send_sets
        self.assertIs(send_set[0][0], raw_batch[0])
        self.assertIs(send_set[1][0], raw_batch[1])

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_sparse_and_dense_phases_match_whole_batch(
        self,
        _get_rank: Any,
        _input_dist_driver: Any,
    ) -> None:
        class ForwardWrapper(torch.nn.Module):
            def __init__(self, module: torch.nn.Module) -> None:
                super().__init__()
                self.module = module
                self.phases: List[str] = []

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                self.phases.append(
                    "dense" if kwargs.get("sparse_outputs") is not None else "sparse"
                )
                return self.module(*args, **kwargs)

        model = self._model(num_layers=2)
        reference = self._model(num_layers=2)
        inputs = self._inputs(num_layers=2, batch_size=4)
        stage = StageWrapper(
            model,
            layers_per_stage=[2],
            stage_size=1,
            process_groups=mock_groups(),
        )
        wrapper = ForwardWrapper(stage.module)
        stage.module = wrapper

        state = stage.sparse_forward_global(inputs, num_microbatches=2)
        run_split_dense(stage, state)
        self.assertEqual(wrapper.phases, ["sparse", "dense", "dense"])
        self.assertTrue(
            all(
                isinstance(seam, torch.Tensor) and seam.grad is not None
                for seams in state.seams
                for seam in seams
                if seam is not None
            )
        )
        stage.sparse_backward_global(state)

        reference_loss, _output = reference(inputs)
        (reference_loss * 2).backward()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad)
            expected = reference_parameters[name].grad
            self.assertIsNotNone(expected)
            assert parameter.grad is not None and expected is not None
            torch.testing.assert_close(parameter.grad, expected)

    def _single_stage(self, layers: List[Any]) -> StageWrapper:
        """A single-stage StageWrapper over hand-built layers."""
        return StageWrapper(
            MaglevTestModel(
                cast(List[MaglevTestLayer], layers),
                StructuredActivationsLayout[MaglevTestActivations](),
            ),
            layers_per_stage=[len(layers)],
            stage_size=1,
            process_groups=mock_groups(),
        )

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_pass_scope_poisons_a_failed_pass(
        self, _get_rank: Any, driver: Any
    ) -> None:
        stage = self._single_stage([_SplitSpyLayer()])
        with stage.pass_scope():
            # A pass's inputs come before its scope, never inside one.
            with self.assertRaisesRegex(RuntimeError, "still running"):
                stage.take_inputs(iter([]), 1)
        with self.assertRaisesRegex(RuntimeError, "weight failed"):
            with stage.pass_scope():
                raise RuntimeError("weight failed")
        with self.assertRaisesRegex(RuntimeError, "restart the job"):
            with stage.pass_scope():
                self.fail("failed communicators must not be reused")
        # The input round is a collective with peers now out of step, so it is
        # refused before it starts.
        with self.assertRaisesRegex(RuntimeError, "restart the job"):
            stage.take_inputs(iter([]), 1)
        driver.return_value.take.assert_not_called()

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_rail_schedules_reject_parameters_outside_the_layers(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        """A weight postproc applies sits outside every layer, so never trains
        under a pass; Maglev1F1B, whose dense forward applies postproc every
        microbatch, still takes it."""
        model = MaglevTestModel(
            cast(List[MaglevTestLayer], [_SplitSpyLayer()]),
            StructuredActivationsLayout[MaglevTestActivations](),
        )
        model.temperature = torch.nn.Parameter(torch.ones(1))
        stage = StageWrapper(
            model, layers_per_stage=[1], stage_size=1, process_groups=mock_groups()
        )
        # Citrine C2: foreach=True for multi-tensor optimizer execution
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        for rail_cls in (MaglevRail1F1B, MaglevRailZB):
            with self.subTest(rail_cls.__name__), self.assertRaisesRegex(
                ValueError, r"outside its layers: \['temperature'\]"
            ):
                rail_cls(stage, optimizer, num_microbatches=1)
        # A head registered on the model counts as a layer, which it is not.
        del model.temperature
        model.append(torch.nn.Linear(2, 2))
        with self.assertRaisesRegex(
            ValueError, r"not MaglevLayers, at \['1 \(Linear\)'\]"
        ):
            reject_parameters_outside_layers(model)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_only_dmp_and_ddp_unwrap_to_the_layers(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        class FakeDMP(torch.nn.Module):
            def __init__(self, module: torch.nn.Module) -> None:
                super().__init__()
                self._dmp_wrapped_module = module

        stage = self._single_stage([_SplitSpyLayer()])
        chain = stage.module
        self.assertTrue(only_data_parallel_wraps(stage.module, stage.layer_chain))
        stage.module = FakeDMP(chain)
        self.assertTrue(only_data_parallel_wraps(stage.module, stage.layer_chain))
        ddp = MagicMock(spec=DistributedDataParallel)
        ddp.module = FakeDMP(chain)
        stage.module = ddp
        self.assertTrue(only_data_parallel_wraps(stage.module, stage.layer_chain))
        stage.module = torch.nn.Sequential(chain)
        self.assertFalse(only_data_parallel_wraps(stage.module, stage.layer_chain))

    def test_accumulate_weight_grads_runs_accumulators_like_upstream(self) -> None:
        class ReusedWeight(torch.nn.Module):
            # The weight feeds two intermediates, so its group exercises the
            # pre-hooks that keep descendants from being counted twice.
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return (value @ self.weight).tanh() @ self.weight

        for name, factory in (
            (
                "mlp",
                lambda: torch.nn.Sequential(
                    torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3)
                ),
            ),
            ("reused", ReusedWeight),
        ):
            with self.subTest(model=name):
                torch.manual_seed(0)
                reference = factory()
                candidate = copy.deepcopy(reference)
                fired: Dict[str, int] = {}
                for param_name, weight in candidate.named_parameters():
                    weight.register_post_accumulate_grad_hook(
                        lambda _w, n=param_name, counts=fired: counts.__setitem__(
                            n, counts.get(n, 0) + 1
                        )
                    )
                for microbatch in range(2):
                    value = torch.randn(5, 3) + microbatch
                    for module, upstream in ((reference, True), (candidate, False)):
                        stage_input = value.clone().requires_grad_()
                        output = module(stage_input)
                        _, param_groups = stage_backward_input(
                            stage_outputs_or_loss=[output],
                            output_grads=[torch.ones_like(output)],
                            input_values=[stage_input],
                            weights=iter(list(module.parameters())),
                        )
                        if upstream:
                            stage_backward_weight(
                                iter(list(module.parameters())), param_groups
                            )
                        else:
                            _accumulate_weight_grads(param_groups)
                for (param_name, expected), actual in zip(
                    reference.named_parameters(), candidate.parameters()
                ):
                    torch.testing.assert_close(
                        actual.grad, expected.grad, rtol=0, atol=0
                    )
                    self.assertEqual(fired[param_name], 2, param_name)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_sparse_forward_rejects_bad_sparse_output(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        """The sparse net must return batch-major floating-point tensors."""

        class BadSparse(torch.nn.Module):
            def __init__(self, value: Any) -> None:
                super().__init__()
                self.value = value

            def forward(self, layer_input: Any) -> Any:
                return self.value

        cases = [
            ("not a tensor", "expected torch.Tensor", "nope"),
            ("wrong leading dim", "expected leading dimension", torch.ones(3, 2)),
            ("integral dtype", "must be floating point", torch.ones(4, 2).long()),
            ("empty tuple", "non-empty tuple", ()),
            (
                "tuple wrong leading dim",
                "expected leading dimension",
                (torch.ones(4, 2), torch.ones(3, 2)),
            ),
            (
                "tuple integral dtype",
                "must be floating point",
                (torch.ones(4, 2), torch.ones(4, 2).long()),
            ),
        ]
        for label, message, value in cases:
            with self.subTest(label):
                layer = _SplitSpyLayer(sparse=BadSparse(value))
                stage = self._single_stage([layer])
                inputs = self._inputs(num_layers=1, batch_size=4)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    stage.sparse_forward_global(inputs, num_microbatches=2)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_tuple_seams_keep_their_dtypes_and_match_whole_batch(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        layer = _TupleSeamLayer()
        reference = MaglevTestModel(
            [cast(MaglevTestLayer, copy.deepcopy(layer))],
            StructuredActivationsLayout[MaglevTestActivations](),
        )
        stage = self._single_stage([layer])
        inputs = self._inputs(num_layers=1, batch_size=4)

        state = stage.sparse_forward_global(inputs, num_microbatches=2)
        seams = [cast(Tuple[torch.Tensor, ...], state.seams[m][0]) for m in range(2)]
        run_split_dense(stage, state)
        self.assertTrue(
            all(
                leaf.grad is not None and leaf.grad.dtype == leaf.dtype
                for seam in seams
                for leaf in seam
            )
        )
        stage.sparse_backward_global(state)
        reference_loss, _output = reference(inputs)
        (reference_loss * 2).backward()

        self.assertEqual(
            [[(leaf.dtype, tuple(leaf.shape)) for leaf in seam] for seam in seams],
            [[(torch.float32, (2, 2)), (torch.bfloat16, (2, 2))]] * 2,
        )
        # Released as they were concatenated.
        self.assertTrue(all(leaf.grad is None for seam in seams for leaf in seam))
        for actual, expected in zip(layer.parameters(), reference.parameters()):
            self.assertIsNotNone(actual.grad)
            torch.testing.assert_close(actual.grad, expected.grad)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_shared_seam_is_one_leaf_and_matches_whole_batch(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        layer = _SharedSeamLayer()
        reference = MaglevTestModel(
            [cast(MaglevTestLayer, copy.deepcopy(layer))],
            StructuredActivationsLayout[MaglevTestActivations](),
        )
        stage = self._single_stage([layer])
        inputs = self._inputs(num_layers=1, batch_size=4)

        state = stage.sparse_forward_global(inputs, num_microbatches=2)
        seams = [cast(Tuple[torch.Tensor, ...], state.seams[m][0]) for m in range(2)]
        self.assertEqual([tuple(seam[0].shape) for seam in seams], [(2, 2)] * 2)
        self.assertIs(seams[0][1], seams[1][1])
        self.assertEqual(tuple(seams[0][1].shape), (5, 2))
        run_split_dense(stage, state)
        stage.sparse_backward_global(state)
        reference_loss, _output = reference(inputs)
        (reference_loss * 2).backward()

        self.assertIsNone(seams[0][1].grad)
        for actual, expected in zip(layer.parameters(), reference.parameters()):
            self.assertIsNotNone(actual.grad)
            torch.testing.assert_close(actual.grad, expected.grad)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_sparse_forward_rejects_shared_positions_past_the_output(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        class PastTheEnd(_SharedSeamLayer):
            def shared_sparse_outputs(self) -> Tuple[int, ...]:
                return (2,)

        stage = self._single_stage([PastTheEnd()])
        inputs = self._inputs(num_layers=1, batch_size=4)
        with self.assertRaisesRegex(ValueError, "shared positions"):
            stage.sparse_forward_global(inputs, num_microbatches=2)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_sparse_backward_releases_seam_gradients_first(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        """Peak memory: the joint backward must not also hold every leaf's piece."""
        stage = self._single_stage([_TupleSeamLayer()])
        inputs = self._inputs(num_layers=1, batch_size=4)
        state = stage.sparse_forward_global(inputs, num_microbatches=2)
        run_split_dense(stage, state)
        pooled = cast(Tuple[torch.Tensor, ...], state.sparse_outputs[0])
        leaves = [
            leaf
            for seams in state.seams
            for leaf in cast(Tuple[torch.Tensor, ...], seams[0])
        ]
        self.assertTrue(all(leaf.grad is not None for leaf in leaves))
        released: List[bool] = []
        for tensor in pooled:
            tensor.register_hook(
                lambda _grad: released.append(all(leaf.grad is None for leaf in leaves))
            )
        stage.sparse_backward_global(state)
        self.assertEqual(released, [True] * len(pooled))

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_sparse_forward_rejects_bad_microbatch_count(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        stage = self._single_stage([_SplitSpyLayer()])
        inputs = self._inputs(num_layers=1, batch_size=4)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            stage.sparse_forward_global(inputs, num_microbatches=0)
        with self.assertRaisesRegex(ValueError, "non-empty microbatches"):
            stage.sparse_forward_global(inputs, num_microbatches=8)
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            stage.sparse_forward_global(inputs, num_microbatches=3)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_sparse_forward_rejects_wrong_split_count(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        """A layer whose splitter disagrees with the schedule is caught."""
        stage = self._single_stage([_SplitSpyLayer(split_count=3)])
        inputs = self._inputs(num_layers=1, batch_size=4)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            stage.sparse_forward_global(inputs, num_microbatches=2)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_weight_backward_without_queued_work(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        """The guard that turns a drain-loop bug into a loud failure."""
        stage = self._single_stage([_SplitSpyLayer()])
        with self.assertRaisesRegex(ValueError, r"no\s+deferred weight work"):
            stage.dense_backward_weight_micro()

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_unused_dense_parameter_has_no_gradient_or_update(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        model = self._model(num_layers=1)
        reference = self._model(num_layers=1)
        for authored in (model, reference):
            cast(MaglevLayer, authored[0]).require_dense().register_parameter(
                "inactive", torch.nn.Parameter(torch.tensor(7.0))
            )
        stage = self._single_stage(list(model))
        inputs = self._inputs(num_layers=1, batch_size=4)
        optimizer = torch.optim.SGD(
            stage.module.parameters(), lr=0.05, weight_decay=0.1, foreach=True
        )
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.05, weight_decay=0.1, foreach=True
        )
        state = stage.sparse_forward_global(inputs, num_microbatches=2)
        run_split_dense(stage, state)
        stage.sparse_backward_global(state)
        reference_loss, _output = reference(inputs)
        (reference_loss * 2).backward()
        inactive = cast(MaglevLayer, model[0]).require_dense().get_parameter("inactive")
        self.assertIsNone(inactive.grad)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(inactive, torch.tensor(7.0), rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad)

    def test_parameters_outside_the_nets_are_refused(self) -> None:
        """A parameter on the layer body would silently never train."""

        class Dense(torch.nn.Module):
            def forward(
                self,
                sparse_output: Any,
                layer_input: torch.Tensor,
                in_activations: Activations = (),
            ) -> Activations:
                return (layer_input,)

        class StrayLayer(MaglevLayer):
            def __init__(self) -> None:
                super().__init__(dense=Dense())
                # Belongs in the dense net; the split backward cannot see it.
                self.stray = torch.nn.Linear(3, 3)

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, 3])),)

        with self.assertRaisesRegex(ValueError, "outside its sparse and dense"):
            reject_parameters_outside_nets([StrayLayer()])


class _Bf16HandoffLayer(_SplitSpyLayer):
    """Hands the next stage one BF16 activation."""

    def __init__(self, first: bool) -> None:
        super().__init__()
        self._first = first
        self._spec = ActivationSpec(torch.Size([-1, 2]), torch.bfloat16)

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return () if self._first else (self._spec,)


def _handoff_stage(rank: int) -> StageWrapper:
    """Rank ``rank``'s stage of two one-layer stages handing off one BF16 tensor."""
    with patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=rank):
        return StageWrapper(
            MaglevModuleList(
                [_Bf16HandoffLayer(first=True), _Bf16HandoffLayer(first=False)]
            ),
            layers_per_stage=[1, 1],
            stage_size=1,
            process_groups=mock_groups(2),
        )


@patch("torchrec.distributed.maglev.stage.dist.isend")
@patch("torchrec.distributed.maglev.stage.InputDistDriver")
class StageHandoffTest(unittest.TestCase):
    def test_send_is_checked_against_the_output_specs(
        self, _driver: Any, isend: Any
    ) -> None:
        """A send that disagrees with the specs the next stage posts its receives
        from -- an FP32 tensor where the spec says BF16, as a wrapper's
        mixed-precision cast produces -- raises before anything is sent."""
        stage = _handoff_stage(0)
        self.assertFalse(stage.is_last)
        with self.assertRaisesRegex(ValueError, "expected dtype torch.bfloat16"):
            stage._start_send_act((torch.ones(3, 2),))
        with self.assertRaisesRegex(ValueError, "expected shape"):
            stage._start_send_act((torch.ones(3, 4, dtype=torch.bfloat16),))
        with self.assertRaisesRegex(ValueError, "expected 1 activation tensors"):
            stage._start_send_act(())
        isend.assert_not_called()

        activation = (torch.ones(3, 2, dtype=torch.bfloat16),)
        stage._start_send_act(activation)
        self.assertEqual(isend.call_count, 1)
        with self.assertRaisesRegex(ValueError, "activation send is still in flight"):
            stage._start_send_act(activation)
        stage._finish_send_act()
        stage._start_send_act(activation)
        self.assertEqual(isend.call_count, 2)

    def test_a_wait_needs_a_posted_receive(self, _driver: Any, _isend: Any) -> None:
        with self.assertRaisesRegex(
            ValueError, "an activation with no receive in flight"
        ):
            _handoff_stage(1)._wait_for_act()
        with self.assertRaisesRegex(ValueError, "a gradient with no receive in flight"):
            _handoff_stage(0)._wait_for_grad()

    def test_an_input_without_a_gradient_sends_zeros(
        self, _driver: Any, isend: Any
    ) -> None:
        stage = _handoff_stage(1)
        received = torch.ones(3, 2, dtype=torch.bfloat16, requires_grad=True)
        stage._start_send_grad((received,))
        (sent,) = isend.call_args.args
        torch.testing.assert_close(sent, torch.zeros_like(received), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "gradient send is still in flight"):
            stage._start_send_grad((received,))
        stage._finish_send_grad()
