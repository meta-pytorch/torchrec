#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple
from unittest.mock import call, MagicMock, patch

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torchrec.distributed.maglev.module import (
    activation_specs_from_tensors,
    Activations,
    ActivationSpec,
    cast_activations,
    check_activations,
    get_structured_activations_layout_metadata_fields,
    MaglevLayer,
    MaglevModuleList,
    ObservedActivationSpecsMixin,
    StructuredActivations,
    StructuredActivationsLayout,
)
from torchrec.distributed.maglev.pipeline import (
    Maglev1F1B,
    Maglev1F1BRecvAhead,
    MaglevPipelineBase,
    MaglevRail,
)
from torchrec.distributed.maglev.stage import (
    _RailLayerChain,
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
from torchrec.modules.embedding_configs import EmbeddingBagConfig

_WEIGHT_SEED = 100
_INPUT_SEED = 500


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


def _make_tables(
    layer_index: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
) -> List[EmbeddingBagConfig]:
    """This layer's feature partition: disjoint 1-feature tables, namespaced per layer."""
    return EmbeddingTablesConfig(
        num_unweighted_features=num_tables,
        num_weighted_features=0,
        embedding_feature_dim=emb_dim,
        base_row_size=num_embeddings,
    ).generate_tables(name_prefix=f"l{layer_index}_")[0]


def _make_input(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    num_float_features: int,
    seed: int,
    device: torch.device,
) -> ModelInput:
    """Deterministic ModelInput (float + sparse), identical on every rank.

    ``ModelInput.generate`` draws from both the ``torch`` and ``random`` RNGs and
    has no seed argument, so seed both here to make the inputs reproducible across
    ranks (required for the distributed-vs-single-process comparison).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    return ModelInput.generate(
        batch_size=batch_size,
        tables=tables,
        weighted_tables=[],
        num_float_features=num_float_features,
        device=device,
    )


def _build_model(
    num_layers: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
    num_float_features: int,
    layer_dim: int,
    device: torch.device,
) -> MaglevTestModel:
    """Author the model as a list of layers, with weights seeded by layer index."""
    activation_layout = StructuredActivationsLayout[MaglevTestActivations]()
    layers: List[MaglevTestLayer] = []
    for layer_index in range(num_layers):
        tables = _make_tables(layer_index, num_tables, num_embeddings, emb_dim)
        torch.manual_seed(_WEIGHT_SEED + layer_index)
        layers.append(
            MaglevTestLayer(
                tables=tables,
                layer_dim=layer_dim,
                is_first=(layer_index == 0),
                activation_layout=activation_layout,
                num_float_features=num_float_features,
                device=device,
            )
        )
    return MaglevTestModel(layers, activation_layout)


class _SplitSpyLayer(MaglevLayer):
    """A minimal Rail layer whose halves and splitter can be made misbehave."""

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


class MaglevModuleListTest(unittest.TestCase):
    """Single-process checks of the authoring API (no distributed setup needed)."""

    def _model(self, num_layers: int = 4) -> MaglevTestModel:
        return _build_model(
            num_layers=num_layers,
            num_tables=2,
            num_embeddings=16,
            emb_dim=4,
            num_float_features=4,
            layer_dim=6,
            device=torch.device("cpu"),
        )

    def _inputs(self, num_layers: int = 4, batch_size: int = 4) -> List[Any]:
        return [
            _make_input(
                _make_tables(l, 2, 16, 4),
                batch_size,
                4,
                _INPUT_SEED + l,
                torch.device("cpu"),
            )
            for l in range(num_layers)
        ]

    def test_standalone_matches_stage_by_stage(self) -> None:
        """Running the model one stage's layers at a time reproduces its forward."""
        model = self._model()
        inputs = self._inputs()

        losses, output = model(inputs)

        # [1, 3]: stage 0 owns layer 0, stage 1 owns layers 1..3.
        activations: Tuple[torch.Tensor, ...] = ()
        for stage_layers in ([0], [1, 2, 3]):
            for i in stage_layers:
                activations = model[i](inputs[i], activations)
        staged_losses, staged_output = model.postproc(activations, inputs[-1])

        torch.testing.assert_close(output, staged_output)
        torch.testing.assert_close(losses, staged_losses)

    def test_model_accepts_different_batch_sizes(self) -> None:
        """Batch-sized specs do not bind standalone execution to construction."""
        model = self._model()
        for batch_size in (2, 5):
            inputs = self._inputs(batch_size=batch_size)
            self.assertEqual(model.get_batch_size(inputs), batch_size)
            losses, output = model(inputs)
            self.assertEqual(losses.shape, torch.Size([]))
            self.assertEqual(output.shape, torch.Size([batch_size]))

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.MaglevProcessGroups.from_scratch")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    @patch("torchrec.distributed.maglev.stage.dist.get_world_size", return_value=1)
    def test_last_stage_loss_only_output_is_opt_in(
        self,
        _get_world_size: Any,
        _get_rank: Any,
        init_process_groups: Any,
        _input_dist_driver: Any,
    ) -> None:
        model = self._model(num_layers=1)
        inputs = self._inputs(num_layers=1, batch_size=4)
        expected_loss, expected_output = model(inputs)
        process_group = MagicMock()
        init_process_groups.return_value = MaglevProcessGroups(
            stage_ranks=((0,),),
            stage_pg=process_group,
            handoff_pgs=(process_group, process_group),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )
        stage = StageWrapper(model, layers_per_stage=[1], stage_size=1)

        losses, output = stage.module(inputs)
        torch.testing.assert_close(losses, expected_loss)
        torch.testing.assert_close(output, expected_output)

        loss_only_stage = StageWrapper(
            model,
            layers_per_stage=[1],
            stage_size=1,
            loss_only_output=True,
        )
        outputs = loss_only_stage.module(inputs)

        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0], expected_loss)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_stage_forward_and_backward_drain_an_input_dist_round(
        self,
        _get_rank: Any,
        input_dist_driver: Any,
    ) -> None:
        process_group = MagicMock()
        process_groups = MaglevProcessGroups(
            stage_ranks=((0,), (1,)),
            stage_pg=process_group,
            handoff_pgs=(process_group, process_group),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )
        model = self._model(num_layers=2)
        raw_batch = self._inputs(num_layers=2, batch_size=4)
        stage = StageWrapper(
            model,
            layers_per_stage=[1, 1],
            stage_size=1,
            process_groups=process_groups,
        )
        microbatch_inputs = [[raw_batch[0]], [raw_batch[0]]]
        driver = input_dist_driver.return_value
        driver.exchange.return_value.wait.return_value = microbatch_inputs
        gradient_hook = MagicMock(side_effect=lambda _grad: None)
        hook_handle = next(stage.module.parameters()).register_hook(gradient_hook)

        with (
            patch.object(stage, "finish_send_act"),
            patch.object(stage, "start_send_act"),
        ):
            loss = stage(raw_batch)

        torch.testing.assert_close(loss, torch.tensor(0.0))
        driver.exchange.assert_called_once()
        send_set = driver.exchange.call_args.args[0]
        self.assertIs(send_set[0][0], raw_batch[0])
        self.assertIs(send_set[1][0], raw_batch[1])
        received_grads = [[torch.ones(4, 6)], [torch.ones(4, 6)]]
        with (
            patch.object(stage, "start_recv_grad") as start_recv_grad,
            patch.object(stage, "wait_for_grad", side_effect=received_grads),
        ):
            loss.backward()

        self.assertEqual(start_recv_grad.call_count, 2)
        gradient_hook.assert_called_once()
        hook_handle.remove()

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_last_stage_loss_backward_drains_retained_graph(
        self,
        _get_rank: Any,
        input_dist_driver: Any,
    ) -> None:
        process_group = MagicMock()
        process_groups = MaglevProcessGroups(
            stage_ranks=((0,),),
            stage_pg=process_group,
            handoff_pgs=(process_group, process_group),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )
        model = self._model(num_layers=1)
        raw_batch = self._inputs(num_layers=1, batch_size=4)
        expected_loss, _output = model(raw_batch)
        stage = StageWrapper(
            model,
            layers_per_stage=[1],
            stage_size=1,
            process_groups=process_groups,
        )
        input_dist_driver.return_value.exchange.return_value.wait.return_value = [
            raw_batch
        ]

        loss = stage(raw_batch)
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()

        self.assertTrue(
            any(parameter.grad is not None for parameter in stage.module.parameters())
        )

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_rail_sparse_and_dense_phases_match_whole_batch(
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

        process_group = MagicMock()
        process_groups = MaglevProcessGroups(
            stage_ranks=((0,),),
            stage_pg=process_group,
            handoff_pgs=(process_group, process_group),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )
        model = self._model(num_layers=2)
        reference = self._model(num_layers=2)
        inputs = self._inputs(num_layers=2, batch_size=4)
        stage = StageWrapper(
            model,
            layers_per_stage=[2],
            stage_size=1,
            process_groups=process_groups,
            enable_rail=True,
        )
        wrapper = ForwardWrapper(stage.module)
        stage.module = wrapper

        state = stage.sparse_forward_global(inputs, num_microbatches=2)
        for microbatch in range(2):
            stage.compute_dense_forward_micro(
                state.dense_inputs[microbatch],
                (),
                microbatch,
                state.seams_for(microbatch),
            )
            stage.dense_backward_act_micro()
            stage.dense_backward_weight_micro()
        self.assertEqual(wrapper.phases, ["sparse", "dense", "dense"])
        self.assertTrue(
            all(
                seam.grad is not None
                for output in state.sparse_outputs
                if output is not None
                for seam in output.seams
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

    def test_base_postproc_must_be_overridden(self) -> None:
        """MaglevModuleList itself cannot score a model."""
        with self.assertRaises(NotImplementedError):
            MaglevModuleList.postproc(self._model(), (), None)

    def test_locate_rank_maps_every_rank_to_a_stage_slot(self) -> None:
        """Each rank resolves to its stage and its position within that HSD."""
        located = [StageWrapper.locate_rank(2, r) for r in range(6)]
        self.assertEqual(located, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_count_stages_requires_whole_stages(self) -> None:
        self.assertEqual(StageWrapper.count_stages(2, world_size=8), 4)
        with self.assertRaises(ValueError):
            StageWrapper.count_stages(3, world_size=8)
        with self.assertRaises(ValueError):
            StageWrapper.count_stages(0, world_size=8)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=2)
    def test_stage_uses_external_noncontiguous_process_groups(
        self,
        _get_rank: Any,
        _input_dist_driver: Any,
    ) -> None:
        stage_pg = MagicMock(name="stage_pg")
        handoff_pg = MagicMock(name="handoff_pg")
        cascade_pg = MagicMock(name="cascade_pg")
        cascade_gloo_pg = MagicMock(name="cascade_gloo_pg")
        process_groups = MaglevProcessGroups(
            stage_ranks=((0, 2), (1, 3)),
            stage_pg=stage_pg,
            handoff_pgs=(handoff_pg, handoff_pg),
            cascade_pg=cascade_pg,
            cascade_gloo_pg=cascade_gloo_pg,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )

        stage = StageWrapper(
            self._model(num_layers=2),
            layers_per_stage=[1, 1],
            stage_size=2,
            process_groups=process_groups,
        )

        self.assertEqual(stage.stage_index, 0)
        self.assertEqual(stage.position, 1)
        self.assertEqual(stage.neighbor_rank(1), 3)
        self.assertIs(stage.stage_pg, stage_pg)
        self.assertIs(stage.cascade_pg, cascade_pg)

    def test_empty_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MaglevModuleList([])

    def test_observed_activation_contract_depends_on_batch_size(self) -> None:
        class ObservedLayer(ObservedActivationSpecsMixin, MaglevLayer):
            def forward(
                self, layer_input: Any, in_activations: Activations = ()
            ) -> Activations:
                return in_activations

        layer = ObservedLayer()
        activations = (
            torch.zeros(2, 3, dtype=torch.float32),
            torch.zeros(2, dtype=torch.int64),
        )

        layer.set_activation_specs(
            activations,
            activations,
            batch_size=None,
            in_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )

        expected = (
            ActivationSpec(torch.Size([-1, 3]), torch.bfloat16),
            ActivationSpec(torch.Size([-1]), torch.int64),
        )
        self.assertEqual(layer.in_activation_specs(), expected)
        self.assertEqual(layer.out_activation_specs(), expected)
        self.assertEqual(
            activation_specs_from_tensors(activations, None, torch.bfloat16), expected
        )
        self.assertEqual(
            activation_specs_from_tensors(activations, 8, torch.bfloat16),
            (
                ActivationSpec(torch.Size([8, 3]), torch.bfloat16),
                ActivationSpec(torch.Size([8]), torch.int64),
            ),
        )
        self.assertEqual(expected[0].materialize_shape(8), torch.Size([8, 3]))
        self.assertEqual(expected[1].materialize_shape(5), torch.Size([5]))
        self.assertEqual(
            ActivationSpec(torch.Size([3, -1, 5])).materialize_shape(7),
            torch.Size([3, 7, 5]),
        )
        with self.assertRaisesRegex(ValueError, "at most one batch dimension"):
            ActivationSpec(torch.Size([-1, 3, -1]))
        with self.assertRaisesRegex(ValueError, "non-negative or -1"):
            ActivationSpec(torch.Size([3, -2]))
        cast = cast_activations(activations, torch.bfloat16, expected)
        check_activations(expected, cast, "dynamic boundary")
        self.assertEqual(cast[0].dtype, torch.bfloat16)
        self.assertEqual(cast[1].dtype, torch.int64)

        with self.assertRaisesRegex(ValueError, "expected shape"):
            check_activations(
                expected,
                (torch.zeros(2, 4), torch.zeros(2, dtype=torch.int64)),
                "dynamic boundary",
            )

    def test_structured_activations_round_trip(self) -> None:
        values = _TestStructuredActivations(
            tensor=torch.ones(2, 4),
            pair=(torch.ones(2, 5), torch.ones(2, 1)),
            sequence=[torch.ones(2), torch.ones(2, dtype=torch.int64)],
            mapping={
                "second": torch.ones(2, 6),
                "first": torch.ones(2, 7),
            },
            numbers=[3, 5],
            counts={"one": 1, "two": 2},
            version=2,
            label="static",
            wrapped=_WrappedTensor(torch.ones(2, 8)),
        )
        layout_type = StructuredActivationsLayout[_TestStructuredActivations]
        layout = layout_type(
            sequence=2,
            mapping=("first", "second"),
            numbers=[3, 5],
            counts={"one": 1, "two": 2},
            version=2,
            label="static",
            wrapped="wrapped",
        )

        activations = layout.pack(values)
        unpacked = layout.unpack(activations)

        self.assertIs(
            layout_type,
            StructuredActivationsLayout[_TestStructuredActivations],
        )
        self.assertEqual(len(activations), 8)
        self.assertIs(unpacked.tensor, values.tensor)
        self.assertIs(unpacked.pair[0], values.pair[0])
        self.assertIs(unpacked.sequence[0], values.sequence[0])
        self.assertEqual(tuple(unpacked.mapping), ("first", "second"))
        self.assertIs(unpacked.mapping["first"], values.mapping["first"])
        self.assertEqual(unpacked.numbers, [3, 5])
        self.assertEqual(unpacked.counts, {"one": 1, "two": 2})
        self.assertEqual(unpacked.version, 2)
        self.assertEqual(unpacked.label, "static")
        self.assertIs(unpacked.wrapped.tensor, values.wrapped.tensor)

    def test_structured_activations_require_exact_metadata_names(self) -> None:
        layout_type = StructuredActivationsLayout[_TestStructuredActivations]

        with self.assertRaisesRegex(TypeError, "sequence"):
            layout_type()
        with self.assertRaisesRegex(TypeError, "unknown"):
            StructuredActivationsLayout[_TensorStructuredActivations](unknown=1)

    def test_structured_activations_report_required_metadata_fields(self) -> None:
        self.assertEqual(
            get_structured_activations_layout_metadata_fields(
                _TestStructuredActivations
            ),
            (
                "sequence",
                "mapping",
                "numbers",
                "counts",
                "version",
                "label",
                "wrapped",
            ),
        )

    def test_structured_activations_reject_extra_tensors(self) -> None:
        layout = StructuredActivationsLayout[_TensorStructuredActivations]()
        values = _TensorStructuredActivations(tensor=torch.ones(2))

        with self.assertRaisesRegex(ValueError, "1 activation tensors"):
            layout.unpack(
                (*layout.pack(values), torch.zeros(1)),
            )

    def test_structured_activations_use_annotations_for_tensor_proxies(self) -> None:
        layout = StructuredActivationsLayout[_TensorStructuredActivations]()
        proxy = object()
        values = _TensorStructuredActivations(tensor=cast(torch.Tensor, proxy))

        self.assertIs(layout.pack(values)[0], proxy)

    def test_layer_class_layout_supports_fsdp2_mixin(self) -> None:
        layer = self._model(num_layers=1)[0]
        wrapped_class = type("FSDPMaglevTestLayer", (FSDPModule, layer.__class__), {})

        layer.__class__ = wrapped_class

        self.assertIsInstance(layer, FSDPModule)

    def test_pipeline_configures_fsdp2_on_each_backward(self) -> None:
        pipeline = cast(Any, object.__new__(MaglevPipelineBase))
        fsdp_module = MagicMock(spec=FSDPModule)
        pipeline._fsdp_modules = [fsdp_module]

        pipeline._configure_fsdp_backward(0, 2)
        pipeline._configure_fsdp_backward(1, 2)

        self.assertEqual(
            fsdp_module.set_requires_gradient_sync.call_args_list,
            [call(False, recurse=False), call(True, recurse=False)],
        )
        self.assertEqual(
            fsdp_module.set_is_last_backward.call_args_list,
            [call(False), call(True)],
        )

    def test_base_pipeline_delegates_full_round_to_stage(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 4
        stage.is_first = True
        stage.in_activation_specs.return_value = ()
        stage_loss = torch.tensor(2.0, requires_grad=True)
        stage.return_value = stage_loss
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        pipeline = MaglevPipelineBase(stage, optimizer)
        raw_batch = object()

        loss = pipeline.progress(iter([raw_batch]))

        self.assertEqual(stage.call_count, 1)
        self.assertIs(stage.call_args.args[0], raw_batch)
        self.assertIs(loss, stage_loss)
        torch.testing.assert_close(stage_loss.grad, torch.tensor(1.0))
        self.assertEqual(pipeline.microbatches_per_pass, 4)
        optimizer.zero_grad.assert_called_once_with()
        optimizer.step.assert_called_once_with()

    def test_1f1b_pairs_steady_state_handoffs(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 2
        stage.stage_index = 0
        stage.is_first = True
        stage.handoff_pg_mode = HandoffPGMode.SHARED
        stage.in_activation_specs.return_value = ()
        stage.take_inputs.return_value = [["first"], ["second"], ["third"]]
        stage.wait_for_act.return_value = ()
        output = (torch.ones(1, requires_grad=True),)
        grads = [torch.ones(1)]
        stage.compute_forward_micro.return_value = output
        stage.send_act_recv_grad.return_value = grads
        stage.compute_backward_micro.return_value = (None, ())
        stage.send_grad_recv_act.return_value = ()
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=3)
        pipeline.progress(iter([object()]))

        stage.start_send_act.assert_called_once_with(output)
        self.assertEqual(
            stage.compute_forward_micro.call_args_list,
            [
                call(["first"], (), 0),
                call(["second"], (), 1),
                call(["third"], (), 2),
            ],
        )
        stage.send_act_recv_grad.assert_has_calls([call(output), call(output)])
        stage.send_grad_recv_act.assert_called_once_with(
            (), recv_next=True, next_stage_input=["third"]
        )
        stage.compute_backward_micro.assert_has_calls(
            [
                call(grads),
                call(grads),
                call(stage.wait_for_grad.return_value),
            ]
        )
        optimizer.step.assert_called_once_with()

    def test_1f1b_split_uses_direction_specific_handoffs(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 2
        stage.stage_index = 0
        stage.is_first = True
        stage.handoff_pg_mode = HandoffPGMode.SPLIT
        stage.in_activation_specs.return_value = ()
        stage.take_inputs.return_value = [["first"], ["second"], ["third"]]
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=3)
        pipeline.progress(iter([object()]))

        self.assertEqual(stage.forward_micro.call_count, 3)
        self.assertEqual(stage.backward_micro.call_count, 3)
        self.assertEqual(
            stage.forward_micro.call_args_list,
            [
                call(["first"], 0),
                call(["second"], 1),
                call(["third"], 2),
            ],
        )
        self.assertEqual(
            stage.start_recv_act.call_args_list,
            [call(["first"]), call(["second"]), call(["third"])],
        )
        stage.send_act_recv_grad.assert_not_called()
        stage.send_grad_recv_act.assert_not_called()
        optimizer.step.assert_called_once_with()

    def test_1f1b_recv_ahead_preserves_split_schedule(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 2
        stage.stage_index = 0
        stage.handoff_pg_mode = HandoffPGMode.SPLIT
        stage.in_activation_specs.return_value = ()
        stage.take_inputs.return_value = [["first"], ["second"], ["third"]]
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        pipeline = Maglev1F1BRecvAhead(stage, optimizer, num_microbatches=3)
        pipeline.progress(iter([object()]))

        events = [
            entry[0]
            for entry in stage.method_calls
            if entry[0]
            in {
                "backward_micro",
                "forward_micro",
                "start_recv_act",
                "start_recv_grad",
            }
        ]
        self.assertEqual(
            events,
            [
                "start_recv_act",
                "start_recv_act",
                "forward_micro",
                "start_recv_grad",
                "forward_micro",
                "start_recv_act",
                "backward_micro",
                "start_recv_grad",
                "forward_micro",
                "start_recv_grad",
                "backward_micro",
                "backward_micro",
            ],
        )
        self.assertEqual(
            stage.start_recv_act.call_args_list,
            [call(["first"]), call(["second"]), call(["third"])],
        )

    def test_rail_requires_rail_enabled_stage(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.rail_enabled = False
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 1
        stage.stage_index = 0
        stage.is_first = True
        stage.in_activation_specs.return_value = ()
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with self.assertRaisesRegex(ValueError, "enable_rail=True"):
            MaglevRail(stage, optimizer, num_microbatches=1)

    def _rail_stage_mock(self, num_stages: int) -> MagicMock:
        stage = MagicMock(spec=StageWrapper)
        stage.rail_enabled = True
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = num_stages
        stage.stage_index = 0
        stage.is_first = True
        stage.handoff_pg_mode = HandoffPGMode.SPLIT
        stage.in_activation_specs.return_value = ()
        return stage

    def _rail_stage(self, layers: List[Any]) -> StageWrapper:
        """A single-stage Rail StageWrapper over hand-built layers."""
        process_group = MagicMock()
        return StageWrapper(
            MaglevTestModel(
                cast(List[MaglevTestLayer], layers),
                StructuredActivationsLayout[MaglevTestActivations](),
            ),
            layers_per_stage=[len(layers)],
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

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_rail_rejects_bad_sparse_output(self, _get_rank: Any, _driver: Any) -> None:
        """The sparse half must return one batch-major floating-point tensor."""

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
        ]
        for label, message, value in cases:
            with self.subTest(label):
                layer = _SplitSpyLayer(sparse=BadSparse(value))
                stage = self._rail_stage([layer])
                inputs = self._inputs(num_layers=1, batch_size=4)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    stage.sparse_forward_global(inputs, num_microbatches=2)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_rail_rejects_bad_microbatch_count(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        stage = self._rail_stage([_SplitSpyLayer()])
        inputs = self._inputs(num_layers=1, batch_size=4)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            stage.sparse_forward_global(inputs, num_microbatches=0)
        with self.assertRaisesRegex(ValueError, "non-empty microbatches"):
            stage.sparse_forward_global(inputs, num_microbatches=8)
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            stage.sparse_forward_global(inputs, num_microbatches=3)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_rail_rejects_wrong_split_count(self, _get_rank: Any, _driver: Any) -> None:
        """A layer whose splitter disagrees with the schedule is caught."""
        stage = self._rail_stage([_SplitSpyLayer(split_count=3)])
        inputs = self._inputs(num_layers=1, batch_size=4)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            stage.sparse_forward_global(inputs, num_microbatches=2)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_rail_weight_backward_without_queued_work(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        """The guard that turns a drain-loop bug into a loud failure."""
        stage = self._rail_stage([_SplitSpyLayer()])
        with self.assertRaisesRegex(ValueError, r"no\s+deferred weight work"):
            stage.dense_backward_weight_micro()

    def test_rail_rejects_parameters_outside_the_halves(self) -> None:
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
                # Belongs in the dense half; the split backward cannot see it.
                self.stray = torch.nn.Linear(3, 3)

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, 3])),)

        with self.assertRaisesRegex(ValueError, "outside its sparse and dense"):
            _RailLayerChain([StrayLayer()])

    def test_maglev_layer_composes_sparse_and_dense_halves(self) -> None:
        class Sparse(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value * 2

        class Dense(torch.nn.Module):
            def forward(
                self,
                pooled: torch.Tensor,
                layer_input: torch.Tensor,
                in_activations: Activations = (),
            ) -> Activations:
                previous = in_activations[0] if in_activations else 0
                return (pooled + layer_input + previous,)

        class Layer(MaglevLayer):
            def __init__(self) -> None:
                super().__init__(sparse=Sparse(), dense=Dense())

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([2, 3])),)

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([2, 3])),)

        value = torch.ones(2, 3)
        (output,) = Layer()(value, (4 * value,))
        torch.testing.assert_close(output, torch.full_like(value, 7))

    def test_maglev_layer_composes_dense_half_without_sparse_half(self) -> None:
        class Dense(torch.nn.Module):
            def forward(
                self,
                sparse_output: Any,
                layer_input: torch.Tensor,
                in_activations: Activations = (),
            ) -> Activations:
                if sparse_output is not None:
                    raise AssertionError("dense-only layer received sparse output")
                return (layer_input,)

        class Layer(MaglevLayer):
            def __init__(self) -> None:
                super().__init__(dense=Dense())

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, 3])),)

        value = torch.ones(2, 3)
        self.assertIs(Layer()(value)[0], value)

    def test_maglev_layer_default_forward_requires_dense_half(self) -> None:
        class Layer(MaglevLayer):
            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, 3])),)

        with self.assertRaisesRegex(ValueError, "without a dense half"):
            Layer()(torch.ones(2, 3))

    def test_maglev_layer_preserves_forward_override_without_halves(self) -> None:
        class LegacyLayer(MaglevLayer):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, 3])),)

            def forward(
                self, layer_input: Any, in_activations: Activations = ()
            ) -> Activations:
                return (self.linear(layer_input),)

        layer = LegacyLayer()

        self.assertEqual(layer(torch.ones(2, 3))[0].shape, torch.Size([2, 3]))
        self.assertIsNone(layer.sparse)
        self.assertIsNone(layer.dense)
        with self.assertRaisesRegex(ValueError, "without a dense half"):
            list(layer.dense_parameters())

    def test_maglev_test_layer_halves_partition_parameters(self) -> None:
        layer = cast(MaglevTestLayer, self._model(num_layers=2)[1])
        sparse = layer.sparse
        self.assertIsNotNone(sparse)
        assert sparse is not None
        dense = layer.require_dense()

        sparse_parameters = {id(parameter) for parameter in sparse.parameters()}
        dense_parameters = {id(parameter) for parameter in layer.dense_parameters()}
        all_parameters = {id(parameter) for parameter in layer.parameters()}

        self.assertTrue(sparse_parameters)
        self.assertTrue(dense_parameters)
        self.assertFalse(sparse_parameters & dense_parameters)
        self.assertEqual(sparse_parameters | dense_parameters, all_parameters)
        self.assertEqual(
            [id(parameter) for parameter in layer.dense_parameters()],
            [id(parameter) for parameter in dense.parameters()],
        )
        self.assertEqual(
            [id(parameter) for parameter in layer.parameters()],
            [
                *[id(parameter) for parameter in sparse.parameters()],
                *[id(parameter) for parameter in dense.parameters()],
            ],
        )

    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    @patch("torchrec.distributed.maglev.stage.dist.new_group")
    def test_from_scratch_split_builds_direction_specific_handoffs(
        self,
        new_group: Any,
        _get_rank: Any,
    ) -> None:
        groups = [MagicMock(name=f"group_{index}") for index in range(9)]
        new_group.side_effect = groups

        process_groups = MaglevProcessGroups.from_scratch(
            stage_size=2,
            num_stages=3,
            handoff_pg_mode=HandoffPGMode.SPLIT,
        )

        self.assertEqual(
            new_group.call_args_list,
            [
                call(ranks=[0, 1]),
                call(ranks=[2, 3]),
                call(ranks=[4, 5]),
                call(ranks=[0, 1, 2, 3, 4, 5]),
                call(ranks=[0, 1, 2, 3, 4, 5]),
                call(ranks=[0, 2, 4], backend=None),
                call(ranks=[1, 3, 5], backend=None),
                call(ranks=[0, 2, 4], backend="gloo"),
                call(ranks=[1, 3, 5], backend="gloo"),
            ],
        )
        self.assertIsNot(process_groups.handoff_pgs[0], process_groups.handoff_pgs[1])

    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=2)
    @patch("torchrec.distributed.maglev.stage.dist.new_group")
    def test_from_device_mesh_reuses_mesh_process_groups(
        self,
        new_group: Any,
        _get_rank: Any,
    ) -> None:
        activation_pg = MagicMock(name="activation_pg")
        gradient_pg = MagicMock(name="gradient_pg")
        cascade_gloo_pgs = [
            MagicMock(name="cascade_gloo_0"),
            MagicMock(name="cascade_gloo_1"),
        ]
        new_group.side_effect = [activation_pg, gradient_pg, *cascade_gloo_pgs]
        stage_pg = MagicMock(name="stage_pg")
        cascade_pg = MagicMock(name="cascade_pg")
        stage_mesh = MagicMock()
        stage_mesh.get_group.return_value = stage_pg
        pipeline_mesh = MagicMock()
        pipeline_mesh.get_group.return_value = cascade_pg
        device_mesh = MagicMock(spec=DeviceMesh)
        device_mesh.ndim = 2
        device_mesh.mesh_dim_names = ("dp", "pp")
        device_mesh.mesh = torch.tensor([[0, 2], [1, 3]])
        device_mesh.__getitem__.side_effect = {
            "dp": stage_mesh,
            "pp": pipeline_mesh,
        }.__getitem__

        process_groups = MaglevProcessGroups.from_device_mesh(
            cast(DeviceMesh, device_mesh),
            stage_mesh_dim_name="dp",
            pipeline_mesh_dim_name="pp",
            handoff_pg_mode=HandoffPGMode.SPLIT,
        )

        self.assertEqual(process_groups.stage_ranks, ((0, 1), (2, 3)))
        self.assertIs(process_groups.stage_pg, stage_pg)
        self.assertEqual(process_groups.handoff_pgs, (activation_pg, gradient_pg))
        self.assertIs(process_groups.cascade_pg, cascade_pg)
        self.assertIs(process_groups.cascade_gloo_pg, cascade_gloo_pgs[0])
        self.assertEqual(
            new_group.call_args_list,
            [
                call(ranks=[0, 1, 2, 3]),
                call(ranks=[0, 1, 2, 3]),
                call(ranks=[0, 2], backend="gloo"),
                call(ranks=[1, 3], backend="gloo"),
            ],
        )

    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    @patch("torchrec.distributed.maglev.stage.dist.new_group")
    def test_from_scratch_shared_reuses_one_handoff_pg(
        self,
        new_group: Any,
        _get_rank: Any,
    ) -> None:
        groups = [MagicMock(name=f"group_{index}") for index in range(8)]
        new_group.side_effect = groups

        process_groups = MaglevProcessGroups.from_scratch(
            stage_size=2,
            num_stages=3,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )

        self.assertEqual(
            new_group.call_args_list,
            [
                call(ranks=[0, 1]),
                call(ranks=[2, 3]),
                call(ranks=[4, 5]),
                call(ranks=[0, 1, 2, 3, 4, 5]),
                call(ranks=[0, 2, 4], backend=None),
                call(ranks=[1, 3, 5], backend=None),
                call(ranks=[0, 2, 4], backend="gloo"),
                call(ranks=[1, 3, 5], backend="gloo"),
            ],
        )
        self.assertIs(process_groups.handoff_pgs[0], process_groups.handoff_pgs[1])
