#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple
from unittest.mock import call, MagicMock, patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.checks import reject_unsplit_layers
from torchrec.distributed.maglev.data_parallel import DDPPassReduction
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
from torchrec.distributed.maglev.pipeline import Maglev1F1B, MaglevPipelineBase
from torchrec.distributed.maglev.process_groups import (
    _count_stages,
    MaglevProcessGroups,
)
from torchrec.distributed.maglev.stage import _LayerChain, StageWrapper
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import (
    MaglevTestActivations,
    MaglevTestLayer,
    MaglevTestModel,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig

_WEIGHT_SEED = 100
_INPUT_SEED = 500


def _test_deferred_ddp_gradient_matches_full_batch(
    rank: int,
    world_size: int,
) -> None:
    class PartiallyUsed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(2, 1, bias=False)
            self.unused = torch.nn.Linear(2, 1, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.used(value)

    with MultiProcessContext(rank, world_size, "gloo"):
        reference = PartiallyUsed()
        distributed = PartiallyUsed()
        with torch.no_grad():
            reference.used.weight.copy_(torch.tensor([[0.25, -0.5]]))
            distributed.load_state_dict(reference.state_dict())
        module = DistributedDataParallel(
            distributed,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
        stage = MagicMock(spec=StageWrapper)
        stage.module = module
        stage.num_stages = 2
        stage.stage_index = 1
        stage.is_first = False
        stage.in_activation_specs.return_value = (MagicMock(),)
        pipeline = Maglev1F1B(
            stage,
            MagicMock(spec=torch.optim.Optimizer),
            num_microbatches=2,
        )
        inputs = (
            (
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[3.0, -1.0]]),
            ),
            (
                torch.tensor([[-2.0, 4.0]]),
                torch.tensor([[0.5, 3.0]]),
            ),
        )
        targets = (
            (torch.tensor([[0.5]]), torch.tensor([[-1.0]])),
            (torch.tensor([[2.0]]), torch.tensor([[1.5]])),
        )

        with pipeline._reduction.accumulate():
            for value, target in zip(inputs[rank], targets[rank]):
                with pipeline._forward_no_sync():
                    loss = (module(value) - target).square().sum()
                loss.backward()
        pipeline._reduction.reduce()

        full_input = torch.cat(
            [value for rank_inputs in inputs for value in rank_inputs]
        )
        full_target = torch.cat(
            [target for rank_targets in targets for target in rank_targets]
        )
        reference_loss = (
            reference(full_input) - full_target
        ).square().sum() / world_size
        reference_loss.backward()

        gradient = module.module.used.weight.grad
        torch.testing.assert_close(gradient, reference.used.weight.grad)
        gathered = [torch.empty_like(gradient) for _ in range(world_size)]
        dist.all_gather(gathered, gradient)
        for rank_gradient in gathered:
            torch.testing.assert_close(rank_gradient, reference.used.weight.grad)
        assert module.module.unused.weight.grad is None


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
    def test_last_stage_loss_only_output_is_opt_in(
        self,
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
            handoff_pgs=(MagicMock(), MagicMock()),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
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
    def test_stage_parameter_names_keep_global_layer_indices(
        self,
        _input_dist_driver: Any,
    ) -> None:
        process_group = MagicMock()
        process_groups = MaglevProcessGroups(
            stage_ranks=((0,), (1,)),
            stage_pg=process_group,
            handoff_pgs=(MagicMock(), MagicMock()),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
        )
        with patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0):
            first_stage = StageWrapper(
                self._model(num_layers=2),
                layers_per_stage=[1, 1],
                stage_size=1,
                process_groups=process_groups,
            )
        with patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=1):
            second_stage = StageWrapper(
                self._model(num_layers=2),
                layers_per_stage=[1, 1],
                stage_size=1,
                process_groups=process_groups,
            )

        first_names = {name for name, _ in first_stage.module.named_parameters()}
        second_names = {name for name, _ in second_stage.module.named_parameters()}
        self.assertTrue(first_names)
        self.assertTrue(second_names)
        self.assertTrue(all(name.startswith("layers.0.") for name in first_names))
        self.assertTrue(all(name.startswith("layers.1.") for name in second_names))
        self.assertTrue(first_names.isdisjoint(second_names))

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_1f1b_phase_ops_match_the_whole_microbatch(
        self,
        _get_rank: Any,
        _input_dist_driver: Any,
    ) -> None:
        """Seams change no gradient: the phase ops give the whole microbatch's
        gradients bit for bit, over two microbatches in 1F1B warmup order."""
        process_group = MagicMock()
        process_groups = MaglevProcessGroups(
            stage_ranks=((0,),),
            stage_pg=process_group,
            handoff_pgs=(MagicMock(), MagicMock()),
            cascade_pg=process_group,
            cascade_gloo_pg=process_group,
        )
        whole = StageWrapper(
            self._model(num_layers=2), [2], 1, process_groups=process_groups
        )
        phased = StageWrapper(
            self._model(num_layers=2), [2], 1, process_groups=process_groups
        )
        microbatches = [
            self._inputs(num_layers=2, batch_size=4),
            self._inputs(num_layers=2, batch_size=3),
        ]
        whole_losses: List[torch.Tensor] = []
        for microbatch_id, stage_input in enumerate(microbatches):
            # The plain call runs each layer's sparse then dense net in one
            # graph, with no detached seam between them.
            whole_losses.append(whole.module(stage_input, ())[0])
            seams = phased.sparse_forward_micro(stage_input, microbatch_id)
            phased._compute_dense_forward_micro(stage_input, (), microbatch_id, seams)
        for whole_loss in whole_losses:
            whole_loss.backward()
            phased_loss, _ = phased._compute_dense_backward_micro([])
            phased.sparse_backward_micro()
            assert phased_loss is not None
            self.assertTrue(torch.equal(whole_loss, phased_loss))
        phased_parameters = dict(phased.module.named_parameters())
        for name, parameter in whole.module.named_parameters():
            expected = parameter.grad
            actual = phased_parameters[name].grad
            assert expected is not None and actual is not None, name
            self.assertTrue(torch.equal(actual, expected), name)

    def test_base_postproc_must_be_overridden(self) -> None:
        """MaglevModuleList itself cannot score a model."""
        with self.assertRaises(NotImplementedError):
            MaglevModuleList.postproc(self._model(), (), None)

    def test_locate_rank_maps_every_rank_to_a_stage_slot(self) -> None:
        """Each rank resolves to its stage and its position within that HSD."""
        stage_ranks = ((0, 1), (2, 3), (4, 5))
        located = [MaglevProcessGroups.locate_rank(stage_ranks, r) for r in range(6)]
        self.assertEqual(located, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_count_stages_requires_whole_stages(self) -> None:
        self.assertEqual(_count_stages(2, world_size=8), 4)
        with self.assertRaises(ValueError):
            _count_stages(3, world_size=8)
        with self.assertRaises(ValueError):
            _count_stages(0, world_size=8)

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=2)
    def test_stage_uses_external_noncontiguous_process_groups(
        self,
        _get_rank: Any,
        _input_dist_driver: Any,
    ) -> None:
        stage_pg = MagicMock(name="stage_pg")
        cascade_pg = MagicMock(name="cascade_pg")
        cascade_gloo_pg = MagicMock(name="cascade_gloo_pg")
        process_groups = MaglevProcessGroups(
            stage_ranks=((0, 2), (1, 3)),
            stage_pg=stage_pg,
            handoff_pgs=(MagicMock(), MagicMock()),
            cascade_pg=cascade_pg,
            cascade_gloo_pg=cascade_gloo_pg,
        )

        stage = StageWrapper(
            self._model(num_layers=2),
            layers_per_stage=[1, 1],
            stage_size=2,
            process_groups=process_groups,
        )

        self.assertEqual(stage.stage_index, 0)
        self.assertEqual(stage.position, 1)
        self.assertEqual(stage._neighbor_rank(1), 3)
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

    def test_1f1b_reduces_once_after_the_pass(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 2
        stage.stage_index = 0
        stage.is_first = True
        stage.in_activation_specs.return_value = ()
        stage.take_inputs.return_value = [["first"], ["second"]]
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=2)
        reduction = MagicMock()
        pipeline._reduction = reduction
        order = MagicMock()
        order.attach_mock(optimizer.zero_grad, "zero_grad")
        order.attach_mock(reduction.accumulate, "accumulate")
        order.attach_mock(reduction.no_sync, "no_sync")
        order.attach_mock(stage.sparse_forward_micro, "sparse_forward_micro")
        order.attach_mock(stage.dense_forward_micro, "dense_forward_micro")
        order.attach_mock(stage.dense_backward_micro, "dense_backward_micro")
        order.attach_mock(stage.sparse_backward_micro, "sparse_backward_micro")
        order.attach_mock(stage.drain_sends, "drain_sends")
        order.attach_mock(reduction.reduce, "reduce")
        order.attach_mock(optimizer.step, "step")

        self.assertIsNone(pipeline.progress(iter([])))

        events = [
            entry[0]
            for entry in order.mock_calls
            if not entry[0].endswith(("__enter__", "__exit__"))
        ]
        self.assertEqual(
            events,
            [
                "zero_grad",
                "accumulate",
                # Stage 0 of 2 warms up one microbatch ahead.
                "no_sync",
                "sparse_forward_micro",
                "dense_forward_micro",
                "no_sync",
                "sparse_forward_micro",
                "dense_forward_micro",
                "dense_backward_micro",
                "sparse_backward_micro",
                "dense_backward_micro",
                "sparse_backward_micro",
                "drain_sends",
                "reduce",
                "step",
            ],
        )
        pipeline.close()
        reduction.close.assert_called_once_with()

    def test_1f1b_ddp_reduces_after_the_final_backward(self) -> None:
        class PartiallyUsed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.used = torch.nn.Linear(2, 1, bias=False)
                self.unused = torch.nn.Linear(2, 1, bias=False)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.used(value).sum()

        with tempfile.TemporaryDirectory() as temp_dir:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{temp_dir}/store",
                rank=0,
                world_size=1,
            )
            try:
                module = DistributedDataParallel(
                    PartiallyUsed(),
                    find_unused_parameters=True,
                    gradient_as_bucket_view=True,
                    static_graph=False,
                )
                stage = MagicMock(spec=StageWrapper)
                stage.module = module
                stage.num_stages = 2
                stage.stage_index = 1
                stage.is_first = False
                stage.in_activation_specs.return_value = (MagicMock(),)
                pipeline = Maglev1F1B(
                    stage,
                    MagicMock(spec=torch.optim.Optimizer),
                    num_microbatches=2,
                )
                self.assertIsInstance(pipeline._reduction, DDPPassReduction)

                with pipeline._reduction.accumulate():
                    for _ in range(2):
                        with pipeline._forward_no_sync():
                            loss = module(torch.ones(1, 2))
                        loss.backward()
                pipeline._reduction.reduce()

                torch.testing.assert_close(
                    module.module.used.weight.grad,
                    torch.full((1, 2), 2.0),
                )
                self.assertIsNone(module.module.unused.weight.grad)
                self.assertTrue(module.require_forward_param_sync)
                pipeline.close()
            finally:
                dist.destroy_process_group()

    def test_base_pipeline_is_not_a_schedule(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 1
        stage.is_first = True
        stage.in_activation_specs.return_value = ()
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with self.assertRaisesRegex(TypeError, "not a schedule"):
            MaglevPipelineBase(stage, optimizer, num_microbatches=1)

    def test_1f1b_split_uses_direction_specific_handoffs(self) -> None:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        stage.num_stages = 2
        stage.stage_index = 0
        stage.is_first = True
        stage.in_activation_specs.return_value = ()
        stage.take_inputs.return_value = [["first"], ["second"], ["third"]]
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=3)
        pipeline.progress(iter([object()]))

        seams = stage.sparse_forward_micro.return_value
        self.assertEqual(
            stage.dense_forward_micro.call_args_list,
            [
                call(["first"], 0, seams),
                call(["second"], 1, seams),
                call(["third"], 2, seams),
            ],
        )
        self.assertEqual(stage.sparse_forward_micro.call_count, 3)
        self.assertEqual(stage.dense_backward_micro.call_count, 3)
        self.assertEqual(stage.sparse_backward_micro.call_count, 3)
        self.assertEqual(
            stage.start_recv_act.call_args_list,
            [call(["first"]), call(["second"]), call(["third"])],
        )
        # Stage 0 of 2: one warmup forward, two steady pairs, one cooldown
        # backward; each receive is posted right before its op.
        events = [
            entry[0]
            for entry in stage.method_calls
            if entry[0]
            in {
                "start_recv_act",
                "sparse_forward_micro",
                "dense_forward_micro",
                "start_recv_grad",
                "dense_backward_micro",
                "sparse_backward_micro",
            }
        ]
        forward = ["start_recv_act", "sparse_forward_micro", "dense_forward_micro"]
        backward = ["start_recv_grad", "dense_backward_micro", "sparse_backward_micro"]
        self.assertEqual(
            events, forward + forward + backward + forward + backward + backward
        )
        optimizer.step.assert_called_once_with()

    def test_maglev_layer_composes_sparse_and_dense_nets(self) -> None:
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

    def test_maglev_layer_composes_dense_net_without_sparse_net(self) -> None:
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

    def test_maglev_layer_default_forward_requires_dense_net(self) -> None:
        class Layer(MaglevLayer):
            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([-1, 3])),)

        with self.assertRaisesRegex(ValueError, "without a dense net"):
            Layer()(torch.ones(2, 3))

    def test_maglev_layer_preserves_forward_override_without_nets(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "without a dense net"):
            list(layer.dense_parameters())
        with self.assertRaisesRegex(
            ValueError, "LegacyLayer at index 0 has no dense net"
        ):
            reject_unsplit_layers([layer])

    def test_maglev_test_layer_nets_partition_parameters(self) -> None:
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

    @patch("torchrec.distributed.maglev.process_groups.dist.get_rank", return_value=0)
    @patch("torchrec.distributed.maglev.process_groups.dist.new_group")
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

    @patch("torchrec.distributed.maglev.process_groups.dist.get_rank", return_value=2)
    @patch("torchrec.distributed.maglev.process_groups.dist.new_group")
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

    def test_one_handoff_group_for_both_directions_is_refused_across_stages(
        self,
    ) -> None:
        group = MagicMock()
        with self.assertRaisesRegex(ValueError, "two distinct process groups"):
            MaglevProcessGroups(
                stage_ranks=((0,), (1,)),
                stage_pg=group,
                handoff_pgs=(group, group),
                cascade_pg=group,
                cascade_gloo_pg=group,
            )
        one_stage = MaglevProcessGroups(
            stage_ranks=((0,),),
            stage_pg=group,
            handoff_pgs=(group, group),
            cascade_pg=group,
            cascade_gloo_pg=group,
        )
        self.assertEqual(one_stage.num_stages, 1)


class _RecordingSparse(torch.nn.Module):
    def __init__(self, name: str, events: List[str]) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self._name = name
        self._events = events

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        self._events.append(f"sparse{self._name}")
        return self.projection(layer_input)


class _RecordingDense(torch.nn.Module):
    def __init__(self, name: str, events: List[str]) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self._name = name
        self._events = events

    def forward(
        self,
        sparse_output: torch.Tensor,
        layer_input: torch.Tensor,
        in_activations: Activations = (),
    ) -> Activations:
        self._events.append(f"dense{self._name}")
        value = sparse_output + in_activations[0] if in_activations else sparse_output
        return (self.projection(value).tanh(),)


class _SplitLayer(MaglevLayer):
    def __init__(self, name: str, events: List[str], body: bool = False) -> None:
        super().__init__(
            sparse=_RecordingSparse(name, events),
            dense=_RecordingDense(name, events),
        )
        self.body: Optional[torch.nn.Module] = torch.nn.Linear(2, 2) if body else None
        self._name = name
        self._events = events

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2])),)

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2])),)


class _ForwardLayer(_SplitLayer):
    """Overrides ``forward``, which no schedule calls."""

    def forward(
        self, layer_input: Any, in_activations: Activations = ()
    ) -> Activations:
        return super().forward(layer_input, in_activations)


class LayerChainTest(unittest.TestCase):
    """A stage's layers on their nets: sparse nets first, then the dense chain."""

    def test_split_layers_run_every_sparse_net_first(self) -> None:
        events: List[str] = []
        torch.manual_seed(7)
        layers = [_SplitLayer("0", events), _SplitLayer("1", events)]
        chain = _LayerChain(layers)
        value = torch.randn(3, 2)

        (out,) = chain([value, value])

        self.assertEqual(events, ["sparse0", "sparse1", "dense0", "dense1"])
        expected = layers[1](value, layers[0](value))[0]
        torch.testing.assert_close(out, expected)

    # StageWrapper refuses before it builds a process group, so these build no
    # group: a refusal that came later would fail on the missing one instead.

    def test_a_layer_that_overrides_forward_is_refused(self) -> None:
        events: List[str] = []
        layers = [_SplitLayer("0", events), _ForwardLayer("1", events)]
        with self.assertRaisesRegex(
            ValueError,
            r"_ForwardLayer at index 1 overrides forward \(_ForwardLayer\.forward\)\. "
            "MaglevPP runs each layer as its sparse net and its dense net",
        ):
            StageWrapper(_SeamModel(layers), [2], 1)
        self.assertEqual(events, [])

    def test_a_layer_without_a_dense_net_is_refused(self) -> None:
        events: List[str] = []
        sparse_only = _SplitLayer("0", events)
        sparse_only.dense = None
        with self.assertRaisesRegex(ValueError, "at index 0 has no dense net"):
            StageWrapper(_SeamModel([sparse_only]), [1], 1)

    def test_parameters_outside_a_split_layers_nets_are_refused(self) -> None:
        events: List[str] = []
        with self.assertRaisesRegex(ValueError, "outside its sparse and dense nets"):
            StageWrapper(_SeamModel([_SplitLayer("0", events, body=True)]), [1], 1)


class _TupleSparse(torch.nn.Module):
    """A float output that trains, integer ids, a frozen output, and a float
    output the dense net never reads."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.unread = torch.nn.Linear(2, 2)

    def forward(self, layer_input: torch.Tensor) -> Any:
        ids = torch.arange(layer_input.shape[0]).unsqueeze(1).expand(-1, 2)
        return (
            self.projection(layer_input),
            ids,
            layer_input * 2,
            self.unread(layer_input),
        )


class _TupleDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(
        self,
        sparse_output: Any,
        layer_input: torch.Tensor,
        in_activations: Activations = (),
    ) -> Activations:
        trained, ids, frozen, _unread = sparse_output
        return (self.projection(trained + ids.float() + frozen).tanh(),)


class _TupleLayer(MaglevLayer):
    def __init__(self, sparse: Optional[torch.nn.Module] = None) -> None:
        super().__init__(
            sparse=sparse if sparse is not None else _TupleSparse(),
            dense=_TupleDense(),
        )

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2])),)


class _SharedSparse(torch.nn.Module):
    """Per-row output, plus a whole table every row of the microbatch reads."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.table = torch.nn.Parameter(torch.randn(5, 2))

    def forward(self, layer_input: torch.Tensor) -> Any:
        return (self.projection(layer_input), self.table * 2)


class _SharedDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(
        self,
        sparse_output: Any,
        layer_input: torch.Tensor,
        in_activations: Activations = (),
    ) -> Activations:
        rows, table = sparse_output
        return (self.projection(rows + table.sum(0)).tanh(),)


class _SharedLayer(MaglevLayer):
    def __init__(self) -> None:
        super().__init__(sparse=_SharedSparse(), dense=_SharedDense())

    def shared_sparse_outputs(self) -> Tuple[int, ...]:
        return (1,)

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 2])),)


class _DictSparse(torch.nn.Module):
    def forward(self, layer_input: torch.Tensor) -> Any:
        return {"pooled": layer_input}


class _SeamModel(MaglevModuleList):
    def preproc(self, model_input: torch.Tensor) -> List[torch.Tensor]:
        return [model_input] * len(self)

    def get_batch_size(self, layer_inputs: Any) -> int:
        return layer_inputs[0].shape[0]

    def postproc(
        self, activations: Activations, layer_input: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return activations[0].square().sum(), activations[0].detach()


class _CallRecorder(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.calls: List[str] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("sparse_only"):
            self.calls.append("sparse")
        elif kwargs.get("sparse_outputs") is not None:
            self.calls.append("dense")
        return self.module(*args, **kwargs)


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class StagePhaseTest(unittest.TestCase):
    """One microbatch on a stage's sparse and dense phases."""

    @staticmethod
    def _stage(layer: MaglevLayer) -> StageWrapper:
        torch.manual_seed(11)
        process_group = MagicMock()
        return StageWrapper(
            _SeamModel([layer]),
            [1],
            1,
            process_groups=MaglevProcessGroups(
                stage_ranks=((0,),),
                stage_pg=process_group,
                handoff_pgs=(MagicMock(), MagicMock()),
                cascade_pg=process_group,
                cascade_gloo_pg=process_group,
            ),
        )

    def test_seams_are_detached_leaves_that_follow_their_source(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        stage = self._stage(_TupleLayer())
        value = torch.randn(3, 2)

        seams = stage.sparse_forward_micro([value], 0)

        assert seams is not None
        trained, ids, frozen, unread = cast(Tuple[torch.Tensor, ...], seams[0])
        for leaf in (trained, ids, frozen, unread):
            self.assertTrue(leaf.is_leaf)
            self.assertIsNone(leaf.grad_fn)
        self.assertTrue(trained.requires_grad)
        self.assertEqual(ids.dtype, torch.int64)
        self.assertFalse(ids.requires_grad)
        self.assertFalse(frozen.requires_grad)
        self.assertTrue(unread.requires_grad)

    def test_sparse_backward_skips_a_seam_without_gradient(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        # A seam the dense net never reads gets no gradient, and its sparse
        # output gets no backward: zero-filling it would still step an optimizer
        # (momentum, say) that a single backward through both nets never runs.
        layer = _TupleLayer()
        stage = self._stage(layer)
        value = torch.randn(3, 2)
        seams = stage.sparse_forward_micro([value], 0)
        stage._compute_dense_forward_micro([value], (), 0, seams)
        stage._compute_dense_backward_micro([])
        assert seams is not None
        trained, _ids, _frozen, unread = cast(Tuple[torch.Tensor, ...], seams[0])
        self.assertIsNotNone(trained.grad)
        self.assertIsNone(unread.grad)

        stage.sparse_backward_micro()

        sparse = cast(_TupleSparse, layer.sparse)
        self.assertIsNotNone(sparse.projection.weight.grad)
        self.assertIsNone(sparse.unread.weight.grad)
        self.assertIsNone(trained.grad)

        reference = _TupleLayer()
        reference.load_state_dict(layer.state_dict())
        loss, _ = _SeamModel([reference])(value)
        loss.backward()
        for (name, actual), expected in zip(
            layer.named_parameters(), reference.parameters()
        ):
            if expected.grad is None:
                self.assertIsNone(actual.grad, name)
            else:
                assert actual.grad is not None, name
                self.assertTrue(torch.equal(actual.grad, expected.grad), name)

    def test_split_stage_calls_its_module_per_phase(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        stage = self._stage(_TupleLayer())
        recorder = _CallRecorder(stage.module)
        stage.module = recorder
        value = torch.randn(3, 2)

        seams = stage.sparse_forward_micro([value], 0)
        stage._compute_dense_forward_micro([value], (), 0, seams)
        # Past the wrapper, straight to the stage's authored layers.
        stage.sparse_forward_micro([value], 1, unwrapped=True)

        self.assertEqual(recorder.calls, ["sparse", "dense"])

    def test_a_non_tensor_sparse_output_is_refused(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        stage = self._stage(_TupleLayer(sparse=_DictSparse()))
        with self.assertRaisesRegex(
            TypeError, "_TupleLayer sparse net returned dict; the stage hands"
        ):
            stage.sparse_forward_micro([torch.randn(3, 2)], 0)

    def test_shared_position_is_one_uncut_leaf_and_matches_whole(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        # Per microbatch, a shared position is that microbatch's whole tensor:
        # one leaf, uncut, with the whole forward's gradients bit for bit.
        layer = _SharedLayer()
        stage = self._stage(layer)
        reference = _SharedLayer()
        reference.load_state_dict(layer.state_dict())
        value = torch.randn(3, 2)

        seams = stage.sparse_forward_micro([value], 0)
        stage._compute_dense_forward_micro([value], (), 0, seams)
        stage._compute_dense_backward_micro([])
        stage.sparse_backward_micro()

        assert seams is not None
        rows, table = cast(Tuple[torch.Tensor, ...], seams[0])
        self.assertEqual(rows.shape, torch.Size([3, 2]))
        self.assertEqual(table.shape, torch.Size([5, 2]))
        loss, _ = _SeamModel([reference])(value)
        loss.backward()
        for (name, actual), expected in zip(
            layer.named_parameters(), reference.parameters()
        ):
            assert actual.grad is not None and expected.grad is not None, name
            self.assertTrue(torch.equal(actual.grad, expected.grad), name)

    def test_1f1b_drains_the_pass_losses(self, _get_rank: Any, driver: Any) -> None:
        stage = self._stage(_TupleLayer())
        value = torch.randn(4, 2)
        driver.return_value.take.return_value = [[value[:2]], [value[2:]]]
        optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=2)

        self.assertIsNone(pipeline.progress(iter([])))

        self.assertEqual(stage.take_pass_losses(), [])

    def test_1f1b_failed_pass_poisons_the_stage(
        self, _get_rank: Any, driver: Any
    ) -> None:
        stage = self._stage(_TupleLayer())
        optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=2)
        # An exhausted dataloader ends training; it does not poison the stage.
        driver.return_value.take.side_effect = StopIteration
        with self.assertRaises(StopIteration):
            pipeline.progress(iter([]))
        driver.return_value.take.side_effect = None
        driver.return_value.take.return_value = [[torch.randn(2, 2)]] * 2
        pipeline.progress(iter([]))
        # A pass that raises does.
        with patch.object(
            stage, "dense_backward_micro", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                pipeline.progress(iter([]))
        with self.assertRaisesRegex(RuntimeError, "out of step"):
            pipeline.progress(iter([]))

    def test_abandon_pass_drops_pending_sparse_forwards(
        self, _get_rank: Any, _driver: Any
    ) -> None:
        stage = self._stage(_TupleLayer())
        stage.sparse_forward_micro([torch.randn(3, 2)], 0)
        stage.abandon_pass()
        with self.assertRaises(IndexError):
            stage.sparse_backward_micro()


class Maglev1F1BTrackerTest(unittest.TestCase):
    """A stage calls its module per phase, so trackers would count each
    microbatch more than once."""

    @staticmethod
    def _stage() -> MagicMock:
        stage = MagicMock(spec=StageWrapper)
        stage.module = torch.nn.Linear(1, 1)
        vars(stage.module)["model_trackers"] = {"tracker": object()}
        stage.num_stages = 1
        stage.stage_index = 0
        stage.is_first = True
        stage.in_activation_specs.return_value = ()
        stage.sparse_net_buffers.return_value = []
        return stage

    def test_1f1b_refuses_model_trackers_on_a_split_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "model trackers"):
            Maglev1F1B(self._stage(), MagicMock(spec=torch.optim.Optimizer), 1)


class MaglevDDPNumericsTest(MultiProcessTestBase):
    def test_deferred_ddp_gradient_matches_full_batch(self) -> None:
        self._run_multi_process_test(
            callable=_test_deferred_ddp_gradient_matches_full_batch,
            world_size=2,
        )
