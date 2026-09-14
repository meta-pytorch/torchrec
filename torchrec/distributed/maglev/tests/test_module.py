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
from typing import Any, cast, Dict, List, Tuple
from unittest.mock import call, MagicMock, patch

import torch
from torch.distributed.fsdp import FSDPModule
from torchrec.distributed.maglev.module import (
    activation_specs_from_tensors,
    Activations,
    ActivationSpec,
    cast_activations,
    get_structured_activations_layout_metadata_fields,
    MaglevLayer,
    MaglevModuleList,
    ObservedActivationSpecsMixin,
    StructuredActivations,
    StructuredActivationsLayout,
)
from torchrec.distributed.maglev.pipeline import MaglevPipelineBase
from torchrec.distributed.maglev.stage import pg_init, StageWrapper
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
    batch_size: int,
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
                batch_size=batch_size,
                activation_layout=activation_layout,
                num_float_features=num_float_features,
                device=device,
            )
        )
    return MaglevTestModel(layers, activation_layout)


class MaglevModuleListTest(unittest.TestCase):
    """Single-process checks of the authoring API (no distributed setup needed)."""

    def _model(self, num_layers: int = 4, batch_size: int = 4) -> MaglevTestModel:
        return _build_model(
            num_layers=num_layers,
            batch_size=batch_size,
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

    def test_postproc_returns_losses_and_output(self) -> None:
        """The model returns the usual (losses, output) pair."""
        model = self._model(batch_size=4)
        losses, output = model(self._inputs(batch_size=4))
        self.assertEqual(losses.shape, torch.Size([]))  # scalar, backward-able
        self.assertEqual(output.shape, torch.Size([4]))  # one prediction per row

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.pg_init")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    @patch("torchrec.distributed.maglev.stage.dist.get_world_size", return_value=1)
    def test_last_stage_loss_only_output_is_opt_in(
        self,
        _get_world_size: Any,
        _get_rank: Any,
        init_process_groups: Any,
        _input_dist_driver: Any,
    ) -> None:
        model = self._model(num_layers=1, batch_size=4)
        inputs = self._inputs(num_layers=1, batch_size=4)
        expected_loss, expected_output = model(inputs)
        process_group = MagicMock()
        init_process_groups.return_value = (
            [process_group],
            (process_group, process_group),
            [process_group],
            [process_group],
        )
        stage = StageWrapper(model, layers_per_stage=[1], stage_size=1)

        losses, output = stage(inputs)
        torch.testing.assert_close(losses, expected_loss)
        torch.testing.assert_close(output, expected_output)

        loss_only_stage = StageWrapper(
            model,
            layers_per_stage=[1],
            stage_size=1,
            loss_only_output=True,
        )
        outputs = loss_only_stage(inputs)

        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0], expected_loss)

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

    def test_empty_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MaglevModuleList([])

    def test_observed_activation_contract_preserves_integer_carriers(self) -> None:
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
            batch_size=8,
            in_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )

        expected = (
            ActivationSpec(torch.Size([8, 3]), torch.bfloat16),
            ActivationSpec(torch.Size([8]), torch.int64),
        )
        self.assertEqual(layer.in_activation_specs(), expected)
        self.assertEqual(layer.out_activation_specs(), expected)
        self.assertEqual(
            activation_specs_from_tensors(activations, 8, torch.bfloat16), expected
        )
        cast = cast_activations(activations, torch.bfloat16, expected)
        self.assertEqual(cast[0].dtype, torch.bfloat16)
        self.assertEqual(cast[1].dtype, torch.int64)

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

    def test_maglev_layer_defines_its_own_architecture(self) -> None:
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
                super().__init__()
                self.sparse = Sparse()
                self.dense = Dense()

            def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return ()

            def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
                return (ActivationSpec(torch.Size([2, 3])),)

            def forward(
                self, layer_input: Any, in_activations: Activations = ()
            ) -> Activations:
                return self.dense(self.sparse(layer_input), layer_input, in_activations)

        value = torch.ones(2, 3)
        (output,) = Layer()(value)
        torch.testing.assert_close(output, torch.full_like(value, 3))

    @patch("torchrec.distributed.maglev.stage.dist.new_group")
    def test_pg_init_builds_separate_nccl_and_gloo_cascades(
        self, new_group: Any
    ) -> None:
        pg_init(stage_size=2, num_stages=3)

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
