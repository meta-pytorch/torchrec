#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from typing import Any, List, Tuple

import torch
from torchrec.distributed.maglev.module import MaglevModuleList
from torchrec.distributed.maglev.pipeline import MaglevPipelineBase
from torchrec.distributed.maglev.stage import StageWrapper
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import MaglevTestLayer, MaglevTestModel
from torchrec.modules.embedding_configs import EmbeddingBagConfig

_WEIGHT_SEED = 100
_INPUT_SEED = 500


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
                num_float_features=num_float_features,
                device=device,
            )
        )
    return MaglevTestModel(layers)


def _run_correctness(
    rank: int,
    world_size: int,
    num_stages: int,
    layers_per_stage: int,
    ranks_per_stage: int,
    batch_size: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
    num_float_features: int,
    layer_dim: int,
) -> None:
    # CPU + gloo keeps the 8-rank repro portable (no 8-GPU requirement).
    with MultiProcessContext(rank=rank, world_size=world_size, backend="gloo"):
        device = torch.device("cpu")
        num_layers = num_stages * layers_per_stage
        partition = [layers_per_stage] * num_stages

        my_stage_index, _position = StageWrapper.locate_rank(ranks_per_stage, rank)

        # Deterministic per-layer inputs / label, identical on every rank (so the
        # two DP lanes of an HSD see the same data and the DP all-reduce is an
        # identity -- keeping the distributed grads exactly comparable to the
        # single-process reference).
        all_tables = [
            _make_tables(l, num_tables, num_embeddings, emb_dim)
            for l in range(num_layers)
        ]
        layer_inputs: List[Any] = [
            _make_input(
                all_tables[l], batch_size, num_float_features, _INPUT_SEED + l, device
            )
            for l in range(num_layers)
        ]
        # ---- distributed pipeline: forward + backward for this rank's stage ----
        # Every rank authors the same full model and hands it to StageWrapper,
        # which keeps only this rank's stage; the kept layers are the model's own
        # modules.
        dist_model = _build_model(
            num_layers,
            batch_size,
            num_tables,
            num_embeddings,
            emb_dim,
            num_float_features,
            layer_dim,
            device,
        )
        # The wrapper derives this rank's stage from stage_size and builds
        # every process group it and the pipeline need.
        stage = StageWrapper(
            model=dist_model,
            layers_per_stage=partition,
            stage_size=ranks_per_stage,
        )
        # Unwrapped: the numerics-transparent path this test compares against a
        # single-process reference. to() materializes the meta-authored layers.
        stage.to(device)
        # lr=0: weights are unchanged, grads remain populated for comparison.
        optimizer = torch.optim.SGD(stage.parameters(), lr=0.0, foreach=True)
        pipeline = MaglevPipelineBase(stage=stage, optimizer=optimizer)
        # One batch, the whole model's per-layer inputs; the stage pulls it,
        # partitions it and all-to-alls it to get its own. No label or criterion:
        # the model scores itself in postproc, against the target carried by the
        # last layer's own ModelInput.
        dist_loss = pipeline.progress(iter([layer_inputs]))

        # ---- single-process reference: the whole MaglevModuleList in-process ----
        ref_model = _build_model(
            num_layers,
            batch_size,
            num_tables,
            num_embeddings,
            emb_dim,
            num_float_features,
            layer_dim,
            device,
        )
        ref_loss, _ref_output = ref_model(layer_inputs)
        ref_loss.backward()

        # 1. Last-stage loss matches the reference.
        if pipeline.is_last:
            assert dist_loss is not None
            torch.testing.assert_close(dist_loss, ref_loss)

        # 2. This stage's gradients (after DP all-reduce) match the reference
        #    model's gradients for the same layers => forward + backward numerics
        #    of the pipelined execution match the standalone model.
        checked = 0
        for offset in range(layers_per_stage):
            layer_index = my_stage_index * layers_per_stage + offset
            ref_params = dict(ref_model[layer_index].named_parameters())
            for name, param in dist_model[layer_index].named_parameters():
                assert param.grad is not None, f"no grad for layer{layer_index}.{name}"
                ref_grad = ref_params[name].grad
                assert (
                    ref_grad is not None
                ), f"no reference grad for layer{layer_index}.{name}"
                torch.testing.assert_close(param.grad, ref_grad)
                checked += 1
        assert checked > 0, "no parameters were checked"


class MaglevPipelineTest(MultiProcessTestBase):
    def test_pipeline_matches_single_process(self) -> None:
        """Multi-layer stages: 8 layers partitioned into 4 stages of 2 layers each."""
        self._run_multi_process_test(
            callable=_run_correctness,
            world_size=8,
            num_stages=4,
            layers_per_stage=2,
            ranks_per_stage=2,
            batch_size=16,
            num_tables=6,
            num_embeddings=64,
            emb_dim=8,
            num_float_features=8,
            layer_dim=12,
        )


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
