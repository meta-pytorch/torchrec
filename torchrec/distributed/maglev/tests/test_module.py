#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from typing import List

import torch
import torch.nn as nn
from torchrec.distributed.maglev.module import MaglevModuleList
from torchrec.distributed.maglev.pipeline import MaglevPipeline
from torchrec.distributed.maglev.stage import build_stage_process_groups, StageWrapper
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import MaglevTestStage
from torchrec.modules.embedding_configs import EmbeddingBagConfig

_WEIGHT_SEED = 100
_INPUT_SEED = 500
_LABEL_SEED = 900


def _make_tables(
    stage_index: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
) -> List[EmbeddingBagConfig]:
    """This stage's feature partition: disjoint 1-feature tables, namespaced per stage."""
    return EmbeddingTablesConfig(
        num_unweighted_features=num_tables,
        num_weighted_features=0,
        embedding_feature_dim=emb_dim,
        base_row_size=num_embeddings,
    ).generate_tables(name_prefix=f"s{stage_index}_")[0]


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


def _build_stage(
    stage_index: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
    num_float_features: int,
    stage_dim: int,
    device: torch.device,
) -> MaglevTestStage:
    """Build a stage with deterministic weights (seeded by stage index)."""
    tables = _make_tables(stage_index, num_tables, num_embeddings, emb_dim)
    torch.manual_seed(_WEIGHT_SEED + stage_index)
    return MaglevTestStage(
        tables=tables,
        stage_dim=stage_dim,
        is_first=(stage_index == 0),
        num_float_features=num_float_features,
        device=device,
    )


def _run_correctness(
    rank: int,
    world_size: int,
    num_stages: int,
    ranks_per_stage: int,
    batch_size: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
    num_float_features: int,
    stage_dim: int,
) -> None:
    # CPU + gloo keeps the 8-rank repro portable (no 8-GPU requirement).
    with MultiProcessContext(rank=rank, world_size=world_size, backend="gloo"):
        device = torch.device("cpu")
        criterion = nn.MSELoss()

        stage_ranks: List[List[int]] = [
            list(range(s * ranks_per_stage, (s + 1) * ranks_per_stage))
            for s in range(num_stages)
        ]
        # Collective: every rank builds every stage's process group.
        stage_pgs = build_stage_process_groups(stage_ranks)
        my_stage_index = rank // ranks_per_stage

        # Deterministic per-stage inputs / label, identical on every rank (so the
        # two DP lanes of an HSD see the same data and the DP all-reduce is an
        # identity -- keeping the distributed grads exactly comparable to the
        # single-process reference).
        all_tables = [
            _make_tables(s, num_tables, num_embeddings, emb_dim)
            for s in range(num_stages)
        ]
        stage_inputs = [
            _make_input(
                all_tables[s], batch_size, num_float_features, _INPUT_SEED + s, device
            )
            for s in range(num_stages)
        ]
        torch.manual_seed(_LABEL_SEED)
        label = torch.randn(batch_size, stage_dim, device=device)

        # ---- distributed pipeline: forward + backward for this rank's stage ----
        stage_module = _build_stage(
            my_stage_index,
            num_tables,
            num_embeddings,
            emb_dim,
            num_float_features,
            stage_dim,
            device,
        )
        stage = StageWrapper(
            module=stage_module,
            stage_pg=stage_pgs[my_stage_index],
            stage_index=my_stage_index,
        )
        pipeline = MaglevPipeline(
            stage=stage,
            stage_ranks=stage_ranks,
            global_rank=rank,
            activation_shape=torch.Size([batch_size, stage_dim]),
            device=device,
        )
        # lr=0: weights are unchanged, grads remain populated for comparison.
        optimizer = torch.optim.SGD(stage.parameters(), lr=0.0, foreach=True)
        dist_loss = pipeline.step(
            stage_input=stage_inputs[my_stage_index],
            optimizer=optimizer,
            label=label if pipeline.is_last else None,
            criterion=criterion if pipeline.is_last else None,
        )

        # ---- single-process reference: the whole MaglevModuleList in-process ----
        ref_stages: List[nn.Module] = [
            _build_stage(
                s,
                num_tables,
                num_embeddings,
                emb_dim,
                num_float_features,
                stage_dim,
                device,
            )
            for s in range(num_stages)
        ]
        ref_model = MaglevModuleList(ref_stages)
        ref_out = ref_model(stage_inputs)
        ref_loss = criterion(ref_out, label)
        ref_loss.backward()

        # 1. Last-stage loss matches the reference.
        if pipeline.is_last:
            assert dist_loss is not None
            torch.testing.assert_close(dist_loss, ref_loss)

        # 2. This stage's gradients (after DP all-reduce) match the reference
        #    stage's gradients => forward + backward numerics of the pipeline
        #    wrapper match the hand-written single-process model.
        ref_params = dict(ref_stages[my_stage_index].named_parameters())
        checked = 0
        for name, param in stage.module.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            ref_grad = ref_params[name].grad
            assert ref_grad is not None, f"no reference grad for {name}"
            torch.testing.assert_close(param.grad, ref_grad)
            checked += 1
        assert checked > 0, "no parameters were checked"


class MaglevPipelineTest(MultiProcessTestBase):
    def test_pipeline_matches_single_process(self) -> None:
        self._run_multi_process_test(
            callable=_run_correctness,
            world_size=8,
            num_stages=4,
            ranks_per_stage=2,
            batch_size=16,
            num_tables=6,
            num_embeddings=64,
            emb_dim=8,
            num_float_features=8,
            stage_dim=12,
        )
