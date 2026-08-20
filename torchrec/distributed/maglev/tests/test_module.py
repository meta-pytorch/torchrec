#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torchrec.distributed.maglev.module import MaglevModuleList
from torchrec.distributed.maglev.pipeline import (
    MaglevPipeline,
    run_1f1b,
    run_1f1b_split,
)
from torchrec.distributed.maglev.stage import (
    build_handoff_process_groups,
    build_stage_process_groups,
    StageWrapper,
)
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import MaglevSplitStage, MaglevTestStage
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
    stage_cls: Type[MaglevTestStage] = MaglevTestStage,
) -> MaglevTestStage:
    """Build a stage with deterministic weights (seeded by stage index)."""
    tables = _make_tables(stage_index, num_tables, num_embeddings, emb_dim)
    torch.manual_seed(_WEIGHT_SEED + stage_index)
    return stage_cls(
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


def _run_schedule_equivalence(
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
    microbatch_counts: List[int],
) -> None:
    """``run_1f1b_split`` must be a pure reordering of ``run_1f1b``.

    Under :class:`Replicated` there is no fused TBE, so embedding weights only
    move at ``optimizer.step()`` -- which runs once, after every microbatch, in
    both schedules. Hoisting the sparse forwards therefore cannot change any
    number: the two arms must agree on the per-microbatch losses and on every
    parameter after the step. Any drift means the schedule mis-pairs a
    microbatch's sparse half with its dense half, its label, or its backward.

    Both arms use ``MaglevSplitStage`` so the only variable is the schedule --
    its derived ``forward`` is pinned against ``MaglevTestStage`` separately by
    :class:`MaglevSplitStageEquivalenceTest`.
    """
    with MultiProcessContext(rank=rank, world_size=world_size, backend="gloo"):
        device = torch.device("cpu")
        criterion = nn.MSELoss()

        stage_ranks: List[List[int]] = [
            list(range(s * ranks_per_stage, (s + 1) * ranks_per_stage))
            for s in range(num_stages)
        ]
        # Built once and shared by every arm: ``new_group`` is collective, so
        # rebuilding per arm would multiply the group count for no benefit.
        stage_pgs = build_stage_process_groups(stage_ranks)
        handoff_pgs = build_handoff_process_groups(stage_ranks)
        my_stage_index = rank // ranks_per_stage

        tables = _make_tables(my_stage_index, num_tables, num_embeddings, emb_dim)

        for num_microbatches in microbatch_counts:
            micro_inputs = [
                _make_input(
                    tables,
                    batch_size,
                    num_float_features,
                    _INPUT_SEED + my_stage_index * 100 + m,
                    device,
                )
                for m in range(num_microbatches)
            ]
            labels = []
            for m in range(num_microbatches):
                torch.manual_seed(_LABEL_SEED + m)
                labels.append(torch.randn(batch_size, stage_dim, device=device))

            arms: Dict[str, List[float]] = {}
            params: Dict[str, Dict[str, torch.Tensor]] = {}
            for name, run_fn in (("1f1b", run_1f1b), ("split", run_1f1b_split)):
                stage = StageWrapper(
                    module=_build_stage(
                        my_stage_index,
                        num_tables,
                        num_embeddings,
                        emb_dim,
                        num_float_features,
                        stage_dim,
                        device,
                        stage_cls=MaglevSplitStage,
                    ),
                    stage_pg=stage_pgs[my_stage_index],
                    stage_index=my_stage_index,
                )
                pipeline = MaglevPipeline(
                    stage=stage,
                    stage_ranks=stage_ranks,
                    global_rank=rank,
                    activation_shape=torch.Size([batch_size, stage_dim]),
                    device=device,
                    handoff_pgs=handoff_pgs,
                )
                arms[name] = run_fn(
                    pipeline=pipeline,
                    microbatch_inputs=micro_inputs,
                    optimizer=stage.configure_optimizer(lr=0.05),
                    labels=labels if pipeline.is_last else None,
                    criterion=criterion if pipeline.is_last else None,
                )
                params[name] = {
                    k: v.detach().clone() for k, v in stage.named_parameters()
                }

            ctx = f"N={num_microbatches} stage={my_stage_index}"
            assert len(arms["split"]) == len(arms["1f1b"]), (
                f"{ctx}: loss count differs -- "
                f"1f1b={len(arms['1f1b'])} split={len(arms['split'])}"
            )
            if pipeline.is_last:
                assert len(arms["1f1b"]) == num_microbatches, (
                    f"{ctx}: last stage returned {len(arms['1f1b'])} losses, "
                    f"expected {num_microbatches}"
                )
            torch.testing.assert_close(
                torch.tensor(arms["split"]),
                torch.tensor(arms["1f1b"]),
                msg=lambda m, c=ctx: f"{c}: per-microbatch losses diverge\n{m}",
            )

            assert params["split"].keys() == params["1f1b"].keys()
            assert params["1f1b"], f"{ctx}: no parameters were compared"
            for key in params["1f1b"]:
                torch.testing.assert_close(
                    params["split"][key],
                    params["1f1b"][key],
                    msg=lambda m, k=key, c=ctx: f"{c}: param {k} diverges\n{m}",
                )

            # The split path must reach the stage through the parallelizer
            # rather than holding a second reference to it: a duplicate entry
            # in ``_modules`` would emit every tensor twice here.
            state_keys = list(stage.state_dict().keys())
            assert len(state_keys) == len(set(state_keys)), (
                f"{ctx}: duplicate state_dict keys: "
                f"{sorted({k for k in state_keys if state_keys.count(k) > 1})}"
            )


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

    def test_split_schedule_matches_1f1b(self) -> None:
        # 3 stages x 2 ranks: ranks_per_stage > 1 keeps the DP all-reduce in the
        # path, and 3 stages gives a distinct (w, prehoist) per stage.
        # microbatch_counts spans the three regimes of the prehoist formula
        # ``min(max(i + 1, N - w), N)``: N < num_stages (floor dominates),
        # N == num_stages, and N > num_stages (ceiling dominates).
        self._run_multi_process_test(
            callable=_run_schedule_equivalence,
            world_size=6,
            num_stages=3,
            ranks_per_stage=2,
            batch_size=16,
            num_tables=6,
            num_embeddings=64,
            emb_dim=8,
            num_float_features=8,
            stage_dim=12,
            microbatch_counts=[2, 3, 5],
        )


class MaglevSplitStageEquivalenceTest(unittest.TestCase):
    """Pin the derivation ``forward == forward_dense o forward_sparse``.

    Single-process, no distribution, no pipeline: seed a ``MaglevSplitStage``
    and a ``MaglevTestStage`` identically and assert all three forward paths
    produce the same output. This is the cheapest correctness gate for the
    split-forward stack: it isolates the stage seam from the schedule, so a
    failure here points at the stage rather than at ``run_1f1b_split``.

    Runs against two stage shapes: the first stage (``is_first=True``, no
    ``scaled_add``, ``prev_output is None``) and a middle stage
    (``is_first=False``, exercises the residual join). Any regression that
    breaks one of the two forms will fail the corresponding case.
    """

    def _run_equivalence(self, stage_index: int, is_first: bool) -> None:
        num_tables = 4
        num_embeddings = 64
        emb_dim = 8
        num_float_features = 8
        stage_dim = 12
        batch_size = 16
        # CPU is enough -- this test verifies math, not device kernels. Keeps
        # the test portable in the same spirit as ``_run_correctness``.
        device = torch.device("cpu")

        tables = _make_tables(stage_index, num_tables, num_embeddings, emb_dim)

        torch.manual_seed(_WEIGHT_SEED + stage_index)
        base_stage = MaglevTestStage(
            tables=tables,
            stage_dim=stage_dim,
            is_first=is_first,
            num_float_features=num_float_features,
            device=device,
        )
        torch.manual_seed(_WEIGHT_SEED + stage_index)
        split_stage = MaglevSplitStage(
            tables=tables,
            stage_dim=stage_dim,
            is_first=is_first,
            num_float_features=num_float_features,
            device=device,
        )

        # ``_make_input`` returns ``test_utils.model_input.ModelInput``, but the
        # stages type ``stage_input`` as ``test_utils.test_model.ModelInput`` (two
        # separate classes with identical duck-typed shape). Cast to bridge the
        # pyre gap; the runtime attributes ``idlist_features`` / ``float_features``
        # are the same on both.
        stage_input: Any = _make_input(
            tables,
            batch_size=batch_size,
            num_float_features=num_float_features,
            seed=_INPUT_SEED + stage_index,
            device=device,
        )
        prev_output: Optional[torch.Tensor]
        if is_first:
            prev_output = None
        else:
            torch.manual_seed(_INPUT_SEED)
            prev_output = torch.randn(batch_size, stage_dim, device=device)

        base_out = base_stage.forward(stage_input, prev_output)
        split_out_derived = split_stage.forward(stage_input, prev_output)
        pooled = split_stage.forward_sparse(stage_input)
        split_out_manual = split_stage.forward_dense(stage_input, pooled, prev_output)

        # Base and derived forward must agree (weights seeded identically).
        torch.testing.assert_close(split_out_derived, base_out)
        # Explicitly-composed halves must match the derived forward -- the
        # invariant the pipeline split relies on.
        torch.testing.assert_close(split_out_manual, base_out)

    def test_first_stage_equivalence(self) -> None:
        # is_first=True: no scaled_add submodule; prev_output=None.
        self._run_equivalence(stage_index=0, is_first=True)

    def test_middle_stage_equivalence(self) -> None:
        # is_first=False: exercises the residual join (scaled_add).
        self._run_equivalence(stage_index=1, is_first=False)
