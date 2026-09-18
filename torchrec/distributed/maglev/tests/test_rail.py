#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import random
import unittest
from typing import Any, cast, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torchrec.distributed.maglev.input_dist import split_dense_inputs
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    MaglevLayer,
    StructuredActivationsLayout,
)
from torchrec.distributed.maglev.pipeline import MaglevRail
from torchrec.distributed.maglev.stage import (
    HandoffPGMode,
    MaglevProcessGroups,
    StageWrapper,
)
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
_NUM_MICROBATCHES = 8
_LR = 0.05


def _tables(layer_index: int) -> List[EmbeddingBagConfig]:
    return EmbeddingTablesConfig(
        num_unweighted_features=2,
        num_weighted_features=0,
        embedding_feature_dim=4,
        base_row_size=32,
    ).generate_tables(name_prefix=f"l{layer_index}_")[0]


def _model(device: torch.device, num_layers: int = 2) -> MaglevTestModel:
    layout = StructuredActivationsLayout[MaglevTestActivations]()
    layers: List[MaglevTestLayer] = []
    for layer_index in range(num_layers):
        torch.manual_seed(_WEIGHT_SEED + layer_index)
        layers.append(
            MaglevTestLayer(
                tables=_tables(layer_index),
                layer_dim=8,
                is_first=layer_index == 0,
                activation_layout=layout,
                num_float_features=4,
                device=device,
            )
        )
    return MaglevTestModel(layers, layout)


def _inputs(
    device: torch.device, seed_offset: int = 0, num_layers: int = 2
) -> List[ModelInput]:
    inputs: List[ModelInput] = []
    for layer_index in range(num_layers):
        torch.manual_seed(_INPUT_SEED + seed_offset + layer_index)
        random.seed(_INPUT_SEED + seed_offset + layer_index)
        inputs.append(
            ModelInput.generate(
                batch_size=16,
                tables=_tables(layer_index),
                weighted_tables=[],
                num_float_features=4,
                device=device,
            )
        )
    return inputs


def _single_stage(model: MaglevTestModel, world_size: int) -> StageWrapper:
    return StageWrapper(
        model=model,
        layers_per_stage=[2],
        stage_size=world_size,
        loss_only_output=True,
        process_groups=MaglevProcessGroups.from_scratch(
            stage_size=world_size,
            num_stages=1,
            handoff_pg_mode=HandoffPGMode.SHARED,
        ),
        enable_rail=True,
    )


def _run_rail_correctness(
    rank: int,
    world_size: int,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="cpu:gloo,cuda:nccl",
    ) as context:
        device = context.device
        inputs = _inputs(device)
        model = _model(device)
        process_groups = MaglevProcessGroups.from_scratch(
            stage_size=1,
            num_stages=2,
            handoff_pg_mode=HandoffPGMode.SPLIT,
        )
        stage = StageWrapper(
            model=model,
            layers_per_stage=[1, 1],
            stage_size=1,
            process_groups=process_groups,
            enable_rail=True,
        )
        stage.to(device)

        local_layer = cast(MaglevTestLayer, model[stage.stage_index])
        sparse = local_layer.sparse
        assert sparse is not None
        calls = {"stage": 0, "sparse": 0, "dense": 0}

        def _count_stage(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            _output: object,
        ) -> None:
            calls["stage"] += 1

        def _count_sparse(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            _output: object,
        ) -> None:
            calls["sparse"] += 1

        def _count_dense(
            _module: torch.nn.Module,
            args: tuple[object, ...],
            _output: object,
        ) -> None:
            calls["dense"] += 1
            dense_input = args[1]
            assert isinstance(dense_input, ModelInput)
            assert dense_input.idlist_features is None

        stage_handle = stage.module.register_forward_hook(_count_stage)
        sparse_handle = sparse.register_forward_hook(_count_sparse)
        dense_handle = local_layer.require_dense().register_forward_hook(_count_dense)
        optimizer = torch.optim.SGD(stage.parameters(), lr=0.05, foreach=True)
        pipeline = MaglevRail(
            stage=stage,
            optimizer=optimizer,
            num_microbatches=8,
        )
        # Two passes: _pending / _rail_seams / _wdense are per-pass state, and a
        # leak in any of them is invisible on the first one.
        for _ in range(2):
            pipeline.progress(itertools.repeat(inputs))
            assert stage.pending_weight_work == 0, "weight work left undrained"
            assert not stage._pending, "microbatch left pending"
            assert not stage._rail_seams, "seam left queued"
        stage_handle.remove()
        sparse_handle.remove()
        dense_handle.remove()

        assert calls == {"stage": 18, "sparse": 2, "dense": 16}

        # Same two passes, unpipelined. Each accumulates 8 microbatch losses,
        # and mse_loss reduces to a mean, so sum_i mean_i == 8 * mean(all).
        reference = _model(device)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.05, foreach=True
        )
        for _ in range(2):
            reference_optimizer.zero_grad()
            reference_loss, _output = reference(inputs)
            (reference_loss * 8).backward()
            reference_optimizer.step()

        reference_layer = cast(MaglevTestLayer, reference[stage.stage_index])
        reference_parameters = dict(reference_layer.named_parameters())
        checked = 0
        for name, parameter in local_layer.named_parameters():
            reference_parameter = reference_parameters[name]
            expected = reference_parameter.grad
            assert parameter.grad is not None
            assert expected is not None
            torch.testing.assert_close(parameter.grad, expected)
            torch.testing.assert_close(parameter, reference_parameter)
            checked += 1
        assert checked > 0


def _dense_only_model(device: torch.device) -> MaglevTestModel:
    """A mixed stage: layer 0 has no sparse half, layer 1 does.

    Covers the paths a uniformly-sparse model never reaches -- ``pooled is None``
    in the sparse phase, a positional ``None`` in ``seams_for``, the ``None``
    filter building ``input_values``, and (on a first stage whose only layer is
    dense-only) the whole-backward fallback in ``dense_backward_act_micro``.
    """
    layout = StructuredActivationsLayout[MaglevTestActivations]()
    torch.manual_seed(_WEIGHT_SEED)
    layers: List[MaglevLayer] = [
        _DenseOnlyLayer(layer_dim=8, num_float_features=4, device=device),
        MaglevTestLayer(
            tables=_tables(1),
            layer_dim=8,
            is_first=False,
            activation_layout=layout,
            num_float_features=4,
            device=device,
        ),
    ]
    cast(MaglevTestLayer, layers[1]).activation_layout = layout
    return MaglevTestModel(cast(List[MaglevTestLayer], layers), layout)


class _DenseOnlyDense(nn.Module):
    """Dense half of a layer with no embeddings: floats in, activation out."""

    def __init__(self, layer_dim: int, num_float_features: int, device) -> None:
        super().__init__()
        self.proj: nn.Linear = nn.Linear(num_float_features, layer_dim, device=device)
        self.layout: StructuredActivationsLayout[MaglevTestActivations] = (
            StructuredActivationsLayout[MaglevTestActivations]()
        )

    def forward(
        self,
        sparse_output: Optional[torch.Tensor],
        layer_input: ModelInput,
        in_activations: Activations = (),
    ) -> Activations:
        assert sparse_output is None, "dense-only layer received a sparse output"
        return self.layout.pack(
            MaglevTestActivations(
                hidden=torch.relu(self.proj(layer_input.float_features))
            )
        )


class _DenseOnlyLayer(MaglevLayer):
    def __init__(self, layer_dim: int, num_float_features: int, device) -> None:
        super().__init__(dense=_DenseOnlyDense(layer_dim, num_float_features, device))
        self._spec: ActivationSpec = ActivationSpec(
            torch.Size([-1, layer_dim]), torch.float32
        )

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (self._spec,)

    def split_dense_input(
        self, layer_input: Any, batch_size: int, num_microbatches: int
    ) -> List[Any]:
        return split_dense_inputs(layer_input, batch_size, num_microbatches)


def _run_rail_dense_only_first_stage(rank: int, world_size: int) -> None:
    with MultiProcessContext(
        rank=rank, world_size=world_size, backend="cpu:gloo,cuda:nccl"
    ) as context:
        device = context.device
        inputs = _inputs(device)
        model = _dense_only_model(device)
        stage = StageWrapper(
            model=model,
            layers_per_stage=[1, 1],
            stage_size=1,
            process_groups=MaglevProcessGroups.from_scratch(
                stage_size=1, num_stages=2, handoff_pg_mode=HandoffPGMode.SPLIT
            ),
            enable_rail=True,
        )
        stage.to(device)
        optimizer = torch.optim.SGD(stage.module.parameters(), lr=_LR, foreach=True)
        pipeline = MaglevRail(
            stage=stage, optimizer=optimizer, num_microbatches=_NUM_MICROBATCHES
        )
        pipeline.progress(itertools.repeat(inputs))
        assert stage.pending_weight_work == 0

        reference = _dense_only_model(device)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=_LR, foreach=True
        )
        reference_loss, _output = reference(inputs)
        (reference_loss * _NUM_MICROBATCHES).backward()
        reference_optimizer.step()

        local = model[stage.stage_index]
        expected = dict(reference[stage.stage_index].named_parameters())
        checked = 0
        for name, parameter in local.named_parameters():
            assert parameter.grad is not None, f"{name} never received a gradient"
            torch.testing.assert_close(parameter.grad, expected[name].grad)
            torch.testing.assert_close(parameter, expected[name])
            checked += 1
        assert checked > 0


def _run_rail_four_stage(
    rank: int,
    world_size: int,
    handoff_pg_mode: HandoffPGMode = HandoffPGMode.SPLIT,
) -> None:
    """Four stages, so a middle stage both receives and sends, and w_lag > 1."""
    with MultiProcessContext(
        rank=rank, world_size=world_size, backend="cpu:gloo,cuda:nccl"
    ) as context:
        device = context.device
        inputs = _inputs(device, num_layers=4)
        model = _model(device, num_layers=4)
        stage = StageWrapper(
            model=model,
            layers_per_stage=[1, 1, 1, 1],
            stage_size=1,
            # The production shape: the last stage's forward_dense returns only
            # (losses,), which no other parity arm exercises.
            loss_only_output=True,
            process_groups=MaglevProcessGroups.from_scratch(
                stage_size=1, num_stages=4, handoff_pg_mode=handoff_pg_mode
            ),
            enable_rail=True,
        )
        stage.to(device)
        optimizer = torch.optim.SGD(stage.module.parameters(), lr=_LR, foreach=True)
        pipeline = MaglevRail(
            stage=stage, optimizer=optimizer, num_microbatches=_NUM_MICROBATCHES
        )
        # ZB1P: the deepest stage defers longest.
        assert pipeline.w_lag == stage.stage_index
        assert pipeline.num_warmup == 3 - stage.stage_index
        pipeline.progress(itertools.repeat(inputs))
        assert stage.pending_weight_work == 0

        reference = _model(device, num_layers=4)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=_LR, foreach=True
        )
        reference_loss, _output = reference(inputs)
        (reference_loss * _NUM_MICROBATCHES).backward()
        reference_optimizer.step()

        expected = dict(reference[stage.stage_index].named_parameters())
        checked = 0
        for name, parameter in model[stage.stage_index].named_parameters():
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected[name].grad)
            torch.testing.assert_close(parameter, expected[name])
            checked += 1
        assert checked > 0


class _UnreducedRail(MaglevRail):
    """``MaglevRail`` with the reduction removed.

    Only exists so the parity test can prove it is not vacuous: without the
    post-backward the two replicas of a stage must actually disagree.
    """

    def _reduce_gradients(self) -> None:
        pass


def _replica_mean(tensor: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    mean = tensor.detach().clone()
    dist.all_reduce(mean, group=pg)
    return mean.div_(pg.size())


def _dense_parameters(stage: StageWrapper) -> List[torch.nn.Parameter]:
    """The parameters FSDP manages here -- the dense halves, and only those.

    The sparse halves are deliberately left unsharded (see :func:`_shard_stage`),
    so their gradients stay per-replica and must not be checked for agreement.
    Under DMP they would be sharded embeddings with no ``.grad`` at all.
    """
    return [
        parameter
        for layer in stage.module.modules()
        if isinstance(layer, MaglevLayer)
        for parameter in layer.require_dense().parameters()
    ]


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """The rank-local shard of a DTensor gradient, or the tensor itself."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _shard_stage(
    stage: StageWrapper,
    device: torch.device,
    num_stages: int,
    shard: bool = False,
) -> None:
    """Apply FSDP2 the way torch's zero-bubble composability tests do.

    Per-dense-half ``fully_shard``, following torch's zero-bubble composability
    test in shape while respecting Maglev's sparse/dense split.

    The mesh is ``(pp, dp)`` with ``dp`` varying fastest, matching how
    ``MaglevProcessGroups`` lays stages out (``divmod(rank, stage_size)``), so
    ``mesh["dp"]`` has exactly ``stage.stage_pg``'s membership.
    """
    mesh = init_device_mesh(
        device.type,
        (num_stages, stage.stage_size),
        mesh_dim_names=("pp", "dp"),
    )
    dp_mesh = mesh["dp"]
    # ``replicate`` rather than ``fully_shard``: both are FSDPModules and take
    # the identical reduction path, but a replicated group leaves every rank
    # holding the whole gradient, so "the replicas agree" is a checkable
    # assertion. Under ``fully_shard`` the ranks hold different shards by
    # design and agreement is meaningless.
    #
    # Only the dense halves. Applying it to a whole layer would turn the
    # embedding table's weight into a DTensor, and aten._embedding_bag then
    # rejects the plain-tensor indices ("got mixed torch.Tensor and DTensor").
    # The sparse half is DMP's to shard, not FSDP2's.
    for layer in stage.module.modules():
        if isinstance(layer, MaglevLayer):
            if shard:
                # reshard_after_forward=True on purpose: the schedule must turn
                # it off itself, or the deferred W's grad-accumulator lookup
                # misses the parameter the I half recorded.
                fully_shard(
                    layer.require_dense(),
                    mesh=dp_mesh,
                    reshard_after_forward=True,
                )
            else:
                replicate(layer.require_dense(), mesh=dp_mesh)


def _run_rail_fsdp_parity(rank: int, world_size: int) -> None:
    """One reduction per pass, landing where a hand-averaged run lands.

    Two stages of two data-parallel replicas, each replica fed different data,
    so an unreduced pass leaves the halves of a stage disagreeing -- which
    ``_UnreducedRail`` asserts before the real schedule runs.

    """
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="cpu:gloo,cuda:nccl",
    ) as context:
        device = context.device

        def _build() -> Tuple[StageWrapper, List[ModelInput]]:
            stage = StageWrapper(
                model=_model(device),
                layers_per_stage=[1, 1],
                stage_size=2,
                process_groups=MaglevProcessGroups.from_scratch(
                    stage_size=2,
                    num_stages=2,
                    handoff_pg_mode=HandoffPGMode.SPLIT,
                ),
                enable_rail=True,
            )
            stage.to(device)
            _shard_stage(stage, device, num_stages=2)
            # Data follows the position, not the stage, so the two stages of a
            # lane agree while the two replicas of a stage do not.
            return stage, _inputs(device, seed_offset=1000 * stage.position)

        stage, inputs = _build()
        unreduced = _UnreducedRail(
            stage=stage,
            # foreach=False: this stage mixes FSDP-sharded dense DTensors with
            # unsharded embedding parameters, and _foreach_add_ refuses the mix.
            # Under DMP the sparse half sits in the fused TBE optimizer instead,
            # so a real stage's outer optimizer sees only DTensors.
            optimizer=torch.optim.SGD(stage.parameters(), lr=_LR, foreach=False),
            num_microbatches=_NUM_MICROBATCHES,
        )
        unreduced.progress(itertools.repeat(inputs))
        # The reference: what a correct reduction must produce. Both arms are
        # built from the same seeds and fed the same data, so the unreduced
        # arm's per-replica gradients are exactly the reduced arm's inputs, and
        # their mean is the answer FSDP2 owes us.
        expected: List[torch.Tensor] = []
        diverged = 0
        for parameter in _dense_parameters(stage):
            grad = parameter.grad
            assert grad is not None, "unreduced pass produced no gradient"
            local = _local(grad)
            mean = _replica_mean(local, stage.stage_pg)
            expected.append(mean)
            if not torch.allclose(local, mean):
                diverged += 1
        assert diverged > 0, "replicas agreed without a reduction; test is vacuous"

        stage, inputs = _build()
        pipeline = MaglevRail(
            stage=stage,
            # foreach=False: this stage mixes FSDP-sharded dense DTensors with
            # unsharded embedding parameters, and _foreach_add_ refuses the mix.
            # Under DMP the sparse half sits in the fused TBE optimizer instead,
            # so a real stage's outer optimizer sees only DTensors.
            optimizer=torch.optim.SGD(stage.parameters(), lr=_LR, foreach=False),
            num_microbatches=_NUM_MICROBATCHES,
        )
        assert pipeline._fsdp_modules, "replicate did not produce an FSDPModule"
        pipeline.progress(itertools.repeat(inputs))
        assert stage.pending_weight_work == 0, "weight work left undrained"
        assert not stage._pending, "microbatch left pending"
        assert not stage._rail_seams, "seam left queued"

        # Against the hand-computed reference, not against itself. Asserting
        # only that the replicas agree would pass for a reduction that averaged
        # the wrong way, summed instead of averaging, or broadcast rank 0 --
        # all of which leave the replicas in perfect agreement.
        checked = 0
        for parameter, reference in zip(_dense_parameters(stage), expected):
            grad = parameter.grad
            assert grad is not None, "reduced pass produced no gradient"
            torch.testing.assert_close(_local(grad), reference)
            checked += 1
        assert checked == len(expected) and checked > 0


def _run_rail_fully_shard(rank: int, world_size: int) -> None:
    """ZeRO-3: `fully_shard` runs the same reduction path as `replicate`.

    Not a gradient-agreement test -- under `fully_shard` each rank holds a
    different shard by construction, so agreement is not well formed. What this
    pins is that the sharded path executes end to end, that the schedule turns
    `reshard_after_forward` off whatever the caller asked for, and that every
    dense parameter comes out of a pass with a gradient.
    """
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="cpu:gloo,cuda:nccl",
    ) as context:
        device = context.device
        stage = StageWrapper(
            model=_model(device),
            layers_per_stage=[1, 1],
            stage_size=2,
            process_groups=MaglevProcessGroups.from_scratch(
                stage_size=2,
                num_stages=2,
                handoff_pg_mode=HandoffPGMode.SPLIT,
            ),
            enable_rail=True,
        )
        stage.to(device)
        _shard_stage(stage, device, num_stages=2, shard=True)
        inputs = _inputs(device, seed_offset=1000 * stage.position)
        pipeline = MaglevRail(
            stage=stage,
            optimizer=torch.optim.SGD(stage.parameters(), lr=_LR, foreach=False),
            num_microbatches=_NUM_MICROBATCHES,
        )
        groups = [
            group
            for module in pipeline._fsdp_modules
            for group in module._get_fsdp_state()._fsdp_param_groups
        ]
        assert groups, "fully_shard produced no parameter group"
        for group in groups:
            assert not group._reshard_after_forward, "schedule left resharding on"

        for _ in range(2):
            pipeline.progress(itertools.repeat(inputs))
            assert stage.pending_weight_work == 0, "weight work left undrained"
            assert not stage._pending, "microbatch left pending"

        checked = 0
        for parameter in _dense_parameters(stage):
            assert parameter.grad is not None, "sharded dense parameter got no gradient"
            checked += 1
        assert checked > 0


def _run_rail_zero0_zero3_equivalence(rank: int, world_size: int) -> None:
    """ZeRO-3 must reduce to the same gradients ZeRO-0 does.

    `test_fsdp_parity` proves the replicated path numerically, but under
    `fully_shard` each rank holds a different shard, so "the replicas agree" is
    not a well-formed assertion there. This closes that gap from the other side:
    run the same model and data both ways and compare the *gathered* gradient.
    Sharding is a data-layout choice, so the full gradient must be identical.
    """
    with MultiProcessContext(
        rank=rank, world_size=world_size, backend="cpu:gloo,cuda:nccl"
    ) as context:
        device = context.device

        def run(shard: bool) -> List[torch.Tensor]:
            stage = StageWrapper(
                model=_model(device),
                layers_per_stage=[1, 1],
                stage_size=2,
                process_groups=MaglevProcessGroups.from_scratch(
                    stage_size=2, num_stages=2, handoff_pg_mode=HandoffPGMode.SPLIT
                ),
                enable_rail=True,
            )
            stage.to(device)
            _shard_stage(stage, device, num_stages=2, shard=shard)
            inputs = _inputs(device, seed_offset=1000 * stage.position)
            pipeline = MaglevRail(
                stage=stage,
                optimizer=torch.optim.SGD(stage.parameters(), lr=_LR, foreach=False),
                num_microbatches=_NUM_MICROBATCHES,
            )
            pipeline.progress(itertools.repeat(inputs))
            grads = []
            for parameter in _dense_parameters(stage):
                grad = parameter.grad
                assert grad is not None, "dense parameter got no gradient"
                # full_tensor() gathers a sharded grad; on a replicated one it
                # is the identity, so both arms are comparable.
                grads.append(
                    grad.full_tensor() if hasattr(grad, "full_tensor") else grad
                )
            return grads

        zero0 = run(shard=False)
        zero3 = run(shard=True)
        assert len(zero0) == len(zero3) and zero0
        for a, b in zip(zero0, zero3):
            torch.testing.assert_close(a, b)


class _DeferralSpyRail(MaglevRail):
    """Records how deep the deferred-weight queue gets during a pass.

    The zero-bubble property is otherwise unobservable: deferring `W` changes
    only *when* weight gradients are computed, so every value-based test passes
    just as well with the deferral removed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.observed: List[int] = []

    def _dense_zero_bubble(self, state: Any) -> None:
        stage = self.stage
        original = stage.dense_backward_act_micro

        def _spy() -> Any:
            out = original()
            self.observed.append(stage.pending_weight_work)
            return out

        # pyre-ignore[8]: deliberate instance-level patch for the duration
        stage.dense_backward_act_micro = _spy
        try:
            super()._dense_zero_bubble(state)
        finally:
            del stage.dense_backward_act_micro


def _run_rail_defers_weight_work(rank: int, world_size: int) -> None:
    """`W` really trails `I` by the stage index -- the reason Rail exists.

    `pending_weight_work` is sampled immediately after each `I`, before the
    drain that follows it. On stage `s` the queue must reach `s` entries; with
    the deferral removed it never exceeds 1 on any stage, and every other test
    in this file still passes.
    """
    with MultiProcessContext(
        rank=rank, world_size=world_size, backend="cpu:gloo,cuda:nccl"
    ) as context:
        device = context.device
        stage = StageWrapper(
            model=_model(device, num_layers=4),
            layers_per_stage=[1, 1, 1, 1],
            stage_size=1,
            process_groups=MaglevProcessGroups.from_scratch(
                stage_size=1, num_stages=4, handoff_pg_mode=HandoffPGMode.SPLIT
            ),
            enable_rail=True,
        )
        stage.to(device)
        pipeline = _DeferralSpyRail(
            stage=stage,
            optimizer=torch.optim.SGD(stage.parameters(), lr=_LR, foreach=True),
            num_microbatches=_NUM_MICROBATCHES,
        )
        pipeline.progress(itertools.repeat(_inputs(device, num_layers=4)))

        assert len(pipeline.observed) == _NUM_MICROBATCHES, pipeline.observed
        # Sampled after I, before its drain, so the queue holds the microbatch
        # just finished plus everything still deferred behind it.
        assert max(pipeline.observed) == stage.stage_index + 1, (
            stage.stage_index,
            pipeline.observed,
        )
        assert stage.pending_weight_work == 0, "weight work left undrained"


def _run_rail_clears_state_after_failed_pass(rank: int, world_size: int) -> None:
    """A pass that raises leaves no microbatch state behind.

    Not a recovery test: after a failed pass the handoff communicators are out
    of step across ranks and the job is over. What this pins is that the local
    FIFOs are empty afterwards, so the wreckage cannot masquerade as a working
    pass -- without it, a caller that catches and retries pops this pass's
    entries against the next pass's gradients and trains on the mispairing.
    """
    with MultiProcessContext(
        rank=rank, world_size=world_size, backend="cpu:gloo,cuda:nccl"
    ) as context:
        device = context.device
        stage = StageWrapper(
            model=_model(device),
            layers_per_stage=[1, 1],
            stage_size=1,
            process_groups=MaglevProcessGroups.from_scratch(
                stage_size=1, num_stages=2, handoff_pg_mode=HandoffPGMode.SPLIT
            ),
            enable_rail=True,
        )
        stage.to(device)
        pipeline = MaglevRail(
            stage=stage,
            optimizer=torch.optim.SGD(stage.parameters(), lr=_LR, foreach=True),
            num_microbatches=_NUM_MICROBATCHES,
        )
        inputs = _inputs(device)

        original = stage.dense_forward_micro

        def _fail_midway(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            if stage._pending:
                raise RuntimeError("injected failure")
            return out

        # pyre-ignore[8]: deliberate instance-level patch
        stage.dense_forward_micro = _fail_midway
        try:
            pipeline.progress(itertools.repeat(inputs))
        except RuntimeError as error:
            assert "injected failure" in str(error), error
        else:
            raise AssertionError("the injected failure did not propagate")
        finally:
            del stage.dense_forward_micro

        assert not stage._pending, "failed pass left microbatches queued"
        assert not stage._rail_seams, "failed pass left seams queued"
        assert stage.pending_weight_work == 0, "failed pass left weight work queued"


@unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two CUDA devices")
class MaglevRailTest(MultiProcessTestBase):
    def test_two_stage_parity(self) -> None:
        # Two stages, so stage 0 has a warmup and w_lag actually defers.
        self._run_multi_process_test(
            callable=_run_rail_correctness,
            world_size=2,
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "needs four CUDA devices")
    def test_four_stage_parity(self) -> None:
        self._run_multi_process_test(
            callable=_run_rail_four_stage,
            world_size=4,
        )

    def test_clears_state_after_failed_pass(self) -> None:
        # A raised pass must not leave queues that a retry would mispair.
        self._run_multi_process_test(
            callable=_run_rail_clears_state_after_failed_pass, world_size=2
        )

    def test_dense_only_first_stage(self) -> None:
        # Stage 0's only layer has no sparse half, so its I half takes the
        # whole-backward fallback; stage 1 is ordinary.
        self._run_multi_process_test(
            callable=_run_rail_dense_only_first_stage,
            world_size=2,
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "needs four CUDA devices")
    def test_four_stage_parity_shared_handoff(self) -> None:
        # Rail never uses the batched crossover that shared mode exists for --
        # I sends its gradient before the weight work, so the two directions are
        # never ready to be paired. It posts each direction independently, which
        # a shared communicator matches per peer and direction in issue order.
        # Maglev1F1B needs a separate _progress_shared; Rail does not.
        self._run_multi_process_test(
            callable=_run_rail_four_stage,
            world_size=4,
            handoff_pg_mode=HandoffPGMode.SHARED,
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "needs four CUDA devices")
    def test_defers_weight_work(self) -> None:
        # The zero-bubble property itself. Nothing else in this file observes it.
        self._run_multi_process_test(
            callable=_run_rail_defers_weight_work, world_size=4
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "needs four CUDA devices")
    def test_zero0_zero3_equivalence(self) -> None:
        # The only numerical check on the sharded path.
        self._run_multi_process_test(
            callable=_run_rail_zero0_zero3_equivalence, world_size=4
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "needs four CUDA devices")
    def test_fully_shard_runs(self) -> None:
        # ZeRO-3. Agreement is not checkable when the ranks hold different
        # shards, so this pins execution and the reshard policy instead.
        self._run_multi_process_test(
            callable=_run_rail_fully_shard,
            world_size=4,
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "needs four CUDA devices")
    def test_fsdp_parity(self) -> None:
        # 2 stages x 2 replicas: stage_size=1 would prove nothing about a
        # reduction, since there would be nobody to reduce with.
        self._run_multi_process_test(
            callable=_run_rail_fsdp_parity,
            world_size=4,
        )
