#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scaffolding the Maglev data-parallel tests share: mock process groups, a
small chain model, its inputs and reference checks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.tensor import DTensor
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    MaglevLayer,
    MaglevModuleList,
)
from torchrec.distributed.maglev.stage import (
    HandoffPGMode,
    MaglevProcessGroups,
    StageWrapper,
)


def mock_groups(num_stages: int = 1) -> MaglevProcessGroups:
    """Mock process groups for ``num_stages`` one-rank stages."""
    group = MagicMock()
    return MaglevProcessGroups(
        stage_ranks=tuple((index,) for index in range(num_stages)),
        stage_pg=group,
        handoff_pgs=(MagicMock(), MagicMock()),
        cascade_pg=group,
        cascade_gloo_pg=group,
        handoff_pg_mode=HandoffPGMode.SPLIT,
    )


def stage_mock() -> MagicMock:
    """A mock one-stage :class:`StageWrapper` around a plain module."""
    stage = MagicMock(spec=StageWrapper)
    stage.module = torch.nn.Linear(1, 1)
    stage.num_stages = 1
    stage.stage_index = 0
    stage.is_first = True
    stage.in_activation_specs.return_value = ()
    return stage


def partial_devices(world_size: int) -> bool:
    """Whether the host has CUDA devices, but fewer than ``world_size``: each
    rank of a Gloo multi-process test still binds the device of its index."""
    return 0 < torch.cuda.device_count() < world_size


class SparseFrontend(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frontend = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.frontend(value).tanh()


class DenseHalf(torch.nn.Module):
    """``extra`` runs only where ``use_extra`` is set, so one replica can use a
    parameter the other does not; ``unused``, when present, never runs, so it
    must end every pass at ``grad=None``."""

    def __init__(self, unused: bool = False) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.extra = torch.nn.Linear(4, 4)
        self.unused: torch.nn.Linear | None = torch.nn.Linear(4, 4) if unused else None
        self.use_extra = True

    def forward(
        self,
        pooled: torch.Tensor,
        _value: torch.Tensor,
        incoming: Activations,
    ) -> Activations:
        value = pooled + incoming[0] if incoming else pooled
        if self.use_extra:
            value = value + self.extra(value).tanh()
        return (self.projection(value).tanh(),)

    def require_unused(self) -> torch.nn.Linear:
        if self.unused is None:
            raise AssertionError("built without the unused projection")
        return self.unused


class Layer(MaglevLayer):
    def __init__(self, first: bool, unused: bool = False) -> None:
        super().__init__(sparse=SparseFrontend(), dense=DenseHalf(unused))
        self.first = first

    def in_activation_specs(self) -> tuple[ActivationSpec, ...]:
        return () if self.first else self.out_activation_specs()

    def out_activation_specs(self) -> tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, 4])),)

    def sparse_half(self) -> SparseFrontend:
        sparse = self.sparse
        if not isinstance(sparse, SparseFrontend):
            raise AssertionError("expected the sparse frontend")
        return sparse

    def dense_half(self) -> DenseHalf:
        dense = self.require_dense()
        if not isinstance(dense, DenseHalf):
            raise AssertionError("expected the dense half")
        return dense


class ChainModel(MaglevModuleList):
    """``num_layers`` layers, each reading the whole raw batch; the loss is the
    last activation's square, summed or, with ``mean_loss``, averaged."""

    def __init__(
        self, num_layers: int = 2, unused: bool = False, mean_loss: bool = False
    ) -> None:
        super().__init__(
            [Layer(first=index == 0, unused=unused) for index in range(num_layers)]
        )
        self.mean_loss = mean_loss
        # Not len(self): a stage built from this model may take its layers out
        # of the list, and the stage still calls this preproc.
        self.num_inputs = num_layers

    def preproc(self, model_input: torch.Tensor) -> list[torch.Tensor]:
        return [model_input] * self.num_inputs

    def get_batch_size(self, layer_inputs: Sequence[Any]) -> int:
        return layer_inputs[0].shape[0]

    def postproc(
        self, activations: Activations, layer_input: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        squares = activations[0].square()
        loss = squares.mean() if self.mean_loss else squares.sum()
        return loss, activations[0].detach()


def as_chain_layer(module: torch.nn.Module) -> Layer:
    if not isinstance(module, Layer):
        raise AssertionError(f"expected a chain layer, got {type(module).__name__}")
    return module


def set_use_extra(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, DenseHalf):
            module.use_extra = enabled


def input_for(replica: int, index: int, device: torch.device) -> torch.Tensor:
    """An 8x4 raw batch, distinct per replica and per ``index``."""
    return torch.arange(32, device=device, dtype=torch.float32).reshape(8, 4) / 32 + (
        replica * 0.3 + index * 0.1
    )


def assert_reduced_layer(layer: torch.nn.Module, reference: torch.nn.Module) -> None:
    """Each weight, and each gradient or its absence, matches the reference's."""
    expected = dict(reference.named_parameters())
    for name, parameter in layer.named_parameters():
        weight = (
            parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        )
        torch.testing.assert_close(weight, expected[name], rtol=1e-4, atol=1e-5)
        grad, expected_grad = parameter.grad, expected[name].grad
        if (grad is None) != (expected_grad is None):
            raise AssertionError(
                f"{name}: grad is {grad}, reference grad is {expected_grad}"
            )
        if grad is not None:
            if isinstance(grad, DTensor):
                grad = grad.full_tensor()
            torch.testing.assert_close(grad, expected_grad, rtol=1e-4, atol=1e-5)


def two_by_two_stage(
    model: ChainModel, device: torch.device
) -> tuple[StageWrapper, Any]:
    """This rank's stage of ``model`` cut over two stages of two replicas.

    The input driver is a mock, returned for the caller to feed: its NCCL
    exchange needs CUDA, and the tests hand each pass its microbatches.
    """
    groups = MaglevProcessGroups.from_scratch(stage_size=2, num_stages=2)
    with patch("torchrec.distributed.maglev.stage.InputDistDriver") as driver:
        stage = StageWrapper(model, [1, 1], stage_size=2, process_groups=groups)
    del model[:]
    stage.to(device)
    return stage, driver


def step_averaged_reference(
    reference: ChainModel,
    optimizer: torch.optim.Optimizer,
    index: int,
    device: torch.device,
    divergent: bool,
) -> None:
    """Step the single-process reference on both replicas' batches, their
    losses averaged; with ``divergent``, replica 0 alone runs ``extra``."""
    optimizer.zero_grad()
    losses = []
    for replica in range(2):
        set_use_extra(reference, replica == 0 or not divergent)
        losses.append(reference(input_for(replica, index, device))[0])
    torch.stack(losses).mean().backward()
    optimizer.step()


def microbatches(
    raw: torch.Tensor, num_microbatches: int, num_layers: int = 1
) -> list[list[torch.Tensor]]:
    """``raw`` cut by rows into ``num_microbatches`` carriers, one value per local layer."""
    return [[part] * num_layers for part in torch.tensor_split(raw, num_microbatches)]


def averaged_passes(
    stage: StageWrapper,
    reference: ChainModel,
    optimizer: torch.optim.Optimizer,
    driver: Any,
    device: torch.device,
    divergent: bool,
    num_microbatches: int,
    passes: int = 3,
) -> Iterator[torch.Tensor]:
    """Yield this replica's raw batch for each pass, after stepping the
    reference on it; once the caller has run the pass, the stage's one layer
    must match the reference's.

    The stubbed input driver hands the pass the batch's rows as
    ``num_microbatches`` microbatches. The chain model's loss sums over rows,
    so the microbatch losses sum to the reference's whole-batch loss.
    """
    for pass_index in range(passes):
        step_averaged_reference(reference, optimizer, pass_index, device, divergent)
        raw = input_for(stage.position, pass_index, device)
        driver.return_value.take.return_value = microbatches(raw, num_microbatches)
        yield raw
        assert_reduced_layer(stage._layers[0], reference[stage.stage_index])
