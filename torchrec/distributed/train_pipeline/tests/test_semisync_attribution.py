#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch
from torch import nn
from torchrec.distributed.train_pipeline.batch_attribution import (
    get_exception_attribution,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    EmbeddingTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSemiSync
from torchrec.streamable import Pipelineable


class _Batch(Pipelineable):
    def __init__(self, embedding: torch.Tensor | None = None) -> None:
        self.embedding = embedding

    def to(self, device: torch.device, non_blocking: bool) -> _Batch:
        raise AssertionError("CPU training must not copy the batch")

    def record_stream(self, stream: torch.Stream) -> None:
        raise AssertionError("CPU training must not record streams")


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, device="cpu"))
        self.output = torch.tensor([7.0], device="cpu")

    def forward(self, batch: Pipelineable | None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(batch, _Batch)
        loss = self.weight * 2
        if batch.embedding is not None:
            loss = loss + batch.embedding
        return loss, self.output


class _CpuSemiSync(TrainPipelineSemiSync[Pipelineable, torch.Tensor]):
    def _mlp_forward(
        self, batch: Pipelineable, context: TrainPipelineContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._model_fwd(batch)

    def start_sparse_data_dist(
        self, batch: Pipelineable | None, context: TrainPipelineContext
    ) -> None:
        pass

    def start_embedding_lookup(
        self, batch: Pipelineable | None, context: EmbeddingTrainPipelineContext
    ) -> None:
        pass


class SemiSyncAttributionTest(unittest.TestCase):
    def _make_pipeline(
        self,
        model: _Model,
        source: torch.Tensor,
        metadata: object | None,
        *,
        stash_gradients: bool = False,
    ) -> _CpuSemiSync:
        embedding = source * 4
        detached = embedding.detach().requires_grad_()
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = _CpuSemiSync(
            model,
            optimizer,
            torch.device("cpu"),
            start_batch=73,
            stash_gradients=stash_gradients,
        )
        pipeline.batches.extend([_Batch(detached), _Batch()])

        current_context = EmbeddingTrainPipelineContext(
            index=73,
            version=1,
            embedding_tensors=[[embedding]],
            embedding_features=[["feature"]],
            detached_embedding_tensors=[[detached]],
            batch_metadata=metadata,
        )
        next_context = EmbeddingTrainPipelineContext(
            index=74,
            version=1,
            batch_metadata=metadata,
        )
        pipeline.contexts.extend([current_context, next_context])
        return pipeline

    def test_backward_paths_and_optimizer_failures_are_unattributed(self) -> None:
        for failure_kind in ("dense_backward", "embedding_backward", "optimizer"):
            with self.subTest(failure_kind=failure_kind):
                model = _Model()
                source = torch.ones(1, device="cpu", requires_grad=True)
                metadata = object()
                pipeline = self._make_pipeline(model, source, metadata)
                error = ValueError(f"{failure_kind} failed")

                if failure_kind in ("dense_backward", "embedding_backward"):

                    def fail_backward(
                        _grad: torch.Tensor,
                        raised_error: ValueError = error,
                    ) -> None:
                        raise raised_error

                    target = (
                        model.weight if failure_kind == "dense_backward" else source
                    )
                    handle = target.register_hook(fail_backward)
                else:

                    def fail_optimizer(
                        _optimizer: torch.optim.Optimizer,
                        _args: tuple[object, ...],
                        _kwargs: dict[str, object],
                        raised_error: ValueError = error,
                    ) -> None:
                        raise raised_error

                    handle = pipeline._optimizer.register_step_pre_hook(fail_optimizer)

                try:
                    with self.assertRaises(ValueError) as raised:
                        pipeline.progress(iter(()))
                finally:
                    handle.remove()

                self.assertIs(error, raised.exception)
                self.assertIsNone(get_exception_attribution(error))

    def test_successful_backward_preserves_dense_and_embedding_gradients(self) -> None:
        model = _Model()
        source = torch.ones(1, device="cpu", requires_grad=True)
        pipeline = self._make_pipeline(model, source, metadata=None)
        dense_gradients: list[torch.Tensor] = []
        hook = model.weight.register_hook(
            lambda grad: dense_gradients.append(grad.clone())
        )
        self.addCleanup(hook.remove)

        output = pipeline.progress(iter(()))

        self.assertIs(model.output, output)
        torch.testing.assert_close(
            dense_gradients,
            [torch.tensor([2.0], device="cpu")],
        )
        torch.testing.assert_close(source.grad, torch.tensor([4.0], device="cpu"))
        torch.testing.assert_close(
            model.weight.detach(),
            torch.tensor([0.8], device="cpu"),
        )
        self.assertIsNone(model.weight.grad)

    def test_gradient_stashing_preserves_optimizer_schedule(self) -> None:
        model = _Model()
        source = torch.ones(1, device="cpu", requires_grad=True)
        pipeline = self._make_pipeline(
            model,
            source,
            metadata=None,
            stash_gradients=True,
        )

        first_output = pipeline.progress(iter(()))

        self.assertIs(model.output, first_output)
        torch.testing.assert_close(
            model.weight.detach(),
            torch.tensor([1.0], device="cpu"),
        )
        self.assertIsNone(model.weight.grad)

        second_output = pipeline.progress(iter(()))

        self.assertIs(model.output, second_output)
        torch.testing.assert_close(
            model.weight.detach(),
            torch.tensor([0.8], device="cpu"),
        )
        self.assertIsNone(model.weight.grad)
