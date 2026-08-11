#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torch
from hypothesis import given, settings, strategies as st
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    TrainPipelineFusedSparseDist,
)
from torchrec.distributed.types import ShardingType

_FUSED_PARAMS: dict[str, bool] = {"stochastic_rounding": False}
_NUM_BATCHES = 12
_BATCH_SIZE = 32


class TrainPipelineFusedSparseDistTest(TrainPipelineSparseDistTestBase):
    def _assert_equal_to_non_pipelined(
        self, sharding_type: str, **pipeline_kwargs: object
    ) -> None:
        """Runs the pipeline against a non-pipelined reference and compares.

        Args:
            sharding_type: sharding type for both models.
            pipeline_kwargs: extra kwargs forwarded to TrainPipelineFusedSparseDist.
        """
        data = self._generate_data(
            num_batches=_NUM_BATCHES,
            batch_size=_BATCH_SIZE,
        )
        dataloader = iter(data)

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, EmbeddingComputeKernel.FUSED.value, _FUSED_PARAMS
        )
        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, sharding_type, EmbeddingComputeKernel.FUSED.value, _FUSED_PARAMS
        )
        copy_state_dict(
            sharded_model.state_dict(), sharded_model_pipelined.state_dict()
        )

        pipeline = TrainPipelineFusedSparseDist(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            # pyre-ignore[6]: kwargs are forwarded verbatim.
            **pipeline_kwargs,
        )

        for batch in data[:-2]:
            batch = batch.to(self.device)
            optim.zero_grad()
            loss, pred = sharded_model(batch)
            loss.backward()
            optim.step()

            pred_pipeline = pipeline.progress(dataloader)

            torch.testing.assert_close(pred, pred_pipeline)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @patch("torch._utils_internal.justknobs_check", return_value=True)
    def test_new_emb_lookup_stream_equal_to_non_pipelined(
        self, _mock_justknobs_check: MagicMock
    ) -> None:
        """
        Tests that running the embedding lookup on a dedicated stream, rather
        than reusing the data-dist stream, still matches non-pipelined results.
        The lookup then consumes input-dist output across a stream boundary.
        """
        self._assert_equal_to_non_pipelined(
            sharding_type=ShardingType.TABLE_WISE.value,
            emb_lookup_stream="new",
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # 8 combinations of the axes below; sample all of them so coverage does not
    # depend on which examples hypothesis happens to draw.
    @settings(max_examples=8, deadline=None)
    @given(
        enqueue_batch_after_forward=st.booleans(),
        embedding_lookup_after_data_dist=st.booleans(),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
    )
    def test_equal_to_non_pipelined(
        self,
        enqueue_batch_after_forward: bool,
        embedding_lookup_after_data_dist: bool,
        sharding_type: str,
    ) -> None:
        """
        Tests TrainPipelineFusedSparseDist with various parameter combinations
        produces same results as non-pipelined execution.
        """
        self._assert_equal_to_non_pipelined(
            sharding_type=sharding_type,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            embedding_lookup_after_data_dist=embedding_lookup_after_data_dist,
        )
