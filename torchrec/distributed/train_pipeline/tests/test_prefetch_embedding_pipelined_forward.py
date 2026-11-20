#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest.mock import MagicMock

from torchrec.distributed.train_pipeline.pipeline_context import (
    PrefetchEmbeddingTrainPipelineContext,
    PrefetchTrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import (
    InSyncPrefetchEmbeddingPipelinedForward,
    PrefetchEmbeddingPipelinedForward,
    PrefetchPipelinedForward,
)
from torchrec.distributed.train_pipeline.types import CallArgs


class TestPrefetchEmbeddingPipelinedForward(unittest.TestCase):
    """Test PrefetchEmbeddingPipelinedForward key functionality"""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_module = MagicMock()
        self.prefetch_context = PrefetchTrainPipelineContext()
        self.mock_args = CallArgs(args=[], kwargs={})

    def test_prefetch_returns_true(self) -> None:
        """Test that prefetch() returns True."""
        forward = PrefetchEmbeddingPipelinedForward(
            name="test_prefetch",
            args=self.mock_args,
            module=self.mock_module,
            context=self.prefetch_context,
        )

        # Test that prefetch returns True
        self.assertIsInstance(forward, PrefetchPipelinedForward)

    def test_call_fails_without_compute_and_output_dist(self) -> None:
        """Test that __call__ fails if compute_and_output_dist is not called first."""
        forward = PrefetchEmbeddingPipelinedForward(
            name="test_call_error",
            args=self.mock_args,
            module=self.mock_module,
            context=self.prefetch_context,
        )

        # Should raise exception when called without compute_and_output_dist
        with self.assertRaises(Exception) as context:
            forward()

        self.assertIn(
            "compute_and_output_dist must be called before __call__",
            str(context.exception),
        )

    def test_call_succeeds_after_compute_and_output_dist(self) -> None:
        """Test that __call__ succeeds when compute_and_output_dist is called first."""
        forward = PrefetchEmbeddingPipelinedForward(
            name="test_call_success",
            args=self.mock_args,
            module=self.mock_module,
            context=self.prefetch_context,
        )

        # Set up mock data in context
        test_data = MagicMock()
        test_ctx = MagicMock()
        self.prefetch_context.module_input_post_prefetch = {
            "test_call_success": test_data
        }
        self.prefetch_context.module_contexts_post_prefetch = {
            "test_call_success": test_ctx
        }

        # Mock the module's compute_and_output_dist method
        mock_awaitable = MagicMock()
        self.mock_module.compute_and_output_dist.return_value = mock_awaitable

        # Call compute_and_output_dist first
        forward.compute_and_output_dist()

        # Now __call__ should succeed and return the awaitable
        result = forward()
        self.assertEqual(result, mock_awaitable)


class InSyncPrefetchEmbeddingPipelinedForwardTest(unittest.TestCase):
    """Test InSyncPrefetchEmbeddingPipelinedForward, used by
    PrefetchTrainPipelineFusedSparseDist.

    The lookup stage has already left an awaitable in `embedding_a2a_requests`, so
    this forward only hands it over. What distinguishes it from the
    InSyncEmbeddingPipelinedForward it derives from is what it does *not* do:
    it does not wait on the awaitable, and it does not consume `module_contexts`.
    """

    _NAME = "sharded_ebc"

    def setUp(self) -> None:
        self.context = PrefetchEmbeddingTrainPipelineContext()
        self.forward: InSyncPrefetchEmbeddingPipelinedForward = (
            InSyncPrefetchEmbeddingPipelinedForward(
                name=self._NAME,
                args=CallArgs(args=[], kwargs={}),
                module=MagicMock(),
                context=self.context,
            )
        )

    def test_call_returns_the_awaitable_unwaited(self) -> None:
        """The wait has to stay in the model, where LazyAwaitable triggers it.

        Waiting here would hand the model a materialized tensor, which is what
        breaks Inductor's layout planning under torch.compile.
        """
        awaitable = MagicMock()
        self.context.embedding_a2a_requests[self._NAME] = awaitable

        result = self.forward()

        self.assertIs(result, awaitable)
        awaitable.wait.assert_not_called()

    def test_call_leaves_module_contexts_alone(self) -> None:
        """The prefetch and lookup stages own this module's context.

        The prefetch stage moved it to `module_contexts_post_prefetch` and the
        lookup stage popped it from there, so a pop here would raise on a
        dict the pipeline still expects to be empty of this module.
        """
        module_context = MagicMock()
        self.context.module_contexts[self._NAME] = module_context
        self.context.embedding_a2a_requests[self._NAME] = MagicMock()

        self.forward()

        self.assertEqual(self.context.module_contexts, {self._NAME: module_context})

    def test_second_call_for_same_module_raises(self) -> None:
        """A pipelined embedding module may only be invoked once per forward."""
        self.context.embedding_a2a_requests[self._NAME] = MagicMock()
        self.context.embedding_a2a_requests["other_ebc"] = MagicMock()

        self.forward()

        with self.assertRaises(AssertionError) as ctx:
            self.forward()
        self.assertIn(f"embedding lookup of {self._NAME}", str(ctx.exception))
        self.assertIn(
            "Available embedding_a2a names: ['other_ebc']", str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
