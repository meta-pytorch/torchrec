#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import types
import unittest
from typing import Any, Dict, Tuple
from unittest import mock

import torch
from torchrec.distributed import embedding as ec
from torchrec.distributed.dist_data import SequenceEmbeddingsAwaitable
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.types import Awaitable
from torchrec.modules import utils
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _FakeShardedEC(types.SimpleNamespace):
    """A stand-in module whose missing methods are ShardedEmbeddingCollection's,
    bound to it, so compute_and_output_dist runs its real helpers."""

    def __getattr__(self, name: str) -> object:
        attr = getattr(ShardedEmbeddingCollection, name)
        return types.MethodType(attr, self) if callable(attr) else attr


class _TensorWait(Awaitable[torch.Tensor]):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def _wait_impl(self) -> torch.Tensor:
        return self.value


class ScopedUnbucketizeComposeTest(unittest.TestCase):
    def setUp(self) -> None:
        for patch in [
            mock.patch.object(ec, "_seq_unbucketize_compose_checks", 0),
            mock.patch.object(ec, "_seq_unbucketize_compose_disabled", False),
            mock.patch.object(ec, "_seq_unbucketize_compose_logged", False),
            mock.patch.object(ec, "_seq_unbucketize_compose_drops", 0),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            mock.patch.object(
                ec.MemoryStashingManager, "is_enabled", return_value=False
            ),
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def _run(
        self,
        enabled: bool,
        dtype: torch.dtype = torch.float32,
        bijection: bool = True,
        training: bool = True,
        empty: bool = False,
        per_sharding_gather: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        ids = torch.tensor([] if empty else [11, 12, 11, 13, 12, 11, 12, 13])
        ids = ids.to(torch.int64)
        unique, reverse = torch.unique(ids, return_inverse=True)
        n, width = unique.numel(), 4
        permute = torch.tensor(
            [] if empty else ([2, 0, 1] if bijection else [0, 0, 2]), dtype=torch.int64
        )
        source = (
            torch.arange(n * width, dtype=torch.float32)
            .view(n, width)
            .to(dtype)
            .requires_grad_()
        )
        original = KeyedJaggedTensor.from_lengths_sync(
            keys=["a", "b"],
            values=ids,
            lengths=torch.tensor([0, 0] if empty else [4, 4]),
        )
        dedup = KeyedJaggedTensor.from_lengths_sync(
            keys=["a", "b"],
            values=unique,
            lengths=torch.tensor([n, 0]),
        )
        ctx = EmbeddingCollectionContext()
        ctx.input_features = [original]
        ctx.reverse_indices = [reverse]
        # pyrefly: ignore [bad-assignment]
        ctx.sharding_contexts = [
            types.SimpleNamespace(features_before_input_dist=dedup)
        ]
        output_wait = SequenceEmbeddingsAwaitable(
            tensor_awaitable=_TensorWait(source),
            unbucketize_permute_tensor=permute,
            embedding_dim=width,
            unbucketize_perm_is_bijection=lambda _: bijection,
        )
        module = _FakeShardedEC(
            _lookups=[lambda _: source],
            _output_dists=[lambda *_: output_wait],
            _sharding_type_to_sharding={"row_wise": None},
            _sharding_types=["row_wise"],
            _dp_index=None,
            _use_packed_jagged_tensor=False,
            _module_fqn=None,
            _embedding_dim_for_sharding_type=lambda _: width,
            post_lookup_tracker_fn=None,
            post_odist_tracker_fn=None,
            _embedding_names_per_sharding=[["a", "b"]],
            _need_indices=False,
            _features_to_permute_indices={},
            _use_gather_select=False,
            _use_gather_select_per_sharding=(
                {"row_wise": True} if per_sharding_gather else None
            ),
            _use_sorted_select=False,
            _use_lean_gather_backward=False,
            _use_index_dedup=True,
            training=training,
        )
        module._compose_reverse_indices = types.MethodType(
            ShardedEmbeddingCollection._compose_reverse_indices, module
        )
        flags = {
            "TORCHREC_SEQ_UNBUCKETIZE_COMPOSE": "1" if enabled else "0",
            "TORCHREC_SEQ_UNBUCKETIZE_COMPOSE_CHECK": "1",
            "TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP": "0",
            "TORCHREC_UNBUCKETIZE_GATHER_BWD": "0",
        }
        with mock.patch.dict(os.environ, flags), mock.patch.object(
            utils, "select_embeddings", wraps=utils.select_embeddings
        ) as select:
            awaitable = ShardedEmbeddingCollection.compute_and_output_dist(
                # pyrefly: ignore [bad-argument-type]
                module,
                ctx,
                # pyrefly: ignore [bad-argument-type]
                [dedup],
            )
            planned = bool(
                ctx.composed_reverse_indices
                and ctx.composed_reverse_indices[0] is not None
            )
            result = awaitable.wait()
            values = torch.cat([result[name].values() for name in ["a", "b"]])
            upstream = (
                torch.arange(values.numel(), dtype=torch.float32)
                .view(values.shape)
                .to(dtype)
            )
            gradient = torch.autograd.grad(values, source, upstream)[0]
            gather_flags = [
                call.kwargs.get("use_gather_select") for call in select.call_args_list
            ]
        expected_source = source.detach().clone().requires_grad_()
        expected = expected_source.index_select(0, permute).index_select(0, reverse)
        expected_grad = torch.autograd.grad(expected, expected_source, upstream)[0]
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)
        return (
            values.detach(),
            gradient,
            {"planned": planned, "gather_flags": gather_flags},
        )

    def test_forward_and_complete_gradients_match_original(self) -> None:
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                baseline = self._run(False, dtype=dtype)
                treatment = self._run(True, dtype=dtype)
                self.assertTrue(treatment[2]["planned"])
                torch.testing.assert_close(treatment[0], baseline[0], rtol=0, atol=0)
                torch.testing.assert_close(treatment[1], baseline[1], rtol=0, atol=0)

    def test_invalid_permutation_retains_accumulating_fallback(self) -> None:
        _, gradient, info = self._run(True, bijection=False)
        self.assertFalse(info["planned"])
        self.assertTrue(torch.equal(gradient[1], torch.zeros_like(gradient[1])))

    def test_eval_does_not_compose(self) -> None:
        _, _, info = self._run(True, training=False)
        self.assertFalse(info["planned"])

    def test_empty_rows(self) -> None:
        values, gradient, _ = self._run(True, empty=True)
        self.assertEqual(values.shape, (0, 4))
        self.assertEqual(gradient.shape, (0, 4))

    def test_current_per_sharding_gather_flag_is_preserved(self) -> None:
        _, _, info = self._run(True, per_sharding_gather=True)
        self.assertTrue(info["gather_flags"])
        self.assertTrue(all(info["gather_flags"]))
