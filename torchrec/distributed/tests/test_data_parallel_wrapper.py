#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.ddp_comm_hooks import (
    bf16_stream_compress_hook,
    CommHook,
    fp16_stream_compress_hook,
    StreamCompressState,
)
from torchrec.distributed.model_parallel import DefaultDataParallelWrapper
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)

# Every rank seeds with this, so the models and the base input agree across
# ranks and across both hooks under test.
SEED = 7


def _ddp_grads_with_hook(
    device: torch.device,
    pg: dist.ProcessGroup,
    model: nn.Module,
    inputs: torch.Tensor,
    state: Any,
    hook: CommHook,
) -> List[torch.Tensor]:
    """Runs three DDP backwards with `hook` registered, returning the gradients."""
    ddp = DistributedDataParallel(
        model.to(device),
        device_ids=[device],
        process_group=pg,
        gradient_as_bucket_view=True,
        # Small buckets so the hook runs several times per backward.
        bucket_cap_mb=1,
    )
    ddp.register_comm_hook(state, hook)

    # More than one backward, so a caller can tell a stream reused across
    # iterations from one reused only within a single backward. Gradients
    # accumulate across all of them, identically for either hook.
    for _ in range(3):
        ddp(inputs).sum().backward()

    return [p.grad.detach().clone() for p in ddp.parameters()]  # pyre-ignore[16]


def _build_model() -> nn.Module:
    """Builds the model, reseeding so every call returns identical weights."""
    torch.manual_seed(SEED)
    return nn.Sequential(*[nn.Linear(512, 512, bias=False) for _ in range(4)])


def _test_compress_comm_hook_matches_default_hook(
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        pg = ctx.pg
        assert pg is not None
        device = ctx.device

        torch.manual_seed(SEED)
        inputs = torch.randn(8, 512, device=device) + rank

        # DDP w/ Stock BF16 compression hook
        default_grads = _ddp_grads_with_hook(
            device,
            pg,
            _build_model(),
            inputs,
            pg,
            ddp_default_hooks.bf16_compress_hook,
        )

        state = StreamCompressState(process_group=pg)

        # Records the stream each call ran against, to check the hook keeps
        # using one rather than taking a fresh one per bucket.
        streams_used: List[Optional[torch.cuda.Stream]] = []

        def recording_hook(
            hook_state: StreamCompressState, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            future = bf16_stream_compress_hook(hook_state, bucket)
            streams_used.append(hook_state.stream)
            return future

        # DDP w/ new BF16 hook + stream tracking
        hook_grads = _ddp_grads_with_hook(
            device,
            pg,
            _build_model(),
            inputs,
            state,
            recording_hook,
        )

        # Both hooks divide by world size, allreduce in BF16 over the same
        # process group and cast back, so the results are bit identical.
        assert len(default_grads) == len(hook_grads)
        for default_grad, hook_grad in zip(default_grads, hook_grads):
            torch.testing.assert_close(hook_grad, default_grad, rtol=0, atol=0)

        # The property the hook exists for: every bucket of every iteration
        # lands on the one stream the state owns. A hook that took a fresh
        # stream per bucket, or reused one within a backward but not across
        # backwards, leaves extra entries here and fails.
        assert state.stream is not None
        # Guards the set comparison below from being satisfied by a single call.
        assert len(streams_used) > 1, streams_used
        assert set(streams_used) == {state.stream}, streams_used


class StreamCompressHookTest(MultiProcessTestBase):
    """Runs the stream hooks under real DDP, against the stock hooks."""

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_compress_comm_hook_matches_default_hook(self) -> None:
        self._run_multi_process_test(
            callable=_test_compress_comm_hook_matches_default_hook,
            world_size=2,
            backend="nccl",
        )


class CommHookSelectionTest(unittest.TestCase):
    """
    Maps `allreduce_comm_precision` to the `(state, hook)` pair to register.

    Runs against a real single-rank gloo group; no hook is ever executed here.
    """

    def setUp(self) -> None:
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )
        self.pg: dist.ProcessGroup = dist.distributed_c10d._get_default_group()
        self.device: torch.device = torch.device("cpu")

    def tearDown(self) -> None:
        dist.destroy_process_group()

    def _hook_for(self, precision: Optional[str]) -> Optional[CommHook]:
        wrapper = DefaultDataParallelWrapper(allreduce_comm_precision=precision)
        selected = wrapper._select_allreduce_comm_hook(self.pg, self.device)
        return None if selected is None else selected[1]

    def test_precision_selects_stock_hook(self) -> None:
        self.assertIs(self._hook_for("bf16"), ddp_default_hooks.bf16_compress_hook)
        self.assertIs(self._hook_for("fp16"), ddp_default_hooks.fp16_compress_hook)

    @parameterized.expand(
        [
            ("bf16_stream", bf16_stream_compress_hook),
            ("fp16_stream", fp16_stream_compress_hook),
        ]
    )
    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
    def test_stream_precision_selects_stream_hook_on_cuda(
        self, precision: str, expected_hook: CommHook
    ) -> None:
        # The precision alone picks the hook, which is what makes stream
        # placement separable from the wire precision. Only the device has to
        # be CUDA; the hooks do not care what backend `pg` uses.
        wrapper = DefaultDataParallelWrapper(allreduce_comm_precision=precision)
        selected = wrapper._select_allreduce_comm_hook(self.pg, torch.device("cuda:0"))
        assert selected is not None
        self.assertIs(selected[1], expected_hook)
        self.assertIsInstance(selected[0], StreamCompressState)

    def test_stream_variants_are_a_noop_off_cuda(self) -> None:
        # There is no stream to create off CUDA, so nothing is registered. The
        # warning is the only signal the caller gets, so it is part of the
        # contract.
        for precision in ("bf16_stream", "fp16_stream"):
            with self.assertLogs(level="WARNING") as logs:
                self.assertIsNone(self._hook_for(precision))
            self.assertTrue(
                any("No comm hook was registered" in line for line in logs.output),
                logs.output,
            )

    def test_stock_precisions_register_with_no_state(self) -> None:
        # torchrec has always passed None here, which the stock hooks read as
        # WORLD. Keep that until it is changed deliberately.
        for precision in ("bf16", "fp16"):
            wrapper = DefaultDataParallelWrapper(allreduce_comm_precision=precision)
            selected = wrapper._select_allreduce_comm_hook(self.pg, self.device)
            assert selected is not None
            self.assertIsNone(selected[0])

    def test_no_precision_means_no_hook(self) -> None:
        self.assertIsNone(self._hook_for(None))
        # Unrecognized values stay a no-op, matching the previous behavior.
        self.assertIsNone(self._hook_for("bfloat16"))

    def test_stream_is_created_lazily(self) -> None:
        # Nothing is allocated until the hook first runs, so a registration that
        # never sees a backward costs no stream.
        state = StreamCompressState(process_group=self.pg)
        self.assertIsNone(state.stream)
