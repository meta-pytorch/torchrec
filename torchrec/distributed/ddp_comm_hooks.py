#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DDP gradient communication hooks, counterparts to the ones PyTorch ships in
`torch.distributed.algorithms.ddp_comm_hooks`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)

CommHook = Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]


@dataclass
class StreamCompressState:
    """
    State registered with DDP for the `*_stream_compress_hook` family.

    Attributes:
        process_group: the group to allreduce the bucket over.
        stream: the decompression stream, created on first use and reused for
            every bucket of every iteration thereafter.
    """

    process_group: dist.ProcessGroup
    stream: Optional[torch.cuda.Stream] = None


def _stream_compress_hook(
    dtype: torch.dtype,
    state: StreamCompressState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Allreduces a gradient bucket in `dtype`, then casts it back to FP32.

    Numerically equivalent to PyTorch's ``bf16_compress_hook`` /
    ``fp16_compress_hook``, but the cast back to FP32 runs on a stream owned by
    `state` rather than one drawn from the global Future pool.

    The stock hooks chain the cast-back with ``Future.then``, which runs the
    callback on a stream taken from PyTorch's 32-entry global pool, so a
    different logical stream is used for every bucket of every iteration, each
    blocked on that bucket's collective by a cross-stream event. Owning the
    stream pins the copy-back to a single one instead, the same one on every
    iteration.

    Args:
        dtype: the reduced precision to allreduce in.
        state: the hook state, holding the process group and the decompression
            stream.
        bucket: the gradient bucket to reduce.

    Returns:
        A future completed with the bucket's FP32 buffer, carrying the
        completion event of the copy-back.
    """
    process_group = state.process_group

    if torch.compiler.is_compiling():
        # Streams and record_stream are not traceable. `_compress_hook` is the
        # dtype-parameterized body the stock hooks are thin wrappers over, and
        # it carries its own functional-collectives path for this case. Defer to
        # it rather than restating that contract here.
        return ddp_default_hooks._compress_hook(dtype, process_group, bucket)

    buffer = bucket.buffer()

    # Divide before the collective so FP16 cannot overflow while summing.
    compressed_tensor = buffer.to(dtype).div_(process_group.size())
    work = dist.all_reduce(compressed_tensor, group=process_group, async_op=True)

    if state.stream is None:
        state.stream = torch.cuda.Stream(device=buffer.device)
    stream = state.stream

    with torch.cuda.stream(stream):
        # Inside the stream context so the copy-back below is ordered after the
        # collective rather than racing it.
        work.wait()

        # Decompress in place to reduce the peak memory.
        buffer.copy_(compressed_tensor)

        # `compressed_tensor` is freed on return while the copy is still pending
        # on `stream`, so the caching allocator must not reuse its block before
        # the copy completes.
        compressed_tensor.record_stream(stream)

        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future(
            devices=[buffer.device]
        )

        # Completing under `stream` records the event that the DDP reducer waits
        # on before it reads the bucket back.
        fut.set_result(buffer)

    return fut


def fp16_stream_compress_hook(
    state: StreamCompressState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Runs `fp16_compress_hook`'s reduction with the copy-back on a dedicated stream.

    Args:
        state: the hook state, holding the process group and the decompression
            stream.
        bucket: the gradient bucket to reduce.

    Returns:
        A future completed with the bucket's FP32 buffer.
    """
    return _stream_compress_hook(torch.float16, state, bucket)


def bf16_stream_compress_hook(
    state: StreamCompressState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Runs `bf16_compress_hook`'s reduction with the copy-back on a dedicated stream.

    Args:
        state: the hook state, holding the process group and the decompression
            stream.
        bucket: the gradient bucket to reduce.

    Returns:
        A future completed with the bucket's FP32 buffer.
    """
    return _stream_compress_hook(torch.bfloat16, state, bucket)
