#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


_PERMUTE_PARAM_SIZE = 6


# Triton TR001: decouple batch tiling from warp count so 16-bit types keep
# enough copy work per thread to amortize indexing and memory instructions.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_BATCH": 1, "BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_BATCH": 2, "BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_BATCH": 4, "BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_BATCH": 4, "BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_BATCH": 8, "BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_BATCH": 8, "BLOCK_SIZE": 128}, num_warps=8),
    ],
    key=["batch_size", "num_permutes"],
)
@triton.jit
def _permute_multi_embedding_kernel(
    example_ptr,
    input0_ptr,
    input1_ptr,
    output0_ptr,
    output1_ptr,
    input_ptrs,
    output_ptrs,
    permutes_ptr,
    in_shapes_ptr,
    out_shapes_ptr,
    batch_size,
    num_permutes,
    PERMUTE_PARAM_SIZE: tl.constexpr,
    SPECIALIZED_POINTERS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    program_id = tl.program_id(0)
    batch_block = program_id // num_permutes
    permute_id = program_id % num_permutes

    permute = permutes_ptr + permute_id * PERMUTE_PARAM_SIZE
    in_tensor = tl.load(permute)
    out_tensor = tl.load(permute + 1)
    in_offset = tl.load(permute + 2).to(tl.int64)
    out_offset = tl.load(permute + 3).to(tl.int64)
    length = tl.load(permute + 4)
    in_length = tl.load(in_shapes_ptr + in_tensor).to(tl.int64)
    out_length = tl.load(out_shapes_ptr + out_tensor).to(tl.int64)
    if SPECIALIZED_POINTERS:
        input_ptr = tl.where(in_tensor == 0, input0_ptr, input1_ptr)
        output_ptr = tl.where(out_tensor == 0, output0_ptr, output1_ptr)
    else:
        input_ptr = tl.load(input_ptrs + in_tensor).to(example_ptr.dtype, bitcast=True)
        output_ptr = tl.load(output_ptrs + out_tensor).to(
            example_ptr.dtype, bitcast=True
        )

    offsets = tl.arange(0, BLOCK_BATCH * BLOCK_SIZE)
    batch_offsets = offsets // BLOCK_SIZE
    column_offsets = offsets % BLOCK_SIZE
    batch_ids = batch_block * BLOCK_BATCH + batch_offsets
    for block_start in range(0, length, BLOCK_SIZE):
        columns = block_start + column_offsets
        mask = (batch_ids < batch_size) & (columns < length)
        value = tl.load(
            input_ptr + batch_ids * in_length + in_offset + columns,
            mask=mask,
        )
        tl.store(
            output_ptr + batch_ids * out_length + out_offset + columns,
            value,
            mask=mask,
        )


# Triton TR001: decouple batch tiling from warp count so 16-bit types keep
# enough copy work per thread to amortize indexing and memory instructions.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_BATCH": 1, "BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_BATCH": 2, "BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_BATCH": 4, "BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_BATCH": 4, "BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_BATCH": 8, "BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_BATCH": 8, "BLOCK_SIZE": 128}, num_warps=8),
    ],
    key=["batch_size", "num_permutes"],
)
@triton.jit
def _permute_multi_embedding_backward_kernel(
    example_ptr,
    grad_output0_ptr,
    grad_output1_ptr,
    grad_input0_ptr,
    grad_input1_ptr,
    grad_output_ptrs,
    grad_input_ptrs,
    permutes_ptr,
    in_shapes_ptr,
    out_shapes_ptr,
    batch_size,
    num_permutes,
    PERMUTE_PARAM_SIZE: tl.constexpr,
    SPECIALIZED_POINTERS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    program_id = tl.program_id(0)
    batch_block = program_id // num_permutes
    permute_id = program_id % num_permutes

    permute = permutes_ptr + permute_id * PERMUTE_PARAM_SIZE
    next_permute = tl.load(permute + 5)
    if next_permute < 0:
        return

    in_tensor = tl.load(permute)
    out_tensor = tl.load(permute + 1)
    in_offset = tl.load(permute + 2).to(tl.int64)
    out_offset = tl.load(permute + 3).to(tl.int64)
    length = tl.load(permute + 4)
    in_length = tl.load(in_shapes_ptr + in_tensor).to(tl.int64)
    out_length = tl.load(out_shapes_ptr + out_tensor).to(tl.int64)
    if SPECIALIZED_POINTERS:
        grad_output_ptr = tl.where(out_tensor == 0, grad_output0_ptr, grad_output1_ptr)
        grad_input_ptr = tl.where(in_tensor == 0, grad_input0_ptr, grad_input1_ptr)
    else:
        grad_output_ptr = tl.load(grad_output_ptrs + out_tensor).to(
            example_ptr.dtype, bitcast=True
        )
        grad_input_ptr = tl.load(grad_input_ptrs + in_tensor).to(
            example_ptr.dtype, bitcast=True
        )

    offsets = tl.arange(0, BLOCK_BATCH * BLOCK_SIZE)
    batch_offsets = offsets // BLOCK_SIZE
    column_offsets = offsets % BLOCK_SIZE
    batch_ids = batch_block * BLOCK_BATCH + batch_offsets
    for block_start in range(0, length, BLOCK_SIZE):
        columns = block_start + column_offsets
        mask = (batch_ids < batch_size) & (columns < length)
        value = tl.load(
            grad_output_ptr + batch_ids * out_length + out_offset + columns,
            mask=mask,
        )
        duplicate = next_permute
        while duplicate > 0 and duplicate < num_permutes:
            duplicate_permute = permutes_ptr + duplicate * PERMUTE_PARAM_SIZE
            duplicate_out_tensor = tl.load(duplicate_permute + 1)
            duplicate_out_offset = tl.load(duplicate_permute + 3).to(tl.int64)
            duplicate_out_length = tl.load(out_shapes_ptr + duplicate_out_tensor).to(
                tl.int64
            )
            if SPECIALIZED_POINTERS:
                duplicate_ptr = tl.where(
                    duplicate_out_tensor == 0,
                    grad_output0_ptr,
                    grad_output1_ptr,
                )
            else:
                duplicate_ptr = tl.load(grad_output_ptrs + duplicate_out_tensor).to(
                    example_ptr.dtype, bitcast=True
                )
            value += tl.load(
                duplicate_ptr
                + batch_ids * duplicate_out_length
                + duplicate_out_offset
                + columns,
                mask=mask,
            )
            duplicate = -tl.load(duplicate_permute + 5)
        tl.store(
            grad_input_ptr + batch_ids * in_length + in_offset + columns,
            value,
            mask=mask,
        )


def _validate_inputs(
    values: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: list[int],
) -> None:
    if not values:
        raise ValueError("values must contain at least one tensor")
    first = values[0]
    if first.device.type != "cuda":
        raise ValueError("triton_permute_multi_embedding requires CUDA tensors")
    if any(
        value.device != first.device
        or value.dtype != first.dtype
        or value.ndim != 2
        or value.shape[0] != first.shape[0]
        for value in values
    ):
        raise ValueError(
            "values must be 2D tensors with matching device, dtype, and batch"
        )
    if permutes.device != first.device or permutes.ndim != 2 or permutes.shape[1] != 6:
        raise ValueError("permutes must be a device tensor with shape [P, 6]")
    if in_shapes.device != first.device or in_shapes.numel() != len(values):
        raise ValueError("in_shapes must contain one entry per input tensor")
    if out_shapes.device != first.device or out_shapes.numel() != len(out_lengths):
        raise ValueError("out_shapes and out_lengths must describe the same outputs")


def _pointer_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Citrine C3 exception: pointer values originate on the host. Pinned staging
    # keeps the required H2D metadata copy asynchronous, matching FBGEMM.
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        pin_memory=True,
    ).to(tensors[0].device, non_blocking=True)


def _forward_impl(
    values: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: list[int],
) -> list[torch.Tensor]:
    _validate_inputs(values, permutes, in_shapes, out_shapes, out_lengths)
    contiguous_values = [value.contiguous() for value in values]
    outputs = [
        values[0].new_empty((values[0].shape[0], out_length))
        for out_length in out_lengths
    ]
    if permutes.shape[0] == 0:
        return outputs
    specialized_pointers = len(contiguous_values) == 2 and len(outputs) == 2
    input_ptrs = (
        permutes if specialized_pointers else _pointer_tensor(contiguous_values)
    )
    output_ptrs = permutes if specialized_pointers else _pointer_tensor(outputs)

    def grid(meta: dict[str, Any]) -> tuple[Any, ...]:
        return (
            triton.cdiv(values[0].shape[0], meta["BLOCK_BATCH"]) * permutes.shape[0],
        )

    _permute_multi_embedding_kernel[grid](
        values[0],
        contiguous_values[0],
        contiguous_values[1] if specialized_pointers else contiguous_values[0],
        outputs[0],
        outputs[1] if specialized_pointers else outputs[0],
        input_ptrs,
        output_ptrs,
        permutes,
        in_shapes,
        out_shapes,
        values[0].shape[0],
        permutes.shape[0],
        PERMUTE_PARAM_SIZE=_PERMUTE_PARAM_SIZE,
        SPECIALIZED_POINTERS=specialized_pointers,
    )
    return outputs


def _backward_impl(
    grad_outputs: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    in_lengths: list[int],
) -> list[torch.Tensor]:
    contiguous_grads = [grad.contiguous() for grad in grad_outputs]
    grad_inputs = [
        grad_outputs[0].new_empty((grad_outputs[0].shape[0], in_length))
        for in_length in in_lengths
    ]
    if permutes.shape[0] == 0:
        return grad_inputs
    specialized_pointers = len(contiguous_grads) == 2 and len(grad_inputs) == 2
    grad_output_ptrs = (
        permutes if specialized_pointers else _pointer_tensor(contiguous_grads)
    )
    grad_input_ptrs = permutes if specialized_pointers else _pointer_tensor(grad_inputs)

    def grid(meta: dict[str, Any]) -> tuple[Any, ...]:
        return (
            triton.cdiv(grad_outputs[0].shape[0], meta["BLOCK_BATCH"])
            * permutes.shape[0],
        )

    _permute_multi_embedding_backward_kernel[grid](
        grad_outputs[0],
        contiguous_grads[0],
        contiguous_grads[1] if specialized_pointers else contiguous_grads[0],
        grad_inputs[0],
        grad_inputs[1] if specialized_pointers else grad_inputs[0],
        grad_output_ptrs,
        grad_input_ptrs,
        permutes,
        in_shapes,
        out_shapes,
        grad_outputs[0].shape[0],
        permutes.shape[0],
        PERMUTE_PARAM_SIZE=_PERMUTE_PARAM_SIZE,
        SPECIALIZED_POINTERS=specialized_pointers,
    )
    return grad_inputs


@torch.library.custom_op(
    "torchrec::triton_permute_multi_embedding",
    mutates_args=(),
    schema="(Tensor[] values, Tensor permutes, Tensor in_shapes, Tensor out_shapes, SymInt[] out_lengths) -> Tensor[]",
)
def triton_permute_multi_embedding(
    values: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: list[int],
) -> list[torch.Tensor]:
    return _forward_impl(values, permutes, in_shapes, out_shapes, out_lengths)


@triton_permute_multi_embedding.register_fake
def _fake_triton_permute_multi_embedding(
    values: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: list[int],
) -> list[torch.Tensor]:
    return [
        values[0].new_empty((values[0].shape[0], out_length))
        for out_length in out_lengths
    ]


@torch.library.custom_op(
    "torchrec::triton_permute_multi_embedding_backward",
    mutates_args=(),
    schema="(Tensor[] grad_outputs, Tensor permutes, Tensor in_shapes, Tensor out_shapes, SymInt[] in_lengths) -> Tensor[]",
)
def _triton_permute_multi_embedding_backward(
    grad_outputs: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    in_lengths: list[int],
) -> list[torch.Tensor]:
    return _backward_impl(grad_outputs, permutes, in_shapes, out_shapes, in_lengths)


@_triton_permute_multi_embedding_backward.register_fake
def _fake_triton_permute_multi_embedding_backward(
    grad_outputs: list[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    in_lengths: list[int],
) -> list[torch.Tensor]:
    return [
        grad_outputs[0].new_empty((grad_outputs[0].shape[0], in_length))
        for in_length in in_lengths
    ]


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    values, permutes, in_shapes, out_shapes, _ = inputs
    ctx.save_for_backward(permutes, in_shapes, out_shapes)
    ctx.in_lengths = [value.shape[1] for value in values]


def _backward(
    ctx: Any, grad_outputs: list[torch.Tensor]
) -> tuple[list[torch.Tensor], None, None, None, None]:
    permutes, in_shapes, out_shapes = ctx.saved_tensors
    grad_inputs = _triton_permute_multi_embedding_backward(
        grad_outputs,
        permutes,
        in_shapes,
        out_shapes,
        ctx.in_lengths,
    )
    return grad_inputs, None, None, None, None


triton_permute_multi_embedding.register_autograd(
    _backward,
    setup_context=_setup_context,
)
