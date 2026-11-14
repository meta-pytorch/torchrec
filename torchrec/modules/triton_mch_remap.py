#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import triton  # @manual
import triton.language as tl  # @manual


@triton.jit
def _mch_remap_kernel(
    sorted_raw_ids,
    remapped_ids_mapping,
    values,
    output,
    num_slots: tl.constexpr,
    num_values: tl.constexpr,
    default_index: tl.constexpr,
    search_steps: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    raw_ids = tl.load(values + offsets, offsets < num_values, other=0)
    low = tl.full((block_size,), 0, tl.int32)
    high = tl.full((block_size,), num_slots - 1, tl.int32)

    for _ in range(search_steps):
        mid = (low + high) // 2
        found = tl.load(sorted_raw_ids + mid)
        advance = found < raw_ids
        low = tl.where(advance, mid + 1, low)
        high = tl.where(advance, high, mid)

    found = tl.load(sorted_raw_ids + low)
    mapped = tl.load(remapped_ids_mapping + low)
    tl.store(
        output + offsets,
        tl.where(found == raw_ids, mapped, default_index),
        offsets < num_values,
    )


def mch_remap_cuda(
    sorted_raw_ids: torch.Tensor,
    remapped_ids_mapping: torch.Tensor,
    values: torch.Tensor,
    default_index: int,
) -> torch.Tensor:
    values = values.contiguous()
    output = torch.empty_like(values)
    if values.numel() == 0:
        return output

    block_size = 256
    _mch_remap_kernel[(triton.cdiv(values.numel(), block_size),)](
        sorted_raw_ids,
        remapped_ids_mapping,
        values,
        output,
        sorted_raw_ids.numel(),
        values.numel(),
        default_index,
        (sorted_raw_ids.numel() - 1).bit_length(),
        block_size,
    )
    return output
