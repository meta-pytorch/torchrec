#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pad uneven TPU all-to-all inputs for KJT and pooled embedding distribution.

Callers keep TPU availability checks at their existing integration points. KJT and
variable-batch embedding split sizes are data-dependent, so this module uses a small
metadata reduction to find padding sizes shared by every rank. Earlier TorchRec split
exchanges provide each rank with its local send and receive splits, but not the full split
matrix or its global maxima.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from torchrec.pt2.checks import is_torchdynamo_compiling


def _pad_dim0(tensor: torch.Tensor, size: int) -> torch.Tensor:
    rows = tensor.shape[0]
    if rows == size:
        return tensor
    padding = torch.zeros(
        (size - rows, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=0)


def padded_all_to_all_single(
    group: dist.ProcessGroup,
    input_tensor: torch.Tensor,
    input_split_sizes: Sequence[int],
    output_split_sizes: Sequence[int],
    padded_split_size: int,
    *,
    even_all_to_all: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Implement uneven dim-0 all-to-all using an even padded collective.

    KJT uses the default raw collective. Pooled embeddings inject TorchRec's existing
    autograd-aware even collective to preserve backward behavior.
    """
    world_size = group.size()
    if input_tensor.ndim == 0:
        raise ValueError("all-to-all input must have at least one dimension")
    if len(input_split_sizes) != world_size or len(output_split_sizes) != world_size:
        raise ValueError("split-size lists must match the process-group size")
    if any(size < 0 for size in (*input_split_sizes, *output_split_sizes)):
        raise ValueError("split sizes must be nonnegative")
    if sum(input_split_sizes) != input_tensor.shape[0]:
        raise ValueError("input split sizes must sum to the input's dim-0 size")
    if padded_split_size < max(1, *input_split_sizes, *output_split_sizes):
        raise ValueError("padded split size is smaller than a real segment")

    padded_input = torch.cat(
        [
            _pad_dim0(segment, padded_split_size)
            for segment in input_tensor.split(list(input_split_sizes), dim=0)
        ],
        dim=0,
    )
    if even_all_to_all is None:
        padded_output = torch.empty_like(padded_input)
        dist.all_to_all_single(
            padded_output,
            padded_input.contiguous(),
            group=group,
        )
    else:
        padded_output = even_all_to_all(padded_input, padded_split_size)

    expected_shape = (padded_split_size * world_size, *input_tensor.shape[1:])
    if padded_output.shape != expected_shape:
        raise ValueError(
            f"even all-to-all returned shape {tuple(padded_output.shape)}, "
            f"expected {expected_shape}"
        )

    source_blocks = padded_output.split(padded_split_size, dim=0)
    return torch.cat(
        [
            source_block[:output_size]
            for source_block, output_size in zip(source_blocks, output_split_sizes)
        ],
        dim=0,
    )


def _global_padding_metadata(
    group: dist.ProcessGroup,
    local_maxima: Sequence[int],
    locally_uneven: bool,
    device: torch.device,
) -> tuple[list[int], bool]:
    local_metadata = torch.tensor(
        [*local_maxima, int(locally_uneven)],
        dtype=torch.int64,
        device=device,
    )
    metadata_size = local_metadata.numel()
    gathered = torch.empty(
        metadata_size * group.size(),
        dtype=local_metadata.dtype,
        device=local_metadata.device,
    )
    dist.all_gather_into_tensor(gathered, local_metadata, group=group)
    gathered = gathered.view(group.size(), metadata_size)
    global_maxima = gathered[:, :-1].amax(dim=0).tolist()
    globally_uneven = bool(gathered[:, -1].amax().item())
    return global_maxima, globally_uneven


def maybe_kjt_a2a_uneven_tpu(
    group: dist.ProcessGroup,
    input_tensors: Sequence[torch.Tensor],
    input_splits: Sequence[Sequence[int]],
    output_splits: Sequence[Sequence[int]],
    device: torch.device,
) -> list[torch.Tensor] | None:
    """Return padded uneven KJT A2A outputs, or ``None`` for the native path."""
    if is_torchdynamo_compiling() or group.size() <= 1 or not input_tensors:
        return None
    if not (len(input_tensors) == len(input_splits) == len(output_splits)):
        raise ValueError("KJT tensors and split lists must have matching lengths")

    local_maxima = [
        max(1, *input_split, *output_split)
        for input_split, output_split in zip(input_splits, output_splits)
    ]
    locally_uneven = any(
        len({*input_split, *output_split}) > 1
        for input_split, output_split in zip(input_splits, output_splits)
    )
    padded_split_sizes, globally_uneven = _global_padding_metadata(
        group, local_maxima, locally_uneven, device
    )
    if not globally_uneven:
        return None

    return [
        padded_all_to_all_single(
            group,
            input_tensor.to(device),
            input_split,
            output_split,
            padded_split_size,
        )
        for input_tensor, input_split, output_split, padded_split_size in zip(
            input_tensors,
            input_splits,
            output_splits,
            padded_split_sizes,
        )
    ]


def maybe_all2all_pooled_uneven_tpu(
    group: dist.ProcessGroup,
    input_embeddings: torch.Tensor,
    batch_size_per_rank: Sequence[int],
    dim_sum_per_rank: Sequence[int],
    *,
    has_codecs: bool,
    even_all_to_all: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor | None:
    """Return padded uneven pooled A2A output, or ``None`` for the native path."""
    if has_codecs or is_torchdynamo_compiling() or group.size() <= 1:
        return None
    if (
        len(batch_size_per_rank) != group.size()
        or len(dim_sum_per_rank) != group.size()
    ):
        raise ValueError("pooled metadata must match the process-group size")
    if any(size < 0 for size in (*batch_size_per_rank, *dim_sum_per_rank)):
        raise ValueError("pooled metadata must be nonnegative")
    if len(set(batch_size_per_rank)) == 1 and len(set(dim_sum_per_rank)) == 1:
        return None

    rank = group.rank()
    batch_size = batch_size_per_rank[rank]
    local_dim = dim_sum_per_rank[rank]
    if input_embeddings.shape != (sum(batch_size_per_rank), local_dim):
        raise ValueError("input embedding shape does not match pooled A2A metadata")

    input_split_sizes = [local_dim * size for size in batch_size_per_rank]
    output_split_sizes = [batch_size * dim for dim in dim_sum_per_rank]
    padded_split_size = max(batch_size_per_rank) * max(dim_sum_per_rank)
    sharded_output = padded_all_to_all_single(
        group,
        input_embeddings.reshape(-1),
        input_split_sizes,
        output_split_sizes,
        max(1, padded_split_size),
        even_all_to_all=even_all_to_all,
    )
    outputs_by_rank = sharded_output.split(output_split_sizes)
    return torch.cat(
        [
            output.view(batch_size, dim)
            for output, dim in zip(outputs_by_rank, dim_sum_per_rank)
        ],
        dim=1,
    )


def maybe_variable_batch_all2all_pooled_uneven_tpu(
    group: dist.ProcessGroup,
    input_embeddings: torch.Tensor,
    batch_size_per_rank_per_feature: Sequence[Sequence[int]],
    batch_size_per_feature_pre_a2a: Sequence[int],
    emb_dim_per_rank_per_feature: Sequence[Sequence[int]],
    *,
    has_codecs: bool,
    even_all_to_all: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor | None:
    """Return padded uneven VBE A2A output, or None for the native path."""
    world_size = group.size()
    if has_codecs or is_torchdynamo_compiling() or world_size <= 1:
        return None
    if (
        len(batch_size_per_rank_per_feature) != world_size
        or len(emb_dim_per_rank_per_feature) != world_size
    ):
        raise ValueError("VBE metadata must match the process-group size")

    rank = group.rank()
    input_split_sizes = [
        sum(
            batch_size * embedding_dim
            for batch_size, embedding_dim in zip(
                batch_size_per_rank_per_feature[destination_rank],
                emb_dim_per_rank_per_feature[rank],
            )
        )
        for destination_rank in range(world_size)
    ]

    feature_index = 0
    output_split_sizes: list[int] = []
    for embedding_dims in emb_dim_per_rank_per_feature:
        output_split_size = 0
        for embedding_dim in embedding_dims:
            output_split_size += (
                batch_size_per_feature_pre_a2a[feature_index] * embedding_dim
            )
            feature_index += 1
        output_split_sizes.append(output_split_size)

    if sum(input_split_sizes) != input_embeddings.numel():
        raise ValueError("VBE input splits must sum to the input size")

    local_maximum = max(1, *input_split_sizes, *output_split_sizes)
    locally_uneven = len({*input_split_sizes, *output_split_sizes}) > 1
    (padded_split_size,), globally_uneven = _global_padding_metadata(
        group,
        [local_maximum],
        locally_uneven,
        input_embeddings.device,
    )
    if not globally_uneven:
        return None

    return padded_all_to_all_single(
        group,
        input_embeddings.reshape(-1),
        input_split_sizes,
        output_split_sizes,
        padded_split_size,
        even_all_to_all=even_all_to_all,
    )
