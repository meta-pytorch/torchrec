#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch


def select_embeddings(
    embeddings: torch.Tensor,
    reverse_indices: torch.Tensor,
    use_gather_select: bool = False,
    use_sorted_select: bool = False,
) -> torch.Tensor:
    """Expand embedding rows using the configured selection implementation.

    With both flags enabled, gather handles index counts up to twice the input
    row count; larger expansions use sorted selection.
    """
    if use_sorted_select and (
        not use_gather_select or reverse_indices.numel() > 2 * embeddings.size(0)
    ):
        from torchrec.modules.sorted_index_select import sorted_index_select

        return sorted_index_select(embeddings, reverse_indices)
    if use_gather_select:
        expanded_indices = reverse_indices.unsqueeze(1).expand(-1, embeddings.size(-1))
        return torch.gather(embeddings, 0, expanded_indices)
    return torch.index_select(embeddings, 0, reverse_indices)
