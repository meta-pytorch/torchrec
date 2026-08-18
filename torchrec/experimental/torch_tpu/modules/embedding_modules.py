#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import torch

# Imported is needed: defines torchrec::embedding_lookup
from torchrec.experimental.torch_tpu.pallas import ops  # noqa: F401


class TPUEmbeddingUnfused(torch.nn.Module):
    """Single-table TPU embedding lookup backed by a Pallas kernel.

    Drop-in replacement for ``nn.Embedding``so that ``torchrec``'s
    ``EmbeddingCollection`` can use it for the ``"tpu"`` device.
    Performs an unfused gather (no pooling); the backward/optimizer
    is not fused.

    Args:
        num_embeddings (int): number of rows in the embedding table.
        embedding_dim (int): width of each embedding row.
        device: device the table lives on (e.g. ``"tpu"``).
        dtype (torch.dtype): dtype of the embedding table.

    Example::

        emb_module = TPUEmbeddingUnfused(
            num_embeddings=10,
            embedding_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._device = device
        self._dtype = dtype
        # Just use a zero tensor
        self._weight = torch.nn.Parameter(
            torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        self.output_dtype = dtype

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._weight

    def split_embedding_weights(self) -> List[torch.Tensor]:
        """Single-table weight list, matching the FBGEMM emb_module interface.

        Lets ``BatchedTPUEmbedding`` (a ``BaseBatchedEmbedding``) use this module as
        its ``emb_module`` and inherit ``split_embedding_weights``/state handling.
        """
        return [self._weight]

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Gather embedding rows for ``input`` indices via the Pallas kernel.

        Args:
            input (torch.Tensor): 1-D int32 tensor of row indices on the TPU.

        Returns:
            torch.Tensor: gathered rows, shape ``[input.numel(), embedding_dim]``.
        """
        return torch.ops.torchrec.embedding_lookup(
            input, self._weight, self._embedding_dim
        )
