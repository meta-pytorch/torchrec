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


class TPUEmbeddingBagUnfused(torch.nn.Module):
    """Single-table TPU pooled embedding lookup backed by a Pallas kernel.

    Performs a pooled gather; the backward/optimizer is not fused.

    Args:
        num_embeddings (int): number of rows in the embedding table.
        embedding_dim (int): width of each embedding row; must be a multiple of 16.
        device: device the table lives on (e.g. ``"tpu"``).
        dtype (torch.dtype): dtype of the embedding table.
        mode (str): Either `mean` or `sum`. Default is `mean`.

    Example::

        emb_module = TPUEmbeddingBagUnfused(
            num_embeddings=10,
            embedding_dim=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    """

    _PADDING_SENTINEL: int = 0

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device],
        dtype: torch.dtype,
        mode: str = "mean",
    ) -> None:
        super().__init__()
        if embedding_dim % 16 != 0:
            raise ValueError(
                "TPUEmbeddingBagUnfused requires embedding_dim to be a multiple of 16"
            )
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._device = device
        self._dtype = dtype
        self._weight = torch.nn.Parameter(
            torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        self.output_dtype = dtype
        self.mode = mode

    @property
    def weight(self) -> torch.nn.Parameter:
        return self._weight

    def split_embedding_weights(self) -> List[torch.Tensor]:
        """Single-table weight list, matching the FBGEMM emb_module interface.

        Lets ``BatchedTPUEmbedding`` (a ``BaseBatchedEmbedding``) use this module as
        its ``emb_module`` and inherit ``split_embedding_weights``/state handling.
        """
        return [self._weight]

    def _build_pooled_idx(
        self,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        B: int,
        K: int,
        sentinel: int,
    ) -> torch.Tensor:
        start = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=lengths.device),
                lengths.cumsum(0)[:-1],
            ]
        )
        col = torch.arange(K, dtype=torch.int32, device=lengths.device)
        gather = start[:, None] + col[None, :]  # [B,K]
        valid = col[None, :] < lengths[:, None]
        gathered = indices[torch.clamp(gather, 0, B * K - 1)]
        return torch.where(
            valid,
            gathered,
            torch.tensor(sentinel, dtype=torch.int32, device=indices.device),
        ).reshape(-1)

    def forward(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Pooled embedding rows for ``input`` indices via the Pallas kernel.

        Args:
            input (torch.Tensor): 1-D row indices on the embedding table's device.
            offsets (torch.Tensor): 1-D bag boundaries on the embedding table's device.

        Returns:
            torch.Tensor: pooled rows, shape ``[offsets.numel() - 1, embedding_dim]``.
        """
        B = offsets.numel() - 1
        if B == 0:
            return self._weight[:0]
        # NACC: Number of accumulators to shorten the reduction within the kernel
        # The pooled embeddings are accumulated round-robin into `NACC`
        # vectors, then reduced into one pooled embedding.
        # Keep this value in sync with pallas.pooled_lookup.NACC.
        NACC = 16
        device = self._weight.device
        if input.device != device or offsets.device != device:
            raise ValueError(
                "input and offsets must already be on the embedding table's device"
            )

        # Repack on-device (no host round trip): only the pool size K needs a host value.
        off = offsets.to(dtype=torch.int64)
        lengths = off[1:] - off[:-1]
        max_len = (
            int(lengths.max().item()) if B > 0 else 0
        )  # scalar sync (only host value K needs)
        # K rounded up to the SparseCore id tile (8) so RBLK*K stays tile-aligned
        K = max(((max_len + 7) // 8) * 8, NACC)

        # Build the [B, K] pad-to-max index matrix
        # This replaces an eager `repeat_interleave` + advanced-index scatter
        indices_d = input.to(dtype=torch.int32)
        pad = B * K - int(indices_d.numel())  # a shape read, no device sync
        if pad > 0:
            indices_d = torch.nn.functional.pad(indices_d, (0, pad))
        elif pad < 0:
            indices_d = indices_d[: B * K]
        idx_flat = self._build_pooled_idx(
            indices=indices_d,
            lengths=lengths.to(torch.int32),
            B=B,
            K=K,
            sentinel=self._PADDING_SENTINEL,
        )

        out = torch.ops.torchrec.embedding_pooled_lookup(
            idx=idx_flat, table=self._weight, pool=K
        )

        # Remove the embedding contributions from padded IDs.
        padding_count = (K - lengths).unsqueeze(1).to(out.dtype)
        out = out - padding_count * self._weight[self._PADDING_SENTINEL].unsqueeze(0)
        if self.mode == "mean":
            denom = lengths.clamp(min=1).unsqueeze(1).to(out.dtype)
            out = out / denom
        return out
