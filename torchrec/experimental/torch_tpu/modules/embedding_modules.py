#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import Enum
from itertools import accumulate
from typing import cast, List, Optional, Tuple

import torch
from torch import nn

# Imported is needed: defines torchrec::embedding_lookup
from torchrec.experimental.torch_tpu.pallas import ops  # noqa: F401


class PooledLookupKernel(Enum):
    PADDED = "padded"
    OFFSET = "offset"
    BATCHED_OFFSET = "batched_offset"


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
        kernel (PooledLookupKernel): Pallas implementation used for pooled lookup.

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
        *,
        kernel: PooledLookupKernel = PooledLookupKernel.OFFSET,
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
        self._kernel = kernel

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

    def _padded_lookup(
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

    def _offset_lookup(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.ops.torchrec.embedding_pooled_lookup_offset(
            input.to(dtype=torch.int32),
            offsets.to(dtype=torch.int32),
            self._weight,
            self._embedding_dim,
        )
        if self.mode == "mean":
            lengths = offsets[1:] - offsets[:-1]
            denominator = lengths.clamp(min=1).unsqueeze(1).to(out.dtype)
            out = out / denominator
        return out

    def forward(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Pool embedding rows using the configured Pallas kernel.

        Args:
            input (torch.Tensor): Row indices on the embedding table's device.
            offsets (torch.Tensor): Bag boundaries on the embedding table's device.

        Returns:
            torch.Tensor: Pooled rows with shape
                ``[offsets.numel() - 1, embedding_dim]``.
        """
        device = self._weight.device
        if input.device != device or offsets.device != device:
            raise ValueError(
                "input and offsets must already be on the embedding table's device"
            )
        if offsets.numel() == 1:
            return self._weight[:0]
        if self._kernel is PooledLookupKernel.OFFSET:
            return self._offset_lookup(input, offsets)
        return self._padded_lookup(input, offsets)


def _construct_weight_offsets(
    embedding_specs: List[Tuple[int, int]],
) -> Tuple[int, List[int]]:
    """Converts global information about embedding table to local information

    Equivalent to "construct_split_state"

    Returns
    - tpu_size: total size
    - offsets: global offsets of flattened table

    """
    tpu_size = 0
    offsets = []
    for row, dim in embedding_specs:
        offsets.append(tpu_size)
        tpu_size += row * dim
    return (tpu_size, offsets)


def _bucket_index_count(num_indices: int) -> int:
    """Round an id count up to a power-of-two bucket."""
    # Smallest id bucket; below this the padding is free and the extra distinct programs
    # are not worth compiling.
    MIN_INDEX_BUCKET = 1024

    if num_indices <= MIN_INDEX_BUCKET:
        return MIN_INDEX_BUCKET
    return 1 << (num_indices - 1).bit_length()


class PallasTableBatchedEmbeddingBags(nn.Module):
    """Batched pooled lookup over a row-stacked collection of embedding tables.

    Args:
        embedding_specs: Local row count and dimension for each table.
        feature_table_map: Table index used by each feature.
        weights_precision: Weight dtype; currently only torch.float32.
        mode: Pooling mode, either sum or mean.
        device: Device on which weights and lookup buffers are allocated.

    Example::

        module = PallasTableBatchedEmbeddingBags(
            embedding_specs=[(8, 16), (4, 16)],
            device=torch.device("cpu"),
        )
    """

    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],
        feature_table_map: Optional[List[int]] = None,
        weights_precision: torch.dtype = torch.float32,
        mode: str = "sum",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("tpu")
        if mode not in {"sum", "mean"}:
            raise ValueError(f"mode must be 'sum' or 'mean', got {mode}")
        self.mode = mode
        self.weights_precision = weights_precision
        assert (
            self.weights_precision == torch.float32
        ), "TPU lookup only supports float32 for now."

        self.embedding_specs = embedding_specs
        rows, dims = zip(*embedding_specs)
        if len(set(dims)) != 1:
            raise ValueError(
                "All TPU-batched tables must have the same local dimension."
            )

        T_ = len(self.embedding_specs)
        assert T_ > 0

        self.size, self.offsets = _construct_weight_offsets(embedding_specs)
        self.feature_table_map: list[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        table_row_offsets = [0] + list(accumulate(rows))
        feature_row_offsets = [
            table_row_offsets[table] for table in self.feature_table_map
        ]
        self.register_buffer(
            "weights_offsets",
            torch.tensor(
                feature_row_offsets,
                dtype=torch.int32,
                device=self.device,
            ),
        )
        self._cached_row_offsets: Optional[torch.Tensor]
        self.register_buffer("_cached_row_offsets", None, persistent=False)
        self._cached_row_offsets_batch_size: int = -1
        feature_weight_offsets = [
            self.offsets[table_idx] for table_idx in self.feature_table_map
        ]

        feature_dims = [dims[t] for t in self.feature_table_map]  # Needed for VBE
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(dims)
        # TODO ADD ASSERT TO MAKE SURE ALL DIMENSIONS SAME
        hash_size_cumsum = [0] + list(accumulate(rows))
        self.total_hash_size: int = int(hash_size_cumsum[-1])

        # Construct the weights
        self.weights = self.construct_weights()

        # Register the buffers needed
        self.register_buffer(
            "D_offsets",
            torch.tensor(D_offsets, dtype=torch.int32, device=self.device),
        )
        # Note feature_weight_offsets need ot be int64, but isn't used within the TPU lookup kernel.
        self.register_buffer(
            "feature_weight_offsets",
            torch.tensor(feature_weight_offsets, dtype=torch.int64, device=self.device),
        )

    def construct_weights(self) -> nn.Parameter:
        rows, dims = zip(*self.embedding_specs)

        # TODO: make this similar to how BatchedFused is doing with normal
        return nn.Parameter(
            torch.zeros(
                size=(sum(rows), max(dims)),
                dtype=self.weights_precision,
                device=self.device,
            )
        )

    def split_embedding_weights(self) -> list[torch.Tensor]:
        splits = []
        for t, (rows, dim) in enumerate(self.embedding_specs):
            offset = self.offsets[t]

            weights = self.weights
            if weights.dim() == 2:
                weights = weights.flatten()
            split_weight = weights.detach()[offset : offset + rows * dim].view(
                rows, dim
            )
            if self.weights.grad is not None:
                split_weight.grad = (
                    self.weights.grad.detach()
                    .flatten()[offset : offset + rows * dim]
                    .view(rows, dim)
                )
            splits.append(split_weight)
        return splits

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        # per_sample_weights: Tensor | None = None,
        # feature_requires_grad: Tensor | None = None,
        # batch_size_per_feature_per_rank: list[list[int]] | None = None,
        # total_unique_indices: int | None = None,
        # hash_zch_identities: Tensor | None = None,
        # hash_zch_runtime_meta: Tensor | None = None,
        # vbe_output: Tensor | None = None,
        # vbe_output_offsets: Tensor | None = None,
    ) -> torch.Tensor:
        """Pool feature-major jagged IDs and return batch-major embeddings.

        Args:
            indices: Flattened local row IDs for all feature bags.
            offsets: Boundaries for feature-major bags.

        Returns:
            Tensor shaped as batch size by features times embedding dimension.
        """
        num_features = len(self.feature_table_map)
        batch_size = (offsets.numel() - 1) // num_features  # ASSUMES NON-VBE

        # Instead of having an offset per feature, repeat it to have one per bag.
        # Cache the result because fixed-batch workloads otherwise rebuild the same
        # device tensor on every forward.
        row_offsets = self._cached_row_offsets
        if row_offsets is None or self._cached_row_offsets_batch_size != batch_size:
            row_offsets = torch.repeat_interleave(
                cast(torch.Tensor, self.weights_offsets), batch_size
            )
            self._cached_row_offsets = row_offsets
            self._cached_row_offsets_batch_size = batch_size

        # `offsets` and `row_offsets` are a fixed [F * B (+1)] for non-VBE, but the id
        # count moves with every batch, so an unbucketed call retraces and recompiles
        # the kernel on every step.
        # TODO: We need to optimize this the right number of bucket
        bucket = _bucket_index_count(indices.numel())
        if bucket != indices.numel():
            indices = nn.functional.pad(indices, (0, bucket - indices.numel()))

        pooled_embs = torch.ops.torchrec.embedding_pooled_batched_lookup_offset(
            indices, offsets, self.weights, row_offsets, self.max_D
        )
        if self.mode == "mean":
            lengths = offsets[1:] - offsets[:-1]
            pooled_embs = pooled_embs / lengths.clamp(min=1).unsqueeze(1).to(
                pooled_embs.dtype
            )

        return (
            pooled_embs.view(num_features, batch_size, self.max_D)
            .permute(1, 0, 2)
            .reshape(batch_size, num_features * self.max_D)
        )
