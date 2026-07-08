#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Shared helpers for the CUDA Graphs dense-path work on ``DLRM``: building a small
unsharded model, generating synthetic ``(dense, KJT)`` batches directly on the
target device, and collecting outputs across batches under the CUDA-graph replay
protocol.
"""

from typing import List, Tuple

import torch
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def build_ebc(
    device: torch.device,
    *,
    embedding_dim: int,
    num_sparse_features: int,
    num_embeddings: int,
) -> EmbeddingBagCollection:
    """Build an ``EmbeddingBagCollection`` of ``num_sparse_features`` equal-dim tables."""
    tables = [
        EmbeddingBagConfig(
            name=f"t{i}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[f"f{i}"],
        )
        for i in range(num_sparse_features)
    ]
    return EmbeddingBagCollection(tables=tables, device=device)


def build_dlrm(
    device: torch.device,
    *,
    embedding_dim: int,
    num_sparse_features: int,
    num_embeddings: int,
    dense_in_features: int,
    over_arch_layer_sizes: Tuple[int, ...] = (512, 256, 1),
) -> DLRM:
    """Build an unsharded base ``DLRM`` on ``device``, in eval mode."""
    ebc = build_ebc(
        device,
        embedding_dim=embedding_dim,
        num_sparse_features=num_sparse_features,
        num_embeddings=num_embeddings,
    )
    model = DLRM(
        embedding_bag_collection=ebc,
        dense_in_features=dense_in_features,
        # Final dense_arch layer must equal embedding_dim (DLRM constraint).
        dense_arch_layer_sizes=[embedding_dim * 2, embedding_dim],
        over_arch_layer_sizes=list(over_arch_layer_sizes),
        dense_device=device,
    )
    return model.to(device).eval()


def generate_batch(
    device: torch.device,
    *,
    batch_size: int,
    num_sparse_features: int,
    num_embeddings: int,
    dense_in_features: int,
    pooling_factor: int,
) -> Tuple[torch.Tensor, KeyedJaggedTensor]:
    """Generate one synthetic ``(dense, sparse-KJT)`` batch directly on ``device``."""
    b, f, length = batch_size, num_sparse_features, pooling_factor
    # Citrine C3: create tensors directly on device, never CPU-then-.to(device).
    dense = torch.rand(b, dense_in_features, device=device)
    lengths = torch.full((f * b,), length, dtype=torch.long, device=device)
    values = torch.randint(
        0, num_embeddings, (f * b * length,), dtype=torch.long, device=device
    )
    kjt = KeyedJaggedTensor(
        keys=[f"f{i}" for i in range(f)], values=values, lengths=lengths
    )
    return dense, kjt


def collect_outputs(
    model: DLRM,
    batches: List[Tuple[torch.Tensor, KeyedJaggedTensor]],
) -> List[torch.Tensor]:
    """Run ``model`` over ``batches``, returning one cloned output per batch.

    ``reduce-overhead`` replays a CUDA graph into static buffers, so mark the step
    boundary before each call and clone each output — otherwise a later replay can
    overwrite a result we still hold. For mixed precision, wrap the call in an
    autocast context; the forwards run inside that dynamic scope.
    """
    outputs: List[torch.Tensor] = []
    for dense, kjt in batches:
        torch.compiler.cudagraph_mark_step_begin()
        outputs.append(
            model(dense_features=dense, sparse_features=kjt).detach().clone()
        )
    return outputs
