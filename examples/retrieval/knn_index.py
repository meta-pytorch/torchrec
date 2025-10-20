#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, Union

import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch


def get_index(
    embedding_dim: int,
    num_centroids: int,
    num_probe: int,
    num_subquantizers: int,
    bits_per_code: int,
    device: Optional[torch.device] = None,
) -> Union[faiss.GpuIndexIVFPQ, faiss.IndexIVFPQ]:
    """
    returns a FAISS IVFPQ index, placed on the device passed in

    Args:
        embedding_dim (int): indexed embedding dimension,
        num_centroids (int): the number of centroids (Voronoi cells),
        num_probe (int): The number of centroids (Voronoi cells) to probe. Must be <= num_centroids. Sweeping powers of 2 for nprobe and picking one of those based on recall statistics (e.g., 1, 2, 4, 8, ..,) is typically done.,
        num_subquantizers (int): the number of subquanitizers in Product Quantization (PQ) compression of subvectors,
        bits_per_code (int): The number of bits for each subvector in Product Quantization (PQ),

    Example::

        get_index()

    """
    if device is not None and device.type == "cuda":
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFPQConfig()
        index = faiss.GpuIndexIVFPQ(
            res,
            embedding_dim,
            num_centroids,
            num_subquantizers,
            bits_per_code,
            faiss.METRIC_L2,
            config,
        )
    else:
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            embedding_dim,
            num_centroids,
            num_subquantizers,
            bits_per_code,
        )
    index.nprobe = num_probe
    return index
