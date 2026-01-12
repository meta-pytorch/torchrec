#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.utils import Batch
from torchrec.modules.embedding_configs import EmbeddingBagConfig


def get_random_dataset(
    batch_size: int,
    num_batches: int,
    num_dense_features: int,
    embedding_bag_configs: List[EmbeddingBagConfig],
    pooling_factors: Optional[Dict[str, int]] = None,
) -> IterableDataset[Batch]:

    if pooling_factors is None:
        pooling_factors = {}

    keys = []
    ids_per_features = []
    hash_sizes = []

    for table in embedding_bag_configs:
        for feature_name in table.feature_names:
            keys.append(feature_name)
            # guess a pooling factor here
            ids_per_features.append(pooling_factors.get(feature_name, 64))
            hash_sizes.append(table.num_embeddings)

    return RandomRecDataset(
        keys=keys,
        batch_size=batch_size,
        hash_sizes=hash_sizes,
        ids_per_features=ids_per_features,
        num_dense=num_dense_features,
        num_batches=num_batches,
    )


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: IterableDataset[Batch],
    device: torch.device,
) -> float:

    start_time = time.perf_counter()

    for data in dataset:
        sparse_features = data.sparse_features.to(device)

        pooled_embeddings = model(sparse_features)
        optimizer.zero_grad()

        vals = []
        for _name, param in pooled_embeddings.to_dict().items():
            vals.append(param)
        torch.cat(vals, dim=1).sum().backward()
        optimizer.step()

    end_time = time.perf_counter()

    return end_time - start_time


def train_one_epoch_fused_optimizer(
    model: torch.nn.Module,
    dataset: IterableDataset[Batch],
    device: torch.device,
) -> float:

    start_time = time.perf_counter()

    for data in dataset:
        sparse_features = data.sparse_features.to(device)
        fused_pooled_embeddings = model(sparse_features)

        fused_vals = []
        for _name, param in fused_pooled_embeddings.to_dict().items():
            fused_vals.append(param)
        torch.cat(fused_vals, dim=1).sum().backward()

    end_time = time.perf_counter()

    return end_time - start_time


def train(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    dataset: IterableDataset[Batch],
    device: torch.device,
    epochs: int = 100,
) -> Tuple[float, float]:

    training_time = []
    for _ in range(epochs):
        if optimizer:
            training_time.append(train_one_epoch(model, optimizer, dataset, device))
        else:
            training_time.append(
                train_one_epoch_fused_optimizer(model, dataset, device)
            )

    return np.mean(training_time), np.std(training_time)
