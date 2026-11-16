#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch.utils.data import DataLoader
from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.datasets.random import RandomRecDataset


def get_dataloader(
    batch_size: int, num_embeddings: int, pin_memory: bool = False, num_workers: int = 0
) -> DataLoader:
    """
    Gets a Random dataloader for the two tower model, containing a two_feature KJT as sparse_features, empty dense_features
    and binary labels

    Args:
        batch_size (int): batch_size
        num_embeddings (int): hash_size of the two embedding tables
        pin_memory (bool): Whether to pin_memory on the GPU
        num_workers (int) Number of dataloader workers

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]

    return DataLoader(
        RandomRecDataset(
            keys=two_tower_column_names,
            batch_size=batch_size,
            hash_size=num_embeddings,
            ids_per_feature=1,
            num_dense=0,
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
