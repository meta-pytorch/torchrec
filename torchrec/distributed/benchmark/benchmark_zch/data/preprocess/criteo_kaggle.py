#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import os
from typing import Any, Dict, List

from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import CAT_FEATURE_COUNT, InMemoryBinaryCriteoIterDataPipe

STAGES = ["train", "val", "test"]


def get_criteo_kaggle_dataloader(
    args: argparse.Namespace,
    configs: Dict[str, Any],
    stage: str = "train",  # "train", "val", "test"
) -> DataLoader:
    dir_path = configs["dataset_path"]
    sparse_part = "sparse.npy"
    datapipe = InMemoryBinaryCriteoIterDataPipe

    # criteo_kaggle has no validation set, so use 2nd half of training set for now.
    # Setting stage to "test" will get the 2nd half of the dataset.
    # Setting root_name to "train" reads from the training set file.
    (root_name, stage) = ("train", "train") if stage == "train" else ("train", "test")
    stage_files: List[List[str]] = [
        [os.path.join(dir_path, f"{root_name}_dense.npy")],
        [os.path.join(dir_path, f"{root_name}_{sparse_part}")],
        [os.path.join(dir_path, f"{root_name}_labels.npy")],
    ]
    batch_size = configs["batch_size"] if args.batch_size is None else args.batch_size
    dataloader = DataLoader(
        datapipe(
            stage,
            *stage_files,
            batch_size=batch_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            drop_last=False,
            shuffle_batches=args.shuffle_batches,
            hashes=(
                [args.num_embeddings] * CAT_FEATURE_COUNT
                if args.input_hash_size is None
                else ([args.input_hash_size] * CAT_FEATURE_COUNT)
            ),
        ),
        batch_size=None,
        collate_fn=lambda x: x,
    )
    return dataloader
