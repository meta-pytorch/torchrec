#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")

    # Dataset related arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["movielens_1m", "criteo_kaggle", "kuairand_1k"],
        default="movielens_1m",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle, kuairand_1k",
    )

    # Model related arguments
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["dlrmv2", "dlrmv3"],
        default="dlrmv3",
        help="model for experiment, current support dlrmv2, dlrmv3. Dlrmv3 is the default",
    )
    parser.add_argument(
        "--num_embeddings",  # ratio of feature ids to embedding table size # 3 axis: x-bath_idx; y-collisions; zembedding table sizes
        type=int,
        default=None,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to None if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
        default=0,
    )

    # Training related arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--sparse_optim",
        type=str,
        default="adagrad",
        help="The optimizer to use for sparse parameters.",
    )
    parser.add_argument(
        "--dense_optim",
        type=str,
        default="adagrad",
        help="The optimizer to use for sparse parameters.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for Adagrad optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay for Adagrad optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.95,
        help="Beta1 for Adagrad optimizer.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adagrad optimizer.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        drop_last=None,
        shuffle_batches=None,
        shuffle_training_set=None,
    )
    parser.add_argument(
        "--input_hash_size",
        type=int,
        default=0,
        help="Input feature value range",
    )
    parser.add_argument(
        "--profiling_result_folder",
        type=str,
        default="profiling_result",
        help="Folder to save profiling results",
    )
    parser.add_argument(
        "--zch_method",
        type=str,
        help="The method to use for zero collision hashing, blank for no zch",
        default="",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=4,
        help="Number of buckets for identity table. Only used for MPZCH. The number of ranks WORLD_SIZE must be a factor of num_buckets, and the number of buckets must be a factor of input_hash_size",
    )
    parser.add_argument(
        "--max_probe",
        type=int,
        default=None,
        help="Number of probes for identity table. Only used for MPZCH",
    )

    # testbed related arguments
    parser.add_argument(
        "--log_path",
        type=str,
        default="log",
        help="Path to save log file without the suffix",
    )
    return parser.parse_args(argv)
