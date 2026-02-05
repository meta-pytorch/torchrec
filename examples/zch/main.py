#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import time

import torch
from torchrec import EmbeddingConfig, KeyedJaggedTensor
from tqdm import tqdm

from .sparse_arch import SparseArch


def main(args: argparse.Namespace) -> None:
    """
    This function tests the performance of a Sparse module with or without the MPZCH feature.
    Arguments:
        use_mpzch: bool, whether to enable MPZCH or not
    Prints:
        duration: time for a forward pass of the Sparse module with or without MPZCH enabled
        collision_rate: the collision rate of the MPZCH feature
    """
    print(f"Is use MPZCH: {args.use_mpzch}")

    # check available devices
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    print(f"Using device: {device}")

    # create an embedding configuration
    embedding_config = [
        EmbeddingConfig(
            name="table_0",
            feature_names=["feature_0"],
            embedding_dim=8,
            num_embeddings=args.num_embeddings_per_table,
        ),
        EmbeddingConfig(
            name="table_1",
            feature_names=["feature_1"],
            embedding_dim=8,
            num_embeddings=args.num_embeddings_per_table,
        ),
    ]

    # generate kjt input list
    input_kjt_list = []
    for _ in range(args.num_iters):
        input_kjt_single = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            # pick a set of 24 random numbers from 0 to args.num_embeddings_per_table
            values=torch.LongTensor(
                list(
                    torch.randint(
                        0, args.num_embeddings_per_table, (3 * args.batch_size,)
                    )
                )
            ),
            lengths=torch.LongTensor([1] * args.batch_size + [2] * args.batch_size),
            weights=None,
        )
        input_kjt_single = input_kjt_single.to(device)
        input_kjt_list.append(input_kjt_single)

    num_requests = args.num_iters * args.batch_size

    # make the model
    model = SparseArch(
        tables=embedding_config,
        device=device,
        return_remapped=True,
        use_mpzch=args.use_mpzch,
        buckets=1,
    )

    # do the forward pass
    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # record the start time
        starter.record()
        for it_idx in tqdm(range(args.num_iters)):
            # ec_out, remapped_ids_out = model(input_kjt_single)
            input_kjt = input_kjt_list[it_idx].to(device)
            ec_out, remapped_ids_out = model(input_kjt)
        # record the end time
        ender.record()
        # wait for the end time to be recorded
        torch.cuda.synchronize()
        duration = starter.elapsed_time(ender) / 1000.0  # convert to seconds
    else:
        # in cpu mode, MPZCH can only run in inference mode, so we profile the model in eval mode
        model.eval()
        if args.use_mpzch:
            # when using MPZCH modules, we need to manually set the modules to be in inference mode
            model._mc_ec._managed_collision_collection._managed_collision_modules[
                "table_0"
            ].reset_inference_mode()
            model._mc_ec._managed_collision_collection._managed_collision_modules[
                "table_1"
            ].reset_inference_mode()

        start_time = time.time()
        for it_idx in tqdm(range(args.num_iters)):
            input_kjt = input_kjt_list[it_idx].to(device)
            ec_out, remapped_ids_out = model(input_kjt)
        end_time = time.time()
        duration = end_time - start_time
    # get qps
    qps = num_requests / duration
    print(f"qps: {qps}")
    # print the duration
    print(f"duration: {duration} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mpzch", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_embeddings_per_table", type=int, default=1000)
    args: argparse.Namespace = parser.parse_args()
    main(args)
