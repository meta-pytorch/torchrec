#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import json
import multiprocessing
import os
import sys
from typing import List

import numpy as np
import torch
from torch import distributed as dist
from torchrec.test_utils import get_free_port
from tqdm import tqdm

from .arguments import parse_args
from .data.get_dataloader import get_dataloader


def main(
    rank: int,
    args: argparse.Namespace,
) -> None:
    # seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # setup environment
    os.environ["RANK"] = str(rank)
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    # get dataset
    train_dataloader = get_dataloader(args.dataset_name, args, "train")
    # test_dataloader = get_dataloader(args.dataset_name, args, "val")

    # feature value set
    feature_name_feature_values_set_dict = {}  # {feature_name: set(feature_values)}
    feature_name_feature_value_count_dict = {}  # {feature_name: feature_value_count}
    feature_name_batch_feature_value_set_dict = (
        {}
    )  # {feature_name: {batch_id: set(feature_values)}}
    # feature remapping dict
    feature_name_feature_remapped_values_source_value_num_query_dict = (
        {}
    )  # {feature_name: {remapped_value: {source_value: num_query}}}

    pbar = tqdm(train_dataloader, desc=f"Rank {rank}")
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        for _, feature_kjt_dict in batch.get_dict().items():
            for feature_name, feature_values_jt in feature_kjt_dict.to_dict().items():
                if feature_name not in feature_name_batch_feature_value_set_dict:
                    feature_name_batch_feature_value_set_dict[feature_name] = {}
                if (
                    batch_idx
                    not in feature_name_batch_feature_value_set_dict[feature_name]
                ):
                    feature_name_batch_feature_value_set_dict[feature_name][
                        batch_idx
                    ] = set()
                # update the feature value set and feature value count
                if feature_name not in feature_name_feature_values_set_dict:
                    feature_name_feature_values_set_dict[feature_name] = set()
                feature_name_feature_values_set_dict[feature_name].update(
                    feature_values_jt.values().tolist()
                )
                feature_name_batch_feature_value_set_dict[feature_name][
                    batch_idx
                ].update(feature_values_jt.values().tolist())
                if feature_name not in feature_name_feature_value_count_dict:
                    feature_name_feature_value_count_dict[feature_name] = {}
                if (
                    feature_name
                    not in feature_name_feature_remapped_values_source_value_num_query_dict
                ):
                    feature_name_feature_remapped_values_source_value_num_query_dict[
                        feature_name
                    ] = {}
                for feature_value in feature_values_jt.values().tolist():
                    if (
                        feature_value
                        not in feature_name_feature_value_count_dict[feature_name]
                    ):
                        feature_name_feature_value_count_dict[feature_name][
                            feature_value
                        ] = 0
                    feature_name_feature_value_count_dict[feature_name][
                        feature_value
                    ] += 1

                    remapped_value = feature_value % args.num_embeddings
                    if (
                        remapped_value
                        not in feature_name_feature_remapped_values_source_value_num_query_dict[
                            feature_name
                        ]
                    ):
                        feature_name_feature_remapped_values_source_value_num_query_dict[
                            feature_name
                        ][
                            remapped_value
                        ] = {}
                    if (
                        feature_value
                        not in feature_name_feature_remapped_values_source_value_num_query_dict[
                            feature_name
                        ][
                            remapped_value
                        ]
                    ):
                        feature_name_feature_remapped_values_source_value_num_query_dict[
                            feature_name
                        ][
                            remapped_value
                        ][
                            feature_value
                        ] = 0
                    feature_name_feature_remapped_values_source_value_num_query_dict[
                        feature_name
                    ][remapped_value][feature_value] += 1

    # do statistics
    stats = {}
    # get feature name to number of unique feature values mapping
    world_size = int(os.environ["WORLD_SIZE"])
    bucket_size = args.input_hash_size // world_size
    for (
        feature_name,
        feature_values_set,
    ) in feature_name_feature_values_set_dict.items():
        if feature_name not in stats:
            stats[feature_name] = {}
        print(f"feature_name: {feature_name}, get num_unique_feature_values")
        stats[feature_name]["num_unique_feature_values"] = len(feature_values_set)
        print(
            f"feature_name: {feature_name}, get feature value distribution with respect to the WORLD_SIZE"
        )
        stats[feature_name]["feature_value_distribution"] = {}
        for feature_value in tqdm(feature_values_set):
            feature_value = int(feature_value) % args.input_hash_size
            bucket_idx = feature_value // bucket_size
            if bucket_idx not in stats[feature_name]["feature_value_distribution"]:
                stats[feature_name]["feature_value_distribution"][bucket_idx] = 0
            stats[feature_name]["feature_value_distribution"][bucket_idx] += 1
        print(
            f"feature_name: {feature_name}, get feature query distribution with respect to the WORLD_SIZE"
        )
        stats[feature_name]["feature_query_distribution"] = {}
        for feature_value, feature_value_count in tqdm(
            feature_name_feature_value_count_dict[feature_name].items()
        ):
            bucket_idx = feature_value // bucket_size
            if bucket_idx not in stats[feature_name]["feature_query_distribution"]:
                stats[feature_name]["feature_query_distribution"][bucket_idx] = 0
            stats[feature_name]["feature_query_distribution"][
                bucket_idx
            ] += feature_value_count
        print(
            f"feature_name: {feature_name}, get feature remapping min-max collision rate"
        )
        stats[feature_name]["feature_remapping_collision"] = {}
        max_num_collisions = 0
        min_num_collisions = 0
        ds_num_collisions = 0
        num_total_queries = 0
        for _, source_value_num_query_count in tqdm(
            feature_name_feature_remapped_values_source_value_num_query_dict[
                feature_name
            ].items()
        ):
            # fetch the number of queries list for the remapped value
            source_value_num_query_count_list = list(
                source_value_num_query_count.values()
            )
            # get the total number of queries for the remapped value
            total_num_query_for_remapped_value = sum(source_value_num_query_count_list)
            num_total_queries += total_num_query_for_remapped_value
            # get the number of collisions for the remapped value
            ## the dataset number of collisions is the number of queries that collide with the first appeared source value that remapped to the remapped value
            ds_num_collisions_for_remapped_value = (
                total_num_query_for_remapped_value
                - source_value_num_query_count_list[0]
            )
            ds_num_collisions += ds_num_collisions_for_remapped_value
            ## the max possible number of collisions is the number of queries that collide with the source value that has the minimal number of queries
            max_num_collisions_for_remapped_value = (
                total_num_query_for_remapped_value
                - min(source_value_num_query_count_list)
            )
            max_num_collisions += max_num_collisions_for_remapped_value
            ## the min possible number of collisions is the number of queries that collide with the source value that has the maximal number
            min_num_collisions_for_remapped_value = (
                total_num_query_for_remapped_value
                - max(source_value_num_query_count_list)
            )
            min_num_collisions += min_num_collisions_for_remapped_value
        # save the results
        stats[feature_name]["feature_remapping_collision"][
            "ds_num_collisions"
        ] = ds_num_collisions
        stats[feature_name]["feature_remapping_collision"][
            "max_num_collisions"
        ] = max_num_collisions
        stats[feature_name]["feature_remapping_collision"][
            "min_num_collisions"
        ] = min_num_collisions
        stats[feature_name]["feature_remapping_collision"][
            "num_total_queries"
        ] = num_total_queries
        stats[feature_name]["feature_remapping_collision"]["ds_num_collisions_rate"] = (
            ds_num_collisions / num_total_queries if num_total_queries > 0 else 0
        )
        stats[feature_name]["feature_remapping_collision"][
            "max_num_collisions_rate"
        ] = (max_num_collisions / num_total_queries if num_total_queries > 0 else 0)
        stats[feature_name]["feature_remapping_collision"][
            "min_num_collisions_rate"
        ] = (min_num_collisions / num_total_queries if num_total_queries > 0 else 0)

    # save results to json
    os.makedirs(args.profiling_result_folder, exist_ok=True)
    output_json_path = os.path.join(
        args.profiling_result_folder, f"dataset_stats_rank_{rank}.json"
    )
    with open(output_json_path, "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    args: argparse.Namespace = parse_args(sys.argv[1:])
    # set environment variables
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())
    # set a multiprocessing context
    ctx: multiprocessing.context.SpawnContext = multiprocessing.get_context("spawn")
    # create a process to perform benchmarking
    processes: List[multiprocessing.context.SpawnProcess] = []
    for rank in range(int(os.environ["WORLD_SIZE"])):
        p: multiprocessing.context.SpawnProcess = ctx.Process(
            target=main,
            args=(rank, args),
        )
        p.start()
        processes.append(p)
