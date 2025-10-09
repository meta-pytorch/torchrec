#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import csv
import json
import multiprocessing
import os
import sys
import time

from typing import Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn

from torch import distributed as dist

# pyre-ignore [21] # NOTE: pyre reports ProfilerActivity is not in torchrec.distributed, but it is in torch.profiler according to https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # @manual //caffe2:torch_tensorboard
from torchrec.metrics.metrics_namespace import MetricPrefix
from torchrec.metrics.rec_metric import RecMetricComputation

from torchrec.test_utils import get_free_port
from tqdm import tqdm

from .arguments import parse_args

from .benchmark_zch_utils import BenchmarkMCProbe, get_logger, get_module_from_instance

from .data.get_dataloader import get_dataloader
from .data.nonzch_remapper import NonZchModRemapperModule
from .models.apply_optimizers import (
    apply_dense_optimizers,
    apply_sparse_optimizers,
    combine_optimizers,
)
from .models.make_model import make_model
from .models.shard_model import shard_model


def main(rank: int, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    # initialize the rank logger
    log_rank_file_path = args.log_path + f"_rank_{rank}.log"
    logger = get_logger(log_file_path=f"{log_rank_file_path}")

    # seed everything for reproducibility
    logger.info(
        f"[rank {rank}] seed everything for reproducibility with seed {args.seed}"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # setup environment
    logger.info(f"[rank {rank}] setup environment")
    os.environ["RANK"] = str(rank)
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(
        f"[rank {rank}] init process group. world size: {world_size}, rank: {rank}, backend: {backend}, device: {device}"
    )

    # get training dataset
    logger.info(f"[rank {rank}] get train dataloader")
    train_dataloader = get_dataloader(args.dataset_name, args, "train")

    # make the model
    logger.info(f"[rank {rank}] make model")
    model, model_configs = make_model(args.model_name, args, device)

    # initialize the non-zch mod module if needed
    nonzch_remapper = None
    if len(args.zch_method) == 0:
        logger.info(f"[rank {rank}] initialize the non-zch mod module")
        nonzch_remapper = NonZchModRemapperModule(
            # pyre-ignore [6] # NOTE: pyre reports model.table_configs is in type `Union[Module, Tensor]`, but we know it is a list of table configs
            table_configs=model.table_configs,
            input_hash_size=args.input_hash_size,
            device=device,
        )

    # apply optimizer to the sparse arch of the model
    logger.info(f"[rank {rank}] apply optimizer to the sparse arch of the model")
    apply_sparse_optimizers(model, args)

    # shard the model
    logger.info(f"[rank {rank}] shard the model")
    model = shard_model(model, device, args)

    # apply optimizer to the dense arch of the model
    logger.info(f"[rank {rank}] apply optimizer to the dense arch of the model")
    dense_optimizer = apply_dense_optimizers(model, args)

    # combine the sparse and dense optimizers
    logger.info(f"[rank {rank}] combine the sparse and dense optimizers")
    optimizer = combine_optimizers(model.fused_optimizer, dense_optimizer)

    # create the benchmark probe if needed
    logger.info(f"[rank {rank}] create the benchmark probe")
    benchmark_probe = None
    if len(args.zch_method) > 0:
        benchmark_probe = BenchmarkMCProbe(
            # pyre-ignore [6] # NOTE: Though in the return type specification to be general we set as nn.Module, but here the returned object is a ManagedCollisionEmbeddingCollection
            mcec=get_module_from_instance(
                model._dmp_wrapped_module,
                model_configs["managed_collision_module_attribute_path"],
            ),
            mc_method=args.zch_method,
            rank=rank,
        )
    else:
        benchmark_probe = BenchmarkMCProbe(
            # pyre-ignore [16] # NOTE: pyre reports nonzch_remapper can be None, but when reach to this branch of condition, we know it is not None
            mcec=nonzch_remapper.mod_modules,
            mc_method="mpzch",  # because non-zch remapper simulates the behavior of mpzch
            rank=rank,
        )

    interval_num_batches_show_qps = 50

    total_time_in_training = 0
    total_num_queries_in_training = 0

    # train the model
    ## code for profiling
    # pyre-ignore [16] # NOTE: pyre reports ProfilerActivity is not in torchrec.distributed, but it is in torch.profiler according to https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # pyre-ignore [16] # NOTE: pyre reports ProfilerActivity is not in torchrec.distributed, but it is in torch.profiler according to https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        activities += [ProfilerActivity.CUDA]
    ## end code for profiling
    with profile(activities=activities, record_shapes=True) as prof:
        for epoch_idx in range(args.epochs):
            model.train()
            starter_list = []
            ender_list = []
            num_queries_per_batch_list = []
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_idx}")
            for batch_idx, batch in enumerate(pbar):
                # batch = batch.to(device)
                batch = batch.to(device)
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                if len(args.zch_method) > 0:
                    benchmark_probe.record_mcec_state(stage="before_fwd")
                # forward pass
                with record_function(f"model training on batch {batch_idx}"):
                    starter.record()
                    # zero the gradients
                    optimizer.zero_grad()
                    loss, (loss_values, pred_logits, labels) = model(batch)
                    loss.backward()
                    optimizer.step()
                    ender.record()
                # do statistics
                num_queries_per_batch = len(labels)
                starter_list.append(starter)
                ender_list.append(ender)
                num_queries_per_batch_list.append(num_queries_per_batch)
                if len(args.zch_method) > 0:
                    benchmark_probe.record_mcec_state(stage="after_fwd")
                    # update zch statistics
                    benchmark_probe.update()
                    # push the zch stats to the queue
                    msg_content = {
                        "epoch_idx": epoch_idx,
                        "batch_idx": batch_idx,
                        "rank": rank,
                        "mch_stats": benchmark_probe.get_mch_stats(),
                    }
                    queue.put(
                        ("mch_stats", msg_content),
                    )
                if (
                    batch_idx % interval_num_batches_show_qps == 0
                    or batch_idx == len(train_dataloader) - 1
                ):
                    if batch_idx == 0:
                        # skip the first batch since it is not a full batch
                        continue
                    # synchronize all the threads to get the exact number of batches
                    torch.cuda.synchronize()
                    # calculate the qps
                    # NOTE: why do this qps calculation every interval_num_batches_show_qps batches?
                    #   because performing this calculation needs to synchronize all the ranks by calling torch.cuda.synchronize()
                    #   and this is a heavy operation (takes several milliseconds). So we only do this calculation every
                    #   interval_num_batches_show_qps batches to reduce the overhead.
                    ## get per batch time list by calculating the time difference between the start and end events of each batch
                    per_batch_time_list = []
                    for i in range(len(starter_list)):
                        per_batch_time_list.append(
                            starter_list[i].elapsed_time(ender_list[i]) / 1000
                        )  # convert to seconds by dividing by 1000
                    ## calculate the total time in the interval
                    total_time_in_interval = sum(per_batch_time_list)
                    ## calculate the total number of queries in the interval
                    total_num_queries_in_interval = sum(num_queries_per_batch_list)
                    ## fabricate the message and total_num_queries_in_interval to the queue
                    interval_start_batch_idx = (
                        batch_idx - interval_num_batches_show_qps
                        if batch_idx >= interval_num_batches_show_qps
                        else 0
                    )  # the start batch index of the interval
                    interval_end_batch_idx = (
                        batch_idx  # the end batch index of the interval
                    )
                    ## fabricate the message content
                    msg_content = {
                        "epoch_idx": epoch_idx,
                        "rank": rank,
                        "interval_start_batch_idx": interval_start_batch_idx,
                        "interval_end_batch_idx": interval_end_batch_idx,
                        "per_batch_time_list": per_batch_time_list,
                        "per_batch_num_queries_list": num_queries_per_batch_list,
                    }
                    ## put the message into the queue
                    queue.put(("duration_and_num_queries", msg_content))
                    qps_per_interval = (
                        total_num_queries_in_interval / total_time_in_interval
                    )
                    total_time_in_training += total_time_in_interval
                    total_num_queries_in_training += total_num_queries_in_interval
                    pbar.set_postfix(
                        {
                            "QPS": qps_per_interval,
                        }
                    )
                    pbar.update(interval_num_batches_show_qps)
                    # reset the lists
                    starter_list = []
                    ender_list = []
                    num_queries_per_batch_list = []
                if batch_idx > 50:
                    # skip rest after collecting data for 50 batches
                    break
                # skip evaluation

    prof.export_chrome_trace(
        f"/home/lizhouyu/tmp/trace_noremap_{args.zch_method if len(args.zch_method) > 0 else 'nonzch'}_{args.model_name}_fullloop_tbsize_{args.num_embeddings}_rank{rank}.json"
    )
    time.sleep(10)
    queue.put(("finished", {"rank": rank}))
    print("finished")
    return

    # print("Total time in training: ", total_time_in_training)
    # print("Total number of queries in training: ", total_num_queries_in_training)
    # print("Average QPS: ", total_num_queries_in_training / total_time_in_training)


def evaluation(
    metric_modules: Dict[str, RecMetricComputation],
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    nonzch_remapper: Optional[NonZchModRemapperModule] = None,
) -> Dict[str, float]:
    """
    Evaluate the model on the given data loader.
    """
    # set model into eval mode
    model.eval()
    # run evaluation and update metrics
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        if nonzch_remapper is not None:
            batch = nonzch_remapper.remap(batch)
        with torch.no_grad():
            loss, (loss_values, pred_logits, labels, weights) = model(batch)
            if len(pred_logits.shape) <= 1:
                pred_logits = pred_logits.unsqueeze(0)
            if len(labels.shape) <= 1:
                labels = labels.unsqueeze(0)
            if len(weights.shape) <= 1:
                weights = weights.unsqueeze(0)
            # update metrics
            for _, metric_module in metric_modules.items():
                metric_module.update(
                    predictions=pred_logits,
                    labels=labels,
                    weights=weights,
                )
    # get the metric values
    metric_values = {}  # {metric_name: metric_value}
    for metric_name, metric_module in metric_modules.items():
        metric_computation_reports = metric_module.compute()
        for metric_computation_report in metric_computation_reports:
            if metric_computation_report.metric_prefix == MetricPrefix.WINDOW:
                metric_values[metric_name] = (
                    metric_computation_report.value.cpu().numpy().tolist()
                )
                break
    # reset metrics modules
    for metric_module in metric_modules.values():
        metric_module.reset()
    # return the metric values
    return metric_values


def statistic(args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    """
    The process to perform statistic calculations
    """
    mch_buffer = (
        {}
    )  # {epcoh_idx:{end_batch_idx: {rank: data_dict}}} where data dict is {metric_name: metric_value}
    num_processed_batches = 0  # counter of the number of processed batches
    world_size = int(os.environ["WORLD_SIZE"])  # world size
    finished_counter = 0  # counter of the number of finished processes

    # create a profiling result folder
    os.makedirs(args.profiling_result_folder, exist_ok=True)
    # create a csv file to save the training_metrics
    training_metrics_file_path = os.path.join(
        args.profiling_result_folder, "training_metrics.csv"
    )
    with open(training_metrics_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_idx",
                "batch_idx",
                "batch_cnt",
                "rank",
                "loss",
            ]
        )
    # create a csv file to save the zch_metrics
    zch_metrics_file_path = os.path.join(
        args.profiling_result_folder, "zch_metrics.csv"
    )
    with open(zch_metrics_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_idx",
                "batch_idx",
                "batch_cnt",
                "feature_name",
                "hit_cnt",
                "total_cnt",
                "insert_cnt",
                "collision_cnt",
                "hit_rate",
                "insert_rate",
                "collision_rate",
                "rank_total_cnt",
                "rank_num_empty_slots",
            ]
        )
    ## create counter for total number of collision and total number of queries
    total_num_collisions = {}  # feature name: total number of collisions
    total_num_queries = {}  # feature name: total number of queries
    # create a csv file to save the qps_metrics
    qps_metrics_file_path = os.path.join(
        args.profiling_result_folder, "qps_metrics.csv"
    )
    with open(qps_metrics_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_idx",
                "batch_idx",
                "batch_cnt",
                "rank",
                "num_queries",
                "duration",
                "qps",
            ]
        )
    # create a csv file to save the eval_metrics
    eval_metrics_file_path = os.path.join(
        args.profiling_result_folder, "eval_metrics.csv"
    )
    with open(eval_metrics_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_idx", "rank", "auc", "ne", "mae", "mse"])

    # create a tensorboard folder and summary writer
    tb_log_folder = os.path.join(args.profiling_result_folder, "tb")
    tb_writer = SummaryWriter(log_dir=tb_log_folder)

    while finished_counter < world_size:
        try:
            # get the data from the queue
            msg_type, msg_content = queue.get(
                timeout=0.5
            )  # data are put into the queue im the form of (msg_type, epoch_idx, batch_idx, rank, rank_data_dict)
        except Exception:
            # if the queue is empty, check if all the processes have finished
            # if finished_counter >= world_size:
            #     print(f"All processes have finished. {finished_counter} / {world_size}")
            #     break
            # else:
            #     continue  # keep waiting for the queue to be filled
            # if queue is empty, check if all the processes have finished
            if finished_counter >= world_size:
                print(f"All processes have finished. {finished_counter} / {world_size}")
                break
            else:
                continue  # keep waiting for the queue to be filled
        # when getting the data, check if the data is from the last batch
        if (
            msg_type == "finished"
        ):  # if the message type is "finished", the process has finished
            rank = msg_content["rank"]
            finished_counter += 1
            print(f"Process {rank} has finished. {finished_counter} / {world_size}")
            continue
        elif msg_type == "mch_stats":
            epoch_idx = msg_content["epoch_idx"]
            batch_idx = msg_content["batch_idx"]
            batch_cnt = msg_content["batch_cnt"]
            rank = msg_content["rank"]
            rank_batch_mch_stats = msg_content["mch_stats"]
            # other wise, aggregate the data into the buffer
            if epoch_idx not in mch_buffer:
                mch_buffer[epoch_idx] = {}
            if batch_idx not in mch_buffer[epoch_idx]:
                mch_buffer[epoch_idx][batch_idx] = {}
            mch_buffer[epoch_idx][batch_idx][rank] = rank_batch_mch_stats
            num_processed_batches += 1
            # check if we have all the data from all the ranks for a batch in an epoch
            # if we have all the data, combine the data from all the ranks
            if len(mch_buffer[epoch_idx][batch_idx]) == world_size:
                # create a dictionary to store the statistics for each batch
                batch_stats = (
                    {}
                )  # {feature_name: {hit_cnt: 0, total_cnt: 0, insert_cnt: 0, collision_cnt: 0}}
                # combine the data from all the ranks
                for mch_stats_rank_idx in mch_buffer[epoch_idx][batch_idx].keys():
                    rank_batch_mch_stats = mch_buffer[epoch_idx][batch_idx][
                        mch_stats_rank_idx
                    ]
                    # for each feature table in the mch stats information
                    for mch_stats_feature_name in rank_batch_mch_stats.keys():
                        # create the dictionary for the feature table if not created
                        if mch_stats_feature_name not in batch_stats:
                            batch_stats[mch_stats_feature_name] = {
                                "hit_cnt": 0,
                                "total_cnt": 0,
                                "insert_cnt": 0,
                                "collision_cnt": 0,
                                "rank_total_cnt": {},  # dictionary of {rank_idx: num_quries_mapped_to_the_rank}
                                "rank_num_empty_slots": {},  # dictionary of {rank_idx: num_empty_slots}
                                "rank_num_unique_queries": {},  # dictionary of {rank_idx: num_unique_queries}
                            }
                        # aggregate the data from all the ranks
                        ## aggregate the hit count
                        batch_stats[mch_stats_feature_name][
                            "hit_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name]["hit_cnt"]
                        ## aggregate the total count
                        batch_stats[mch_stats_feature_name][
                            "total_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name]["total_cnt"]
                        ## aggregate the insert count
                        batch_stats[mch_stats_feature_name][
                            "insert_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name]["insert_cnt"]
                        ## aggregate the collision count
                        batch_stats[mch_stats_feature_name][
                            "collision_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name][
                            "collision_cnt"
                        ]
                        ## for rank total count, get the data from the rank data dict
                        batch_stats[mch_stats_feature_name]["rank_total_cnt"][
                            mch_stats_rank_idx
                        ] = rank_batch_mch_stats[mch_stats_feature_name][
                            "rank_total_cnt"
                        ]
                        ## for rank num empty slots, get the data from the rank data dict
                        batch_stats[mch_stats_feature_name]["rank_num_empty_slots"][
                            mch_stats_rank_idx
                        ] = rank_batch_mch_stats[mch_stats_feature_name][
                            "num_empty_slots"
                        ]
                        ## for rank num unique queries, get the data from the rank data dict
                        batch_stats[mch_stats_feature_name]["rank_num_unique_queries"][
                            mch_stats_rank_idx
                        ] = rank_batch_mch_stats[mch_stats_feature_name][
                            "num_unique_queries"
                        ]
                # clear the buffer for the batch
                del mch_buffer[epoch_idx][batch_idx]
                # save the zch statistics to a file and tensorboard
                tb_prefix = "collision management"
                with open(zch_metrics_file_path, "a") as f:
                    writer = csv.writer(f)
                    for feature_name, stats in batch_stats.items():
                        # calculate rate for each rank
                        hit_rate = stats["hit_cnt"] / stats["total_cnt"]
                        insert_rate = stats["insert_cnt"] / stats["total_cnt"]
                        collision_rate = stats["collision_cnt"] / stats["total_cnt"]
                        rank_total_cnt = json.dumps(stats["rank_total_cnt"])
                        rank_num_empty_slots = json.dumps(stats["rank_num_empty_slots"])
                        # update the total number of collisions and total number of queries
                        if feature_name not in total_num_collisions:
                            total_num_collisions[feature_name] = 0
                        if feature_name not in total_num_queries:
                            total_num_queries[feature_name] = 0
                        total_num_collisions[feature_name] += stats["collision_cnt"]
                        total_num_queries[feature_name] += stats["total_cnt"]
                        overall_collision_rate = (
                            total_num_collisions[feature_name]
                            / total_num_queries[feature_name]
                            if total_num_queries[feature_name] > 0
                            else 0
                        )
                        # write the zch statitics to csv file
                        writer.writerow(
                            [
                                epoch_idx,
                                batch_idx,
                                batch_cnt,
                                feature_name,
                                stats["hit_cnt"],
                                stats["total_cnt"],
                                stats["insert_cnt"],
                                stats["collision_cnt"],
                                hit_rate,
                                insert_rate,
                                collision_rate,
                                rank_total_cnt,
                                rank_num_empty_slots,
                            ]
                        )
                        # write the zch statitics to tensorboard
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/hit_cnt",
                            stats["hit_cnt"],
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/total_cnt",
                            stats["total_cnt"],
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/insert_cnt",
                            stats["insert_cnt"],
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/collision_cnt",
                            stats["collision_cnt"],
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/hit_rate",
                            hit_rate,
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/insert_rate",
                            insert_rate,
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/collision_rate",
                            collision_rate,
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/total_num_collisions",
                            total_num_collisions[feature_name],
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/total_num_queries",
                            total_num_queries[feature_name],
                            batch_cnt,
                        )
                        tb_writer.add_scalar(
                            f"{tb_prefix}/{feature_name}/overall_collision_rate",
                            overall_collision_rate,
                            batch_cnt,
                        )
                        ## convert rank idx to string for displaying in tensorboard
                        rank_total_cnt_scalar_dict = dict(
                            [
                                (str(rank_idx), rank_total_cnt)
                                for (rank_idx, rank_total_cnt) in stats[
                                    "rank_total_cnt"
                                ].items()
                            ]
                        )
                        tb_writer.add_scalars(
                            f"{tb_prefix}/{feature_name}/number of queries mapped to rank",
                            rank_total_cnt_scalar_dict,
                            batch_cnt,
                        )
                        ## convert rank idx to string for displaying in tensorboard
                        rank_num_empty_slots_scalar_dict = dict(
                            [
                                (str(rank_idx), rank_num_empty_slots)
                                for (rank_idx, rank_num_empty_slots) in stats[
                                    "rank_num_empty_slots"
                                ].items()
                            ]
                        )
                        tb_writer.add_scalars(
                            f"{tb_prefix}/{feature_name}/number of empty slots in rank",
                            rank_num_empty_slots_scalar_dict,
                            batch_cnt,
                        )
                        ## convert rank idx to string for displaying in tensorboard
                        rank_num_unique_queries_scalar_dict = dict(
                            [
                                (str(rank_idx), rank_num_unique_queries)
                                for (rank_idx, rank_num_unique_queries) in stats[
                                    "rank_num_unique_queries"
                                ].items()
                            ]
                        )
                        tb_writer.add_scalars(
                            f"{tb_prefix}/{feature_name}/number of unique queries in rank",
                            rank_num_unique_queries_scalar_dict,
                            batch_cnt,
                        )
        elif msg_type == "duration_and_num_queries":
            epoch_idx = msg_content["epoch_idx"]
            rank = msg_content["rank"]
            interval_start_batch_idx = msg_content["interval_start_batch_idx"]
            interval_start_batch_cnt = msg_content["interval_start_batch_cnt"]
            per_batch_time_list = msg_content["per_batch_time_list"]
            per_batch_num_queries_list = msg_content["per_batch_num_queries_list"]
            # save the qps statistics to a file and tensorboard
            tb_prefix = "efficiency"
            with open(qps_metrics_file_path, "a") as f:
                writer = csv.writer(f)
                for i in range(len(per_batch_time_list)):
                    qps = (
                        per_batch_num_queries_list[i] / per_batch_time_list[i]
                        if per_batch_time_list[i] > 0
                        else 0
                    )
                    writer.writerow(
                        [
                            epoch_idx,
                            str(interval_start_batch_idx + i),
                            str(interval_start_batch_cnt + i),
                            rank,
                            per_batch_num_queries_list[i],
                            per_batch_time_list[i],
                            qps,
                        ]
                    )
                    # write the qps statistics to tensorboard
                    tb_writer.add_scalar(
                        f"{tb_prefix}/qps",
                        qps,
                        interval_start_batch_cnt + i,
                    )
                    tb_writer.add_scalars(
                        f"{tb_prefix}/number of queries",
                        {str(rank): per_batch_num_queries_list[i]},
                        interval_start_batch_cnt + i,
                    )
                    tb_writer.add_scalars(
                        f"{tb_prefix}/training duration",
                        {str(rank): per_batch_time_list[i]},
                        interval_start_batch_cnt + i,
                    )
        elif msg_type == "training_metrics":
            epoch_idx = msg_content["epoch_idx"]
            rank = msg_content["rank"]
            interval_start_batch_idx = msg_content["interval_start_batch_idx"]
            interval_start_batch_cnt = msg_content["interval_start_batch_cnt"]
            per_batch_loss_list = msg_content["per_batch_loss_list"]
            # save the training metrics to a file and tensorboard
            tb_prefix = "training"
            with open(training_metrics_file_path, "a") as f:
                writer = csv.writer(f)
                for i in range(len(per_batch_loss_list)):
                    # write the training metrics to csv file
                    writer.writerow(
                        [
                            epoch_idx,
                            str(interval_start_batch_idx + i),
                            str(interval_start_batch_cnt + i),
                            rank,
                            per_batch_loss_list[i],
                        ]
                    )
                    # write the training metrics to tensorboard
                    tb_writer.add_scalars(
                        f"{tb_prefix}/loss",
                        {str(rank): per_batch_loss_list[i]},
                        interval_start_batch_cnt + i,
                    )
        elif msg_type == "eval_result":
            epoch_idx = msg_content["epoch_idx"]
            rank = msg_content["rank"]
            eval_result_dict = msg_content["eval_result_dict"]
            # save the evaluation result to a file and tensorboard
            tb_prefix = "evaluation"
            with open(eval_metrics_file_path, "a") as f:
                # write the evaluation result to csv file
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch_idx,
                        rank,
                        eval_result_dict["auc"] if "auc" in eval_result_dict else "",
                        eval_result_dict["ne"] if "ne" in eval_result_dict else "",
                        eval_result_dict["mae"] if "mae" in eval_result_dict else "",
                        eval_result_dict["mse"] if "mse" in eval_result_dict else "",
                    ]
                )
                # write the evaluation result to tensorboard
                # rebuild {task_idx: {metric_name: {rank: metric_value}}} dict
                tb_eval_result_dict = (
                    {}
                )  # {task_idx: {metric_name: {rank: metric_value}}}
                for metric_name, task_metric_value_list in eval_result_dict.items():
                    for task_idx, metric_value in enumerate(task_metric_value_list):
                        if task_idx not in tb_eval_result_dict:
                            tb_eval_result_dict[task_idx] = {}
                        if metric_name not in tb_eval_result_dict[task_idx]:
                            tb_eval_result_dict[task_idx][metric_name] = {}
                        tb_eval_result_dict[task_idx][metric_name][
                            str(rank)
                        ] = metric_value
                # display the evaluation result in tensorboard for each task
                for task_idx in tb_eval_result_dict.keys():
                    for metric_name, metric_value_dict in tb_eval_result_dict[
                        task_idx
                    ].items():
                        tb_writer.add_scalars(
                            f"{tb_prefix}/task_{task_idx}/{metric_name}",
                            metric_value_dict,
                            epoch_idx,
                        )
        else:
            # raise a warning if the message type is not recognized
            print("Warning: Unknown message type")
            continue


if __name__ == "__main__":
    args: argparse.Namespace = parse_args(sys.argv[1:])

    # set environment variables
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())
    # set a multiprocessing context
    ctx: multiprocessing.context.SpawnContext = multiprocessing.get_context("spawn")
    # create a queue to communicate between processes
    queue: multiprocessing.Queue = ctx.Queue()
    # # create a process to perform statistic calculations
    stat_process: multiprocessing.context.SpawnProcess = ctx.Process(
        target=statistic, args=(args, queue)
    )  # create a process to perform statistic calculations
    stat_process.start()
    # create a process to perform benchmarking
    train_processes: List[multiprocessing.context.SpawnProcess] = []
    for rank in range(int(os.environ["WORLD_SIZE"])):
        p = ctx.Process(
            target=main,
            args=(rank, args, queue),
        )
        p.start()
        train_processes.append(p)

    # wait for the training processes to finish
    for p in train_processes:
        p.join()
    # wait for the statistic process to finish
    stat_process.join()
