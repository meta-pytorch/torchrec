#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import os
import sys
from typing import Dict

import torch
import yaml

from torchrec.metrics.auc import AUCMetricComputation
from torchrec.metrics.mae import MAEMetricComputation
from torchrec.metrics.mse import MSEMetricComputation
from torchrec.metrics.ne import NEMetricComputation
from torchrec.metrics.rec_metric import RecMetricComputation


def get_metric_modules(
    rank: int, args: argparse.Namespace, device: torch.device
) -> Dict[str, RecMetricComputation]:
    # get the dataset configuration from the yaml file
    dataset_name = args.dataset_name
    assert os.path.exists(
        os.path.join(os.path.dirname(__file__), "configs", f"{dataset_name}.yaml")
    ), f"Dataset {dataset_name} not found"
    with open(
        os.path.join(os.path.dirname(__file__), "configs", f"{dataset_name}.yaml"), "r"
    ) as f:
        configs = yaml.safe_load(f)
    metric_modules = {}  # dictionary of {metric_name: metric_module}
    # get the task_type: task_count mapping
    task_type_count_dict = {}  # dictionary of {task_type: task_count}
    ## get the tasks from configs
    multitask_configs = configs["multitask_configs"]
    ## get the task_type: task_count
    for task_info_dict in multitask_configs:
        task_type = task_info_dict["task_type"]
        if task_type not in task_type_count_dict:
            task_type_count_dict[task_type] = 0
        task_type_count_dict[task_type] += 1
    # instantiate the metric modules
    for task_type, task_count in task_type_count_dict.items():
        if task_type == "regression":
            metric_modules["mae"] = MAEMetricComputation(
                my_rank=rank,
                batch_size=(
                    configs["batch_size"]
                    if args.batch_size is None
                    else args.batch_size
                ),
                n_tasks=task_count,
                window_size=sys.maxsize,
                compute_on_all_ranks=True,
            ).to(device)
            metric_modules["mse"] = MSEMetricComputation(
                my_rank=rank,
                batch_size=(
                    configs["batch_size"]
                    if args.batch_size is None
                    else args.batch_size
                ),
                n_tasks=task_count,
                window_size=sys.maxsize,
                compute_on_all_ranks=True,
            ).to(device)
        elif task_type == "classification":
            metric_modules["ne"] = NEMetricComputation(
                my_rank=rank,
                batch_size=(
                    configs["batch_size"]
                    if args.batch_size is None
                    else args.batch_size
                ),
                n_tasks=task_count,
                window_size=sys.maxsize,
                compute_on_all_ranks=True,
            ).to(device)
            metric_modules["auc"] = AUCMetricComputation(
                my_rank=rank,
                batch_size=(
                    configs["batch_size"]
                    if args.batch_size is None
                    else args.batch_size
                ),
                n_tasks=task_count,
                window_size=sys.maxsize,
                compute_on_all_ranks=True,
            ).to(device)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    return metric_modules
