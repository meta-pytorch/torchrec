#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
import unittest
import uuid

import torch
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec import test_utils

from ..bert4rec_main import main


class MainTest(unittest.TestCase):
    @classmethod
    def _run_trainer_QPS_ddp(cls) -> None:
        main(
            [
                "--dataset_name",
                "random",
                "--random_user_count",
                "3000",
                "--random_item_count",
                "500000",
                "--random_size",
                "8000000",
                "--lr",
                "0.001",
                "--mask_prob",
                "0.2",
                "--weight_decay",
                "0.00001",
                "--train_batch_size",
                "8",
                "--val_batch_size",
                "8",
                "--max_len",
                "16",
                "--emb_dim",
                "128",
                "--num_epochs",
                "1",
                "--mode",
                "ddp",
            ]
        )

    @classmethod
    def _run_trainer_QPS_dmp(cls) -> None:
        main(
            [
                "--dataset_name",
                "random",
                "--random_user_count",
                "3000",
                "--random_item_count",
                "500000",
                "--random_size",
                "8000000",
                "--lr",
                "0.001",
                "--mask_prob",
                "0.2",
                "--weight_decay",
                "0.00001",
                "--train_batch_size",
                "8",
                "--val_batch_size",
                "8",
                "--max_len",
                "16",
                "--emb_dim",
                "128",
                "--num_epochs",
                "1",
                "--mode",
                "dmp",
            ]
        )

    @classmethod
    def _run_trainer_random_ddp(cls) -> None:
        main(
            [
                "--dataset_name",
                "random",
                "--random_user_count",
                "60",
                "--random_item_count",
                "20",
                "--random_size",
                "800",
                "--lr",
                "0.001",
                "--mask_prob",
                "0.2",
                "--weight_decay",
                "0.00001",
                "--train_batch_size",
                "32",
                "--val_batch_size",
                "32",
                "--max_len",
                "30",
                "--emb_dim",
                "32",
                "--num_epochs",
                "5",
                "--mode",
                "ddp",
            ]
        )

    @classmethod
    def _run_trainer_random_dmp(cls) -> None:
        main(
            [
                "--dataset_name",
                "random",
                "--random_user_count",
                "60",
                "--random_item_count",
                "20",
                "--random_size",
                "800",
                "--lr",
                "0.001",
                "--mask_prob",
                "0.2",
                "--weight_decay",
                "0.00001",
                "--train_batch_size",
                "32",
                "--val_batch_size",
                "32",
                "--max_len",
                "30",
                "--emb_dim",
                "32",
                "--num_epochs",
                "5",
                "--mode",
                "dmp",
            ]
        )

    @classmethod
    def _run_trainer_ml_20m(cls) -> None:
        main(
            [
                "--dataset_name",
                "ml-20m",
                "--dataset_path",
                "/home/USER_NAME/datasets/",
                "--export_root",
                "/home/USER_NAME/saved_models/",
                "--lr",
                "0.001",
                "--mask_prob",
                "0.2",
                "--weight_decay",
                "0.00001",
                "--train_batch_size",
                "64",
                "--val_batch_size",
                "64",
                "--max_len",
                "200",
                "--emb_dim",
                "64",
                "--num_epochs",
                "10",
            ]
        )

    @classmethod
    def _run_trainer_ml_1m(cls) -> None:
        main(
            [
                "--dataset_name",
                "ml-1m",
                "--dataset_path",
                "/home/USER_NAME/datasets/",
                "--export_root",
                "/home/USER_NAME/saved_models/",
                "--lr",
                "0.001",
                "--mask_prob",
                "0.2",
                "--weight_decay",
                "0.00001",
                "--train_batch_size",
                "256",
                "--val_batch_size",
                "256",
                "--test_batch_size",
                "256",
                "--max_len",
                "100",
                "--emb_dim",
                "256",
                "--num_epochs",
                "30",
            ]
        )

    @test_utils.skip_if_asan
    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_main_function_random_ddp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer_random_ddp)()

    @test_utils.skip_if_asan
    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_main_function_random_dmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )
            elastic_launch(config=lc, entrypoint=self._run_trainer_random_dmp)()
