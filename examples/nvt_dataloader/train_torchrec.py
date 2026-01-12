#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import time
from typing import cast, Iterator, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics as metrics
import torchrec
import torchrec.distributed as trec_dist
import torchrec.optim as trec_optim
from nvt_binary_dataloader import NvtBinaryDataloader
from pyre_extensions import none_throws
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.fused_embedding_modules import fuse_embedding_optimizer
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="local batch size to use for training",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--binary_path",
        type=str,
        default="/data/criteo_1tb/criteo_binary/split/",
        help="Location for binary datafiles",
    )
    parser.add_argument(
        "--change_lr",
        dest="change_lr",
        action="store_true",
        help="Flag to determine whether learning rate should be changed part way through training.",
    )
    parser.add_argument(
        "--lr_change_point",
        type=float,
        default=0.80,
        help="The point through training at which learning rate should change to the value set by"
        " lr_after_change_point. The default value is 0.80 which means that 80% through the total iterations (totaled"
        " across all epochs), the learning rate will change.",
    )
    parser.add_argument(
        "--lr_after_change_point",
        type=float,
        default=3.0,
        help="Learning rate after change point in first epoch.",
    )
    parser.add_argument(
        "--throughput_check_freq_within_epoch",
        type=int,
        default=1000,
        help="Frequency at QPS will be output within an epoch.",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=10000,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    return parser.parse_args(argv)


def _eval(
    train_pipeline: TrainPipelineSparseDist, it: Iterator[Batch]
) -> Tuple[float, float, float]:
    train_pipeline._model.eval()

    device = train_pipeline._device
    auroc = metrics.AUROC(compute_on_step=False).to(device)
    accuracy = metrics.Accuracy(compute_on_step=False).to(device)
    val_losses = []
    step = 0
    with torch.no_grad():
        while True:
            try:
                loss, logits, labels = train_pipeline.progress(it)
                val_losses.append(loss)
                preds = torch.sigmoid(logits)

                labels = labels.to(torch.int32)
                auroc(preds, labels)
                accuracy(preds, labels)
                step += 1
            except StopIteration:
                break
    auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    bce_loss = torch.mean(torch.stack(val_losses))
    return (auroc_result, accuracy_result, bce_loss)


def main(argv: List[str]):
    args = parse_args(argv)
    rank = int(os.environ["LOCAL_RANK"])

    print("Running with args", args)

    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_embeddings_per_feature = None
    if args.num_embeddings_per_feature is not None:
        num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
    train_loader = NvtBinaryDataloader(
        binary_file_path=os.path.join(args.binary_path, "train"),
        batch_size=args.batch_size,
    ).get_dataloader(rank=rank, world_size=world_size)
    val_loader = NvtBinaryDataloader(
        binary_file_path=os.path.join(args.binary_path, "validation"),
        batch_size=args.batch_size,
    ).get_dataloader(rank=rank, world_size=world_size)

    test_loader = NvtBinaryDataloader(
        binary_file_path=os.path.join(args.binary_path, "test"),
        batch_size=args.batch_size,
    ).get_dataloader(rank=rank, world_size=world_size)

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(num_embeddings_per_feature)[feature_idx]
                if num_embeddings_per_feature is not None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    train_model = DLRMTrain(
        DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=list(
                map(int, args.dense_arch_layer_sizes.split(","))
            ),
            over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
            dense_device=device,
        ),
    )

    train_model = fuse_embedding_optimizer(
        train_model,
        optimizer_type=(
            torchrec.optim.RowWiseAdagrad if args.adagrad else torch.optim.SGD
        ),
        optimizer_kwargs={"learning_rate": args.learning_rate},
        device=torch.device("meta"),
    )

    sharders = cast(
        List[ModuleSharder[nn.Module]],
        [
            FusedEmbeddingBagCollectionSharder(),
        ],
    )

    pg = dist.GroupMember.WORLD
    hbm_cap = torch.cuda.get_device_properties(device).total_memory
    local_world_size = trec_dist.comm.get_local_size(world_size)
    model = DistributedModelParallel(
        module=train_model,
        device=device,
        env=trec_dist.ShardingEnv.from_process_group(pg),
        plan=trec_dist.planner.EmbeddingShardingPlanner(
            topology=trec_dist.planner.Topology(
                world_size=world_size,
                compute_device=device.type,
                local_world_size=local_world_size,
                hbm_cap=hbm_cap,
            ),
            storage_reservation=trec_dist.planner.storage_reservations.HeuristicalStorageReservation(
                percentage=0.25,
            ),
            batch_size=args.batch_size,
        ).collective_plan(train_model, sharders, pg),
        sharders=sharders,
    )

    non_fused_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: (
            torch.optim.Adagrad(params, lr=args.learning_rate)
            if args.adagrad
            else torch.optim.SGD(params, lr=args.learning_rate)
        ),
    )

    opt = trec_optim.keyed.CombinedOptimizer(
        [non_fused_optimizer, model.fused_optimizer]
    )

    train_pipeline = TrainPipelineSparseDist(
        model,
        opt,
        device,
    )

    throughput = ThroughputMetric(
        batch_size=args.batch_size,
        world_size=world_size,
        window_seconds=30,
        warmup_steps=10,
    )

    changing_point_steps = (
        TOTAL_TRAINING_SAMPLES * args.lr_change_point // args.batch_size // world_size
    )
    for epoch in range(args.epochs):
        print(f"starting the {epoch} epoch now")
        start_time = time.time()
        it = iter(train_loader)
        step = 0
        losses = []
        while True:
            try:
                train_pipeline._model.train()
                loss, _logits, _labels = train_pipeline.progress(it)

                if args.change_lr and step == changing_point_steps:
                    print(f"Changing learning rate to: {args.lr_after_change_point}")
                    optimizer = train_pipeline._optimizer
                    lr = args.lr_after_change_point
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                throughput.update()
                losses.append(loss)

                if step % args.throughput_check_freq_within_epoch == 0 and step != 0:
                    # infra calculation
                    throughput_val = throughput.compute()
                    if rank == 0:
                        print("step", step)
                        print("throughput", throughput_val)
                    losses = []
                if step % args.validation_freq_within_epoch == 0 and step != 0:
                    # metrics calculation
                    validation_it = iter(val_loader)
                    auroc_result, accuracy_result, bce_loss = _eval(
                        train_pipeline, validation_it
                    )
                    if rank == 0:
                        print(f"AUROC over validation set: {auroc_result}.")
                        print(f"Accuracy over validation set: {accuracy_result}.")
                        print(
                            "binary cross entropy loss",
                            bce_loss / (args.batch_size),
                        )

                    test_it = iter(test_loader)
                    auroc_result, accuracy_result, bce_loss = _eval(
                        train_pipeline, test_it
                    )
                    if rank == 0:
                        print(f"AUROC over test set: {auroc_result}.")
                        print(f"Accuracy over test set: {accuracy_result}.")

                step += 1

            except StopIteration:
                print("Reached stop iteration")
                break
        train_time = time.time()
        if rank == 0:
            print(f"this epoch training takes {train_time - start_time}")

        # eval
        val_it = iter(val_loader)
        auroc_result, accuracy_result, bce_loss = _eval(train_pipeline, val_it)
        if rank == 0:
            print(f"AUROC over validation set: {auroc_result}.")
            print(f"Accuracy over validation set: {accuracy_result}.")
            print(
                "binary cross entropy loss over validation set",
                bce_loss / (args.batch_size),
            )
        # test
        test_it = iter(test_loader)
        auroc_result, accuracy_result, bce_loss = _eval(train_pipeline, test_it)
        if rank == 0:
            print(f"AUROC over test set: {auroc_result}.")
            print(f"Accuracy over test set: {accuracy_result}.")
            print(
                "binary cross entropy loss over test set",
                bce_loss / (args.batch_size),
            )


if __name__ == "__main__":
    main(sys.argv[1:])
