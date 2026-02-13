#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3


# from pathlib import Path
import argparse
import os
import sys
from typing import Any, cast, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ModuleSharder
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from tqdm import tqdm


# OSS import
try:

    # @manual=//torchrec/github/examples/bert4rec:bert4rec_metrics
    from bert4rec_metrics import recalls_and_ndcgs_for_ks

    # @manual=//torchrec/github/examples/bert4rec/data:bert4rec_movielens_datasets
    from data.bert4rec_movielens_datasets import Bert4RecPreprocsser, get_raw_dataframe

    # @manual=//torchrec/github/examples/bert4rec/dataloader:bert4rec_movielens_dataloader
    from dataloader.bert4rec_movielens_dataloader import Bert4RecDataloader

    # @manual=//torchrec/github/examples/bert4rec/models:bert4rec
    from models.bert4rec import BERT4Rec
except ImportError:
    pass

# internal import
try:
    from .bert4rec_metrics import recalls_and_ndcgs_for_ks  # noqa F811
    from .data.bert4rec_movielens_datasets import (  # noqa F811
        Bert4RecPreprocsser,
        get_raw_dataframe,
    )
    from .dataloader.bert4rec_movielens_dataloader import (  # noqa F811
        Bert4RecDataloader,
    )
    from .models.bert4rec import BERT4Rec  # noqa F811
except ImportError:
    pass


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec + lightning app")
    parser.add_argument(
        "--min_user_count", type=int, default=5, help="minimum user ratings count"
    )
    parser.add_argument(
        "--min_item_count",
        type=int,
        default=0,
        help="minimum item count for each valid user",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="max length of the Bert embedding dimension",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.15,
        help="probability of the mask",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ml-1m",
        help="dataset for experiment, current support ml-1m, ml-20m",
    )
    parser.add_argument(
        "--min_rating",
        type=int,
        default=0,
        help="minimum valid rating",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="the number of epoch to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--decay_step",
        type=int,
        default="25",
        help="the step of weight decay",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="gamma of the lr scheduler",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="train batch size",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=128,
        help="val batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="test batch size",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=256,
        help="dimension of the hidden layer embedding",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="number of header of attention",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="number of layers of attention",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a folder containing the dataset.",
    )
    parser.add_argument(
        "--export_root",
        type=str,
        default="",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--random_user_count",
        type=int,
        default=10,
        help="number of random users",
    )
    parser.add_argument(
        "--random_item_count",
        type=int,
        default=30,
        help="number of random items",
    )
    parser.add_argument(
        "--random_size",
        type=int,
        default=300,
        help="number of random sample size",
    )
    parser.add_argument(
        "--dupe_factor",
        type=int,
        default=3,
        help="number of duplication while generating the random masked seqs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dmp",
        help="dmp (distirbuted model parallel) or ddp (distributed data parallel)",
    )
    return parser.parse_args(argv)


def _dict_mean(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
    return mean_dict


def _to_kjt(seqs: torch.LongTensor, device: torch.device) -> KeyedJaggedTensor:
    seqs_list = list(seqs)
    lengths = torch.IntTensor([value.size(0) for value in seqs_list])
    values = torch.cat(seqs_list, dim=0)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["item"], values=values, lengths=lengths
    ).to(device)
    return kjt


def _calculate_metrics(
    model: Union[DDP, DMP],
    batch: List[torch.LongTensor],
    metric_ks: List[int],
    device: torch.device,
) -> Dict[str, float]:
    seqs, candidates, labels = batch
    kjt = _to_kjt(seqs, device)
    scores = model(kjt)
    scores = scores[:, -1, :]
    scores = scores.gather(1, candidates)

    metrics = recalls_and_ndcgs_for_ks(scores, labels, metric_ks)
    return metrics


def _train_one_epoch(
    model: Union[DDP, DMP],
    train_loader: data_utils.DataLoader,
    device: torch.device,
    optimizer: optim.Adam,
    lr_scheduler: optim.lr_scheduler.StepLR,
    epoch: int,
) -> None:
    """
    Train model for 1 epoch. Helper function for train_val_test.

    Args:
        model (Union[DDP, DMP]): DMP or DDP model contains the Bert4Rec.
        train_loader (data_utils.DataLoader): DataLoader used for training.
        device (torch.device): the device to train/val/test
        optimizer (optim.Adam): Adam optimizer to train the model
        lr_scheduler (optim.lr_scheduler.StepLR): scheduler to control the learning rate
        epoch (int): the current epoch number

    Returns:
        None.
    """
    model.train()
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    loss_logs = []
    train_iterator = iter(train_loader)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    outputs = [None for _ in range(dist.get_world_size())]
    for _ in tqdm(iter(int, 1), desc=f"Epoch {epoch+1}"):
        try:
            batch = next(train_iterator)
            batch = [x.to(device) for x in batch]

            optimizer.zero_grad()
            seqs, labels = batch

            kjt = _to_kjt(seqs, device)
            logits = model(kjt)  # B x T x V

            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            labels = labels.view(-1)  # B*T
            loss = ce(logits, labels)

            loss.backward()

            optimizer.step()

            loss_logs.append(loss.item())

        except StopIteration:
            break
    dist.all_gather_object(outputs, sum(loss_logs) / len(loss_logs))
    if dist.get_rank() == 0:
        #  _SupportsSum)]]` but got `List[None]`.
        print(f"Epoch {epoch + 1}, average loss { (sum(outputs) or 0) /len(outputs)}")
    lr_scheduler.step()


def _validate(
    model: Union[DDP, DMP],
    val_loader: data_utils.DataLoader,
    device: torch.device,
    epoch: int,
    metric_ks: List[int],
    is_testing: bool = False,
) -> None:
    """
    Evaluate model. Computes and prints metrics including Recalls and NDCGs. Helper
    function for train_val_test.

    Args:
        model (Union[DDP, DMP]): DMP or DDP model contains the Bert4Rec.
        val_loader (data_utils.DataLoader): DataLoader used for validation.
        device (torch.device): the device to train/val/test
        epoch (int): the current epoch number
        metric_ks (List[int]): the metrics we want to validate
        is_testing (bool): if validation or testing

    Returns:
        None.
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    outputs = [None for _ in range(dist.get_world_size())]
    keys = ["Recall@1", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10"]
    metrics_log: Dict[str, List[float]] = {key: [] for key in keys}

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            batch = [x.to(device) for x in batch]

            metrics = _calculate_metrics(model, batch, metric_ks, device)

            for key in keys:
                metrics_log[key].append(metrics[key])

    metrics_avg = {
        key: sum(values) / len(values) for key, values in metrics_log.items()
    }
    dist.all_gather_object(outputs, metrics_avg)
    if dist.get_rank() == 0:
        print(
            f"{'Epoch ' + str(epoch + 1) if not is_testing else 'Test'}, metrics {_dict_mean(outputs)}"
        )


def train_val_test(
    model: Union[DDP, DMP],
    train_loader: data_utils.DataLoader,
    val_loader: data_utils.DataLoader,
    test_loader: data_utils.DataLoader,
    device: torch.device,
    optimizer: optim.Adam,
    lr_scheduler: optim.lr_scheduler.StepLR,
    num_epochs: int,
    metric_ks: List[int],
    export_root: str,
) -> None:
    """
    Train/validation/test loop. Ensure the dataloader will do the shuffling on each rank
    and will output the performance metrics like recalls and ndcgs

    Args:
        model (Union[DDP, DMP]): DMP or DDP model contains the Bert4Rec.
        train_loader (data_utils.DataLoader): DataLoader used for training.
        val_loader (data_utils.DataLoader): DataLoader used for validation.
        test_loader (data_utils.DataLoader): DataLoader used for testing.
        device (torch.device): the device to train/val/test
        optimizer (optim.Adam): Adam optimizer to train the model
        lr_scheduler (optim.lr_scheduler.StepLR): scheduler to control the learning rate
        num_epochs (int): the number of epochs to train
        metric_ks (List[int]): the metrics we want to validate
        export_root (str): the export root of the saved models

    Returns:
        None.
    """
    _validate(model, val_loader, device, -1, metric_ks)
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        _train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            lr_scheduler,
            epoch,
        )
        _validate(model, val_loader, device, epoch, metric_ks)
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                export_root + f"epoch_{epoch + 1}_model.pth",
            )
            print(f"epoch {epoch + 1} model has been saved to {export_root}")
    _validate(model, test_loader, device, num_epochs, metric_ks, True)


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Bert4Rec Model
    (https://arxiv.org/abs/1904.06690). The Bert4Rec model contains both data parallel
    components (e.g. transformation blocks) and model parallel
    components (e.g. item embeddings). The Bert4Rec model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or the movielens dataset
    (https://grouplens.org/datasets/movielens/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)
    use_dmp = args.mode == "dmp"
    metric_ks: List[int] = (
        [1, 5, 10, 20, 50, 100] if args.dataset_name != "random" else [1, 5, 10]
    )
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()

    raw_data = get_raw_dataframe(
        args.dataset_name,
        args.random_user_count,
        args.random_item_count,
        args.random_size,
        args.min_rating,
        args.dataset_path,
    )

    df = Bert4RecPreprocsser(
        raw_data,
        args.min_rating,
        args.min_user_count,
        args.min_item_count,
        args.dataset_name,
        args.max_len,
        args.mask_prob,
        args.dupe_factor,
    ).get_processed_dataframes()
    # 0 for padding, item_count + 1 for mask
    vocab_size = len(df["smap"]) + 2
    bert4recDataloader = Bert4RecDataloader(
        df,
        args.train_batch_size,
        args.val_batch_size,
        args.test_batch_size,
    )
    (
        train_loader,
        val_loader,
        test_loader,
    ) = bert4recDataloader.get_pytorch_dataloaders(rank, world_size)
    model_bert4rec = BERT4Rec(
        vocab_size,
        args.max_len,
        emb_dim=args.emb_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    if use_dmp:
        fused_params: Dict[str, Any] = {}
        fused_params["optimizer"] = EmbOptimType.ADAM
        fused_params["learning_rate"] = args.lr
        fused_params["weight_decay"] = args.weight_decay
        model = DMP(
            module=model_bert4rec,
            device=device,
            sharders=[
                cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder(fused_params))
            ],
        )
        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(model.named_parameters())),
            lambda params: optim.Adam(
                params, lr=args.lr, weight_decay=args.weight_decay
            ),
        )

        optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    else:
        device_ids = [rank] if backend == "nccl" else None
        """
        Another way to do DDP is to specify the sharding_type for TorchRec as Data_parallel
        Here we provide an example of how to do it:
        First we constraint the sharding_types to only use data_parallel in sharder,
        then we use DMP to wrap it:

            sharding_types = [ShardingType.DATA_PARALLEL.value]
            constraints[
                "item_embedding"
            ] = torchrec.distributed.planner.ParameterConstraints(sharding_types=sharding_types)
            sharders = [
                cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder(fused_params))
            ]
            pg = dist.GroupMember.WORLD
            model = DMP(
                module=model_bert4rec,
                device=device,
                plan=torchrec.distributed.planner.EmbeddingShardingPlanner(
                topology=torchrec.distributed.planner.Topology(
                    world_size=world_size,
                    compute_device=device.type,
                ),
                constraints=constraints
            ).collective_plan(model_bert4rec, sharders, pg),
                sharders=sharders,
            )

        """
        model = DDP(model_bert4rec, device_ids=device_ids)
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.gamma
    )

    train_val_test(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        #  `Union[CombinedOptimizer, Adam]`.
        optimizer,
        lr_scheduler,
        args.num_epochs,
        metric_ks,
        args.export_root,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
