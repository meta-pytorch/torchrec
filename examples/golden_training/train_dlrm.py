#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from typing import List, Optional

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.utils.data import IterableDataset
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from tqdm import tqdm


def _get_random_dataset(
    num_embeddings: int,
    batch_size: int = 32,
) -> IterableDataset:
    return RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=batch_size,
        hash_size=num_embeddings,
        ids_per_feature=1,
        num_dense=len(DEFAULT_INT_NAMES),
    )


@record
def main() -> None:
    train()


def train(
    num_embeddings: int = 1024**2,
    embedding_dim: int = 128,
    dense_arch_layer_sizes: Optional[List[int]] = None,
    over_arch_layer_sizes: Optional[List[int]] = None,
    learning_rate: float = 0.1,
    num_iterations: int = 1000,
    qcomm_forward_precision: Optional[CommType] = CommType.FP16,
    qcomm_backward_precision: Optional[CommType] = CommType.BF16,
) -> None:
    """
    Constructs and trains a DLRM model (using random dummy data). Each script is run on each process (rank) in SPMD fashion.
    The embedding layers will be sharded across available ranks

    qcomm_forward_precision: Compression used in forwards pass. FP16 is the recommended usage. INT8 and FP8 are in development, but feel free to try them out.
    qcomm_backward_precision: Compression used in backwards pass. We recommend using BF16 to ensure training stability.

    The effects of quantized comms will be most apparent in large training jobs across multiple nodes where inter host communication is expensive.
    """
    if dense_arch_layer_sizes is None:
        dense_arch_layer_sizes = [64, embedding_dim]
    if over_arch_layer_sizes is None:
        over_arch_layer_sizes = [64, 1]

    # Init process_group , device, rank, backend
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)

    # Construct DLRM module
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    dlrm_model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dense_device=device,
    )
    train_model = DLRMTrain(dlrm_model)

    apply_optimizer_in_backward(
        RowWiseAdagrad,
        train_model.model.sparse_arch.parameters(),
        {"lr": learning_rate},
    )
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                # pyrefly: ignore[bad-argument-type]
                forward_precision=qcomm_forward_precision,
                # pyrefly: ignore[bad-argument-type]
                backward_precision=qcomm_backward_precision,
            )
        )
        if backend == "nccl"
        else None
    )
    sharder = EmbeddingBagCollectionSharder(qcomm_codecs_registry=qcomm_codecs_registry)

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        # pyrefly: ignore[bad-argument-type]
        sharders=[sharder],
    )

    non_fused_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=learning_rate),
    )
    # Overlap comm/compute/device transfer during training through train_pipeline
    train_pipeline = TrainPipelineSparseDist(
        model,
        non_fused_optimizer,
        device,
    )

    # train model
    train_iterator = iter(
        _get_random_dataset(
            num_embeddings=num_embeddings,
        )
    )
    for _ in tqdm(range(int(num_iterations)), mininterval=5.0):
        train_pipeline.progress(train_iterator)


if __name__ == "__main__":
    main()
