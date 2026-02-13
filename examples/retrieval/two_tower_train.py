#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import os
from typing import List, Optional

import click
import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import inference as trec_infer
from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.inference.state_dict_transform import (
    state_dict_gather,
    state_dict_to_device,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# OSS import
try:
    # @manual=//torchrec/github/examples/retrieval/data:dataloader
    from data.dataloader import get_dataloader

    # @manual=//torchrec/github/examples/retrieval:knn_index
    from knn_index import get_index

    # @manual=//torchrec/github/examples/retrieval/modules:two_tower
    from modules.two_tower import TwoTower, TwoTowerTrainTask
except ImportError:
    pass

# internal import
try:
    from .data.dataloader import get_dataloader  # noqa F811
    from .knn_index import get_index  # noqa F811
    from .modules.two_tower import TwoTower, TwoTowerTrainTask  # noqa F811
except ImportError:
    pass


@click.command()
@click.option(
    "--save_dir",
    type=click.STRING,
    default=None,
    help="Directory to save model and faiss index. If None, nothing is saved",
)
def main(save_dir: Optional[str]) -> None:
    train(save_dir=save_dir)


def train(
    num_embeddings: int = 1024**2,
    embedding_dim: int = 64,
    layer_sizes: Optional[List[int]] = None,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    num_iterations: int = 100,
    num_centroids: int = 100,
    num_subquantizers: int = 8,
    bits_per_code: int = 8,
    num_probe: int = 8,
    save_dir: Optional[str] = None,
) -> None:
    """
    Trains a simple Two Tower (UV) model, which is a simplified version of [A Dual Augmented Two-tower Model for Online Large-scale Recommendation](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf).
    Torchrec is used to shard the model, and is pipelined so that dataloading, data-parallel to model-parallel comms, and forward/backward are overlapped.
    It is trained on random data in the format of [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) dataset in SPMD fashion.
    The distributed model is gathered to CPU.
    The item (movie) towers embeddings are used to train a FAISS [IVFPQ](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint) index, which is serialized.
    The resulting `KNNIndex` can be queried with batched `torch.Tensor`, and will return the distances and indices for the approximate K nearest neighbors of the query embeddings. The model itself is also serialized.

    Args:
        num_embeddings (int): The number of embeddings the embedding table
        embedding_dim (int): embedding dimension of both embedding tables
        layer_sizes (List[int]): list representing layer sizes of the MLP. Last size is the final embedding size
        learning_rate (float): learning_rate
        batch_size (int): batch size to use for training
        num_iterations (int): number of train batches
        num_centroids (int): The number of centroids (Voronoi cells)
        num_subquantizers (int): The number of subquanitizers in Product Quantization (PQ) compression of subvectors
        bits_per_code (int): The number of bits for each subvector in Product Quantization (PQ)
        num_probe (int): The number of centroids (Voronoi cells) to probe. Must be <= num_centroids. Sweeping powers of 2 for nprobe and picking one of those based on recall statistics (e.g., 1, 2, 4, 8, ..,) is typically done.
        save_dir (Optional[str]): Directory to save model and faiss index. If None, nothing is saved
    """
    if layer_sizes is None:
        layer_sizes = [128, 64]

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)

    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_name],
        )
        for feature_name in two_tower_column_names
    ]
    embedding_bag_collection = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_model = TwoTower(
        embedding_bag_collection=embedding_bag_collection,
        layer_sizes=layer_sizes,
        device=device,
    )
    two_tower_train_task = TwoTowerTrainTask(two_tower_model)
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        two_tower_train_task.two_tower.ebc.parameters(),
        {"lr": learning_rate},
    )
    model = DistributedModelParallel(
        module=two_tower_train_task,
        device=device,
    )

    optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.Adam(params, lr=learning_rate),
    )

    dataloader = get_dataloader(
        batch_size=batch_size,
        num_embeddings=num_embeddings,
        pin_memory=(backend == "nccl"),
    )
    dl_iterator = iter(dataloader)
    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )

    # Train model
    for _ in range(num_iterations):
        try:
            train_pipeline.progress(dl_iterator)
        except StopIteration:
            break

    checkpoint_pg = dist.new_group(backend="gloo")
    # Copy sharded state_dict to CPU.
    cpu_state_dict = state_dict_to_device(
        model.state_dict(), pg=checkpoint_pg, device=torch.device("cpu")
    )

    ebc_cpu = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),
    )
    two_tower_cpu = TwoTower(
        embedding_bag_collection=ebc_cpu,
        layer_sizes=layer_sizes,
    )
    two_tower_train_cpu = TwoTowerTrainTask(two_tower_cpu)
    if rank == 0:
        two_tower_train_cpu = two_tower_train_cpu.to_empty(device="cpu")
    state_dict_gather(cpu_state_dict, two_tower_train_cpu.state_dict())
    dist.barrier()

    # Create and train FAISS index for the item (movie) tower on CPU
    if rank == 0:
        index = get_index(
            embedding_dim=embedding_dim,
            num_centroids=num_centroids,
            num_probe=num_probe,
            num_subquantizers=num_subquantizers,
            bits_per_code=bits_per_code,
            device=torch.device("cpu"),
        )

        values = torch.tensor(list(range(num_embeddings)), device=torch.device("cpu"))
        kjt = KeyedJaggedTensor(
            keys=two_tower_column_names,
            values=values,
            lengths=torch.tensor(
                [0] * num_embeddings + [1] * num_embeddings,
                device=torch.device("cpu"),
            ),
        )

        # Get the embeddings of the item(movie) tower by querying model
        with torch.no_grad():
            lookups = two_tower_cpu.ebc(kjt)[two_tower_column_names[1]]
            item_embeddings = two_tower_cpu.candidate_proj(lookups)
        index.train(item_embeddings)
        index.add(item_embeddings)

        if save_dir is not None:
            save_dir = save_dir.rstrip("/")
            quant_model = trec_infer.modules.quantize_embeddings(
                model, dtype=torch.qint8, inplace=True
            )
            torch.save(quant_model.state_dict(), f"{save_dir}/model.pt")
            faiss.write_index(index, f"{save_dir}/faiss.index")


if __name__ == "__main__":
    main()
