#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import click
import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch
from torchrec import inference as trec_infer
from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# OSS import
try:

    # @manual=//torchrec/github/examples/retrieval:knn_index
    from knn_index import get_index

    # @manual=//torchrec/github/examples/retrieval/modules:two_tower
    from modules.two_tower import (  # noqa F811
        convert_TwoTower_to_TwoTowerRetrieval,
        TwoTowerRetrieval,
    )
except ImportError:
    pass

# internal import
try:
    from .data.dataloader import get_dataloader  # noqa F811
    from .knn_index import get_index  # noqa F811
    from .modules.two_tower import (  # noqa F811
        convert_TwoTower_to_TwoTowerRetrieval,
        TwoTowerRetrieval,
    )
except ImportError:
    pass


@click.command()
@click.option(
    "--load_dir",
    type=click.STRING,
    default=None,
    help="Directory to load model and faiss index from. If None, uses random data",
)
def main(load_dir: Optional[str]) -> None:
    infer(load_dir=load_dir)


def infer(
    num_embeddings: int = 1024 * 1024,
    embedding_dim: int = 64,
    layer_sizes: Optional[List[int]] = None,
    num_centroids: int = 100,
    k: int = 100,
    num_subquantizers: int = 8,
    bits_per_code: int = 8,
    num_probe: int = 8,
    model_device_idx: int = 0,
    faiss_device_idx: int = 0,
    batch_size: int = 32,
    load_dir: Optional[str] = None,
    world_size: int = 2,
) -> None:
    """
    Loads the serialized model and FAISS index from `two_tower_train.py`.
    A `TwoTowerRetrieval` model is instantiated, which wraps the `KNNIndex`, the query (user) tower and the candidate item (movie) tower inside an `nn.Module`.
    The retreival model is quantized using [`torchrec.quant`](https://pytorch.org/torchrec/torchrec.quant.html).
    The serialized `TwoTower` model weights trained before are converted into `TwoTowerRetrieval` which are loaded into the retrieval model.
    The seralized trained FAISS index is also loaded.
    The entire retreival model can be queried with a batch of candidate (user) ids and returns logits which can be used in ranking.

    Args:
        num_embeddings (int): The number of embeddings the embedding table
        embedding_dim (int): embedding dimension of both embedding tables
        layer_sizes (str): Comma separated list representing layer sizes of the MLP. Last size is the final embedding size
        num_centroids (int): The number of centroids (Voronoi cells)
        k (int): The number of nearest neighbors to retrieve
        num_subquantizers (int): The number of subquanitizers in Product Quantization (PQ) compression of subvectors
        bits_per_code (int): The number of bits for each subvector in Product Quantization (PQ)
        num_probe (int): The number of centroids (Voronoi cells) to probe. Must be <= num_centroids. Sweeping powers of 2 for nprobe and picking one of those based on recall statistics (e.g., 1, 2, 4, 8, ..,) is typically done.
        model_device_idx (int): device index to place model on
        faiss_device_idx (int): device index to place FAISS index on
        batch_size (int): batch_size of the random batch used to query Retrieval model at the end of the script
        load_dir (Optional[str]): Directory to load model and faiss index from. If None, uses random data
    """
    if layer_sizes is None:
        layer_sizes = [128, 64]
    assert torch.cuda.is_available(), "This example requires a GPU"

    device: torch.device = torch.device(f"cuda:{model_device_idx}")
    torch.cuda.set_device(device)

    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    ebcs = []
    for feature_name in two_tower_column_names:
        config = EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_name],
            data_type=DataType.FP16,
        )
        ebcs.append(
            EmbeddingBagCollection(
                tables=[config],
                device=torch.device("meta"),
            )
        )

    retrieval_sd = None
    if load_dir is not None:
        load_dir = load_dir.rstrip("/")
        index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            faiss_device_idx,
            faiss.read_index(f"{load_dir}/faiss.index"),
        )
        two_tower_sd = torch.load(f"{load_dir}/model.pt", weights_only=True)
        retrieval_sd = convert_TwoTower_to_TwoTowerRetrieval(
            two_tower_sd,
            [f"t_{two_tower_column_names[0]}"],
            [f"t_{two_tower_column_names[1]}"],
        )
    else:
        embeddings = torch.rand((num_embeddings, embedding_dim)).to(
            torch.device(f"cuda:{faiss_device_idx}")
        )
        index = get_index(
            embedding_dim=embedding_dim,
            num_centroids=num_centroids,
            num_probe=num_probe,
            num_subquantizers=num_subquantizers,
            bits_per_code=bits_per_code,
            device=torch.device(f"cuda:{faiss_device_idx}"),
        )
        index.train(embeddings)
        index.add(embeddings)

    retrieval_model = TwoTowerRetrieval(
        index,
        ebcs[0],
        ebcs[1],
        layer_sizes,
        k,
        device,
        dtype=torch.float16,
    )

    constraints = {}
    for feature_name in two_tower_column_names:
        constraints[f"t_{feature_name}"] = ParameterConstraints(
            sharding_types=[ShardingType.TABLE_WISE.value],
            compute_kernels=[EmbeddingComputeKernel.QUANT.value],
        )

    quant_model = trec_infer.modules.quantize_embeddings(
        retrieval_model,
        dtype=torch.qint8,
        inplace=True,
        output_dtype=torch.float16,
    )

    dmp = DistributedModelParallel(
        module=quant_model,
        device=device,
        env=ShardingEnv.from_local(world_size=world_size, rank=model_device_idx),
        init_data_parallel=False,
    )
    if retrieval_sd is not None:
        dmp.load_state_dict(retrieval_sd)

    # query with random batch
    values = torch.randint(0, num_embeddings, (batch_size,), device=device)
    batch = KeyedJaggedTensor(
        keys=[two_tower_column_names[0]],
        values=values,
        lengths=torch.tensor([1] * batch_size, device=device),
    )
    dmp(batch)


if __name__ == "__main__":
    main()
