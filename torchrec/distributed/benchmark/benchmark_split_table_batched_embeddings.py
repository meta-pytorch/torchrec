#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Dict, List

import click

import torch

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.benchmark.base import benchmark_func
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@click.command()
@click.option("--num-embeddings", default=100_000)
@click.option("--embedding-dim", default=128)
@click.option("--num-tables", default=4)
@click.option("--batch-size", default=262144)
@click.option("--bag-size", default=10)
def main(
    num_embeddings: int,
    embedding_dim: int,
    num_tables: int,
    batch_size: int,
    bag_size: int,
) -> None:
    if embedding_dim == 0:
        for embedding_dim in [4, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            op_bench(num_embeddings, embedding_dim, num_tables, batch_size, bag_size)
    else:
        op_bench(num_embeddings, embedding_dim, num_tables, batch_size, bag_size)


def op_bench(
    num_embeddings: int,
    embedding_dim: int,
    num_tables: int,
    batch_size: int,
    bag_size: int,
) -> None:
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                num_embeddings,
                embedding_dim,
                EmbeddingLocation.DEVICE,
                (
                    ComputeDevice.CUDA
                    if torch.cuda.is_available()
                    else ComputeDevice.CPU
                ),
            )
        ]
        * num_tables,
        optimizer=OptimType.EXACT_ADAGRAD,
        learning_rate=0.1,
        eps=0.1,
        weights_precision=SparseType.FP32,
        stochastic_rounding=False,
        output_dtype=SparseType.FP32,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.NONE,
    )

    def _func_to_benchmark(
        kjts: List[Dict[str, KeyedJaggedTensor]],
        model: torch.nn.Module,
    ) -> torch.Tensor:
        kjt = kjts[0]["feature"]
        return model.forward(kjt.values(), kjt.offsets())

    # breakpoint()  # import fbvscode; fbvscode.set_trace()
    tables = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            name="table_0",
            feature_names=["feature_0"],
        )
    ]
    inputs = ModelInput.generate(
        tables=tables,
        weighted_tables=[],
        batch_size=batch_size,
        world_size=1,
        num_float_features=0,
        pooling_avg=10,
        device=torch.device("cuda"),
    )[0].idlist_features

    result = benchmark_func(
        name=f"SplitTableBatchedEmbeddingBagsCodegen-{num_embeddings}-{embedding_dim}-{num_tables}-{batch_size}-{bag_size}",
        bench_inputs=[{"feature": inputs}],
        prof_inputs=[{"feature": inputs}],
        num_benchmarks=10,
        num_profiles=10,
        profile_dir=".",
        world_size=1,
        func_to_benchmark=_func_to_benchmark,
        benchmark_func_kwargs={"model": emb},
        rank=0,
        pre_gpu_load=3,
    )

    print(result)


if __name__ == "__main__":
    main()
