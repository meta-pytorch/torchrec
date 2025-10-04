#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Any, List

import click
import torch
from torchrec.distributed.benchmark.base import benchmark_func
from torchrec.distributed.embedding import EmbeddingCollectionContext
from torchrec.distributed.embedding_sharding import _set_sharding_context_post_a2a
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _set_sharding_context_post_a2a_previous(
    kjts: List[KeyedJaggedTensor],
    ctx: EmbeddingCollectionContext,
) -> None:
    for kjt, sharding_context in zip(kjts, getattr(ctx, "sharding_contexts", [])):
        if (
            hasattr(sharding_context, "batch_size_per_rank_per_feature")
            and kjt.variable_stride_per_key()
            and kjt.stride_per_key_per_rank()
        ):
            sharding_context.batch_size_per_rank_per_feature = [
                [
                    kjt.stride_per_key_per_rank()[i][j]
                    for i in range(len(kjt.stride_per_key_per_rank()))
                ]
                for j in range(len(kjt.stride_per_key_per_rank()[0]))
            ]


# buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_set_sharding_context_post_a2a -- --num_list=0 --num_keys=0 | grep set_sharding_context_post_a2a


@click.command()
@click.option("--num_list", default=100)
@click.option("--num_keys", default=100)
def main(
    num_list: int,
    num_keys: int,
) -> None:
    if num_list == 0 and num_keys == 0:
        for num_list in [100, 1000, 10000]:
            for num_keys in [10, 100]:
                op_bench(num_list, num_keys, _set_sharding_context_post_a2a_previous)
                op_bench(num_list, num_keys, _set_sharding_context_post_a2a)
    else:
        op_bench(num_list, num_keys, _set_sharding_context_post_a2a_previous)
        op_bench(num_list, num_keys, _set_sharding_context_post_a2a)


def op_bench(
    num_list: int,
    num_keys: int,
    func_to_benchmark: Any,  # pyre-ignore[2]
) -> None:
    kjts = [
        KeyedJaggedTensor(
            keys=["dummy_id"] * num_keys,
            values=torch.IntTensor([1] * num_keys),
            lengths=torch.IntTensor([1] * num_keys),
            stride_per_key_per_rank=[[1]] * num_keys,
        )
        for _ in range(num_list)
    ]
    for kjt in kjts:
        kjt._variable_stride_per_key = True
    ctx = EmbeddingCollectionContext(
        sharding_contexts=[
            SequenceShardingContext(batch_size_per_rank_per_feature=[])
            for _ in range(num_list)
        ]
    )

    bench_inputs = []

    result = benchmark_func(
        name=f"{func_to_benchmark.__name__}-{num_list}-{num_keys}",
        bench_inputs=bench_inputs,
        prof_inputs=bench_inputs,
        num_benchmarks=10,
        num_profiles=2,
        profile_dir=".",
        world_size=1,
        func_to_benchmark=func_to_benchmark,
        benchmark_func_kwargs={"kjts": kjts, "ctx": ctx},
        rank=0,
        pre_gpu_load=0,
        device_type="cpu",
    )
    print(result)


if __name__ == "__main__":
    main()
