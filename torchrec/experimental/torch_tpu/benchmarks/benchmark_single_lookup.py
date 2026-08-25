#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Accuracy and latency benchmark for TPU Pallas embedding lookups."""

import argparse
import os
import time
from typing import Any, cast, List, Optional, Tuple

import torch

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
from torch_tpu._internal import profiler, sync
from torchrec.distributed.batched_embedding_kernel import (
    BatchedTPUEmbedding,
    BatchedTPUEmbeddingBag,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE = 8
PROFILE_NUMBER_TIMES = 10
BENCHMARK_TIMES = 100
BATCH_SIZES = [10, 40, 160, 640, 1024, 2048, 1024 * 32]
EMBEDDING_DIMS = [32, 64, 128, 256, 512]
BENCH_BATCH_SIZE = 2048


def make_configs(collection: str, dim: int) -> List[Any]:
    if collection == "ec":
        return [
            EmbeddingConfig(
                name="user_table",
                embedding_dim=dim,
                num_embeddings=100000,
                feature_names=["user"],
            ),
            EmbeddingConfig(
                name="product_table",
                embedding_dim=dim,
                num_embeddings=1000000,
                feature_names=["product"],
            ),
        ]
    return [
        EmbeddingBagConfig(
            name="user_table",
            embedding_dim=dim,
            num_embeddings=100000,
            feature_names=["user"],
            pooling=PoolingType.SUM,
        ),
        EmbeddingBagConfig(
            name="product_table",
            embedding_dim=dim,
            num_embeddings=1000000,
            feature_names=["product"],
            pooling=PoolingType.SUM,
        ),
    ]


def grouped_config(collection: str, configs: List[Any]) -> GroupedEmbeddingConfig:
    pooling = PoolingType.NONE if collection == "ec" else PoolingType.SUM
    return GroupedEmbeddingConfig(
        data_type=DataType.FP32,
        pooling=pooling,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=EmbeddingComputeKernel.UNFUSED_TPU,
        embedding_tables=[
            ShardedEmbeddingTable(
                name=config.name,
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
                data_type=DataType.FP32,
                feature_names=config.feature_names,
                pooling=pooling,
                is_weighted=False,
                has_feature_processor=False,
                compute_kernel=EmbeddingComputeKernel.UNFUSED_TPU,
                local_rows=config.num_embeddings,
                local_cols=config.embedding_dim,
            )
            for config in configs
        ],
    )


def make_modules(
    collection: str, configs: List[Any]
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    config = grouped_config(collection, configs)
    if collection == "ec":
        return (
            EmbeddingCollection(tables=configs, device=torch.device("cpu")),
            BatchedTPUEmbedding(config=config, device=torch.device("tpu")),
        )
    return (
        EmbeddingBagCollection(tables=configs, device=torch.device("cpu")),
        BatchedTPUEmbeddingBag(config=config, device=torch.device("tpu")),
    )


def copy_weights(
    collection: str,
    cpu_collection: torch.nn.Module,
    tpu_kernel: torch.nn.Module,
    configs: List[Any],
) -> None:
    assert isinstance(tpu_kernel, (BatchedTPUEmbedding, BatchedTPUEmbeddingBag))
    for weight, config in zip(tpu_kernel.split_embedding_weights(), configs):
        if collection == "ec":
            assert isinstance(cpu_collection, EmbeddingCollection)
            source = cpu_collection.embeddings[config.name].weight
        else:
            assert isinstance(cpu_collection, EmbeddingBagCollection)
            source = cpu_collection.embedding_bags[config.name].weight
        weight.copy_(cast(torch.Tensor, source))


def reference_output(
    collection: str, module: torch.nn.Module, kjt: KeyedJaggedTensor
) -> torch.Tensor:
    if collection == "ec":
        assert isinstance(module, EmbeddingCollection)
        jt_dict = module(kjt)
        return torch.cat([jt_dict[key].values() for key in kjt.keys()], dim=0)
    assert isinstance(module, EmbeddingBagCollection)
    keyed_tensor = module(kjt)
    return torch.cat([keyed_tensor[key] for key in kjt.keys()], dim=1)


def tpu_output(module: torch.nn.Module, kjt: KeyedJaggedTensor) -> torch.Tensor:
    assert isinstance(module, (BatchedTPUEmbedding, BatchedTPUEmbeddingBag))
    return module(kjt)


def to_tpu(kjt: KeyedJaggedTensor) -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=kjt.keys(),
        values=kjt.values().to(device="tpu", dtype=torch.int32),
        lengths=kjt.lengths().to(device="tpu"),
    )


def generate_input(configs: List[Any], batch_size: int) -> KeyedJaggedTensor:
    model_input, _ = ModelInput.generate(
        batch_size=batch_size,
        world_size=WORLD_SIZE,
        num_float_features=0,
        tables=configs,
        weighted_tables=[],
        pooling_avg=5,
        random_seed=100,
    )
    kjt = model_input.idlist_features
    assert isinstance(kjt, KeyedJaggedTensor)
    return kjt


def run_accuracy(collection: str) -> None:
    configs = make_configs(collection, 32)
    cpu_collection, tpu_kernel = make_modules(collection, configs)
    with torch.no_grad():
        copy_weights(collection, cpu_collection, tpu_kernel, configs)
        for batch_size in BATCH_SIZES:
            kjt = generate_input(configs, batch_size)
            expected = reference_output(collection, cpu_collection, kjt)
            tpu_result = tpu_output(tpu_kernel, to_tpu(kjt))
            sync.synchronize(tpu_result, wait=True)
            actual = tpu_result.to("cpu")
            assert expected.shape == actual.shape, f"{expected.shape} != {actual.shape}"
            error = (expected - actual).abs()
            assert torch.allclose(expected, actual, atol=1e-5)
            print(
                f"accuracy collection={collection} batch_size={batch_size}: "
                f"shape={tuple(actual.shape)}, max_abs_diff={error.max().item():.3e}"
            )


def run_profile(collection: str, traces_dir: Optional[str]) -> None:
    if traces_dir is None:
        print("\nTRACE_DIR unset, skipping the profiled step")
        return

    configs = make_configs(collection, 32)
    _, tpu_kernel = make_modules(collection, configs)
    print(f"Profiler start, traces written to {traces_dir}")
    tpu_result: torch.Tensor | None = None
    with torch.no_grad(), profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.TPU,
        ],
        on_trace_ready=profiler.xprof_trace_handler(dir_name=traces_dir),
    ):
        for _ in range(PROFILE_NUMBER_TIMES):
            kjt = generate_input(configs, 2048)
            tpu_result = tpu_output(tpu_kernel, to_tpu(kjt))
            sync.synchronize(tpu_result, wait=True)

    assert tpu_result is not None
    print("Profiled TPU output shape:", tuple(tpu_result.shape))


def run_benchmark(collection: str, lookup_mode: str, embedding_dims: List[int]) -> None:
    print(
        f"\n========== Benchmark: collection={collection}, kernel={lookup_mode}, "
        f"iters={BENCHMARK_TIMES} =========="
    )
    results = []
    for dim in embedding_dims:
        configs = make_configs(collection, dim)
        cpu_collection, tpu_kernel = make_modules(collection, configs)
        kjt = generate_input(configs, BENCH_BATCH_SIZE)
        kjt_tpu = to_tpu(kjt)

        with torch.no_grad():
            copy_weights(collection, cpu_collection, tpu_kernel, configs)
            _ = reference_output(collection, cpu_collection, kjt)
            tpu_result = tpu_output(tpu_kernel, kjt_tpu)
            sync.synchronize(tpu_result, wait=True)

            tpu_times = []
            cpu_times = []
            for _ in range(BENCHMARK_TIMES):
                start = time.perf_counter()
                _ = reference_output(collection, cpu_collection, kjt)
                cpu_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                tpu_result = tpu_output(tpu_kernel, kjt_tpu)
                sync.synchronize(tpu_result, wait=True)
                tpu_times.append(time.perf_counter() - start)

        tpu_ms = sum(tpu_times) / len(tpu_times) * 1000
        cpu_ms = sum(cpu_times) / len(cpu_times) * 1000
        results.append((dim, tpu_ms, cpu_ms))
        print(
            f"  dim={dim}: TPU({lookup_mode})={tpu_ms:.2f} ms, "
            f"CPU={cpu_ms:.2f} ms, speedup={cpu_ms / tpu_ms:.2f}x"
        )

    print(
        f"\n=== Summary: collection={collection}, kernel={lookup_mode}, "
        f"iters={BENCHMARK_TIMES} ==="
    )
    print(f"{'dim':>6} {'TPU (ms)':>12} {'CPU (ms)':>12} {'speedup':>10}")
    for dim, tpu_ms, cpu_ms in results:
        print(f"{dim:>6} {tpu_ms:>12.2f} {cpu_ms:>12.2f} {cpu_ms / tpu_ms:>9.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookup-mode",
        default=os.environ.get("LOOKUP_MODE", "v1_sc"),
        choices=["v1_tpu", "v1_sc"],
        help="Embedding lookup kernel to use (default from $LOOKUP_MODE or v1_sc).",
    )
    parser.add_argument(
        "--collection",
        choices=["ec", "ebc"],
        default="ec",
        help="Benchmark EmbeddingCollection or EmbeddingBagCollection.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        choices=EMBEDDING_DIMS,
        help="Run only one embedding dimension from the standard sweep.",
    )
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    lookup_mode = args.lookup_mode
    collection = args.collection

    os.environ["LOOKUP_MODE"] = lookup_mode
    from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401

    embedding_dims = (
        [args.embedding_dim] if args.embedding_dim is not None else EMBEDDING_DIMS
    )
    print(f"LOOKUP_MODE={lookup_mode}, collection={collection}")
    if not args.skip_accuracy:
        run_accuracy(collection)
    if not args.skip_profile:
        run_profile(collection, os.environ.get("TRACE_DIR"))
    run_benchmark(collection, lookup_mode, embedding_dims)


if __name__ == "__main__":
    main()
