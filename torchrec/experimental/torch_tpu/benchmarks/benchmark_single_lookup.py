#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Accuracy + latency benchmark for the single-lookup TPU Pallas embedding kernel.
"""

import argparse
import os
import time
from typing import List

import torch

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
from torch_tpu._internal import profiler, sync
from torchrec.distributed.batched_embedding_kernel import BatchedTPUEmbedding
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE = 8
PROFILE_NUMBER_TIMES = 10
BENCHMARK_TIMES = 100

embedding_configs: List[EmbeddingConfig] = [
    EmbeddingConfig(
        name="user_table",
        embedding_dim=32,
        num_embeddings=100000,
        feature_names=["user"],
    ),
    EmbeddingConfig(
        name="product_table",
        embedding_dim=32,
        num_embeddings=1000000,
        feature_names=["product"],
    ),
]


def make_configs(dim: int) -> List[EmbeddingConfig]:
    # Mirrors the module-level `embedding_configs`, varying only embedding_dim.
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


def grouped_config(configs: List[EmbeddingConfig]) -> GroupedEmbeddingConfig:
    """One `UNFUSED_TPU` group holding every table, as the sharded path builds it."""
    return GroupedEmbeddingConfig(
        data_type=DataType.FP32,
        pooling=PoolingType.NONE,
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
                pooling=PoolingType.NONE,
                is_weighted=False,
                has_feature_processor=False,
                compute_kernel=EmbeddingComputeKernel.UNFUSED_TPU,
                local_rows=config.num_embeddings,
                local_cols=config.embedding_dim,
            )
            for config in configs
        ],
    )


def reference_output(
    ref_ec: EmbeddingCollection, kjt: KeyedJaggedTensor
) -> torch.Tensor:
    """`EmbeddingCollection` output flattened to match `BatchedTPUEmbedding`."""
    ref_jt = ref_ec(kjt)
    return torch.cat([ref_jt[key].values() for key in kjt.keys()], dim=0)


def main() -> None:
    # CLI flag overrides the env var, which overrides the v1_sc default.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookup-mode",
        default=os.environ.get("LOOKUP_MODE", "v1_sc"),
        choices=["v1_tpu", "v1_sc"],
        help="Embedding lookup kernel to use (default from $LOOKUP_MODE or v1_sc).",
    )
    args = parser.parse_args()
    lookup_mode = args.lookup_mode

    # The flag must be in the env before either module is imported.
    os.environ["LOOKUP_MODE"] = lookup_mode
    from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401

    print(f"LOOKUP_MODE = {lookup_mode}")

    traces_dir = os.environ.get("TRACE_DIR")

    # --- Reference: TorchRec EmbeddingCollection (uses nn.Embedding on CPU) ---
    ref_module = EmbeddingCollection(
        tables=embedding_configs, device=torch.device("cpu")
    )

    # --- TPU: the UNFUSED_TPU compute kernel, i.e. what the sharded path runs ---
    tpu_kernel = BatchedTPUEmbedding(
        config=grouped_config(embedding_configs), device=torch.device("tpu")
    )

    # Copy weights from reference into the TPU tables so outputs are comparable.
    # split_embedding_weights() returns one weight per table, in table order.
    for weight, config in zip(tpu_kernel.split_embedding_weights(), embedding_configs):
        # pyre-ignore[6]: EmbeddingCollection.embeddings is a ModuleDict, so indexing it types as Module.
        weight.data.copy_(ref_module.embeddings[config.name].weight.data)

    # --- Accuracy test: compare outputs across different batch sizes ---
    BATCH_SIZES = [10, 40, 160, 640, 1024, 2048, 1024 * 32]

    for batch_size in BATCH_SIZES:
        print("Create Model Input")
        model_input, _ = ModelInput.generate(
            batch_size=batch_size,
            world_size=WORLD_SIZE,
            num_float_features=0,
            tables=embedding_configs,
            weighted_tables=[],
            pooling_avg=5,
            random_seed=100,
        )
        # CPU embedding needs int64 on torch 2.13
        kjt = model_input.idlist_features
        assert isinstance(kjt, KeyedJaggedTensor)
        # Convert to int32/TPU outside the forward, before ref_module caches a CPU _jt_dict on kjt.
        kjt_tpu = KeyedJaggedTensor(
            keys=kjt.keys(),
            values=kjt.values().to(device="tpu", dtype=torch.int32),
            lengths=kjt.lengths(),
        )
        print("Lookup on CPU")
        ref_vals = reference_output(ref_module, kjt)
        print("Lookup on TPU")
        tpu_vals = tpu_kernel(kjt_tpu).to("cpu")

        print(f"\n=== batch_size={batch_size} ===")
        print(f"  ref shape={ref_vals.shape}, tpu shape={tpu_vals.shape}")
        assert ref_vals.shape == tpu_vals.shape, f"{ref_vals.shape} != {tpu_vals.shape}"
        err = (ref_vals - tpu_vals).abs()
        if err.max() > 1e-5:
            print("CPU", ref_vals)
            print("TPU", tpu_vals)
        assert torch.allclose(ref_vals, tpu_vals, atol=1e-5)
        print(f"    match=True, max abs diff={err.max().item():.3e}")
        sync.synchronize()

    # --- Profiler: single step with torch_tpu xprof trace ---
    if traces_dir is None:
        print("\nTRACE_DIR unset, skipping the profiled step")
    else:
        print(f"Profiler Start, traces printed to {traces_dir}")
        tpu_output: torch.Tensor | None = None
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.TPU,
            ],
            on_trace_ready=profiler.xprof_trace_handler(dir_name=traces_dir),
        ):
            for _ in range(PROFILE_NUMBER_TIMES):
                model_input, _ = ModelInput.generate(
                    batch_size=2048,
                    world_size=WORLD_SIZE,
                    num_float_features=0,
                    tables=embedding_configs,
                    weighted_tables=[],
                    pooling_avg=5,
                    random_seed=100,
                )
                # CPU embedding needs int64
                kjt = model_input.idlist_features
                assert isinstance(kjt, KeyedJaggedTensor)

                # Convert to int32/TPU outside the forward.
                kjt_tpu = KeyedJaggedTensor(
                    keys=kjt.keys(),
                    values=kjt.values().to(device="tpu", dtype=torch.int32),
                    lengths=kjt.lengths(),  # stays on CPU so to_dict() splits on CPU
                )
                tpu_output = tpu_kernel(kjt_tpu)

        assert (
            tpu_output is not None
        ), "profiled step never ran; PROFILE_NUMBER_TIMES must be > 0"
        print("Profiled TPU output shape:", tuple(tpu_output.shape))

    # --- Benchmark here -------
    EMBEDDING_DIMS = [32, 64, 128, 256, 512]
    BENCH_BATCH_SIZE = 2048

    print(
        f"\n========== Benchmark sweep: kernel={lookup_mode}, iters={BENCHMARK_TIMES} =========="
    )
    results = []
    for dim in EMBEDDING_DIMS:
        configs = make_configs(dim)
        # Both sides are the same compute kernel, so the only variable is the
        # backend the lookup op dispatches to -- Pallas on TPU, the reference
        # gather on CPU. Timing EmbeddingCollection on CPU instead would charge
        # the CPU column for to_dict()/JaggedTensor work the TPU column skips.
        tpu_kernel = BatchedTPUEmbedding(
            config=grouped_config(configs), device=torch.device("tpu")
        )
        cpu_kernel = BatchedTPUEmbedding(
            config=grouped_config(configs), device=torch.device("cpu")
        )

        # warm-up both backends
        model_input, _ = ModelInput.generate(
            batch_size=BENCH_BATCH_SIZE,
            world_size=WORLD_SIZE,
            num_float_features=0,
            tables=configs,
            weighted_tables=[],
            pooling_avg=5,
            random_seed=100,
        )
        # CPU embedding needs int64
        kjt = model_input.idlist_features
        assert isinstance(kjt, KeyedJaggedTensor)
        # Convert to int32/TPU
        kjt_tpu = KeyedJaggedTensor(
            keys=kjt.keys(),
            values=kjt.values().to(device="tpu", dtype=torch.int32),
            lengths=kjt.lengths(),
        )
        _ = tpu_kernel(kjt_tpu)
        _ = cpu_kernel(kjt)
        sync.synchronize()

        tpu_times = []
        cpu_times = []
        for _ in range(BENCHMARK_TIMES):
            model_input, _ = ModelInput.generate(
                batch_size=BENCH_BATCH_SIZE,
                world_size=WORLD_SIZE,
                num_float_features=0,
                tables=configs,
                weighted_tables=[],
                pooling_avg=5,
                random_seed=100,
            )
            # CPU embedding needs int64
            kjt = model_input.idlist_features
            assert isinstance(kjt, KeyedJaggedTensor)
            # Convert to int32/TPU
            kjt_tpu = KeyedJaggedTensor(
                keys=kjt.keys(),
                values=kjt.values().to(device="tpu", dtype=torch.int32),
                lengths=kjt.lengths(),
            )

            # CPU reference gather
            start = time.perf_counter()
            _ = cpu_kernel(kjt)
            cpu_times.append(time.perf_counter() - start)

            # TPU Pallas kernel
            sync.synchronize()
            start = time.perf_counter()
            _ = tpu_kernel(kjt_tpu)
            sync.synchronize()
            tpu_times.append(time.perf_counter() - start)

        tpu_ms = sum(tpu_times) / len(tpu_times) * 1000
        cpu_ms = sum(cpu_times) / len(cpu_times) * 1000
        results.append((dim, tpu_ms, cpu_ms))
        print(
            f"  dim={dim}: TPU({lookup_mode})={tpu_ms:.2f} ms, "
            f"CPU={cpu_ms:.2f} ms, speedup={cpu_ms / tpu_ms:.2f}x"
        )

    print(f"\n=== Summary: kernel={lookup_mode}, iters={BENCHMARK_TIMES} ===")
    print(f"{'dim':>6} {'TPU (ms)':>12} {'CPU (ms)':>12} {'speedup':>10}")
    for dim, tpu_ms, cpu_ms in results:
        print(f"{dim:>6} {tpu_ms:>12.2f} {cpu_ms:>12.2f} {cpu_ms / tpu_ms:>9.2f}x")


if __name__ == "__main__":
    main()
