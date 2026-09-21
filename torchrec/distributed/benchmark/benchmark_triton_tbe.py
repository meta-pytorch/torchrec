#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Trace-derived Triton TBE benchmarks with an FBGEMM CUDA baseline.

The request shapes come from Kineto metadata for production column-wise embedding
lookups. Where exact table cardinalities are unavailable or aggregate metadata would
produce cache-resident tables, the profile uses fixed, randomly sampled table sizes
with realistic total weight footprints.

Example:

    buck2 run @fbcode//mode/opt \
        fbcode//torchrec/distributed/benchmark:benchmark_triton_tbe -- \
        triton_tbe_forward --workload=cmf_vle_fb_ad_vpv_offline

Run the matching FBGEMM baseline by replacing ``triton_tbe_forward`` with
``fbgemm_tbe_forward``.

Run the managed-memory comparison with ``triton_uvm_tbe_forward`` and
``fbgemm_uvm_tbe_forward``.

Run bounded managed-memory launches with ``triton_uvm_capped_tbe_forward``.

Run the managed-memory cache comparison with
``triton_uvm_caching_tbe_forward`` and ``fbgemm_uvm_caching_tbe_forward``.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm

try:
    from fbgemm_gpu.tbe.config.embedding_config import (
        BoundsCheckMode,
        EmbeddingLocation,
        PoolingMode,
    )
except ImportError:
    from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
        BoundsCheckMode,
        EmbeddingLocation,
        PoolingMode,
    )

from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    cmd_conf,
)
from torchrec.distributed.triton_tbe.triton_table_batched_embeddings import (
    TritonTableBatchedEmbeddingBags,
)
from torchrec.distributed.triton_tbe.triton_uvm_caching_table_batched_embeddings import (
    TritonUVMCachingTableBatchedEmbeddingBags,
)
from torchrec.distributed.triton_tbe.triton_uvm_table_batched_embeddings import (
    TritonUVMCappedTableBatchedEmbeddingBags,
    TritonUVMTableBatchedEmbeddingBags,
)

logger: logging.Logger = logging.getLogger(__name__)

# pyrefly: ignore[missing-argument]
_cc = cmd_conf()

_LENGTH_BUILD_CHUNK = 1 << 20


@dataclass(frozen=True)
class TraceShape:
    feature_batch_sizes: tuple[int, ...]
    num_indices: int
    max_batch_per_rank: int | None = None

    @property
    def num_bags(self) -> int:
        return sum(self.feature_batch_sizes)


@dataclass(frozen=True)
class TraceWorkload:
    table_rows: tuple[int, ...]
    embedding_dims: tuple[int, ...]
    shapes: tuple[TraceShape, ...]
    num_ranks: int | None = None


@dataclass(frozen=True)
class TraceRequest:
    indices: torch.Tensor
    offsets: torch.Tensor
    batch_size_per_feature_per_rank: list[list[int]] | None


@dataclass
class TraceTBEForwardConfig(BenchFuncConfig):
    name: str = ""
    world_size: int = 1
    device_type: str = "cuda"
    profile_dir: str = "."
    num_benchmarks: int = 100
    num_profiles: int = 10
    workload: str = "cmf_vle_fb_ad_vpv_offline"
    shape_index: int = 0
    all_shapes: bool = False
    seed: int = 42
    uvm_host_mapped: bool = False
    cache_algorithm: str = "lru"
    forward_block_limit: int = 0
    vbe_forward_block_limit: int = 0
    uvm_cache_bytes: int = 0


@dataclass(frozen=True)
class TritonTBEKernelSpec:
    module_class: type[TritonTableBatchedEmbeddingBags]
    uses_managed_memory: bool
    supports_block_limits: bool = False
    supports_cache: bool = False


_TRITON_TBE_KERNELS: dict[str, TritonTBEKernelSpec] = {
    "triton": TritonTBEKernelSpec(TritonTableBatchedEmbeddingBags, False),
    "triton_uvm": TritonTBEKernelSpec(TritonUVMTableBatchedEmbeddingBags, True),
    "triton_uvm_capped": TritonTBEKernelSpec(
        TritonUVMCappedTableBatchedEmbeddingBags,
        True,
        supports_block_limits=True,
    ),
    "triton_uvm_caching": TritonTBEKernelSpec(
        TritonUVMCachingTableBatchedEmbeddingBags,
        True,
        supports_cache=True,
    ),
}


_TRACE_WORKLOADS: dict[str, TraceWorkload] = {
    "cmf_vle_fb_ad_vpv_offline": TraceWorkload(
        # Randomly sampled with seed 42 and fixed for reproducibility. The FP16
        # weights total exactly 10 GB.
        table_rows=(395_427_677, 229_572_323),
        embedding_dims=(8, 8),
        shapes=(
            TraceShape((24_332_144, 4_080), 24_333_157, 240_288),
            TraceShape((24_995_184, 4_064), 24_996_120, 241_376),
            TraceShape((23_793_584, 4_016), 23_794_477, 236_064),
            TraceShape((24_617_648, 4_064), 24_618_572, 232_672),
            TraceShape((24_588_368, 4_048), 24_589_253, 237_968),
        ),
        num_ranks=128,
    ),
    "cmf_vle_user_conv_ads_event_long_retention": TraceWorkload(
        # Randomly sampled with seed 43 and fixed for reproducibility. The FP16
        # weights total exactly 10 GB.
        table_rows=(75_327_265, 414_756_718, 134_916_017),
        embedding_dims=(8, 8, 8),
        shapes=(
            TraceShape((13_566_048, 1_963_760, 1_963_760), 32_955_824, 134_960),
            TraceShape((13_960_048, 2_019_344, 2_019_344), 33_901_749, 136_016),
            TraceShape((13_282_928, 1_926_672, 1_926_672), 32_307_556, 134_912),
            TraceShape((13_692_848, 1_981_664, 1_981_664), 33_261_650, 130_016),
            TraceShape((13_654_672, 1_987_096, 1_987_096), 33_277_289, 130_784),
        ),
        num_ranks=128,
    ),
    "cmf_ebc": TraceWorkload(
        # Randomly sampled with seed 42 and fixed here for reproducible comparisons.
        # The skewed tables retain the trace's table count and sum/max dimensions;
        # their FP16 weights total 9.999998928 GB.
        table_rows=(
            2_144_222,
            9_692_111,
            7_153_649,
            3_780_603,
            3_745_277,
            5_284_712,
            5_941_361,
            2_688_465,
            4_102_227,
            1_437_625,
            1_859_649,
            26_408_384,
            2_152_945,
            2_372_867,
            8_740_201,
        ),
        embedding_dims=(64, 40, 64, 72, 40, 40, 72, 88, 56, 80, 80, 48, 112, 96, 56),
        shapes=(
            TraceShape((196_608,) * 15, 85_606_593),
            TraceShape((196_608,) * 15, 85_398_310),
            TraceShape((196_608,) * 15, 85_297_569),
            TraceShape((196_608,) * 15, 85_583_487),
            TraceShape((196_608,) * 15, 85_272_046),
        ),
    ),
    "cmf_vle_public_sparse": TraceWorkload(
        table_rows=(12_500_000,),
        embedding_dims=(8,),
        shapes=(
            TraceShape((12_044_043,), 12_043_905, 143_165),
            TraceShape((12_098_277,), 12_098_114, 147_239),
            TraceShape((11_909_628,), 11_909_472, 139_738),
            TraceShape((11_967_270,), 11_967_126, 141_906),
            TraceShape((12_029_680,), 12_029_544, 147_665),
        ),
        num_ranks=96,
    ),
}


def _partition_evenly(total: int, parts: int) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + int(part < remainder) for part in range(parts)]


def _partition_with_observed_max(
    total: int,
    parts: int,
    observed_max: int,
) -> list[int]:
    if observed_max < (total + parts - 1) // parts:
        raise ValueError(
            f"observed max {observed_max} cannot partition {total} over {parts} ranks"
        )
    return [observed_max] + _partition_evenly(total - observed_max, parts - 1)


def _make_batch_size_per_feature_per_rank(
    shape: TraceShape,
    num_ranks: int,
) -> list[list[int]]:
    if shape.max_batch_per_rank is None:
        raise ValueError("VBE shape requires max_batch_per_rank")
    result = []
    for feature, batch_size in enumerate(shape.feature_batch_sizes):
        if feature == 0:
            per_rank = _partition_with_observed_max(
                batch_size,
                num_ranks,
                shape.max_batch_per_rank,
            )
        else:
            per_rank = _partition_evenly(batch_size, num_ranks)

        # Avoid aligning every feature's largest partitions to the same rank.
        rotate = (feature * 17) % num_ranks
        result.append(per_rank[rotate:] + per_rank[:rotate])
    return result


def _allocate_indices_per_feature(shape: TraceShape) -> list[int]:
    scaled = [shape.num_indices * batch for batch in shape.feature_batch_sizes]
    counts = [value // shape.num_bags for value in scaled]
    remaining = shape.num_indices - sum(counts)
    remainders = sorted(
        range(len(counts)),
        key=lambda feature: scaled[feature] % shape.num_bags,
        reverse=True,
    )
    for feature in remainders[:remaining]:
        counts[feature] += 1
    return counts


def _fill_lengths(lengths: torch.Tensor, num_indices: int) -> None:
    num_bags = lengths.numel()
    base, extra = divmod(num_indices, num_bags)
    lengths.fill_(base)
    if extra == 0:
        return

    # Citrine C3: construct the large request tensors directly on the target GPU.
    for start in range(0, num_bags, _LENGTH_BUILD_CHUNK):
        end = min(start + _LENGTH_BUILD_CHUNK, num_bags)
        positions = torch.arange(
            start,
            end,
            dtype=torch.int64,
            device=lengths.device,
        )
        previous = torch.div(positions * extra, num_bags, rounding_mode="floor")
        current = torch.div(
            (positions + 1) * extra,
            num_bags,
            rounding_mode="floor",
        )
        lengths[start:end] += current - previous


def _make_request(
    workload: TraceWorkload,
    shape: TraceShape,
    device: torch.device,
) -> TraceRequest:
    feature_indices = _allocate_indices_per_feature(shape)
    lengths = torch.empty(shape.num_bags, dtype=torch.int64, device=device)
    indices = torch.empty(shape.num_indices, dtype=torch.int64, device=device)

    bag_cursor = 0
    index_cursor = 0
    for batch_size, num_indices, table_rows in zip(
        shape.feature_batch_sizes,
        feature_indices,
        workload.table_rows,
    ):
        _fill_lengths(lengths[bag_cursor : bag_cursor + batch_size], num_indices)
        indices[index_cursor : index_cursor + num_indices].random_(0, table_rows)
        bag_cursor += batch_size
        index_cursor += num_indices

    offsets = torch.empty(shape.num_bags + 1, dtype=torch.int64, device=device)
    offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=offsets[1:])

    return TraceRequest(
        indices=indices,
        offsets=offsets,
        batch_size_per_feature_per_rank=(
            _make_batch_size_per_feature_per_rank(shape, workload.num_ranks)
            if workload.num_ranks is not None
            else None
        ),
    )


def _make_triton_tbe(
    workload: TraceWorkload,
    device: torch.device,
    *,
    kernel_name: str,
    forward_block_limit: int = 0,
    vbe_forward_block_limit: int = 0,
    uvm_cache_bytes: int = 0,
    uvm_host_mapped: bool = False,
    cache_algorithm: str = "lru",
) -> TritonTableBatchedEmbeddingBags:
    if kernel_name not in _TRITON_TBE_KERNELS:
        raise ValueError(
            f"Triton TBE kernel must be one of {sorted(_TRITON_TBE_KERNELS)}, "
            f"got {kernel_name}"
        )
    kernel = _TRITON_TBE_KERNELS[kernel_name]
    module_kwargs: dict[str, Any] = {}
    if kernel.uses_managed_memory:
        module_kwargs["uvm_host_mapped"] = uvm_host_mapped
    if kernel.supports_block_limits:
        module_kwargs.update(
            {
                "forward_block_limit": forward_block_limit,
                "vbe_forward_block_limit": vbe_forward_block_limit,
            }
        )
    elif kernel.supports_cache:
        cache_assoc = torch.cuda.get_device_properties(device).warp_size
        bytes_per_cache_set = (
            cache_assoc * max(workload.embedding_dims) * torch.float16.itemsize
        )
        cache_sets = uvm_cache_bytes // bytes_per_cache_set
        if cache_sets <= 0:
            raise ValueError(
                f"uvm_cache_bytes must hold at least {cache_assoc} cache rows"
            )
        module_kwargs.update(
            {
                "cache_sets": cache_sets,
                "cache_algorithm": cache_algorithm,
            }
        )
    module = kernel.module_class(
        embedding_specs=list(zip(workload.table_rows, workload.embedding_dims)),
        feature_table_map=list(range(len(workload.table_rows))),
        weights_precision=torch.float16,
        output_dtype=torch.float32,
        stochastic_rounding=False,
        learning_rate=0.01,
        eps=0.1,
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        device=device,
        fused_bounds_check=False,
        **module_kwargs,
    )
    with torch.no_grad():
        weight = (
            torch.ops.fbgemm.uvm_to_cpu(module.weight)
            if kernel.uses_managed_memory
            else module.weight
        )
        weight.uniform_(-0.01, 0.01)
    return module


def _make_fbgemm_tbe(
    workload: TraceWorkload,
    device: torch.device,
    *,
    uvm: bool = False,
    uvm_cache_bytes: int = 0,
    uvm_host_mapped: bool = False,
    cache_algorithm: str = "lru",
) -> SplitTableBatchedEmbeddingBagsCodegen:
    use_uvm_cache = uvm_cache_bytes > 0
    cache_kwargs: dict[str, Any] = {}
    if use_uvm_cache:
        try:
            cache_algorithm_value = CacheAlgorithm[cache_algorithm.upper()]
        except KeyError as error:
            raise ValueError("cache_algorithm must be LRU or LFU") from error
        cache_assoc = 32
        bytes_per_cache_set = (
            cache_assoc * max(workload.embedding_dims) * SparseType.FP16.bit_rate()
        ) // 8
        cache_sets = uvm_cache_bytes // bytes_per_cache_set
        if cache_sets <= 0:
            raise ValueError(
                f"uvm_cache_bytes must hold at least {cache_assoc} cache rows"
            )
        cache_kwargs = {
            "cache_algorithm": cache_algorithm_value,
            "cache_precision": SparseType.FP16,
            "cache_sets": cache_sets,
        }
    module = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                rows,
                dim,
                (
                    EmbeddingLocation.MANAGED_CACHING
                    if use_uvm_cache
                    else EmbeddingLocation.MANAGED if uvm else EmbeddingLocation.DEVICE
                ),
                ComputeDevice.CUDA,
            )
            for rows, dim in zip(workload.table_rows, workload.embedding_dims)
        ],
        feature_table_map=list(range(len(workload.table_rows))),
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.01,
        eps=0.1,
        weights_precision=SparseType.FP16,
        output_dtype=SparseType.FP32,
        stochastic_rounding=False,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.V2_WARNING,
        uvm_host_mapped=uvm_host_mapped,
        **cache_kwargs,
    ).to(device)
    module.init_embedding_weights_uniform(-0.01, 0.01)
    return module


def _run_forward(
    _batch_inputs: list[dict[str, Any]],
    module: torch.nn.Module,
    request: TraceRequest,
) -> None:
    module(
        request.indices,
        request.offsets,
        batch_size_per_feature_per_rank=(request.batch_size_per_feature_per_rank),
    )


def _run_backend(
    kernel_name: str,
    workload_name: str,
    workload: TraceWorkload,
    shape_index: int,
    shape: TraceShape,
    request: TraceRequest,
    config: TraceTBEForwardConfig,
    device: torch.device,
) -> None:
    module: torch.nn.Module
    if kernel_name in _TRITON_TBE_KERNELS:
        module = _make_triton_tbe(
            workload,
            device,
            kernel_name=kernel_name,
            forward_block_limit=config.forward_block_limit,
            vbe_forward_block_limit=config.vbe_forward_block_limit,
            uvm_cache_bytes=config.uvm_cache_bytes,
            uvm_host_mapped=config.uvm_host_mapped,
            cache_algorithm=config.cache_algorithm,
        )
    elif kernel_name in {"fbgemm", "fbgemm_uvm", "fbgemm_uvm_caching"}:
        module = _make_fbgemm_tbe(
            workload,
            device,
            uvm=kernel_name != "fbgemm",
            uvm_cache_bytes=(
                config.uvm_cache_bytes if kernel_name == "fbgemm_uvm_caching" else 0
            ),
            uvm_host_mapped=config.uvm_host_mapped,
            cache_algorithm=config.cache_algorithm,
        )
    else:
        raise ValueError(f"Unsupported TBE kernel: {kernel_name}")

    _run_forward([], module, request)
    torch.cuda.synchronize(device)

    suffix = f"_{config.name}" if config.name else ""
    result = benchmark_func(
        func_to_benchmark=_run_forward,
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs={"module": module, "request": request},
        sample_count=shape.num_indices,
        **config.benchmark_func_kwargs(
            name=f"{kernel_name}_tbe_forward_{workload_name}_shape_{shape_index}{suffix}",
            rank=0,
        ),
    )
    print(result)


def _run_trace_workload(
    config: TraceTBEForwardConfig,
    kernel_name: str,
) -> None:
    config.maybe_enable_expandable_segments()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_triton_tbe requires a CUDA device")
    config.set_log_level()

    if config.workload not in _TRACE_WORKLOADS:
        raise ValueError(
            f"--workload must be one of {sorted(_TRACE_WORKLOADS)}, "
            f"got {config.workload}"
        )
    workload = _TRACE_WORKLOADS[config.workload]
    if config.shape_index < 0 or config.shape_index >= len(workload.shapes):
        raise ValueError(
            f"--shape-index must be in [0, {len(workload.shapes)}), "
            f"got {config.shape_index}"
        )

    shape_indices = (
        range(len(workload.shapes)) if config.all_shapes else (config.shape_index,)
    )
    device = torch.device(torch.cuda.current_device())
    for shape_index in shape_indices:
        shape = workload.shapes[shape_index]
        torch.manual_seed(config.seed + shape_index)
        request = _make_request(workload, shape, device)
        logger.info(
            "workload=%s shape=%d T=%d R=%s bags=%d indices=%d mean_L=%.6f",
            config.workload,
            shape_index,
            len(workload.table_rows),
            workload.num_ranks if workload.num_ranks is not None else "fixed-B",
            shape.num_bags,
            shape.num_indices,
            shape.num_indices / shape.num_bags,
        )
        torch.manual_seed(config.seed)
        _run_backend(
            kernel_name,
            config.workload,
            workload,
            shape_index,
            shape,
            request,
            config,
            device,
        )


def register_benchmark(
    config: type[TraceTBEForwardConfig],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        func.__annotations__ = {"config": config, "return": None}
        # pyrefly: ignore[missing-attribute]
        _cc.register(func)
        return func

    return decorator


@dataclass
class TritonTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark TorchRec's Triton TBE on a trace-derived production shape."""


@register_benchmark(TritonTBEForwardConfig)
def triton_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "triton")


@dataclass
class FbgemmTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark the FBGEMM CUDA TBE baseline on the identical request."""


@register_benchmark(FbgemmTBEForwardConfig)
def fbgemm_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "fbgemm")


@dataclass
class TritonUVMTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark Triton TBE with managed-memory weights."""


@register_benchmark(TritonUVMTBEForwardConfig)
def triton_uvm_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "triton_uvm")


@dataclass
class TritonUVMCappedTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark Triton UVM TBE with bounded forward launches."""

    forward_block_limit: int = 48
    vbe_forward_block_limit: int = 48


@register_benchmark(TritonUVMCappedTBEForwardConfig)
def triton_uvm_capped_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "triton_uvm_capped")


@dataclass
class TritonUVMCachingTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark Triton UVM TBE with a native HBM row cache."""

    uvm_cache_bytes: int = 2 * 1024**3


@register_benchmark(TritonUVMCachingTBEForwardConfig)
def triton_uvm_caching_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "triton_uvm_caching")


@dataclass
class FbgemmUVMCachingTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark FBGEMM UVM TBE with an equivalently sized HBM cache."""

    uvm_cache_bytes: int = 2 * 1024**3


@register_benchmark(FbgemmUVMCachingTBEForwardConfig)
def fbgemm_uvm_caching_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "fbgemm_uvm_caching")


@dataclass
class FbgemmUVMTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark FBGEMM TBE with managed-memory weights."""


@register_benchmark(FbgemmUVMTBEForwardConfig)
def fbgemm_uvm_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_workload(config, "fbgemm_uvm")


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
