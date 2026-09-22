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

Example comparing both backends on a custom training workload:

    buck2 run @fbcode//mode/opt \
        fbcode//torchrec/distributed/benchmark:benchmark_triton_tbe -- \
        tbe_comparison --num_tables=2 --num_embeddings=1000000,2000000 \
        --embedding_dim=128,256 --batch_size=1024 --bag_size=20

Use ``triton_tbe`` or ``fbgemm_tbe`` to benchmark one backend. The legacy
forward-only entry points also support managed-memory, bounded launches, and
managed-memory cache comparisons.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator

import torch
import yaml
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.cache.cache_config import CacheAlgorithm
from fbgemm_gpu.tbe.utils import generate_requests_for_grouped_tables, TBERequest

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
from torchrec.distributed.triton_tbe import (
    triton_table_batched_embeddings as triton_tbe_module,
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
class TBEModuleSpec:
    table_rows: list[int]
    embedding_dims: list[int]
    feature_table_map: list[int]


@dataclass(frozen=True)
class TBEComparisonWorkload(TBEModuleSpec):
    batch_sizes: list[int]
    bag_sizes: list[int]
    variable_batch: bool


@dataclass
class TBEComparisonConfig(BenchFuncConfig):
    """Compare FBGEMM and Triton TBE on a configurable training workload."""

    name: str = ""
    world_size: int = 1
    device_type: str = "cuda"
    profile_dir: str = "."
    num_benchmarks: int = 10
    num_profiles: int = 10
    num_tables: int = 1
    num_embeddings: str = "1000000"
    embedding_dim: str = "128"
    feature_table_map: str = ""
    batch_size: str = "1024"
    bag_size: str = "20"
    alpha: float = 1.0
    weighted: bool = False
    vbe: bool = False
    weights_precision: str = "fp16"
    output_dtype: str = "fp32"
    optimizer: str = "exact_row_wise_adagrad"
    fused_bounds_check: bool = False
    enable_triton_tbe_optimizations: bool = field(
        default=False,
        metadata={
            "cmd_conf_aliases": ("--enable-triton-tbe-optimizations",),
        },
    )
    use_clc: bool = False
    check_correctness: bool = True
    num_warmups: int = 3
    deterministic: bool = True
    seed: int = 42
    workload_catalog: str = ""
    workload: str = ""
    list_workloads: bool = False
    shape_index: int = 0
    all_shapes: bool = False
    uvm_host_mapped: bool = False
    cache_algorithm: str = "lru"
    forward_block_limit: int = 0
    vbe_forward_block_limit: int = 0
    uvm_cache_bytes: int = 0
    run_backward: bool = True


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


def load_workload_catalog(
    path: str,
) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    with open(path) as yaml_file:
        catalog = yaml.safe_load(yaml_file)
    if not isinstance(catalog, dict):
        raise ValueError(f"workload catalog {path} must contain a mapping")
    workload_type = catalog.get("workload_type")
    if workload_type not in ("generated", "trace"):
        raise ValueError(
            f"workload catalog {path} workload_type must be generated or trace"
        )
    defaults = catalog.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"workload catalog {path} defaults must be a mapping")
    workloads = catalog.get("workloads")
    if not isinstance(workloads, dict):
        raise ValueError(f"workload catalog {path} must contain a workloads mapping")
    return workload_type, defaults, workloads


def _parse_trace_workload(raw_workload: dict[str, Any]) -> TraceWorkload:
    return TraceWorkload(
        table_rows=tuple(raw_workload["table_rows"]),
        embedding_dims=tuple(raw_workload["embedding_dims"]),
        shapes=tuple(
            TraceShape(
                feature_batch_sizes=tuple(shape["feature_batch_sizes"]),
                num_indices=shape["num_indices"],
                max_batch_per_rank=shape.get("max_batch_per_rank"),
            )
            for shape in raw_workload["shapes"]
        ),
        num_ranks=raw_workload.get("num_ranks"),
    )


def _resolve_comparison_config(
    config: TBEComparisonConfig,
) -> tuple[TBEComparisonConfig | None, TraceWorkload | None]:
    if not config.workload_catalog:
        if config.workload or config.list_workloads:
            raise ValueError(
                "--workload_catalog is required with --workload or --list_workloads"
            )
        return config, None

    workload_type, defaults, workloads = load_workload_catalog(config.workload_catalog)
    if config.list_workloads:
        print("\n".join(workloads))
        return None, None
    if not config.workload:
        raise ValueError("--workload is required with --workload_catalog")
    if config.workload not in workloads:
        raise ValueError(
            f"--workload must be one of {sorted(workloads)}, got {config.workload}"
        )
    raw_workload = workloads[config.workload]
    resolved_config = replace(config, **defaults)
    if workload_type == "trace":
        return resolved_config, _parse_trace_workload(raw_workload)
    return replace(resolved_config, **raw_workload), None


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


def _make_trace_request(
    workload: TraceWorkload,
    shape: TraceShape,
    device: torch.device,
) -> TBERequest:
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

    return TBERequest(
        indices,
        offsets,
        None,
        (
            _make_batch_size_per_feature_per_rank(shape, workload.num_ranks)
            if workload.num_ranks is not None
            else None
        ),
    )


def _parse_int_list(
    value: str,
    option: str,
    allow_zero: bool = False,
) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"--{option} must contain integers, got {value}") from error
    minimum = 0 if allow_zero else 1
    if not values or any(item < minimum for item in values):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"--{option} values must be {requirement}, got {values}")
    return values


def _expand_values(values: list[int], size: int, option: str) -> list[int]:
    if len(values) == 1:
        return values * size
    if len(values) != size:
        raise ValueError(
            f"--{option} must contain one value or {size} values, got {len(values)}"
        )
    return values


def _parse_comparison_workload(config: TBEComparisonConfig) -> TBEComparisonWorkload:
    if config.num_tables <= 0:
        raise ValueError(f"--num_tables must be positive, got {config.num_tables}")

    if config.feature_table_map:
        feature_table_map = _parse_int_list(
            config.feature_table_map,
            "feature_table_map",
            allow_zero=True,
        )
    else:
        feature_table_map = list(range(config.num_tables))

    if any(table < 0 or table >= config.num_tables for table in feature_table_map):
        raise ValueError(
            f"--feature_table_map entries must be in [0, {config.num_tables}), "
            f"got {feature_table_map}"
        )
    missing_tables = sorted(set(range(config.num_tables)) - set(feature_table_map))
    if missing_tables:
        raise ValueError(f"feature table map is missing tables {missing_tables}")

    num_features = len(feature_table_map)
    batch_sizes = _parse_int_list(config.batch_size, "batch_size")
    return TBEComparisonWorkload(
        table_rows=_expand_values(
            _parse_int_list(config.num_embeddings, "num_embeddings"),
            config.num_tables,
            "num_embeddings",
        ),
        embedding_dims=_expand_values(
            _parse_int_list(config.embedding_dim, "embedding_dim"),
            config.num_tables,
            "embedding_dim",
        ),
        feature_table_map=feature_table_map,
        batch_sizes=_expand_values(
            batch_sizes,
            num_features,
            "batch_size",
        ),
        bag_sizes=_expand_values(
            _parse_int_list(config.bag_size, "bag_size"),
            num_features,
            "bag_size",
        ),
        variable_batch=config.vbe or len(batch_sizes) > 1,
    )


def _make_triton_tbe(
    workload: TBEModuleSpec,
    config: TBEComparisonConfig,
    device: torch.device,
    *,
    kernel_name: str = "triton",
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
    weights_precision = SparseType(config.weights_precision)
    output_dtype = SparseType(config.output_dtype)
    module_kwargs: dict[str, Any] = {}
    if kernel.uses_managed_memory:
        module_kwargs["uvm_host_mapped"] = uvm_host_mapped
    if kernel.supports_block_limits:
        module_kwargs.update(
            {
                "forward_block_limit": config.forward_block_limit,
                "vbe_forward_block_limit": config.vbe_forward_block_limit,
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
        feature_table_map=workload.feature_table_map,
        weights_precision=weights_precision.as_dtype(),
        output_dtype=output_dtype.as_dtype(),
        optimizer=OptimType(config.optimizer),
        fused_bounds_check=config.fused_bounds_check,
        enable_triton_tbe_optimizations=config.enable_triton_tbe_optimizations,
        device=device,
        stochastic_rounding=False,
        learning_rate=0.01,
        eps=0.1,
        **module_kwargs,
    )
    with torch.no_grad():
        weight = (
            torch.ops.fbgemm.uvm_to_cpu(module.weight)
            if kernel.uses_managed_memory
            else module.weight
        )
        weight.uniform_(-1.0, 1.0)
    return module


def _make_fbgemm_tbe(
    workload: TBEModuleSpec,
    config: TBEComparisonConfig,
    device: torch.device,
    *,
    uvm: bool = False,
    uvm_cache_bytes: int = 0,
    uvm_host_mapped: bool = False,
    cache_algorithm: str = "lru",
) -> SplitTableBatchedEmbeddingBagsCodegen:
    weights_precision = SparseType(config.weights_precision)
    output_dtype = SparseType(config.output_dtype)
    use_uvm_cache = uvm_cache_bytes > 0
    cache_kwargs: dict[str, Any] = {}
    if use_uvm_cache:
        try:
            cache_algorithm_value = CacheAlgorithm[cache_algorithm.upper()]
        except KeyError as error:
            raise ValueError("cache_algorithm must be LRU or LFU") from error
        cache_assoc = 32
        bytes_per_cache_set = (
            cache_assoc * max(workload.embedding_dims) * weights_precision.bit_rate()
        ) // 8
        cache_sets = uvm_cache_bytes // bytes_per_cache_set
        if cache_sets <= 0:
            raise ValueError(
                f"uvm_cache_bytes must hold at least {cache_assoc} cache rows"
            )
        cache_kwargs = {
            "cache_algorithm": cache_algorithm_value,
            "cache_precision": weights_precision,
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
        feature_table_map=workload.feature_table_map,
        optimizer=OptimType(config.optimizer),
        learning_rate=0.01,
        eps=0.1,
        weights_precision=weights_precision,
        output_dtype=output_dtype,
        stochastic_rounding=False,
        pooling_mode=PoolingMode.SUM,
        bounds_check_mode=BoundsCheckMode.V2_WARNING,
        uvm_host_mapped=uvm_host_mapped,
        **cache_kwargs,
    ).to(device)
    module.init_embedding_weights_uniform(-1.0, 1.0)
    return module


def _clone_request(request: TBERequest, requires_grad: bool) -> TBERequest:
    per_sample_weights = request.per_sample_weights
    if per_sample_weights is not None:
        per_sample_weights = per_sample_weights.detach().clone()
        per_sample_weights.requires_grad_(requires_grad)
    return TBERequest(
        request.indices.long(),
        request.offsets.long(),
        per_sample_weights,
        request.Bs_per_feature_per_rank,
    )


def _comparison_forward(
    _batch_inputs: list[dict[str, Any]],
    module: torch.nn.Module,
    request: TBERequest,
) -> torch.Tensor:
    indices = request.indices
    offsets = request.offsets
    if isinstance(module, SplitTableBatchedEmbeddingBagsCodegen):
        indices = indices.long()
        offsets = offsets.long()
    return module(
        indices,
        offsets,
        request.per_sample_weights,
        batch_size_per_feature_per_rank=request.Bs_per_feature_per_rank,
    )


@dataclass
class _RequestStream:
    requests: list[TBERequest]
    index: int = 0

    def next(self) -> TBERequest:
        request = self.requests[self.index % len(self.requests)]
        self.index += 1
        return request


def _benchmark_comparison_forward(
    batch_inputs: list[dict[str, Any]],
    module: torch.nn.Module,
    request_stream: _RequestStream,
) -> torch.Tensor:
    return _comparison_forward(batch_inputs, module, request_stream.next())


@dataclass
class _BackwardState:
    output: torch.Tensor | None = None


def _prepare_comparison_backward(
    _batch_inputs: list[dict[str, Any]],
    module: torch.nn.Module,
    request_stream: _RequestStream,
    grad_output: torch.Tensor,
    backward_state: _BackwardState,
) -> None:
    request = request_stream.next()
    if request.per_sample_weights is not None:
        request.per_sample_weights.grad = None
    backward_state.output = _comparison_forward([], module, request)


def _comparison_backward(
    _batch_inputs: list[dict[str, Any]],
    module: torch.nn.Module,
    request_stream: _RequestStream,
    grad_output: torch.Tensor,
    backward_state: _BackwardState,
) -> None:
    if backward_state.output is None:
        raise RuntimeError("backward iteration was not prepared")
    backward_state.output.backward(grad_output)
    backward_state.output = None


@contextmanager
def _deterministic_algorithms(enabled: bool) -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=previous_warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _benchmark_comparison_operation(
    config: TBEComparisonConfig,
    name: str,
    operation: Callable[..., Any],
    module: torch.nn.Module,
    requests: list[TBERequest],
    sample_count: int,
    grad_output: torch.Tensor | None = None,
    backward_state: _BackwardState | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "module": module,
        "request_stream": _RequestStream(requests),
    }
    if grad_output is not None:
        kwargs["grad_output"] = grad_output
    if backward_state is not None:
        kwargs["backward_state"] = backward_state
    warmup_kwargs: dict[str, Any] = kwargs | {
        "request_stream": _RequestStream([requests[0]])
    }
    for _ in range(config.num_warmups):
        if backward_state is not None:
            _prepare_comparison_backward([], **warmup_kwargs)
        operation([], **warmup_kwargs)
    torch.cuda.synchronize()
    result = benchmark_func(
        func_to_benchmark=operation,
        iteration_setup_func=(
            _prepare_comparison_backward if backward_state is not None else None
        ),
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs=kwargs,
        sample_count=sample_count,
        **config.benchmark_func_kwargs(name=name, rank=0),
    )
    print(result)


def _estimate_comparison_traffic(
    workload: TBEComparisonWorkload,
    config: TBEComparisonConfig,
) -> tuple[int, int]:
    bytes_per_weight = SparseType(config.weights_precision).bit_rate() / 8
    forward_bytes = int(
        bytes_per_weight
        * sum(
            batch_size * bag_size * workload.embedding_dims[table]
            for batch_size, bag_size, table in zip(
                workload.batch_sizes,
                workload.bag_sizes,
                workload.feature_table_map,
            )
        )
    )
    return forward_bytes, 2 * forward_bytes


def _make_benchmark_modules(
    config: TBEComparisonConfig,
    workload: TBEModuleSpec,
    backend: str,
) -> dict[str, torch.nn.Module]:
    device = torch.device(torch.cuda.current_device())
    modules: dict[str, torch.nn.Module] = {}
    if backend == "comparison":
        modules["fbgemm"] = _make_fbgemm_tbe(workload, config, device)
        modules["triton"] = _make_triton_tbe(workload, config, device)
    elif backend in _TRITON_TBE_KERNELS:
        modules[backend] = _make_triton_tbe(
            workload,
            config,
            device,
            kernel_name=backend,
            uvm_cache_bytes=config.uvm_cache_bytes,
            uvm_host_mapped=config.uvm_host_mapped,
            cache_algorithm=config.cache_algorithm,
        )
    elif backend in ("fbgemm", "fbgemm_uvm", "fbgemm_uvm_caching"):
        modules[backend] = _make_fbgemm_tbe(
            workload,
            config,
            device,
            uvm=backend != "fbgemm",
            uvm_cache_bytes=(
                config.uvm_cache_bytes if backend == "fbgemm_uvm_caching" else 0
            ),
            uvm_host_mapped=config.uvm_host_mapped,
            cache_algorithm=config.cache_algorithm,
        )
    if not modules:
        raise ValueError(f"unsupported backend {backend}")

    if backend == "comparison":
        fbgemm_module = modules["fbgemm"]
        triton_module = modules["triton"]
        assert isinstance(fbgemm_module, SplitTableBatchedEmbeddingBagsCodegen)
        assert isinstance(triton_module, TritonTableBatchedEmbeddingBags)
        fbgemm_weight = fbgemm_module.weights_dev
        assert isinstance(fbgemm_weight, torch.Tensor)
        with torch.no_grad():
            triton_module.weight.copy_(fbgemm_weight)
    return modules


def _make_grad_output(
    config: TBEComparisonConfig,
    request: TBERequest,
    output: torch.Tensor,
) -> torch.Tensor | None:
    if not config.run_backward:
        return None
    grad_output = torch.randn_like(output)
    if request.Bs_per_feature_per_rank is None:
        grad_output = grad_output.clamp(min=-10.0, max=10.0).to(
            SparseType(config.weights_precision).as_dtype()
        )
    return grad_output


def _run_benchmark_case(
    config: TBEComparisonConfig,
    workload: TBEModuleSpec,
    requests: list[TBERequest],
    forward_bytes: int,
    backend: str,
    case_name: str = "",
) -> None:
    modules = _make_benchmark_modules(config, workload, backend)
    request = requests[0]

    outputs = {
        name: _comparison_forward([], module, request)
        for name, module in modules.items()
    }
    if config.check_correctness and backend == "comparison":
        tolerance = 1e-3 if config.weights_precision == "fp32" else 1e-2
        torch.testing.assert_close(
            outputs["triton"],
            outputs["fbgemm"],
            rtol=tolerance,
            atol=tolerance,
        )
        logger.info("FBGEMM and Triton forward outputs match")
    grad_output = _make_grad_output(config, request, next(iter(outputs.values())))
    del outputs

    print(f"Estimated forward traffic: {forward_bytes / 1.0e9:.3f} GB")
    if config.run_backward:
        print(f"Estimated backward traffic: {2 * forward_bytes / 1.0e9:.3f} GB")

    sample_count = request.indices.numel()
    suffix_parts = [part for part in (case_name, config.name) if part]
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    for name, module in modules.items():
        _benchmark_comparison_operation(
            config,
            f"{name}_tbe_forward{suffix}",
            _benchmark_comparison_forward,
            module,
            requests,
            sample_count,
        )
    if config.run_backward:
        assert grad_output is not None
        with _deterministic_algorithms(config.deterministic):
            for name, module in modules.items():
                _benchmark_comparison_operation(
                    config,
                    f"{name}_tbe_backward{suffix}",
                    _comparison_backward,
                    module,
                    [_clone_request(request, config.weighted) for request in requests],
                    sample_count,
                    grad_output,
                    _BackwardState(),
                )


def _run_generated_benchmark(config: TBEComparisonConfig, backend: str) -> None:
    config.maybe_enable_expandable_segments()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_triton_tbe requires a CUDA device")
    config.set_log_level()
    if config.num_warmups < 0:
        raise ValueError(
            f"--num_warmups must be non-negative, got {config.num_warmups}"
        )

    workload = _parse_comparison_workload(config)
    torch.manual_seed(config.seed)
    requests = generate_requests_for_grouped_tables(
        max(config.num_benchmarks, config.num_profiles, 1),
        (workload.batch_sizes if workload.variable_batch else workload.batch_sizes[0]),
        config.num_tables,
        max(workload.bag_sizes),
        max(workload.table_rows),
        Ls=workload.bag_sizes,
        feature_table_map=workload.feature_table_map,
        Es=workload.table_rows,
        alpha=config.alpha,
        weighted=config.weighted,
    )
    forward_bytes, _ = _estimate_comparison_traffic(workload, config)
    _run_benchmark_case(
        config,
        workload,
        requests,
        forward_bytes,
        backend,
        config.workload,
    )


def _run_trace_benchmark(
    config: TBEComparisonConfig,
    workload: TraceWorkload,
    backend: str,
) -> None:
    config.maybe_enable_expandable_segments()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_triton_tbe requires a CUDA device")
    config.set_log_level()
    if config.num_warmups < 0:
        raise ValueError(
            f"--num_warmups must be non-negative, got {config.num_warmups}"
        )
    if config.shape_index < 0 or config.shape_index >= len(workload.shapes):
        raise ValueError(
            f"--shape_index must be in [0, {len(workload.shapes)}), "
            f"got {config.shape_index}"
        )

    shape_indices = (
        range(len(workload.shapes)) if config.all_shapes else (config.shape_index,)
    )
    device = torch.device(torch.cuda.current_device())
    module_spec = TBEModuleSpec(
        table_rows=list(workload.table_rows),
        embedding_dims=list(workload.embedding_dims),
        feature_table_map=list(range(len(workload.table_rows))),
    )
    bytes_per_weight = SparseType(config.weights_precision).bit_rate() / 8

    for shape_index in shape_indices:
        shape = workload.shapes[shape_index]
        torch.manual_seed(config.seed + shape_index)
        request = _make_trace_request(workload, shape, device)
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
        feature_indices = _allocate_indices_per_feature(shape)
        forward_bytes = int(
            bytes_per_weight
            * sum(
                num_indices * dim
                for num_indices, dim in zip(
                    feature_indices,
                    workload.embedding_dims,
                )
            )
        )
        torch.manual_seed(config.seed)
        _run_benchmark_case(
            config,
            module_spec,
            [request],
            forward_bytes,
            backend,
            f"{config.workload}_shape_{shape_index}",
        )


def register_benchmark(
    config: type[BenchFuncConfig],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        func.__annotations__ = {"config": config, "return": None}
        # pyrefly: ignore[missing-attribute]
        _cc.register(func)
        return func

    return decorator


def _run_selected_benchmark(config: TBEComparisonConfig, backend: str) -> None:
    resolved_config, trace_workload = _resolve_comparison_config(config)
    if resolved_config is None:
        return
    previous_has_tlx = triton_tbe_module.has_tlx
    triton_tbe_module.has_tlx = resolved_config.use_clc
    try:
        if trace_workload is None:
            _run_generated_benchmark(resolved_config, backend)
        else:
            _run_trace_benchmark(resolved_config, trace_workload, backend)
    finally:
        triton_tbe_module.has_tlx = previous_has_tlx


@register_benchmark(TBEComparisonConfig)
def triton_tbe(config: TBEComparisonConfig) -> None:
    """Benchmark Triton TBE without requiring an FBGEMM implementation."""

    _run_selected_benchmark(config, "triton")


@register_benchmark(TBEComparisonConfig)
def fbgemm_tbe(config: TBEComparisonConfig) -> None:
    """Benchmark FBGEMM TBE without constructing the Triton implementation."""

    _run_selected_benchmark(config, "fbgemm")


@register_benchmark(TBEComparisonConfig)
def tbe_comparison(config: TBEComparisonConfig) -> None:
    """Benchmark FBGEMM and Triton TBE on the same configurable workload."""

    _run_selected_benchmark(config, "comparison")


@dataclass
class TraceTBEForwardConfig(TBEComparisonConfig):
    """Run a forward-only trace workload with legacy benchmark defaults."""

    profile_dir: str = "."
    num_benchmarks: int = 100
    num_profiles: int = 10
    num_warmups: int = 0
    deterministic: bool = False
    run_backward: bool = False


@dataclass
class TritonTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark TorchRec Triton TBE on a trace-derived production shape."""


def _run_trace_forward(config: TraceTBEForwardConfig, backend: str) -> None:
    if not config.workload_catalog or not config.workload:
        raise ValueError("trace benchmarks require --workload_catalog and --workload")
    _run_selected_benchmark(config, backend)


@register_benchmark(TritonTBEForwardConfig)
def triton_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "triton")


@dataclass
class FbgemmTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark FBGEMM CUDA TBE on the identical request."""


@register_benchmark(FbgemmTBEForwardConfig)
def fbgemm_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "fbgemm")


@dataclass
class TritonUVMTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark Triton TBE with managed-memory weights."""


@register_benchmark(TritonUVMTBEForwardConfig)
def triton_uvm_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "triton_uvm")


@dataclass
class TritonUVMCappedTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark Triton UVM TBE with bounded forward launches."""

    forward_block_limit: int = 48
    vbe_forward_block_limit: int = 48


@register_benchmark(TritonUVMCappedTBEForwardConfig)
def triton_uvm_capped_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "triton_uvm_capped")


@dataclass
class TritonUVMCachingTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark Triton UVM TBE with a native HBM row cache."""

    uvm_cache_bytes: int = 2 * 1024**3


@register_benchmark(TritonUVMCachingTBEForwardConfig)
def triton_uvm_caching_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "triton_uvm_caching")


@dataclass
class FbgemmUVMCachingTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark FBGEMM UVM TBE with an equivalently sized HBM cache."""

    uvm_cache_bytes: int = 2 * 1024**3


@register_benchmark(FbgemmUVMCachingTBEForwardConfig)
def fbgemm_uvm_caching_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "fbgemm_uvm_caching")


@dataclass
class FbgemmUVMTBEForwardConfig(TraceTBEForwardConfig):
    """Benchmark FBGEMM TBE with managed-memory weights."""


@register_benchmark(FbgemmUVMTBEForwardConfig)
def fbgemm_uvm_tbe_forward(config: TraceTBEForwardConfig) -> None:
    _run_trace_forward(config, "fbgemm_uvm")


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
