#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Microbenchmarks for TorchRec's Triton ops, each against its fbgemm CUDA baseline.

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_triton_ops -- \
        bounds_check_triton --name=$(hg whereami | cut -c 1-10)

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_triton_ops \
        permute_2d_triton --name=$(git rev-parse --short HEAD || echo $USER)

Every benchmark is single-rank and single-process: these are kernel microbenchmarks,
so there is no MultiProcessContext and no collective.

see README.md for more details
"""

import logging
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

import torch
import triton
import triton.language as tl
from fbgemm_gpu.quantize_utils import (
    bf16_to_fp32,
    fp32_to_bf16_with_clamp,
    fp32_to_mx4,
    mx4_to_float,
)
from torch.autograd.profiler import record_function

try:
    from fbgemm_gpu.tbe.config.embedding_config import BoundsCheckMode
except ImportError:
    # Base layers predating the tbe.config leaf module still re-export it here.
    from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode

from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    cmd_conf,
)
from torchrec.distributed.triton_tbe.triton_table_batched_embeddings import (
    _bounds_check_offsets_kernel,
    _repair_offsets_kernel,
)
from torchrec.sparse.jagged_tensor import _kt_regroup_arguments, JaggedTensor
from torchrec.sparse.triton_batch_index_select import triton_batch_index_select_dim0
from torchrec.sparse.triton_permute_2d import (
    MIN_SEGMENTS,
    PERSEG_MIN_MEAN,
    triton_permute_2d_sparse_data,
)
from torchrec.sparse.triton_permute_multi_embedding import (
    triton_permute_multi_embedding,
)
from torchrec.sparse.triton_quantized_comm import (
    triton_bfloat16_quantized_to_float,
    triton_float_to_bfloat16_quantized,
    triton_float_to_fused8bitrowwise_quantized,
    triton_float_to_mx4_quantized,
    triton_fused8bitrowwise_quantized_to_float,
    triton_mx4_quantized_to_float,
)

logger: logging.Logger = logging.getLogger(__name__)

# permute_2D_sparse_data lives in sparse_ops. Importing triton_permute_2d does not pull
# in jagged_tensor, which is where torchrec normally registers these, so load them here
# the same guarded way. bounds_check_indices needs no equivalent: importing the Triton
# TBE module imports fbgemm_gpu, which registers the TBE ops on package init.
try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu/codegen:index_select_ops")
except OSError:
    pass

# pyrefly: ignore[missing-argument]
_cc = cmd_conf()

# Mirrors the launch in TritonTableBatchedEmbeddingBags.forward. Both are part of what
# is being measured, so they are pinned here rather than left to a default.
_BOUNDS_CHECK_BLOCK_SIZE = 256
_BOUNDS_CHECK_NUM_WARPS = 8

# The module default is V2_WARNING, which _bounds_check_config() splits into the base
# mode plus a version: the C++ op only accepts FATAL/WARNING/IGNORE, and V2 is selected
# by the separate bounds_check_version argument.
_BOUNDS_CHECK_MODE_WARNING: int = int(BoundsCheckMode.WARNING)
_BOUNDS_CHECK_VERSION_V2 = 2


#################################### util functions ####################################
def _pick(numel: int, pct: int, device: torch.device) -> Optional[torch.Tensor]:
    """Indices of ``pct`` percent of ``numel`` entries, or None when pct <= 0."""
    if pct <= 0 or numel == 0:
        return None
    count = max(1, (numel * pct) // 100)
    return torch.randperm(numel, device=device)[:count]


################################# framework components #################################
@dataclass
class TritonOpConfig(BenchFuncConfig):
    name: str = ""
    world_size: int = 1
    device_type: str = "cuda"
    profile_dir: str = "."
    num_benchmarks: int = 100
    num_profiles: int = 10
    seed: int = 42
    debug_mode: bool = False
    run_warmup: bool = True

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        """Build everything the benchmark consumes.

        Runs once per invocation, outside the timed region, so tensor allocation and
        data generation never land in the measurement.
        """
        return {}

    def validate_outputs(self, kwargs: Dict[str, Any]) -> None:
        """Validate benchmark state after all timed and profiled iterations."""


def _make_benchmark_kwargs(arg: TritonOpConfig, device: torch.device) -> Dict[str, Any]:
    new_keys = {f.name for f in fields(type(arg))} - {
        f.name for f in fields(TritonOpConfig)
    }
    kwargs: Dict[str, Any] = {key: getattr(arg, key) for key in new_keys}
    kwargs |= arg.make_inputs(device)
    return kwargs


# single-rank runner
def single_rank_runner(
    arg: TritonOpConfig,
    bench_func: Callable[..., None],
) -> None:
    assert torch.cuda.is_available(), "these kernels require a CUDA device"

    arg.set_log_level()

    # debug mode only works with vscode for now.
    if arg.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    # Same seed for every subcommand, so a Triton/CUDA pair sees identical input.
    torch.manual_seed(arg.seed)
    device = torch.device(arg.device_type)

    func_name = getattr(bench_func, "__name__", arg.name)
    name: str = f"{func_name}_{arg.name}" if arg.name else func_name

    # Warm up outside the measurement, then reconstruct inputs so mutating ops do not
    # turn a requested dirty-path measurement into a clean-path measurement.
    if arg.run_warmup:
        warmup_kwargs = _make_benchmark_kwargs(arg, device)
        bench_func([], **warmup_kwargs)
        del warmup_kwargs
    torch.manual_seed(arg.seed)
    kwargs = _make_benchmark_kwargs(arg, device)

    result = benchmark_func(
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs=kwargs,
        func_to_benchmark=bench_func,
        rank=0,
        # Input is empty, actual traffic is determined by the benchmark function
        sample_count=0,
        **arg.benchmark_func_kwargs(name=name),
    )

    arg.validate_outputs(kwargs)
    print(result)


def register_benchmark(
    config: type[TritonOpConfig],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator factory: register a benchmark function with the CLI, bound to the
    given config class. The decorated function is the per-iteration benchmark and
    its name is the CLI subcommand. Define the config class first, then:

    @register_benchmark(BoundsCheckConfig)
    def bounds_check_triton(_batch_inputs, offsets, ..., **_kwargs): ...
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        def dispatch(arg: TritonOpConfig) -> None:
            single_rank_runner(arg=arg, bench_func=func)

        # CLI subcommand key = benchmark function name; the annotation must be the
        # concrete config class so cmd_conf builds its argparse from its fields
        dispatch.__name__ = func.__name__
        dispatch.__annotations__ = {"arg": config, "return": None}
        # pyrefly: ignore[missing-attribute]
        _cc.register(dispatch)
        return func

    return decorator


############################ autotune demonstration ##################################
def _power_of_two_bucket(value: int) -> int:
    if value <= 0:
        raise ValueError("autotune dimensions must be positive")
    return 1 << (value - 1).bit_length()


def _autotune_jagged_copy_repr(specialization: Any) -> str:
    constants = specialization.constants
    return (
        "_autotune_jagged_copy_kernel"
        f"_KB{constants['batch_size_bucket']}"
        f"_KL{constants['length_bucket']}"
        f"_KD{constants['dim_bucket']}"
        f"_BB{constants['BLOCK_BATCH']}"
        f"_BL{constants['BLOCK_LENGTH']}"
        f"_BD{constants['BLOCK_DIM']}"
    )


# Triton TR001: autotune independent jagged batch, length, and dimension tiles.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_BATCH": 1, "BLOCK_LENGTH": 1, "BLOCK_DIM": 32},
            num_warps=1,
        ),
        triton.Config(
            {"BLOCK_BATCH": 1, "BLOCK_LENGTH": 2, "BLOCK_DIM": 64},
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_BATCH": 2, "BLOCK_LENGTH": 2, "BLOCK_DIM": 64},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_BATCH": 2, "BLOCK_LENGTH": 4, "BLOCK_DIM": 128},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_BATCH": 4, "BLOCK_LENGTH": 4, "BLOCK_DIM": 128},
            num_warps=8,
        ),
    ],
    key=["batch_size_bucket", "length_bucket", "dim_bucket"],
)
@triton.jit(repr=_autotune_jagged_copy_repr)
def _autotune_jagged_copy_kernel(
    values_ptr,
    offsets_ptr,
    output_ptr,
    batch_size,
    dim,
    batch_size_bucket: tl.constexpr,
    length_bucket: tl.constexpr,
    dim_bucket: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_LENGTH: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
) -> None:
    batch_ids = tl.program_id(0) * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    length_ids = tl.program_id(1) * BLOCK_LENGTH + tl.arange(0, BLOCK_LENGTH)
    dim_ids = tl.program_id(2) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    batch_mask = batch_ids < batch_size
    starts = tl.load(offsets_ptr + batch_ids, mask=batch_mask, other=0)
    ends = tl.load(offsets_ptr + batch_ids + 1, mask=batch_mask, other=0)
    value_rows = starts[:, None, None] + length_ids[None, :, None]
    value_offsets = value_rows * dim + dim_ids[None, None, :]
    mask = (
        batch_mask[:, None, None]
        & (value_rows < ends[:, None, None])
        & (dim_ids[None, None, :] < dim)
    )
    values = tl.load(values_ptr + value_offsets, mask=mask)
    tl.store(output_ptr + value_offsets, values, mask=mask)


@dataclass
class AutotuneWorkload:
    name: str
    batch_size: int
    max_length: int
    dim: int
    dtype: str
    batch_size_bucket: int
    length_bucket: int
    dim_bucket: int
    input: JaggedTensor
    output: torch.Tensor
    grid: Callable[[Dict[str, Any]], tuple[Any, ...]]


@dataclass
class AutotuneTritonConfig(TritonOpConfig):
    """Trace bucket- and dtype-dependent choices made by Triton autotuning.

    The record_function range identifies both the exact jagged shape and its bucketed
    autotune key. The GPU kernel symbol independently contains the key buckets and the
    selected tile sizes, while its CUDA block dimension divided by 32 gives num_warps.

    run command:
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops \
        autotune_triton --name=h100
    """

    small_batch_size: int = 3
    small_max_length: int = 17
    small_dim: int = 48
    same_bucket_batch_size: int = 4
    same_bucket_max_length: int = 31
    same_bucket_dim: int = 63
    large_batch_size: int = 1000
    large_max_length: int = 100
    large_dim: int = 96
    base_dtype: str = "float32"
    comparison_dtype: str = "bfloat16"

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        workloads: List[AutotuneWorkload] = []
        cases = (
            (
                "small_base_dtype",
                self.small_batch_size,
                self.small_max_length,
                self.small_dim,
                self.base_dtype,
            ),
            (
                "same_bucket_base_dtype",
                self.same_bucket_batch_size,
                self.same_bucket_max_length,
                self.same_bucket_dim,
                self.base_dtype,
            ),
            (
                "large_base_dtype",
                self.large_batch_size,
                self.large_max_length,
                self.large_dim,
                self.base_dtype,
            ),
            (
                "large_comparison_dtype",
                self.large_batch_size,
                self.large_max_length,
                self.large_dim,
                self.comparison_dtype,
            ),
        )
        for name, batch_size, max_length, dim, dtype_name in cases:
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(dtype_name)
            if dtype is None:
                raise ValueError(
                    "base_dtype and comparison_dtype must be float32, float16, "
                    "or bfloat16"
                )
            length_values = [
                max_length if index == 0 else 1 + (index * 37) % max_length
                for index in range(batch_size)
            ]
            lengths = torch.tensor(length_values, device=device, dtype=torch.int64)
            offsets = torch.zeros(batch_size + 1, device=device, dtype=torch.int64)
            torch.cumsum(lengths, dim=0, out=offsets[1:])
            values = torch.randn(sum(length_values), dim, device=device, dtype=dtype)
            jagged_input = JaggedTensor(values=values, lengths=lengths, offsets=offsets)
            batch_size_bucket = _power_of_two_bucket(batch_size)
            length_bucket = _power_of_two_bucket(max_length)
            dim_bucket = _power_of_two_bucket(dim)

            def grid(
                meta: Dict[str, Any],
                grid_batch_size: int = batch_size,
                grid_max_length: int = max_length,
                grid_dim: int = dim,
            ) -> tuple[Any, ...]:
                return (
                    triton.cdiv(grid_batch_size, meta["BLOCK_BATCH"]),
                    triton.cdiv(grid_max_length, meta["BLOCK_LENGTH"]),
                    triton.cdiv(grid_dim, meta["BLOCK_DIM"]),
                )

            workloads.append(
                AutotuneWorkload(
                    name=name,
                    batch_size=batch_size,
                    max_length=max_length,
                    dim=dim,
                    dtype=dtype_name,
                    batch_size_bucket=batch_size_bucket,
                    length_bucket=length_bucket,
                    dim_bucket=dim_bucket,
                    input=jagged_input,
                    output=torch.empty_like(values),
                    grid=grid,
                )
            )
        return {"workloads": workloads}


@register_benchmark(AutotuneTritonConfig)
def autotune_triton(
    _batch_inputs: List[Dict[str, Any]],
    workloads: List[AutotuneWorkload],
    **_kwargs: Dict[str, Any],
) -> None:
    for workload in workloads:
        with record_function(
            "## autotune_triton "
            f"case={workload.name} "
            f"shape[batch_size={workload.batch_size},"
            f"max_length={workload.max_length},dim={workload.dim},"
            f"dtype={workload.dtype}] "
            f"bucket[batch_size={workload.batch_size_bucket},"
            f"length={workload.length_bucket},dim={workload.dim_bucket}] ##"
        ):
            _autotune_jagged_copy_kernel[workload.grid](
                workload.input.values(),
                workload.input.offsets(),
                workload.output,
                workload.batch_size,
                workload.dim,
                workload.batch_size_bucket,
                workload.length_bucket,
                workload.dim_bucket,
            )


######################## autotune restore_value demonstration ########################
_AUTOTUNE_ADD_ONE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]


# Triton TR001: autotune the pointwise tile size used for the mutation.
@triton.autotune(
    configs=_AUTOTUNE_ADD_ONE_CONFIGS,
    key=["numel"],
    restore_value=["values"],
)
@triton.jit
def _autotune_add_one_with_restore_kernel(
    values,
    numel,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    value = tl.load(values + offsets, mask=mask)
    tl.store(values + offsets, value + 1, mask=mask)


# Triton TR001: use the same choices as the restore_value treatment.
@triton.autotune(
    configs=_AUTOTUNE_ADD_ONE_CONFIGS,
    key=["numel"],
)
@triton.jit
def _autotune_add_one_without_restore_kernel(
    values,
    numel,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    value = tl.load(values + offsets, mask=mask)
    tl.store(values + offsets, value + 1, mask=mask)


@dataclass
class AutotuneRestoreValueState:
    values: torch.Tensor
    sample_indices: torch.Tensor
    logical_invocations: int = 0


def _read_autotune_restore_value_samples(
    state: AutotuneRestoreValueState,
) -> List[float]:
    return state.values.index_select(0, state.sample_indices).cpu().tolist()


def _trace_autotune_restore_value_samples(
    state: AutotuneRestoreValueState,
) -> None:
    if not torch.autograd.profiler._is_profiler_enabled:
        return

    # The readback intentionally synchronizes only profiled iterations. Keeping it in
    # a separate range makes the validation overhead explicit and excludes it from the
    # autotune_restore_value timing range.
    with record_function("## autotune_restore_value validation readback ##"):
        observed = _read_autotune_restore_value_samples(state)
    observed_label = ",".join(f"{value:g}" for value in observed)
    with record_function(
        "## autotune_restore_value validation result "
        f"logical_invocations={state.logical_invocations} "
        f"sample_values={observed_label} ##"
    ):
        pass


@dataclass
class AutotuneRestoreValueConfig(TritonOpConfig):
    """Expose the correctness and memory cost of restoring a mutating input.

    The default FP16 input is 12 GB, representative of an embedding table mutated by
    TBE backward. Clearing the autotune cache before each outer iteration makes every
    iteration exercise candidate benchmarking. Warmup is disabled so that disabling
    retune_each_iteration shows one cold call followed by cached steady-state calls.
    """

    num_benchmarks: int = 0
    memory_snapshot: bool = True
    run_warmup: bool = False
    numel: int = 6_000_000_000
    restore_value: bool = True
    retune_each_iteration: bool = True

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        if self.numel <= 0:
            raise ValueError("numel must be positive")
        # Citrine C3: allocate the large input directly on its target GPU.
        values = torch.zeros(self.numel, device=device, dtype=torch.float16)
        sample_indices = torch.tensor(
            [0, self.numel // 2, self.numel - 1],
            device=device,
        )
        return {
            "state": AutotuneRestoreValueState(
                values=values,
                sample_indices=sample_indices,
            )
        }

    def validate_outputs(self, kwargs: Dict[str, Any]) -> None:
        state = kwargs["state"]
        assert isinstance(state, AutotuneRestoreValueState)
        observed = _read_autotune_restore_value_samples(state)
        expected = float(state.logical_invocations)
        cache_is_warm_before_inputs = self.run_warmup and not self.retune_each_iteration
        if self.restore_value or cache_is_warm_before_inputs:
            if observed != [expected] * len(observed):
                raise AssertionError(
                    "restore_value failed to preserve one mutation per logical "
                    f"invocation: expected {expected}, observed {observed}"
                )
        elif not all(value > expected for value in observed):
            raise AssertionError(
                "unrestored autotune did not expose candidate mutations: "
                f"expected values above {expected}, observed {observed}"
            )
        logger.info(
            "Validated autotune mutation count: logical=%d observed=%s",
            state.logical_invocations,
            observed,
        )


@register_benchmark(AutotuneRestoreValueConfig)
def autotune_restore_value(
    _batch_inputs: List[Dict[str, Any]],
    state: AutotuneRestoreValueState,
    restore_value: bool,
    retune_each_iteration: bool,
    **_kwargs: Dict[str, Any],
) -> None:
    kernel = (
        _autotune_add_one_with_restore_kernel
        if restore_value
        else _autotune_add_one_without_restore_kernel
    )
    if retune_each_iteration:
        kernel.cache.clear()

    numel = state.values.numel()

    def grid(meta: Dict[str, Any]) -> tuple[Any, ...]:
        return (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    with record_function(
        "## autotune_restore_value "
        f"bytes={state.values.nbytes} "
        f"restore_value={restore_value} "
        f"retune_each_iteration={retune_each_iteration} ##"
    ):
        kernel[grid](state.values, numel)
    state.logical_invocations += 1
    _trace_autotune_restore_value_samples(state)


############################### TBE bounds check configs ###############################
@dataclass
class BoundsCheckConfig(TritonOpConfig):
    """Shared input generation for both bounds-check backends.

    Corrupt-input caveat: a bounds check in WARNING mode is a *repair* operation. Both
    backends rewrite the offending entries on the first iteration, so with a non-zero
    corruption percentage the remaining iterations measure already-clean data and the
    reported cost understates the dirty path. Pass --num_benchmarks=1 for a true
    dirty-path number.

    The two corruption knobs are not interchangeable. bad_offsets_pct breaks the offsets
    array and is the only one _bounds_check_offsets_kernel can see; oob_pct puts row ids
    out of range, which only the CUDA op and the in-gather _load_checked_index check.
    """

    # Defaults follow the shape used by the Triton-vs-FBGEMM TBE benchmark so numbers
    # are comparable against it.
    num_tables: int = 32
    batch_size: int = 131072
    bag_size: int = 20
    num_embeddings: int = 10_000_000
    oob_pct: int = 0
    bad_offsets_pct: int = 0

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        """Build (indices, offsets, rows_per_table, warning) in the TBE CSR layout.

        ``offsets`` is feature-major with ``num_tables * batch_size + 1`` entries,
        matching what TritonTableBatchedEmbeddingBags.forward receives from a KJT.
        """
        total_bags = self.num_tables * self.batch_size
        lengths = torch.randint(
            low=0,
            high=2 * self.bag_size + 1,
            size=(total_bags,),
            dtype=torch.int64,
            device=device,
        )
        offsets = torch.zeros(total_bags + 1, dtype=torch.int64, device=device)
        torch.cumsum(lengths, 0, out=offsets[1:])
        # Setup-time sync only; nothing in the timed region reads back to host.
        num_indices = int(offsets[-1].item())

        indices = torch.randint(
            low=0,
            high=self.num_embeddings,
            size=(num_indices,),
            dtype=torch.int64,
            device=device,
        )

        oob = _pick(num_indices, self.oob_pct, device)
        if oob is not None:
            # Past the end of every table, so the row-id range test fires.
            indices[oob] = self.num_embeddings + 1

        bad = _pick(total_bags, self.bad_offsets_pct, device)
        if bad is not None:
            # Negative trips both `starts < 0` on this lane and `starts > ends` on the
            # previous one, which is what the offsets kernel is looking for.
            offsets[bad] = -1

        logger.info(
            "T=%d B=%d L=%d -> %d bags, %d indices",
            self.num_tables,
            self.batch_size,
            self.bag_size,
            total_bags,
            num_indices,
        )
        return {
            "indices": indices,
            "offsets": offsets,
            "rows_per_table": torch.full(
                (self.num_tables,),
                self.num_embeddings,
                dtype=torch.int64,
                device=device,
            ),
            "warning": torch.zeros(1, dtype=torch.int64, device=device),
            "num_indices": num_indices,
            "total_bags": total_bags,
        }


@dataclass
class BoundsCheckTritonConfig(BoundsCheckConfig):
    """
    run commands:
    1. offsets kernel only (default)
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops bounds_check_triton \
        --name=clean

    2. include the repair kernel
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops bounds_check_triton \
        --name=repair \
        --include_repair=True

    3. dirty offsets, one iteration (see the caveat on BoundsCheckConfig)
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops bounds_check_triton \
        --name=dirty \
        --bad_offsets_pct=5 --num_benchmarks=1

    use case:
        time _bounds_check_offsets_kernel on its own. The kernels are launched directly
        rather than through TritonTableBatchedEmbeddingBags, for two reasons: the module
        would wrap them in a full forward, and its fused path silently falls back to the
        CUDA op unless mode is WARNING and there is no VBE, no per-sample weights, no
        hoisted transpose, and no AMD -- so a module-level benchmark can quietly measure
        the wrong backend.
    """

    include_repair: bool = False


@register_benchmark(BoundsCheckTritonConfig)
def bounds_check_triton(
    _batch_inputs: List[Dict[str, Any]],
    offsets: torch.Tensor,
    warning: torch.Tensor,
    num_indices: int,
    total_bags: int,
    include_repair: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## zero warning counter ##"):
        # The module zeroes the counter on every forward, so it is inside the timed
        # region for both backends and cancels out of the comparison.
        warning.zero_()

    with record_function("## bounds check offsets ##"):
        # constexpr params and the num_warps launch knob are part of Triton's launch
        # protocol, which the type checker does not model.
        _bounds_check_offsets_kernel[
            (triton.cdiv(total_bags, _BOUNDS_CHECK_BLOCK_SIZE),)
        ](
            offsets,
            warning,
            num_indices,
            total_bags,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_SIZE=_BOUNDS_CHECK_BLOCK_SIZE,
            # pyrefly: ignore[unexpected-keyword]
            num_warps=_BOUNDS_CHECK_NUM_WARPS,
        )

    if include_repair:
        with record_function("## repair offsets ##"):
            # Serial O(total_bags) scan on one program, guarded on the warning counter,
            # so this is a near-free early-exit unless the offsets are actually broken.
            _repair_offsets_kernel[(1,)](
                offsets,
                warning,
                num_indices,
                total_bags,
                # pyrefly: ignore[unexpected-keyword]
                num_warps=1,
            )


@register_benchmark(BoundsCheckConfig)
def bounds_check_cuda(
    _batch_inputs: List[Dict[str, Any]],
    indices: torch.Tensor,
    offsets: torch.Tensor,
    rows_per_table: torch.Tensor,
    warning: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    """Benchmark FBGEMM's CUDA baseline on the same seed and shape.

    It validates row IDs and offsets, while the Triton benchmark only times its offsets
    kernel because row validation is fused into the gather loop.
    """
    with record_function("## zero warning counter ##"):
        warning.zero_()

    with record_function("## bounds_check_indices ##"):
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            _BOUNDS_CHECK_MODE_WARNING,
            warning,
            None,
            bounds_check_version=_BOUNDS_CHECK_VERSION_V2,
        )


################################ 2D permute configs ####################################
@dataclass
class Permute2dConfig(TritonOpConfig):
    """Shared input generation for both 2D-permute backends.

    Defaults give 1,048,576 segments, above triton_permute_2d.MIN_SEGMENTS (700k) where
    should_use_triton() takes over, at a mean length below PERSEG_MIN_MEAN so the
    load-balanced kernel is the one measured. Raise mean_pooling_factor past
    PERSEG_MIN_MEAN to measure the per-segment kernel instead.
    """

    num_features: int = 1024
    permute_batch_size: int = 1024
    mean_pooling_factor: int = 1
    has_weight: bool = False
    gpu_backlog_ms: float = 5.0

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        """Build (permute, lengths, values, weights, permuted_lengths_sum).

        ``permute`` is a full permutation, so ``permuted_lengths_sum`` is just the
        total. Real call sites also pass subsets and repeats, which is why fbgemm
        carries the sum as a separate argument; this keeps the simple case so the two
        backends move exactly the same bytes.
        """
        lengths = torch.randint(
            low=0,
            high=max(2 * self.mean_pooling_factor, 2),
            size=(self.num_features, self.permute_batch_size),
            dtype=torch.int32,
            device=device,
        )
        permuted_lengths_sum = int(lengths.sum().item())
        values = torch.randint(
            low=0,
            high=int(1e5),
            size=(permuted_lengths_sum,),
            dtype=torch.int32,
            device=device,
        )
        permute = torch.randperm(self.num_features, device=device).to(torch.int32)

        num_segments = self.num_features * self.permute_batch_size
        logger.info(
            "%d segments (MIN_SEGMENTS=%d), %d values, mean length %.2f -> %s kernel",
            num_segments,
            MIN_SEGMENTS,
            permuted_lengths_sum,
            permuted_lengths_sum / max(num_segments, 1),
            (
                "per-segment"
                if permuted_lengths_sum >= num_segments * PERSEG_MIN_MEAN
                else "blocked"
            ),
        )
        if num_segments < MIN_SEGMENTS:
            logger.warning(
                "%d segments is below MIN_SEGMENTS=%d, where should_use_triton() defers "
                "to fbgemm; this does not reflect a shape the Triton path would serve.",
                num_segments,
                MIN_SEGMENTS,
            )
        return {
            "permute": permute,
            "lengths": lengths,
            "values": values,
            "weights": (
                torch.rand(permuted_lengths_sum, dtype=torch.float32, device=device)
                if self.has_weight
                else None
            ),
            "permuted_lengths_sum": permuted_lengths_sum,
        }


@register_benchmark(Permute2dConfig)
def permute_2d_triton(
    _batch_inputs: List[Dict[str, Any]],
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    permuted_lengths_sum: int,
    **_kwargs: Dict[str, Any],
) -> None:
    """Benchmark TorchRec's Triton replacement for permute_2D_sparse_data."""
    with record_function("## triton_permute_2d_sparse_data ##"):
        triton_permute_2d_sparse_data(
            permute, lengths, values, weights, permuted_lengths_sum
        )


@register_benchmark(Permute2dConfig)
def permute_2d_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    permuted_lengths_sum: int,
    **_kwargs: Dict[str, Any],
) -> None:
    """Benchmark FBGEMM's CUDA permute_2D_sparse_data baseline."""
    with record_function("## permute_2D_sparse_data ##"):
        torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, values, weights, permuted_lengths_sum
        )


######################## batch index select dim 0 configs ############################
@dataclass
class BatchIndexSelectDim0Config(TritonOpConfig):
    """Inputs matching VariableBatchEmbeddingBagCollectionAwaitable.

    Embedding dimensions cycle through embedding_dims. Setting different minimum
    and maximum input row counts exercises variable batch sizes before all-to-all;
    output_batch_size is the common reconstructed batch size.
    """

    num_features: int = 165
    min_input_rows: int = 1024
    max_input_rows: int = 4096
    output_batch_size: int = 4096
    embedding_dims: str = "16,16,16,16,48,80,112,128,160"
    run_backward: bool = False
    dtype: str = "float16"
    gpu_backlog_ms: float = 20.0

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype)
        if dtype is None:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.min_input_rows <= 0 or self.max_input_rows < self.min_input_rows:
            raise ValueError("input row bounds must be positive and ordered")
        if self.max_input_rows > self.output_batch_size:
            raise ValueError(
                "input row counts cannot exceed the reconstructed output batch size"
            )
        dimensions = [int(value) for value in self.embedding_dims.split(",")]
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise ValueError("embedding_dims must contain positive integers")
        input_columns = [
            dimensions[index % len(dimensions)] for index in range(self.num_features)
        ]
        input_rows = torch.randint(
            self.min_input_rows,
            self.max_input_rows + 1,
            (self.num_features,),
        ).tolist()
        inputs = torch.randn(
            sum(rows * columns for rows, columns in zip(input_rows, input_columns)),
            device=device,
            dtype=dtype,
            requires_grad=self.run_backward,
        )
        indices = torch.cat(
            [
                torch.cat(
                    [
                        torch.arange(rows, device=device, dtype=torch.int64),
                        torch.randint(
                            0,
                            rows,
                            (self.output_batch_size - rows,),
                            device=device,
                            dtype=torch.int64,
                        ),
                    ]
                )[torch.randperm(self.output_batch_size, device=device)]
                for rows in input_rows
            ]
        )
        grad_output = (
            torch.randn(
                self.output_batch_size * sum(input_columns),
                device=device,
                dtype=dtype,
            )
            if self.run_backward
            else None
        )
        return {
            "inputs": inputs,
            "indices": indices,
            "input_rows": input_rows,
            "input_columns": input_columns,
            "grad_output": grad_output,
        }


def _run_batch_index_select_backward(
    output: torch.Tensor,
    inputs: torch.Tensor,
    grad_output: Optional[torch.Tensor],
    run_backward: bool,
) -> None:
    if run_backward:
        assert grad_output is not None
        torch.autograd.grad(output, inputs, grad_output)


@register_benchmark(BatchIndexSelectDim0Config)
def batch_index_select_dim0_triton(
    _batch_inputs: List[Dict[str, Any]],
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_rows: List[int],
    input_columns: List[int],
    grad_output: Optional[torch.Tensor],
    output_batch_size: int,
    run_backward: bool,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_batch_index_select_dim0 ##"):
        output = triton_batch_index_select_dim0(
            inputs, indices, output_batch_size, input_rows, input_columns
        )
        _run_batch_index_select_backward(output, inputs, grad_output, run_backward)


@register_benchmark(BatchIndexSelectDim0Config)
def batch_index_select_dim0_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_rows: List[int],
    input_columns: List[int],
    grad_output: Optional[torch.Tensor],
    output_batch_size: int,
    run_backward: bool,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## fbgemm_batch_index_select_dim0 ##"):
        output = torch.ops.fbgemm.batch_index_select_dim0(
            inputs=inputs,
            indices=indices,
            input_num_indices=[output_batch_size] * len(input_columns),
            input_rows=input_rows,
            input_columns=input_columns,
            permute_output_dim_0_1=True,
        )
        _run_batch_index_select_backward(output, inputs, grad_output, run_backward)


@register_benchmark(BatchIndexSelectDim0Config)
def batch_index_select_dim0_torch(
    _batch_inputs: List[Dict[str, Any]],
    inputs: torch.Tensor,
    indices: torch.Tensor,
    input_rows: List[int],
    input_columns: List[int],
    grad_output: Optional[torch.Tensor],
    output_batch_size: int,
    run_backward: bool,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## torch_batch_index_select_dim0 ##"):
        input_splits = inputs.split(
            [rows * columns for rows, columns in zip(input_rows, input_columns)]
        )
        index_splits = indices.split(output_batch_size)
        output = torch.cat(
            [
                input_part.view(rows, columns).index_select(0, index_part)
                for input_part, index_part, rows, columns in zip(
                    input_splits, index_splits, input_rows, input_columns
                )
            ],
            dim=1,
        ).flatten()
        _run_batch_index_select_backward(output, inputs, grad_output, run_backward)


######################## quantized communication configs ############################
@dataclass
class QuantizedCommConfig(TritonOpConfig):
    """FP32 communication tensors quantized independently by their last dimension."""

    num_rows: int = 65536
    num_columns: int = 32
    gpu_backlog_ms: float = 20.0

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        if self.num_rows < 0 or self.num_columns <= 0:
            raise ValueError("num_rows must be nonnegative and num_columns positive")
        input = torch.randn(
            self.num_rows,
            self.num_columns,
            dtype=torch.float32,
            device=device,
        )
        quantized = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input)
        return {
            "input": input,
            "fused_8bit": quantized,
            "bfloat16": fp32_to_bf16_with_clamp(input),
            "mx4": fp32_to_mx4(input),
        }


@register_benchmark(QuantizedCommConfig)
def fused_8bit_rowwise_quantize_triton(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_float_to_fused8bitrowwise_quantized ##"):
        triton_float_to_fused8bitrowwise_quantized(input)


@register_benchmark(QuantizedCommConfig)
def fused_8bit_rowwise_quantize_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## fbgemm_float_to_fused8bitrowwise_quantized ##"):
        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input)


@register_benchmark(QuantizedCommConfig)
def fused_8bit_rowwise_dequantize_triton(
    _batch_inputs: List[Dict[str, Any]],
    fused_8bit: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_fused8bitrowwise_quantized_to_float ##"):
        triton_fused8bitrowwise_quantized_to_float(fused_8bit)


@register_benchmark(QuantizedCommConfig)
def fused_8bit_rowwise_dequantize_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    fused_8bit: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## fbgemm_fused8bitrowwise_quantized_to_float ##"):
        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(fused_8bit)


@register_benchmark(QuantizedCommConfig)
def fused_8bit_rowwise_roundtrip_triton(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_fused8bitrowwise_roundtrip ##"):
        triton_fused8bitrowwise_quantized_to_float(
            triton_float_to_fused8bitrowwise_quantized(input)
        )


@register_benchmark(QuantizedCommConfig)
def fused_8bit_rowwise_roundtrip_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## fbgemm_fused8bitrowwise_roundtrip ##"):
        torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
            torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input)
        )


@register_benchmark(QuantizedCommConfig)
def bfloat16_quantize_triton(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_float_to_bfloat16_quantized ##"):
        triton_float_to_bfloat16_quantized(input)


@register_benchmark(QuantizedCommConfig)
def bfloat16_quantize_qcomm(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## qcomm_float_to_bfloat16_quantized ##"):
        fp32_to_bf16_with_clamp(input)


@register_benchmark(QuantizedCommConfig)
def bfloat16_dequantize_triton(
    _batch_inputs: List[Dict[str, Any]],
    bfloat16: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_bfloat16_quantized_to_float ##"):
        triton_bfloat16_quantized_to_float(bfloat16)


@register_benchmark(QuantizedCommConfig)
def bfloat16_dequantize_qcomm(
    _batch_inputs: List[Dict[str, Any]],
    bfloat16: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## qcomm_bfloat16_quantized_to_float ##"):
        bf16_to_fp32(bfloat16)


@register_benchmark(QuantizedCommConfig)
def bfloat16_roundtrip_triton(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_bfloat16_roundtrip ##"):
        triton_bfloat16_quantized_to_float(triton_float_to_bfloat16_quantized(input))


@register_benchmark(QuantizedCommConfig)
def bfloat16_roundtrip_qcomm(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## qcomm_bfloat16_roundtrip ##"):
        bf16_to_fp32(fp32_to_bf16_with_clamp(input))


@register_benchmark(QuantizedCommConfig)
def mx4_quantize_triton(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## torchrec_triton_float_to_mx4_quantized ##"):
        triton_float_to_mx4_quantized(input)


@register_benchmark(QuantizedCommConfig)
def mx4_quantize_qcomm(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## qcomm_triton_float_to_mx4_quantized ##"):
        fp32_to_mx4(input)


@register_benchmark(QuantizedCommConfig)
def mx4_dequantize_triton(
    _batch_inputs: List[Dict[str, Any]],
    mx4: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## torchrec_triton_mx4_quantized_to_float ##"):
        triton_mx4_quantized_to_float(mx4)


@register_benchmark(QuantizedCommConfig)
def mx4_dequantize_qcomm(
    _batch_inputs: List[Dict[str, Any]],
    mx4: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## qcomm_triton_mx4_quantized_to_float ##"):
        mx4_to_float(mx4)


@register_benchmark(QuantizedCommConfig)
def mx4_roundtrip_triton(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## torchrec_triton_mx4_roundtrip ##"):
        triton_mx4_quantized_to_float(triton_float_to_mx4_quantized(input))


@register_benchmark(QuantizedCommConfig)
def mx4_roundtrip_qcomm(
    _batch_inputs: List[Dict[str, Any]],
    input: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## qcomm_triton_mx4_roundtrip ##"):
        mx4_to_float(fp32_to_mx4(input))


############################ pooled regroup configs ###################################
@dataclass
class RegroupConfig(TritonOpConfig):
    """Inputs for cached-metadata multi-tensor pooled-embedding regroup."""

    batch_size: int = 1024
    num_dense_features: int = 20
    num_sparse_features: int = 1000
    dense_dim: int = 64
    sparse_dim: int = 128
    num_groups: int = 2
    skipped_features: int = 0
    duplicate_features: int = 0
    run_backward: bool = False
    dtype: str = "float32"

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype)
        if dtype is None:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.run_backward and self.skipped_features > 0:
            raise ValueError(
                "backward with skipped features has unspecified gradients in the "
                "FBGEMM-compatible contract"
            )
        keys = [
            [f"dense_{i}" for i in range(self.num_dense_features)],
            [f"sparse_{i}" for i in range(self.num_sparse_features)],
        ]
        lengths = [
            [self.dense_dim] * self.num_dense_features,
            [self.sparse_dim] * self.num_sparse_features,
        ]
        values = [
            torch.randn(
                self.batch_size,
                sum(tensor_lengths),
                device=device,
                dtype=dtype,
                requires_grad=self.run_backward,
            )
            for tensor_lengths in lengths
        ]

        all_keys = keys[0] + keys[1]
        if self.skipped_features >= len(all_keys):
            raise ValueError("skipped_features must leave at least one feature")
        selected_keys = all_keys[self.skipped_features :]
        groups: List[List[str]] = [[] for _ in range(self.num_groups)]
        for index, key in enumerate(selected_keys):
            groups[index % self.num_groups].append(key)
        for index in range(self.duplicate_features):
            groups[index % self.num_groups].append(
                selected_keys[index % len(selected_keys)]
            )

        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            values[0], keys, lengths, groups
        )
        grad_outputs = (
            [
                torch.randn(self.batch_size, length, device=device, dtype=dtype)
                for length in out_lengths
            ]
            if self.run_backward
            else None
        )
        return {
            "values": values,
            "permutes": permutes,
            "in_shapes": in_shapes,
            "out_shapes": out_shapes,
            "out_lengths": out_lengths,
            "grad_outputs": grad_outputs,
        }


def _run_backward(
    outputs: List[torch.Tensor],
    values: List[torch.Tensor],
    grad_outputs: Optional[List[torch.Tensor]],
    run_backward: bool,
) -> None:
    if run_backward:
        assert grad_outputs is not None
        torch.autograd.grad(outputs, values, grad_outputs)


@register_benchmark(RegroupConfig)
def regroup_triton(
    _batch_inputs: List[Dict[str, Any]],
    values: List[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: List[int],
    grad_outputs: Optional[List[torch.Tensor]],
    run_backward: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_permute_multi_embedding ##"):
        outputs = triton_permute_multi_embedding(
            values, permutes, in_shapes, out_shapes, out_lengths
        )
        _run_backward(outputs, values, grad_outputs, run_backward)


@register_benchmark(RegroupConfig)
def regroup_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    values: List[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: List[int],
    grad_outputs: Optional[List[torch.Tensor]],
    run_backward: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## fbgemm_permute_multi_embedding ##"):
        outputs = torch.ops.fbgemm.permute_multi_embedding(
            values, permutes, in_shapes, out_shapes, out_lengths
        )
        _run_backward(outputs, values, grad_outputs, run_backward)


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
