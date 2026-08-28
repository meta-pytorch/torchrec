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
from torchrec.sparse.jagged_tensor import _kt_regroup_arguments
from torchrec.sparse.triton_permute_2d import (
    MIN_SEGMENTS,
    PERSEG_MIN_MEAN,
    triton_permute_2d_sparse_data,
)
from torchrec.sparse.triton_permute_multi_embedding import (
    triton_permute_multi_embedding,
)

logger: logging.Logger = logging.getLogger(__name__)

# permute_2D_sparse_data lives in sparse_ops. Importing triton_permute_2d does not pull
# in jagged_tensor, which is where torchrec normally registers these, so load them here
# the same guarded way. bounds_check_indices needs no equivalent: importing the Triton
# TBE module imports fbgemm_gpu, which registers the TBE ops on package init.
try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
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

    def make_inputs(self, device: torch.device) -> Dict[str, Any]:
        """Build everything the benchmark consumes.

        Runs once per invocation, outside the timed region, so tensor allocation and
        data generation never land in the measurement.
        """
        return {}


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

    print(result)


def register_benchmark(
    config: type[TritonOpConfig],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator factory: register a benchmark function with the CLI, bound to the
    given config class. The decorated function is the per-iteration benchmark and
    its name is the CLI subcommand. Define the config class first, then:

    @register_benchmark(BoundsCheckTritonConfig)
    def bounds_check_triton(_batch_inputs, offsets, ..., **_kwargs): ...
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        def dispatch(arg: TritonOpConfig) -> None:
            single_rank_runner(arg=arg, bench_func=func)

        # CLI subcommand key = benchmark function name; the annotation must be the
        # concrete config subclass so cmd_conf builds its argparse from its fields
        dispatch.__name__ = func.__name__
        dispatch.__annotations__ = {"arg": config, "return": None}
        # pyrefly: ignore[missing-attribute]
        _cc.register(dispatch)
        return func

    return decorator


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


@dataclass
class BoundsCheckCudaConfig(BoundsCheckConfig):
    """
    run commands:
    1. baseline against bounds_check_triton on identical input
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops bounds_check_cuda \
        --name=clean

    use case:
        the fbgemm CUDA op the Triton kernels replace, run on the same seed and shape.
        Note it does strictly more work than the Triton offsets kernel: it validates row
        ids as well as the offsets array, which the Triton path folds into the gather
        loop instead (_load_checked_index). Read the pair as backend A/B on the offsets
        sweep, not as an equal-work comparison.
    """


@register_benchmark(BoundsCheckCudaConfig)
def bounds_check_cuda(
    _batch_inputs: List[Dict[str, Any]],
    indices: torch.Tensor,
    offsets: torch.Tensor,
    rows_per_table: torch.Tensor,
    warning: torch.Tensor,
    **_kwargs: Dict[str, Any],
) -> None:
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


@dataclass
class Permute2dTritonConfig(Permute2dConfig):
    """
    run commands:
    1. blocked kernel (default): many short segments
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops permute_2d_triton \
        --name=blocked

    2. per-segment kernel: fewer, longer segments
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops permute_2d_triton \
        --name=perseg \
        --mean_pooling_factor=1024

    use case:
        the Triton replacement for fbgemm.permute_2D_sparse_data, which wins on the two
        regimes fbgemm's per-segment launch shape handles badly: very many short
        segments, and length skew across keys. Pair with permute_2d_fbgemm on the same
        seed and shape.
    """


@register_benchmark(Permute2dTritonConfig)
def permute_2d_triton(
    _batch_inputs: List[Dict[str, Any]],
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    permuted_lengths_sum: int,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_permute_2d_sparse_data ##"):
        triton_permute_2d_sparse_data(
            permute, lengths, values, weights, permuted_lengths_sum
        )


@dataclass
class Permute2dFbgemmConfig(Permute2dConfig):
    """
    run commands:
    1. baseline against permute_2d_triton on identical input
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops permute_2d_fbgemm \
        --name=blocked

    2. baseline against permute_2d_triton on identical input with pf=1024
    > python -m torchrec.distributed.benchmark.benchmark_triton_ops permute_2d_fbgemm \
        --name=blocked_1024 \
        --mean_pooling_factor=1024

    use case:
        the fbgemm CUDA baseline. It hands each block 32 consecutive segments and picks
        vectorised or scalar loads per segment, which is efficient for few long aligned
        segments and degrades as segment count and key-level length skew grow.
    """


@register_benchmark(Permute2dFbgemmConfig)
def permute_2d_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    permuted_lengths_sum: int,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## permute_2D_sparse_data ##"):
        torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths, values, weights, permuted_lengths_sum
        )


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
        grad_outputs = [
            torch.randn(self.batch_size, length, device=device, dtype=dtype)
            for length in out_lengths
        ]
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
    grad_outputs: List[torch.Tensor],
    run_backward: bool,
) -> None:
    if run_backward:
        torch.autograd.grad(outputs, values, grad_outputs)


@dataclass
class RegroupTritonConfig(RegroupConfig):
    """Benchmark the Triton multi-tensor pooled-embedding regroup."""


@register_benchmark(RegroupTritonConfig)
def regroup_triton(
    _batch_inputs: List[Dict[str, Any]],
    values: List[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: List[int],
    grad_outputs: List[torch.Tensor],
    run_backward: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## triton_permute_multi_embedding ##"):
        outputs = triton_permute_multi_embedding(
            values, permutes, in_shapes, out_shapes, out_lengths
        )
        _run_backward(outputs, values, grad_outputs, run_backward)


@dataclass
class RegroupFbgemmConfig(RegroupConfig):
    """Benchmark the FBGEMM multi-tensor pooled-embedding regroup."""


@register_benchmark(RegroupFbgemmConfig)
def regroup_fbgemm(
    _batch_inputs: List[Dict[str, Any]],
    values: List[torch.Tensor],
    permutes: torch.Tensor,
    in_shapes: torch.Tensor,
    out_shapes: torch.Tensor,
    out_lengths: List[int],
    grad_outputs: List[torch.Tensor],
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
