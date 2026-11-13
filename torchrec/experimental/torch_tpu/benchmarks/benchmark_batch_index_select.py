#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Microbenchmark Pallas batch index select, backward, and its grouping stage.

Preview without importing Torch or initializing TPU:
    python benchmark_batch_index_select.py --suite standard --dry-run

Run in an environment configured by the TorchTPU distributed launcher:
    python benchmark_batch_index_select.py --suite smoke
    python benchmark_batch_index_select.py --suite custom --batch-size 11837 \
        --features 4 --input-rows 16,72,2712,82192 --input-columns 16 \
        --unique-rows 16,32,256,2048 --index-layout spread

B counts requested rows per feature. R counts stored input rows per feature.
U counts referenced unique rows, and D is the local embedding width. Backward
includes grouping; standalone grouping and zero-fill timings are diagnostic
components, whose latencies should not be added to the backward latency.
Timings include Python dispatch and a blocking TPU synchronization per call,
but exclude input generation, host/device transfers, and correctness checks.
The first call includes compilation/cache loading and is reported separately.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    input_rows: tuple[int, ...]
    input_columns: tuple[int, ...]
    unique_rows: tuple[int, ...]
    distribution: str = "uniform"
    index_layout: str = "compact"
    dtype: str = "float32"
    hot_fraction: float = 0.9

    def validate(self) -> None:
        if self.batch_size < 1 or not self.input_rows:
            raise ValueError("B and feature count must be positive")
        if not len(self.input_rows) == len(self.input_columns) == len(self.unique_rows):
            raise ValueError("R, D, and U must have one entry per feature")
        for rows, dim, unique in zip(
            self.input_rows, self.input_columns, self.unique_rows
        ):
            if rows < 1 or dim < 16 or dim % 16:
                raise ValueError(
                    "R must be positive and D must be a positive multiple of 16"
                )
            if not 1 <= unique <= min(rows, self.batch_size):
                raise ValueError("Each U must satisfy 1 <= U <= min(R, B)")
        if not 0 <= self.hot_fraction <= 1:
            raise ValueError("hot_fraction must lie in [0, 1]")


def make_indices(case: Case, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    result = []
    for rows, unique in zip(case.input_rows, case.unique_rows):
        if case.index_layout == "compact":
            row_ids = list(range(unique))
        elif case.index_layout == "tail":
            row_ids = list(range(rows - unique, rows))
        else:
            row_ids = rng.sample(range(rows), unique)
            if rows - 1 not in row_ids:
                row_ids[-1] = rows - 1
        extra = case.batch_size - unique
        if case.distribution == "hot" and unique > 1:
            hot = round(extra * case.hot_fraction)
            positions = [0] * hot + [1 + i % (unique - 1) for i in range(extra - hot)]
        elif case.distribution == "zipf":
            positions = rng.choices(
                range(unique),
                weights=[1 / (i + 1) ** 1.2 for i in range(unique)],
                k=extra,
            )
        else:
            positions = [i % unique for i in range(extra)]
        indices = row_ids + [row_ids[i] for i in positions]
        rng.shuffle(indices)
        result.append(indices)
    return result


def custom_cases(args: argparse.Namespace) -> list[Case]:
    def expand(text: str) -> tuple[int, ...]:
        values = tuple(int(v) for v in text.split(","))
        return values * args.features if len(values) == 1 else values

    rows, dims, unique = (
        expand(args.input_rows),
        expand(args.input_columns),
        expand(args.unique_rows),
    )
    if any(len(values) != args.features for values in (rows, dims, unique)):
        raise ValueError("Custom lists must have one entry or --features entries")
    cases = [
        Case(
            "custom",
            args.batch_size,
            rows,
            dims,
            unique,
            args.distribution,
            args.index_layout,
            args.dtype,
            args.hot_fraction,
        )
    ]
    return cases


def standard_cases() -> list[Case]:
    base = Case("baseline", 2048, (512,) * 8, (64,) * 8, (256,) * 8)
    cases = [base]
    for batch in (16, 512, 4096, 11837, 32768):
        cases.append(
            replace(
                base,
                name=f"batch_{batch}",
                batch_size=batch,
                unique_rows=(min(256, batch),) * 8,
            )
        )
    for features in (1, 4, 28, 64):
        cases.append(
            replace(
                base,
                name=f"features_{features}",
                input_rows=(512,) * features,
                input_columns=(64,) * features,
                unique_rows=(256,) * features,
            )
        )
    for dim in (16, 32, 128, 256):
        cases.append(replace(base, name=f"width_{dim}", input_columns=(dim,) * 8))
    for unique in (1, 16, 512):
        cases.append(replace(base, name=f"unique_{unique}", unique_rows=(unique,) * 8))
    for distribution in ("hot", "zipf"):
        cases.append(replace(base, name=distribution, distribution=distribution))
    for rows in (2048, 8192, 82192):
        for layout in ("compact", "spread"):
            cases.append(
                replace(
                    base,
                    name=f"stored_{rows}_{layout}",
                    input_rows=(rows,) * 8,
                    index_layout=layout,
                )
            )
    cases.extend(
        [
            replace(
                base,
                name="mixed_rows_and_widths",
                input_rows=(16, 72, 2712, 82192) * 2,
                input_columns=(16, 32, 64, 128) * 2,
                unique_rows=(16, 32, 256, 512) * 2,
                index_layout="spread",
            ),
            replace(base, name="one_hot_feature", unique_rows=(1,) + (256,) * 7),
            replace(base, name="bf16", dtype="bfloat16"),
        ]
    )
    return cases


def make_cases(args: argparse.Namespace) -> list[Case]:
    if args.suite == "custom":
        cases = custom_cases(args)
    elif args.suite == "smoke":
        cases = [
            Case("balanced", 32, (8, 16), (16, 32), (8, 8)),
            Case("hot_row", 64, (16,), (16,), (4,), "hot"),
            Case("retained_tail", 17, (32, 16), (16, 32), (8, 8), index_layout="tail"),
        ]
    else:
        cases = standard_cases()
    cases = [
        case
        for case in cases
        if not args.case
        or any(fnmatch.fnmatchcase(case.name, pattern) for pattern in args.case)
    ]
    if not cases:
        raise ValueError("No cases matched --case")
    for case in cases:
        case.validate()
    return cases


def describe(case: Case, indices: list[list[int]]) -> dict[str, Any]:
    counts = [Counter(feature) for feature in indices]
    element_bytes = 4 if case.dtype == "float32" else 2
    return {
        **asdict(case),
        "features": len(case.input_rows),
        "index_elements": case.batch_size * len(case.input_rows),
        "forward_output_elements": case.batch_size * sum(case.input_columns),
        "input_gradient_elements": sum(
            r * d for r, d in zip(case.input_rows, case.input_columns)
        ),
        "input_gradient_bytes": element_bytes
        * sum(r * d for r, d in zip(case.input_rows, case.input_columns)),
        "unused_rows": [r - len(c) for r, c in zip(case.input_rows, counts)],
        "largest_duplicate_group": [max(c.values()) for c in counts],
        "maximum_referenced_row": [max(c) for c in counts],
    }


def check_grouping(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    case: Case,
) -> None:
    import torch

    grad_rows, offsets, group_row_ids = (value.cpu() for value in output)
    features = len(case.input_rows)
    expected_rows = indices.view(features, case.batch_size).argsort(dim=1, stable=True)
    torch.testing.assert_close(grad_rows.to(torch.int64), expected_rows)
    if offsets.shape != (features, case.batch_size + 1):
        raise AssertionError("Grouping offsets must have B + 1 entries per feature")
    if group_row_ids.shape != (features, case.batch_size):
        raise AssertionError("Grouping row IDs must have B entries per feature")
    for f, feature_indices in enumerate(indices.view(features, case.batch_size)):
        rows, counts = torch.unique(feature_indices, sorted=True, return_counts=True)
        counts = torch.cat(
            (counts, torch.zeros(case.batch_size - counts.numel(), dtype=counts.dtype))
        )
        expected_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int64), counts.cumsum(0))
        )
        torch.testing.assert_close(offsets[f].to(torch.int64), expected_offsets)
        torch.testing.assert_close(
            group_row_ids[f, : rows.numel()].to(torch.int64), rows
        )


def measure(
    call: Callable[[], Any],
    synchronize: Callable[[Any], None],
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, float]:
    for _ in range(warmup):
        synchronize(call())
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(iterations):
            synchronize(call())
        times.append((time.perf_counter() - start) * 1000 / iterations)
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "trials": trials,
        "iterations_per_trial": iterations,
    }


def run_case(
    case: Case,
    indices_list: list[list[int]],
    args: argparse.Namespace,
    group_op: Callable[..., Any],
    synchronize: Callable[[Any], None],
) -> list[dict[str, Any]]:
    import torch
    from torchrec.experimental.torch_tpu.pallas import impl

    dtype = getattr(torch, case.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    sizes = [r * d for r, d in zip(case.input_rows, case.input_columns)]
    inputs = torch.randn(sum(sizes), generator=generator).to(dtype)
    indices = torch.tensor(indices_list, dtype=torch.int64).flatten()
    inverse = indices.view(len(case.input_rows), case.batch_size)
    gathered = torch.cat(
        [
            feature.view(r, d).index_select(0, index)
            for feature, r, d, index in zip(
                inputs.split(sizes), case.input_rows, case.input_columns, inverse
            )
        ],
        dim=1,
    ).flatten()
    grad_output = torch.randn(gathered.shape, generator=generator).to(dtype)
    grad_features = grad_output.view(case.batch_size, -1).split(
        case.input_columns, dim=1
    )
    expected_grad = torch.cat(
        [
            torch.zeros(r, d, dtype=torch.float32)
            .index_add_(0, index, grad.float())
            .to(dtype)
            .flatten()
            for r, d, index, grad in zip(
                case.input_rows, case.input_columns, inverse, grad_features
            )
        ]
    )
    inputs_tpu, indices_tpu, grad_tpu = (
        inputs.to("tpu"),
        indices.to("tpu"),
        grad_output.to("tpu"),
    )
    grouping_indices = indices_tpu.to(torch.int32)
    for value in (inputs_tpu, indices_tpu, grad_tpu, grouping_indices):
        synchronize(value)
    rows, dims = list(case.input_rows), list(case.input_columns)
    calls = {
        "forward": lambda: impl.batch_index_select_dim0_tpu(
            inputs_tpu, indices_tpu, rows, dims
        ),
        "backward": lambda: impl.batch_index_select_dim0_backward_tpu_impl(
            grad_tpu, indices_tpu, rows, dims, sum(sizes)
        ),
        "grouping": lambda: group_op(
            indices=grouping_indices, num_features=len(rows), batch_size=case.batch_size
        ),
        "zero": lambda: torch.zeros_like(inputs_tpu),
    }
    records = []
    rtol = (
        args.rtol
        if args.rtol is not None
        else (1e-4 if dtype == torch.float32 else 0.03)
    )
    atol = (
        args.atol
        if args.atol is not None
        else (1e-4 if dtype == torch.float32 else 0.05)
    )
    for operation in args.operations:
        record = {
            **describe(case, indices_list),
            "operation": operation,
            "rtol": rtol,
            "atol": atol,
        }
        try:
            start = time.perf_counter()
            output = calls[operation]()
            synchronize(output)
            record["first_call_ms"] = (time.perf_counter() - start) * 1000
            if operation == "grouping":
                check_grouping(
                    cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], output),
                    indices,
                    case,
                )
            else:
                expected = {
                    "forward": gathered,
                    "backward": expected_grad,
                    "zero": torch.zeros_like(inputs),
                }[operation]
                actual = output.cpu()
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                if operation == "backward":
                    for gradient, rows_count, dim, feature_indices in zip(
                        actual.split(sizes),
                        case.input_rows,
                        case.input_columns,
                        inverse,
                    ):
                        used = torch.zeros(rows_count, dtype=torch.bool)
                        used[feature_indices] = True
                        if gradient.view(rows_count, dim)[~used].count_nonzero().item():
                            raise AssertionError(
                                "Unused stored rows have nonzero gradients"
                            )
            record.update(
                measure(
                    calls[operation],
                    synchronize,
                    args.warmup,
                    args.iterations,
                    args.trials,
                )
            )
            record["status"] = "ok"
        except (AssertionError, RuntimeError, ValueError) as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite", choices=("smoke", "standard", "custom"), default="standard"
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Select case names; shell-quoted wildcard patterns allowed",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("forward", "backward", "grouping", "zero"),
        default=["grouping", "forward", "backward", "zero"],
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument(
        "--input-rows",
        default="512",
        help="Custom stored-row counts, comma-separated or one for all features",
    )
    parser.add_argument(
        "--input-columns",
        default="64",
        help="Custom local widths, comma-separated or one for all features",
    )
    parser.add_argument(
        "--unique-rows",
        default="256",
        help="Custom referenced-unique counts, comma-separated or one for all features",
    )
    parser.add_argument(
        "--distribution", choices=("uniform", "hot", "zipf"), default="uniform"
    )
    parser.add_argument(
        "--index-layout", choices=("compact", "spread", "tail"), default="compact"
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Custom input/gradient dtype",
    )
    parser.add_argument(
        "--hot-fraction",
        type=float,
        default=0.9,
        help="Fraction of extra hot-distribution references assigned to one row",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/alitehrani/batch_index_benchmark.jsonl"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print workload descriptions without importing Torch or initializing TPU",
    )
    args = parser.parse_args()
    if args.warmup < 0 or min(args.iterations, args.trials, args.features) < 1:
        parser.error(
            "Warmup must be nonnegative; iterations, trials and features must be positive"
        )
    if any(
        value is not None and (not math.isfinite(value) or value < 0)
        for value in (args.rtol, args.atol)
    ):
        parser.error("Tolerances must be finite and nonnegative")
    output = args.output.resolve()
    if not output.is_relative_to(Path("/tmp/alitehrani")):
        parser.error("Store benchmark artifacts under /tmp/alitehrani/")
    args.output = output
    return args


def main() -> None:
    args = parse_args()
    cases = make_cases(args)
    if args.dry_run:
        for case in cases:
            print(json.dumps(describe(case, make_indices(case, args.seed))))
        return

    import torch
    import torch.distributed as dist
    import torch_tpu  # pyre-ignore[21]: Available in the TPU environment.
    from torch_tpu._internal import pallas, sync  # pyre-ignore[21]
    from torchrec.experimental.torch_tpu.pallas import batch_index

    _ = torch_tpu
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    try:
        kernel_sha256 = hashlib.sha256(
            Path(batch_index.__file__).read_bytes()
        ).hexdigest()
    except OSError:
        kernel_sha256 = None
    group_op = pallas.jax_op(
        "torchrec_pallas_bench::group_batches_by_unique",
        batch_index.group_batches_by_unique,
    )

    def synchronize(output: Any) -> None:
        for tensor in output if isinstance(output, tuple) else (output,):
            sync.synchronize(tensor, wait=True)

    output_path = args.output.with_name(
        f"{args.output.stem}.rank{rank}{args.output.suffix}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    try:
        with output_path.open("w") as output_file, torch.no_grad():
            for case in cases:
                records = run_case(
                    case, make_indices(case, args.seed), args, group_op, synchronize
                )
                for record in records:
                    record.update(
                        rank=rank,
                        world_size=dist.get_world_size(),
                        torch_version=torch.__version__,
                        timing="synchronized_wall_clock",
                        kernel_sha256=kernel_sha256,
                        seed=args.seed,
                        warmup=args.warmup,
                    )
                    line = json.dumps(record)
                    output_file.write(line + "\n")
                    output_file.flush()
                    print(line, flush=True)
                    failures += record["status"] != "ok"
    finally:
        dist.destroy_process_group()
    print(f"Results: {output_path}; failed checks: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
