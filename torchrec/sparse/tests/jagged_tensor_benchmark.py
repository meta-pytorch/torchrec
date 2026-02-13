#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for KeyedTensor regrouping operations.

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/sparse/tests:jagged_tensor_benchmark -- --batch_size=1024 --n_dense=20 --n_sparse=1000

OSS (external):
    python -m torchrec.sparse.tests.jagged_tensor_benchmark --batch_size=1024 --n_dense=20 --n_sparse=1000
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List

import torch
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
)
from torchrec.modules.regroup import KTRegroupAsDict
from torchrec.sparse.jagged_tensor import (
    _fbgemm_permute_pooled_embs,
    _regroup_keyed_tensors,
    KeyedTensor,
    permute_multi_embedding,
    regroup_kts,
)
from torchrec.sparse.tests.utils import build_groups, build_kts

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RunOptions(BenchFuncConfig):
    """
    Configuration options for running KeyedTensor regrouping benchmarks.

    This class defines the parameters that control how the benchmark is executed,
    including tensor dimensions, batch configuration, and profiling options.

    Args:
        batch_size (int): Batch size for the benchmark. Default is 1024.
        n_dense (int): Total number of dense embeddings. Default is 20.
        n_sparse (int): Total number of sparse embeddings. Default is 1000.
        dim_dense (int): Dimension of dense embeddings. Default is 64.
        dim_sparse (int): Dimension of sparse embeddings. Default is 128.
        n_groups (int): Total number of regrouping groups. Default is 2.
        run_backward (bool): Whether to run backward pass. Default is False.
        profile_dir (str): Directory to save profiling results. If empty, profiling is disabled.
            Default is "" (disabled).
        name (str): Name of the profiling file. Default is "jagged_tensor_benchmark".
    """

    ALL_NAMES: List[str] = field(
        default_factory=lambda: [
            "torch_generic",
            "kt.regroup",
            "regroup_module",
            "permute_multi_embs",
            "regroup_kts",
            "permute_pooled_embs",
        ],
        repr=False,
    )

    name: str = "all"

    batch_size: int = 1024
    n_dense: int = 20
    n_sparse: int = 1000
    dim_dense: int = 64
    dim_sparse: int = 128
    n_groups: int = 2
    run_backward: bool = False

    # Override defaults from BenchFuncConfig
    world_size: int = 1
    num_benchmarks: int = 20
    num_profiles: int = 10
    device_type: str = "cuda"
    debug_mode: bool = False
    profile_dir: str = "."

    duplicates: bool = False
    fn: Callable[..., List[torch.Tensor]] | None = None

    _iter_index: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        match self.name:
            case "torch_generic":
                self.fn = _regroup_keyed_tensors
            case "kt.regroup":
                self.fn = KeyedTensor.regroup
            case "regroup_module":

                module: torch.nn.Module | None = None

                def regroup_module(
                    keyed_tensors: List[KeyedTensor],
                    groups: List[List[str]],
                ) -> List[torch.Tensor]:
                    nonlocal module
                    if module is None:
                        module = KTRegroupAsDict(
                            groups=groups, keys=[str(i) for i in range(self.n_groups)]
                        )
                    return list(module.forward(keyed_tensors).values())

                self.fn = regroup_module
            case "permute_multi_embs":
                self.fn = permute_multi_embedding
            case "regroup_kts":
                self.fn = regroup_kts
            case "permute_pooled_embs":
                if self.duplicates:
                    self.fn = _regroup_keyed_tensors
                    self.name = "torch_generic_fallback"
                else:
                    self.fn = _fbgemm_permute_pooled_embs
            case "all":
                pass
            case _:
                raise ValueError(f"Unknown code name: {self.name}")
        if self.name != "all":
            self.name += "_dup" if self.duplicates else ""

    def __iter__(self) -> Iterator["RunOptions"]:
        """
        Iterate over benchmark configurations.

        If code_name is "all", yields a new RunOptions for each benchmark type.
        Otherwise, yields only self.

        Returns:
            Iterator of RunOptions objects.
        """
        self._iter_index = 0
        return self

    def __next__(self) -> "RunOptions":
        """
        Get the next benchmark configuration.

        Returns:
            Next RunOptions object with a specific code_name.

        Raises:
            StopIteration: When all configurations have been yielded.
        """
        if self.name != "all":
            if self._iter_index == 0:
                self._iter_index += 1
                return self
            raise StopIteration

        if self._iter_index >= len(self.ALL_NAMES):
            raise StopIteration

        name = self.ALL_NAMES[self._iter_index]
        self._iter_index += 1

        new_option = copy.copy(self)
        new_option.name = name
        new_option.__post_init__()
        return new_option

    def build_kts(
        self,
    ) -> List[KeyedTensor]:
        """
        Build KeyedTensors for benchmarking.

        Creates a list of KeyedTensors with the configured dimensions,
        batch size, and device settings.

        Returns:
            List[KeyedTensor]: List of KeyedTensors for benchmarking.
        """
        device = torch.device(self.device_type)
        return build_kts(
            self.n_dense,
            self.n_sparse,
            self.dim_dense,
            self.dim_sparse,
            self.batch_size,
            device,
            self.run_backward,
        )


def runner(
    run_option: RunOptions,
) -> BenchmarkResult:
    """
    Run benchmark for a single configuration.

    Args:
        run_option: Run options containing benchmark configuration.

    Returns:
        BenchmarkResult object for the benchmark.
    """
    device = torch.device(run_option.device_type)
    kts = run_option.build_kts()
    labels = torch.randint(0, 1, (run_option.batch_size,), device=device).float()
    groups = build_groups(kts, run_option.n_groups, duplicates=run_option.duplicates)

    fn = run_option.fn
    fn_kwargs = {"keyed_tensors": kts, "groups": groups}

    # Initial call to warm up
    # pyrefly: ignore[not-callable]
    fn(**fn_kwargs)

    def _func_to_benchmark(
        _bench_inputs: List[Any],
        # pyrefly: ignore[bad-function-definition]
        fn: Callable[..., List[torch.Tensor]] = fn,
        fn_kwargs: Dict[str, Any] = fn_kwargs,
        run_backward: bool = run_option.run_backward,
        labels: torch.Tensor = labels,
    ) -> None:
        result = fn(**fn_kwargs)
        if run_backward:
            vectors = [tensor.reshape(1, -1) for tensor in result]
            loss = torch.concat(vectors, dim=1).sum()
            loss.backward()

    result = benchmark_func(
        rank=0,
        func_to_benchmark=_func_to_benchmark,
        bench_inputs=[{}],
        prof_inputs=[{}] * run_option.num_profiles,
        benchmark_func_kwargs={},
        **run_option.benchmark_func_kwargs(),
    )

    logger.info(result.prettify())
    logger.info("\nMarkdown format:\n%s", result)

    return result


@cmd_conf
def main(
    run_option: RunOptions,
) -> None:
    """
    Main entry point for the jagged tensor benchmark.

    Args:
        run_option: Configuration options for the benchmark.
    """
    run_option.set_log_level()
    if run_option.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    results: List[BenchmarkResult] = []
    for option in run_option:
        results.append(runner(option))

    print(BenchmarkResult.print_table(results))


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
