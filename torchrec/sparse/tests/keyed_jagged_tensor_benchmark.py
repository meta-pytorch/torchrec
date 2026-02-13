#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import sys

import click

# Need this for PT2 compile
# Otherwise will get error
# NotImplementedError: fbgemm::permute_1D_sparse_data: We could not find the abstract impl for this operator.
from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from torchrec.sparse.tests.keyed_jagged_tensor_benchmark_lib import (
    bench,
    DEFTAULT_BENCHMARK_FUNCS,
    TransformType,
)

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", stream=sys.stdout)
logger.setLevel(logging.DEBUG)


@click.command()
@click.option(
    "--num-repeat",
    default=20,
    help="Number of times method under test is run",
)
@click.option(
    "--num-warmup",
    default=10,
    help="Number of times method under test is run for warmup",
)
@click.option(
    "--num-features",
    default=128,
    help="Total number of sparse features per KJT",
)
@click.option(
    "--batch-size",
    default=4096,
    help="Batch size per KJT (assumes non-VBE)",
)
@click.option(
    "--mean-pooling-factor",
    default=100,
    help="Avg pooling factor for KJT",
)
@click.option(
    "--num-workers",
    default=4,
    help="World size to simulate for dist_init",
)
@click.option(
    "--test-pt2/--no-test-pt2",
    type=bool,
    default=False,
    help="Whether to benchmark PT2 Eager",
)
@click.option(
    "--kjt-funcs",
    type=str,
    default=",".join(DEFTAULT_BENCHMARK_FUNCS),
    help="kjt functions to benchmark",
)
@click.option(
    "--run-modes",
    type=str,
    default=",".join([member.name for member in TransformType]),
    help="kjt functions to benchmark",
)
def main(
    num_repeat: int,
    num_warmup: int,
    num_features: int,
    batch_size: int,
    mean_pooling_factor: int,
    num_workers: int,
    test_pt2: bool,
    kjt_funcs: str,
    run_modes: str,
) -> None:
    bench(
        num_repeat,
        num_warmup,
        num_features,
        batch_size,
        mean_pooling_factor,
        num_workers,
        test_pt2,
        kjt_funcs,
        run_modes,
    )


if __name__ == "__main__":
    main()
