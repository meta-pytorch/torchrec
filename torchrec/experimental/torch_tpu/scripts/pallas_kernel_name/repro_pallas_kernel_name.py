#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reproduce Pallas inner-kernel versus TorchTPU module naming."""

# @lint-ignore-every AUTODEPS2 Standalone repro runs directly on a TPU pod.
import sys

import jax
import torch
import torch_tpu  # noqa: F401  (registers the TPU device)
from jax.experimental import pallas as pl
from torch_tpu._internal import pallas, profiler, sync


ADD_INNER_NAME = "named_pallas_add"
ADD_OUTER_NAME = "pallas_name_repro::add"
MULTIPLY_INNER_NAME = "named_pallas_multiply"
MULTIPLY_OUTER_NAME = "pallas_name_repro::multiply"


def _add_kernel(x_ref, y_ref, out_ref) -> None:
    out_ref[...] = x_ref[...] + y_ref[...]


def _multiply_kernel(x_ref, y_ref, out_ref) -> None:
    out_ref[...] = x_ref[...] * y_ref[...]


@pallas.jax_op(ADD_OUTER_NAME)
def named_add(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        _add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        name=ADD_INNER_NAME,
    )(x, y)


@pallas.jax_op(MULTIPLY_OUTER_NAME)
def named_multiply(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        _multiply_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        name=MULTIPLY_INNER_NAME,
    )(x, y)


def main(trace_dir: str) -> None:
    x = torch.ones(128, dtype=torch.float32, device="tpu")
    y = torch.full((128,), 2.0, dtype=torch.float32, device="tpu")

    add_output = named_add(x, y)
    multiply_output = named_multiply(x, y)
    sync.synchronize(add_output, wait=True)
    sync.synchronize(multiply_output, wait=True)

    print(
        "Add MLIR module:",
        sync.computation_mlir(add_output).splitlines()[0],
    )
    print(
        "Multiply MLIR module:",
        sync.computation_mlir(multiply_output).splitlines()[0],
    )
    print("Inner Pallas names:", ADD_INNER_NAME, MULTIPLY_INNER_NAME)
    print("Outer Torch operation names:", ADD_OUTER_NAME, MULTIPLY_OUTER_NAME)

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.TPU,
        ],
        on_trace_ready=profiler.xprof_trace_handler(dir_name=trace_dir),
    ):
        add_output = named_add(x, y)
        sync.synchronize(add_output, wait=True)
        multiply_output = named_multiply(x, y)
        sync.synchronize(multiply_output, wait=True)

    torch.testing.assert_close(add_output.cpu(), torch.full((128,), 3.0))
    torch.testing.assert_close(multiply_output.cpu(), torch.full((128,), 2.0))
    print("Trace written to", trace_dir)
    print("Inspect XLA Modules for both `tt_jit_custom_kernel` events.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} TRACE_DIR")
    main(sys.argv[1])
