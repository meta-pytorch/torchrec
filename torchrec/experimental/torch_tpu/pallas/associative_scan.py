#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp


@jax.jit
def complete_cumsum_jax(array: jax.Array) -> jax.Array:
    """Return the exact int32 exclusive-zero/inclusive-prefix offsets."""
    # TPU callers pass nonnegative lengths whose sum is the indexed values count;
    # TPU sparse kernels use int32 offsets, so valid inputs cannot exceed INT32_MAX.
    # TODO: Test support for int64, or look into adding overflow condition outside.
    array = array.astype(jnp.int32)
    if array.shape[0] == 0:
        return jnp.zeros((1,), dtype=jnp.int32)
    prefix = jax.lax.associative_scan(jax.lax.add, array)
    return jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), prefix])
