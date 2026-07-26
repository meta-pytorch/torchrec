#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def pipeline_forward(
    named_modules_and_inputs: Dict[str, Tuple[torch.nn.Module, KeyedJaggedTensor]],
) -> Dict[str, Any]:
    """
    Pipeline the forward pass of multiple embedding modules.

    For local (non-sharded) embedding modules, the function calls forward
    directly. For sharded embedding modules, it launches all ``input_dist``
    calls first to overlap the all-to-all input distribution across modules,
    then waits on the awaitables and runs ``compute_and_output_dist``.

    Args:
        named_modules_and_inputs: A mapping from a user-defined name to a
            tuple of ``(module, kjt_input)``. ``module`` must be either a
            local embedding module or a ``ShardedModule`` wrapping one;
            ``kjt_input`` is the ``KeyedJaggedTensor`` to feed into that
            module's forward pass.

    Returns:
        A dictionary mapping each name from ``named_modules_and_inputs`` to
        the corresponding module's forward output.

    Example:
        >>> from torchrec.modules.embedding_pipeline import pipeline_forward
        >>> # `ec_a` and `ebc_b` are EmbeddingCollection and EmbeddingBagCollection
        >>> # instances respectively. When the model is sharded, they become sharded
        >>> # modules. But the code below still works.
        >>> # `kjt_a` and `kjt_b` are KeyedJaggedTensor inputs.
        >>> outputs = pipeline_forward({
        ...     "ec_a": (ec_a, kjt_a),
        ...     "ebc_b": (ebc_b, kjt_b),
        ... })
        >>> pooled_a = outputs["ec_a"]
        >>> pooled_b = outputs["ebc_b"]
    """
    from torchrec.distributed.types import ShardedModule

    results: Dict[str, Any] = {}
    sharded: List[Tuple[str, ShardedModule, KeyedJaggedTensor]] = []

    for name, (module, kjt_input) in named_modules_and_inputs.items():
        if isinstance(module, ShardedModule):
            sharded.append((name, module, kjt_input))
        else:
            results[name] = module(kjt_input)

    if sharded:
        contexts = []
        input_dist_awaitables = []
        for _name, module, kjt_input in sharded:
            ctx = module.create_context()
            contexts.append(ctx)
            input_dist_awaitables.append(module.input_dist(ctx, kjt_input))

        dist_awaitables = [a.wait() for a in input_dist_awaitables]
        dist_inputs = [a.wait() for a in dist_awaitables]

        for i, (name, module, _kjt_input) in enumerate(sharded):
            results[name] = module.compute_and_output_dist(contexts[i], dist_inputs[i])

    return results
