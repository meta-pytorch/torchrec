#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import random
import unittest
from typing import Any, Callable, List, Sequence

import torch
from torchrec.sparse.jagged_tensor import KeyedTensor


def build_kts(
    dense_features: int,
    sparse_features: int,
    dim_dense: int,
    dim_sparse: int,
    batch_size: int,
    device: torch.device,
    run_backward: bool,
) -> List[KeyedTensor]:
    key_dim = 1
    dense_embs = [
        torch.randn(batch_size, dim_dense, device=device, requires_grad=run_backward)
        for i in range(dense_features)
    ]
    dense_keys = [f"dense_{i}" for i in range(dense_features)]
    dense_kt = KeyedTensor.from_tensor_list(dense_keys, dense_embs, key_dim)

    sparse_embs = [
        torch.randn(batch_size, dim_sparse, device=device, requires_grad=run_backward)
        for i in range(sparse_features)
    ]
    sparse_keys = [f"sparse_{i}" for i in range(sparse_features)]
    sparse_kt = KeyedTensor.from_tensor_list(sparse_keys, sparse_embs, key_dim)
    return [dense_kt, sparse_kt]


def build_groups(
    kts: List[KeyedTensor],
    num_groups: int,
    skips: bool = False,
    duplicates: bool = False,
) -> List[List[str]]:
    all_keys = []
    for kt in kts:
        all_keys.extend(kt.keys())
    allocation = [random.randint(0, num_groups - 1) for _ in range(len(all_keys))]
    groups = [[] for _ in range(num_groups)]
    for i, key in enumerate(allocation):
        groups[key].append(all_keys[i])
    if skips:
        for group in groups:
            if len(group) > 1:
                group.pop(random.randint(0, len(group) - 1))
    if duplicates:
        for group in groups:
            group.append(random.choice(all_keys))
    return groups


def repeat_test(
    *args: List[Any], **kwargs: Sequence[Any]
) -> Callable[..., Callable[..., None]]:
    def decorate(f: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(f)
        def decorator(self: unittest.TestCase) -> None:
            queue = [(arg, {}) for arg in args] if args else [((), {})]
            # pyrefly: ignore[bad-assignment]
            for k, values in kwargs.items():
                new_queue = []
                for a, d in queue:
                    for v in values:
                        new_d = d.copy()
                        new_d[k] = v
                        new_queue.append((a, new_d))
                queue = new_queue
            for a, d in queue:
                print(f"running {f.__name__} {a} {d}")
                f(self, *a, **d)

        return decorator

    return decorate
