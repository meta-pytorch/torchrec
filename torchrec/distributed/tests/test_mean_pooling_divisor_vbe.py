#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
from torchrec.distributed.embeddingbag import _create_mean_pooling_divisor
from torchrec.modules.embedding_configs import PoolingType

BATCH = 4
DIMS: dict[str, int] = {"f_a": 3, "f_b": 2}
# Variable batch: f_a has one row per sample, f_b one row shared by all samples.
STRIDES: dict[str, int] = {"f_a": 4, "f_b": 1}
INVERSE: dict[str, list[int]] = {"f_a": [0, 1, 2, 3], "f_b": [0, 0, 0, 0]}
LENGTHS: dict[str, list[int]] = {"f_a": [1, 2, 3, 4], "f_b": [5]}


def _divisor(
    input_keys: list[str],
    sharded_keys: list[str],
    inverse_keys_as_tuple: bool = False,
) -> torch.Tensor:
    """
    Divisor for a variable-batch KJT whose features are all RW / TWRW MEAN.
    `input_keys` is the input KJT's key order, which the inverse indices
    follow; `sharded_keys` is the order after the input dist permute.
    """
    inverse_keys = (
        cast(list[str], tuple(input_keys)) if inverse_keys_as_tuple else input_keys
    )
    lengths = torch.tensor([n for k in sharded_keys for n in LENGTHS[k]])
    key_to_inverse_index = {k: i for i, k in enumerate(input_keys)}
    return _create_mean_pooling_divisor(
        lengths=lengths,
        keys=sharded_keys,
        offsets=torch.cat([torch.zeros(1, dtype=lengths.dtype), lengths.cumsum(0)]),
        stride=BATCH,
        stride_per_key=[STRIDES[k] for k in sharded_keys],
        dim_per_key=torch.tensor([DIMS[k] for k in sharded_keys]),
        pooling_type_to_rs_features={
            PoolingType.SUM.value: [],
            PoolingType.MEAN.value: sharded_keys,
        },
        embedding_names=sharded_keys,
        embedding_dims=[DIMS[k] for k in sharded_keys],
        variable_batch_per_feature=True,
        kjt_inverse_order=torch.tensor([key_to_inverse_index[k] for k in sharded_keys]),
        kjt_key_indices={k: i for i, k in enumerate(sharded_keys)},
        kt_key_ordering=torch.arange(len(sharded_keys)),
        inverse_indices=(
            inverse_keys,
            torch.tensor([INVERSE[k] for k in input_keys]),
        ),
    )


def _expected(sharded_keys: list[str]) -> torch.Tensor:
    rows = [
        [
            float(LENGTHS[k][INVERSE[k][sample]])
            for k in sharded_keys
            for _ in range(DIMS[k])
        ]
        for sample in range(BATCH)
    ]
    return torch.tensor(rows) + 1e-6


class MeanPoolingDivisorVariableBatchTest(unittest.TestCase):
    def test_keys_not_permuted(self) -> None:
        keys = ["f_a", "f_b"]
        torch.testing.assert_close(_divisor(keys, keys), _expected(keys))

    def test_keys_not_permuted_inverse_keys_tuple(self) -> None:
        # A tuple of the same keys compares unequal to the list, so the
        # realign runs, but with an identity order it must not change anything.
        keys = ["f_a", "f_b"]
        torch.testing.assert_close(
            _divisor(keys, keys, inverse_keys_as_tuple=True), _expected(keys)
        )

    def test_keys_permuted(self) -> None:
        # Without the realign, each feature reads the other feature's lengths.
        sharded_keys = ["f_b", "f_a"]
        torch.testing.assert_close(
            _divisor(["f_a", "f_b"], sharded_keys), _expected(sharded_keys)
        )

    def test_keys_permuted_out_of_bounds(self) -> None:
        # Without the realign, f_a's inverse indices get f_b's offset and
        # index past the end of the lengths.
        sharded_keys = ["f_a", "f_b"]
        torch.testing.assert_close(
            _divisor(["f_b", "f_a"], sharded_keys), _expected(sharded_keys)
        )


if __name__ == "__main__":
    unittest.main()
