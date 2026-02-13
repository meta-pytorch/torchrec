#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

import torch
from hypothesis import assume, given, settings, strategies as st, Verbosity
from tensordict import TensorDict
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt


class TestTensorDict(unittest.TestCase):
    @given(device_str=st.sampled_from(["cpu", "meta", "cuda"]))
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_kjt_input(self, device_str: str) -> None:
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)
        values = torch.tensor([0, 1, 2, 3, 2, 3, 4], device=device)
        kjt = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=values,
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 7], device=device),
        )
        features = maybe_td_to_kjt(kjt)
        self.assertEqual(features, kjt)

    @given(device_str=st.sampled_from(["cpu", "meta", "cuda"]))
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_td_kjt(self, device_str: str) -> None:
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)
        values = torch.tensor([0, 1, 2, 3, 2, 3, 4], device=device)
        lengths = torch.tensor([2, 0, 1, 1, 1, 2], device=device)
        data = {
            "f2": torch.nested.nested_tensor_from_jagged(
                torch.tensor([2, 3], device=device),
                lengths=torch.tensor([1, 1], device=device),
            ),
            "f1": torch.nested.nested_tensor_from_jagged(
                torch.arange(2, device=device),
                offsets=torch.tensor([0, 2, 2], device=device),
            ),
            "f3": torch.nested.nested_tensor_from_jagged(
                torch.tensor([2, 3, 4], device=device),
                lengths=torch.tensor([1, 2], device=device),
            ),
        }
        # pyrefly: ignore[bad-specialization]
        td = TensorDict(
            data,  # type: ignore[arg-type]
            device=device,
            batch_size=[2],
        )

        # pyrefly: ignore[bad-argument-type]
        features = maybe_td_to_kjt(td, ["f1", "f2", "f3"])
        torch.testing.assert_close(features.values(), values)
        torch.testing.assert_close(features.lengths(), lengths)
