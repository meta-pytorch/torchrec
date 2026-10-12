#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from __future__ import annotations

import unittest
from typing import Any, List
from unittest.mock import MagicMock

import torch
from torchrec.modules.embedding_pipeline import pipeline_forward
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _MockAwaitable:
    def __init__(self, value: Any) -> None:
        self._value = value

    def wait(self) -> Any:
        return self._value


class _TrackedShardedModule(torch.nn.Module):
    def __init__(self, name: str, output: Any, call_order: List[str]) -> None:
        super().__init__()
        self._name = name
        self._output = output
        self._call_order = call_order

    def create_context(self) -> Any:
        return MagicMock()

    def input_dist(self, ctx: Any, *args: Any, **kwargs: Any) -> _MockAwaitable:
        self._call_order.append(f"{self._name}.input_dist")
        return _MockAwaitable(_MockAwaitable("dist_input"))

    def compute_and_output_dist(self, ctx: Any, dist_input: Any) -> Any:
        self._call_order.append(f"{self._name}.compute_and_output_dist")
        return self._output


class TestPipelineForward(unittest.TestCase):
    def test_empty_input(self) -> None:
        results = pipeline_forward({})
        self.assertEqual(results, {})

    def test_non_sharded_only(self) -> None:
        module_a = MagicMock(spec=torch.nn.Module)
        module_a.return_value = "output_a"
        module_b = MagicMock(spec=torch.nn.Module)
        module_b.return_value = "output_b"

        kjt_a = MagicMock(spec=KeyedJaggedTensor)
        kjt_b = MagicMock(spec=KeyedJaggedTensor)

        results = pipeline_forward(
            {
                "a": (module_a, kjt_a),
                "b": (module_b, kjt_b),
            }
        )

        self.assertEqual(results["a"], "output_a")
        self.assertEqual(results["b"], "output_b")
        module_a.assert_called_once_with(kjt_a)
        module_b.assert_called_once_with(kjt_b)

    def test_sharded_pipeline_order(self) -> None:
        from torchrec.distributed.types import ShardedModule

        call_order: List[str] = []
        mod_a = _TrackedShardedModule("a", "result_a", call_order)
        mod_b = _TrackedShardedModule("b", "result_b", call_order)

        ShardedModule.register(type(mod_a))

        kjt = MagicMock(spec=KeyedJaggedTensor)

        results = pipeline_forward(
            {
                "a": (mod_a, kjt),
                "b": (mod_b, kjt),
            }
        )

        self.assertEqual(results["a"], "result_a")
        self.assertEqual(results["b"], "result_b")

        input_dist_indices = [i for i, c in enumerate(call_order) if "input_dist" in c]
        compute_indices = [
            i for i, c in enumerate(call_order) if "compute_and_output_dist" in c
        ]
        self.assertTrue(
            max(input_dist_indices) < min(compute_indices),
            f"All input_dist should complete before any compute: {call_order}",
        )

    def test_mixed_sharded_and_non_sharded(self) -> None:
        from torchrec.distributed.types import ShardedModule

        call_order: List[str] = []
        sharded_mod = _TrackedShardedModule("sharded", "sharded_result", call_order)
        ShardedModule.register(type(sharded_mod))

        non_sharded_mod = MagicMock(spec=torch.nn.Module)
        non_sharded_mod.return_value = "direct_result"

        kjt = MagicMock(spec=KeyedJaggedTensor)

        results = pipeline_forward(
            {
                "non_sharded": (non_sharded_mod, kjt),
                "sharded": (sharded_mod, kjt),
            }
        )

        self.assertEqual(results["non_sharded"], "direct_result")
        self.assertEqual(results["sharded"], "sharded_result")
        non_sharded_mod.assert_called_once_with(kjt)

    def test_result_keys_match_input_keys(self) -> None:
        module = MagicMock(spec=torch.nn.Module)
        module.return_value = "output"
        kjt = MagicMock(spec=KeyedJaggedTensor)

        results = pipeline_forward(
            {
                "foo": (module, kjt),
                "bar": (module, kjt),
            }
        )

        self.assertEqual(set(results.keys()), {"foo", "bar"})
