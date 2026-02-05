#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import os
import unittest
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.distributed._shard import sharded_tensor, sharding_spec
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    KeyedOptimizerWrapper,
    OptimizerWrapper,
)
from torchrec.test_utils import get_free_port


class DummyOptimizerModule:
    def __init__(
        self,
        tensor: torch.Tensor,
    ) -> None:
        self.tensor = tensor

    def state_dict(self) -> Dict[str, Any]:
        return {"tensor": self.tensor}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.tensor.detach().copy_(state_dict["tensor"])


class TestKeyedOptimizer(unittest.TestCase):
    def _assert_state_dict_equals(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> None:
        self.assertEqual(dict1["param_groups"], dict2["param_groups"])
        self.assertEqual(
            dict1["state"]["param_2"],
            dict2["state"]["param_2"],
        )
        torch.testing.assert_close(
            dict1["state"]["param_1"]["tensor"],
            dict2["state"]["param_1"]["tensor"],
        )
        torch.testing.assert_close(
            dict1["state"]["param_1"]["nested_dictionary"]["tensor"],
            dict2["state"]["param_1"]["nested_dictionary"]["tensor"],
        )
        torch.testing.assert_close(
            dict1["state"]["param_1"]["optimizer_module"]["tensor"],
            dict2["state"]["param_1"]["optimizer_module"]["tensor"],
        )

        torch.testing.assert_close(
            dict1["state"]["param_1"]["sharded_tensor"].local_shards()[0].tensor,
            dict2["state"]["param_1"]["sharded_tensor"].local_shards()[0].tensor,
        )

    def test_load_state_dict(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        # Set up example KeyedOptimizer.
        param_1_t, param_2_t = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        param_1, param_2 = Variable(param_1_t), Variable(param_2_t)
        keyed_optimizer = KeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {
                param_1: {
                    "one": 1.0,
                    "tensor": torch.tensor([5.0, 6.0]),
                    "sharded_tensor": sharded_tensor.full(
                        sharding_spec.ChunkShardingSpec(
                            dim=0, placements=["rank:0/cpu"]
                        ),
                        (4,),
                        fill_value=1.0,
                    ),
                    "nested_dictionary": {
                        "tensor": torch.tensor([7.0, 8.0]),
                    },
                    "optimizer_module": DummyOptimizerModule(torch.tensor([9.0, 10.0])),
                },
                param_2: {"two": 2.0},
            },
            [
                {
                    "params": [param_1],
                    "param_group_val_0": 3.0,
                    "param_group_val_1": 4.0,
                },
                {
                    "params": [param_2],
                    "param_group_val_0": 5.0,
                    "param_group_val_1": 6.0,
                },
            ],
        )
        keyed_optimizer.save_param_groups(True)

        # Assert state_dict is as expected.
        state: Dict[str, Any] = {
            "param_1": {
                "one": 1.0,
                "tensor": torch.tensor([5.0, 6.0]),
                "sharded_tensor": sharded_tensor.full(
                    sharding_spec.ChunkShardingSpec(dim=0, placements=["rank:0/cpu"]),
                    (4,),
                    fill_value=1.0,
                ),
                "nested_dictionary": {
                    "tensor": torch.tensor([7.0, 8.0]),
                },
                "optimizer_module": {
                    "tensor": torch.tensor([9.0, 10.0]),
                },
            },
            "param_2": {"two": 2.0},
        }
        param_groups: List[Dict[str, Any]] = [
            {
                "params": ["param_1"],
                "param_group_val_0": 3.0,
                "param_group_val_1": 4.0,
            },
            {
                "params": ["param_2"],
                "param_group_val_0": 5.0,
                "param_group_val_1": 6.0,
            },
        ]
        expected_state_dict = {
            "state": state,
            "param_groups": param_groups,
        }
        self._assert_state_dict_equals(
            expected_state_dict, keyed_optimizer.state_dict()
        )

        # Modify state dict and call load_state_dict.
        expected_state_dict["state"]["param_1"]["one"] = 10.0
        expected_state_dict["state"]["param_1"]["tensor"] = torch.tensor([50.0, 60.0])
        expected_state_dict["state"]["param_1"]["sharded_tensor"] = sharded_tensor.full(
            sharding_spec.ChunkShardingSpec(dim=0, placements=["rank:0/cpu"]),
            (4,),
            fill_value=10.0,
        )
        expected_state_dict["state"]["param_1"]["nested_dictionary"]["tensor"] = (
            torch.tensor([70.0, 80.0])
        )
        expected_state_dict["state"]["param_1"]["optimizer_module"]["tensor"] = (
            torch.tensor([90.0, 100.0])
        )
        expected_state_dict["param_groups"][0]["param_group_val_0"] = 8.0
        expected_state_dict["param_groups"][1]["param_group_val_1"] = 9.0

        keyed_optimizer.load_state_dict(expected_state_dict)
        self._assert_state_dict_equals(
            expected_state_dict, keyed_optimizer.state_dict()
        )
        dist.destroy_process_group()

    def test_non_param_state_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "All state keys must be params."):
            param_1_t = torch.tensor([1.0, 2.0])
            param_1 = Variable(param_1_t)
            KeyedOptimizer(
                {"param_1": param_1},
                {param_1: 1.0, "non_param_state_key": 2.0},
                [{"params": [param_1], "param_group_val_0": 3.0}],
            )

    def test_init_state_with_momentum(self) -> None:
        dense = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float))
        sparse = torch.nn.Parameter(torch.ones((1, 4), dtype=torch.float))
        opt = KeyedOptimizerWrapper(
            {"dense": dense, "sparse": sparse},
            lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.1),
        )
        opt.init_state({"sparse"})

        self.assertTrue(dense.grad is not None)
        self.assertFalse(dense.grad.is_sparse)
        self.assertTrue("momentum_buffer" in opt.state_dict()["state"]["dense"])

        self.assertTrue(sparse.grad is not None)
        self.assertTrue(sparse.grad.is_sparse)
        self.assertTrue("momentum_buffer" in opt.state_dict()["state"]["sparse"])

    def test_init_state_no_momentum(self) -> None:
        dense = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float))
        sparse = torch.nn.Parameter(torch.ones((1, 4), dtype=torch.float))
        opt = KeyedOptimizerWrapper(
            {"dense": dense, "sparse": sparse},
            lambda params: torch.optim.SGD(params, lr=0.1),
        )
        opt.init_state({"sparse"})

        self.assertTrue(dense.grad is not None)
        self.assertFalse(dense.grad.is_sparse)

        self.assertTrue(sparse.grad is not None)
        self.assertTrue(sparse.grad.is_sparse)

        self.assertTrue("state" in opt.state_dict())
        self.assertFalse(opt.state_dict()["state"])

    def test_pickle(self) -> None:
        dense = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float))
        sparse = torch.nn.Parameter(torch.ones((1, 4), dtype=torch.float))
        opt = KeyedOptimizerWrapper(
            {"dense": dense, "sparse": sparse},
            lambda params: torch.optim.SGD(params, lr=0.1),
        )
        opt.init_state({"sparse"})

        bytesIO = io.BytesIO()
        torch.save(opt, bytesIO)
        bytesIO.seek(0)
        reload_opt = torch.load(bytesIO, weights_only=False)

        for k in reload_opt.state_dict():
            self.assertEqual(
                opt.state_dict()[k],
                reload_opt.state_dict()[k],
            )


class TestCombinedOptimizer(unittest.TestCase):
    def test_pickle(self) -> None:
        # Set up example KeyedOptimizer 1.
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer_1 = KeyedOptimizer(
            {"param_1": param_1},
            {param_1: {"one": 1.0}},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )

        # Set up example KeyedOptimizer 2.
        param_2_t = torch.tensor([-1.0, -2.0])
        param_2 = Variable(param_2_t)
        keyed_optimizer_2 = KeyedOptimizer(
            {"param_2": param_2},
            {param_2: {"two": -1.0}},
            [{"params": [param_2], "param_group_val_0": -2.0}],
        )

        combined_optimizer = CombinedOptimizer(
            [("ko1", keyed_optimizer_1), ("", keyed_optimizer_2)]
        )

        bytesIO = io.BytesIO()
        torch.save(combined_optimizer, bytesIO)
        bytesIO.seek(0)
        reload_combined_optimizer = torch.load(bytesIO, weights_only=False)

        for k in reload_combined_optimizer.state_dict():
            self.assertEqual(
                combined_optimizer.state_dict()[k],
                reload_combined_optimizer.state_dict()[k],
            )

    def test_load_state_dict(self) -> None:
        # Set up example KeyedOptimizer 1.
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer_1 = KeyedOptimizer(
            {"param_1": param_1},
            {param_1: {"one": 1.0}},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )

        # Set up example KeyedOptimizer 2.
        param_2_t = torch.tensor([-1.0, -2.0])
        param_2 = Variable(param_2_t)
        keyed_optimizer_2 = KeyedOptimizer(
            {"param_2": param_2},
            {param_2: {"two": -1.0}},
            [{"params": [param_2], "param_group_val_0": -2.0}],
        )

        combined_optimizer = CombinedOptimizer(
            [("ko1", keyed_optimizer_1), ("", keyed_optimizer_2)]
        )
        combined_optimizer.save_param_groups(True)

        combined_optimizer_state_dict = combined_optimizer.state_dict()
        combined_optimizer_state_dict["state"]["ko1.param_1"] = {"one": 999}
        combined_optimizer_state_dict["state"]["param_2"] = {"two": 998}
        combined_optimizer_state_dict["param_groups"][0]["param_group_val_0"] = 997
        combined_optimizer_state_dict["param_groups"][1]["param_group_val_0"] = 996

        combined_optimizer.load_state_dict(combined_optimizer_state_dict)

        # Check that optimizers in the combined optimizer have their state and
        # param_groups updated.
        self.assertEqual(keyed_optimizer_1.state[param_1], {"one": 999})
        self.assertEqual(keyed_optimizer_2.state[param_2], {"two": 998})
        self.assertEqual(keyed_optimizer_1.param_groups[0]["param_group_val_0"], 997)
        self.assertEqual(keyed_optimizer_2.param_groups[0]["param_group_val_0"], 996)


class TestOptimizerWrapper(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer = KeyedOptimizer(
            {"param_1": param_1},
            {param_1: {"one": 1.0}},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )
        optimizer_wrapper = OptimizerWrapper(keyed_optimizer)
        optimizer_wrapper.save_param_groups(True)

        optimizer_wrapper_state_dict = optimizer_wrapper.state_dict()
        optimizer_wrapper_state_dict["state"]["param_1"] = {"one": 999}
        optimizer_wrapper_state_dict["param_groups"][0]["param_group_val_0"] = 998
        optimizer_wrapper.load_state_dict(optimizer_wrapper_state_dict)

        # Check that both keyed_optimizer and optimizer_wrapper have their state and
        # param_groups updated.
        self.assertEqual(keyed_optimizer.state[param_1], {"one": 999})
        self.assertEqual(optimizer_wrapper.state[param_1], {"one": 999})
        self.assertEqual(keyed_optimizer.param_groups[0]["param_group_val_0"], 998)
        self.assertEqual(optimizer_wrapper.param_groups[0]["param_group_val_0"], 998)
