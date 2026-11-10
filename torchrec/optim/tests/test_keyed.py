#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest
from copy import deepcopy
from typing import Any, Dict, List, Mapping
from unittest.mock import patch

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
from torchrec.test_utils import init_process_group_single_rank


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
    def test_legacy_checkpoint_resume(self) -> None:
        for optimizer_type in (torch.optim.SGD, torch.optim.AdamW):
            for save_groups in (False, True):
                with self.subTest(optimizer=optimizer_type, save_groups=save_groups):
                    kwargs = {"lr": 0.03, "weight_decay": 0.2}
                    if optimizer_type is torch.optim.SGD:
                        kwargs["momentum"] = 0.9
                    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                    reference = optimizer_type([p], **kwargs)
                    for step in range(4):
                        p.grad = torch.tensor([0.2 * (step + 1), -0.3])
                        reference.step()
                    saved: dict[str, Any] = {
                        "state": {"weight": deepcopy(reference.state[p])}
                    }
                    if save_groups:
                        saved["param_groups"] = [
                            {**reference.param_groups[0], "params": ["weight"]}
                        ]
                    q = torch.nn.Parameter(p.detach().clone())
                    restored = optimizer_type([q], **kwargs)
                    child = KeyedOptimizerWrapper(
                        {"weight": q}, lambda _, optimizer=restored: optimizer
                    )
                    wrapped = OptimizerWrapper(
                        CombinedOptimizer([OptimizerWrapper(child)])
                    )
                    wrapped.save_param_groups(save_groups)
                    wrapped.init_state()
                    with torch.no_grad():
                        q.copy_(p)
                    with patch.object(
                        restored, "post_load_state_dict", create=True
                    ) as legacy_hook:
                        wrapped.load_state_dict(deepcopy(saved))
                        legacy_hook.assert_not_called()
                    actual = wrapped.state_dict()
                    self.assertEqual(set(actual), set(saved))
                    self.assertEqual(
                        actual.get("param_groups"), saved.get("param_groups")
                    )
                    torch.testing.assert_close(
                        actual["state"], saved["state"], rtol=0, atol=0
                    )
                    invalid = deepcopy(saved)
                    invalid["checkpoint_metadata"] = {"unexpected": torch.tensor(1)}
                    with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
                        wrapped.load_state_dict(invalid)
                    actual = wrapped.state_dict()
                    self.assertEqual(
                        actual.get("param_groups"), saved.get("param_groups")
                    )
                    torch.testing.assert_close(
                        actual["state"], saved["state"], rtol=0, atol=0
                    )
                    for step in range(4, 9):
                        p.grad = torch.tensor([0.2 * (step + 1), -0.3])
                        q.grad = p.grad.clone()
                        reference.step()
                        wrapped.step()
                        torch.testing.assert_close(q, p, rtol=0, atol=0)
                        torch.testing.assert_close(
                            restored.state[q], reference.state[p], rtol=0, atol=0
                        )
                        self.assertNotIn("checkpoint_metadata", wrapped.state_dict())

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
        init_process_group_single_rank("gloo")

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
        # pyrefly: ignore[bad-index]
        self.assertEqual(keyed_optimizer_1.param_groups[0]["param_group_val_0"], 997)
        # pyrefly: ignore[bad-index]
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
        # pyrefly: ignore[bad-index]
        self.assertEqual(keyed_optimizer.param_groups[0]["param_group_val_0"], 998)
        # pyrefly: ignore[bad-index]
        self.assertEqual(optimizer_wrapper.param_groups[0]["param_group_val_0"], 998)


class MetadataSGD(torch.optim.SGD):
    def __init__(self, params: list[torch.Tensor], layout: int) -> None:
        super().__init__(params, lr=0.1, momentum=0.9)
        self.layout = layout
        self.loaded_momentum: torch.Tensor | None = None

    def get_checkpoint_metadata(self) -> dict[str, torch.Tensor]:
        return {"layout": torch.tensor(self.layout)}

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        layout = metadata.get("layout")
        if (
            set(metadata) != {"layout"}
            or not isinstance(layout, torch.Tensor)
            or not torch.equal(layout, torch.tensor(self.layout))
        ):
            raise ValueError("Incompatible test layout")

    def post_load_state_dict(self) -> None:
        p = self.param_groups[0]["params"][0]
        self.loaded_momentum = self.state[p]["momentum_buffer"].clone()


class TestCheckpointMetadata(unittest.TestCase):
    def test_metadata_parameter_group_order(self) -> None:
        def make(order: list[list[str]]) -> KeyedOptimizerWrapper:
            params = {key: torch.nn.Parameter(torch.ones(2)) for key in ("a", "b", "c")}
            inner = MetadataSGD([params[key] for key in order[0]], layout=0)
            for group in order[1:]:
                inner.add_param_group({"params": [params[key] for key in group]})
            wrapped = KeyedOptimizerWrapper(params, lambda _: inner)
            wrapped.init_state()
            return wrapped

        source = make([["a", "b"], ["c"]])
        saved = deepcopy(source.state_dict())
        for order in (
            [["a", "b"], ["c"]],
            [["b", "a"], ["c"]],
            [["c"], ["a", "b"]],
            [["a"], ["b", "c"]],
        ):
            with self.subTest(order=order):
                target = make(order)
                for state in target.state.values():
                    state["momentum_buffer"].fill_(7)
                before = deepcopy(target.state_dict())
                if order == [["a", "b"], ["c"]]:
                    target.load_state_dict(saved)
                    torch.testing.assert_close(
                        target.state_dict(), saved, rtol=0, atol=0
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "parameter names or order"):
                        target.load_state_dict(saved)
                    torch.testing.assert_close(
                        target.state_dict(), before, rtol=0, atol=0
                    )

    def _make_pair(self, value: float) -> tuple[OptimizerWrapper, list[MetadataSGD]]:
        children: list[KeyedOptimizer | tuple[str, KeyedOptimizer]] = []
        raw = []
        for index, key in enumerate(("a", "b")):
            p = torch.nn.Parameter(torch.tensor([value]))
            inner = MetadataSGD([p], layout=index)
            child = KeyedOptimizerWrapper(
                {key: p}, lambda _, optimizer=inner: optimizer
            )
            p.grad = torch.full_like(p, value + index)
            child.step()
            children.append(OptimizerWrapper(child))
            raw.append(inner)
        wrapped = OptimizerWrapper(CombinedOptimizer(children))
        wrapped.save_param_groups(True)
        return wrapped, raw

    def test_anonymous_metadata_roundtrip(self) -> None:
        source, _ = self._make_pair(7.0)
        saved = deepcopy(source.state_dict())
        self.assertEqual(set(saved["checkpoint_metadata"]), {"0:", "1:"})
        target, raw = self._make_pair(-3.0)
        target.load_state_dict(saved)
        actual = target.state_dict()
        self.assertEqual(actual["param_groups"], saved["param_groups"])
        for key in ("state", "checkpoint_metadata"):
            torch.testing.assert_close(actual[key], saved[key], rtol=0, atol=0)
        for key, inner in zip(("a", "b"), raw):
            torch.testing.assert_close(
                inner.loaded_momentum,
                saved["state"][key]["momentum_buffer"],
                rtol=0,
                atol=0,
            )

    def test_metadata_rejection_before_any_state_copy(self) -> None:
        source, _ = self._make_pair(7.0)
        saved = deepcopy(source.state_dict())
        for fault in (
            "missing",
            "extra",
            "second_layout",
            "swapped",
            "keys_dtype",
            "keys_shape",
            "keys_type",
            "keys_value",
        ):
            with self.subTest(fault=fault):
                target, raw = self._make_pair(-3.0)
                for inner in raw:
                    inner.param_groups[0]["lr"] = 0.7
                before = deepcopy(target.state_dict())
                invalid = deepcopy(saved)
                metadata = invalid["checkpoint_metadata"]
                if fault == "missing":
                    del metadata["1:"]
                elif fault == "extra":
                    metadata["2:"] = deepcopy(metadata["1:"])
                elif fault == "second_layout":
                    metadata["1:"]["optimizer"]["layout"] = torch.tensor(99)
                elif fault == "swapped":
                    metadata["0:"], metadata["1:"] = metadata["1:"], metadata["0:"]
                else:
                    keys = metadata["1:"]["parameter_keys"]
                    replacements = {
                        "keys_dtype": keys.to(torch.int64),
                        "keys_shape": keys.unsqueeze(0),
                        "keys_type": keys.tolist(),
                        "keys_value": keys ^ 1,
                    }
                    metadata["1:"]["parameter_keys"] = replacements[fault]
                with self.assertRaises(ValueError):
                    target.load_state_dict(invalid)
                actual = target.state_dict()
                self.assertEqual(actual["param_groups"], before["param_groups"])
                for key in ("state", "checkpoint_metadata"):
                    torch.testing.assert_close(actual[key], before[key], rtol=0, atol=0)
                for inner in raw:
                    self.assertIsNone(inner.loaded_momentum)
