#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Callable, Dict, List, Tuple

import torch
import torch.utils._pytree as pytree
from hypothesis import assume, given, settings, strategies as st, Verbosity
from torch.fx._pytree import tree_flatten_spec
from torchrec.sparse.jagged_tensor import (
    _fbgemm_permute_pooled_embs,
    _kt_regroup_arguments,
    _regroup_keyed_tensors,
    KeyedTensor,
    permute_multi_embedding,
    regroup_kts,
)
from torchrec.sparse.tests.utils import build_groups, build_kts
from torchrec.test_utils import skip_if_asan_class

torch.fx.wrap("len")


class TestKeyedTensor(unittest.TestCase):
    def test_key_lookup(self) -> None:
        tensor_list = [
            torch.Tensor([[1.0, 1.0]]),
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]),
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=0, key_dim=0)
        self.assertEqual(kt.key_dim(), 0)

        self.assertTrue(torch.equal(kt["dense_0"], tensor_list[0]))
        self.assertTrue(torch.equal(kt["dense_1"], tensor_list[1]))

    def test_key_lookup_dim_1(self) -> None:
        tensor_list = [
            torch.tensor([[1.0, 1.0]]).T,
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim=1)
        self.assertEqual(kt.key_dim(), 1)
        self.assertTrue(torch.equal(kt["dense_0"], tensor_list[0]))
        self.assertTrue(torch.equal(kt["dense_1"], tensor_list[1]))

    def test_to_dict(self) -> None:
        tensor_list = [
            torch.Tensor([[1.0, 1.0]]),
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]),
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=0, key_dim=0)
        self.assertEqual(kt.key_dim(), 0)

        d = kt.to_dict()
        for key in keys:
            self.assertTrue(torch.equal(kt[key], d[key]))

    def test_to_dict_dim_1(self) -> None:
        tensor_list = [
            torch.tensor([[1.0, 1.0]]).T,
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim=1)
        self.assertEqual(kt.key_dim(), 1)

        d = kt.to_dict()
        for key in keys:
            self.assertTrue(torch.equal(kt[key], d[key]))

    def test_regroup_single_kt(self) -> None:
        tensor_list = [torch.randn(2, 3) for i in range(5)]
        key_dim = 1
        keys = ["dense_0", "dense_1", "dense_2", "dense_3", "dense_4"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim)
        grouped_tensors = KeyedTensor.regroup(
            [kt], [["dense_0", "dense_4"], ["dense_1", "dense_3"], ["dense_2"]]
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[0], torch.cat([tensor_list[0], tensor_list[4]], key_dim)
            )
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[1], torch.cat([tensor_list[1], tensor_list[3]], key_dim)
            )
        )
        self.assertTrue(torch.equal(grouped_tensors[2], tensor_list[2]))

    def test_regroup_multiple_kt(self) -> None:
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 4), torch.randn(2, 8), torch.randn(2, 2)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3), torch.randn(2, 10)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        grouped_tensors = KeyedTensor.regroup(
            [kt_1, kt_2], [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[0],
                torch.cat(
                    [tensor_list_1[0], tensor_list_2[1], tensor_list_1[2]], key_dim
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[1],
                torch.cat([tensor_list_1[1], tensor_list_2[0]], key_dim),
            )
        )

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        regroup_func=st.sampled_from(
            [
                KeyedTensor.regroup,
                regroup_kts,
                permute_multi_embedding,
                _fbgemm_permute_pooled_embs,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=15, deadline=None)
    def test_regroup_kts(
        self, regroup_func: Callable[..., List[torch.Tensor]], device_str: str
    ) -> None:
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)

        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=device,
            run_backward=False,
        )
        groups = build_groups(kts=kts, num_groups=2)
        refs = _regroup_keyed_tensors(kts, groups)
        outputs = regroup_func(kts, groups)
        for ref, output in zip(refs, outputs):
            self.assertEqual(ref.device, output.device)
            if device_str == "meta":
                self.assertEqual(ref.shape, output.shape)
            else:
                torch.testing.assert_close(ref, output)

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        regroup_func=st.sampled_from(
            [
                KeyedTensor.regroup,
                regroup_kts,
                permute_multi_embedding,
                _fbgemm_permute_pooled_embs,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_regroup_kts_inference(
        self, regroup_func: Callable[..., List[torch.Tensor]], device_str: str
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)
        with torch.inference_mode():
            kts = build_kts(
                dense_features=20,
                sparse_features=20,
                dim_dense=64,
                dim_sparse=128,
                batch_size=128,
                device=device,
                run_backward=False,
            )
            groups = build_groups(kts=kts, num_groups=2)
            refs = _regroup_keyed_tensors(kts, groups)
            outputs = regroup_func(kts, groups)
        for ref, output in zip(refs, outputs):
            self.assertEqual(ref.device, output.device)
            if device_str == "meta":
                self.assertEqual(ref.shape, output.shape)
            else:
                torch.testing.assert_close(ref, output)

    def test_regroup_backward_skips_and_duplicates(self) -> None:
        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=torch.device("cpu"),
            run_backward=True,
        )
        groups = build_groups(kts=kts, num_groups=2, skips=True, duplicates=True)
        labels = torch.randint(0, 1, (128,), device=torch.device("cpu")).float()

        tensor_groups = KeyedTensor.regroup(kts, groups)
        pred0 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, labels).sum()
        actual_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        actual_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        # clear grads are return
        kts[0].values().grad = None
        kts[1].values().grad = None

        tensor_groups = _regroup_keyed_tensors(kts, groups)
        pred1 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, labels).sum()
        expected_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        expected_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

    def test_size_in_bytes(self) -> None:
        # Test size_in_bytes with float32 tensor (4 bytes per element)
        tensor_list = [
            torch.Tensor([[1.0, 1.0]]),
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]),
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=0, key_dim=0)

        # Expected: 6 float32 elements (2 + 4) * 4 bytes = 24 bytes
        expected_size = 6 * 4
        self.assertEqual(kt.size_in_bytes(), expected_size)

    def test_size_in_bytes_different_dtypes(self) -> None:
        # Test size_in_bytes with float64 tensor (8 bytes per element)
        tensor_list = [
            torch.DoubleTensor([[1.0, 1.0]]),
            torch.DoubleTensor([[2.0, 2.0], [3.0, 3.0]]),
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=0, key_dim=0)

        # Expected: 6 float64 elements (2 + 4) * 8 bytes = 48 bytes
        expected_size = 6 * 8
        self.assertEqual(kt.size_in_bytes(), expected_size)

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "cuda"]),
        regroup_func=st.sampled_from(
            [
                KeyedTensor.regroup,
                regroup_kts,
                permute_multi_embedding,
                _fbgemm_permute_pooled_embs,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_regroup_backward(
        self, regroup_func: Callable[..., List[torch.Tensor]], device_str: str
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)
        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=device,
            run_backward=True,
        )
        groups = build_groups(kts=kts, num_groups=2, skips=False, duplicates=False)
        labels = torch.randint(0, 1, (128,), device=device).float()

        tensor_groups = KeyedTensor.regroup(kts, groups)
        pred0 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, labels).sum()
        actual_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        actual_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        # clear grads are return
        kts[0].values().grad = None
        kts[1].values().grad = None

        tensor_groups = regroup_func(kts, groups)
        pred1 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, labels).sum()
        expected_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        expected_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

    def test_regroup_multiple_kt_duplicate_keys(self) -> None:
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 4) for i in range(2)]
        keys_1 = ["dense_0", "dense_1"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(3)]
        keys_2 = ["sparse_0", "sparse_1", "dense_2"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        grouped_tensors = KeyedTensor.regroup(
            [kt_1, kt_2], [["dense_0", "sparse_1"], ["dense_1", "sparse_0", "dense_0"]]
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[0],
                torch.cat([tensor_list_1[0], tensor_list_2[1]], key_dim),
            )
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[1],
                torch.cat(
                    [tensor_list_1[1], tensor_list_2[0], tensor_list_1[0]], key_dim
                ),
            )
        )

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        regroup_func=st.sampled_from(
            [
                KeyedTensor.regroup,
                regroup_kts,
                permute_multi_embedding,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_regroup_scriptable(
        self, regroup_func: Callable[..., List[torch.Tensor]], device_str: str
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)

        class MyModule(torch.nn.Module):
            def forward(self, inputs: List[KeyedTensor]) -> List[torch.Tensor]:
                # user provided, not model input
                groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
                return regroup_func(inputs, groups)

        m = MyModule()
        script_model = torch.jit.script(m)
        # input
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3, device=device) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3, device=device) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        inputs = [kt_1, kt_2]
        outputs = script_model(inputs)  # pyre-ignore[29]
        refs = _regroup_keyed_tensors(
            inputs, [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
        )
        for ref, output in zip(refs, outputs):
            self.assertEqual(ref.device, output.device)
            if device_str == "meta":
                self.assertEqual(ref.shape, output.shape)
            else:
                torch.testing.assert_close(ref, output)

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        regroup_func=st.sampled_from(
            [
                KeyedTensor.regroup,
                regroup_kts,
                permute_multi_embedding,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_regroup_scriptable_inference(
        self, regroup_func: Callable[..., List[torch.Tensor]], device_str: str
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)

        class MyModule(torch.nn.Module):
            def forward(self, inputs: List[KeyedTensor]) -> List[torch.Tensor]:
                # user provided, not model input
                groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
                return regroup_func(inputs, groups)

        m = MyModule()
        script_model = torch.jit.script(m)
        with torch.inference_mode():
            # input
            key_dim = 1
            tensor_list_1 = [torch.randn(2, 3, device=device) for i in range(3)]
            keys_1 = ["dense_0", "dense_1", "dense_2"]
            kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
            tensor_list_2 = [torch.randn(2, 3, device=device) for i in range(2)]
            keys_2 = ["sparse_0", "sparse_1"]
            kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
            inputs = [kt_1, kt_2]
            outputs = script_model(inputs)  # pyre-ignore[29
            refs = _regroup_keyed_tensors(
                inputs, [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
            )
        for ref, output in zip(refs, outputs):
            self.assertEqual(ref.device, output.device)
            if device_str == "meta":
                self.assertEqual(ref.shape, output.shape)
            else:
                torch.testing.assert_close(ref, output)

    def test_regroup_fxable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(
                self, inputs: List[KeyedTensor], groups: List[List[str]]
            ) -> List[torch.Tensor]:
                return KeyedTensor.regroup(inputs, groups)

        m = MyModule()

        # input
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        inputs = [kt_1, kt_2]
        groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]

        # ensure that symbolic tracing works
        gm = torch.fx.symbolic_trace(m)
        results = m(inputs, groups)
        traced_results = gm(inputs, groups)
        self.assertEqual(len(results), len(traced_results))
        for result, traced_result in zip(results, traced_results):
            self.assertTrue(torch.equal(result, traced_result))

    def test_regroup_as_dict_scriptable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, inputs: List[KeyedTensor]) -> Dict[str, torch.Tensor]:
                groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
                keys = ["group_0", "group_1"]
                return KeyedTensor.regroup_as_dict(inputs, groups, keys)

        m = MyModule()
        torch.jit.script(m)

    def test_regroup_as_dict_fxable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, inputs: List[KeyedTensor]) -> Dict[str, torch.Tensor]:
                groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
                keys = ["group_0", "group_1"]
                return KeyedTensor.regroup_as_dict(inputs, groups, keys)

        m = MyModule()

        # input
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        inputs = [kt_1, kt_2]

        # ensure that symbolic tracing works
        gm = torch.fx.symbolic_trace(m)
        results = m(inputs)
        traced_results = gm(inputs)
        self.assertEqual(len(results), len(traced_results))
        for result, traced_result in zip(results.values(), traced_results.values()):
            self.assertTrue(torch.equal(result, traced_result))

    def test_scriptable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedTensor) -> torch.Tensor:
                values = input["any"].values()
                return values

        m = MyModule()
        torch.jit.script(m)

    def test_string_none(self) -> None:
        jag_tensor = KeyedTensor(
            [],
            [],
            torch.Tensor(),
        )

        self.assertEqual(
            str(jag_tensor),
            "KeyedTensor()\n",
        )

    def test_string_basic(self) -> None:
        tensor_list = [
            torch.tensor([[1.0]]),
        ]
        keys = ["key"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim=0)

        self.assertEqual(
            str(kt),
            'KeyedTensor({\n    "key": [[1.0]]\n})\n',
        )

    def test_string_values(self) -> None:
        tensor_list = [
            torch.tensor([[1.0, 1.0]]).T,
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list)

        self.assertEqual(
            str(kt),
            'KeyedTensor({\n    "dense_0": [[1.0], [1.0]],\n    "dense_1": [[2.0, 3.0], [2.0, 3.0]]\n})\n',
        )

    def test_pytree(self) -> None:
        tensor_list = [
            torch.Tensor([[1.0, 1.0]]).T,
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=1, key_dim=1)
        # generate the out_spec in the torch.export run
        flattened, out_spec = pytree.tree_flatten(kt)

        # first element of flattened list should be the kt._values
        self.assertTrue(torch.equal(flattened[0], kt.values()))
        # re-construct the unflattened kt from the flattened list plus the out_spec
        unflattened = pytree.tree_unflatten(flattened, out_spec)

        self.assertTrue(isinstance(unflattened, KeyedTensor))
        self.assertListEqual(unflattened.keys(), keys)
        self.assertListEqual(unflattened._length_per_key, kt._length_per_key)

        # for ir export, key order in KT could change
        tensor_list = [
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]).T,
            torch.Tensor([[1.0, 1.0]]).T,
        ]
        keys = ["dense_1", "dense_0"]
        kt2 = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=1, key_dim=1)

        # flatten the kt2 based on previously generated out_spec
        # this is to mimic the exported_program module run
        # the kt2 could have different key order but out_spec is the same
        flattened2 = tree_flatten_spec(kt2, out_spec)

        # re-construct the unflattened kt from the flattened list plus the out_spec
        # the rebuilt kt2 should contain the same effective data as kt (ignoring key order)
        unflattened2 = pytree.tree_unflatten(flattened2, out_spec)
        self.assertTrue(isinstance(unflattened2, KeyedTensor))
        self.assertSetEqual(set(unflattened.keys()), set(unflattened2.keys()))
        for key in kt.keys():
            torch.testing.assert_close(unflattened[key], unflattened2[key])
            torch.testing.assert_close(kt[key], unflattened2[key])


class TestKeyedTensorRegroupOp(unittest.TestCase):
    # pyre-ignore[56]
    @given(device_str=st.sampled_from(["cpu", "meta", "cuda"]))
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_kt_regroup_arguments(self, device_str: str) -> None:
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)

        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            torch.empty(0, device=device), keys, lengths, groups
        )
        ref_permutes = [
            [0, 0, 0, 0, 3, 4],  # f1, jump to 4, as a start
            [1, 0, 0, 3, 5, 0],  # f3
            [0, 1, 3, 0, 4, 0],  # f2
            [1, 2, 5, 0, 6, 0],  # f4
            [0, 2, 0, 6, 3, -6],  # f1 jump to 6, as in a jump sequence
            [2, 2, 0, 9, 8, 0],  # f6
            [0, 3, 0, 0, 3, -8],  # f1 jump stop, as out of boundary
            [1, 3, 11, 3, 7, 0],  # f5
        ]
        if device_str == "meta":
            self.assertEqual(permutes.shape, (len(ref_permutes), len(ref_permutes[0])))
            self.assertEqual(in_shapes.shape, (3,))
            self.assertEqual(out_shapes.shape, (4,))
        else:
            self.assertTrue(
                torch.equal(
                    permutes,
                    torch.tensor(ref_permutes, dtype=torch.int32, device=device),
                )
            )
            self.assertEqual(in_shapes.tolist(), [7, 18, 8])
            self.assertEqual(out_shapes.tolist(), [8, 4, 17, 10])
        self.assertEqual(out_lengths, [8, 4, 17, 10])

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        batch_size=st.sampled_from([16, 128, 1024]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_multi_permute_forward(self, device_str: str, batch_size: int) -> None:
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)

        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        with torch.inference_mode():
            values = [torch.randn(batch_size, sum(L), device=device) for L in lengths]
            permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
                values[0], keys, lengths, groups
            )
            outputs = torch.ops.fbgemm.permute_multi_embedding(
                values, permutes, in_shapes, out_shapes, out_lengths
            )

        if device_str == "meta":
            for out, ref in zip(outputs, out_lengths):
                self.assertEqual(out.shape, (batch_size, ref))
        else:
            refs = [[] for _ in groups]
            for i in range(permutes.size(0)):
                in_idx, out, in_start, _, length, _ = permutes[i].tolist()
                refs[out].append(values[in_idx][:, in_start : (in_start + length)])
            refs = [torch.cat(ref, dim=1) for ref in refs]
            for out, ref in zip(outputs, refs):
                torch.testing.assert_close(out, ref)

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["meta", "cpu", "cuda"]),
        dtype=st.sampled_from(
            [
                torch.float,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_multi_permute_dtype(self, device_str: str, dtype: torch.dtype) -> None:
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)

        batch_size = 4
        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        values = [
            torch.randn(batch_size, sum(L), device=device, dtype=dtype) for L in lengths
        ]
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            values[0], keys, lengths, groups
        )
        outputs = torch.ops.fbgemm.permute_multi_embedding(
            values, permutes, in_shapes, out_shapes, out_lengths
        )

        if device_str == "meta":
            for out, ref in zip(outputs, out_lengths):
                self.assertEqual(out.shape, (batch_size, ref))
        else:
            refs = [[] for _ in groups]
            for i in range(permutes.size(0)):
                in_idx, out, in_start, _, length, _ = permutes[i].tolist()
                refs[out].append(values[in_idx][:, in_start : (in_start + length)])
            refs = [torch.cat(ref, dim=1) for ref in refs]
            for out, ref in zip(outputs, refs):
                torch.testing.assert_close(out, ref)
                self.assertEqual(out.dtype, ref.dtype)

    # pyre-ignore[56]
    @given(
        zipped_args=st.sampled_from(
            [
                ("cpu", 32, [[3, 4], [5, 6, 7], [8]]),
                ("cuda", 128, [[96, 256], [512, 128, 768], [1024]]),
            ],
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_multi_permute_backward(
        self, zipped_args: Tuple[str, int, List[List[int]]]
    ) -> None:
        device_str, batch_size, lengths = zipped_args
        # assumption only fails when using cuda but device == 0.
        assume(device_str != "cuda" or torch.cuda.device_count() > 0)
        device = torch.device(device_str)

        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        values = [
            torch.randn(batch_size, sum(lens), device=device, requires_grad=True)
            for lens in lengths
        ]
        ref_values = [v.detach() for v in values]
        for v in ref_values:
            v.requires_grad = True
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            values[0], keys, lengths, groups
        )
        refs = [[] for _ in groups]
        for i in range(permutes.size(0)):
            in_idx, out_idx, in_start, _, length, _ = permutes[i].tolist()
            refs[out_idx].append(ref_values[in_idx][:, in_start : (in_start + length)])
        refs = [torch.cat(ref, dim=1) for ref in refs]
        outputs = torch.ops.fbgemm.permute_multi_embedding(
            values, permutes, in_shapes, out_shapes, out_lengths
        )
        for out, ref in zip(outputs, refs):
            self.assertTrue(torch.allclose(out, ref))

        ref_loss, loss = refs[0].sum(), outputs[0].sum()
        for i in range(1, len(refs)):
            ref_loss += (i + 1.1) * refs[i].sum()
            loss += (i + 1.1) * outputs[i].sum()
        ref_loss.backward()
        loss.backward()
        for val, ref in zip(values, ref_values):
            val_grad, ref_grad = val.grad, ref.grad
            assert isinstance(val_grad, torch.Tensor)
            self.assertTrue(torch.allclose(val_grad, ref_grad))

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "cuda"]),
        batch_size=st.sampled_from([16, 1024]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_multi_permute_noncontiguous(
        self, device_str: str, batch_size: int
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)
        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        values = [
            torch.randn(sum(lens), batch_size, device=device, requires_grad=True)
            for lens in lengths
        ]
        non_contiguous = [v.t() for v in values]
        for value in non_contiguous:
            self.assertFalse(value.is_contiguous())
        ref_values = [v.detach() for v in values]
        for v in ref_values:
            v.requires_grad = True
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            non_contiguous[0], keys, lengths, groups
        )
        refs = [[] for _ in groups]
        for i in range(permutes.size(0)):
            in_idx, out_idx, in_start, _, length, _ = permutes[i].tolist()
            refs[out_idx].append(ref_values[in_idx][in_start : (in_start + length), :])
        refs = [torch.cat(ref).t() for ref in refs]
        outputs = torch.ops.fbgemm.permute_multi_embedding(
            non_contiguous, permutes, in_shapes, out_shapes, out_lengths
        )
        for out, ref in zip(outputs, refs):
            self.assertTrue(torch.allclose(out, ref))

        ref_loss, loss = refs[0].sum(), outputs[0].sum()
        for i in range(1, len(refs)):
            ref_loss += (i + 1.1) * refs[i].sum()
            loss += (i + 1.1) * outputs[i].sum()
        ref_loss.backward()
        loss.backward()
        for val, ref in zip(values, ref_values):
            val_grad, ref_grad = val.grad, ref.grad
            assert isinstance(val_grad, torch.Tensor)
            self.assertTrue(torch.allclose(val_grad, ref_grad))

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        batch_size=st.sampled_from([16, 1024]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_kt_regroup_arguments_op(self, device_str: str, batch_size: int) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)
        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        device = torch.device(device)
        embs = [torch.randn(batch_size, sum(L), device=device) for L in lengths]
        permutes, in_shapes, out_shapes, out_lengths = (
            torch.ops.fbgemm.kt_regroup_arguments(
                embs[0],
                keys,
                lengths,
                groups,
            )
        )
        ref_permutes = [
            [0, 0, 0, 0, 3, 4],  # f1, jump to 4, as a start
            [1, 0, 0, 3, 5, 0],  # f3
            [0, 1, 3, 0, 4, 0],  # f2
            [1, 2, 5, 0, 6, 0],  # f4
            [0, 2, 0, 6, 3, -6],  # f1 jump to 6, as in a jump sequence
            [2, 2, 0, 9, 8, 0],  # f6
            [0, 3, 0, 0, 3, -8],  # f1 jump stop, as out of boundary
            [1, 3, 11, 3, 7, 0],  # f5
        ]
        if device_str == "meta":
            self.assertEqual(permutes.shape, (len(ref_permutes), len(ref_permutes[0])))
            self.assertEqual(in_shapes.shape, (3,))
            self.assertEqual(out_shapes.shape, (4,))
        else:
            self.assertTrue(
                torch.equal(
                    permutes,
                    torch.tensor(ref_permutes, dtype=torch.int32, device=device),
                )
            )
            self.assertEqual(in_shapes.tolist(), [7, 18, 8])
            self.assertEqual(out_shapes.tolist(), [8, 4, 17, 10])
        self.assertEqual(out_lengths, [8, 4, 17, 10])

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "meta", "cuda"]),
        batch_size=st.sampled_from([16, 1024]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_keyed_tensor_regroup_forward(
        self, device_str: str, batch_size: int
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)
        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        permutes = [
            [0, 0, 0, 0, 3, 4],  # f1, jump to 4, as a start
            [1, 0, 0, 3, 5, 0],  # f3
            [0, 1, 3, 0, 4, 0],  # f2
            [1, 2, 5, 0, 6, 0],  # f4
            [0, 2, 0, 6, 3, -6],  # f1 jump to 6, as in a jump sequence
            [2, 2, 0, 9, 8, 0],  # f6
            [0, 3, 0, 0, 3, -8],  # f1 jump stop, as out of boundary
            [1, 3, 11, 3, 7, 0],  # f5
        ]
        with torch.inference_mode():
            values = [
                torch.randn(batch_size, sum(lens), device=device) for lens in lengths
            ]
            refs = [[] for _ in groups]
            for p in permutes:
                in_idx, out_idx, in_start, _, length, _ = p
                refs[out_idx].append(values[in_idx][:, in_start : (in_start + length)])
            refs = [torch.cat(ref, dim=1) for ref in refs]
            outputs = torch.ops.fbgemm.regroup_keyed_tensor(
                values,
                keys,
                lengths,
                groups,
            )
        for out, ref in zip(outputs, refs):
            if device_str == "meta":
                self.assertEqual(out.shape, ref.shape)
            else:
                torch.testing.assert_close(out, ref)

    # pyre-ignore[56]
    @given(
        device_str=st.sampled_from(["cpu", "cuda"]),
        batch_size=st.sampled_from([16, 1024]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_keyed_tensor_regroup_backward(
        self, device_str: str, batch_size: int
    ) -> None:
        if device_str == "cuda" and not torch.cuda.is_available():
            return
        else:
            device = torch.device(device_str)
        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        values = [
            torch.randn(batch_size, sum(lens), device=device, requires_grad=True)
            for lens in lengths
        ]
        ref_values = [v.detach() for v in values]
        for v in ref_values:
            v.requires_grad = True
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            values[0], keys, lengths, groups
        )
        refs = [[] for _ in groups]
        for i in range(permutes.size(0)):
            in_idx, out_idx, in_start, _, length, _ = permutes[i].tolist()
            refs[out_idx].append(ref_values[in_idx][:, in_start : (in_start + length)])
        refs = [torch.cat(ref, dim=1) for ref in refs]
        outputs = torch.ops.fbgemm.regroup_keyed_tensor(
            values,
            keys,
            lengths,
            groups,
        )
        for out, ref in zip(outputs, refs):
            self.assertTrue(torch.allclose(out, ref))

        ref_loss, loss = refs[0].sum(), outputs[0].sum()
        for i in range(1, len(refs)):
            ref_loss += (i + 1.1) * refs[i].sum()
            loss += (i + 1.1) * outputs[i].sum()
        ref_loss.backward()
        loss.backward()
        for val, ref in zip(values, ref_values):
            val_grad, ref_grad = val.grad, ref.grad
            assert isinstance(val_grad, torch.Tensor)
            self.assertTrue(torch.allclose(val_grad, ref_grad))


@skip_if_asan_class
class TestKeyedTensorGPU(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.cuda.current_device()

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_regroup_backward_skips_and_duplicates(self) -> None:
        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=self.device,
            run_backward=True,
        )
        groups = build_groups(kts=kts, num_groups=2, skips=True, duplicates=True)
        labels = torch.randint(0, 1, (128,), device=self.device).float()

        tensor_groups = KeyedTensor.regroup(kts, groups)
        pred0 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, labels).sum()
        actual_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        actual_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        # clear grads are return
        kts[0].values().grad = None
        kts[1].values().grad = None

        tensor_groups = _regroup_keyed_tensors(kts, groups)
        pred1 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, labels).sum()
        expected_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        expected_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_regroup_backward(self) -> None:
        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=self.device,
            run_backward=True,
        )
        groups = build_groups(kts=kts, num_groups=2, skips=False, duplicates=False)
        labels = torch.randint(0, 1, (128,), device=self.device).float()

        tensor_groups = KeyedTensor.regroup(kts, groups)
        pred0 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, labels).sum()
        actual_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        actual_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        # clear grads are return
        kts[0].values().grad = None
        kts[1].values().grad = None

        tensor_groups = _regroup_keyed_tensors(kts, groups)
        pred1 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, labels).sum()
        expected_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        expected_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)
