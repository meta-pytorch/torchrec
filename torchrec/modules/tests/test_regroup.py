#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torch.fx
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.modules.embedding_configs import data_type_to_dtype
from torchrec.modules.regroup import (
    _split_to_tensor_dict,
    _to_tensor_dict,
    KTRegroupAsDict,
    PermuteMultiEmbedding,
)
from torchrec.sparse.jagged_tensor import _all_keys_used_once, KeyedTensor
from torchrec.sparse.tests.utils import build_groups, build_kts
from torchrec.types import DataType


class KTRegroupAsDictTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=torch.device("cpu"),
            run_backward=True,
        )
        self.num_groups = 2
        self.keys = ["user", "object"]
        self.labels = torch.randint(0, 1, (128,), device=torch.device("cpu")).float()

    def new_kts(self) -> None:
        self.kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=torch.device("cpu"),
            run_backward=True,
        )

    def test_regroup_backward_skips_and_duplicates(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=True, duplicates=True
        )
        self.assertIs(_all_keys_used_once(self.kts, groups), False)

        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)

        # first run
        tensor_groups = regroup_module(self.kts)
        pred0 = tensor_groups["user"].sum(dim=1).mul(tensor_groups["object"].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, self.labels).sum()
        actual_kt_0_grad, actual_kt_1_grad = torch.autograd.grad(
            loss, [self.kts[0].values(), self.kts[1].values()]
        )

        # clear grads so can reuse inputs
        self.kts[0].values().grad = None
        self.kts[1].values().grad = None

        tensor_groups = KeyedTensor.regroup_as_dict(
            keyed_tensors=self.kts, groups=groups, keys=self.keys
        )
        pred1 = tensor_groups["user"].sum(dim=1).mul(tensor_groups["object"].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, self.labels).sum()
        expected_kt_0_grad, expected_kt_1_grad = torch.autograd.grad(
            loss, [self.kts[0].values(), self.kts[1].values()]
        )

        torch.allclose(pred0, pred1)
        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

        # second run
        self.new_kts()
        tensor_groups = regroup_module(self.kts)
        pred0 = tensor_groups["user"].sum(dim=1).mul(tensor_groups["object"].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, self.labels).sum()
        actual_kt_0_grad, actual_kt_1_grad = torch.autograd.grad(
            loss, [self.kts[0].values(), self.kts[1].values()]
        )

        # clear grads so can reuse inputs
        self.kts[0].values().grad = None
        self.kts[1].values().grad = None

        tensor_groups = KeyedTensor.regroup_as_dict(
            keyed_tensors=self.kts, groups=groups, keys=self.keys
        )
        pred1 = tensor_groups["user"].sum(dim=1).mul(tensor_groups["object"].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, self.labels).sum()
        expected_kt_0_grad, expected_kt_1_grad = torch.autograd.grad(
            loss, [self.kts[0].values(), self.kts[1].values()]
        )

        torch.allclose(pred0, pred1)
        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

    def test_regroup_backward(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=False, duplicates=False
        )
        self.assertIs(_all_keys_used_once(self.kts, groups), True)

        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)
        tensor_groups = regroup_module(self.kts)
        pred0 = tensor_groups["user"].sum(dim=1).mul(tensor_groups["object"].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, self.labels).sum()
        actual_kt_0_grad, actual_kt_1_grad = torch.autograd.grad(
            loss, [self.kts[0].values(), self.kts[1].values()]
        )

        # clear grads so can reuse inputs
        self.kts[0].values().grad = None
        self.kts[1].values().grad = None

        tensor_groups = KeyedTensor.regroup_as_dict(
            keyed_tensors=self.kts, groups=groups, keys=self.keys
        )
        pred1 = tensor_groups["user"].sum(dim=1).mul(tensor_groups["object"].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, self.labels).sum()
        expected_kt_0_grad, expected_kt_1_grad = torch.autograd.grad(
            loss, [self.kts[0].values(), self.kts[1].values()]
        )

        torch.allclose(pred0, pred1)
        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

    def test_fx_and_jit_regroup(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=False, duplicates=False
        )
        self.assertIs(_all_keys_used_once(self.kts, groups), True)

        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)
        # first pass
        regroup_module(self.kts)

        # now trace
        gm = torch.fx.symbolic_trace(regroup_module)
        jit_gm = torch.jit.script(gm)

        out = jit_gm(self.kts)
        eager_out = regroup_module(self.kts)
        for key in out.keys():
            torch.allclose(out[key], eager_out[key])

    def test_fx_and_jit_regroup_captures_lazy_init(self) -> None:
        """
        Regression test: when KTRegroupAsDict is FX-traced as a submodule
        of a parent module before its first forward call, the lazy-init
        guard does not skip the module_init free function, so module_init
        is captured into the resulting FX graph as a call_function node.
        torch.jit.script must then succeed on that graph -- without
        @torch.jit.ignore on module_init, TS would try to compile its
        body and abort at script time with:

          RuntimeError: 'KTRegroupAsDict' object has no attribute or
          method '_init_fbgemm_regroup'. 'module_init' is being compiled
          since it was called from 'GraphModule.forward'

        This test only covers script-time success. Runtime dispatch of
        the captured call requires the regroup module to be the original
        Python instance (not a scripted submodule that strips its
        non-TS methods); production achieves that via a cross-CU
        torch.package setup that isn't reproducible in-process.
        """
        from torchrec.modules.regroup import module_init

        class _Parent(torch.nn.Module):
            def __init__(self, groups: list[list[str]], keys: list[str]) -> None:
                super().__init__()
                self.regroup = KTRegroupAsDict(groups=groups, keys=keys)

            def forward(self, kts: list[KeyedTensor]) -> dict[str, torch.Tensor]:
                return self.regroup(kts)

        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=False, duplicates=False
        )
        parent = _Parent(groups=groups, keys=self.keys)
        self.assertFalse(parent.regroup._is_inited)

        # Trace BEFORE running forward, so module_init is captured into
        # the resulting FX graph (rather than being skipped by the lazy-init guard).
        gm = torch.fx.symbolic_trace(parent)
        captured_targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn(module_init, captured_targets)

        # Must not raise. Without @torch.jit.ignore on module_init this
        # would fail with the RuntimeError above during compilation.
        torch.jit.script(gm)

    def test_fx_and_jit_regroup_with_multi_device(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=False, duplicates=False
        )
        self.assertIs(_all_keys_used_once(self.kts, groups), True)

        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)
        # first pass
        regroup_module(self.kts)

        # now trace
        gm = torch.fx.symbolic_trace(regroup_module)
        jit_gm = torch.jit.script(gm)

        out = jit_gm(self.kts)
        eager_out = regroup_module(self.kts)
        for key in out.keys():
            torch.allclose(out[key], eager_out[key])

    def test_fx_and_jit_regroup_skips_and_duplicates(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=True, duplicates=True
        )
        self.assertIs(_all_keys_used_once(self.kts, groups), False)

        regroup_module = KTRegroupAsDict(
            groups=groups, keys=self.keys, multi_device=True
        )
        # first pass
        regroup_module(self.kts)

        # now trace
        gm = torch.fx.symbolic_trace(regroup_module)
        jit_gm = torch.jit.script(gm)

        out = jit_gm(self.kts)
        eager_out = regroup_module(self.kts)
        for key in out.keys():
            torch.allclose(out[key], eager_out[key])

    @given(data_type=st.sampled_from([DataType.BF16, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_regroup_cast(self, data_type: DataType) -> None:
        dtype = data_type_to_dtype(data_type)
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=True, duplicates=True
        )
        self.assertIs(_all_keys_used_once(self.kts, groups), False)

        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)
        cast_regroup = KTRegroupAsDict(
            groups=groups, keys=self.keys, emb_dtype=data_type
        )

        eager_out = regroup_module(self.kts)
        cast_out = cast_regroup(self.kts)

        for key in eager_out.keys():
            self.assertEqual(cast_out[key].dtype, dtype)
            torch.allclose(cast_out[key], eager_out[key].to(dtype))

    def test_permute_multi_embedding_none_out_lengths(self) -> None:
        groups = [["f1", "f2"], ["f3"]]
        permute = PermuteMultiEmbedding(groups)
        values = [torch.randn(2, 4), torch.randn(2, 8)]
        result = permute(values)
        self.assertEqual(len(result), len(values))
        for r, v in zip(result, values):
            self.assertTrue(torch.equal(r, v))

    def test_to_tensor_dict(self) -> None:
        keys = ["a", "b", "c"]
        values = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        result = _to_tensor_dict(keys, values)
        self.assertEqual(list(result.keys()), keys)
        for k, v in zip(keys, values):
            self.assertTrue(torch.equal(result[k], v))

    def test_split_to_tensor_dict(self) -> None:
        keys = ["x", "y"]
        values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = _split_to_tensor_dict(keys, values)
        self.assertEqual(list(result.keys()), keys)
        for k, v in zip(keys, values):
            self.assertTrue(torch.equal(result[k], v))

    def test_fx_trace_with_none_out_lengths(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=False, duplicates=False
        )
        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)
        eager_out = regroup_module(self.kts)
        gm = torch.fx.symbolic_trace(regroup_module)
        fx_out = gm(self.kts)
        for key in eager_out.keys():
            self.assertTrue(torch.equal(eager_out[key], fx_out[key]))
