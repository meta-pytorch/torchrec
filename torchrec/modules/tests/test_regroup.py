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
from torchrec.modules.regroup import KTRegroupAsDict
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
        assert _all_keys_used_once(self.kts, groups) is False

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
        assert _all_keys_used_once(self.kts, groups) is True

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
        assert _all_keys_used_once(self.kts, groups) is True

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

    def test_fx_and_jit_regroup_with_multi_device(self) -> None:
        groups = build_groups(
            kts=self.kts, num_groups=self.num_groups, skips=False, duplicates=False
        )
        assert _all_keys_used_once(self.kts, groups) is True

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
        assert _all_keys_used_once(self.kts, groups) is False

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
        assert _all_keys_used_once(self.kts, groups) is False

        regroup_module = KTRegroupAsDict(groups=groups, keys=self.keys)
        cast_regroup = KTRegroupAsDict(
            groups=groups, keys=self.keys, emb_dtype=data_type
        )

        eager_out = regroup_module(self.kts)
        cast_out = cast_regroup(self.kts)

        for key in eager_out.keys():
            self.assertEqual(cast_out[key].dtype, dtype)
            torch.allclose(cast_out[key], eager_out[key].to(dtype))
