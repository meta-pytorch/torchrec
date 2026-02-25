#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, List
from unittest.mock import MagicMock

import torch
from torch import nn
from torchrec.distributed.comm_ops import Request
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionAwaitable
from torchrec.distributed.train_pipeline.backward_injection import (
    InjectionSite,
    OutputDistSite,
    register_backward_hook,
)
from torchrec.distributed.types import ShardingType


class SimpleModel(nn.Module):
    """Simple nested model for testing InjectionSite."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_a = nn.Linear(4, 4)
        self.layer_b = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_b(self.layer_a(x))


class InjectionSiteTest(unittest.TestCase):
    """Unit tests for InjectionSite (no GPU/distributed required)."""

    def test_find_target_module(self) -> None:
        model = SimpleModel()
        self.assertIs(
            InjectionSite(fqn="layer_a").find_target_module(model), model.layer_a
        )
        self.assertIsNone(InjectionSite(fqn="nonexistent").find_target_module(model))

    def test_find_grad_tensor_nested(self) -> None:
        site = InjectionSite(fqn="")
        grad = torch.tensor([1.0], requires_grad=True)
        nested = (torch.tensor([0.0]), {"k": [torch.tensor([0.0]), grad]})
        self.assertIs(site.find_grad_tensor(nested), grad)
        self.assertIsNone(site.find_grad_tensor(torch.tensor([1.0])))

    def test_register_hook_nonexistent_raises(self) -> None:
        site = InjectionSite(fqn="nonexistent.module")
        with self.assertRaises(ValueError):
            register_backward_hook(site, SimpleModel(), lambda grad: None)

    def test_register_hook_persists_and_removable(self) -> None:
        """Hook fires every iteration; removing it stops firing."""
        model = SimpleModel()
        site = InjectionSite(fqn="layer_a")
        call_count: List[int] = [0]

        handle = register_backward_hook(
            site,
            model,
            lambda grad: call_count.__setitem__(0, call_count[0] + 1),
        )

        for _ in range(3):
            model.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
        self.assertEqual(call_count[0], 3)

        handle.remove()
        model.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        self.assertEqual(call_count[0], 3)


def _make_request_with_dummy_tensor() -> Request:
    """Create a mock Request with a dummy_tensor that requires grad."""
    req = MagicMock(spec=Request)
    req.dummy_tensor = torch.tensor([1.0], requires_grad=True)
    return req


def _make_ebc_awaitable(
    sharding_types: List[str],
    awaitables: List[Any] | None = None,
) -> EmbeddingBagCollectionAwaitable:
    """Create a mock EBC awaitable with given sharding types."""
    mock = MagicMock(spec=EmbeddingBagCollectionAwaitable)
    mock._sharding_types = sharding_types
    if awaitables is None:
        awaitables = []
        for _ in sharding_types:
            w = MagicMock()
            w._tensor_awaitable = _make_request_with_dummy_tensor()
            awaitables.append(w)
    mock._awaitables = awaitables
    return mock


class OutputDistSiteTest(unittest.TestCase):
    """Unit tests for OutputDistSite.find_grad_tensor."""

    def test_ebc_awaitable(self) -> None:
        site = OutputDistSite(fqn="ebc", sharding_type=ShardingType.TABLE_WISE)
        result = site.find_grad_tensor(
            _make_ebc_awaitable([ShardingType.TABLE_WISE.value])
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.requires_grad)

    def test_selects_correct_sharding_type(self) -> None:
        """With multiple sharding types, returns tensor for the matching one."""
        site = OutputDistSite(fqn="ebc", sharding_type=ShardingType.COLUMN_WISE)

        tw_req = _make_request_with_dummy_tensor()
        cw_req = _make_request_with_dummy_tensor()

        tw_awaitable = MagicMock()
        tw_awaitable._tensor_awaitable = tw_req
        cw_awaitable = MagicMock()
        cw_awaitable._tensor_awaitable = cw_req

        ebc = _make_ebc_awaitable(
            [ShardingType.TABLE_WISE.value, ShardingType.COLUMN_WISE.value],
            [tw_awaitable, cw_awaitable],
        )
        self.assertIs(site.find_grad_tensor(ebc), cw_req.dummy_tensor)

    def test_sharding_type_mismatch_raises(self) -> None:
        site = OutputDistSite(fqn="ebc", sharding_type=ShardingType.COLUMN_WISE)
        with self.assertRaises(RuntimeError):
            site.find_grad_tensor(_make_ebc_awaitable([ShardingType.TABLE_WISE.value]))

    def test_unsupported_type_raises(self) -> None:
        site = OutputDistSite(fqn="ebc", sharding_type=ShardingType.TABLE_WISE)
        with self.assertRaises(RuntimeError):
            site.find_grad_tensor("not_an_awaitable")

    def test_hasattr_fallback(self) -> None:
        """Objects with _awaitables/_sharding_types but not EBC/EC use hasattr fallback."""
        site = OutputDistSite(fqn="ebc", sharding_type=ShardingType.TABLE_WISE)

        class FakeAwaitable:
            pass

        w = MagicMock()
        w._tensor_awaitable = _make_request_with_dummy_tensor()

        fake = FakeAwaitable()
        fake._awaitables = [w]  # pyre-ignore[16]
        fake._sharding_types = [ShardingType.TABLE_WISE.value]  # pyre-ignore[16]

        self.assertIsNotNone(site.find_grad_tensor(fake))
