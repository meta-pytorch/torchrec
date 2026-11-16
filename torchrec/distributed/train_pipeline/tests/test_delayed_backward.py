#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import unittest
from dataclasses import dataclass
from io import BytesIO
from typing import cast

import torch
import torch.distributed as dist
from torchrec.distributed.comm_ops import Request
from torchrec.distributed.train_pipeline.delayed_backward import (
    detach_output_dist_tensors,
    get_output_dist_requests,
    restore_delayed_backward_tensors,
    run_delayed_backward,
)
from torchrec.distributed.types import LazyAwaitable, NoWait


class _Awaitables(LazyAwaitable[torch.Tensor]):
    def __init__(self, awaitables: list[object]) -> None:
        super().__init__()
        self._awaitables = awaitables

    def _wait_impl(self) -> torch.Tensor:
        raise AssertionError("Request discovery must not wait on the output")


@dataclass
class _AwaitablesPerSharding:
    _awaitables_per_sharding: list[object]


@dataclass
class _TensorAwaitable:
    _tensor_awaitable: object


def _request(dummy_tensor: torch.Tensor) -> Request[torch.Tensor]:
    request: Request[torch.Tensor] = Request(
        cast(dist.ProcessGroup, object()), dummy_tensor.device
    )
    request.dummy_tensor = dummy_tensor
    return request


class DelayedBackwardTest(unittest.TestCase):
    def test_packaged_module_can_run_delayed_backward(self) -> None:
        buffer = BytesIO()
        module_name = "torchrec.distributed.train_pipeline.delayed_backward"
        with torch.package.PackageExporter(buffer) as exporter:
            exporter.intern(module_name)
            exporter.extern("**")
            exporter.save_module(module_name)
        buffer.seek(0)
        importer = torch.package.PackageImporter(buffer)
        packaged_module = importer.import_module(module_name)
        x = torch.tensor([2.0], requires_grad=True)
        request = _request(x.square())
        records = packaged_module.detach_output_dist_tensors(request)

        request.dummy_tensor.sum().backward()

        self.assertIsNone(x.grad)
        packaged_module.run_delayed_backward(records)
        torch.testing.assert_close(x.grad, torch.tensor([4.0]))

    def test_get_output_dist_requests_discovers_nested_requests(self) -> None:
        requests = [_request(torch.ones(1, requires_grad=True)) for _ in range(4)]
        output = (
            _Awaitables(
                [
                    _TensorAwaitable(requests[0]),
                    NoWait(requests[3]),
                ]
            ),
            [
                _AwaitablesPerSharding([requests[1], _TensorAwaitable(requests[0])]),
                (_TensorAwaitable(requests[2]),),
            ],
        )

        self.assertEqual(
            get_output_dist_requests(output),
            requests[:3],
        )

    def test_run_delayed_backward_matches_joint_backward(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(4, dtype=torch.double, requires_grad=True)
        y = torch.randn(4, dtype=torch.double, requires_grad=True)
        x_reference = x.detach().clone().requires_grad_()
        y_reference = y.detach().clone().requires_grad_()

        shared = x.square()
        request_0 = _request(shared.sin())
        request_1 = _request(shared.exp())
        records = detach_output_dist_tensors(
            _Awaitables([_TensorAwaitable(request_0), _TensorAwaitable(request_1)])
        )
        loss = (
            (request_0.dummy_tensor * 0.7 + request_1.dummy_tensor * -0.2 + y.cos())
            .square()
            .sum()
        )
        loss.backward()

        self.assertIsNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(records[0].proxy_tensor.grad)
        self.assertIsNotNone(records[1].proxy_tensor.grad)
        run_delayed_backward(records)

        shared_reference = x_reference.square()
        reference_loss = (
            (
                shared_reference.sin() * 0.7
                + shared_reference.exp() * -0.2
                + y_reference.cos()
            )
            .square()
            .sum()
        )
        reference_loss.backward()
        torch.testing.assert_close(x.grad, x_reference.grad)
        torch.testing.assert_close(y.grad, y_reference.grad)

    def test_run_delayed_backward_skips_unused_proxy(self) -> None:
        x_used = torch.tensor([2.0], requires_grad=True)
        x_unused = torch.tensor([3.0], requires_grad=True)
        used = x_used.square()
        unused = x_unused.sin()
        used_request = _request(used)
        unused_request = _request(unused)
        records = detach_output_dist_tensors(
            _Awaitables([used_request, unused_request])
        )

        used_request.dummy_tensor.sum().backward()
        self.assertIsNotNone(records[0].proxy_tensor.grad)
        self.assertIsNone(records[1].proxy_tensor.grad)
        run_delayed_backward(records)

        torch.testing.assert_close(x_used.grad, torch.tensor([4.0]))
        self.assertIsNone(x_unused.grad)
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            run_delayed_backward(records)

    def test_restore_reconnects_original_graph(self) -> None:
        x = torch.tensor([2.0], requires_grad=True)
        original = x.pow(3)
        request = _request(original)
        records = detach_output_dist_tensors(request)

        self.assertIs(request.dummy_tensor, records[0].proxy_tensor)
        restore_delayed_backward_tensors(records)
        self.assertIs(request.dummy_tensor, original)
        request.dummy_tensor.sum().backward()

        torch.testing.assert_close(x.grad, torch.tensor([12.0]))
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            restore_delayed_backward_tensors(records)

    def test_detach_rejects_an_already_detached_request(self) -> None:
        request = _request(torch.ones(1, requires_grad=True).square())
        records = detach_output_dist_tensors(request)

        with self.assertRaisesRegex(RuntimeError, "already detached"):
            detach_output_dist_tensors(request)

        restore_delayed_backward_tensors(records)

    def test_detach_rejects_dummy_without_upstream_graph(self) -> None:
        request = _request(torch.ones(1, requires_grad=True))

        with self.assertRaisesRegex(RuntimeError, "not connected"):
            detach_output_dist_tensors(request)
