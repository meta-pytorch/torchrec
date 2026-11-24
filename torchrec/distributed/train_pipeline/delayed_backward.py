#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Utilities for delaying backward across output-distribution boundaries."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchrec.distributed.comm_ops import Request
from torchrec.distributed.types import NoWait


_DELAYED_BACKWARD_RECORD_ATTRIBUTE = "_torchrec_delayed_backward_record"
_OUTPUT_DIST_CHILD_ATTRIBUTES = (
    "_awaitables",
    "_awaitables_per_sharding",
    "_tensor_awaitable",
)


@dataclass(eq=False)
class DelayedBackwardTensor:
    """One detached output-distribution autograd boundary."""

    request: Request[Any]
    original_tensor: torch.Tensor
    proxy_tensor: torch.Tensor
    _active: bool = field(default=True, init=False, repr=False)


def get_output_dist_requests(output: Any) -> list[Request[Any]]:
    """Returns output-dist requests reachable from an awaitable output.

    Supported output structures include nested tuples/lists and the common
    TorchRec awaitable fields. ``NoWait`` branches are intentionally ignored
    because they do not contain an asynchronous communication boundary.
    """

    requests: list[Request[Any]] = []
    seen_request_ids: set[int] = set()
    seen_output_ids: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, NoWait):
            return
        if isinstance(value, Request):
            request_id = id(value)
            if request_id not in seen_request_ids:
                seen_request_ids.add(request_id)
                requests.append(value)
            return

        value_id = id(value)
        if value_id in seen_output_ids:
            return
        seen_output_ids.add(value_id)

        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return

        try:
            value_attributes = vars(value)
        except TypeError:
            return
        for attribute in _OUTPUT_DIST_CHILD_ATTRIBUTES:
            if attribute in value_attributes:
                visit(value_attributes[attribute])

    visit(output)
    return requests


def detach_output_dist_tensors(output: Any) -> list[DelayedBackwardTensor]:
    """Cuts autograd at every asynchronous output-dist request in ``output``.

    The request's original dummy tensor retains the upstream graph. Its proxy
    is used by the wait operation and accumulates the downstream gradient.
    A request may have only one active delayed-backward boundary at a time.
    """

    requests = get_output_dist_requests(output)
    originals: list[torch.Tensor] = []
    for request in requests:
        if hasattr(request, _DELAYED_BACKWARD_RECORD_ATTRIBUTE):
            raise RuntimeError("Output-dist request is already detached")
        original_tensor = request.dummy_tensor
        if not isinstance(original_tensor, torch.Tensor):
            raise RuntimeError("Output-dist request has no dummy tensor to detach")
        if not original_tensor.requires_grad:
            raise RuntimeError("Output-dist dummy tensor does not require gradients")
        if original_tensor.grad_fn is None:
            raise RuntimeError(
                "Output-dist dummy tensor is not connected to an upstream graph"
            )
        originals.append(original_tensor)

    records = [
        DelayedBackwardTensor(
            request=request,
            original_tensor=original_tensor,
            proxy_tensor=original_tensor.detach().requires_grad_(),
        )
        for request, original_tensor in zip(requests, originals)
    ]
    for record in records:
        record.request.dummy_tensor = record.proxy_tensor
        setattr(record.request, _DELAYED_BACKWARD_RECORD_ATTRIBUTE, record)
    return records


def _validate_active_records(records: Sequence[DelayedBackwardTensor]) -> None:
    seen_record_ids: set[int] = set()
    for record in records:
        record_id = id(record)
        if record_id in seen_record_ids:
            raise RuntimeError("Delayed-backward record was provided more than once")
        seen_record_ids.add(record_id)

        if not record._active:
            raise RuntimeError("Delayed-backward record has already been consumed")
        if (
            getattr(record.request, _DELAYED_BACKWARD_RECORD_ATTRIBUTE, None)
            is not record
        ):
            raise RuntimeError("Delayed-backward record does not own its request")
        if record.request.dummy_tensor is not record.proxy_tensor:
            raise RuntimeError("Output-dist request no longer references its proxy")


def restore_delayed_backward_tensors(
    records: Sequence[DelayedBackwardTensor],
) -> None:
    """Reconnects requests to their original tensors without running backward."""

    _validate_active_records(records)
    for record in records:
        record.request.dummy_tensor = record.original_tensor
        delattr(record.request, _DELAYED_BACKWARD_RECORD_ATTRIBUTE)
        record._active = False


def run_delayed_backward(records: Sequence[DelayedBackwardTensor]) -> None:
    """Runs all ready delayed boundaries in one autograd traversal.

    A proxy with no gradient did not contribute to the downstream loss and is
    skipped. All records are consumed, including skipped records. Requests are
    not restored because their wait already consumed the proxy; restoration is
    only for reconnecting a boundary before wait.
    """

    _validate_active_records(records)
    tensors: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    for record in records:
        gradient = record.proxy_tensor.grad
        if gradient is not None:
            tensors.append(record.original_tensor)
            gradients.append(gradient)

    for record in records:
        delattr(record.request, _DELAYED_BACKWARD_RECORD_ATTRIBUTE)
        record._active = False
    if tensors:
        torch.autograd.backward(tensors, gradients)
