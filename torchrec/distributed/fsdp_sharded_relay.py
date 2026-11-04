#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_api import _ReduceOp, ReduceScatter
from torchcomms import ReduceOp as TorchCommReduceOp, TorchComm


@runtime_checkable
class _RelayBackend(Protocol):
    def sharded_relay_multi_group_all_reduce(
        self,
        tensors: list[torch.Tensor],
        op: TorchCommReduceOp,
        all_active_ranks: list[list[int]],
        per_group_counts: list[int],
        async_op: bool = False,
    ) -> object | None: ...

    def sharded_relay_multi_group_reduce_scatter(
        self,
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        op: TorchCommReduceOp,
        all_active_ranks: list[list[int]],
        per_group_recv_counts: list[int],
        async_op: bool = False,
    ) -> object | None: ...


class ShardedRelayCollective:
    """One fixed FSDP bucket on a node-wide RCCLX communicator.

    Groups use communicator-local rank IDs, in process-group order. Every
    communicator rank must call the same buckets in the same order, including
    ranks helping other groups. Counts describe the all-reduce buffer or the
    reduce-scatter output, and must agree across all ranks before execution.
    Use separate instances for each module and collective dimension.
    """

    def __init__(
        self,
        comm: object,
        rank: int,
        node_ranks: Sequence[int],
        active_groups: Sequence[Sequence[int]],
        counts: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.groups = [list(g) for g in active_groups]
        self.counts = list(counts)
        self.node_ranks = list(node_ranks)
        flat_ranks = [r for g in self.groups for r in g]
        if sorted(flat_ranks) != list(range(len(node_ranks))):
            raise ValueError("Active groups must partition the node communicator")
        if len(set(node_ranks)) != len(node_ranks) or rank not in flat_ranks:
            raise ValueError("Invalid node ranks or communicator rank")
        sizes = {len(g) for g in self.groups}
        if len(sizes) != 1 or sizes.pop() not in (2, 4) or len(self.groups) < 2:
            raise ValueError("Relay requires disjoint 2- or 4-rank groups and helpers")
        if len(counts) != len(self.groups) or any(c <= 0 for c in counts):
            raise ValueError("Provide a positive bucket count for every active group")
        if device.type != "cuda" or torch.version.hip is None:
            raise ValueError("Sharded relay requires an AMD GPU")
        self.my_group = next(i for i, g in enumerate(self.groups) if rank in g)
        self.dtype = dtype
        self.device = device
        # Citrine C3: allocate the helper placeholder directly on its device.
        self.placeholder = torch.empty(1, dtype=dtype, device=device)
        backend = comm.unsafe_get_backend() if isinstance(comm, TorchComm) else comm
        if not isinstance(backend, _RelayBackend):
            raise ValueError(
                "Communicator requires native RCCLX multi-group relay support"
            )
        self.backend = backend

    def _validate(
        self, tensor: torch.Tensor, group: dist.ProcessGroup, count: int
    ) -> None:
        expected_ranks = [self.node_ranks[r] for r in self.groups[self.my_group]]
        if dist.get_process_group_ranks(group) != expected_ranks:
            raise ValueError("FSDP process-group order differs from relay active ranks")
        if (
            tensor.numel() != count
            or tensor.dtype != self.dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("FSDP bucket differs from the configured relay bucket")

    def _group_tensors(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        return [
            tensor.reshape(-1) if i == self.my_group else self.placeholder
            for i in range(len(self.groups))
        ]


class ShardedRelayAllReduce(ShardedRelayCollective):
    def __call__(
        self, tensor: torch.Tensor, group: dist.ProcessGroup, op: _ReduceOp
    ) -> None:
        self._validate(tensor, group, self.counts[self.my_group])
        if op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
            dist.all_reduce(tensor, group=group, op=op)
            return
        self.backend.sharded_relay_multi_group_all_reduce(
            self._group_tensors(tensor),
            op=(
                TorchCommReduceOp.SUM
                if op == dist.ReduceOp.SUM
                else TorchCommReduceOp.AVG
            ),
            all_active_ranks=self.groups,
            per_group_counts=self.counts,
            async_op=False,
        )


class ShardedRelayReduceScatter(ShardedRelayCollective, ReduceScatter):
    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work | None:
        count = self.counts[self.my_group]
        self._validate(output_tensor, group, count)
        self._validate(input_tensor, group, count * group.size())
        if op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
            return dist.reduce_scatter_single(
                output_tensor, input_tensor, group=group, op=op, async_op=async_op
            )
        self.backend.sharded_relay_multi_group_reduce_scatter(
            self._group_tensors(input_tensor),
            self._group_tensors(output_tensor),
            op=(
                TorchCommReduceOp.SUM
                if op == dist.ReduceOp.SUM
                else TorchCommReduceOp.AVG
            ),
            all_active_ranks=self.groups,
            per_group_recv_counts=self.counts,
            async_op=False,
        )
        # The synchronous TorchComm route orders completion on the caller stream.
        return None
