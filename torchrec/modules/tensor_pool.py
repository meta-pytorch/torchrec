#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import torch
from torchrec.modules.object_pool import ObjectPool
from torchrec.modules.utils import deterministic_dedup


@torch.fx.wrap
def _fx_assert_device(ids: torch.Tensor, device: torch.device) -> None:
    assert ids.device == device
    assert ids.dtype in [torch.int32, torch.int64]


@torch.fx.wrap
def _fx_assert_pool_size(ids: torch.Tensor, pool_size: int) -> None:
    assert torch.all(ids < pool_size).item()


class TensorPool(ObjectPool[torch.Tensor]):
    """
    TensorPool represents a collection of torch.Tensor with uniform dimension.
    It is effectively a 2D tensor of size [pool_size, dim], where each [1,dim] row
    tensor is associated with an unique index which can be set up with update().
    Each row tensor making up the tensor pool can be quried by its index with lookup().

    Args:
        pool_size (int): total number of rows of tensors in the pool
        dim (int): dimension that each tensor in the pool
        dtype (torch.dtype): dtype of the tensors in the pool
        device (Optional[torch.device]): default device
        loaded_values (Optional[torch.Tensor]): pre-defined values to initialize the pool
        enable_uvm (bool): if set to true, the pool will be allocated on UVM

    Call Args:
        ids: 1D torch.Tensor of ids to look up

    Returns:
        torch.Tensor of shape [ids.size(0), dim]

    Example::

        dense_pool = TensorPool(
            pool_size=10,
            dim=2,
            dtype=torch.float
        )

        # Update
        ids = torch.Tensor([1, 9])
        update_values = torch.Tensor([[1.0, 2.0],[3.0,4.0]])
        dense_pool.update(ids=ids, values=update_values)

        # Lookup
        lookup_values = dense_pool.lookup(ids=ids)

        print(lookup_values)
        # tensor([[1., 2.],
        #        [3., 4.]])
    """

    def __init__(
        self,
        pool_size: int,
        dim: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        loaded_values: Optional[torch.Tensor] = None,
        enable_uvm: bool = False,
    ) -> None:
        super().__init__()
        self._pool_size = pool_size
        self._dtype = dtype
        self._device = device if device is not None else torch.device("meta")
        self._dim = dim
        self._enable_uvm = enable_uvm
        # TODO enable multiple lookup on unsharded module

        self.register_buffer(
            "_pool",
            torch.zeros(
                (self._pool_size, self._dim),
                dtype=self._dtype,
                device=self._device,
            ),
        )
        if loaded_values is not None:
            self._pool = loaded_values

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        torch._assert(self._device is not None, "self._device should already be set")
        return self._device

    @property
    def pool(self) -> torch.Tensor:
        return self._pool

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        _fx_assert_device(ids, self._device)
        _fx_assert_pool_size(ids, self._pool_size)
        return self._pool[ids]

    def update(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        assert values.dim() == 2
        assert values.size(1) == self._dim
        assert values.dtype == self._dtype
        assert values.device == self._device, f"{values.device} != {self._device}"
        _fx_assert_device(ids, self._device)
        _fx_assert_pool_size(ids, self._pool_size)

        # If duplicate ids are passed in for update, only the last one is kept
        deduped_ids, dedup_permutation = deterministic_dedup(ids)
        self._pool[deduped_ids] = values[dedup_permutation]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lookup(ids)
