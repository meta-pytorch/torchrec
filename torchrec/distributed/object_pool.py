#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import abstractmethod
from typing import Generic, Type

import torch
from torch._prims_common import is_integer_dtype
from torchrec.distributed.types import (
    Awaitable,
    DistOut,
    LazyAwaitable,
    Out,
    ShardedModule,
    ShrdCtx,
)
from torchrec.modules.object_pool import ObjectPool


class ShardedObjectPool(
    Generic[Out, DistOut, ShrdCtx],
    ObjectPool[Out],
    ShardedModule[torch.Tensor, DistOut, Out, ShrdCtx],
):
    """
    An abstract distributed K-V store supports update and lookup on torch.Tensor and KeyedJaggedTensor.

    To use the update() function, users need to implement _update_preproc(), _ids_dist(), _update_local(), update_value_dist()
    To use the lookup() function, users need to implement _ids_dist(), _lookup_local(), lookup_value_dist()
    """

    def __init__(self) -> None:
        # pyrefly: ignore[missing-attribute]
        super().__init__()

    @abstractmethod
    def _update_preproc(self, values: Out) -> Out:
        """
        Sanity check and preproc input values
        """
        ...

    @abstractmethod
    def _update_ids_dist(
        self, ctx: ShrdCtx, ids: torch.Tensor
    ) -> Awaitable[Awaitable[torch.Tensor]]: ...

    @abstractmethod
    def _update_local(
        self, ctx: ShrdCtx, ids: torch.Tensor, values: DistOut
    ) -> None: ...

    @abstractmethod
    def _update_values_dist(self, ctx: ShrdCtx, values: Out) -> Awaitable[DistOut]: ...

    @abstractmethod
    def _lookup_ids_dist(
        self, ctx: ShrdCtx, ids: torch.Tensor
    ) -> Awaitable[Awaitable[torch.Tensor]]: ...

    @abstractmethod
    def _lookup_local(self, ctx: ShrdCtx, ids: torch.Tensor) -> DistOut: ...

    @abstractmethod
    def _lookup_values_dist(
        self, ctx: ShrdCtx, values: DistOut
    ) -> LazyAwaitable[Out]: ...

    @abstractmethod
    def create_context(self) -> ShrdCtx:
        pass

    # pyrefly: ignore[bad-override]
    def forward(self, ids: torch.Tensor) -> LazyAwaitable[Out]:
        """
        Perform distributed lookup on the pool using `ids`

        It comprises 3 stages:

        1) IDs received at each rank must be distributed via all2all to the correct ranks.
        2) Each rank receives the correct IDs, and looks up the values locally
        3) Each rank distributes the values from local lookup to other ranks. Note that this step depends on IDs dist because we need to know the batch dimension of tensors to send to all other ranks.

        Refer to docstring for `ShardedTensorPool` and `ShardedKeyedJaggedTensorPool` for examples.
        """
        torch._assert(is_integer_dtype(ids.dtype), "ids type must be int")

        ctx = self.create_context()
        id_dist = self._lookup_ids_dist(ctx, ids).wait().wait()
        local_lookup = self._lookup_local(ctx, id_dist)
        dist_values = self._lookup_values_dist(ctx, local_lookup)
        return dist_values

    # pyrefly: ignore[bad-override]
    def lookup(self, ids: torch.Tensor) -> LazyAwaitable[Out]:
        return self.forward(ids)

    # pyrefly: ignore[bad-param-name-override]
    def update(self, ids: torch.Tensor, values: Out) -> None:
        """
        Perform distributed update on the pool mapping `ids` to `values`

        Args:
            ids (torch.Tensor): 1D tensor containing ids to be updated
            values (torch.Tensor): tensor where first dim must equal number of ids

        It comprises 4 stages:

        1) Optional preproc stage for the values tensor received
        2) Distribute IDs to correct ranks
        3) Distribute value tensor/KJT to correct ranks
        4) Each rank will now have the IDs to update and the corresponding values tensor/KJT, and can update locally

        Refer to docstring for `ShardedTensorPool` and `ShardedKeyedJaggedTensorPool` for examples.
        """
        torch._assert(is_integer_dtype(ids.dtype), "ids type must be int")
        values = self._update_preproc(values=values)
        ctx = self.create_context()
        dist_ids = self._update_ids_dist(ctx=ctx, ids=ids).wait().wait()
        dist_values = self._update_values_dist(ctx=ctx, values=values).wait()
        self._update_local(ctx=ctx, ids=dist_ids, values=dist_values)

    # These below aren't used, instead we have lookup_ids_dist/lookup_local/lookup_values_dist, and corresponding update
    def input_dist(
        self,
        ctx: ShrdCtx,
        *input,
        **kwargs,
        #  value of `None`.
        # pyrefly: ignore[bad-return]
    ) -> Awaitable[Awaitable[torch.Tensor]]:
        pass

    # pyrefly: ignore[bad-return]
    def compute(self, ctx: ShrdCtx, dist_input: torch.Tensor) -> DistOut:
        pass

    #  `None`.
    # pyrefly: ignore[bad-return]
    def output_dist(self, ctx: ShrdCtx, output: DistOut) -> LazyAwaitable[Out]:
        pass

    @property
    def unsharded_module_type(self) -> Type[ObjectPool[Out]]:
        return ObjectPool[Out]
