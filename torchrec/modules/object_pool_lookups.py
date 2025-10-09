#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import abc
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    EmbeddingLocation,
    PoolingMode,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch.autograd.profiler import record_function
from torchrec.modules.utils import jagged_index_select_with_empty
from torchrec.sparse.jagged_tensor import JaggedTensor


torch.fx.wrap("jagged_index_select_with_empty")


class KeyedJaggedTensorPoolLookup(abc.ABC, torch.nn.Module):
    """
    Abstract base class for KeyedJaggedTensor pool lookups

    Implementations of this class should define methods for
     - lookup using ids
     - update values associated with ids
     - returning states that should be saved
        and loaded in state_dict()

    Args:
        pool_size (int): size of the pool
        feature_max_lengths (Dict[str,int]): Dict mapping feature name to max length that
            its values can have for any given batch. The underlying storage representation
            for the KJT pool is currently a padded 2D tensor, so this information is
            needed.
        is_weighted (bool): Boolean indicating whether or not the KJTs will have weights
            that need to be stored separately.
        device (torch.device): device that KJTs should be placed on

    Example:
        Other classes should inherit from this class and implement the
        abstract methods.
    """

    _pool_size: int
    _feature_max_lengths: Dict[str, int]
    _is_weighted: bool
    _total_lengths: int
    _total_lengths_t: torch.Tensor
    _key_lengths: torch.Tensor
    _jagged_lengths: torch.Tensor
    _jagged_offsets: torch.Tensor
    _device: torch.device

    def __init__(
        self,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        is_weighted: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._pool_size = pool_size
        self._feature_max_lengths = feature_max_lengths
        self._device = device
        self._total_lengths = sum(self._feature_max_lengths.values())
        self._total_lengths_t = torch.tensor(
            [self._total_lengths], device=device, dtype=torch.int32
        )
        self._is_weighted = is_weighted

        self._key_lengths = torch.zeros(
            (self._pool_size, len(self._feature_max_lengths)),
            dtype=torch.int32,
            device=self._device,
        )

        lengths, offsets = self._infer_jagged_lengths_inclusive_offsets()
        self._jagged_lengths = lengths
        self._jagged_offsets = offsets

    def _infer_jagged_lengths_inclusive_offsets(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths_sum = self._key_lengths.sum(dim=1)
        padded_lengths = self._total_lengths_t - lengths_sum
        jagged_lengths = torch.stack([lengths_sum, padded_lengths], dim=1).flatten()
        return (
            jagged_lengths,
            torch.ops.fbgemm.asynchronous_inclusive_cumsum(jagged_lengths),
        )

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """
        Override _load_from_state_dict in torch.nn.Module.
        """
        torch.nn.Module._load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        lengths, offsets = self._infer_jagged_lengths_inclusive_offsets()
        self._jagged_lengths = lengths
        self._jagged_offsets = offsets

    @abc.abstractmethod
    def lookup(self, ids: torch.Tensor) -> JaggedTensor:
        pass

    @abc.abstractmethod
    def update(self, ids: torch.Tensor, values: JaggedTensor) -> None:
        # assume that at this point there are no duplicate ids, and all preproc is done by KJTPool
        pass

    def forward(self, ids: torch.Tensor) -> JaggedTensor:
        """
        Forward performs a lookup using the given ids

        Args:
            ids (torch.Tensor): Tensor of IDs to lookup

        Returns:
            JaggedTensor: JaggedTensor containing the merged
                values, lengths and weights associated with the ids
                for all the features of the KJT pool.
        """
        return self.lookup(ids)

    def states_to_register(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield "values", self._tbe_state
        yield "key_lengths", self._key_lengths
        if self._is_weighted:
            yield "weights", self._tbe_weights_state


class TensorJaggedIndexSelectLookup(KeyedJaggedTensorPoolLookup):
    _values_dtype: torch.dtype
    _values: torch.Tensor
    _weights: torch.Tensor

    def __init__(
        self,
        pool_size: int,
        values_dtype: torch.dtype,
        feature_max_lengths: Dict[str, int],
        is_weighted: bool,
        device: torch.device,
    ) -> None:
        super().__init__(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            device=device,
            is_weighted=is_weighted,
        )

        self._values_dtype = values_dtype

        self._values = torch.zeros(
            (self._pool_size, self._total_lengths),
            dtype=self._values_dtype,
            device=self._device,
        )

        if self._is_weighted:
            self._weights = torch.zeros(
                (self._pool_size, self._total_lengths),
                dtype=torch.float,
                device=self._device,
            )
        else:
            # to appease torchscript
            self._weights = torch.empty((0,), dtype=torch.float, device=self._device)

    def lookup(self, ids: torch.Tensor) -> JaggedTensor:
        """
        Example:
        memory layout is
        values = [
            [1], [2, 2], 0, 0, 0,0
            [11,11],[12,12,12],0,0
        ]
        lengths = [
            1,2
            2,3
        ]

        We can consider this as a jagged tensor with
        [
            [1,2,2], [0,0,0,0],
            [11,11,12,12,12],[0,0]
        ]
        where we can combine all values together, and all padded values together.
        The index to select into is then 2*ids (doubled because of padding index).

        jagged_index_select will let us retrieve
        [1,2,2,11,11,12,12,12], that we can then massage into
        [
            [1], [2,2]
            [11,11] [12,12,12]
        ]

        Later (not in this method), we turn this into appropriate KJT format,
        using jagged index select to transpose into
        [
            [1] [11, 11]
            [2,2] [12,12,12]
        ]
        """

        with record_function("## KJTPool Lookup ##"):
            key_lengths_for_ids = self._key_lengths[ids]
            lookup_indices = 2 * ids
            lengths = self._jagged_lengths[lookup_indices]
            offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(lengths)
            values = jagged_index_select_with_empty(
                self._values.flatten().unsqueeze(-1),
                lookup_indices,
                self._jagged_offsets,
                offsets,
            )
            weights = torch.jit.annotate(Optional[torch.Tensor], None)
            if self._is_weighted:
                weights = jagged_index_select_with_empty(
                    self._weights.flatten().unsqueeze(-1),
                    lookup_indices,
                    self._jagged_offsets,
                    offsets,
                )

        return JaggedTensor(
            values=values,
            weights=weights,
            lengths=key_lengths_for_ids.flatten(),
        )

    def update(self, ids: torch.Tensor, values: JaggedTensor) -> None:

        with record_function("## KJTPool update ##"):
            key_lengths = (
                # pyre-ignore
                values.lengths()
                .view(-1, len(self._feature_max_lengths))
                .sum(axis=1)
            )
            key_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(key_lengths)
            padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
                values.values(),
                [key_offsets],
                [self._total_lengths],
                0,
            )

            self._values[ids] = padded_values.to(self._values.dtype)
            self._key_lengths[ids] = (
                values.lengths()
                .view(-1, len(self._feature_max_lengths))
                .to(self._key_lengths.dtype)
            )

            if values.weights_or_none() is not None:
                padded_weights = torch.ops.fbgemm.jagged_to_padded_dense(
                    values.weights(),
                    [key_offsets],
                    [self._total_lengths],
                    0,
                )
                self._weights[ids] = padded_weights

            lengths, offsets = self._infer_jagged_lengths_inclusive_offsets()
            self._jagged_lengths = lengths
            self._jagged_offsets = offsets


class UVMCachingInt64Lookup(KeyedJaggedTensorPoolLookup):
    def __init__(
        self,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        is_weighted: bool,
        device: torch.device,
    ) -> None:
        super().__init__(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            device=device,
            is_weighted=is_weighted,
        )

        # memory layout will be
        # [f1 upper bits][f2 upper bits][upper bits paddings][f1 lower bits][f2 lower bits][lower bits paddings]

        # TBE requires dim to be divisible by 4
        self._bit_dims: int = ((self._total_lengths + 4 - 1) // 4) * 4

        self._bit_dims_t: torch.Tensor = torch.tensor(
            [self._bit_dims], dtype=torch.int32, device=self._device
        )

        self._tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    pool_size,
                    2 * self._bit_dims,
                    EmbeddingLocation.MANAGED,
                    ComputeDevice.CUDA,
                ),
            ],
            pooling_mode=PoolingMode.NONE,
            device=device,
        )
        self._tbe_state: torch.Tensor = (
            self._tbe.split_embedding_weights()[0].flatten().view(pool_size, -1)
        )

        if self._is_weighted:
            # pyre-ignore
            self._tbe_weights = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        pool_size,
                        self._bit_dims,
                        (
                            EmbeddingLocation.MANAGED_CACHING
                            if device != torch.device("meta")
                            else EmbeddingLocation.MANAGED
                        ),
                        ComputeDevice.CUDA,
                    ),
                ],
                pooling_mode=PoolingMode.NONE,
                device=device,
            )
            self._tbe_weights_state: torch.Tensor = (
                self._tbe_weights.split_embedding_weights()[0]
                .flatten()
                .view(pool_size, -1)
            )

    def lookup(self, ids: torch.Tensor) -> JaggedTensor:
        with record_function("## UVMCachingInt64Lookup lookup ##"):
            output = self._tbe(
                indices=ids,
                offsets=torch.tensor([0, ids.shape[0]], device=self._device),
            )

            output_int_split = output.view(torch.int32).split(
                [self._bit_dims, self._bit_dims], dim=1
            )
            output_int_upper = output_int_split[0].to(torch.int64) << 32
            output_int_lower = output_int_split[1].to(torch.int64) & 0xFFFFFFFF

            kjt_dense_values = output_int_upper | output_int_lower

            key_lengths_for_ids = self._key_lengths[ids]
            lookup_indices = 2 * ids
            lengths = self._jagged_lengths[lookup_indices]
            offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(lengths)
            values = jagged_index_select_with_empty(
                kjt_dense_values.flatten().unsqueeze(-1),
                lookup_indices,
                self._jagged_offsets,
                offsets,
            )

        return JaggedTensor(
            values=values,
            lengths=key_lengths_for_ids.flatten(),
        )

    def update(self, ids: torch.Tensor, values: JaggedTensor) -> None:
        with record_function("## UVMCachingInt64Lookup update ##"):
            key_lengths = (
                # pyre-ignore
                values.lengths()
                .view(-1, len(self._feature_max_lengths))
                .sum(axis=1)
            )
            key_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(key_lengths)
            padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
                values.values(),
                [key_offsets],
                [self._bit_dims],
                0,
            )

            values_upper_bits = (padded_values >> 32).to(torch.int32)
            values_lower_bits = (padded_values & 0xFFFFFFFF).to(torch.int32)

            state = torch.cat([values_upper_bits, values_lower_bits], dim=1).view(
                torch.float32
            )

            self._tbe_state[ids] = state

            self._key_lengths[ids] = (
                values.lengths()
                .view(-1, len(self._feature_max_lengths))
                .to(self._key_lengths.dtype)
            )

            lengths, offsets = self._infer_jagged_lengths_inclusive_offsets()
            self._jagged_lengths = lengths
            self._jagged_offsets = offsets


class UVMCachingInt32Lookup(KeyedJaggedTensorPoolLookup):
    def __init__(
        self,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        is_weighted: bool,
        device: torch.device,
    ) -> None:
        super().__init__(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            device=device,
            is_weighted=is_weighted,
        )

        # memory layout will be
        # f1        f2
        # [f1 bits] [f2 bits] padding
        # TBE requires dim to be divisible by 4.
        self._bit_dims: int = ((self._total_lengths + 4 - 1) // 4) * 4
        self._bit_dims_t: torch.Tensor = torch.tensor(
            [self._bit_dims], dtype=torch.int32, device=self._device
        )

        self._tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    pool_size,
                    self._bit_dims,
                    (
                        EmbeddingLocation.MANAGED_CACHING
                        if self._device.type != "meta"
                        else EmbeddingLocation.DEVICE
                    ),
                    ComputeDevice.CUDA,
                ),
            ],
            pooling_mode=PoolingMode.NONE,
            device=device,
        )

        self._tbe_state: torch.Tensor = (
            self._tbe.split_embedding_weights()[0].flatten().view(pool_size, -1)
        )

        if self._is_weighted:
            # pyre-ignore
            self._tbe_weights = SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        pool_size,
                        self._bit_dims,
                        (
                            EmbeddingLocation.MANAGED_CACHING
                            if self._device.type != "meta"
                            else EmbeddingLocation.DEVICE
                        ),
                        ComputeDevice.CUDA,
                    ),
                ],
                pooling_mode=PoolingMode.NONE,
                device=device,
            )
            self._tbe_weights_state: torch.Tensor = (
                self._tbe_weights.split_embedding_weights()[0]
                .flatten()
                .view(pool_size, -1)
            )

    def lookup(self, ids: torch.Tensor) -> JaggedTensor:
        with record_function("## UVMCachingInt32Lookup lookup ##"):
            output = self._tbe(
                indices=ids,
                offsets=torch.tensor([0, ids.shape[0]], device=self._device),
            )

            kjt_dense_values = output.view(torch.int32)

            key_lengths_for_ids = self._key_lengths[ids]
            lookup_indices = 2 * ids
            lengths = self._jagged_lengths[lookup_indices]
            offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(lengths)
            values = jagged_index_select_with_empty(
                kjt_dense_values.flatten().unsqueeze(-1),
                lookup_indices,
                self._jagged_offsets,
                offsets
            )

        return JaggedTensor(
            values=values.flatten(),
            lengths=key_lengths_for_ids.flatten(),
        )

    def update(self, ids: torch.Tensor, values: JaggedTensor) -> None:
        with record_function("## UVMCachingInt32Lookup update##"):
            key_lengths = (
                # pyre-ignore
                values.lengths()
                .view(-1, len(self._feature_max_lengths))
                .sum(axis=1)
            )
            key_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(key_lengths)
            state = torch.ops.fbgemm.jagged_to_padded_dense(
                values.values(),
                [key_offsets],
                [self._bit_dims],
                0,
            ).view(torch.float32)

            self._tbe_state[ids] = state

            self._key_lengths[ids] = (
                values.lengths()
                .view(-1, len(self._feature_max_lengths))
                .to(self._key_lengths.dtype)
            )

            lengths, offsets = self._infer_jagged_lengths_inclusive_offsets()
            self._jagged_lengths = lengths
            self._jagged_offsets = offsets


class TensorPoolLookup(abc.ABC, torch.nn.Module):
    """
    Abstract base class for tensor pool lookups

    Implementations of this class should define methods for
     - lookup using ids
     - update values associated with ids
     - returning states that should be saved
        and loaded in state_dict()
     - setting state from loaded values

    Args:
        pool_size (int): size of the pool
        dim (int): dimension of the tensors in the pool
        dtype (torch.dtype): dtype of the tensors in the pool
        device (torch.device): device of the tensors in the pool

    Example:
        Other classes should inherit this base class and implement the
        abstract methods.
    """

    def __init__(
        self,
        pool_size: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._pool_size = pool_size
        self._dim = dim
        self._dtype = dtype
        self._device = device

    @abc.abstractmethod
    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def update(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        # assume that at this point there are no duplicate ids, and all preproc is done by TensorPool
        pass

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Forward performs a lookup using the given ids

        Args:
            ids (torch.Tensor): Tensor of IDs to lookup

        Returns:
            torch.Tensor: Tensor of values associated with the given ids
        """
        return self.lookup(ids)

    @abc.abstractmethod
    def states_to_register(self) -> Iterator[Tuple[str, torch.Tensor]]:
        pass

    @abc.abstractmethod
    def set_state(self, loaded_values: torch.Tensor) -> None:
        pass


class TensorLookup(TensorPoolLookup):
    def __init__(
        self,
        pool_size: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
        enable_uvm: bool = False,
    ) -> None:
        super().__init__(
            pool_size=pool_size,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        self._enable_uvm = enable_uvm
        self._pool: torch.Tensor = (
            torch.zeros(
                (self._pool_size, self._dim),
                out=torch.ops.fbgemm.new_unified_tensor(
                    torch.zeros(
                        (self._pool_size, self._dim),
                        device=device,
                        dtype=dtype,
                    ),
                    [self._pool_size * self._dim],
                    False,
                ),
            )
            if self._enable_uvm
            else torch.zeros(
                (self._pool_size, self._dim),
                dtype=self._dtype,
                device=self._device,
            )
        )

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        torch._assert(
            ids.device.type == self._device.type,
            "ids.device.type does not match self._device.type",
        )
        with record_function("## TensorPool Lookup ##"):
            ret = self._pool[ids]
        return ret

    def update(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        with record_function("## TensorPool update ##"):
            self._pool[ids] = values

    def set_state(self, loaded_values: torch.Tensor) -> None:
        self._pool.copy_(loaded_values)

    def states_to_register(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield "_pool", self._pool


class UVMCachingFloatLookup(TensorPoolLookup):
    def __init__(
        self,
        pool_size: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:

        super().__init__(
            pool_size=pool_size,
            dim=dim,
            dtype=dtype,
            device=device,
        )

        sparse_type = SparseType.from_dtype(self._dtype)

        self._tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    self._pool_size,
                    self._dim,
                    (
                        EmbeddingLocation.MANAGED_CACHING
                        if self._device.type != "meta"
                        else EmbeddingLocation.DEVICE
                    ),
                    ComputeDevice.CUDA,
                ),
            ],
            pooling_mode=PoolingMode.NONE,
            device=device,
            weights_precision=sparse_type,
            output_dtype=sparse_type,
        )

        self._tbe_state: torch.Tensor = (
            self._tbe.split_embedding_weights()[0].flatten().view(pool_size, -1)
        )

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        torch._assert(
            ids.device.type == self._device.type,
            "ids.device.type does not match self._device.type",
        )
        with record_function("## UVMCachingFloatLookup lookup ##"):
            output = self._tbe(
                indices=ids,
                offsets=torch.tensor([0, ids.shape[0]], device=self._device),
            )
        return output

    def update(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        with record_function("## UVMCachingFloatLookup update ##"):
            self._tbe_state[ids] = values

    def states_to_register(self) -> Iterator[Tuple[str, torch.Tensor]]:
        yield "_pool", self._tbe_state

    def set_state(self, loaded_values: torch.Tensor) -> None:
        self._tbe_state.copy_(loaded_values)
