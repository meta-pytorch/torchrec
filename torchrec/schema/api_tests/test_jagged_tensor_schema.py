#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import unittest
from typing import Dict, List, Optional, Tuple, Union

import torch
from torchrec.schema.utils import is_signature_compatible
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class StableJaggedTensor:
    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "JaggedTensor":
        return JaggedTensor(torch.empty(0))

    @staticmethod
    def from_dense_lengths(
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "JaggedTensor":
        return JaggedTensor(torch.empty(0))

    @staticmethod
    def from_dense(
        values: List[torch.Tensor],
        weights: Optional[List[torch.Tensor]] = None,
    ) -> "JaggedTensor":
        return JaggedTensor(torch.empty(0))

    def to_dense(self) -> List[torch.Tensor]:
        return []

    def to_dense_weights(self) -> Optional[List[torch.Tensor]]:
        pass

    def to_padded_dense(
        self,
        desired_length: Optional[int] = None,
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        return torch.empty(0)

    def to_padded_dense_weights(
        self,
        desired_length: Optional[int] = None,
        padding_value: float = 0.0,
    ) -> Optional[torch.Tensor]:
        pass

    def device(self) -> torch.device:
        return torch.device("cpu")

    def lengths(self) -> torch.Tensor:
        return torch.empty(0)

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        pass

    def offsets(self) -> torch.Tensor:
        return torch.empty(0)

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        pass

    def values(self) -> torch.Tensor:
        return torch.empty(0)

    def weights(self) -> torch.Tensor:
        return torch.empty(0)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        pass

    def to(self, device: torch.device, non_blocking: bool = False) -> "JaggedTensor":
        return JaggedTensor(torch.empty(0))

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        pass


class StableKeyedJaggedTensor:
    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[
            Union[List[List[int]], torch.IntTensor]
        ] = None,
        # Below exposed to ensure torch.script-able
        stride_per_key: Optional[List[int]] = None,
        length_per_key: Optional[List[int]] = None,
        lengths_offset_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
        jt_dict: Optional[Dict[str, JaggedTensor]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> None:
        pass

    @staticmethod
    def from_offsets_sync(
        keys: List[str],
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    @staticmethod
    def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    @staticmethod
    def concat(
        kjt_list: List["KeyedJaggedTensor"],
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    @staticmethod
    def empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def sync(self) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def unsync(self) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def device(self) -> torch.device:
        return torch.device("cpu")

    def lengths(self) -> torch.Tensor:
        return torch.empty(0)

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        pass

    def offsets(self) -> torch.Tensor:
        return torch.empty(0)

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        pass

    def keys(self) -> List[str]:
        return []

    def values(self) -> torch.Tensor:
        return torch.empty(0)

    def weights(self) -> torch.Tensor:
        return torch.empty(0)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        pass

    def stride(self) -> int:
        return 0

    def stride_per_key(self) -> List[int]:
        return []

    def stride_per_key_per_rank(self) -> List[List[int]]:
        return []

    def variable_stride_per_key(self) -> bool:
        return False

    def inverse_indices(self) -> Tuple[List[str], torch.Tensor]:
        return ([], torch.empty(0))

    def inverse_indices_or_none(self) -> Optional[Tuple[List[str], torch.Tensor]]:
        pass

    def _key_indices(self) -> Dict[str, int]:
        return {}

    def length_per_key(self) -> List[int]:
        return []

    def length_per_key_or_none(self) -> Optional[List[int]]:
        pass

    def offset_per_key(self) -> List[int]:
        return []

    def offset_per_key_or_none(self) -> Optional[List[int]]:
        pass

    def lengths_offset_per_key(self) -> List[int]:
        return []

    def index_per_key(self) -> Dict[str, int]:
        return {}

    def split(self, segments: List[int]) -> List["KeyedJaggedTensor"]:
        return []

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def flatten_lengths(self) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def __getitem__(self, key: str) -> JaggedTensor:
        return JaggedTensor(torch.empty(0))

    def to_dict(self) -> Dict[str, JaggedTensor]:
        return {}

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        pass

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def pin_memory(self) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))

    def dist_labels(self) -> List[str]:
        return []

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        return []

    def dist_tensors(self) -> List[torch.Tensor]:
        return []

    @staticmethod
    def dist_init(
        keys: List[str],
        tensors: List[torch.Tensor],
        variable_stride_per_key: bool,
        num_workers: int,
        recat: Optional[torch.Tensor],
        stride_per_rank: Optional[List[int]],
        stagger: int = 1,
    ) -> "KeyedJaggedTensor":
        return KeyedJaggedTensor([], torch.empty(0))


class StableKeyedTensor:
    def __init__(
        self,
        keys: List[str],
        length_per_key: List[int],
        values: torch.Tensor,
        key_dim: int = 1,
        # Below exposed to ensure torch.script-able
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
    ) -> None:
        pass

    @staticmethod
    def from_tensor_list(
        keys: List[str], tensors: List[torch.Tensor], key_dim: int = 1, cat_dim: int = 1
    ) -> "KeyedTensor":
        return KeyedTensor([], [], torch.empty(0))

    def keys(self) -> List[str]:
        return []

    def values(self) -> torch.Tensor:
        return torch.empty(0)

    def key_dim(self) -> int:
        return 0

    def device(self) -> torch.device:
        return torch.device("cpu")

    def offset_per_key(self) -> List[int]:
        return []

    def length_per_key(self) -> List[int]:
        return []

    def _key_indices(self) -> Dict[str, int]:
        return {}

    def __getitem__(self, key: str) -> torch.Tensor:
        return torch.empty(0)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {}

    @staticmethod
    def regroup(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]]
    ) -> List[torch.Tensor]:
        return []

    @staticmethod
    def regroup_as_dict(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]], keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        return {}

    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        pass

    def to(self, device: torch.device, non_blocking: bool = False) -> "KeyedTensor":
        return KeyedTensor([], [], torch.empty(0))


class TestJaggedTensorSchema(unittest.TestCase):
    def test_kjt(self) -> None:
        stable_kjt_funcs = inspect.getmembers(
            StableKeyedJaggedTensor, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_kjt_funcs:
            self.assertTrue(getattr(KeyedJaggedTensor, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(KeyedJaggedTensor, func_name)),
                )
            )

    def test_jt(self) -> None:
        stable_jt_funcs = inspect.getmembers(
            StableJaggedTensor, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_jt_funcs:
            self.assertTrue(getattr(JaggedTensor, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(JaggedTensor, func_name)),
                )
            )

    def test_kt(self) -> None:
        stable_kt_funcs = inspect.getmembers(
            StableKeyedTensor, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_kt_funcs:
            self.assertTrue(getattr(KeyedTensor, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(KeyedTensor, func_name)),
                )
            )
