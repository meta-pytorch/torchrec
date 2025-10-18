#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple, Union

import torch
from torchrec.modules.embedding_configs import data_type_to_dtype
from torchrec.sparse.jagged_tensor import (
    _desugar_keyed_tensors,
    _kt_regroup_arguments,
    KeyedTensor,
)
from torchrec.types import CacheMixin, DataType


@torch.fx.wrap
def _get_kts_values(kts: List[KeyedTensor]) -> List[torch.Tensor]:
    return [kt.values() for kt in kts]


@torch.fx.wrap
def _permuted_values(
    kts: List[KeyedTensor], remap: List[Tuple[int, str]], dim: int
) -> torch.Tensor:
    embedding_dicts = [kt.to_dict() for kt in kts]
    values = [embedding_dicts[idx][key] for (idx, key) in remap]
    return torch.cat(values, dim=dim)


@torch.fx.wrap
def module_init(module: "KTRegroupAsDict", keyed_tensors: List[KeyedTensor]) -> None:
    assert len(keyed_tensors) > 0, "Empty list provided"
    assert all(
        kt.device() == keyed_tensors[0].device() for kt in keyed_tensors
    ), "All inputs should be on the same device."
    assert all(
        kt.key_dim() == keyed_tensors[0].key_dim() for kt in keyed_tensors
    ), "All inputs should have the same key_dim"
    module._dim = keyed_tensors[0].key_dim()

    if module._dim == 1:
        module._init_fbgemm_regroup(keyed_tensors)
    else:
        module._init_regroup(keyed_tensors)
    module._is_inited = True


class PermuteMultiEmbedding(torch.nn.Module):
    """
    Module to handle cached tensors and running FBGEMM
    op for KT. This separate module allows fx tracing through
    all the logic in KTRegroupAsDict while keeping what's necessary
    for exposing set_device and allowing tensors to be moved to
    the appropriate device during model processing.

    Args:
        groups (List[List[str]]): Groups from KTRegroupAsDict
        multi_device (bool): Whether to move buffers to current guarded device

    Example:
        groups = [['f1', 'f2'], ['f3']]
        permute_module = PermuteMultiEmbedding(groups)
    """

    def __init__(self, groups: List[List[str]], multi_device: bool = False) -> None:
        """
        Args:
            groups (List[List[str]]): Groups from KTRegroupAsDict
            multi_device (bool): Whether to move buffers to current guarded device
        """
        super().__init__()
        self._groups = groups
        self.register_buffer("_permutes", torch.empty(0), persistent=False)
        self.register_buffer("_in_shapes", torch.empty(0), persistent=False)
        self.register_buffer("_out_shapes", torch.empty(0), persistent=False)
        self._out_lengths: Optional[List[int]] = None

        # When multi_device is True, the input values could be on
        # different devices when the model got loaded,
        # We need to move buffer to the device of the first value
        # in the input list.
        self._multi_device = multi_device

    def init_tensors(
        self,
        permute: torch.Tensor,
        in_shapes: torch.Tensor,
        out_shapes: torch.Tensor,
        out_lengths: List[int],
    ) -> None:
        # no need to pin_memory() or to(..., non_blocking=True) since occurs only once
        self._permutes = permute
        self._in_shapes = in_shapes
        self._out_shapes = out_shapes
        self._out_lengths = out_lengths

    @torch.jit.export
    def set_device(self, device: str) -> None:
        self._permutes = self._permutes.to(device)
        self._in_shapes = self._in_shapes.to(device)
        self._out_shapes = self._out_shapes.to(device)

    def forward(self, values: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            values (List[torch.Tensor]): List of tensors to permute

        Returns:
            List[torch.Tensor]: permuted values
        """
        permutes = self._permutes
        in_shapes = self._in_shapes
        out_shapes = self._out_shapes
        if self._multi_device:
            device = values[0].device
            # Non-blocking, assume permute_multi_embedding will be called with in the same stream
            permutes = permutes.to(device, non_blocking=True)
            in_shapes = in_shapes.to(device, non_blocking=True)
            out_shapes = out_shapes.to(device, non_blocking=True)

        return torch.ops.fbgemm.permute_multi_embedding(
            values,
            permutes,
            in_shapes,
            out_shapes,
            self._out_lengths,
        )


def _to_tensor_dict(
    keys: List[str], values: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]
) -> Dict[str, torch.Tensor]:
    return {key: values[i] for i, key in enumerate(keys)}


class KTRegroupAsDict(torch.nn.Module, CacheMixin):
    """
    KTRegroupAsDict is a nn.Module that mirrors beahvior of static method KeyedTensor.regroup_as_dict()

    The advantage of using this module it caches the regrouping logic after first batch.

    Args:
        groups (List[List[str]]): features per output group
        keys (List[str]): key of each output group
        emb_dtype (Optional[DataType]): embedding data type
        multi_device (bool): Whether to move buffers to current guarded device

    Example::

        keys = ['object', 'user']
        groups = [['f1', 'f2'], ['f3']]
        regroup_module = KTRegroupAsDict(groups, keys)


        tensor_list = [torch.randn(2, 4), torch.randn(2, 8), torch.randn(2, 2)]
        kts = [KeyedTensor.from_tensor_list(['f1', 'f2', 'f3' ], tensor_list)]
        out = regroup_module(kts)

    """

    def __init__(
        self,
        groups: List[List[str]],
        keys: List[str],
        emb_dtype: Optional[DataType] = None,
        multi_device: bool = False,
    ) -> None:
        """
        Args:
            groups (List[List[str]]): features per output group
            keys (List[str]): key of each output group
            emb_dtype (Optional[DataType]): embedding data type
            multi_device (bool): Whether to move buffers to current guarded device
        """
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        assert len(groups) == len(keys), "Groups and keys should have same length"
        self._groups = groups
        self._keys = keys
        self._is_inited = False

        # cached values populated on first forward call
        self._dim: int = 1
        self._use_fbgemm_regroup: bool = False
        self._splits: List[int] = []
        self._idx_key_pairs: List[Tuple[int, str]] = []
        self._permute_pooled_embs_impl = PermuteMultiEmbedding(groups, multi_device)
        self._emb_dtype = emb_dtype

    def _init_fbgemm_regroup(self, kts: List[KeyedTensor]) -> None:
        self._use_fbgemm_regroup = True
        keys, lengths, values = _desugar_keyed_tensors(kts)
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            values[0],
            keys,
            lengths,
            self._groups,
        )
        # no need to pin_memory() or to(..., non_blocking=True) since occurs only once
        self._permute_pooled_embs_impl.init_tensors(
            permutes,
            in_shapes,
            out_shapes,
            out_lengths,
        )

    def _init_regroup(self, kts: List[KeyedTensor]) -> None:
        lengths = [kt.length_per_key() for kt in kts]
        indices = [kt._key_indices() for kt in kts]

        key_to_idx: dict[str, int] = {}
        for i, kt in enumerate(kts):
            for key in kt.keys():
                if key in key_to_idx:
                    raise RuntimeError(
                        f"Duplicate key {key} found in KeyedTensors, undefined behavior"
                    )
                key_to_idx[key] = i

        splits: List[int] = []
        idx_key_pairs: List[Tuple[int, str]] = []
        for group in self._groups:
            group_length = 0
            for name in group:
                idx_key_pairs.append((key_to_idx[name], name))
                group_length += lengths[key_to_idx[name]][
                    indices[key_to_idx[name]][name]
                ]
            splits.append(group_length)

        self._splits = splits
        self._idx_key_pairs = idx_key_pairs

    def embedding_cast(self, embs: List[torch.Tensor]) -> List[torch.Tensor]:
        if self._emb_dtype is None:
            return embs
        dtype = data_type_to_dtype(self._emb_dtype)
        return [emb.to(dtype=dtype) for emb in embs]

    def forward(self, keyed_tensors: List[KeyedTensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            keyed_tensors (List[KeyedTensor]): list of KeyedTensors to regroup

        Returns:
            Dict[str, torch.Tensor]: dictionary of tensors keyed by the keys.
        """
        if not self._is_inited:
            module_init(self, keyed_tensors)

        if self._use_fbgemm_regroup:
            values = _get_kts_values(keyed_tensors)
            values = self.embedding_cast(values)
            permuted_values = self._permute_pooled_embs_impl(values)
            return _to_tensor_dict(self._keys, permuted_values)
        else:
            permuted_values = _permuted_values(
                keyed_tensors, self._idx_key_pairs, self._dim
            )
            permuted_values = self.embedding_cast([permuted_values])[0]
            splitted_values = torch.split(permuted_values, self._splits, dim=self._dim)
            return _to_tensor_dict(self._keys, splitted_values)

    def clear_cache(self) -> None:
        self._is_inited = False
