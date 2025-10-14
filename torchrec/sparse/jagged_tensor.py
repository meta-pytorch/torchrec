#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging

import operator

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.autograd.profiler import record_function
from torch.fx._pytree import register_pytree_flatten_spec, TreeSpec
from torch.utils._pytree import GetAttrKey, KeyEntry, register_pytree_node
from torchrec.pt2.checks import (
    is_non_strict_exporting,
    is_pt2_compiling,
    is_torchdynamo_compiling,
    pt2_check_size_nonzero,
    pt2_checks_all_is_size,
    pt2_checks_tensor_slice,
    pt2_guard_size_oblivious,
)
from torchrec.streamable import Pipelineable

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_cpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_multi_embedding_ops_cpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_multi_embedding_ops_gpu"
    )
except OSError:
    pass


logger: logging.Logger = logging.getLogger()


def _pin_and_move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    moving a tensor from cpu to cuda using pinned memory (non_blocking) is generally faster
    """
    if is_pt2_compiling():
        # TODO: remove once FakeTensor supports pin_memory() and to(..., non_blocking=True)
        return tensor.to(device=device)

    return (
        tensor.pin_memory().to(device=device, non_blocking=True)
        if device.type == "cuda" and tensor.device.type == "cpu"
        else tensor.to(device=device, non_blocking=True)
    )


def _cumsum(o: List[int]) -> List[int]:
    """
    python-list version of converting lengths --> offsets
    """
    ret = [0] * (len(o) + 1)
    for i in range(len(o)):
        ret[i + 1] = ret[i] + o[i]
    return ret


def _to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)


@torch.jit.script_if_tracing
def _batched_lengths_to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    (f, b) = lengths.shape
    offsets_0 = lengths.new_zeros((f, 1))
    offsets_1 = torch.cumsum(lengths, dim=-1).to(lengths.dtype)
    offsets = torch.cat([offsets_0, offsets_1], dim=-1)
    return offsets


def _maybe_compute_lengths(
    lengths: Optional[torch.Tensor], offsets: Optional[torch.Tensor]
) -> torch.Tensor:
    if lengths is None:
        assert offsets is not None
        lengths = torch.diff(offsets)
    return lengths


def _maybe_compute_offsets(
    lengths: Optional[torch.Tensor], offsets: Optional[torch.Tensor]
) -> torch.Tensor:
    if offsets is None:
        assert lengths is not None
        offsets = _to_offsets(lengths)
    return offsets


def _get_weights_or_throw(weights: Optional[torch.Tensor]) -> torch.Tensor:
    assert weights is not None, "This (Keyed)JaggedTensor doesn't have weights."
    return weights


def _get_lengths_offset_per_key_or_throw(
    lengths_offset_per_key: Optional[List[int]],
) -> List[int]:
    assert (
        lengths_offset_per_key is not None
    ), "This (Keyed)JaggedTensor doesn't have lengths_offset_per_key."
    return lengths_offset_per_key


def _get_stride_per_key_or_throw(stride_per_key: Optional[List[int]]) -> List[int]:
    assert (
        stride_per_key is not None
    ), "This (Keyed)JaggedTensor doesn't have stride_per_key."
    return stride_per_key


def _get_inverse_indices_or_throw(
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]],
) -> Tuple[List[str], torch.Tensor]:
    assert inverse_indices is not None, "This KJT doesn't have inverse indices."
    return inverse_indices


def _assert_offsets_or_lengths_is_provided(
    offsets: Optional[torch.Tensor], lengths: Optional[torch.Tensor]
) -> None:
    assert offsets is not None or lengths is not None, "Must provide lengths or offsets"


@torch.fx.wrap
# keep for legacy use cases
def _regroup_keyed_tensors(
    keyed_tensors: List["KeyedTensor"], groups: List[List[str]]
) -> List[torch.Tensor]:

    embedding_dicts = [keyed_tensor.to_dict() for keyed_tensor in keyed_tensors]
    lengths = [keyed_tensor.length_per_key() for keyed_tensor in keyed_tensors]
    indices = [keyed_tensor._key_indices() for keyed_tensor in keyed_tensors]
    key_dim = keyed_tensors[0].key_dim()

    key_to_idx: dict[str, int] = {}
    for i, keyed_tensor in enumerate(keyed_tensors):
        for key in keyed_tensor.keys():
            key_to_idx[key] = i

    # Rearrange values based on groups with a single torch.cat operation.
    split_lengths: List[int] = []
    cat_input: List[torch.Tensor] = []
    for group in groups:
        group_length = 0
        for name in group:
            cat_input.append(embedding_dicts[key_to_idx[name]][name])
            group_length += lengths[key_to_idx[name]][indices[key_to_idx[name]][name]]
        split_lengths.append(group_length)
    rearranged_values = torch.cat(cat_input, key_dim)

    return list(rearranged_values.split(split_lengths, dim=key_dim))


@torch.fx.wrap
def _all_keys_used_once(
    keyed_tensors: List["KeyedTensor"], groups: List[List["str"]]
) -> bool:
    flat_keys: List[str] = []
    flat_groups: List[str] = []
    for keyed_tensor in keyed_tensors:
        flat_keys.extend(keyed_tensor.keys())
    for sub_group in groups:
        flat_groups.extend(sub_group)
    # jit.script does not support set, so we use a dict to represent the set
    key_set: Dict[str, int] = {key: 1 for key in flat_keys}
    group_set: Dict[str, int] = {key: 1 for key in flat_groups}
    return len(key_set) == len(group_set) == len(flat_keys) == len(flat_groups)


@torch.fx.wrap
def permute_multi_embedding(
    keyed_tensors: List["KeyedTensor"], groups: List[List["str"]]
) -> List[torch.Tensor]:
    keys, lengths, values = _desugar_keyed_tensors(keyed_tensors)
    permutes, in_shape, out_shape, out_lengths = torch.ops.fbgemm.kt_regroup_arguments(
        values[0], keys, lengths, groups
    )
    permuted_values = torch.ops.fbgemm.permute_multi_embedding(
        values,
        permutes,
        in_shape,
        out_shape,
        out_lengths,
    )
    return permuted_values


@torch.fx.wrap
def regroup_kts(
    keyed_tensors: List["KeyedTensor"], groups: List[List["str"]]
) -> List[torch.Tensor]:
    keys, lengths, values = _desugar_keyed_tensors(keyed_tensors)
    return torch.ops.fbgemm.regroup_keyed_tensor(
        values,
        keys,
        lengths,
        groups,
    )


@torch.fx.wrap
def _fbgemm_permute_pooled_embs(
    keyed_tensors: List["KeyedTensor"], groups: List[List["str"]]
) -> List[torch.Tensor]:
    keys, lengths, values = _desugar_keyed_tensors(keyed_tensors)
    permute, inv_permute, offsets, inv_offsets, splits = _remap_to_groups(
        keys, lengths, groups
    )
    values = torch.concat(values, dim=1)
    device = values.device
    permuted_values = torch.ops.fbgemm.permute_pooled_embs_auto_grad(
        values,
        _pin_and_move(offsets, device),
        _pin_and_move(permute, device),
        _pin_and_move(inv_offsets, device),
        _pin_and_move(inv_permute, device),
    )
    return list(torch.split(permuted_values, splits, dim=1))


@torch.fx.wrap
def _desugar_keyed_tensors(
    kts: List["KeyedTensor"],
) -> Tuple[List[List[str]], List[List[int]], List[torch.Tensor]]:
    """
    Desugar a list of KeyedTensors into basic data structure
    """
    return (
        [kt.keys() for kt in kts],
        [kt.length_per_key() for kt in kts],
        [kt.values() for kt in kts],
    )


@torch.fx.wrap
def _remap_to_groups(
    keys: List[List[str]],
    key_lengths: List[List[int]],
    groups: List[List[str]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Given a list of keys and lengths per key for each group, return the permute indices, inverse_permute indices, offsets, inv_offsets, splits.
    The output is used to re-arrange values based on groups with a single cat operation.
    """

    lengths: List[int] = []
    flat_keys: List[str] = []
    flat_groups: List[str] = []

    for sub_keys_length in key_lengths:
        lengths.extend(sub_keys_length)
    for sub_keys in keys:
        flat_keys.extend(sub_keys)

    for sub_group in groups:
        flat_groups.extend(sub_group)

    key_splits = [len(sub_group) for sub_group in groups]

    index_map = {key: idx for idx, key in enumerate(flat_keys)}
    permute = [index_map[key] for key in flat_groups]
    inv_lengths = [lengths[i] for i in permute]
    splits = _sum_by_splits(inv_lengths, key_splits)

    inv_permute = [0] * len(permute)
    for i, p in enumerate(permute):
        inv_permute[p] = i

    offsets = torch.tensor(_cumsum(lengths), dtype=torch.int64)
    inv_offsets = torch.tensor(_cumsum(inv_lengths), dtype=torch.int64)
    permute = torch.tensor(permute, dtype=torch.int64)
    inv_permute = torch.tensor(inv_permute, dtype=torch.int64)

    return permute, inv_permute, offsets, inv_offsets, splits


def _kt_regroup_arguments(
    value: torch.Tensor,
    keys: List[List[str]],
    key_lengths: List[List[int]],
    groups: List[List[str]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    returns: permutes, in_shapes, out_shapes, out_lengths
    """
    #  key => (tensor_idx, key_index)
    key_map: Dict[str, Tuple[int, int]] = {
        key: (tensor_idx, key_idx)
        for tensor_idx, tensor in enumerate(keys)
        for key_idx, key in enumerate(tensor)
    }

    #  [offsets per tensor]
    in_offsets: List[List[int]] = [[] for _ in key_lengths]
    for i, tensor in enumerate(key_lengths):
        in_offsets[i] = _cumsum(tensor)
    in_lengths: List[int] = [sum(lengths) for lengths in key_lengths]

    # set total_permutes as the jump stop sign
    total_permutes: int = sum(len(tensor) for tensor in groups)
    out_lengths: List[int] = [0] * len(groups)

    # [input_tensor_idx, output_tensor_idx, input_start, output_start, length, jump]
    permute_param = 6
    permutes: List[List[int]] = [[0] * permute_param for _ in range(total_permutes)]

    # record the last seen index, so that can make the jump from last_seen to current
    last_seen: Dict[str, int] = {}
    permute_idx = 0
    for output_tensor_idx, output_tenser in enumerate(groups):
        output_start = 0
        for output_key in output_tenser:
            input_tensor_idx, input_key_idx = key_map[output_key]
            input_start = in_offsets[input_tensor_idx][input_key_idx]
            length = key_lengths[input_tensor_idx][input_key_idx]

            # add jump data
            if output_key not in last_seen:
                jump = 0  # don't need to jump yet
                # positive as a potential jump start
                last_seen[output_key] = permute_idx
            else:
                prev = last_seen[output_key]
                if prev >= 0:  # positive ==> it's a jump start
                    # jump to current idx, positive as the jump start
                    permutes[prev][5] = permute_idx
                else:  # it's already in a jump sequence, mark as negative
                    permutes[-prev][5] = -permute_idx
                # mark last_seen negative since it's already in jump
                last_seen[output_key] = -permute_idx
                # it's a potential jump stop
                jump = -total_permutes

            permutes[permute_idx][:] = [
                input_tensor_idx,
                output_tensor_idx,
                input_start,
                output_start,
                length,
                jump,
            ]
            permute_idx += 1
            output_start += length
        out_lengths[output_tensor_idx] = output_start

    permute_tensor = torch.tensor(permutes, dtype=torch.int32)
    in_shapes = torch.tensor(in_lengths, dtype=torch.int32)
    out_shapes = torch.tensor(out_lengths, dtype=torch.int32)
    device = value.device
    permute_tensor = _pin_and_move(permute_tensor, device)
    in_shapes = _pin_and_move(in_shapes, device)
    out_shapes = _pin_and_move(out_shapes, device)
    return (
        permute_tensor,
        in_shapes,
        out_shapes,
        out_lengths,
    )


def _values_string(values: torch.Tensor, start: int, end: int) -> str:
    size = values.size()
    if len(size) == 1:
        return "[" + ", ".join([str(value.item()) for value in values[start:end]]) + "]"
    elif len(size) == 2:
        values_list: List[str] = []
        for value in values[start:end]:
            values_list.append("[" + ", ".join([str(s.item()) for s in value]) + "]")
        return "[" + ", ".join(values_list) + "]"
    else:
        raise ValueError(
            "the values dimension is larger than 2, we don't support printing"
        )


def _jagged_values_string(
    values: torch.Tensor,
    offsets: torch.Tensor,
    offset_start: int,
    offset_end: int,
) -> str:
    return (
        "["
        + ", ".join(
            [
                # pyre-fixme[6]: For 2nd param expected `int` but got `Tensor`.
                # pyre-fixme[6]: For 3rd param expected `int` but got `Tensor`.
                _values_string(values, offsets[index], offsets[index + 1])
                for index in range(offset_start, offset_end)
            ]
        )
        + "]"
    )


@torch.fx.wrap
def _optional_mask(
    tensor: Optional[torch.Tensor], mask: torch.Tensor
) -> Optional[torch.Tensor]:

    return tensor[mask] if tensor is not None else None


@torch.fx.wrap
# pyre-ignore
def _arange(*args, **kwargs) -> torch.Tensor:
    return torch.arange(*args, **kwargs)


def _permute_tensor_by_segments(
    tensor: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    output_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Permutes a tensor by segments according to recat tensor.

    For variable stride tensors we permute across length per key, which reduces the
    number of permute indices and lengthens each sequence.
    `keyed_jagged_index_select_dim1` more efficiently parallelizes work for each permute
    index and sequence across multiple thread blocks.

    For permuting KJT with weights that are not of float type (i.e. storing
    bucketization position tensor of longs in weights), `permute_1D_sparse_data` is used
    instead of `keyed_jagged_index_select_dim1` which doesn't support non float weights.

    NOTE:
        `keyed_jagged_index_select_dim1` is only supported for CUDA.
    """
    if tensor.device.type == "cuda" and (
        weights is None or weights.dtype == torch.float32
    ):
        output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values=tensor,
            lengths=segment_sizes,
            offsets=_to_offsets(segment_sizes),
            indices=recat,
            batch_size=segment_sizes.numel(),
            weights=weights,
            selected_lengths_sum=output_size,
        )
        permuted_tensor = output[0]
        permuted_weights = output[2] if weights is not None else None
    else:
        (
            _,
            permuted_tensor,
            permuted_weights,
        ) = torch.ops.fbgemm.permute_1D_sparse_data(
            recat,
            segment_sizes,
            tensor,
            weights,
            output_size,
        )
    return permuted_tensor, permuted_weights


@torch.fx.wrap
def _kjt_concat(
    kjt_list: List["KeyedJaggedTensor"],
) -> "KeyedJaggedTensor":
    if len(kjt_list) == 0:
        raise ValueError("Can't concat empty KJT list")

    is_weighted: bool = kjt_list[0].weights_or_none() is not None
    has_length_per_key: bool = True

    length_per_key: List[int] = []
    keys: List[str] = []
    value_list: List[torch.Tensor] = []
    weight_list: List[torch.Tensor] = []
    length_list: List[torch.Tensor] = []
    stride_per_key_per_rank: List[List[int]] = []
    stride: Optional[int] = None
    inv_idx_keys: List[str] = []
    inv_idx_tensors: List[torch.Tensor] = []

    variable_stride_per_key_list = [kjt.variable_stride_per_key() for kjt in kjt_list]
    assert all(variable_stride_per_key_list) or not any(
        variable_stride_per_key_list
    ), "variable stride per key must be consistent for all KJTs"
    variable_stride_per_key = all(variable_stride_per_key_list)

    for i, kjt in enumerate(kjt_list):
        curr_is_weighted: bool = kjt.weights_or_none() is not None
        if is_weighted != curr_is_weighted:
            raise ValueError("Can't merge weighted KJT with unweighted KJT")
        _length_per_key: Optional[List[int]] = None
        if kjt._length_per_key is None:
            has_length_per_key = False
        else:
            _length_per_key = kjt._length_per_key
        if has_length_per_key and _length_per_key is not None:
            length_per_key += _length_per_key
        keys += kjt.keys()
        value_list.append(kjt.values())
        if is_weighted:
            weight_list.append(kjt.weights())
        length_list.append(kjt.lengths())
        if variable_stride_per_key:
            stride_per_key_per_rank += kjt.stride_per_key_per_rank()
        elif stride is None:
            stride = kjt.stride()
        else:
            assert stride == kjt.stride(), "strides must be consistent for all KJTs"
        if kjt.inverse_indices_or_none() is not None:
            assert (
                len(inv_idx_tensors) == i
            ), "inverse indices must be consistent for all KJTs"
            inv_idx_keys += kjt.inverse_indices()[0]
            inv_idx_tensors.append(kjt.inverse_indices()[1])
        else:
            assert (
                len(inv_idx_tensors) == 0
            ), "inverse indices must be consistent for all KJTs"

    return KeyedJaggedTensor(
        keys=keys,
        values=torch.cat(value_list, dim=0),
        weights=torch.cat(weight_list, dim=0) if is_weighted else None,
        lengths=torch.cat(length_list, dim=0),
        stride=stride,
        stride_per_key_per_rank=(
            stride_per_key_per_rank if variable_stride_per_key else None
        ),
        length_per_key=length_per_key if has_length_per_key else None,
        inverse_indices=(
            (inv_idx_keys, torch.cat(inv_idx_tensors))
            if len(inv_idx_tensors) == len(kjt_list)
            else None
        ),
    )


class JaggedTensorMeta(abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta):
    pass


class JaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """
    Represents an (optionally weighted) jagged tensor.

    A `JaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. See `KeyedJaggedTensor` for full example.

    Implementation is torch.jit.script-able.

    NOTE:
        We will NOT do input validation as it's expensive, you should always pass in the
        valid lengths, offsets, etc.

    Args:
        values (torch.Tensor): values tensor in dense representation.
        weights (Optional[torch.Tensor]): if values have weights. Tensor with same shape
            as values.
        lengths (Optional[torch.Tensor]): jagged slices, represented as lengths.
        offsets (Optional[torch.Tensor]): jagged slices, represented as cumulative
            offsets.
    """

    _fields = ["_values", "_weights", "_lengths", "_offsets"]

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:

        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        _assert_offsets_or_lengths_is_provided(offsets, lengths)
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "JaggedTensor":
        """
        Constructs an empty JaggedTensor.

        Args:
            is_weighted (bool): whether the JaggedTensor has weights.
            device (Optional[torch.device]): device for JaggedTensor.
            values_dtype (Optional[torch.dtype]): dtype for values.
            weights_dtype (Optional[torch.dtype]): dtype for weights.
            lengths_dtype (torch.dtype): dtype for lengths.

        Returns:
            JaggedTensor: empty JaggedTensor.
        """
        weights = (
            torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        )
        return JaggedTensor(
            values=torch.empty(0, dtype=values_dtype, device=device),
            offsets=torch.empty(0, dtype=lengths_dtype, device=device),
            lengths=torch.empty(0, dtype=lengths_dtype, device=device),
            weights=weights,
        )

    @staticmethod
    def from_dense_lengths(
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from values and lengths tensors, with optional weights.
        Note that `lengths` is still of shape (B,), where B is the batch size.

        Args:
            values (torch.Tensor): dense representation of values.
            lengths (torch.Tensor): jagged slices, represented as lengths.
            weights (Optional[torch.Tensor]): if values have weights, tensor with
                the same shape as values.

        Returns:
            JaggedTensor: JaggedTensor created from 2D dense tensor.
        """

        mask2d = (
            _arange(end=values.size(1), device=values.device).expand(values.size(0), -1)
        ) < lengths.unsqueeze(-1)
        return JaggedTensor(
            values=values[mask2d],
            weights=_optional_mask(weights, mask2d),
            lengths=lengths,
        )

    @staticmethod
    def from_dense(
        values: List[torch.Tensor],
        weights: Optional[List[torch.Tensor]] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from list of tensors as values, with optional weights.
        `lengths` will be computed, of shape (B,), where B is `len(values)` which
        represents the batch size.

        Args:
            values (List[torch.Tensor]): a list of tensors for dense representation
            weights (Optional[List[torch.Tensor]]): if values have weights, tensor with
                the same shape as values.

        Returns:
            JaggedTensor: JaggedTensor created from 2D dense tensor.

        Example::

            values = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            weights = [
                torch.Tensor([1.0]),
                torch.Tensor(),
                torch.Tensor([7.0, 8.0]),
                torch.Tensor([10.0, 11.0, 12.0]),
            ]
            j1 = JaggedTensor.from_dense(
                values=values,
                weights=weights,
            )

            # j1 = [[1.0], [], [7.0, 8.0], [10.0, 11.0, 12.0]]
        """

        values_tensor = torch.cat(values, dim=0)
        lengths = torch.tensor(
            [value.size(0) for value in values],
            dtype=torch.int32,
            device=values_tensor.device,
        )
        weights_tensor = torch.cat(weights, dim=0) if weights is not None else None

        return JaggedTensor(
            values=values_tensor,
            weights=weights_tensor,
            lengths=lengths,
        )

    def to_dense(self) -> List[torch.Tensor]:
        """
        Constructs a dense-representation of the JT's values.

        Returns:
            List[torch.Tensor]: list of tensors.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, offsets=offsets)

            values_list = jt.to_dense()

            # values_list = [
            #     torch.tensor([1.0, 2.0]),
            #     torch.tensor([]),
            #     torch.tensor([3.0]),
            #     torch.tensor([4.0]),
            #     torch.tensor([5.0]),
            #     torch.tensor([6.0, 7.0, 8.0]),
            # ]
        """
        tensor_list = []
        for index in range(self.offsets().size(0) - 1):
            offset = self.offsets()[index].item()
            next_offset = self.offsets()[index + 1].item()
            tensor_list.append(self.values()[offset:next_offset])
        return tensor_list

    def to_dense_weights(self) -> Optional[List[torch.Tensor]]:
        """
        Constructs a dense-representation of the JT's weights.

        Returns:
            Optional[List[torch.Tensor]]: list of tensors, `None` if no weights.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

            weights_list = jt.to_dense_weights()

            # weights_list = [
            #     torch.tensor([0.1, 0.2]),
            #     torch.tensor([]),
            #     torch.tensor([0.3]),
            #     torch.tensor([0.4]),
            #     torch.tensor([0.5]),
            #     torch.tensor([0.6, 0.7, 0.8]),
            # ]
        """
        if self.weights_or_none() is None:
            return None
        tensor_list = []
        for index in range(self.offsets().size(0) - 1):
            offset = self.offsets()[index].item()
            next_offset = self.offsets()[index + 1].item()
            tensor_list.append(self.weights()[offset:next_offset])
        return tensor_list

    def to_padded_dense(
        self,
        desired_length: Optional[int] = None,
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Constructs a 2D dense tensor from the JT's values of shape (B, N,).

        Note that `B` is the length of self.lengths() and
        `N` is the longest feature length or `desired_length`.

        If `desired_length` > `length` we will pad with `padding_value`, otherwise we
        will select the last value at `desired_length`.

        Args:
            desired_length (int): the length of the tensor.
            padding_value (float): padding value if we need to pad.

        Returns:
            torch.Tensor: 2d dense tensor.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, offsets=offsets)

            dt = jt.to_padded_dense(
                desired_length=2,
                padding_value=10.0,
            )

            # dt = [
            #     [1.0, 2.0],
            #     [10.0, 10.0],
            #     [3.0, 10.0],
            #     [4.0, 10.0],
            #     [5.0, 10.0],
            #     [6.0, 7.0],
            # ]
        """
        if desired_length is None:
            N = int(torch.max(self.lengths()).item())
        else:
            N = desired_length
        return torch.ops.fbgemm.jagged_to_padded_dense(
            self.values(), [self.offsets()], [N], padding_value
        )

    def to_padded_dense_weights(
        self,
        desired_length: Optional[int] = None,
        padding_value: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Constructs a 2D dense tensor from the JT's weights of shape (B, N,).

        Note that `B` (batch size) is the length of self.lengths() and
        `N` is the longest feature length or `desired_length`.

        If `desired_length` > `length` we will pad with `padding_value`, otherwise we
        will select the last value at `desired_length`.

        Like `to_padded_dense` but for the JT's weights instead of values.

        Args:
            desired_length (int): the length of the tensor.
            padding_value (float): padding value if we need to pad.

        Returns:
            Optional[torch.Tensor]: 2d dense tensor, `None` if no weights.

        Example::

            values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
            jt = JaggedTensor(values=values, weights=weights, offsets=offsets)

            d_wt = jt.to_padded_dense_weights(
                desired_length=2,
                padding_value=1.0,
            )

            # d_wt = [
            #     [0.1, 0.2],
            #     [1.0, 1.0],
            #     [0.3, 1.0],
            #     [0.4, 1.0],
            #     [0.5, 1.0],
            #     [0.6, 0.7],
            # ]
        """
        if self.weights_or_none() is None:
            return None
        if desired_length is None:
            N = int(torch.max(self.lengths()).item())
        else:
            N = desired_length
        return torch.ops.fbgemm.jagged_to_padded_dense(
            self.weights(), [self.offsets()], [N], padding_value
        )

    def device(self) -> torch.device:
        """
        Get JaggedTensor device.

        Returns:
            torch.device: the device of the values tensor.
        """
        return self._values.device

    def lengths(self) -> torch.Tensor:
        """
        Get JaggedTensor lengths. If not computed, compute it from offsets.

        Returns:
            torch.Tensor: the lengths tensor.
        """
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        """
        Get JaggedTensor lengths. If not computed, return None.

        Returns:
            Optional[torch.Tensor]: the lengths tensor.
        """
        return self._lengths

    def offsets(self) -> torch.Tensor:
        """
        Get JaggedTensor offsets. If not computed, compute it from lengths.

        Returns:
            torch.Tensor: the offsets tensor.
        """
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        """
        Get JaggedTensor offsets. If not computed, return None.

        Returns:
            Optional[torch.Tensor]: the offsets tensor.
        """
        return self._offsets

    def values(self) -> torch.Tensor:
        """
        Get JaggedTensor values.

        Returns:
            torch.Tensor: the values tensor.
        """
        return self._values

    def weights(self) -> torch.Tensor:
        """
        Get JaggedTensor weights. If None, throw an error.

        Returns:
            torch.Tensor: the weights tensor.
        """
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        """
        Get JaggedTensor weights. If None, return None.

        Returns:
            Optional[torch.Tensor]: the weights tensor.
        """
        return self._weights

    def to(self, device: torch.device, non_blocking: bool = False) -> "JaggedTensor":
        """
        Move the JaggedTensor to the specified device.

        Args:
            device (torch.device): the device to move to.
            non_blocking (bool): whether to perform the copy asynchronously.

        Returns:
            JaggedTensor: the moved JaggedTensor.
        """
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        return JaggedTensor(
            values=self._values.to(device, non_blocking=non_blocking),
            weights=(
                weights.to(device, non_blocking=non_blocking)
                if weights is not None
                else None
            ),
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
        )

    @torch.jit.unused
    # pyre-fixme[14]: `record_stream` overrides method defined in `Multistreamable`
    #  inconsistently.
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def __str__(self) -> str:
        offsets = self.offsets()

        if self._weights is None:
            return (
                "JaggedTensor({\n    "
                + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1)
                + "\n})\n"
            )

        return (
            "JaggedTensor({\n"
            + '    "values": '
            + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1)
            + ',\n    "weights": '
            + _jagged_values_string(
                _get_weights_or_throw(self._weights), offsets, 0, len(offsets) - 1
            )
            + "\n})\n"
        )


def _jt_flatten(
    t: JaggedTensor,
) -> Tuple[List[Optional[torch.Tensor]], None]:
    return [getattr(t, a) for a in JaggedTensor._fields], None


def _jt_flatten_with_keys(
    t: JaggedTensor,
) -> Tuple[List[Tuple[KeyEntry, Optional[torch.Tensor]]], None]:
    values, context = _jt_flatten(t)
    # pyre can't tell that GetAttrKey implements the KeyEntry protocol
    return [  # pyre-ignore[7]
        (GetAttrKey(k), v) for k, v in zip(JaggedTensor._fields, values)
    ], context


def _jt_unflatten(values: List[Optional[torch.Tensor]], context: None) -> JaggedTensor:
    return JaggedTensor(*values)


def _jt_flatten_spec(t: JaggedTensor, spec: TreeSpec) -> List[Optional[torch.Tensor]]:
    return [getattr(t, a) for a in JaggedTensor._fields]


register_pytree_node(
    JaggedTensor,
    _jt_flatten,
    _jt_unflatten,
    flatten_with_keys_fn=_jt_flatten_with_keys,
    serialized_type_name="torchrec.sparse.jagged_tensor.JaggedTensor",
)
register_pytree_flatten_spec(JaggedTensor, _jt_flatten_spec)


def _assert_tensor_has_no_elements_or_has_integers(
    tensor: Optional[torch.Tensor], tensor_name: str
) -> None:
    if torch.compiler.is_compiling() or tensor is None:
        # Skipping the check tensor.numel() == 0 to not guard on pt2 symbolic shapes.
        # TODO(ivankobzarev): Use guard_size_oblivious to pass tensor.numel() == 0 once it is torch scriptable.
        return

    assert pt2_guard_size_oblivious(tensor.numel() == 0) or tensor.dtype in [
        torch.long,
        torch.int,
        torch.short,
        torch.int8,
        torch.uint8,
    ], "{} must be of integer type, but got {}".format(tensor_name, tensor.dtype)


def _maybe_compute_index_per_key(
    keys: List[str],
    index_per_key: Optional[Dict[str, int]],
) -> Dict[str, int]:
    if index_per_key is None:
        index_per_key = {key: i for i, key in enumerate(keys)}
    return index_per_key


def _maybe_compute_stride_kjt(
    keys: List[str],
    stride: Optional[int],
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    stride_per_key_per_rank: Optional[torch.IntTensor],
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
) -> int:
    if stride is None:
        if len(keys) == 0:
            stride = 0
        elif (
            stride_per_key_per_rank is not None and stride_per_key_per_rank.numel() > 0
        ):
            # For VBE KJT, batch size should be based on inverse_indices when set.
            if inverse_indices is not None and inverse_indices[1].numel() > 0:
                return inverse_indices[1].shape[-1]

            s = stride_per_key_per_rank.sum(dim=1).max().item()
            if not torch.jit.is_scripting() and is_non_strict_exporting():
                stride = torch.sym_int(s)
            else:
                stride = int(s)

        elif offsets is not None and offsets.numel() > 0:
            stride = (offsets.numel() - 1) // len(keys)
        elif lengths is not None:
            stride = lengths.numel() // len(keys)
        else:
            stride = 0
    return stride


def _use_segment_sum_csr(stride_per_key: List[int]) -> bool:
    """
    `segment_sum_csr` performs poorly for small number of segments and many elements
    in each segment to sum. This function uses an empirically calculated equation,
    derived from fitting a quadratic regression to an interval of elements and elements
    per segment that match performance between the kernel and PyTorch solution, to
    determine the threshold of when to use `segment_sum_csr`.
    """
    if is_pt2_compiling():
        # dynamo symbolic shapes can not pass this condition without concrete stride values
        return False

    elements_per_segment = sum(stride_per_key) / len(stride_per_key)
    segment_threshold_float = (
        1.39771
        + 0.0000312222 * elements_per_segment
        + 1.63949e-10 * elements_per_segment**2
    )
    if not torch.jit.is_scripting() and is_non_strict_exporting():
        segment_threshold = torch.sym_int(segment_threshold_float)
    else:
        segment_threshold = int(segment_threshold_float)

    return len(stride_per_key) >= segment_threshold


def _length_per_key_from_stride_per_key(
    lengths: torch.Tensor, stride_per_key: List[int]
) -> List[int]:
    ret: List[int] = []
    if _use_segment_sum_csr(stride_per_key):
        stride_per_key_offsets = _to_offsets(
            _pin_and_move(
                torch.tensor(stride_per_key, dtype=torch.int32), lengths.device
            )
        )
        ret = torch.jit.annotate(
            List[int],
            torch.ops.fbgemm.segment_sum_csr(
                1, stride_per_key_offsets, lengths
            ).tolist(),
        )
    else:
        tensor_list: List[torch.Tensor] = [
            torch.sum(chunk).view(1) for chunk in torch.split(lengths, stride_per_key)
        ]
        if len(tensor_list) == 0:
            return []

        ret = torch.jit.annotate(List[int], torch.cat(tensor_list).tolist())

    pt2_checks_all_is_size(ret)
    return ret


def _maybe_compute_length_per_key(
    keys: List[str],
    stride: int,
    stride_per_key: List[int],
    variable_stride_per_key: bool,
    length_per_key: Optional[List[int]],
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
) -> List[int]:
    if length_per_key is None:
        if (
            len(keys)
            and values is not None
            and values.is_meta
            and not is_non_strict_exporting()
        ):
            # create dummy lengths per key when on meta device
            total_length = values.numel()
            _length = [total_length // len(keys)] * len(keys)
            _length[0] += total_length % len(keys)
        elif len(keys) and lengths is not None:
            _length: List[int] = (
                _length_per_key_from_stride_per_key(lengths, stride_per_key)
                if variable_stride_per_key
                else (
                    torch.sum(
                        pt2_check_size_nonzero(lengths.view(len(keys), stride)), dim=1
                    ).tolist()
                    if pt2_guard_size_oblivious(lengths.numel() != 0)
                    else [0] * len(keys)
                )
            )
        elif len(keys) and offsets is not None and len(offsets) > 0:
            _length: List[int] = (
                _length_per_key_from_stride_per_key(torch.diff(offsets), stride_per_key)
                if variable_stride_per_key
                else torch.sum(torch.diff(offsets).view(-1, stride), dim=1).tolist()
            )
        else:
            _length: List[int] = []
        length_per_key = _length
        pt2_checks_all_is_size(length_per_key)

    return length_per_key


def _maybe_compute_offset_per_key(
    keys: List[str],
    stride: int,
    stride_per_key: List[int],
    variable_stride_per_key: bool,
    length_per_key: Optional[List[int]],
    offset_per_key: Optional[List[int]],
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
) -> Tuple[List[int], List[int]]:
    if length_per_key is None:
        _length_per_key: List[int] = _maybe_compute_length_per_key(
            keys=keys,
            stride=stride,
            stride_per_key=stride_per_key,
            variable_stride_per_key=variable_stride_per_key,
            length_per_key=length_per_key,
            lengths=lengths,
            offsets=offsets,
            values=values,
        )

        if not torch.jit.is_scripting() and is_non_strict_exporting():
            # only torch.export non-strict case
            return (
                _length_per_key,
                (
                    torch.ops.fbgemm.asynchronous_complete_cumsum(
                        torch._refs.tensor(
                            _length_per_key,
                            dtype=torch.int32,
                            device=torch.device("cpu"),
                            pin_memory=False,
                            requires_grad=False,
                        )
                    ).tolist()
                    if len(_length_per_key) > 0
                    else []
                ),
            )
        else:
            return _length_per_key, _cumsum(_length_per_key)
    elif offset_per_key is None:
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            # only torch.export non-strict case
            return (
                length_per_key,
                (
                    torch.ops.fbgemm.asynchronous_complete_cumsum(
                        torch._refs.tensor(
                            length_per_key,
                            dtype=torch.int32,
                            device=torch.device("cpu"),
                            pin_memory=False,
                            requires_grad=False,
                        )
                    ).tolist()
                    if len(length_per_key) > 0
                    else []
                ),
            )
        else:
            return length_per_key, _cumsum(length_per_key)
    else:
        return length_per_key, offset_per_key


def _jagged_tensor_string(
    key: str,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    offsets: torch.Tensor,
    offset_start: int,
    offset_end: int,
) -> str:
    if weights is None:
        return '"{}": '.format(key) + _jagged_values_string(
            values, offsets, offset_start, offset_end
        )

    return (
        '"{}"'.format(key)
        + ': {\n        "values": '
        + _jagged_values_string(values, offsets, offset_start, offset_end)
        + ',\n        "weights": '
        + _jagged_values_string(
            _get_weights_or_throw(weights), offsets, offset_start, offset_end
        )
        + "\n    }"
    )


class ComputeKJTToJTDict(torch.nn.Module):
    """Converts a KeyedJaggedTensor to a dict of JaggedTensors.

    Args:

    Example::
        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        would return

        {
            "Feature0": JaggedTensor([[V0,V1],None,V2]),
            "Feature1": JaggedTensor([V3,V4,[V5,V6,V7]]),
        }
    """

    def forward(
        self, keyed_jagged_tensor: "KeyedJaggedTensor"
    ) -> Dict[str, JaggedTensor]:
        """
        Converts a KeyedJaggedTensor into a dict of JaggedTensors.

        Args:
            keyed_jagged_tensor (KeyedJaggedTensor): tensor to convert
        Returns:
            Dict[str, JaggedTensor]
        """
        return _maybe_compute_kjt_to_jt_dict(
            stride=keyed_jagged_tensor.stride(),
            stride_per_key=keyed_jagged_tensor.stride_per_key(),
            keys=keyed_jagged_tensor.keys(),
            length_per_key=keyed_jagged_tensor.length_per_key(),
            values=keyed_jagged_tensor.values(),
            lengths=keyed_jagged_tensor.lengths(),
            variable_stride_per_key=keyed_jagged_tensor.variable_stride_per_key(),
            weights=keyed_jagged_tensor.weights_or_none(),
            jt_dict=keyed_jagged_tensor._jt_dict,
        )


class ComputeJTDictToKJT(torch.nn.Module):
    """Converts a dict of JaggedTensors to KeyedJaggedTensor.
    Args:

    Example:
    passing in jt_dict
        {
            "Feature0": JaggedTensor([[V0,V1],None,V2]),
            "Feature1": JaggedTensor([V3,V4,[V5,V6,V7]]),
        }
    Returns::
    kjt with content:
    #              0       1        2  <-- dim_1
    # "Feature0"   [V0,V1] None    [V2]
    # "Feature1"   [V3]    [V4]    [V5,V6,V7]
    #   ^
    #  dim_0

    """

    def forward(self, jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
        """
        Args:
            jt_dict: a dict of JaggedTensor
        Returns:
            KeyedJaggedTensor
        """
        return KeyedJaggedTensor.from_jt_dict(jt_dict)


@torch.fx.wrap
def _maybe_compute_kjt_to_jt_dict(
    stride: int,
    stride_per_key: List[int],
    keys: List[str],
    length_per_key: List[int],
    values: torch.Tensor,
    lengths: torch.Tensor,
    variable_stride_per_key: bool,
    weights: Optional[torch.Tensor],
    jt_dict: Optional[Dict[str, JaggedTensor]],
) -> Dict[str, JaggedTensor]:
    if not length_per_key:
        return {}

    if jt_dict is not None:
        return jt_dict

    _jt_dict: Dict[str, JaggedTensor] = {}
    if not torch.jit.is_scripting() and is_pt2_compiling():
        cat_size = 0
        total_size = values.size(0)
        for i in length_per_key:
            cat_size += i
            torch._check(cat_size <= total_size)
        torch._check(cat_size == total_size)
        torch._check_is_size(stride)
    values_list = torch.split(values, length_per_key)
    if variable_stride_per_key:
        split_lengths = torch.split(lengths, stride_per_key)
        split_offsets = [
            torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            for lengths in split_lengths
        ]
    elif pt2_guard_size_oblivious(lengths.numel() > 0):
        strided_lengths = lengths.view(len(keys), stride)
        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            torch._check(strided_lengths.size(0) > 0)
            torch._check(strided_lengths.size(1) > 0)
        split_lengths = torch.unbind(
            strided_lengths,
            dim=0,
        )
        split_offsets = torch.unbind(
            _batched_lengths_to_offsets(strided_lengths),
            dim=0,
        )
    else:
        split_lengths = torch.unbind(lengths, dim=0)
        split_offsets = torch.unbind(lengths, dim=0)

    if weights is not None:
        weights_list = torch.split(weights, length_per_key)
        for idx, key in enumerate(keys):
            length = split_lengths[idx]
            offset = split_offsets[idx]
            _jt_dict[key] = JaggedTensor(
                lengths=length,
                offsets=offset,
                values=values_list[idx],
                weights=weights_list[idx],
            )
    else:
        for idx, key in enumerate(keys):
            length = split_lengths[idx]
            offset = split_offsets[idx]
            _jt_dict[key] = JaggedTensor(
                lengths=length,
                offsets=offset,
                values=values_list[idx],
            )
    return _jt_dict


@torch.fx.wrap
def _kjt_empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
    # empty like function fx wrapped, also avoids device hardcoding
    stride, stride_per_key_per_rank = (
        (None, kjt._stride_per_key_per_rank)
        if kjt._stride_per_key_per_rank is not None and kjt.variable_stride_per_key()
        else (kjt.stride(), None)
    )

    return KeyedJaggedTensor(
        keys=[],
        values=torch.empty(0, device=kjt.device(), dtype=kjt.values().dtype),
        weights=(
            None
            if kjt.weights_or_none() is None
            else torch.empty(0, device=kjt.device(), dtype=kjt.weights().dtype)
        ),
        lengths=torch.empty(0, device=kjt.device(), dtype=kjt.lengths().dtype),
        stride=stride,
        stride_per_key_per_rank=stride_per_key_per_rank,
    )


def _sum_by_splits(input_list: List[int], splits: List[int]) -> List[int]:
    return [
        sum(input_list[sum(splits[:i]) : sum(splits[:i]) + n])
        for i, n in enumerate(splits)
    ]


@torch.fx.wrap
def jt_is_equal(jt_1: "JaggedTensor", jt_2: "JaggedTensor") -> bool:
    """This function checks if two JaggedTensors are equal by comparing their internal representations.
    The comparison is done by comparing the values of the internal representations themselves.
    For optional fields, None values are treated as equal.

    Args:
        jt_1 (JaggedTensor): the first JaggedTensor
        jt_2 (JaggedTensor): the second JaggedTensor

    Returns:
        bool: True if both JaggedTensors have the same values
    """

    if not isinstance(jt_1, JaggedTensor) or not isinstance(jt_2, JaggedTensor):
        return False

    if not _check_attributes(jt_1.values(), jt_2.values(), torch.allclose):
        return False

    _force_length_offset_computation(jt_1)
    _force_length_offset_computation(jt_2)

    attributes_to_check = [
        (jt_1.weights_or_none(), jt_2.weights_or_none()),
        (jt_1.lengths_or_none(), jt_2.lengths_or_none()),
        (jt_1.offsets_or_none(), jt_2.offsets_or_none()),
    ]

    for attr_1, attr_2 in attributes_to_check:
        if not _check_attributes(
            attr_1,
            attr_2,
            torch.allclose if isinstance(attr_1, torch.Tensor) else operator.eq,
        ):
            return False

    return True


@torch.fx.wrap
def kjt_is_equal(kjt_1: "KeyedJaggedTensor", kjt_2: "KeyedJaggedTensor") -> bool:
    """This function checks if two KeyedJaggedTensors are equal by comparing their internal representations.
    The comparison is done by comparing the values of the internal representations themselves.
    For optional fields, None values are treated as equal.
    We compare the keys by ensuring that they have the same length and that the corresponding keys are the same order and same values.

    Args:
        kjt_1 (KeyedJaggedTensor): the first KeyedJaggedTensor
        kjt_2 (KeyedJaggedTensor): the second KeyedJaggedTensor

    Returns:
        bool: True if both KeyedJaggedTensors have the same values
    """
    if not isinstance(kjt_1, KeyedJaggedTensor) or not isinstance(
        kjt_2, KeyedJaggedTensor
    ):
        return False

    # check for missing/extra keys
    if len(kjt_1.keys()) != len(kjt_2.keys()):
        return False

    # check if all keys are equal and in same order
    for a, b in zip(kjt_1.keys(), kjt_2.keys()):
        if a != b:
            return False

    if not _check_attributes(kjt_1.values(), kjt_2.values(), torch.allclose):
        return False

    _force_length_offset_computation(kjt_1)
    _force_length_offset_computation(kjt_2)
    # sync length and offset per key as well
    kjt_1.sync()
    kjt_2.sync()

    attributes_to_check = [
        (kjt_1.lengths_or_none(), kjt_2.lengths_or_none()),
        (kjt_1.weights_or_none(), kjt_2.weights_or_none()),
        (kjt_1.offsets_or_none(), kjt_2.offsets_or_none()),
        (kjt_1.length_per_key_or_none(), kjt_2.length_per_key_or_none()),
        (kjt_1.offset_per_key_or_none(), kjt_2.offset_per_key_or_none()),
        (kjt_1.stride(), kjt_2.stride()),
    ]

    for attr_1, attr_2 in attributes_to_check:
        if not _check_attributes(
            attr_1,
            attr_2,
            torch.allclose if isinstance(attr_1, torch.Tensor) else operator.eq,
        ):
            return False

    return True


def _force_length_offset_computation(
    kjt: Union["KeyedJaggedTensor", "JaggedTensor"]
) -> None:
    """Helper function to force length/offset computation for KJT or JT
    Mainly used for testing equality, as equal KJT's/JT's can be formed from just using lengths or offsets.
    One can be derived from the other so to ensure properly equality checking we force the computation of
    the other attribute if it can be done.
    """
    offsets = kjt.offsets_or_none()
    lengths = kjt.lengths_or_none()
    if offsets is not None and lengths is None:
        kjt.lengths()
    elif lengths is not None and offsets is None:
        kjt.offsets()


def _check_attributes(
    attr_1: Union[torch.Tensor, List[int], List[str], int, None],
    attr_2: Union[torch.Tensor, List[int], List[str], int, None],
    comparison_func: Callable[[Any, Any], bool],  # pyre-ignore[2]
) -> bool:
    """Helper function to check if two attributes are equal.

    Args:
        attr_1: The first attribute.
        attr_2: The second attribute.
        comparison_func (function): Function to compare the attributes.

    Returns:
        bool: False if the attributes are not equal or one is None while the other isn't, otherwise True.
    """
    if attr_1 is not None and attr_2 is not None:
        # allclose throws error for different tensor sizes, we check manually for this
        if (
            comparison_func == torch.allclose
            and attr_1.size() != attr_2.size()  # pyre-ignore[16]
        ):
            return False
        if not comparison_func(attr_1, attr_2):
            return False
    elif attr_1 is not None or attr_2 is not None:
        return False

    return True


def _maybe_compute_lengths_offset_per_key(
    lengths_offset_per_key: Optional[List[int]],
    stride_per_key: Optional[List[int]],
    stride: Optional[int],
    keys: List[str],
) -> Optional[List[int]]:
    if lengths_offset_per_key is not None:
        return lengths_offset_per_key
    elif stride_per_key is not None:
        return _cumsum(stride_per_key)
    elif stride is not None:
        return _cumsum([stride] * len(keys))
    else:
        return None


def _maybe_compute_stride_per_key(
    stride_per_key: Optional[List[int]],
    stride_per_key_per_rank: Optional[torch.IntTensor],
    stride: Optional[int],
    keys: List[str],
) -> Optional[List[int]]:
    if stride_per_key is not None:
        return stride_per_key
    elif stride_per_key_per_rank is not None:
        if stride_per_key_per_rank.dim() != 2:
            # after permute the kjt could be empty
            return []
        rt: List[int] = stride_per_key_per_rank.sum(dim=1).tolist()
        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            pt2_checks_all_is_size(rt)
        return rt
    elif stride is not None:
        return [stride] * len(keys)
    else:
        return None


class KeyedJaggedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """Represents an (optionally weighted) keyed jagged tensor.

    A `KeyedJaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. Keyed on first dimension and jagged on the last
    dimension.

    Implementation is torch.jit.script-able.

    Args:
        keys (List[str]): keys to the jagged Tensor.
        values (torch.Tensor): values tensor in dense representation.
        weights (Optional[torch.Tensor]): if the values have weights. Tensor with the
            same shape as values.
        lengths (Optional[torch.Tensor]): jagged slices, represented as lengths.
        offsets (Optional[torch.Tensor]): jagged slices, represented as cumulative
            offsets.
        stride (Optional[int]): number of examples per batch.
        stride_per_key_per_rank (Optional[Union[torch.IntTensor, List[List[int]]]]):
            batch size (number of examples) per key per rank, with the outer list
            representing the keys and the inner list representing the values.
            Each value in the inner list represents the number of examples in the batch
            from the rank of its index in a distributed context.
        length_per_key (Optional[List[int]]): start length for each key.
        offset_per_key (Optional[List[int]]): start offset for each key and final
            offset.
        index_per_key (Optional[Dict[str, int]]): index for each key.
        jt_dict (Optional[Dict[str, JaggedTensor]]): dictionary of keys to JaggedTensors.
            Allow ability to make to_dict() lazy/cacheable.
        inverse_indices (Optional[Tuple[List[str], torch.Tensor]]): inverse indices to
            expand deduplicated embedding output for variable stride per key.

    Example::

        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        dim_0: keyed dimension (ie. `Feature0`, `Feature1`)
        dim_1: optional second dimension (ie. batch size)
        dim_2: The jagged dimension which has slice lengths between 0-3 in the above example

        # We represent this data with following inputs:

        values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7]  # V == any tensor datatype
        weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7]  # W == any tensor datatype
        lengths: torch.Tensor = [2, 0, 1, 1, 1, 3]  # representing the jagged slice
        offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8]  # offsets from 0 for each jagged slice
        keys: List[str] = ["Feature0", "Feature1"]  # correspond to each value of dim_0
        index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1": 1}  # index for each key
        offset_per_key: List[int] = [0, 3, 8]  # start offset for each key and final offset
    """

    # This is the subset of fields on KJT which are required (all other fields
    # can be derived from these fields, and are only cached)
    _fields = [
        "_values",
        "_weights",
        "_lengths",
        "_offsets",
        "_stride_per_key_per_rank",
        "_inverse_indices",
    ]

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[
            Union[torch.IntTensor, List[List[int]]]
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
        """
        This is the constructor for KeyedJaggedTensor is jit.scriptable and PT2 compatible.
        It is important only to assign attributes here or do input checks to support various
        internal inference optimizations.  By convention the attirbute is named same as input arg, just
        with leading underscore
        """
        self._keys: List[str] = keys
        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets
        self._stride: Optional[int] = stride
        if not torch.jit.is_scripting() and is_torchdynamo_compiling():
            # in pt2.compile the stride_per_key_per_rank has to be torch.Tensor or None
            # does not take List[List[int]]
            assert not isinstance(stride_per_key_per_rank, list)

        self._stride_per_key_per_rank: Optional[torch.IntTensor] = (
            torch.IntTensor(stride_per_key_per_rank, device="cpu")
            if isinstance(stride_per_key_per_rank, list)
            else stride_per_key_per_rank
        )
        self._stride_per_key: Optional[List[int]] = stride_per_key
        self._length_per_key: Optional[List[int]] = length_per_key
        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._lengths_offset_per_key: Optional[List[int]] = lengths_offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key
        self._jt_dict: Optional[Dict[str, JaggedTensor]] = jt_dict
        self._inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = (
            inverse_indices
        )
        # this is only needed for torch.compile case
        self._pt2_stride_per_key_per_rank: Optional[List[List[int]]] = None

        # legacy attribute, for backward compatabilibity
        self._variable_stride_per_key: Optional[bool] = None

        # validation logic
        if not torch.jit.is_scripting():
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
            self._init_pt2_checks()

    def _init_pt2_checks(self) -> None:
        if torch.jit.is_scripting() or not is_torchdynamo_compiling():
            return
        if self._stride_per_key is not None:
            pt2_checks_all_is_size(self._stride_per_key)

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
        """
        Constructs a KeyedJaggedTensor from a list of keys, values, and offsets.

        Args:
            keys (List[str]): list of keys.
            values (torch.Tensor): values tensor in dense representation.
            offsets (torch.Tensor): jagged slices, represented as cumulative offsets.
            weights (Optional[torch.Tensor]): if the values have weights. Tensor with the
                same shape as values.
            stride (Optional[int]): number of examples per batch.
            stride_per_key_per_rank (Optional[List[List[int]]]): batch size
                (number of examples) per key per rank, with the outer list representing the
                keys and the inner list representing the values.
            inverse_indices (Optional[Tuple[List[str], torch.Tensor]]): inverse indices to
                expand deduplicated embedding output for variable stride per key.

        Returns:
            KeyedJaggedTensor: constructed KeyedJaggedTensor.
        """
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

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
        """
        Constructs a KeyedJaggedTensor from a list of keys, lengths, and offsets.
        Same as `from_offsets_sync` except lengths are used instead of offsets.

        Args:
            keys (List[str]): list of keys.
            values (torch.Tensor): values tensor in dense representation.
            lengths (torch.Tensor): jagged slices, represented as lengths.
            weights (Optional[torch.Tensor]): if the values have weights. Tensor with the
                same shape as values.
            stride (Optional[int]): number of examples per batch.
            stride_per_key_per_rank (Optional[List[List[int]]]): batch size
                (number of examples) per key per rank, with the outer list representing the
                keys and the inner list representing the values.
            inverse_indices (Optional[Tuple[List[str], torch.Tensor]]): inverse indices to
                expand deduplicated embedding output for variable stride per key.

        Returns:
            KeyedJaggedTensor: constructed KeyedJaggedTensor.
        """
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=inverse_indices,
        )
        return kjt.sync()

    @staticmethod
    def concat(
        kjt_list: List["KeyedJaggedTensor"],
    ) -> "KeyedJaggedTensor":
        """
        Concatenates a list of KeyedJaggedTensors into a single KeyedJaggedTensor.

        Args:
            kjt_list (List[KeyedJaggedTensor]): list of KeyedJaggedTensors to be concatenated.

        Returns:
            KeyedJaggedTensor: concatenated KeyedJaggedTensor.
        """
        return _kjt_concat(kjt_list)

    @staticmethod
    def empty(
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        values_dtype: Optional[torch.dtype] = None,
        weights_dtype: Optional[torch.dtype] = None,
        lengths_dtype: torch.dtype = torch.int32,
    ) -> "KeyedJaggedTensor":
        """
        Constructs an empty KeyedJaggedTensor.

        Args:
            is_weighted (bool): whether the KeyedJaggedTensor is weighted or not.
            device (Optional[torch.device]): device on which the KeyedJaggedTensor will be placed.
            values_dtype (Optional[torch.dtype]): dtype of the values tensor.
            weights_dtype (Optional[torch.dtype]): dtype of the weights tensor.
            lengths_dtype (torch.dtype): dtype of the lengths tensor.

        Returns:
            KeyedJaggedTensor: empty KeyedJaggedTensor.
        """
        weights = (
            torch.empty(0, dtype=weights_dtype, device=device) if is_weighted else None
        )
        return KeyedJaggedTensor(
            keys=torch.jit.annotate(List[str], []),
            values=torch.empty(0, dtype=values_dtype, device=device),
            weights=weights,
            lengths=torch.empty(0, dtype=lengths_dtype, device=device),
            stride=0,
        )

    @staticmethod
    def empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        """
        Constructs an empty KeyedJaggedTensor with the same device and dtypes as the input KeyedJaggedTensor.

        Args:
            kjt (KeyedJaggedTensor): input KeyedJaggedTensor.

        Returns:
            KeyedJaggedTensor: empty KeyedJaggedTensor.
        """
        return _kjt_empty_like(kjt)

    @staticmethod
    def from_jt_dict(jt_dict: Dict[str, JaggedTensor]) -> "KeyedJaggedTensor":
        """
        Constructs a KeyedJaggedTensor from a dictionary of JaggedTensors.
        Automatically calls `kjt.sync()` on newly created KJT.

        NOTE:
            This function will ONLY work if the JaggedTensors all
            have the same "implicit" batch_size dimension.

        Basically, we can visualize JaggedTensors as 2-D tensors
        of the format of [batch_size x variable_feature_dim].
        In the case, we have some batch without a feature value,
        the input JaggedTensor could just not include any values.

        But KeyedJaggedTensor (by default) typically pad "None"
        so that all the JaggedTensors stored in the KeyedJaggedTensor
        have the same batch_size dimension. That is, in the case,
        the JaggedTensor input didn't automatically pad
        for the empty batches, this function would error / not work.

        Consider the visualization of the following KeyedJaggedTensor:
        #              0       1        2  <-- dim_1
        # "Feature0"   [V0,V1] None    [V2]
        # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        #   ^
        #  dim_0

        Now if the input jt_dict = {
            # "Feature0"   [V0,V1] [V2]
            # "Feature1"   [V3]    [V4]    [V5,V6,V7]
        } and the "None" is left out from each JaggedTensor,
        then this function would fail as we would not correctly
        be able to pad "None" as it does not technically know
        the correct batch / place to pad within the JaggedTensor.

        Essentially, the lengths Tensor inferred by this function
        would be [2, 1, 1, 1, 3] indicating variable batch_size
        dim_1 violates the existing assumption / precondition
        that KeyedJaggedTensor's should have fixed batch_size dimension.

        Args:
            jt_dict (Dict[str, JaggedTensor]): dictionary of JaggedTensors.

        Returns:
            KeyedJaggedTensor: constructed KeyedJaggedTensor.
        """
        kjt_keys = list(jt_dict.keys())
        kjt_vals_list: List[torch.Tensor] = []
        kjt_lens_list: List[torch.Tensor] = []
        kjt_weights_list: List[torch.Tensor] = []
        stride_per_key: List[int] = []
        for jt in jt_dict.values():
            stride_per_key.append(len(jt.lengths()))
            kjt_vals_list.append(jt.values())
            kjt_lens_list.append(jt.lengths())
            weight = jt.weights_or_none()
            if weight is not None:
                kjt_weights_list.append(weight)
        kjt_vals = torch.concat(kjt_vals_list)
        kjt_lens = torch.concat(kjt_lens_list)
        kjt_weights = (
            torch.concat(kjt_weights_list) if len(kjt_weights_list) > 0 else None
        )
        kjt_stride, kjt_stride_per_key_per_rank = (
            (stride_per_key[0], None)
            if all(s == stride_per_key[0] for s in stride_per_key)
            else (None, torch.IntTensor(stride_per_key, device="cpu").reshape(-1, 1))
        )
        kjt = KeyedJaggedTensor(
            keys=kjt_keys,
            values=kjt_vals,
            weights=kjt_weights,
            lengths=kjt_lens,
            stride=kjt_stride,
            stride_per_key_per_rank=kjt_stride_per_key_per_rank,
        ).sync()
        return kjt

    def sync(self) -> "KeyedJaggedTensor":
        """
        Synchronizes the KeyedJaggedTensor by computing the offset_per_key and length_per_key.

        Returns:
            KeyedJaggedTensor: synced KeyedJaggedTensor.
        """
        if not is_torchdynamo_compiling():
            self.length_per_key()
            self.offset_per_key()
        return self

    def unsync(self) -> "KeyedJaggedTensor":
        """
        Unsyncs the KeyedJaggedTensor by clearing the offset_per_key and length_per_key.

        Returns:
            KeyedJaggedTensor: unsynced KeyedJaggedTensor.
        """
        self._length_per_key = None
        self._offset_per_key = None
        return self

    def device(self) -> torch.device:
        """
        Returns the device of the KeyedJaggedTensor.

        Returns:
            torch.device: device of the KeyedJaggedTensor.
        """
        return self._values.device

    def lengths(self) -> torch.Tensor:
        """
        Returns the lengths of the KeyedJaggedTensor.
        If the lengths are not computed yet, it will compute them.

        Returns:
            torch.Tensor: lengths of the KeyedJaggedTensor.
        """
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def lengths_or_none(self) -> Optional[torch.Tensor]:
        """
        Returns the lengths of the KeyedJaggedTensor or None if they are not computed yet.

        Returns:
            torch.Tensor: lengths of the KeyedJaggedTensor.
        """
        return self._lengths

    def offsets(self) -> torch.Tensor:
        """
        Returns the offsets of the KeyedJaggedTensor.
        If the offsets are not computed yet, it will compute them.

        Returns:
            torch.Tensor: offsets of the KeyedJaggedTensor.
        """
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def offsets_or_none(self) -> Optional[torch.Tensor]:
        """
        Returns the offsets of the KeyedJaggedTensor or None if they are not computed yet.

        Returns:
            torch.Tensor: offsets of the KeyedJaggedTensor.
        """
        return self._offsets

    def keys(self) -> List[str]:
        """
        Returns the keys of the KeyedJaggedTensor.

        Returns:
            List[str]: keys of the KeyedJaggedTensor.
        """
        return self._keys

    def values(self) -> torch.Tensor:
        """
        Returns the values of the KeyedJaggedTensor.

        Returns:
            torch.Tensor: values of the KeyedJaggedTensor.
        """
        return self._values

    def weights(self) -> torch.Tensor:
        """
        Returns the weights of the KeyedJaggedTensor.
        If weights is None, this will throw an error.

        Returns:
            torch.Tensor: weights of the KeyedJaggedTensor.
        """
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        """
        Returns the weights of the KeyedJaggedTensor or None if they don't exist.

        Returns:
            torch.Tensor: weights of the KeyedJaggedTensor.
        """
        return self._weights

    def stride(self) -> int:
        """
        Returns the stride of the KeyedJaggedTensor.
        If stride is None, this will compute it.

        Returns:
            int: stride of the KeyedJaggedTensor.
        """
        stride = _maybe_compute_stride_kjt(
            self._keys,
            self._stride,
            self._lengths,
            self._offsets,
            self._stride_per_key_per_rank,
            self._inverse_indices,
        )
        self._stride = stride
        return stride

    def stride_per_key(self) -> List[int]:
        """
        Returns the stride per key of the KeyedJaggedTensor.
        If stride per key is None, this will compute it.

        Returns:
            List[int]: stride per key of the KeyedJaggedTensor.
        """
        stride_per_key = _maybe_compute_stride_per_key(
            self._stride_per_key,
            self._stride_per_key_per_rank,
            self.stride(),
            self._keys,
        )
        self._stride_per_key = stride_per_key
        return _get_stride_per_key_or_throw(stride_per_key)

    def stride_per_key_per_rank(self) -> List[List[int]]:
        """
        Returns the stride per key per rank of the KeyedJaggedTensor.

        Returns:
            List[List[int]]: stride per key per rank of the KeyedJaggedTensor.
        """
        # making a local reference to the class variable to make jit.script behave
        _stride_per_key_per_rank = self._stride_per_key_per_rank
        if (
            not torch.jit.is_scripting()
            and is_torchdynamo_compiling()
            and _stride_per_key_per_rank is not None
        ):
            if self._pt2_stride_per_key_per_rank is not None:
                return self._pt2_stride_per_key_per_rank
            stride_per_key_per_rank = _stride_per_key_per_rank.tolist()
            for stride_per_rank in stride_per_key_per_rank:
                pt2_checks_all_is_size(stride_per_rank)
            self._pt2_stride_per_key_per_rank = stride_per_key_per_rank
            return stride_per_key_per_rank
        return (
            []
            if _stride_per_key_per_rank is None
            else _stride_per_key_per_rank.tolist()
        )

    def variable_stride_per_key(self) -> bool:
        """
        Returns whether the KeyedJaggedTensor has variable stride per key.
        NOTE: `self._variable_stride_per_key` could be `False` when `self._stride_per_key_per_rank`
        is not `None`. It might be assigned to False externally/intentionally, usually the
        `self._stride_per_key_per_rank` is trivial.

        Returns:
            bool: whether the KeyedJaggedTensor has variable stride per key.
        """
        if self._variable_stride_per_key is not None:
            return self._variable_stride_per_key
        return self._stride_per_key_per_rank is not None

    def inverse_indices(self) -> Tuple[List[str], torch.Tensor]:
        """
        Returns the inverse indices of the KeyedJaggedTensor.
        If inverse indices are None, this will throw an error.

        Returns:
            Tuple[List[str], torch.Tensor]: inverse indices of the KeyedJaggedTensor.
        """
        return _get_inverse_indices_or_throw(self._inverse_indices)

    def inverse_indices_or_none(self) -> Optional[Tuple[List[str], torch.Tensor]]:
        """
        Returns the inverse indices of the KeyedJaggedTensor or None if they don't exist.

        Returns:
            Optional[Tuple[List[str], torch.Tensor]]: inverse indices of the KeyedJaggedTensor.
        """
        return self._inverse_indices

    def _key_indices(self) -> Dict[str, int]:
        _index_per_key: Dict[str, int] = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key = _index_per_key
        return _index_per_key

    def length_per_key(self) -> List[int]:
        """
        Returns the length per key of the KeyedJaggedTensor.
        If length per key is None, this will compute it.

        Returns:
            List[int]: length per key of the KeyedJaggedTensor.
        """
        _length_per_key = _maybe_compute_length_per_key(
            keys=self._keys,
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            variable_stride_per_key=self.variable_stride_per_key(),
            length_per_key=self._length_per_key,
            lengths=self._lengths,
            offsets=self._offsets,
            values=self._values,
        )
        self._length_per_key = _length_per_key
        return _length_per_key

    def length_per_key_or_none(self) -> Optional[List[int]]:
        """
        Returns the length per key of the KeyedJaggedTensor or None if it hasn't been computed.

        Returns:
            List[int]: length per key of the KeyedJaggedTensor.
        """
        return self._length_per_key

    def offset_per_key(self) -> List[int]:
        """
        Returns the offset per key of the KeyedJaggedTensor.
        If offset per key is None, this will compute it.

        Returns:
            List[int]: offset per key of the KeyedJaggedTensor.
        """
        _length_per_key, _offset_per_key = _maybe_compute_offset_per_key(
            keys=self._keys,
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            variable_stride_per_key=self.variable_stride_per_key(),
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            lengths=self._lengths,
            offsets=self._offsets,
            values=self._values,
        )
        self._length_per_key = _length_per_key
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def offset_per_key_or_none(self) -> Optional[List[int]]:
        """
        Returns the offset per key of the KeyedJaggedTensor or None if it hasn't been computed.

        Returns:
            List[int]: offset per key of the KeyedJaggedTensor.
        """
        return self._offset_per_key

    def lengths_offset_per_key(self) -> List[int]:
        """
        Returns the lengths offset per key of the KeyedJaggedTensor.
        If lengths offset per key is None, this will compute it.

        Returns:
            List[int]: lengths offset per key of the KeyedJaggedTensor.
        """
        if self.variable_stride_per_key():
            _lengths_offset_per_key = _maybe_compute_lengths_offset_per_key(
                self._lengths_offset_per_key,
                self.stride_per_key(),
                None,
                self._keys,
            )
        else:
            _lengths_offset_per_key = _maybe_compute_lengths_offset_per_key(
                self._lengths_offset_per_key, None, self.stride(), self._keys
            )

        self._lengths_offset_per_key = _lengths_offset_per_key
        return _get_lengths_offset_per_key_or_throw(_lengths_offset_per_key)

    def index_per_key(self) -> Dict[str, int]:
        """
        Returns the index per key of the KeyedJaggedTensor.

        Returns:
            Dict[str, int]: index per key of the KeyedJaggedTensor.
        """
        return self._key_indices()

    def split(self, segments: List[int]) -> List["KeyedJaggedTensor"]:
        """
        Splits the KeyedJaggedTensor into a list of KeyedJaggedTensor.

        Args:
            segments (List[int]): list of segments.

        Returns:
            List[KeyedJaggedTensor]: list of KeyedJaggedTensor.
        """
        split_list: List[KeyedJaggedTensor] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        # use local copy/ref for self._stride_per_key_per_rank to satisfy jit.script
        _stride_per_key_per_rank = self._stride_per_key_per_rank
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]
            stride_per_key_per_rank = (
                _stride_per_key_per_rank[start:end, :]
                if self.variable_stride_per_key()
                and _stride_per_key_per_rank is not None
                else None
            )
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(self)
            elif segment == 0:
                empty_int_list: List[int] = torch.jit.annotate(List[int], [])
                split_list.append(
                    KeyedJaggedTensor(
                        keys=keys,
                        values=torch.tensor(
                            empty_int_list,
                            device=self.device(),
                            dtype=self._values.dtype,
                        ),
                        weights=(
                            None
                            if self.weights_or_none() is None
                            else torch.tensor(
                                empty_int_list,
                                device=self.device(),
                                dtype=self.weights().dtype,
                            )
                        ),
                        lengths=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        offsets=torch.tensor(
                            empty_int_list, device=self.device(), dtype=torch.int
                        ),
                        stride=self._stride,
                        stride_per_key_per_rank=stride_per_key_per_rank,
                        stride_per_key=None,
                        length_per_key=None,
                        lengths_offset_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                        jt_dict=None,
                        inverse_indices=None,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]

                if not torch.jit.is_scripting() and is_non_strict_exporting():
                    sz = sum(split_length_per_key)

                    [torch._check_is_size(length) for length in split_length_per_key]
                    torch._check(start_offset <= self._values.size(0))
                    torch._check(sz <= self._values.size(0))
                    torch._check_is_size(start_offset)

                    torch._check(start_offset + sz <= self._values.size(0))

                    lengths_start = self.lengths_offset_per_key()[start]
                    lengths_sz = self.lengths_offset_per_key()[end] - lengths_start

                    _lengths = torch.narrow(
                        self.lengths(), 0, lengths_start, lengths_sz
                    )

                    if self.weights_or_none() is not None:
                        torch._check(start_offset + sz <= self.weights().size(0))
                        torch._check(start_offset <= self.weights().size(0))

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=torch.narrow(self._values, 0, start_offset, sz),
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else torch.narrow(self.weights(), 0, start_offset, sz)
                            ),
                            lengths=_lengths,
                            offsets=None,
                            stride=self._stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            stride_per_key=None,
                            length_per_key=split_length_per_key,
                            lengths_offset_per_key=None,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
                else:
                    pt2_checks_tensor_slice(self._values, start_offset, end_offset)

                    lengths_offset_per_key: List[int] = self.lengths_offset_per_key()
                    pt2_checks_tensor_slice(
                        self.lengths(),
                        lengths_offset_per_key[start],
                        lengths_offset_per_key[end],
                    )

                    split_list.append(
                        KeyedJaggedTensor(
                            keys=keys,
                            values=self._values[start_offset:end_offset],
                            weights=(
                                None
                                if self.weights_or_none() is None
                                else self.weights()[start_offset:end_offset]
                            ),
                            lengths=self.lengths()[
                                lengths_offset_per_key[start] : lengths_offset_per_key[
                                    end
                                ]
                            ],
                            offsets=None,
                            stride=self._stride,
                            stride_per_key_per_rank=stride_per_key_per_rank,
                            stride_per_key=None,
                            length_per_key=split_length_per_key,
                            lengths_offset_per_key=None,
                            offset_per_key=None,
                            index_per_key=None,
                            jt_dict=None,
                            inverse_indices=None,
                        )
                    )
            start = end
            start_offset = end_offset
        return split_list

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensor":
        """
        Permutes the KeyedJaggedTensor.

        Args:
            indices (List[int]): list of indices.
            indices_tensor (Optional[torch.Tensor]): tensor of indices.

        Returns:
            KeyedJaggedTensor: permuted KeyedJaggedTensor.
        """
        if indices_tensor is None:
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )

        length_per_key = self.length_per_key()
        permuted_keys: List[str] = []
        permuted_length_per_key: List[int] = []
        permuted_length_per_key_sum = 0
        for index in indices:
            key = self.keys()[index]
            permuted_keys.append(key)
            permuted_length_per_key.append(length_per_key[index])

        stride_per_key = self._stride_per_key
        permuted_stride_per_key = (
            [stride_per_key[i] for i in indices] if stride_per_key is not None else None
        )

        _stride_per_key_per_rank = self._stride_per_key_per_rank
        permuted_stride_per_key_per_rank = (
            _stride_per_key_per_rank[indices, :]
            if self.variable_stride_per_key() and _stride_per_key_per_rank is not None
            else None
        )

        permuted_length_per_key_sum = sum(permuted_length_per_key)
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            torch._check_is_size(permuted_length_per_key_sum)
            torch._check(permuted_length_per_key_sum != -1)
            torch._check(permuted_length_per_key_sum != 0)

        if self.variable_stride_per_key():
            length_per_key_tensor = _pin_and_move(
                torch.tensor(self.length_per_key()), self.device()
            )
            stride_per_key_tensor = _pin_and_move(
                torch.tensor(self.stride_per_key()), self.device()
            )
            permuted_lengths, _ = _permute_tensor_by_segments(
                self.lengths(),
                stride_per_key_tensor,
                indices_tensor,
                None,
            )
            permuted_values, permuted_weights = _permute_tensor_by_segments(
                self.values(),
                length_per_key_tensor,
                indices_tensor,
                self.weights_or_none(),
            )
        elif is_torchdynamo_compiling() and not torch.jit.is_scripting():
            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                indices_tensor,
                self.lengths(),
                self.values(),
                self.stride(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )
        else:
            (
                permuted_lengths,
                permuted_values,
                permuted_weights,
            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                indices_tensor,
                self.lengths().view(len(self._keys), -1),
                self.values(),
                self.weights_or_none(),
                permuted_length_per_key_sum,
            )

        kjt = KeyedJaggedTensor(
            keys=permuted_keys,
            values=permuted_values,
            weights=permuted_weights,
            lengths=permuted_lengths.view(-1),
            offsets=None,
            stride=self._stride,
            stride_per_key_per_rank=permuted_stride_per_key_per_rank,
            stride_per_key=permuted_stride_per_key,
            length_per_key=permuted_length_per_key if len(permuted_keys) > 0 else None,
            lengths_offset_per_key=None,
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )
        return kjt

    def flatten_lengths(self) -> "KeyedJaggedTensor":
        stride_per_key_per_rank = (
            self._stride_per_key_per_rank if self.variable_stride_per_key() else None
        )
        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values,
            weights=self._weights,
            lengths=self.lengths().view(-1),
            offsets=None,
            stride=self._stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=None,
            length_per_key=self.length_per_key(),
            lengths_offset_per_key=None,
            offset_per_key=None,
            index_per_key=None,
            jt_dict=None,
            inverse_indices=None,
        )

    def __getitem__(self, key: str) -> JaggedTensor:
        """
        Returns the JaggedTensor for the given key.

        Args:
            key (str): key.

        Returns:
            JaggedTensor: JaggedTensor for the given key.
        """
        offset_per_key = self.offset_per_key()
        index = self._key_indices()[key]
        start_offset = offset_per_key[index]
        end_offset = (
            offset_per_key[index + 1]
            if index + 1 < len(offset_per_key)
            else start_offset
        )

        if not torch.jit.is_scripting() and is_non_strict_exporting():
            length_per_key = self.length_per_key()
            _lengths = torch.narrow(
                self.lengths(),
                0,
                self.lengths_offset_per_key()[index],
                self.lengths_offset_per_key()[index + 1]
                - self.lengths_offset_per_key()[index],
            )
            sz = length_per_key[index]

            torch._check_is_size(start_offset)
            torch._check_is_size(sz)
            torch._check(start_offset <= self.values().size(0))
            torch._check(sz <= self.values().size(0))

            if self.weights_or_none() is not None:
                torch._check(start_offset <= self.weights().size(0))
                torch._check(sz <= self.weights().size(0))

            return JaggedTensor(
                values=torch.narrow(
                    self.values(),
                    0,
                    start_offset,
                    sz,
                ),
                weights=(
                    None
                    if self.weights_or_none() is None
                    else torch.narrow(
                        self.weights(),
                        0,
                        start_offset,
                        sz,
                    )
                ),
                lengths=_lengths,
                offsets=None,
            )
        else:
            pt2_checks_tensor_slice(self._values, start_offset, end_offset)

            return JaggedTensor(
                values=self._values[start_offset:end_offset],
                weights=(
                    None
                    if self.weights_or_none() is None
                    else self.weights()[start_offset:end_offset]
                ),
                lengths=self.lengths()[
                    self.lengths_offset_per_key()[
                        index
                    ] : self.lengths_offset_per_key()[index + 1]
                ],
                offsets=None,
            )

    def to_dict(self) -> Dict[str, JaggedTensor]:
        """
        Returns a dictionary of JaggedTensor for each key.
        Will cache result in self._jt_dict.

        Returns:
            Dict[str, JaggedTensor]: dictionary of JaggedTensor for each key.
        """
        if not torch.jit.is_scripting() and is_non_strict_exporting():
            logger.warning(
                "Trying to non-strict torch.export KJT to_dict, which is extremely slow and not recommended!"
            )
        _jt_dict = _maybe_compute_kjt_to_jt_dict(
            stride=self.stride(),
            stride_per_key=self.stride_per_key(),
            keys=self.keys(),
            length_per_key=self.length_per_key(),
            lengths=self.lengths(),
            values=self.values(),
            variable_stride_per_key=self.variable_stride_per_key(),
            weights=self.weights_or_none(),
            jt_dict=self._jt_dict,
        )
        self._jt_dict = _jt_dict
        return _jt_dict

    @torch.jit.unused
    # pyre-fixme[14]: `record_stream` overrides method defined in `Multistreamable`
    #  inconsistently.
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> "KeyedJaggedTensor":
        """
        Returns a copy of KeyedJaggedTensor in the specified device and dtype.

        Args:
            device (torch.device): the desired device of the copy.
            non_blocking (bool): whether to copy the tensors in a non-blocking fashion.
            dtype (Optional[torch.dtype]): the desired data type of the copy.

        Returns:
            KeyedJaggedTensor: the copied KeyedJaggedTensor.
        """
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride_per_key_per_rank = (
            self._stride_per_key_per_rank if self.variable_stride_per_key() else None
        )
        length_per_key = self._length_per_key
        lengths_offset_per_key = self._lengths_offset_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key
        stride_per_key = self._stride_per_key
        jt_dict = self._jt_dict
        inverse_indices = self._inverse_indices
        if inverse_indices is not None:
            inverse_indices = (
                inverse_indices[0],
                inverse_indices[1].to(device, non_blocking=non_blocking),
            )
        if weights is not None:
            if dtype is not None:
                weights = weights.to(
                    dtype=dtype, device=device, non_blocking=non_blocking
                )
            else:
                weights = weights.to(device=device, non_blocking=non_blocking)

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            weights=weights,
            lengths=(
                lengths.to(device, non_blocking=non_blocking)
                if lengths is not None
                else None
            ),
            offsets=(
                offsets.to(device, non_blocking=non_blocking)
                if offsets is not None
                else None
            ),
            stride=self._stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=stride_per_key,
            length_per_key=length_per_key,
            lengths_offset_per_key=lengths_offset_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            inverse_indices=inverse_indices,
        )

    def __str__(self) -> str:
        if len(self._keys) == 0 or self._offsets is None and self._lengths is None:
            return "KeyedJaggedTensor()\n"
        offsets = self.offsets()

        return (
            "KeyedJaggedTensor({\n"
            + ",\n".join(
                [
                    "    "
                    + _jagged_tensor_string(
                        self._keys[index],
                        self._values,
                        self._weights,
                        offsets,
                        sum(self.stride_per_key()[:index]),
                        sum(self.stride_per_key()[: index + 1]),
                    )
                    for index in range(len(self._keys))
                ]
            )
            + "\n})\n"
        )

    def pin_memory(self) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        stride_per_key_per_rank = (
            self._stride_per_key_per_rank if self.variable_stride_per_key() else None
        )
        inverse_indices = self._inverse_indices
        if inverse_indices is not None:
            inverse_indices = (inverse_indices[0], inverse_indices[1].pin_memory())

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.pin_memory(),
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=self._stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=self._stride_per_key,
            length_per_key=self._length_per_key,
            lengths_offset_per_key=self._lengths_offset_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
            jt_dict=None,
            inverse_indices=inverse_indices,
        )

    def dist_labels(self) -> List[str]:
        labels = ["lengths", "values"]
        if self.variable_stride_per_key():
            labels.append("strides")
        if self.weights_or_none() is not None:
            labels.append("weights")
        return labels

    def dist_splits(self, key_splits: List[int]) -> List[List[int]]:
        batch_size_per_split = _sum_by_splits(self.stride_per_key(), key_splits)
        length_per_split = _sum_by_splits(self.length_per_key(), key_splits)
        splits = [batch_size_per_split, length_per_split]
        if self.variable_stride_per_key():
            splits.append(key_splits)
        if self.weights_or_none() is not None:
            splits.append(length_per_split)
        return splits

    def dist_tensors(self) -> List[torch.Tensor]:
        tensors = [self.lengths(), self.values()]
        if self.variable_stride_per_key():
            strides = _pin_and_move(torch.tensor(self.stride_per_key()), self.device())
            tensors.append(strides)
        if self.weights_or_none() is not None:
            tensors.append(self.weights())
        return tensors

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
        assert len(tensors) in [2, 3, 4]
        lengths = tensors[0]
        values = tensors[1]
        stride_per_rank_per_key = tensors[2] if variable_stride_per_key else None
        weights = (
            tensors[-1]
            if (variable_stride_per_key and len(tensors) == 4)
            or (not variable_stride_per_key and len(tensors) == 3)
            else None
        )

        if variable_stride_per_key:
            assert stride_per_rank_per_key is not None
            stride_per_key_per_rank: torch.Tensor = stride_per_rank_per_key.view(
                num_workers, len(keys)
            ).T.cpu()

            strides_cumsum: torch.Tensor = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(stride_per_rank_per_key)
            ).cpu()

            cumsum_lengths = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

            n = strides_cumsum.size(0)
            strides_cumsum_from_1 = torch.narrow(
                strides_cumsum, dim=0, start=1, length=n - 1
            )
            strides_cumsum_to_minus_1 = torch.narrow(
                strides_cumsum, dim=0, start=0, length=n - 1
            )
            length_per_key_tensor = (
                cumsum_lengths[strides_cumsum_from_1]
                - cumsum_lengths[strides_cumsum_to_minus_1]
            )

            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    lengths, _ = _permute_tensor_by_segments(
                        lengths,
                        stride_per_rank_per_key,
                        torch.jit._unwrap_optional(recat),
                        None,
                    )
                    values, weights = _permute_tensor_by_segments(
                        values,
                        length_per_key_tensor,
                        torch.jit._unwrap_optional(recat),
                        weights,
                    )

            if stride_per_key_per_rank.numel() == 0:
                stride_per_key_per_rank = torch.zeros(
                    (len(keys), 1), device="cpu", dtype=torch.int64
                )

            if stagger > 1:
                indices = torch.arange(num_workers).view(stagger, -1).T.reshape(-1)
                stride_per_key_per_rank = stride_per_key_per_rank[:, indices]

            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                weights=weights,
                lengths=lengths,
                stride_per_key_per_rank=stride_per_key_per_rank,
            )
            return kjt.sync()
        else:
            assert stride_per_rank is not None
            with record_function("## all2all_data:recat_values ##"):
                if recat is not None:
                    stride = stride_per_rank[0]

                    single_batch_per_rank = True
                    if not is_torchdynamo_compiling():
                        single_batch_per_rank = all(
                            s == stride for s in stride_per_rank
                        )
                    if (
                        single_batch_per_rank
                        and is_torchdynamo_compiling()
                        and not torch.jit.is_scripting()
                    ):
                        (
                            lengths,
                            values,
                            weights,
                        ) = torch.ops.fbgemm.permute_2D_sparse_data_input1D(
                            torch.jit._unwrap_optional(recat),
                            lengths,
                            values,
                            stride,
                            weights,
                            values.numel(),
                        )
                    elif single_batch_per_rank:
                        if stride > 0:
                            (
                                lengths,
                                values,
                                weights,
                            ) = torch.ops.fbgemm.permute_2D_sparse_data(
                                torch.jit._unwrap_optional(recat),
                                lengths.view(-1, stride),
                                values,
                                weights,
                                values.numel(),
                            )
                            lengths = lengths.view(-1)
                    else:  # variable batch size per rank
                        (
                            lengths,
                            values,
                            weights,
                        ) = torch.ops.fbgemm.permute_1D_sparse_data(
                            torch.jit._unwrap_optional(recat),
                            lengths.view(-1),
                            values,
                            weights,
                            values.numel(),
                        )
            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                weights=weights,
                lengths=lengths,
                stride=sum(stride_per_rank),
            )
            return kjt.sync()


def _kjt_flatten(
    t: KeyedJaggedTensor,
) -> Tuple[List[Optional[torch.Tensor]], List[str]]:
    """
    Used by PyTorch's pytree utilities for serialization and processing.
    Extracts tensor attributes of a KeyedJaggedTensor and returns them
    as a flat list, along with the necessary metadata to reconstruct the KeyedJaggedTensor.

    Component tensors are returned as dynamic attributes.
    KJT metadata are added as static specs.

    Returns:
        Tuple containing:
            - List[Optional[torch.Tensor]]: All tensor attributes (_values, _weights, _lengths,
              _offsets, _stride_per_key_per_rank, and the tensor part of _inverse_indices if present)
            - Tuple[List[str], List[str]]: Metadata needed for reconstruction:
                - List of keys from the original KeyedJaggedTensor
                - List of inverse indices keys (if present, otherwise empty list)
    """
    values = [getattr(t, a) for a in KeyedJaggedTensor._fields[:-1]]
    values.append(t._inverse_indices[1] if t._inverse_indices is not None else None)

    return values, t._keys


def _kjt_flatten_with_keys(
    t: KeyedJaggedTensor,
) -> Tuple[List[Tuple[KeyEntry, Optional[torch.Tensor]]], List[str]]:
    values, context = _kjt_flatten(t)
    # pyre can't tell that GetAttrKey implements the KeyEntry protocol
    return [  # pyre-ignore[7]
        (GetAttrKey(k), v) for k, v in zip(KeyedJaggedTensor._fields, values)
    ], context


def _kjt_unflatten(
    values: List[Optional[torch.Tensor]],
    context: List[str],  # context is _keys
) -> KeyedJaggedTensor:
    if len(values) < len(KeyedJaggedTensor._fields):
        values.extend([None] * (len(KeyedJaggedTensor._fields) - len(values)))
    elif len(values) > len(KeyedJaggedTensor._fields):
        raise ValueError(
            f"Too many values provided for KeyedJaggedTensor: {len(values)} vs {len(KeyedJaggedTensor._fields)}"
        )
    return KeyedJaggedTensor(
        context,
        *values[:-2],
        stride_per_key_per_rank=values[-2],
        inverse_indices=(context, values[-1]) if values[-1] is not None else None,
    )


def _kjt_flatten_spec(
    t: KeyedJaggedTensor, spec: TreeSpec
) -> List[Optional[torch.Tensor]]:
    values = [getattr(t, a) for a in KeyedJaggedTensor._fields[:-1]]
    values.append(t._inverse_indices[1] if t._inverse_indices is not None else None)

    return values


register_pytree_node(
    KeyedJaggedTensor,
    _kjt_flatten,
    _kjt_unflatten,
    flatten_with_keys_fn=_kjt_flatten_with_keys,
    serialized_type_name="torchrec.sparse.jagged_tensor.KeyedJaggedTensor",
)
register_pytree_flatten_spec(KeyedJaggedTensor, _kjt_flatten_spec)


def flatten_kjt_list(
    kjt_arr: List[KeyedJaggedTensor],
) -> Tuple[List[Optional[torch.Tensor]], List[List[str]]]:
    _flattened_data = []
    _flattened_context = []
    for t in kjt_arr:
        _values, _context = _kjt_flatten(t)
        _flattened_data.extend(_values)
        _flattened_context.append(_context)
    return _flattened_data, _flattened_context


def unflatten_kjt_list(
    values: List[Optional[torch.Tensor]], contexts: List[List[str]]
) -> List[KeyedJaggedTensor]:
    num_kjt_fields = len(KeyedJaggedTensor._fields)
    length = len(values)
    return [
        _kjt_unflatten(
            values[j * num_kjt_fields : (j + 1) * num_kjt_fields],
            contexts[j],
        )
        for j in range(length // num_kjt_fields)
    ]


def _maybe_compute_offset_per_key_kt(
    length_per_key: List[int],
    offset_per_key: Optional[List[int]],
) -> List[int]:
    if offset_per_key is None:
        offset_per_key = _cumsum(length_per_key)
    return offset_per_key


def _keyed_values_string(values: torch.Tensor) -> str:
    return (
        "["
        + ", ".join([_values_string(value, 0, len(value)) for value in values])
        + "]"
    )


class KeyedTensor(Pipelineable, metaclass=JaggedTensorMeta):
    """
    KeyedTensor holds a concatenated list of dense tensors, each of which can be
    accessed by a key.

    The keyed dimension can be of variable length (length_per_key).
    Common use cases uses include storage of pooled embeddings of different dimensions.

    Implementation is torch.jit.script-able.

    Args:
        keys (List[str]): list of keys.
        length_per_key (List[int]): length of each key along key dimension.
        values (torch.Tensor): dense tensor, concatenated typically along key dimension.
        key_dim (int): key dimension, zero indexed - defaults to 1
            (typically B is 0-dimension).

    Example::

        # kt is KeyedTensor holding

        #                         0           1           2
        #     "Embedding A"    [1,1]       [1,1]        [1,1]
        #     "Embedding B"    [2,1,2]     [2,1,2]      [2,1,2]
        #     "Embedding C"    [3,1,2,3]   [3,1,2,3]    [3,1,2,3]

        tensor_list = [
            torch.tensor([[1,1]] * 3),
            torch.tensor([[2,1,2]] * 3),
            torch.tensor([[3,1,2,3]] * 3),
        ]

        keys = ["Embedding A", "Embedding B", "Embedding C"]

        kt = KeyedTensor.from_tensor_list(keys, tensor_list)

        kt.values()
        # torch.Tensor(
        #     [
        #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
        #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
        #         [1, 1, 2, 1, 2, 3, 1, 2, 3],
        #     ]
        # )

        kt["Embedding B"]
        # torch.Tensor([[2, 1, 2], [2, 1, 2], [2, 1, 2]])
    """

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
        self._keys = keys
        self._length_per_key = length_per_key
        self._values = values
        self._key_dim = key_dim

        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key

    @staticmethod
    def from_tensor_list(
        keys: List[str], tensors: List[torch.Tensor], key_dim: int = 1, cat_dim: int = 1
    ) -> "KeyedTensor":
        """
        Create a KeyedTensor from a list of tensors. The tensors are concatenated
        along the cat_dim. The keys are used to index the tensors.

        Args:
            keys (List[str]): list of keys.
            tensors (List[torch.Tensor]): list of tensors.
            key_dim (int): key dimension, zero indexed - defaults to 1
                (typically B is 0-dimension).
            cat_dim (int): dimension along which to concatenate the tensors - defaults

        Returns:
            KeyedTensor: keyed tensor.
        """
        length_per_key = [tensor.shape[key_dim] for tensor in tensors]
        return KeyedTensor(
            keys=keys,
            length_per_key=length_per_key,
            values=torch.cat(tensors, dim=cat_dim),
            key_dim=key_dim,
        )

    def keys(self) -> List[str]:
        """
        Returns:
            List[str]: list of keys.
        """
        return self._keys

    def values(self) -> torch.Tensor:
        """
        Get the values tensor.

        Returns:
            torch.Tensor: dense tensor, concatenated typically along key dimension.
        """
        return self._values

    def key_dim(self) -> int:
        """
        Returns:
            int: key dimension, zero indexed - typically B is 0-dimension.
        """
        return self._key_dim

    def device(self) -> torch.device:
        """
        Returns:
            torch.device: device of the values tensor.
        """
        return self._values.device

    def offset_per_key(self) -> List[int]:
        """
        Get the offset of each key along key dimension.
        Compute and cache if not already computed.

        Returns:
            List[int]: offset of each key along key dimension.
        """
        _offset_per_key = _maybe_compute_offset_per_key_kt(
            self._length_per_key,
            self._offset_per_key,
        )
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def length_per_key(self) -> List[int]:
        """
        Returns:
            List[int]: length of each key along key dimension.
        """
        return self._length_per_key

    def _key_indices(self) -> Dict[str, int]:
        """
        Get the indices of each key.
        Compute and cache if not already computed.

        Returns:
            Dict[str, int]: indices of each key.
        """
        _index_per_key = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key = _index_per_key
        return _index_per_key

    def __getitem__(self, key: str) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: tensor for the given key.
        """
        index = self._key_indices()[key]
        start = self.offset_per_key()[index]
        length = self._length_per_key[index]
        return self._values.narrow(dim=self._key_dim, start=start, length=length)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict[str, torch.Tensor]: dictionary of tensors keyed by the keys.
        """
        indices = self._key_indices()
        lengths = self._length_per_key
        split_values = self._values.split(lengths, dim=self._key_dim)
        return {key: split_values[index] for (key, index) in indices.items()}

    @staticmethod
    def regroup(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]]
    ) -> List[torch.Tensor]:
        """
        Regroup a list of KeyedTensors into a list of tensors.

        Args:
            keyed_tensors (List[KeyedTensor]): list of KeyedTensors.
            groups (List[List[str]]): list of groups of keys.

        Returns:
            List[torch.Tensor]: list of tensors.
        """
        # Fast path, one-to-one correspondence between keyed_tensors and groups
        if _all_keys_used_once(keyed_tensors, groups) is True:
            return _fbgemm_permute_pooled_embs(keyed_tensors, groups)
        else:  # Fallback to slow path otherwise
            return _regroup_keyed_tensors(keyed_tensors, groups)

    @staticmethod
    def regroup_as_dict(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]], keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Regroup a list of KeyedTensors into a dictionary of tensors.

        Args:
            keyed_tensors (List[KeyedTensor]): list of KeyedTensors.
            groups (List[List[str]]): list of groups of keys.
            keys (List[str]): list of keys.

        Returns:
            Dict[str, torch.Tensor]: dictionary of tensors.
        """
        ret: Dict[str, torch.Tensor] = {}
        assert len(groups) == len(keys), "Groups and keys should have same length"
        tensor_list = KeyedTensor.regroup(keyed_tensors, groups)
        for i, key in enumerate(keys):
            ret[key] = tensor_list[i]
        return ret

    @torch.jit.unused
    # pyre-fixme[14]: `record_stream` overrides method defined in `Multistreamable`
    #  inconsistently.
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)

    def to(self, device: torch.device, non_blocking: bool = False) -> "KeyedTensor":
        """
        Moves the values tensor to the specified device.

        Args:
            device (torch.device): device to move the values tensor to.
            non_blocking (bool): whether to perform the operation asynchronously
                (default: False).

        Returns:
            KeyedTensor: keyed tensor with values tensor moved to the specified device.
        """
        return KeyedTensor(
            keys=self._keys,
            length_per_key=self._length_per_key,
            values=self._values.to(device, non_blocking=non_blocking),
            key_dim=self._key_dim,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
        )

    def __str__(self) -> str:
        if len(self._keys) == 0:
            return "KeyedTensor()\n"

        return (
            "KeyedTensor({\n"
            + ",\n".join(
                [
                    '    "{}": '.format(key) + _keyed_values_string(self[key])
                    for key in self._keys
                ]
            )
            + "\n})\n"
        )


def _kt_flatten(
    kt: KeyedTensor,
) -> Tuple[List[torch.Tensor], Tuple[List[str], List[int]]]:
    return [kt._values], (kt._keys, kt._length_per_key)


def _kt_unflatten(
    values: List[torch.Tensor], context: Tuple[List[str], List[int]]
) -> KeyedTensor:
    return KeyedTensor(context[0], context[1], values[0])


def _kt_flatten_spec(kt: KeyedTensor, spec: TreeSpec) -> List[torch.Tensor]:
    _keys, _length_per_key = spec.context
    #  please read https://fburl.com/workplace/8bei5iju for more context,
    #  you can also consider use short_circuit_pytree_ebc_regroup with KTRegroupAsDict
    logger.warning(
        "KT's key order might change from spec from the torch.export, this could have perf impact. "
        f"{kt.keys()} vs {_keys}"
    )
    res = permute_multi_embedding([kt], [_keys])
    return [res[0]]


# The assumption here in torch.exporting KeyedTensor is that _length_per_key is static
register_pytree_node(
    KeyedTensor, _kt_flatten, _kt_unflatten, serialized_type_name="KeyedTensor"
)
register_pytree_flatten_spec(KeyedTensor, _kt_flatten_spec)
