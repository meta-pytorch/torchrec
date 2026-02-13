#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional

import torch
from torchrec.modules.object_pool import ObjectPool
from torchrec.modules.object_pool_lookups import (
    KeyedJaggedTensorPoolLookup,
    TensorJaggedIndexSelectLookup,
    UVMCachingInt64Lookup,
)
from torchrec.modules.utils import deterministic_dedup, jagged_index_select_with_empty
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


@torch.fx.wrap
def _fx_assert_device(ids: torch.Tensor, device: torch.device) -> None:
    assert ids.device == device
    assert ids.dtype in [torch.int32, torch.int64]


@torch.fx.wrap
def _fx_wrap_lookup(
    ids: torch.Tensor,
    keys: List[str],
    feature_max_lengths: Dict[str, int],
    is_weighted: bool,
    values_dtype: torch.dtype,
    device: torch.device,
    lookup: TensorJaggedIndexSelectLookup,  # This type enforement is a hack to make it work with torch.jit.script
    weigth_dtype: Optional[torch.dtype] = None,
) -> KeyedJaggedTensor:
    jt_lookup: JaggedTensor = lookup.lookup(ids)

    row_major_to_feature_major_permute = (
        torch.arange((ids.shape[0] * len(feature_max_lengths)), device=device)
        .view(-1, len(feature_max_lengths))
        .t()
        .flatten()
    )

    lengths = jt_lookup.lengths().flatten()[row_major_to_feature_major_permute]
    output_offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(lengths)
    values = jagged_index_select_with_empty(
        jt_lookup.values().flatten().unsqueeze(-1),
        row_major_to_feature_major_permute,
        jt_lookup.offsets().flatten()[1:],
        output_offsets,
    )
    values, lengths = values.flatten(), lengths.flatten()

    weights = torch.jit.annotate(Optional[torch.Tensor], None)
    if jt_lookup.weights_or_none() is not None:
        weights = jagged_index_select_with_empty(
            jt_lookup.weights().flatten().unsqueeze(-1),
            row_major_to_feature_major_permute,
            jt_lookup.offsets().flatten()[1:],
            output_offsets,
        )
        weights = weights.flatten()

    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=values,
        lengths=lengths,
        weights=weights,
    )


class KeyedJaggedTensorPool(ObjectPool[KeyedJaggedTensor]):
    """
    KeyedJaggedTensorPool represents a collection of KeyedJaggedTensor (KJT)
    with an index over the batch (non-jagged) dimension. For example, if a KJT
    has 2 features "Feature0" and "Feature1" with stride of 2 (i.e. batch dim = 2),
    KeyedJaggedTensorPool allows associating an index with batch 0 of "Feature0" and
    "Feature1". Example:

    #              "Feature0"  "Feature1" < dim_0
    #  batch_0      [V0,V1]       None    <-- associated with index 2
    #  batch_1       [V3]         [V4]    <-- associated with index 0
    #    ^
    #  dim_1

    This is useful when one needs to associate entity IDs to sparse features with
    jagged dimension, for example during hard negative sampling.

    Args:
        pool_size (int): total number of batches that can be stored in the pool
        feature_max_lengths (Dict[str,int]): Mapping from feature name in KJT
            to the maximum size of the jagged slices for the feature.
        is_weighted (bool): whether KJT values have weights that need to be stored.
        device (Optional[torch.device]): default device
        enable_uvm (bool): if set to true, the pool will be allocated on UVM

    Call Args:
        ids: 1D torch.Tensor of ids to look up

    Returns:
        KeyedJaggedTensor with uniform stride of ids.size(0)

    Example::

        feature_max_lengths = {"feature0": 2, "feature1": 3}

        kjt_pool = KeyedJaggedTensorPool(
            pool_size=10,
            feature_max_lengths=feature_max_lengths,
            values_dtype=torch.float,
        )

        # Update
        kjt_pool.update(
            ids=torch.tensor([1,0,2]), # Assign different indices along batch dim
            values=kjt,
        )

        # Lookup
        lookup_kjt = kjt_pool.lookup(ids=torch.Tensor([2,0]))

        print(lookup_kjt)
        # KeyedJaggedTensor({
        #     "feature0": [[v2], [v0, v1]]
        #     "feature1": [[v5,v6,v7], [v4]]
        # })

    """

    def __init__(
        self,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype = torch.int64,
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        enable_uvm: bool = False,
    ) -> None:
        super().__init__()
        self._pool_size = pool_size
        self._feature_max_lengths: Dict[str, int] = feature_max_lengths
        self._total_lengths = sum(self._feature_max_lengths.values())
        self._values_dtype = values_dtype
        self._is_weighted = is_weighted
        self._device = device if device is not None else torch.device("meta")
        self._enable_uvm = enable_uvm

        self._permute_feature = None
        self.register_buffer(
            "_feature_max_lengths_t",
            torch.tensor(
                list(feature_max_lengths.values()),
                dtype=torch.int32,
                device=self._device,
            ),
            persistent=False,
        )

        self._keys = list(self._feature_max_lengths.keys())
        self._lookup: KeyedJaggedTensorPoolLookup = None
        if self._enable_uvm and values_dtype == torch.int64:
            self._lookup = UVMCachingInt64Lookup(
                pool_size, feature_max_lengths, is_weighted, self._device
            )
        else:
            self._lookup = TensorJaggedIndexSelectLookup(
                pool_size,
                values_dtype,
                feature_max_lengths,
                is_weighted,
                self._device,
            )

        if self._lookup is None:
            raise ValueError(
                f"Cannot create lookup for {self._enable_uvm=} {self._values_dtype}"
            )

        for fqn, tensor in self._lookup.states_to_register():
            self.register_buffer(
                fqn,
                tensor,
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
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        self._lookup._values = state_dict[prefix + "values"]
        self._lookup._key_lengths = state_dict[prefix + "key_lengths"]

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def feature_max_lengths(self) -> Dict[str, int]:
        return self._feature_max_lengths

    @property
    def values_dtype(self) -> torch.dtype:
        return self._values_dtype

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted

    def lookup(self, ids: torch.Tensor) -> KeyedJaggedTensor:
        _fx_assert_device(ids, self._device)
        return _fx_wrap_lookup(
            ids,
            self._keys,
            self._feature_max_lengths,
            self._is_weighted,
            self._values_dtype,
            self._device,
            self._lookup,
            self._weights.dtype if self._is_weighted else None,
        )

    def _update_preproc(self, values: KeyedJaggedTensor) -> KeyedJaggedTensor:
        """
        2 steps:
        1. Permute/filter KJT keys to be the same as in feature_max_lengths
        2. Ensure the max_lengths of input is within the feature_max_lengths
        """
        if self._permute_feature is None:
            self._permute_feature = []
            for feature in self._feature_max_lengths.keys():
                for j, kjt_feature in enumerate(values.keys()):
                    if feature == kjt_feature:
                        self._permute_feature.append(j)

        valid_input = values.permute(self._permute_feature)
        max_elements, _max_indices = (
            valid_input.lengths().reshape(len(self._keys), -1).max(dim=1)
        )

        assert torch.all(
            max_elements <= self._feature_max_lengths_t
        ).item(), "input KJT has a feature that exceeds specified max lengths"

        return valid_input

    def update(self, ids: torch.Tensor, values: KeyedJaggedTensor) -> None:
        _fx_assert_device(ids, self._device)

        kjt = self._update_preproc(values)
        assert kjt.values().dtype == self._values_dtype

        # If duplicate ids are passed in for update, only the last one is kept
        deduped_ids, dedup_permutation = deterministic_dedup(ids)
        arange_idx = torch.arange(
            values.stride() * len(self._keys), device=self._device
        )
        feature_major_to_row_major_permute = (arange_idx.view(len(self._keys), -1).t())[
            dedup_permutation, :
        ].flatten()

        row_major_lengths = kjt.lengths()[feature_major_to_row_major_permute]
        row_major_offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(
            row_major_lengths
        )
        row_major_values = jagged_index_select_with_empty(
            kjt.values().unsqueeze(-1),
            feature_major_to_row_major_permute,
            kjt.offsets()[1:],
            row_major_offsets,
        )

        row_major_values = row_major_values.flatten()

        row_major_lengths = row_major_lengths.flatten()

        row_major_weights = None
        if self._is_weighted:
            row_major_weights = jagged_index_select_with_empty(
                kjt.weights().unsqueeze(-1),
                feature_major_to_row_major_permute,
                kjt.offsets()[1:],
                row_major_offsets,
            )
            row_major_weights = row_major_weights.flatten()

        self._lookup.update(
            deduped_ids,
            JaggedTensor(
                values=row_major_values,
                lengths=row_major_lengths.flatten(),
                weights=row_major_weights,
            ),
        )
