#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


def validate_keyed_jagged_tensor(
    kjt: KeyedJaggedTensor, configs: Optional[List[EmbeddingBagConfig]] = None
) -> bool:
    """
    Validates the inputs that construct a KeyedJaggedTensor.

    Any invalid input will result in a ValueError being thrown.

    Returns:
        bool: True if all validations pass (including feature range),
              False if feature range validation fails (soft warning).
    """
    _validate_lengths_and_offsets(kjt)
    _validate_keys(kjt)
    _validate_weights(kjt)
    if configs is not None:
        return _validate_feature_range(kjt, configs)
    else:
        return True


def _validate_lengths_and_offsets(kjt: KeyedJaggedTensor) -> bool:
    """
    Validates the lengths and offsets of a KJT.

    - At least one of lengths or offsets is provided
    - If both are provided, they are consistent with each other
    - The dimensions of these tensors align with the values tensor

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    lengths = kjt.lengths_or_none()
    offsets = kjt.offsets_or_none()
    if lengths is None and offsets is None:
        raise ValueError(
            "lengths and offsets cannot be both empty in KeyedJaggedTensor"
        )
    elif lengths is not None and offsets is not None:
        _validate_lengths_and_offsets_consistency(lengths, offsets, kjt.values())
    elif lengths is not None:
        _validate_lengths(lengths, kjt.values())
    elif offsets is not None:
        _validate_offsets(offsets, kjt.values())
    return True


def _validate_lengths_and_offsets_consistency(
    lengths: torch.Tensor, offsets: torch.Tensor, values: torch.Tensor
) -> None:
    _validate_lengths(lengths, values)
    _validate_offsets(offsets, values)

    if lengths.numel() != offsets.numel() - 1:
        raise ValueError(
            f"Expected lengths size to be 1 more than offsets size, but got lengths size: {lengths.numel()} and offsets size: {offsets.numel()}"
        )

    if not lengths.equal(torch.diff(offsets)):
        raise ValueError("offsets is not equal to the cumulative sum of lengths")
    return True


def _validate_lengths(lengths: torch.Tensor, values: torch.Tensor) -> bool:
    """
    Validates lengths tensor.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    if lengths.sum().item() != values.numel():
        raise ValueError(
            f"Sum of lengths must equal the number of values, but got {lengths.sum().item()} and {values.numel()}"
        )
    return True


def _validate_offsets(offsets: torch.Tensor, values: torch.Tensor) -> bool:
    """
    Validates offsets tensor.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    if offsets.numel() == 0:
        raise ValueError("offsets cannot be empty")

    if offsets[0] != 0:
        raise ValueError(f"Expected first offset to be 0, but got {offsets[0]} instead")

    if offsets[-1] != values.numel():
        raise ValueError(
            f"The last element of offsets must equal to the number of values, but got {offsets[-1]} and {values.numel()}"
        )
    return True


def _validate_keys(kjt: KeyedJaggedTensor) -> bool:
    """
    Validates KJT keys, assuming the lengths/offsets input are valid.

    - keys must be unique
    - For non-VBE cases, the size of lengths is divisible by the number of keys

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    keys = kjt.keys()

    if len(set(keys)) != len(keys):
        raise ValueError("keys must be unique")

    lengths = kjt.lengths_or_none()
    offsets = kjt.offsets_or_none()
    if lengths is not None:
        lengths_size = lengths.numel()
    else:
        assert offsets is not None
        lengths_size = offsets.numel() - 1

    if len(keys) == 0 and lengths_size > 0:
        raise ValueError("keys is empty but lengths or offsets is not")
    elif len(keys) > 0:
        # TODO: Validate KJT for VBE cases
        if not kjt.variable_stride_per_key():
            if lengths_size % len(keys) != 0:
                raise ValueError(
                    f"lengths size must be divisible by keys size, but got {lengths_size} and {len(keys)}"
                )
    return True


def _validate_weights(kjt: KeyedJaggedTensor) -> bool:
    """
    Validates if the KJT weights has the same size as values.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    weights = kjt.weights_or_none()
    if weights is not None and weights.numel() != kjt.values().numel():
        raise ValueError(
            f"weights size must equal to values size, but got {weights.numel()} and {kjt.values().numel()}"
        )
    return True


def _validate_feature_range(
    kjt: KeyedJaggedTensor, configs: List[EmbeddingBagConfig]
) -> bool:
    """
    Validates if the KJT feature range is valid.

    Returns:
        bool: True if all features are within valid range, False otherwise.
    """
    feature_to_range_map: Dict[str, int] = {}
    for config in configs:
        for feature in config.feature_names:
            feature_to_range_map[feature] = config.num_embeddings

    if len(kjt.keys() & feature_to_range_map.keys()) == 0:
        logger.info(
            f"None of KJT._keys {kjt.keys()} in the config {feature_to_range_map.keys()}"
        )
        return True

    valid = True
    jtd = kjt.to_dict()
    for feature, jt in jtd.items():
        if feature not in feature_to_range_map:
            logger.info(f"Feature {feature} is not in the config")
            continue
        if jt.values().numel() == 0:
            continue
        min_value, max_value = jt.values().min(), jt.values().max()
        if min_value < 0 or max_value >= feature_to_range_map[feature]:
            logger.warning(
                f"Feature {feature} has range {min_value, max_value} "
                f"which is out of range {0, feature_to_range_map[feature]}"
            )
            valid = False
    return valid
