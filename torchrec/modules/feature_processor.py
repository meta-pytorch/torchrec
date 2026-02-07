#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.fx.tracer import is_fx_tracing
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


# Will be deprecated soon, please use PositionWeightedProcessor, see full doc below
class BaseFeatureProcessor(nn.Module):
    """
    Abstract base class for feature processor.
    """

    @abc.abstractmethod
    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        pass


@torch.fx.wrap
def position_weighted_module_update_features(
    features: Dict[str, JaggedTensor],
    weighted_features: Dict[str, JaggedTensor],
) -> Dict[str, JaggedTensor]:
    features.update(weighted_features)
    return features


@torch.jit.script_if_tracing
@torch.fx.wrap
def offsets_to_range_traceble(
    offsets: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    return torch.ops.fbgemm.offsets_range(offsets.long(), torch.numel(values))


# Will be deprecated soon, please use PositionWeightedProcessor, see full doc below
class PositionWeightedModule(BaseFeatureProcessor):
    """
    Adds position weights to id list features.

    Args:
        max_feature_lengths (Dict[str, int]): feature name to `max_length` mapping.
            `max_length`, a.k.a truncation size, specifies the maximum number of ids
            each sample has. For each feature, its position weight parameter size is
            `max_length`.
    """

    def __init__(
        self,
        max_feature_lengths: Dict[str, int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.max_feature_lengths = max_feature_lengths
        self.position_weights: nn.ParameterDict = nn.ParameterDict()
        for key, length in max_feature_lengths.items():
            self.position_weights[key] = nn.Parameter(
                torch.empty([length], device=device)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for key, _length in self.max_feature_lengths.items():
                self.position_weights[key].fill_(1.0)

    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (Dict[str, JaggedTensor]): dictionary of keys to `JaggedTensor`,
                representing the features.

        Returns:
            Dict[str, JaggedTensor]: same as input features with `weights` field being populated.
        """

        weighted_features: Dict[str, JaggedTensor] = {}
        for key, position_weight in self.position_weights.items():
            seq = offsets_to_range_traceble(
                features[key].offsets(), features[key].values()
            )
            weighted_features[key] = JaggedTensor(
                values=features[key].values(),
                lengths=features[key].lengths(),
                offsets=features[key].offsets(),
                weights=torch.gather(position_weight, dim=0, index=seq),
            )
        return position_weighted_module_update_features(features, weighted_features)


class BaseGroupedFeatureProcessor(nn.Module):
    """
    Abstract base class for grouped feature processor
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        pass


class PositionWeightedProcessor(BaseGroupedFeatureProcessor):
    """
    PositionWeightedProcessor represents a processor to apply position weight to a KeyedJaggedTensor.

    It can handle both unsharded and sharded input and output corresponding output

    Args:
        max_feature_lengths (Dict[str, int]): Dict of feature_lengths, the key is the feature_name and value is length.
        device (Optional[torch.device]): default compute device.

    Example::

        keys=["Feature0", "Feature1", "Feature2"]
        values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7])
        lengths=torch.tensor([2, 0, 1, 1, 1, 3, 2, 3, 0])
        features = KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths)
        pw = FeatureProcessorCollection(
            feature_processor_modules={key: PositionWeightedFeatureProcessor(max_feature_length=100) for key in keys}
        )
        result = pw(features)
        # result is
        # KeyedJaggedTensor({
        #     "Feature0": {
        #         "values": [[0, 1], [], [2]],
        #         "weights": [[1.0, 1.0], [], [1.0]]
        #     },
        #     "Feature1": {
        #         "values": [[3], [4], [5, 6, 7]],
        #         "weights": [[1.0], [1.0], [1.0, 1.0, 1.0]]
        #     },
        #     "Feature2": {
        #         "values": [[3, 4], [5, 6, 7], []],
        #         "weights": [[1.0, 1.0], [1.0, 1.0, 1.0], []]
        #     }
        # })
    """

    def __init__(
        self,
        max_feature_lengths: Dict[str, int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.max_feature_lengths = max_feature_lengths
        for length in self.max_feature_lengths.values():
            if length <= 0:
                raise
        self.position_weights: nn.ParameterDict = nn.ParameterDict()
        for key, length in max_feature_lengths.items():
            self.position_weights[key] = nn.Parameter(
                torch.empty([length], device=device).fill_(1.0)
            )

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        """
        In unsharded or non-pipelined model, the input features both contain fp_feature
        and non_fp_features, and the output will filter out non_fp features
        In sharded pipelining model, the input features can only contain either none
        or all feature_processed features, since the input feature comes from the
        input_dist() of ebc which will filter out the keys not in the ebc. And the
        input size is same as output size

        Args:
            features (KeyedJaggedTensor): input features

        Returns:
            KeyedJaggedTensor
        """
        if is_fx_tracing():
            features_dict = features.to_dict()
            weighted_features_names: List[str] = []
            weighted_features_values: List[torch.Tensor] = []
            weighted_features_lengths: List[torch.Tensor] = []
            weighted_features_weights: List[torch.Tensor] = []
            for key, position_weight in self.position_weights.items():
                seq = torch.ops.fbgemm.offsets_range(
                    features_dict[key].offsets().long(),
                    torch.numel(features_dict[key].values()),
                )
                weighted_features_names.append(key)
                weighted_features_values.append(features_dict[key].values())
                weighted_features_lengths.append(features_dict[key].lengths())
                weighted_features_weights.append(
                    torch.gather(position_weight, dim=0, index=seq)
                )
            return KeyedJaggedTensor.from_lengths_sync(
                keys=weighted_features_names,
                values=torch.cat(weighted_features_values),
                lengths=torch.cat(weighted_features_lengths),
                weights=torch.cat(weighted_features_weights),
            )
        else:
            feature_names = features.keys()
            lengths = features.lengths()
            offsets = features.offsets()
            values = features.values()
            length_per_key = features.length_per_key()
            weights = features.weights_or_none()
            batch_size = features.stride()
            vbe = features.variable_stride_per_key()
            stride_per_key_per_rank = features.stride_per_key_per_rank()

            has_fp_id_list_feature = False
            has_normal_id_list_feature = False

            if weights is None:
                cat_seq = torch.ops.fbgemm.offsets_range(
                    offsets.long(), torch.numel(values)
                )
            else:
                # for row-wise sharding
                cat_seq = weights.long()
            seqs = torch.split(cat_seq, features.length_per_key())

            for feature_name in feature_names:
                if feature_name in self.max_feature_lengths:
                    has_fp_id_list_feature = True
                else:
                    has_normal_id_list_feature = True

            # in sharded pipelining model, the input features can only contain either none
            # or all feature_processed features, since the input feature comes from the
            # input_dist() of ebc which will filter out the keys not in the ebc
            # for the input features both contain fp_feature and normal_features, it could be
            # unsharded or non-pipelined sharded models
            if has_fp_id_list_feature:
                # for sharded pipeling
                if not has_normal_id_list_feature:
                    processed_features_weights: List[torch.Tensor] = []
                    for feature_index, feature_name in enumerate(feature_names):
                        processed_weight = torch.gather(
                            self.position_weights[feature_name],
                            dim=0,
                            index=seqs[feature_index],
                        )
                        processed_features_weights.append(processed_weight)
                    fp_features = KeyedJaggedTensor(
                        keys=feature_names,
                        values=values,
                        weights=torch.cat(processed_features_weights),
                        lengths=lengths,
                        offsets=offsets,
                        stride=batch_size,
                        length_per_key=length_per_key,
                        offset_per_key=features.offset_per_key(),
                        index_per_key=features._key_indices(),
                        stride_per_key_per_rank=(
                            stride_per_key_per_rank if vbe else None
                        ),
                    )
                # for unsharded or sharded non-pipeling
                else:
                    feature_values = values.split(length_per_key)
                    feature_lengths = lengths.split(batch_size)
                    processed_features_names: List[str] = []
                    processed_features_lengths: List[torch.Tensor] = []
                    processed_features_values: List[torch.Tensor] = []
                    processed_features_weights: List[torch.Tensor] = []
                    processed_features_batch_sizes = []
                    for feature_index, feature_name in enumerate(feature_names):
                        if feature_name in self.max_feature_lengths:
                            feature_value = feature_values[feature_index]
                            feature_length = feature_lengths[feature_index]
                            processed_weight = torch.gather(
                                self.position_weights[feature_name],
                                dim=0,
                                index=seqs[feature_index],
                            )
                            processed_features_names.append(feature_name)
                            processed_features_lengths.append(feature_length)
                            processed_features_values.append(feature_value)
                            processed_features_weights.append(processed_weight)
                            if vbe:
                                processed_features_batch_sizes.append(
                                    stride_per_key_per_rank[feature_index]
                                )
                    fp_features = KeyedJaggedTensor.from_lengths_sync(
                        keys=processed_features_names,
                        values=torch.cat(processed_features_values),
                        lengths=torch.cat(processed_features_lengths),
                        weights=torch.cat(processed_features_weights),
                        stride_per_key_per_rank=(
                            processed_features_batch_sizes if vbe else None
                        ),
                    )
                return fp_features
            # normal id_list feature
            else:
                return features

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, param in self.position_weights.items():
            destination[prefix + f"position_weights.{name}"] = (
                param if keep_vars else param.detach()
            )
        return destination
