#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.embedding_types import BaseGroupedFeatureProcessor
from torchrec.distributed.utils import append_prefix
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# Will be deprecated soon, please use PositionWeightedProcessor, see the full
# doc under modules/feature_processor.py
class GroupedPositionWeightedModule(BaseGroupedFeatureProcessor):
    def __init__(
        self, max_feature_lengths: Dict[str, int], device: Optional[torch.device] = None
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
        self.register_buffer(
            "_dummy_weights",
            torch.tensor(
                max(self.max_feature_lengths.values()),
                device=device,
            ).fill_(1.0),
        )

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        if features.weights_or_none() is None:
            cat_seq = torch.ops.fbgemm.offsets_range(
                features.offsets().long(), torch.numel(features.values())
            )
        else:
            # for row-wise sharding
            cat_seq = features.weights().long()
        seqs = torch.split(cat_seq, features.length_per_key())
        weights_list = []
        for key, seq in zip(features.keys(), seqs):
            if key in self.max_feature_lengths:
                weights_list.append(
                    torch.gather(self.position_weights[key], dim=0, index=seq)
                )
            else:
                weights_list.append(
                    # pyrefly: ignore[bad-index]
                    self._dummy_weights[: self.max_feature_lengths[key]]
                )
        weights = torch.cat(weights_list)

        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            weights=weights,
            lengths=features.lengths(),
            offsets=features.offsets(),
            stride=features.stride(),
            length_per_key=features.length_per_key(),
            stride_per_key_per_rank=(
                features.stride_per_key_per_rank()
                if features.variable_stride_per_key()
                else None
            ),
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.position_weights.items():
            yield append_prefix(prefix, f"position_weights.{name}"), param

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    # pyrefly: ignore[bad-override]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyrefly: ignore[missing-attribute]
            destination._metadata = OrderedDict()
        for name, param in self.position_weights.items():
            destination[prefix + f"position_weights.{name}"] = (
                param if keep_vars else param.detach()
            )
        return destination
