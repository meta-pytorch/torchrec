#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import torch
from tensordict import TensorDict
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def maybe_td_to_kjt(
    features: KeyedJaggedTensor, keys: Optional[List[str]] = None
) -> KeyedJaggedTensor:
    if torch.jit.is_scripting():
        assert isinstance(features, KeyedJaggedTensor)
        return features
    if isinstance(features, TensorDict):
        if keys is None:
            # pyrefly: ignore[no-matching-overload]
            keys = list(features.keys())
        values = torch.cat([features[key]._values for key in keys], dim=0)
        lengths = torch.cat(
            [
                (
                    (features[key]._lengths)
                    if features[key]._lengths is not None
                    else torch.diff(features[key]._offsets)
                )
                for key in keys
            ],
            dim=0,
        )
        return KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
        )
    else:
        return features
