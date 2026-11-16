#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Optional

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _tensor_eq_or_none(
    t1: Optional[torch.Tensor],
    t2: Optional[torch.Tensor],
    out_of_order: bool = False,
    length: Optional[torch.Tensor] = None,
) -> bool:
    if t1 is None and t2 is None:
        return True
    elif t1 is None and t2 is not None:
        return False
    elif t1 is not None and t2 is None:
        return False

    assert t1 is not None
    assert t2 is not None

    if t1.dtype != t2.dtype:
        return False

    if not out_of_order:
        return torch.equal(t1, t2)

    assert length is not None
    is_int = not torch.is_floating_point(t1)
    vals_1 = t1.tolist()
    vals_2 = t2.tolist()
    current_offset = 0
    for i in length.tolist():
        if i == 0:
            continue
        sorted_vals_1 = sorted(vals_1[current_offset : current_offset + i])
        sorted_vals_2 = sorted(vals_2[current_offset : current_offset + i])
        if is_int:
            if sorted_vals_1 != sorted_vals_2:
                return False
        else:
            for left, right in zip(
                sorted_vals_1,
                sorted_vals_2,
            ):
                if not math.isclose(left, right):
                    return False
        current_offset += i
    return True


def keyed_jagged_tensor_equals(
    kjt1: Optional[KeyedJaggedTensor],
    kjt2: Optional[KeyedJaggedTensor],
    is_pooled_features: bool = False,
) -> bool:
    if kjt1 is None and kjt2 is None:
        return True
    elif kjt1 is None and kjt2 is not None:
        return False
    elif kjt1 is not None and kjt2 is None:
        return False

    assert kjt1 is not None
    assert kjt2 is not None
    if not (
        kjt1.keys() == kjt2.keys()
        and _tensor_eq_or_none(kjt1.lengths(), kjt2.lengths())
    ):
        return False

    return _tensor_eq_or_none(
        kjt1.values(), kjt2.values(), is_pooled_features, kjt1.lengths()
    ) and _tensor_eq_or_none(
        kjt1._weights, kjt2._weights, is_pooled_features, kjt1.lengths()
    )
