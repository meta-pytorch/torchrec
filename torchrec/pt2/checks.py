#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

import torch

from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

USE_TORCHDYNAMO_COMPILING_PATH: bool = False


def set_use_torchdynamo_compiling_path(val: bool) -> None:
    global USE_TORCHDYNAMO_COMPILING_PATH
    USE_TORCHDYNAMO_COMPILING_PATH = val


def get_use_torchdynamo_compiling_path() -> bool:
    global USE_TORCHDYNAMO_COMPILING_PATH
    return USE_TORCHDYNAMO_COMPILING_PATH


try:
    if torch.jit.is_scripting():
        raise Exception()

    from torch.compiler import (
        is_compiling as is_compiler_compiling,
        is_dynamo_compiling as _is_torchdynamo_compiling,
    )

    def is_torchdynamo_compiling() -> bool:
        if torch.jit.is_scripting():
            return False

        # Can not use global variable here, as it is not supported in TorchScript
        # (It parses full method src even there is a guard torch.jit.is_scripting())
        return get_use_torchdynamo_compiling_path() or _is_torchdynamo_compiling()

    def is_non_strict_exporting() -> bool:
        return not is_torchdynamo_compiling() and is_compiler_compiling()

except Exception:
    # BC for torch versions without compiler and torch deploy path
    def is_torchdynamo_compiling() -> bool:
        return False

    def is_non_strict_exporting() -> bool:
        return False


def is_pt2_compiling() -> bool:
    return is_torchdynamo_compiling() or is_compiler_compiling()


def pt2_checks_tensor_slice(
    tensor: torch.Tensor, start_offset: int, end_offset: int, dim: int = 0
) -> None:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return

    torch._check_is_size(start_offset)
    torch._check_is_size(end_offset)
    torch._check_is_size(end_offset - start_offset)
    torch._check(start_offset <= tensor.size(dim))
    torch._check(end_offset <= tensor.size(dim))
    torch._check(end_offset >= start_offset)


def pt2_checks_all_is_size(x: List[int]) -> List[int]:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    for i in x:
        torch._check_is_size(i)
    return x


def pt2_check_size_nonzero(x: torch.Tensor) -> torch.Tensor:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    for i in range(x.dim()):
        torch._check(x.size(i) > 0)
    return x


def pt2_guard_size_oblivious(x: bool) -> bool:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    return guard_size_oblivious(x)
