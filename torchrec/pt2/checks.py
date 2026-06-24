#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

import torch

try:
    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true
except ImportError:
    # Forward-compatibility: older torch builds bundled in prod-pinned
    # lowering/processing packages predate guard_or_false/guard_or_true.
    #
    # This fallback is never actually invoked in practice: older torch only
    # exists in the deserialization/lowering backend, where is_pt2_compiling() is
    # False, so the consuming pt2_guard_or_* functions return x before ever
    # calling these. It exists only so the module can be imported against older
    # torch (the forward-compat condition that was failing).
    #
    # guard_size_oblivious is the exact primitive these two helpers replaced in
    # D100530526 and reproduces their behavior for the only call sites
    # (jagged_tensor: numel() == 0 -> guard_or_false; numel() != 0 / > 0 ->
    # guard_or_true), so it is faithful for the size predicates used here. It is
    # intentionally not a general-purpose equivalent (guard_or_false biases
    # False, guard_or_true biases True); that mismatch is moot given the above.
    #
    # Kept at module scope (not inlined) because pt2_guard_or_* are TorchScript
    # compiled, and TorchScript rejects in-function import statements even in the
    # dead is_scripting() branch.
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    def guard_or_false(x: bool) -> bool:
        return guard_size_oblivious(x)

    def guard_or_true(x: bool) -> bool:
        return guard_size_oblivious(x)


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
    torch._C._log_api_usage_once("torchrec.pt2.checks.import_failure.torch_compiler")

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

    torch._check(start_offset >= 0)
    torch._check(end_offset >= 0)
    torch._check(end_offset - start_offset >= 0)
    torch._check(start_offset <= tensor.size(dim))
    torch._check(end_offset <= tensor.size(dim))
    torch._check(end_offset >= start_offset)


def pt2_checks_all_is_size(x: List[int]) -> List[int]:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    for i in x:
        torch._check(i >= 0)
    return x


def pt2_check_size_nonzero(x: torch.Tensor) -> torch.Tensor:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    for i in range(x.dim()):
        torch._check(x.size(i) > 0)
    return x


def pt2_guard_or_false(x: bool) -> bool:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    return guard_or_false(x)


def pt2_guard_or_true(x: bool) -> bool:
    if torch.jit.is_scripting() or not is_pt2_compiling():
        return x

    return guard_or_true(x)
