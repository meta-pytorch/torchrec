#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import itertools
from typing import List, Tuple

import torch
import torch._prims_common as utils


def down_size(N: int, size: torch.Size) -> Tuple[int, int]:
    assert size[-1] % N == 0, f"{size} last dim not divisible by {N}"
    # pyrefly: ignore[bad-return]
    return (*size[:-1], size[-1] // N)


def up_size(N: int, size: torch.Size) -> Tuple[int, int]:
    # pyrefly: ignore[bad-return]
    return (*size[:-1], size[-1] * N)


def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.
    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list
    Example:
        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r


def find_arg_of_type(it, t):
    for x in it:
        if isinstance(x, t):
            return x
    return None


class UIntXTensor(torch.Tensor):
    """
    A Tensor subclass of uint8 dtype, that represents Tensor with X-bit elements.
    The last dimension must be divisible by (8 // X).

    __torch_dispatch__ special handling:
    .view(dtype=torch.uint8) - returns the underlying uint8 data.

    .slice,.view - works in UIntX units, dimension values must be divisible by (8 // X).

    .detach,.clone - work as an op on underlying uint8 data.
    """

    # pyrefly: ignore [bad-override, bad-param-name-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, N: int, elem):
        assert elem.dtype is torch.uint8
        return torch.Tensor._make_wrapper_subclass(
            cls, up_size(N, elem.shape), dtype=torch.uint8
        )

    def __init__(self, N: int, elem: torch.Tensor) -> None:
        self.N: int = N
        self.elem: torch.Tensor = elem

    def tolist(self) -> List:
        return self.elem.tolist()

    # pyrefly: ignore[bad-override]
    def __repr__(self) -> str:
        return f"UInt{8 // self.N}Tensor(shape={self.shape}, elem={self.elem})"

    @classmethod
    # pyrefly: ignore [bad-override, bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):  # noqa: C901
        if func is torch.ops.aten.detach.default:
            # Temp workaround to avoid 'Cannot set version_counter for inference tensor'
            with torch.inference_mode(False):
                (self,) = args
                # pyrefly: ignore[missing-argument]
                return cls(func(self.elem))
        elif func is torch.ops.aten.clone.default:
            (self,) = args
            # pyrefly: ignore[missing-argument]
            return cls(func(self.elem))
        elif func is torch.ops.aten.copy_.default:
            (self, src) = args
            self.elem.copy_(src.elem)
            return self
        elif func is torch.ops.aten.view.dtype:
            # .view(dtype=uint8) is the way to get the underlying uint8 data
            self, dtype = args
            if dtype == torch.uint8:
                return self.elem
        elif func is torch.ops.aten._to_copy.default:
            (self,) = args
            dtype = find_arg_of_type(
                # pyrefly: ignore[missing-attribute]
                itertools.chain(args, kwargs.values()),
                torch.dtype,
            )
            device = find_arg_of_type(
                # pyrefly: ignore[missing-attribute]
                itertools.chain(args, kwargs.values()),
                torch.device,
            )
            # Handle only to device
            if device:
                assert dtype is None or dtype == torch.uint8
                # pyrefly: ignore[missing-argument]
                return cls(self.elem.to(device))
        elif func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == self.dim() - 1:
                # hard case
                if step != 1:
                    raise NotImplementedError(f"slice step={step}")
                assert start % self.N == 0, start
                assert end >= self.shape[dim] or end % self.N == 0, end
                # pyrefly: ignore[missing-argument]
                return cls(
                    torch.ops.aten.slice.Tensor(
                        self.elem, dim, start // self.N, end // self.N, 1
                    ),
                )
            else:
                # pyrefly: ignore[missing-argument]
                return cls(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step),
                )
        if func is torch.ops.aten.view.default:
            self, size = args
            size = utils.infer_size(size, self.numel())
            assert not kwargs
            # pyrefly: ignore[bad-argument-type, missing-argument]
            return cls(self.elem.reshape(down_size(self.N, size)))
        elif func is torch.ops.aten.select.int:
            self, dim, index = args
            if dim != self.dim() - 1:
                # pyrefly: ignore[missing-argument]
                return cls(torch.ops.aten.select.int(self.elem, dim, index))
            else:
                raise NotImplementedError(f"select dim={dim}")

        raise NotImplementedError(f"{func} args:{args} kwargs:{kwargs}")


class UInt4Tensor(UIntXTensor):
    N: int = 2

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        return UIntXTensor.__new__(cls, cls.N, elem)

    def __init__(self, elem: torch.Tensor) -> None:
        super().__init__(UInt4Tensor.N, elem)


class UInt2Tensor(UIntXTensor):
    N: int = 4

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        return UIntXTensor.__new__(cls, cls.N, elem)

    def __init__(self, elem: torch.Tensor) -> None:
        super().__init__(UInt2Tensor.N, elem)
