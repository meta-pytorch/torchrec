#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utility functions for Torch TPU Embedding modules."""

import os

import torch

# Installs the `torch.tpu` module that the functions below call.
import torch_tpu  # pylint: disable=unused-import  # ruff: ignore[unused-import]


def get_device() -> torch.device:
    """Returns the TPU PyTorch device."""
    return torch.device("tpu")


def _has_tpu_devices() -> bool:
    """Returns whether TorchTPU has TPU devices in this process.

    TorchTPU only makes the TPU the current accelerator where it has TPU devices
    of a generation that it supports; elsewhere, e.g. on GPU hosts,
    `torch.accelerator` refers to other devices, whose availability isn't checked
    so that e.g. CUDA isn't initialized.
    """
    accelerator = torch.accelerator.current_accelerator()
    return (
        accelerator is not None
        and accelerator.type == "tpu"
        and torch.accelerator.is_available()
    )


def get_num_sc_per_device(
    num_sc_per_device: int | None = None,
) -> int:
    """Returns the number of SparseCores per device.

    Args:
      num_sc_per_device: Optional override for the number of SparseCores.

    Returns:
      `num_sc_per_device` if given, else the value of the `NUM_SC_PER_DEVICE`
      environment variable if set, else the number of SparseCores per device of
      TorchTPU's TPU devices (0 if they have none or TorchTPU has no TPU devices,
      e.g. on CPU or GPU hosts).
    """
    if num_sc_per_device is not None:
        return num_sc_per_device

    env_sc = os.getenv("NUM_SC_PER_DEVICE")
    if env_sc is not None and env_sc.isdigit():
        return int(env_sc)

    # `torch.tpu.get_device_properties()` raises without TPU devices.
    if not _has_tpu_devices():
        return 0
    return torch.tpu.get_device_properties().sparse_cores_per_device


def get_sc_simd_width(
    sc_simd_width: int | None = None,
) -> int:
    """Returns the SIMD width of each SparseCore.

    Args:
      sc_simd_width: Optional override for SparseCore SIMD width.

    Returns:
      `sc_simd_width` if given, else the value of the `SC_SIMD_WIDTH` environment
      variable if set, else the SparseCore SIMD width of TorchTPU's TPU devices.

    Raises:
      ValueError: If the SIMD width is neither given nor set in `SC_SIMD_WIDTH`,
        and TorchTPU has no TPU devices or they have no SparseCores.
    """
    if sc_simd_width is not None:
        return sc_simd_width

    env_simd = os.getenv("SC_SIMD_WIDTH")
    if env_simd is not None and env_simd.isdigit():
        return int(env_simd)

    # `torch.tpu.get_device_properties()` raises without TPU devices.
    if _has_tpu_devices():
        properties = torch.tpu.get_device_properties()
        if properties.sparse_core_simd_width is not None:
            return properties.sparse_core_simd_width
        device = properties.name
    else:
        device = "no TPU device"
    raise ValueError(
        f"expected a TPU device with SparseCores, got {device}; pass the"
        " SparseCore SIMD width or set SC_SIMD_WIDTH"
    )


def get_hbm_word_size_in_4b(
    hbm_word_size_in_4b: int | None = None,
) -> int:
    """Returns the HBM word size in 4-byte words.

    Args:
      hbm_word_size_in_4b: Optional override for HBM word size in 4-byte words.

    Returns:
      HBM word size in 4-byte words.
    """
    if hbm_word_size_in_4b is not None:
        return hbm_word_size_in_4b

    env_word_size = os.getenv("HBM_WORD_SIZE_IN_4B")
    if env_word_size is not None and env_word_size.isdigit():
        return int(env_word_size)

    return 8


def get_compile_count() -> int:
    """Returns total PyTorch Dynamo frame compile count."""
    return sum(torch._dynamo.convert_frame.FRAME_COMPILE_COUNTER.values())


def to_device_in_sparsecore_layout(
    t: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Copies a host tensor to `device`, in the SparseCore layout if it's a TPU.

    SparseCore kernels read and write tables row-major in 8-element tiles, the
    {1,0:T(8)} layout. Allocating tables and optimizer slots in it avoids a
    full-table relayout around SparseCore kernels, since XLA's default layout for
    narrow tables is {0,1:T(8,128)}.

    Args:
      t: The tensor to copy, on the host.
      device: The device to copy `t` to.

    Returns:
      A copy of `t` on `device`, in the SparseCore layout if `device` is a TPU and
      in the default layout otherwise.
    """
    if device.type != "tpu":
        return t.to(device)
    with torch.tpu.LayoutContext(
        torch.tpu.TpuLayout(minor_to_major=[1, 0], tiles=[[8]])
    ):
        return t.to(device)
