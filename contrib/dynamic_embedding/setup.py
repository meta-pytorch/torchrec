#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys

import torch
from setuptools import find_packages
from skbuild import setup

extra_cmake_args = []

if sys.platform == "linux":
    _nvcc_paths = (
        []
        if os.getenv("CMAKE_CUDA_COMPILER") is None
        else [os.getenv("CMAKE_CUDA_COMPILER")]
    ) + [
        "/usr/bin/nvcc",
        "/usr/local/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/cuda/bin/nvcc",
    ]
    for _nvcc_path in _nvcc_paths:
        try:
            os.stat(_nvcc_path)
            extra_cmake_args.append(f"-DCMAKE_CUDA_COMPILER={_nvcc_path}")
            break
        except FileNotFoundError:
            pass
    else:
        raise RuntimeError(f"Cannot find nvcc in [{','.join(_nvcc_paths)}]")

    if os.getenv("CUDA_TOOLKIT_ROOT_DIR") is None:
        extra_cmake_args.append(
            f'-DCUDA_TOOLKIT_ROOT_DIR={os.path.abspath(os.path.join(os.path.dirname(_nvcc_path), ".."))}'
        )
    else:
        extra_cmake_args.append(
            f"-DCUDA_TOOLKIT_ROOT_DIR={os.getenv('CUDA_TOOLKIT_ROOT_DIR')}"
        )

setup(
    name="torchrec_dynamic_embedding",
    package_dir={"": "src"},
    packages=find_packages("src"),
    cmake_args=[
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DTDE_TORCH_BASE_DIR={os.path.dirname(torch.__file__)}",
        "-DTDE_WITH_TESTING=OFF",
    ]
    + extra_cmake_args,
    cmake_install_dir="src",
    version="0.0.1",
)
