#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


export PYTORCH_CUDA_PKG=""
export CONDA_ENV="build_binary"

if [[ ${MATRIX_PYTHON_VERSION} = '3.14t' ]]; then
    # use conda-forge to install python3.14t
    conda create -y -n "${CONDA_ENV}" python-freethreading=3.14
    conda run -n "${CONDA_ENV}" python -c "import sys; print(f'python GIL enabled: {sys._is_gil_enabled()}')"
elif [[ ${MATRIX_PYTHON_VERSION} = '3.13t' ]]; then
    # use conda-forge to install python3.13t
    conda create -y -n "${CONDA_ENV}" python="3.13" python-freethreading -c conda-forge
    conda run -n "${CONDA_ENV}" python -c "import sys; print(f'python GIL enabled: {sys._is_gil_enabled()}')"
else
    conda create -y -n "${CONDA_ENV}" python="${MATRIX_PYTHON_VERSION}"
fi

conda run -n "${CONDA_ENV}" python --version

# Install pytorch, torchrec and fbgemm as per
# installation instructions on following page
# https://github.com/meta-pytorch/torchrec#installations


# figure out CUDA VERSION
if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
    if [[ ${MATRIX_GPU_ARCH_VERSION} = '11.8' ]]; then
        export CUDA_VERSION="cu118"
    elif [[ ${MATRIX_GPU_ARCH_VERSION} = '12.1' ]]; then
        export CUDA_VERSION="cu121"
    elif [[ ${MATRIX_GPU_ARCH_VERSION} = '12.6' ]]; then
        export CUDA_VERSION="cu126"
    elif [[ ${MATRIX_GPU_ARCH_VERSION} = '12.8' ]]; then
        export CUDA_VERSION="cu128"
    elif [[ ${MATRIX_GPU_ARCH_VERSION} = '12.9' ]]; then
        export CUDA_VERSION="cu129"
    elif [[ ${MATRIX_GPU_ARCH_VERSION} = '13.0' ]]; then
        export CUDA_VERSION="cu130"
    else
        export CUDA_VERSION="cu126"
    fi
else
    export CUDA_VERSION="cpu"
fi

# figure out URL
if [[ ${MATRIX_CHANNEL} = 'nightly' ]]; then
    export PYTORCH_URL="https://download.pytorch.org/whl/nightly/${CUDA_VERSION}"
elif [[ ${MATRIX_CHANNEL} = 'test' ]]; then
    export PYTORCH_URL="https://download.pytorch.org/whl/test/${CUDA_VERSION}"
elif [[ ${MATRIX_CHANNEL} = 'release' ]]; then
    export PYTORCH_URL="https://download.pytorch.org/whl/${CUDA_VERSION}"
fi


echo "CU_VERSION: ${CUDA_VERSION}"
echo "MATRIX_CHANNEL: ${MATRIX_CHANNEL}"
echo "CONDA_ENV: ${CONDA_ENV}"

# shellcheck disable=SC2155
export CONDA_PREFIX=$(conda run -n "${CONDA_ENV}" printenv CONDA_PREFIX)


# Set LD_LIBRARY_PATH to fix the runtime error with fbgemm_gpu not
# being able to locate libnvrtc.so
# NOTE: The order of the entries in LD_LIBRARY_PATH matters
echo "[NOVA] Setting LD_LIBRARY_PATH ..."
conda env config vars set -n ${CONDA_ENV}  \
    LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/lib:/usr/lib64:${LD_LIBRARY_PATH}"


# install pytorch
# switch back to conda once torch nightly is fixed
# if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
#     export PYTORCH_CUDA_PKG="pytorch-cuda=${MATRIX_GPU_ARCH_VERSION}"
# fi

conda run -n "${CONDA_ENV}" pip install torch --index-url "$PYTORCH_URL"

# install fbgemm
conda run -n "${CONDA_ENV}" pip install fbgemm-gpu --index-url "$PYTORCH_URL"

# install other requirements
conda run -n "${CONDA_ENV}" pip install -r requirements.txt

# install torchrec
conda run -n "${CONDA_ENV}" pip install torchrec --index-url "$PYTORCH_URL"

# Run small import test
conda run -n "${CONDA_ENV}" python -c "import torch; import fbgemm_gpu; import torchrec"

# check directory
ls -R

# check if cuda available
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.cuda.is_available())"

# check cuda version
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.version.cuda)"

# Finally run smoke test
conda run -n "${CONDA_ENV}" pip install iopath
if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
    conda run -n "${CONDA_ENV}" torchrun --nnodes=1 --nproc_per_node=gpu test_installation.py
else
    conda run -n "${CONDA_ENV}" torchrun --nnodes=1 --nproc_per_node=1 test_installation.py --cpu_only
fi


# redo for pypi release

if [[ ${MATRIX_CHANNEL} != 'release' ]]; then
    exit 0
else
    # Check version matches only for release binaries
    torchrec_version=$(conda run -n "${CONDA_ENV}" pip show torchrec | grep Version | cut -d' ' -f2)
    fbgemm_version=$(conda run -n "${CONDA_ENV}" pip show fbgemm_gpu | grep Version | cut -d' ' -f2)

    if [ "$torchrec_version" != "$fbgemm_version" ]; then
        echo "Error: TorchRec package version does not match FBGEMM package version"
        exit 1
    fi
fi

if [[ ${MATRIX_PYTHON_VERSION} = '3.14' ]]; then
    # conda currently doesn't support 3.14 unless using the forge channel
    conda create -y -n "${CONDA_ENV}" python="3.14" -c conda-forge
elif [[ ${MATRIX_PYTHON_VERSION} = '3.13t' ]]; then
    # use conda-forge to install python3.13t
    conda create -y -n "${CONDA_ENV}" python="3.13" python-freethreading -c conda-forge
    conda run -n "${CONDA_ENV}" python -c "import sys; print(f'python GIL enabled: {sys._is_gil_enabled()}')"
else
    conda create -y -n "${CONDA_ENV}" python="${MATRIX_PYTHON_VERSION}"
fi


conda run -n "${CONDA_ENV}" python --version

# we only have one cuda version for pypi build
if [[ ${MATRIX_GPU_ARCH_VERSION} != '12.6' ]]; then
    exit 0
fi

echo "checking pypi release"
conda run -n "${CONDA_ENV}" pip install torch
conda run -n "${CONDA_ENV}" pip install fbgemm-gpu
conda run -n "${CONDA_ENV}" pip install torchrec

# Check version matching again for PyPI
torchrec_version=$(conda run -n "${CONDA_ENV}" pip show torchrec | grep Version | cut -d' ' -f2)
fbgemm_version=$(conda run -n "${CONDA_ENV}" pip show fbgemm_gpu | grep Version | cut -d' ' -f2)

if [ "$torchrec_version" != "$fbgemm_version" ]; then
    echo "Error: TorchRec package version does not match FBGEMM package version"
    exit 1
fi

# check directory
ls -R

# check if cuda available
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.cuda.is_available())"

# check cuda version
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.version.cuda)"

# Finally run smoke test
conda run -n "${CONDA_ENV}" pip install iopath
conda run -n "${CONDA_ENV}" torchrun --nnodes=1 --nproc_per_node=gpu test_installation.py
