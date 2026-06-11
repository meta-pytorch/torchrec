#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Auto version mapping: torchrec 1.X -> torch 2.(X+5)
# e.g. torchrec 1.4 -> torch 2.9, 1.5 -> 2.10, 1.6 -> 2.11
get_expected_torch_version() {
    local torchrec_ver="$1"
    local minor
    minor=$(echo "$torchrec_ver" | cut -d'.' -f2)
    if [[ -z "$minor" || ! "$minor" =~ ^[0-9]+$ ]]; then
        echo "Cannot parse torchrec version: $torchrec_ver" >&2
        return 1
    fi
    echo "2.$((minor + 5))"
}

# Validate installed package versions against expected versions
validate_versions() {
    local expected_torchrec_ver="$1"
    local expected_torch_ver="$2"

    local torchrec_ver
    torchrec_ver=$(conda run -n "${CONDA_ENV}" pip show torchrec | grep Version | cut -d' ' -f2)
    local fbgemm_ver
    fbgemm_ver=$(conda run -n "${CONDA_ENV}" pip show fbgemm_gpu | grep Version | cut -d' ' -f2)
    local torch_ver
    torch_ver=$(conda run -n "${CONDA_ENV}" pip show torch | grep Version | cut -d' ' -f2)

    echo "Installed versions: torchrec=$torchrec_ver, fbgemm_gpu=$fbgemm_ver, torch=$torch_ver"
    echo "Expected versions: torchrec=${expected_torchrec_ver}, fbgemm_gpu=${expected_torchrec_ver}, torch=${expected_torch_ver}.*"

    local failed=0
    if [[ "$torchrec_ver" != "$expected_torchrec_ver"* ]]; then
        echo "Error: torchrec version mismatch: got $torchrec_ver, expected ${expected_torchrec_ver}*"
        failed=1
    fi
    if [[ "$fbgemm_ver" != "$expected_torchrec_ver"* ]]; then
        echo "Error: fbgemm_gpu version mismatch: got $fbgemm_ver, expected ${expected_torchrec_ver}*"
        failed=1
    fi
    if [[ "$torch_ver" != "$expected_torch_ver"* ]]; then
        echo "Error: torch version mismatch: got $torch_ver, expected ${expected_torch_ver}*"
        failed=1
    fi
    if [[ "$failed" -eq 1 ]]; then
        exit 1
    fi
    echo "All package versions validated successfully."
}

# Read expected version from version.txt
EXPECTED_TORCHREC_VERSION=$(tr -d '[:space:]' < version.txt)
EXPECTED_TORCH_VERSION=$(get_expected_torch_version "$EXPECTED_TORCHREC_VERSION")
if [[ $? -ne 0 ]]; then
    echo "Failed to determine expected torch version for torchrec=$EXPECTED_TORCHREC_VERSION"
    exit 1
fi
echo "Expected torchrec/fbgemm version: $EXPECTED_TORCHREC_VERSION"
echo "Expected torch version: $EXPECTED_TORCH_VERSION.*"

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

if [[ ${MATRIX_CHANNEL} = 'nightly' ]]; then
    # Nightly: torch and fbgemm-gpu nightlies share a daily build cadence, so
    # installing the latest of each keeps them ABI-compatible. Versions use
    # dev/date formats that the X+5 mapping can't parse, so don't pin.
    conda run -n "${CONDA_ENV}" pip install torch --index-url "$PYTORCH_URL"
    conda run -n "${CONDA_ENV}" pip install fbgemm-gpu --index-url "$PYTORCH_URL"
else
    # Release/test: a published fbgemm_gpu release is ABI-locked to a specific
    # torch (fbgemm 1.X is built against torch 2.(X+5)). The latest stable torch
    # can be NEWER than the latest fbgemm release (e.g. torch 2.12 vs fbgemm 1.6),
    # which crashes on import with:
    #   fbgemm_gpu_config.so: undefined symbol: ..._ZNR5torch7Library4_defE...
    # So install fbgemm-gpu FIRST, then pin torch to the version that matches the
    # fbgemm that actually got installed (derive it, don't trust version.txt,
    # which may name a release whose binaries aren't published yet).
    conda run -n "${CONDA_ENV}" pip install fbgemm-gpu --index-url "$PYTORCH_URL"
    INSTALLED_FBGEMM_VERSION=$(conda run -n "${CONDA_ENV}" pip show fbgemm_gpu | grep Version | cut -d' ' -f2)
    COMPATIBLE_TORCH_VERSION=$(get_expected_torch_version "$INSTALLED_FBGEMM_VERSION")
    if [[ $? -ne 0 ]]; then
        echo "Failed to determine compatible torch version for fbgemm_gpu=$INSTALLED_FBGEMM_VERSION"
        exit 1
    fi
    echo "Installed fbgemm_gpu=$INSTALLED_FBGEMM_VERSION -> pinning torch==${COMPATIBLE_TORCH_VERSION}.*"
    conda run -n "${CONDA_ENV}" pip install "torch==${COMPATIBLE_TORCH_VERSION}.*" --index-url "$PYTORCH_URL"
fi

# install other requirements
# NOTE: requirements.txt has a floating 'fbgemm-gpu>=1.4.0'; fbgemm_gpu is
# already installed above and satisfies it, so pip will not re-resolve/upgrade
# it here (no --upgrade), preserving the torch<->fbgemm pairing chosen above.
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


# Validate all package versions (skip nightly — versions use dev/date formats)
if [[ ${MATRIX_CHANNEL} != 'nightly' ]]; then
    validate_versions "$EXPECTED_TORCHREC_VERSION" "$EXPECTED_TORCH_VERSION"
fi


# redo for pypi release

if [[ ${MATRIX_CHANNEL} != 'release' ]]; then
    exit 0
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
# Same ABI rule as above, but for the PyPI index: install fbgemm_gpu first, then
# pin torch to the version matching the installed fbgemm (fbgemm 1.X -> torch
# 2.(X+5)) so the pair is never ABI-incompatible.
conda run -n "${CONDA_ENV}" pip install fbgemm-gpu
PYPI_FBGEMM_VERSION=$(conda run -n "${CONDA_ENV}" pip show fbgemm_gpu | grep Version | cut -d' ' -f2)
PYPI_COMPATIBLE_TORCH_VERSION=$(get_expected_torch_version "$PYPI_FBGEMM_VERSION")
if [[ $? -ne 0 ]]; then
    echo "Failed to determine compatible torch version for fbgemm_gpu=$PYPI_FBGEMM_VERSION"
    exit 1
fi
echo "PyPI fbgemm_gpu=$PYPI_FBGEMM_VERSION -> pinning torch==${PYPI_COMPATIBLE_TORCH_VERSION}.*"
conda run -n "${CONDA_ENV}" pip install "torch==${PYPI_COMPATIBLE_TORCH_VERSION}.*"
conda run -n "${CONDA_ENV}" pip install torchrec

# Validate all package versions for PyPI release
validate_versions "$EXPECTED_TORCHREC_VERSION" "$EXPECTED_TORCH_VERSION"

# check directory
ls -R

# check if cuda available
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.cuda.is_available())"

# check cuda version
conda run -n "${CONDA_ENV}" python -c "import torch; print(torch.version.cuda)"

# Finally run smoke test
conda run -n "${CONDA_ENV}" pip install iopath
conda run -n "${CONDA_ENV}" torchrun --nnodes=1 --nproc_per_node=gpu test_installation.py
