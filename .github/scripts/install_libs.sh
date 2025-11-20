#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "CU_VERSION: ${CU_VERSION}"
echo "CHANNEL: ${CHANNEL}"
echo "CONDA_ENV: ${CONDA_ENV}"

if [[ $CU_VERSION = cu* ]]; then
    # Setting LD_LIBRARY_PATH fixes the runtime error with fbgemm_gpu not
    # being able to locate libnvrtc.so
    echo "[NOVA] Setting LD_LIBRARY_PATH ..."
    conda env config vars set -p ${CONDA_ENV}  \
        LD_LIBRARY_PATH="/usr/local/lib:${CUDA_HOME}/lib64:${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
else
    echo "[NOVA] Setting LD_LIBRARY_PATH ..."
    conda env config vars set -p ${CONDA_ENV}  \
        LD_LIBRARY_PATH="/usr/local/lib:${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
fi

if [ "$CHANNEL" = "nightly" ]; then
    ${CONDA_RUN} pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/"$CU_VERSION"
elif [ "$CHANNEL" = "test" ]; then
    ${CONDA_RUN} pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/test/"$CU_VERSION"
fi


${CONDA_RUN} pip install importlib-metadata click PyYAML
