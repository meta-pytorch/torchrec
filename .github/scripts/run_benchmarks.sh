#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exit immediately if a command exits with a non-zero status.
set -e

python --version
python -c "import torch, fbgemm_gpu, torchrec; print(f'torch {torch.__version__}\nfbgemm_gpu {fbgemm_gpu.__version__}\ntorchrec {torchrec.__version__}')"
nvidia-smi

# torchrec directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TORCHREC_DIR=$(realpath "$SCRIPT_DIR"/../../torchrec)
YAML_DIR="$TORCHREC_DIR"/distributed/benchmark/yaml

# working directory
WORK_DIR=$(pwd)

echo "torchrec directory: $TORCHREC_DIR"
echo "working directory: $WORK_DIR"

# base pipeline
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
    --yaml_config="$YAML_DIR"/base_pipeline_light.yml \
    --memory_snapshot=True

# SDD pipeline
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
    --yaml_config="$YAML_DIR"/base_pipeline_light.yml \
    --memory_snapshot=True \
    --pipeline="sparse" \
    --name="sparse_data_dist_light"
