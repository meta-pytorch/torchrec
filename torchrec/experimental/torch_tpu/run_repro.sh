#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Minimal launcher for repro_collective_lowering.py -- delegates to the run harness in
# fb/experiments/torchtpu (run_pod.sh) with this file's path relative to that dir.
#   ./run_repro.sh          # multi-host (world=32) -- reproduces the failure (SIGABRT)
#   ./run_repro.sh single   # single-host (world=8) control -- lowers cleanly

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="$(cd "${HERE}/../../fb/experiments/torchtpu" && pwd)"
REPRO_REL="../../../experimental/torch_tpu/repro_collective_lowering.py"

cd "${HARNESS}"
if [ "${1:-}" = "single" ]; then
  exec ./run_pod.sh run "${REPRO_REL}"
fi
exec ./run_pod.sh run16 "${REPRO_REL}"
