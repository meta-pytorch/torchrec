#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os


def main():
    """
    For filtering out certain cuda versions in the build matrix that
    determines with nightly builds are run. This ensures TorchRec is
    always consistent in version compatibility with FBGEMM.
    """

    full_matrix_string = os.environ["MAT"]
    full_matrix = json.loads(full_matrix_string)
    """
    Matrix contents can be found in a github "Build Linux Wheels"
    log output. A typical example is
    {
        "include": [
            {
            "python_version": "3.9",
            "gpu_arch_type": "cpu",
            "gpu_arch_version": "",
            "desired_cuda": "cpu",
            "container_image": "pytorch/manylinux2_28-builder:cpu",
            "package_type": "manywheel",
            "build_name": "manywheel-py3_9-cpu",
            "validation_runner": "linux.2xlarge",
            "installation": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu",
            "channel": "nightly",
            "upload_to_base_bucket": "no",
            "stable_version": "2.7.1",
            "use_split_build": false
            },
            {
            "python_version": "3.9",
            "gpu_arch_type": "cuda",
            "gpu_arch_version": "12.6",
            "desired_cuda": "cu126",
            "container_image": "pytorch/manylinux2_28-builder:cuda12.6",
            "package_type": "manywheel",
            "build_name": "manywheel-py3_9-cuda12_6",
            "validation_runner": "linux.g5.4xlarge.nvidia.gpu",
            "installation": "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126",
            "channel": "nightly",
            "upload_to_base_bucket": "no",
            "stable_version": "2.7.1",
            "use_split_build": false
            }
        ]
    }
    """

    new_matrix_entries = []

    for entry in full_matrix["include"]:
        # if entry["desired_cuda"] == "cu130":
        #     # fbgemm only supports cuda 12.6, 12.8, 12.9 and 13.0
        #     continue
        if entry["python_version"] in ("3.9"):
            # stop python3.9 support, and skipp python3.14 due to incompatibility with torch.compile
            # for python version: https://devguide.python.org/versions/
            continue
        new_matrix_entries.append(entry)

    new_matrix = {"include": new_matrix_entries}
    print(json.dumps(new_matrix))


if __name__ == "__main__":
    main()
