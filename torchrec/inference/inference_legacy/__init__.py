#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Inference

Torchrec inference provides a Torch.Deploy based library for GPU inference.

These includes:
    - Model packaging in Python
        - `PredictModule` and `PredictFactory` are the contracts between the Python model authoring and the C++ model serving.
        - `PredictFactoryPackager` can be used to package a PredictFactory class using torch.package.
    - Model serving in C++
        - `BatchingQueue` is a generalized config-based request tensor batching implementation.
        - `GPUExecutor` handles the forward call into the inference model inside Torch.Deploy.

We implemented an example of how to use this library with the TorchRec DLRM model.
    - `examples/dlrm/inference/dlrm_packager.py`: this demonstrates how to export the DLRM model as a torch.package.
    - `examples/dlrm/inference/dlrm_predict.py`: this shows how to use `PredictModule` and `PredictFactory` based on an existing model.
"""

from . import model_packager  # noqa
