# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchrec.experimental.torch_tpu.pallas import ops
from torchrec.experimental.torch_tpu.pallas.lookup import (
    batched_tpu_embedding_lookup,
    single_tpu_embedding_lookup,
)

__all__ = [
    "ops",
    "batched_tpu_embedding_lookup",
    "single_tpu_embedding_lookup",
]
