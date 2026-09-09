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

# Re-exports resolve lazily (PEP 562); see the package __init__ one level up for the
# rationale. Here specifically: `embedding_modules` reaches jax through `pallas`, so
# eager re-exports made the cheap, torch-only `embedding_configs` unreachable without
# it.

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchrec.experimental.torch_tpu.modules.embedding_configs import (
        SparseCoreEmbeddingConfig,
    )
    from torchrec.experimental.torch_tpu.modules.embedding_modules import (
        TPUEmbeddingUnfused,
    )
    from torchrec.experimental.torch_tpu.modules.fused_embedding_modules import (
        SparseCoreFusedEmbeddingBagCollection,
        SparseCoreFusedEmbeddingCollection,
    )


_SYMBOL_TO_MODULE: dict[str, str] = {
    "SparseCoreEmbeddingConfig": "torchrec.experimental.torch_tpu.modules.embedding_configs",
    "SparseCoreFusedEmbeddingBagCollection": "torchrec.experimental.torch_tpu.modules.fused_embedding_modules",
    "SparseCoreFusedEmbeddingCollection": "torchrec.experimental.torch_tpu.modules.fused_embedding_modules",
    "TPUEmbeddingUnfused": "torchrec.experimental.torch_tpu.modules.embedding_modules",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(module_name), name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted([*globals(), *_SYMBOL_TO_MODULE])


__all__ = [
    "SparseCoreEmbeddingConfig",
    "SparseCoreFusedEmbeddingBagCollection",
    "SparseCoreFusedEmbeddingCollection",
    "TPUEmbeddingUnfused",
]
