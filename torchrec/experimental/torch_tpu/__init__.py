#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Re-exports resolve lazily (PEP 562) rather than at package-import time.
#
# Importing ANY submodule of this package runs this file first, so eager re-exports
# made every sibling depend on the union of their requirements: `jax` (via
# modules.embedding_modules -> pallas), `etils`, and the `pybind_input_preprocessing`
# extension, which setup.py deliberately does not build. None of those exist in an OSS
# install, so pure-torch modules such as `uneven_all_to_all` -- and their tests --
# became unimportable there.
#
# Resolving on first attribute access keeps `from torchrec.experimental.torch_tpu
# import <symbol>` working while confining each dependency to the code that needs it.
# Touching a symbol whose backing module is unavailable still raises the underlying
# ImportError, naming the missing package.

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchrec.experimental.torch_tpu.checkpoint.planners import (
        SparseCoreLoadPlanner,
        SparseCoreSavePlanner,
    )
    from torchrec.experimental.torch_tpu.datasets.dataloader import (
        SparseCoreBatch,
        SparseCoreDataLoader,
    )
    from torchrec.experimental.torch_tpu.datasets.input_preprocessing import (
        KeyedSparseCorePreprocessedInput,
        SparseCoreInputPreprocessor,
        SparseCorePreprocessedInput,
    )
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
    "KeyedSparseCorePreprocessedInput": "torchrec.experimental.torch_tpu.datasets.input_preprocessing",
    "SparseCoreBatch": "torchrec.experimental.torch_tpu.datasets.dataloader",
    "SparseCoreDataLoader": "torchrec.experimental.torch_tpu.datasets.dataloader",
    "SparseCoreEmbeddingConfig": "torchrec.experimental.torch_tpu.modules.embedding_configs",
    "SparseCoreFusedEmbeddingBagCollection": "torchrec.experimental.torch_tpu.modules.fused_embedding_modules",
    "SparseCoreFusedEmbeddingCollection": "torchrec.experimental.torch_tpu.modules.fused_embedding_modules",
    "SparseCoreInputPreprocessor": "torchrec.experimental.torch_tpu.datasets.input_preprocessing",
    "SparseCoreLoadPlanner": "torchrec.experimental.torch_tpu.checkpoint.planners",
    "SparseCorePreprocessedInput": "torchrec.experimental.torch_tpu.datasets.input_preprocessing",
    "SparseCoreSavePlanner": "torchrec.experimental.torch_tpu.checkpoint.planners",
    "TPUEmbeddingUnfused": "torchrec.experimental.torch_tpu.modules.embedding_modules",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(module_name), name)
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted([*globals(), *_SYMBOL_TO_MODULE])


__all__ = [
    "SparseCoreEmbeddingConfig",
    "SparseCoreFusedEmbeddingBagCollection",
    "SparseCoreFusedEmbeddingCollection",
    "TPUEmbeddingUnfused",
    "SparseCoreInputPreprocessor",
    "KeyedSparseCorePreprocessedInput",
    "SparseCorePreprocessedInput",
    "SparseCoreBatch",
    "SparseCoreDataLoader",
    "SparseCoreSavePlanner",
    "SparseCoreLoadPlanner",
]
