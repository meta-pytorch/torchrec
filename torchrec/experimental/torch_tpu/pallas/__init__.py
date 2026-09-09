#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Re-exports resolve lazily (PEP 562); see the package __init__ one level up for the
# rationale. Here specifically: `lookup` imports jax, which an OSS install does not
# have, and `modules.embedding_modules` imports `pallas.ops`. Eagerly, that made jax a
# hard requirement of the embedding modules even though `ops` itself only needs torch.

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchrec.experimental.torch_tpu.pallas import ops
    from torchrec.experimental.torch_tpu.pallas.lookup import (
        batched_tpu_embedding_lookup,
        single_tpu_embedding_lookup,
    )


_SYMBOL_TO_MODULE: dict[str, str] = {
    "batched_tpu_embedding_lookup": "torchrec.experimental.torch_tpu.pallas.lookup",
    "ops": "torchrec.experimental.torch_tpu.pallas.ops",
    "single_tpu_embedding_lookup": "torchrec.experimental.torch_tpu.pallas.lookup",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    # `ops` is a re-exported submodule, so it resolves to the module itself; every
    # other entry names a symbol defined inside its module.
    attr = module if module_name.endswith(f".{name}") else getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted([*globals(), *_SYMBOL_TO_MODULE])


__all__ = [
    "ops",
    "batched_tpu_embedding_lookup",
    "single_tpu_embedding_lookup",
]
