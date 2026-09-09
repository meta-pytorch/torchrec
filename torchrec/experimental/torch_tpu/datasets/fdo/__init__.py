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

# Re-exports resolve lazily (PEP 562); see the package __init__ two levels up for the
# rationale. Here the two backing modules have disjoint requirements -- `absl` for the
# CSV client, `etils` for the client interface -- so eager re-exports made each depend
# on the other's.

import importlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchrec.experimental.torch_tpu.datasets.fdo.csv_file_fdo_client import (
        CSVFileFDOClient,
    )
    from torchrec.experimental.torch_tpu.datasets.fdo.fdo_client import (
        FDOClient,
        KeyedSparseCoreInputStats,
        SparseCoreInputStats,
    )


_SYMBOL_TO_MODULE: dict[str, str] = {
    "CSVFileFDOClient": "torchrec.experimental.torch_tpu.datasets.fdo.csv_file_fdo_client",
    "FDOClient": "torchrec.experimental.torch_tpu.datasets.fdo.fdo_client",
    "KeyedSparseCoreInputStats": "torchrec.experimental.torch_tpu.datasets.fdo.fdo_client",
    "SparseCoreInputStats": "torchrec.experimental.torch_tpu.datasets.fdo.fdo_client",
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
    "FDOClient",
    "CSVFileFDOClient",
    "SparseCoreInputStats",
    "KeyedSparseCoreInputStats",
]
