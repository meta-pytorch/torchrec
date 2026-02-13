#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from torchrec.optim.keyed import KeyedOptimizer


class DummyKeyedOptimizer(KeyedOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # pyrefly: ignore[bad-override]
    def step(self, closure: Any) -> None:
        pass  # Override NotImplementedError.
