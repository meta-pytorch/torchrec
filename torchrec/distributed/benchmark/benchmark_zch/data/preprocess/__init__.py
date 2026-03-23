#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from .criteo_kaggle import get_criteo_kaggle_dataloader
from .kuairand_1k import get_kuairand_1k_dataloader
from .kuairand_27k import get_kuairand_27k_dataloader  # noqa: F401
from .movielens_1m import get_movielens_1m_dataloader  # noqa: F401
