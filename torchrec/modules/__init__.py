#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Common Modules

The torchrec modules contain a collection of various modules.

These modules include:
    - extensions of `nn.Embedding` and `nn.EmbeddingBag`, called `EmbeddingBagCollection`
      and `EmbeddingCollection` respectively.
    - established modules such as `DeepFM <https://arxiv.org/pdf/1703.04247.pdf>`_ and
      `CrossNet <https://arxiv.org/abs/1708.05123>`_.
    - common module patterns such as `MLP` and `SwishLayerNorm`.
    - custom modules for TorchRec such as `PositionWeightedModule` and
      `LazyModuleExtensionMixin`.
    - `EmbeddingTower` and `EmbeddingTowerCollection`, logical "tower" of embeddings
      passed to provided interaction module.
"""

from . import (  # noqa  # noqa  # noqa  # noqa  # noqa  # noqa  # noqa  # noqa  # noqa
    activation,
    crossnet,
    deepfm,
    embedding_configs,
    embedding_modules,
    embedding_tower,
    feature_processor,
    lazy_extension,
    mlp,
)
