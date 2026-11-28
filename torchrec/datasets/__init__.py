#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Datasets

Torchrec contains two popular RecSys datasets, the `Kaggle/Criteo Display Advertising <https://www.kaggle.com/c/criteo-display-ad-challenge/>`_ Dataset
and the `MovieLens 20M <https://grouplens.org/datasets/movielens/20m/>`_ Dataset.

Additionally, it contains a RandomDataset, which is useful to generate random data in the same format as the above.

Lastly, it contains scripts and utilities for pre-processing, loading, etc.

Example::

    from torchrec.datasets.criteo import criteo_kaggle
    datapipe = criteo_terabyte(
        ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
    )
    datapipe = dp.iter.Batcher(datapipe, 100)
    datapipe = dp.iter.Collator(datapipe)
    batch = next(iter(datapipe))
"""

from torchrec.datasets import (
    criteo as criteo,
    movielens as movielens,
    random as random,
    utils as utils,
)
