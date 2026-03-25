#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Torchrec Jagged Tensors

It has 3 classes: JaggedTensor, KeyedJaggedTensor, KeyedTensor.

JaggedTensor

It represents an (optionally weighted) jagged tensor. A JaggedTensor is a
tensor with a jagged dimension which is dimension whose slices may be of
different lengths. See KeyedJaggedTensor docstring for full example and further
information.

KeyedJaggedTensor

KeyedJaggedTensor has additional "Key" information. Keyed on first dimension,
and jagged on last dimension. Please refer to KeyedJaggedTensor docstring for full example and
further information.

KeyedTensor

KeyedTensor holds a concatenated list of dense tensors each of which can be accessed by a key.
Keyed dimension can be variable length (length_per_key). Common use cases uses include storage
of pooled embeddings of different dimensions. Please refer to KeyedTensor docstring for full
example and further information.
"""

from . import jagged_tensor  # noqa
