#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Quantization

Torchrec provides a quantized version of EmbeddingBagCollection for inference.
It relies on fbgemm quantized ops.
This reduces the size of the model weights and speeds up model execution.

Example:
    >>> import torch.quantization as quant
    >>> import torchrec.quant as trec_quant
    >>> import torchrec as trec
    >>> qconfig = quant.QConfig(
    >>>     activation=quant.PlaceholderObserver,
    >>>     weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
    >>> )
    >>> quantized = quant.quantize_dynamic(
    >>>     module,
    >>>     qconfig_spec={
    >>>         trec.EmbeddingBagCollection: qconfig,
    >>>     },
    >>>     mapping={
    >>>         trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
    >>>     },
    >>>     inplace=inplace,
    >>> )
"""

from torchrec.quant.embedding_modules import EmbeddingBagCollection  # noqa
