#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchrec import distributed as distributed, quant as quant
from torchrec.fx import tracer as tracer
from torchrec.modules.embedding_configs import (
    DataType as DataType,
    EmbeddingBagConfig as EmbeddingBagConfig,
    EmbeddingConfig as EmbeddingConfig,
    PoolingType as PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection as EmbeddingBagCollection,
    EmbeddingBagCollectionInterface as EmbeddingBagCollectionInterface,
    EmbeddingCollection as EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import (
    JaggedTensor as JaggedTensor,
    KeyedJaggedTensor as KeyedJaggedTensor,
    KeyedTensor as KeyedTensor,
)
from torchrec.streamable import (
    Multistreamable as Multistreamable,
    Pipelineable as Pipelineable,
)

try:
    # @manual=//torchrec/fb:version
    # pyrefly: ignore[missing-import]
    from .version import __version__ as __version__, github_version as github_version
except ImportError:
    pass
