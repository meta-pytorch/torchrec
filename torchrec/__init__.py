#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torchrec.distributed  # noqa
import torchrec.quant  # noqa
from torchrec.fx import tracer  # noqa
from torchrec.modules.embedding_configs import (  # noqa
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (  # noqa
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
)  # noqa
from torchrec.sparse.jagged_tensor import (  # noqa
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.streamable import Multistreamable, Pipelineable  # noqa

try:
    # @manual=//torchrec/fb:version
    from .version import __version__, github_version  # noqa
except ImportError:
    pass
