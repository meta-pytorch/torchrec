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

"""Custom Embedding Configurations using SparseCore on TPU."""

import dataclasses
from typing import Any, List, Optional, Union

from torchrec.modules import embedding_configs

PoolingType = embedding_configs.PoolingType
EmbeddingConfig = embedding_configs.EmbeddingConfig
EmbeddingBagConfig = embedding_configs.EmbeddingBagConfig
dataclass = dataclasses.dataclass


# TODO: Add support for table stacking here.
# We should think about how to take the list of EmbeddingConfigs and stack them
# properly.
@dataclass
class SparseCoreEmbeddingConfig:
    """Wrapper config class that attaches TPU sequence bounds to a standard BaseEmbeddingConfig."""

    config: Union[EmbeddingConfig, EmbeddingBagConfig]
    max_seq_len: Optional[int] = None
    max_ids_per_partition: int = 256
    max_unique_ids_per_partition: int = 256
    suggested_coo_buffer_size_per_device: int = 32

    @property
    def name(self) -> str:
        return self.config.name

    # TODO: We need to make sure that this number is divisible by 8.
    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    # TODO: We need to make sure that this number is divisible by
    # the total number of SC devices.
    @property
    def num_embeddings(self) -> int:
        return self.config.num_embeddings

    @property
    def feature_names(self) -> List[str]:
        return self.config.feature_names

    @property
    def init_fn(self) -> Any:
        return self.config.init_fn

    @property
    def pooling(self) -> PoolingType:
        if isinstance(self.config, EmbeddingBagConfig):
            return self.config.pooling
        raise AttributeError("Wrapped config is not an EmbeddingBagConfig")
