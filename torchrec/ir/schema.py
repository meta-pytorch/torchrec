#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List, Optional, Tuple

from torchrec.modules.embedding_configs import DataType, PoolingType


# Same as EmbeddingBagConfig but serializable
@dataclass
class EmbeddingBagConfigMetadata:
    num_embeddings: int
    embedding_dim: int
    name: str
    data_type: DataType
    feature_names: List[str]
    weight_init_max: Optional[float]
    weight_init_min: Optional[float]
    need_pos: bool
    pooling: PoolingType


@dataclass
class EBCMetadata:
    tables: List[EmbeddingBagConfigMetadata]
    is_weighted: bool
    device: Optional[str]


@dataclass
class FPEBCMetadata:
    is_fp_collection: bool
    features: List[str]


@dataclass
class PositionWeightedModuleMetadata:
    max_feature_length: int


@dataclass
class PositionWeightedModuleCollectionMetadata:
    max_feature_lengths: List[Tuple[str, int]]


@dataclass
class KTRegroupAsDictMetadata:
    groups: List[List[str]]
    keys: List[str]
    emb_dtype: Optional[str]
