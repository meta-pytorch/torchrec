#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import List, Optional

import torch
from torchrec.distributed.embedding_sharding import EmbeddingShardingContext
from torchrec.distributed.embedding_types import KJTList
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


class SequenceShardingContext(EmbeddingShardingContext):
    """
    Stores KJTAllToAll context and reuses it in SequenceEmbeddingsAllToAll.
    SequenceEmbeddingsAllToAll has the same comm pattern as KJTAllToAll.

    Attributes:
        features_before_input_dist (Optional[KeyedJaggedTensor]): stores the original
            KJT before input dist.
        input_splits(List[int]): stores the input splits of KJT AlltoAll.
        output_splits (List[int]): stores the output splits of KJT AlltoAll.
        unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute order of
            KJT bucketize (for row-wise sharding only).
        lengths_after_input_dist (Optional[torch.Tensor]): stores the KJT length after
            input dist.
    """

    # Torch Dynamo does not support default_factory=list:
    # https://github.com/pytorch/pytorch/issues/120108
    # TODO(ivankobzarev): Make this a dataclass once supported

    def __init__(
        self,
        # Fields of EmbeddingShardingContext
        batch_size_per_rank: Optional[List[int]] = None,
        batch_size_per_rank_per_feature: Optional[List[List[int]]] = None,
        batch_size_per_feature_pre_a2a: Optional[List[int]] = None,
        variable_batch_per_feature: bool = False,
        # Fields of SequenceShardingContext
        features_before_input_dist: Optional[KeyedJaggedTensor] = None,
        input_splits: Optional[List[int]] = None,
        output_splits: Optional[List[int]] = None,
        sparse_features_recat: Optional[torch.Tensor] = None,
        unbucketize_permute_tensor: Optional[torch.Tensor] = None,
        lengths_after_input_dist: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            batch_size_per_rank,
            batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a,
            variable_batch_per_feature,
        )
        self.features_before_input_dist: Optional[KeyedJaggedTensor] = (
            features_before_input_dist
        )
        self.input_splits: List[int] = input_splits if input_splits is not None else []
        self.output_splits: List[int] = (
            output_splits if output_splits is not None else []
        )
        self.sparse_features_recat: Optional[torch.Tensor] = sparse_features_recat
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = (
            unbucketize_permute_tensor
        )
        self.lengths_after_input_dist: Optional[torch.Tensor] = lengths_after_input_dist

    def record_stream(self, stream: torch.Stream) -> None:
        if self.features_before_input_dist is not None:
            self.features_before_input_dist.record_stream(stream)
        if self.sparse_features_recat is not None:
            self.sparse_features_recat.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.lengths_after_input_dist is not None:
            self.lengths_after_input_dist.record_stream(stream)


@dataclass
class InferSequenceShardingContext(Multistreamable):
    """
    Stores inference context and reuses it in sequence embedding output_dist or result return.

    Attributes:
        features KJTList: stores the shards of KJT after input dist.
        features_before_input_dist KJT: stores the original input KJT (before input dist).
        unbucketize_permute_tensor Optional[torch.Tensor]: stores unbucketize tensor, only for RowWise sharding.
    """

    features: KJTList
    features_before_input_dist: Optional[KeyedJaggedTensor] = None
    unbucketize_permute_tensor: Optional[torch.Tensor] = None
    bucket_mapping_tensor: Optional[torch.Tensor] = None
    bucketized_length: Optional[torch.Tensor] = None
    embedding_names_per_rank: Optional[List[List[str]]] = None

    def record_stream(self, stream: torch.Stream) -> None:
        for feature in self.features:
            feature.record_stream(stream)
        if self.features_before_input_dist is not None:
            self.features_before_input_dist.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.bucket_mapping_tensor is not None:
            self.bucket_mapping_tensor.record_stream(stream)
        if self.bucketized_length is not None:
            self.bucketized_length.record_stream(stream)
