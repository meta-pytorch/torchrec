#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, MISSING
from typing import List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from .model_input import ModelInput, VariableBatchModelInput


@dataclass
class ModelInputConfig:
    # fixed size model input

    num_batches: int
    batch_size: int
    num_float_features: int
    feature_pooling_avg: int
    device: Optional[str] = None
    use_offsets: bool = False
    long_kjt_indices: bool = True
    long_kjt_offsets: bool = True
    long_kjt_lengths: bool = True
    pin_memory: bool = True
    use_variable_batch: bool = False
    num_dummy_tensor: int = 0
    power_law_alpha: Optional[float] = (
        None  # If set, use power-law distribution for indices
    )

    def __post_init__(self):
        assert self.num_batches is not MISSING, "--num_batches must be specified"
        assert self.batch_size is not MISSING, "--batch_size must be specified"
        assert (
            self.num_float_features is not MISSING
        ), "--num_float_features must be specified"
        assert (
            self.feature_pooling_avg is not MISSING
        ), "--feature_pooling_avg must be specified"

    def generate_batches(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
    ) -> List[ModelInput]:
        """
        Generate model input data for benchmarking.

        Args:
            tables: List of embedding tables

        Returns:
            A list of ModelInput objects representing the generated batches
        """
        device = torch.device(self.device) if self.device is not None else None

        if self.use_variable_batch:
            return [
                VariableBatchModelInput.generate(
                    batch_size=self.batch_size,
                    num_float_features=self.num_float_features,
                    tables=tables,
                    weighted_tables=weighted_tables,
                    use_offsets=self.use_offsets,
                    indices_dtype=(
                        torch.int64 if self.long_kjt_indices else torch.int32
                    ),
                    offsets_dtype=(
                        torch.int64 if self.long_kjt_offsets else torch.int32
                    ),
                    lengths_dtype=(
                        torch.int64 if self.long_kjt_lengths else torch.int32
                    ),
                    device=device,
                    pin_memory=self.pin_memory,
                    num_dummy_tensor=self.num_dummy_tensor,
                )
                for _ in range(self.num_batches)
            ]

        return [
            ModelInput.generate(
                batch_size=self.batch_size,
                tables=tables,
                weighted_tables=weighted_tables,
                num_float_features=self.num_float_features,
                pooling_avg=self.feature_pooling_avg,
                use_offsets=self.use_offsets,
                device=device,
                indices_dtype=(torch.int64 if self.long_kjt_indices else torch.int32),
                offsets_dtype=(torch.int64 if self.long_kjt_offsets else torch.int32),
                lengths_dtype=(torch.int64 if self.long_kjt_lengths else torch.int32),
                pin_memory=self.pin_memory,
                power_law_alpha=self.power_law_alpha,
                num_dummy_tensor=self.num_dummy_tensor,
            )
            for _ in range(self.num_batches)
        ]
