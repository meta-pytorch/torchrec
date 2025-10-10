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
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from .model_input import ModelInput


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
            )
            for batch_size in range(self.num_batches)
        ]
