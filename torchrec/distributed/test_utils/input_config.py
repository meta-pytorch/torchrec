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

from .model_input import (
    generate_overlapping_batches,
    ModelInput,
    VariableBatchModelInput,
)


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
    overlap_ratio: Optional[float] = (
        None  # If set, generate batches with controlled index overlap
    )
    overlap_tables: Optional[List[str]] = (
        None  # If set, only apply overlap to these table names
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
        world_size: int = 1,
        rank: int = 0,
    ) -> List[ModelInput]:
        """
        Generate model input data for benchmarking.

        Args:
            tables: List of embedding tables
            weighted_tables: List of weighted embedding tables
            world_size: Number of ranks (used with overlap_ratio)
            rank: Current rank (used with overlap_ratio)

        Returns:
            A list of ModelInput objects representing the generated batches
        """
        device = torch.device(self.device) if self.device is not None else None

        if self.overlap_ratio is not None:
            # generate_overlapping_batches produces batches for ALL ranks at
            # once, but each rank calls this method independently. We seed the
            # RNG deterministically so every rank produces the same global
            # batches, then each rank picks its own slice. The original RNG
            # state is saved and restored so the seeding does not affect
            # subsequent random operations (e.g. weight init, dropout).
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(42)
            all_batches = generate_overlapping_batches(
                tables=tables,
                batch_size=self.batch_size,
                world_size=world_size,
                num_batches=self.num_batches,
                overlap_ratio=self.overlap_ratio,
                weighted_tables=weighted_tables,
                num_float_features=self.num_float_features,
                pooling_avg=self.feature_pooling_avg,
                device=device or torch.device("cpu"),
                indices_dtype=(torch.int64 if self.long_kjt_indices else torch.int32),
                lengths_dtype=(torch.int64 if self.long_kjt_lengths else torch.int32),
                overlap_tables=self.overlap_tables,
                pin_memory=self.pin_memory,
            )
            torch.random.set_rng_state(rng_state)
            return [all_batches[b][rank] for b in range(self.num_batches)]

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
