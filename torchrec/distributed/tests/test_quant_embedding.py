#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from torchrec.distributed.quant_embedding import create_infer_embedding_sharding
from torchrec.distributed.sharding.cw_sequence_sharding import (
    InferCwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    InferRwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.tw_sequence_sharding import (
    InferTwSequenceEmbeddingSharding,
)
from torchrec.distributed.types import ShardingEnv, ShardingType


class CreateInferEmbeddingShardingTest(unittest.TestCase):
    def test_device_sharding_types(self) -> None:
        """Dispatch only: checks the class picked per device, not behavior of specific accelerator."""
        cases = [
            (ShardingType.TABLE_WISE.value, InferTwSequenceEmbeddingSharding),
            (ShardingType.ROW_WISE.value, InferRwSequenceEmbeddingSharding),
            (ShardingType.COLUMN_WISE.value, InferCwSequenceEmbeddingSharding),
        ]
        for device_type in ("cuda", "mtia", "xpu"):
            for sharding_type, expected in cases:
                with self.subTest(
                    device_type=device_type, sharding_type=sharding_type
                ), patch(
                    "torchrec.distributed.quant_embedding.get_device_from_sharding_infos",
                    return_value=device_type,
                ):
                    sharding = create_infer_embedding_sharding(
                        sharding_type,
                        [],
                        ShardingEnv.from_local(world_size=2, rank=0),
                        torch.device(device_type, 0),
                    )
                    self.assertIsInstance(sharding, expected)
