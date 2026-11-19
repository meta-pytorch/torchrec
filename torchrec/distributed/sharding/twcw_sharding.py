#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional

import torch
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.sharding.cw_sharding import CwPooledEmbeddingSharding
from torchrec.distributed.types import QuantizedCommCodecs, ShardingEnv


class TwCwPooledEmbeddingSharding(CwPooledEmbeddingSharding):
    """
    Shards embedding bags table-wise column-wise, i.e.. a given embedding table is
    partitioned along its columns and the table slices are placed on all ranks
    within a host group.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            sharding_infos,
            env,
            device,
            permute_embeddings=permute_embeddings,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
