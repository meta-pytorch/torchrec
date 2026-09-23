#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        return_remapped: bool = False,
        input_hash_size: List[int] | int = 4000,
        allow_in_place_embed_weight_update: bool = False,
        use_mpzch: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped
        hash_sizes = [
            input_hash_size if isinstance(input_hash_size, int) else input_hash_size[i]
            for i in range(len(tables))
        ]
        mc_modules: dict[str, ManagedCollisionModule] = {}
        if use_mpzch:
            num_buckets: List[int] = [
                int(t.total_num_buckets) if t.total_num_buckets is not None else 4
                for t in tables
            ]
            # Parameters hard-coded from test_quant_mc_embedding
            mc_modules["table_0"] = HashZchManagedCollisionModule(
                zch_size=(tables[0].num_embeddings),
                input_hash_size=hash_sizes[0],
                device=device,
                total_num_buckets=num_buckets[0],
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_0"],
                    single_ttl=1,
                ),
                max_probe=5,
            )
            mc_modules["table_1"] = HashZchManagedCollisionModule(
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=hash_sizes[1],
                total_num_buckets=num_buckets[1],
                eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_1"],
                    single_ttl=1,
                ),
                max_probe=5,
            )
        else:
            mc_modules["table_0"] = MCHManagedCollisionModule(
                zch_size=(tables[0].num_embeddings),
                input_hash_size=hash_sizes[0],
                device=device,
                eviction_interval=2,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )
            mc_modules["table_1"] = MCHManagedCollisionModule(
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=hash_sizes[1],
                eviction_interval=2,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )

        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(
                    tables=tables,
                    device=device,
                ),
                ManagedCollisionCollection(
                    # pyrefly: ignore[bad-argument-type]
                    managed_collision_modules=mc_modules,
                    embedding_configs=tables,
                ),
                return_remapped_features=self._return_remapped,
                allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, JaggedTensor]]]:
        ec_out, remapped_ids_out = self._mc_ec(kjt)
        pred = torch.cat(
            [ec_out[key].values() for key in ["feature_0", "feature_1"]],
            dim=0,
        )
        loss = pred.mean()
        return loss, remapped_ids_out
