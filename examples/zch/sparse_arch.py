#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchrec import (
    EmbeddingCollection,
    EmbeddingConfig,
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)

# For MPZCH
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)

# For MPZCH
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection

# For original MC
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)


class SparseArch(nn.Module):
    """
    Class SparseArch
    An example of SparseArch with 2 tables, each with 2 features.
    It looks up the corresponding embedding for incoming KeyedJaggedTensors with 2 features
    and returns the corresponding embeddings.

    Parameters:
        tables(List[EmbeddingConfig]): List of EmbeddingConfig that defines the embedding table
        device(torch.device): device on which the embedding table should be placed
        buckets(int): number of buckets for each table
        input_hash_size(int): input hash size for each table
        return_remapped(bool): whether to return remapped features, if so, the return will be
            a tuple of (Embedding(KeyedTensor), Remapped_ID(KeyedJaggedTensor)), otherwise, the return will be
            a tuple of (Embedding(KeyedTensor), None)
        is_inference(bool): whether to use inference mode. In inference mode, the module will not update the embedding table
        use_mpzch(bool): whether to use MPZCH or not. If true, the module will use MPZCH managed collision module,
            otherwise, it will use original MC managed collision module
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        buckets: int = 4,
        input_hash_size: int = 4000,
        return_remapped: bool = False,
        is_inference: bool = False,
        use_mpzch: bool = False,
    ) -> None:
        super().__init__()
        self._return_remapped = return_remapped

        mc_modules = {}

        if (
            use_mpzch
        ):  # if using the MPZCH module, we create a HashZchManagedCollisionModule for each table
            mc_modules["table_0"] = HashZchManagedCollisionModule(
                is_inference=is_inference,
                zch_size=(
                    tables[0].num_embeddings
                ),  # the zch size, that is, the size of local embedding table, should be the same as the size of the embedding table
                input_hash_size=input_hash_size,  # the input hash size, that is, the size of the input id space
                device=device,  # the device on which the embedding table should be placed
                total_num_buckets=buckets,  # the number of buckets, the detailed explanation of the use of buckets can be found in the readme file
                eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,  # the eviction policy name, in this example use the single ttl eviction policy, which assume an id is evictable if it has been in the table longer than the ttl (time to live)
                eviction_config=HashZchEvictionConfig(  # Here we need to specify for each feature, what is the ttl, that is, how long an id can stay in the table before it is evictable
                    features=[
                        "feature_0"
                    ],  # because we only have one feature "feature_0" in this table, so we only need to specify the ttl for this feature
                    single_ttl=1,  # The unit of ttl is hour. Let's set the ttl to be default to 1, which means an id is evictable if it has been in the table for more than one hour.
                ),
            )
            mc_modules["table_1"] = HashZchManagedCollisionModule(
                is_inference=is_inference,
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=input_hash_size,
                total_num_buckets=buckets,
                eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=["feature_1"],
                    single_ttl=1,
                ),
            )
        else:  # if not using the MPZCH module, we create a MCHManagedCollisionModule for each table
            mc_modules["table_0"] = MCHManagedCollisionModule(
                zch_size=(tables[0].num_embeddings),
                input_hash_size=input_hash_size,
                device=device,
                eviction_interval=2,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )
            mc_modules["table_1"] = MCHManagedCollisionModule(
                zch_size=(tables[1].num_embeddings),
                device=device,
                input_hash_size=input_hash_size,
                eviction_interval=1,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )

        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(
                    tables=tables,
                    device=device,
                ),
                ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=tables,
                ),
                return_remapped_features=self._return_remapped,
            )
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:
        return self._mc_ec(kjt)
