#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Any, cast, Dict, List, Optional, Type

import torch
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.mc_embedding_modules import (
    BaseManagedCollisionEmbeddingCollectionSharder,
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class ManagedCollisionEmbeddingCollectionContext(EmbeddingCollectionContext):

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        evictions_per_table: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        remapped_kjt: Optional[KJTList] = None,
    ) -> None:
        super().__init__(sharding_contexts, input_features, reverse_indices)
        self.evictions_per_table: Optional[Dict[str, Optional[torch.Tensor]]] = (
            evictions_per_table
        )
        self.remapped_kjt: Optional[KJTList] = remapped_kjt

    def record_stream(self, stream: torch.Stream) -> None:
        super().record_stream(stream)
        if self.evictions_per_table:
            #  pyre-ignore
            for value in self.evictions_per_table.values():
                if value is None:
                    continue
                value.record_stream(stream)
        if self.remapped_kjt is not None:
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
            self.remapped_kjt.record_stream(stream)


class ShardedManagedCollisionEmbeddingCollection(
    BaseShardedManagedCollisionEmbeddingCollection[
        ManagedCollisionEmbeddingCollectionContext
    ]
):
    def __init__(
        self,
        module: ManagedCollisionEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ec_sharder: EmbeddingCollectionSharder,
        mc_sharder: ManagedCollisionCollectionSharder,
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            ec_sharder,
            mc_sharder,
            env,
            device,
        )

    # For consistency with embeddingbag
    @property
    def _embedding_collection(self) -> ShardedEmbeddingCollection:
        return cast(ShardedEmbeddingCollection, self._embedding_module)

    def create_context(
        self,
    ) -> ManagedCollisionEmbeddingCollectionContext:
        return ManagedCollisionEmbeddingCollectionContext(sharding_contexts=[])


class ManagedCollisionEmbeddingCollectionSharder(
    BaseManagedCollisionEmbeddingCollectionSharder[ManagedCollisionEmbeddingCollection]
):
    def __init__(
        self,
        ec_sharder: Optional[EmbeddingCollectionSharder] = None,
        mc_sharder: Optional[ManagedCollisionCollectionSharder] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            ec_sharder
            or EmbeddingCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
            ),
            mc_sharder or ManagedCollisionCollectionSharder(),
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    def shard(
        self,
        module: ManagedCollisionEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedManagedCollisionEmbeddingCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedManagedCollisionEmbeddingCollection(
            module,
            params,
            # pyre-ignore [6]
            ec_sharder=self._e_sharder,
            mc_sharder=self._mc_sharder,
            env=env,
            device=device,
        )

    @property
    def module_type(self) -> Type[ManagedCollisionEmbeddingCollection]:
        return ManagedCollisionEmbeddingCollection

    @property
    def sharded_module_type(self) -> Type[ShardedManagedCollisionEmbeddingCollection]:
        return ShardedManagedCollisionEmbeddingCollection
