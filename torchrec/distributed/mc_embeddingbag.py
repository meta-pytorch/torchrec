# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Optional, Type, TypeVar

import torch
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.mc_embedding_modules import (
    BaseManagedCollisionEmbeddingCollectionSharder,
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder
from torchrec.distributed.types import (
    Awaitable,
    Multistreamable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.modules.utils import SequenceVBEContext
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


@dataclass
class ManagedCollisionEmbeddingBagCollectionContext(EmbeddingBagCollectionContext):
    evictions_per_table: Optional[Dict[str, Optional[torch.Tensor]]] = None
    remapped_kjt: Optional[KJTList] = None
    seq_vbe_ctx: List[SequenceVBEContext] = field(default_factory=list)

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


class ShardedManagedCollisionEmbeddingBagCollection(
    BaseShardedManagedCollisionEmbeddingCollection[
        ManagedCollisionEmbeddingBagCollectionContext
    ]
):
    def __init__(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        mc_sharder: ManagedCollisionCollectionSharder,
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            ebc_sharder,
            mc_sharder,
            env,
            device,
        )

    # For backwards compat, some references still to self._embedding_bag_collection
    @property
    def _embedding_bag_collection(self) -> ShardedEmbeddingBagCollection:
        return cast(ShardedEmbeddingBagCollection, self._embedding_module)

    def create_context(
        self,
    ) -> ManagedCollisionEmbeddingBagCollectionContext:
        return ManagedCollisionEmbeddingBagCollectionContext(sharding_contexts=[])

    def input_dist(
        self,
        ctx: ShrdCtx,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:

        ctx.variable_batch_per_feature = features.variable_stride_per_key()
        ctx.inverse_indices = features.inverse_indices_or_none()

        if self._managed_collision_collection._has_uninitialized_input_dists:
            self._managed_collision_collection._create_input_dists(
                input_feature_names=features.keys()
            )
            self._managed_collision_collection._has_uninitialized_input_dists = False

            # pyre-ignore [16]
            if ctx.variable_batch_per_feature:
                if self._return_remapped_features:
                    raise NotImplementedError(
                        "VBE is not supported currently for return_remapped_features=True."
                    )

                # pyre-ignore
                self._embedding_module._create_inverse_indices_permute_indices(
                    ctx.inverse_indices  # pyre-ignore [16]
                )

        return self._managed_collision_collection.input_dist(
            # pyre-fixme [6]
            ctx,
            features,
        )


class ManagedCollisionEmbeddingBagCollectionSharder(
    BaseManagedCollisionEmbeddingCollectionSharder[
        ManagedCollisionEmbeddingBagCollection
    ]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        mc_sharder: Optional[ManagedCollisionCollectionSharder] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            ebc_sharder
            or EmbeddingBagCollectionSharder(
                fused_params=fused_params, qcomm_codecs_registry=qcomm_codecs_registry
            ),
            mc_sharder or ManagedCollisionCollectionSharder(),
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    def shard(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedManagedCollisionEmbeddingBagCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedManagedCollisionEmbeddingBagCollection(
            module,
            params,
            # pyre-ignore [6]
            ebc_sharder=self._e_sharder,
            mc_sharder=self._mc_sharder,
            env=env,
            device=device,
        )

    @property
    def module_type(self) -> Type[ManagedCollisionEmbeddingBagCollection]:
        return ManagedCollisionEmbeddingBagCollection
