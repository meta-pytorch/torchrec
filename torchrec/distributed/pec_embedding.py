#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Type

import torch
import torch.nn as nn
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.pec_collision_handlers import (
    CollisionHandlerBase,
    CollisionResult,
    CollisionSplits,
    create_collision_handler,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.pec_embedding_modules import PECEmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


@dataclass
class PECEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Per-module context for ShardedPECEmbeddingCollection.

    Inherits from EmbeddingCollectionContext so it can be passed directly
    to the inner ShardedEmbeddingCollection's compute/output_dist methods.
    """

    prev_remapped_feature_values: List[torch.Tensor] | None = None


class ShardedPECEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,  # CompIn
        List[torch.Tensor],  # DistOut
        Dict[str, JaggedTensor],  # Out
        PECEmbeddingCollectionContext,  # ShrdCtx
    ],
):
    """Sharded PEC Embedding Collection. Wraps ShardedEmbeddingCollection via composition."""

    def __init__(
        self,
        module: PECEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ec_sharder: EmbeddingCollectionSharder,
        env: ShardingEnv,
        device: torch.device,
        qcomm_codecs_registry: Dict[str, QuantizedCommCodecs] | None = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)

        self._embedding_collection: ShardedEmbeddingCollection = ec_sharder.shard(
            module._embedding_collection,
            table_name_to_parameter_sharding,
            env=env,
            device=device,
        )
        self._env: ShardingEnv = env

        assert env.process_group is not None
        self._collision_handlers: List[CollisionHandlerBase] = []
        for (
            sharding_type,
            sharding,
        ) in self._embedding_collection._sharding_type_to_sharding.items():
            self._collision_handlers.append(
                create_collision_handler(
                    sharding_type=sharding_type,
                    device=device,
                    grouped_emb_configs=sharding._grouped_embedding_configs,  # pyre-ignore[16]
                    table_name_to_config=self._embedding_collection._table_name_to_config,
                    process_group=env.process_group,
                    checker_type=module._checker_type,
                )
            )

    def create_context(self) -> PECEmbeddingCollectionContext:
        return PECEmbeddingCollectionContext()

    def input_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        return self._embedding_collection.input_dist(ctx, features)

    def compute(
        self,
        ctx: PECEmbeddingCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        return self._embedding_collection.compute(ctx, dist_input)

    def output_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        return self._embedding_collection.output_dist(ctx, output)

    def compute_and_output_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        input: KJTList,
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        return self._embedding_collection.compute_and_output_dist(ctx, input)

    def detect_collisions(
        self,
        ctx: PECEmbeddingCollectionContext,
        dist_input: KJTList,
    ) -> List[CollisionResult]:
        """Detects collisions for each sharding group's features.

        Reads ctx.prev_remapped_feature_values (set by the pipeline from the
        previous batch's CollisionResult). Returns masks only — KJT splitting
        is done externally via split_features_by_values_mask().

        Args:
            ctx: PEC context for this batch. Pipeline sets
                ctx.prev_remapped_feature_values from the previous batch's
                CollisionResult.remapped_feature_values (None for first batch).
            dist_input: KJTList from input_dist (one KJT per sharding group)

        Returns:
            List of CollisionResult (one per sharding group, typically just one for RW)
        """
        prev_values = ctx.prev_remapped_feature_values
        results: List[CollisionResult] = []
        for i, (handler, features) in enumerate(
            zip(self._collision_handlers, dist_input)
        ):
            prev = prev_values[i] if prev_values is not None else None
            results.append(handler.detect_collisions(features, prev))
        return results

    def collision_split_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        nonoverlapped_features: List[KeyedJaggedTensor],
    ) -> List[Awaitable[CollisionSplits]]:
        """Exchanges per-rank overlapped/nonoverlapped split sizes via AllToAll.

        Delegates to each handler's split_dist. Each awaitable resolves
        to CollisionSplits with per-rank send/receive counts for
        overlapped and nonoverlapped partitions.

        Args:
            ctx: PEC context (sharding_contexts must be set from input_dist)
            nonoverlapped_features: one nonoverlapped KJT per
                sharding group, from split_features_by_values_mask()
                # TODO: test with multiple sharding groups

        Returns:
            List of Awaitable[CollisionSplits], one per sharding group.
        """
        return [
            handler.split_dist(
                nol_features,
                sharding_ctx,
            )
            for handler, nol_features, sharding_ctx in zip(
                self._collision_handlers,
                nonoverlapped_features,
                ctx.sharding_contexts,
            )
        ]

    def permute_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        features_per_group: List[KeyedJaggedTensor],
        forward_overlap_masks: List[torch.Tensor],
    ) -> List[Awaitable[torch.Tensor]]:
        """Distributes collision partition permutations via AllToAll.

        Delegates to each handler's permute_dist. Each awaitable resolves
        to a forward_permute tensor for reordering merged [ol, nol]
        embeddings back to original order.

        Args:
            ctx: PEC context (sharding_contexts must be set from input_dist)
            features_per_group: per-group features after input_dist (for lengths)
            forward_overlap_masks: per-group bool masks for overlapped values

        Returns:
            List of Awaitable[torch.Tensor], one per sharding group, each
            resolving to a forward_permute tensor.
        """
        return [
            handler.permute_dist(
                features,
                mask,
                sharding_ctx,
            )
            for handler, features, mask, sharding_ctx in zip(
                self._collision_handlers,
                features_per_group,
                forward_overlap_masks,
                ctx.sharding_contexts,
            )
        ]

    def compute_and_output_dist_in_partition(
        self,
        ctx: PECEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
        collision_splits: CollisionSplits,
        is_overlapped: bool,
    ) -> List[Awaitable[torch.Tensor]]:
        """Embedding lookup + output dist for one partition (overlapped or nonoverlapped).

        Performs lookup on the partition's features using the inner EC's lookup
        module, then AllToAll's the embeddings back to trainer using the
        collision-specific splits. The embedding AllToAll goes in the reverse
        direction of the features AllToAll, so collision output_splits become
        embedding input_splits and vice versa.

        Args:
            ctx: PEC context (sharding_contexts must be set from input_dist)
            features: partition features (ol or nol KJT from split_features)
            collision_splits: splits from collision_split_dist().wait()
            is_overlapped: True for overlapped partition, False for nonoverlapped

        Returns:
            List of Awaitable[torch.Tensor], one per sharding group.
        """
        partition_idx = 0 if is_overlapped else 1
        awaitables: List[Awaitable[torch.Tensor]] = []

        for lookup, odist, sharding_ctx, sharding_type in zip(
            self._embedding_collection._lookups,
            self._embedding_collection._output_dists,
            ctx.sharding_contexts,
            self._embedding_collection._sharding_type_to_sharding,
        ):
            # Shallow copy to avoid mutating the original sharding context
            partition_ctx = copy(sharding_ctx)

            # The original lengths_after_input_dist reflects the full batch.
            # Replace with this partition's lengths so the embedding AllToAll
            # knows the correct per-feature sequence lengths for recat.
            partition_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            # Embedding AllToAll reverses the direction of features AllToAll:
            partition_ctx.input_splits = collision_splits.output_splits[partition_idx]
            partition_ctx.output_splits = collision_splits.input_splits[partition_idx]

            # upt maps positions in the full (pre-split) batch. Each partition
            # only has a subset of values, so the full upt would have wrong size
            # and wrong index mappings. PEC handles reordering after merging both
            # partitions via forward_permute from permute_dist.
            partition_ctx.unbucketize_permute_tensor = None

            embedding_dim = self._embedding_collection._embedding_dim_for_sharding_type(
                sharding_type
            )
            embs = lookup(features)
            awaitables.append(odist(embs.view(-1, embedding_dim), partition_ctx))

        return awaitables


class PECEmbeddingCollectionSharder(BaseEmbeddingSharder[PECEmbeddingCollection]):
    """Sharder for PECEmbeddingCollection. Enforces RW-only sharding."""

    def __init__(
        self,
        ec_sharder: EmbeddingCollectionSharder | None = None,
        fused_params: Dict[str, Any] | None = None,
        qcomm_codecs_registry: Dict[str, QuantizedCommCodecs] | None = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry)
        self._embedding_collection_sharder: EmbeddingCollectionSharder = (
            ec_sharder
            or EmbeddingCollectionSharder(
                fused_params=fused_params,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    @property
    def module_type(self) -> Type[PECEmbeddingCollection]:
        return PECEmbeddingCollection

    def shardable_parameters(
        self, module: PECEmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return self._embedding_collection_sharder.shardable_parameters(
            module._embedding_collection
        )

    def shard(
        self,
        module: PECEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: torch.device | None = None,
        module_fqn: str | None = None,
    ) -> "ShardedPECEmbeddingCollection":
        if device is None:
            device = torch.device("cuda")

        return ShardedPECEmbeddingCollection(
            module,
            params,
            ec_sharder=self._embedding_collection_sharder,
            env=env,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
