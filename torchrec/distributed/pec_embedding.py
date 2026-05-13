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
    CollisionPermutation,
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
from torchrec.modules.utils import construct_jagged_tensors
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


@dataclass
class PECEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Per-module context for ShardedPECEmbeddingCollection.

    Inherits from EmbeddingCollectionContext so it can be passed directly
    to the inner ShardedEmbeddingCollection's compute/output_dist methods.
    """

    prev_remapped_feature_values: List[torch.Tensor] | None = None


class PECEmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    """Merges overlapped and nonoverlapped embedding partitions.

    On wait, concatenates [ol, nol] embeddings per sharding group and applies
    the corresponding forward_permute to restore original order, then constructs
    Dict[str, JaggedTensor] output via construct_jagged_tensors.
    """

    def __init__(
        self,
        overlapped_awaitables: List[Awaitable[torch.Tensor]],
        nonoverlapped_awaitables: List[Awaitable[torch.Tensor]],
        permutations: List[CollisionPermutation],
        features_per_sharding: List[KeyedJaggedTensor],
        embedding_names_per_sharding: List[List[str]],
        need_indices: bool,
        features_to_permute_indices: Dict[str, List[int]],
    ) -> None:
        super().__init__()
        # PEC-specific: partition awaitables and permutation results (per group)
        self._overlapped_awaitables = overlapped_awaitables
        self._nonoverlapped_awaitables = nonoverlapped_awaitables
        self._permutations = permutations

        # From EC: used by construct_jagged_tensors to build output
        self._features_per_sharding = features_per_sharding
        self._embedding_names_per_sharding = embedding_names_per_sharding
        self._need_indices = need_indices

        # CW-only: reorders column shards from rank-grouped to original order.
        # Empty dict for RW sharding (PEC currently only supports RW).
        self._features_to_permute_indices = features_to_permute_indices

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}

        for ol_aw, nol_aw, perm, features, embedding_names in zip(
            self._overlapped_awaitables,
            self._nonoverlapped_awaitables,
            self._permutations,
            self._features_per_sharding,
            self._embedding_names_per_sharding,
        ):
            ol_embs = ol_aw.wait()
            nol_embs = nol_aw.wait()

            merged_embs = torch.cat([ol_embs, nol_embs], dim=0)
            embeddings = torch.index_select(merged_embs, 0, perm.forward_permute)

            jt_dict.update(
                construct_jagged_tensors(
                    embeddings=embeddings,
                    features=features,
                    embedding_names=embedding_names,
                    need_indices=self._need_indices,
                    features_to_permute_indices=self._features_to_permute_indices,
                )
            )

        return jt_dict


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
        overlap_results: List[CollisionResult],
        prev_ctx: PECEmbeddingCollectionContext | None = None,
        prev_features_per_group: List[KeyedJaggedTensor] | None = None,
    ) -> List[Awaitable[CollisionPermutation]]:
        """Distributes forward and backward partition permutations via AllToAll.

        PEC partitions each batch into overlapped (ol) and nonoverlapped (nol)
        values relative to the previous batch. The nol partition's lookup can
        start early in batch i - 1 since those values don't conflict. The ol
        partition must wait for batch i-1's gradient update before lookup.

        Forward mask (batch i): produces forward_permute for merging [ol, nol]
        embeddings back to original order after both lookups complete.

        Backward mask (batch i-1): produces backward_permute for re-splitting
        batch i-1's gradients into ol/nol. The ol gradient update of batch i - 1
        must finish before batch i's ol lookup can proceed. The nol gradient
        update of batch i - 1 can be deferred further into batch i since those
        values don't conflict.

        Both AllToAlls are fired together per handler for batching. The
        backward mask lives on the previous batch's sharding ranks, so its
        AllToAll needs prev_features (for lengths) and prev_ctx (for splits
        and unbucketize_permute). The pipeline saves these across batches.

        Args:
            ctx: PEC context for current batch (sharding contexts as state)
            features_per_group: current batch features after input_dist
            overlap_results: from detect_collisions, contains both forward
                and backward overlap masks
            prev_ctx: previous batch's PEC context, needed for backward mask
                AllToAll (None for first batch)
            prev_features_per_group: previous batch's features after input_dist,
                needed for backward mask AllToAll (None for first batch)

        Returns:
            List of Awaitable[CollisionPermutation], one per sharding group.
        """
        awaitables: List[Awaitable[CollisionPermutation]] = []

        for i, (handler, features, result, sharding_ctx) in enumerate(
            zip(
                self._collision_handlers,
                features_per_group,
                overlap_results,
                ctx.sharding_contexts,
            )
        ):
            bwd_mask = result.backward_overlap_mask
            prev_features = None
            prev_sharding_ctx = None

            if bwd_mask is not None:
                assert prev_ctx is not None
                assert prev_features_per_group is not None
                prev_features = prev_features_per_group[i]
                prev_sharding_ctx = prev_ctx.sharding_contexts[i]

            awaitables.append(
                handler.permute_dist(
                    features,
                    sharding_ctx,
                    result.forward_overlap_mask,
                    prev_features=prev_features,
                    prev_sharding_ctx=prev_sharding_ctx,
                    backward_overlap_mask=bwd_mask,
                )
            )

        return awaitables

    def compute_and_output_dist_in_partition(
        self,
        ctx: PECEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
        collision_splits: CollisionSplits,
        is_overlapped: bool,
    ) -> List[Awaitable[torch.Tensor]]:
        """Performs embedding lookup and output AllToAll for one partition.

        Looks up embeddings for the partition's features using the inner EC's
        lookup module, then AllToAll's the results back using partition-specific
        splits. The embedding AllToAll reverses the direction of the features
        AllToAll, so collision output_splits become embedding input_splits.

        Args:
            ctx: PEC context (sharding_contexts must be set from input_dist)
            features: partition features (ol or nol KJT)
            collision_splits: splits from split_dist
            is_overlapped: True for overlapped partition, False for nonoverlapped

        Returns:
            List of Awaitable[torch.Tensor], one per sharding group.
        """
        partition_idx = 0 if is_overlapped else 1
        ec = self._embedding_collection
        awaitables: List[Awaitable[torch.Tensor]] = []

        for lookup, odist, sharding_ctx, sharding_type in zip(
            ec._lookups,
            ec._output_dists,
            ctx.sharding_contexts,
            ec._sharding_type_to_sharding,
        ):
            partition_ctx = copy(sharding_ctx)

            # Replace full-batch lengths with this partition's lengths.
            partition_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )

            # Reverse direction: collision output → embedding input.
            partition_ctx.input_splits = collision_splits.output_splits[partition_idx]
            partition_ctx.output_splits = collision_splits.input_splits[partition_idx]

            # upt maps positions in the full (pre-split) batch — wrong size for
            # a partition. Reordering is handled by forward_permute after merge.
            partition_ctx.unbucketize_permute_tensor = None

            embedding_dim = ec._embedding_dim_for_sharding_type(sharding_type)
            embs = lookup(features)
            awaitables.append(odist(embs.view(-1, embedding_dim), partition_ctx))

        return awaitables

    def merge_partitioned_embeddings(
        self,
        ctx: PECEmbeddingCollectionContext,
        overlapped_awaitables: List[Awaitable[torch.Tensor]],
        nonoverlapped_awaitables: List[Awaitable[torch.Tensor]],
        permutations: List[CollisionPermutation],
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        """Creates a LazyAwaitable that merges ol/nol embeddings on wait.

        On wait: waits on both partition awaitables, concatenates [ol, nol],
        applies forward_permute to restore original order, and constructs
        the final Dict[str, JaggedTensor].

        Args:
            ctx: PEC context (sharding_contexts must be set from input_dist)
            overlapped_awaitables: from compute_and_output_dist_in_partition(is_overlapped=True)
            nonoverlapped_awaitables: from compute_and_output_dist_in_partition(is_overlapped=False)
            permutations: from permute_dist, one per sharding group

        Returns:
            LazyAwaitable resolving to Dict[str, JaggedTensor].
        """
        ec = self._embedding_collection
        features_per_sharding = [
            # pyre-ignore[6]
            sharding_ctx.features_before_input_dist
            for sharding_ctx in ctx.sharding_contexts
        ]
        return PECEmbeddingCollectionAwaitable(
            overlapped_awaitables=overlapped_awaitables,
            nonoverlapped_awaitables=nonoverlapped_awaitables,
            permutations=permutations,
            features_per_sharding=features_per_sharding,
            embedding_names_per_sharding=ec._embedding_names_per_sharding,
            need_indices=ec._need_indices,
            features_to_permute_indices=ec._features_to_permute_indices,
        )


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
