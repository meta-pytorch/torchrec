#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import torch
import torch.nn as nn
from torchrec.distributed.dist_data import SplitsAllToAllAwaitable
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


@dataclass
class CollisionSplits:
    """Result of waiting on CollisionSplitsAwaitable.

    input_splits: [ol_per_rank, nol_per_rank] — how many values this rank
        sends to each other rank per partition.
    output_splits: [ol_received, nol_received] — how many values this rank
        receives from each other rank per partition.
    """

    input_splits: List[List[int]]
    output_splits: List[List[int]]


class CollisionSplitsAwaitable(Awaitable[List[CollisionSplits]]):
    """Awaitable for PEC collision split sizes, one per sharding group.

    Wraps a SplitsAllToAllAwaitable that exchanges nonoverlapped per-rank
    counts. On wait(), derives overlapped received splits from
    total - nol_received, then assembles per-group CollisionSplits.

    Args:
        ol_input_splits: per-group overlapped send counts, each [world_size].
        nol_input_splits: per-group nonoverlapped send counts, each [world_size].
        splits_awaitable: SplitsAllToAll that exchanges nol counts across ranks.
        total_input_splits: per-group total receive counts from input_dist,
            each [world_size]. Used to derive ol_received = total - nol_received.
    """

    def __init__(
        self,
        ol_input_splits: List[List[int]],
        nol_input_splits: List[List[int]],
        splits_awaitable: SplitsAllToAllAwaitable,
        total_input_splits: List[List[int]],
    ) -> None:
        super().__init__()
        self._ol_input_splits = ol_input_splits
        self._nol_input_splits = nol_input_splits
        self._splits_awaitable = splits_awaitable
        self._total_input_splits = total_input_splits

    def _wait_impl(self) -> List[CollisionSplits]:
        result = self._splits_awaitable.wait()
        splits: List[CollisionSplits] = []
        for nol_received, ol_input, nol_input, total in zip(
            result,
            self._ol_input_splits,
            self._nol_input_splits,
            self._total_input_splits,
        ):
            ol_received = [total[r] - nol_received[r] for r in range(len(nol_received))]
            splits.append(
                CollisionSplits(
                    input_splits=[ol_input, nol_input],
                    output_splits=[ol_received, nol_received],
                )
            )
        return splits


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
        """Detect collisions for each sharding group's features.

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
    ) -> CollisionSplitsAwaitable:
        """Starts AllToAll to exchange per-rank nonoverlapped/overlapped split
        sizes.

        For each sharding group, computes how many nonoverlapped values this
        rank sends to each other rank (nol_per_rank), derives the overlapped
        counts (ol_per_rank = total - nol), and bundles all groups into a
        single SplitsAllToAll. On wait, the received nol counts are used to
        derive the received ol counts.

        Args:
            ctx: PEC context (sharding_contexts must be set from input_dist)
            nonoverlapped_features: one nonoverlapped KJT per
                sharding group, from split_features_by_values_mask()

        Returns:
            CollisionSplitsAwaitable that resolves to CollisionSplits.
            # TODO: test with multiple sharding groups
        """
        # Per-group value counts for the overlapped partition. Derived from
        # output splits form input dist, and will be used in output dist.
        ol_input_splits: List[List[int]] = []

        # Per-group value counts for non-overlapped partition. Derived from
        # output splits from input dist, and will be used in output dist.
        nol_input_splits: List[List[int]] = []

        # Per-group split tensors for the nonoverlapped partition. Exchanged
        # via SplitsAllToAll.
        nol_splits: List[torch.Tensor] = []

        # Per-group total receive counts from input_dist, used by the
        # awaitable to derive ol_received = total - nol_received.
        total_input_splits: List[List[int]] = []

        for handler, nol_features, sharding_ctx in zip(
            self._collision_handlers,
            nonoverlapped_features,
            ctx.sharding_contexts,
        ):
            assert sharding_ctx.batch_size_per_rank is not None
            nol = handler.compute_nonoverlapped_per_rank(
                nol_features, sharding_ctx.batch_size_per_rank
            )
            ol = (
                torch.tensor(
                    sharding_ctx.output_splits,
                    device=nol.device,
                    dtype=nol.dtype,
                )
                - nol
            )

            nol_splits.append(nol)

            ol_input_splits.append(ol.tolist())
            nol_input_splits.append(nol.tolist())
            total_input_splits.append(sharding_ctx.input_splits)

        assert self._env.process_group is not None
        splits_awaitable = SplitsAllToAllAwaitable(nol_splits, self._env.process_group)

        return CollisionSplitsAwaitable(
            ol_input_splits=ol_input_splits,
            nol_input_splits=nol_input_splits,
            splits_awaitable=splits_awaitable,
            total_input_splits=total_input_splits,
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
