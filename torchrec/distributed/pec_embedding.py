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
from typing import Any, Dict, List, Tuple, Type

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
    create_overlap_handler,
    MaskDistAwaitable,
    OverlapHandler,
    OverlapMasks,
    OverlapSplits,
    split_kjt_by_values_mask,
)
from torchrec.distributed.pec_comm_ops import PECAll2AllSeqInfo, PECAll2AllSeqWait
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
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


def _mask_dist(
    handler: OverlapHandler,
    features: KeyedJaggedTensor | None,
    sharding_ctx: SequenceShardingContext | None,
    prev_features: KeyedJaggedTensor | None,
    prev_sharding_ctx: SequenceShardingContext | None,
    masks: OverlapMasks,
) -> Tuple[MaskDistAwaitable | None, MaskDistAwaitable | None]:
    """Starts forward and backward mask AllToAlls for one sharding group.

    Kicks off both AllToAlls (non-blocking) before any local work, so
    they overlap with the subsequent _split_kjt computation. Returns
    None for a direction whose mask is None (first/last batch).

    Args:
        handler: sharding-specific overlap handler.
        features: current batch features after input_dist. None for last batch.
        sharding_ctx: current batch sharding context. None for last batch.
        prev_features: previous batch features after input_dist. None for first batch.
        prev_sharding_ctx: previous batch sharding context. None for first batch.
        masks: overlap masks from detect_overlap.

    Returns:
        (forward_mask_awaitable, backward_mask_awaitable). Either may be None.
    """
    fwd_aw = None
    if masks.forward_overlap_mask is not None:
        assert features is not None and sharding_ctx is not None
        fwd_aw = handler.mask_dist(
            features,
            sharding_ctx,
            masks.forward_overlap_mask,
        )

    bwd_aw = None
    if masks.backward_overlap_mask is not None:
        assert prev_features is not None and prev_sharding_ctx is not None
        bwd_aw = handler.mask_dist(
            prev_features,
            prev_sharding_ctx,
            masks.backward_overlap_mask,
        )

    return fwd_aw, bwd_aw


def _split_kjt(
    features: KeyedJaggedTensor | None,
    prev_features: KeyedJaggedTensor | None,
    masks: OverlapMasks,
) -> Tuple[
    KeyedJaggedTensor | None,
    KeyedJaggedTensor | None,
    KeyedJaggedTensor | None,
    KeyedJaggedTensor | None,
]:
    """Splits current and prev features into OL/NOL partitions for one group.

    Args:
        features: current batch features. None for last batch.
        prev_features: previous batch features. None for first batch.
        masks: overlap masks from detect_overlap.

    Returns:
        (ol_kjt, nol_kjt, bwd_ol_kjt, bwd_nol_kjt). Any pair may be
        (None, None) if the corresponding mask is None.
    """
    ol_kjt, nol_kjt = None, None
    if masks.forward_overlap_mask is not None:
        assert features is not None
        ol_kjt, nol_kjt = split_kjt_by_values_mask(
            features,
            masks.forward_overlap_mask,
        )

    bwd_ol_kjt, bwd_nol_kjt = None, None
    if masks.backward_overlap_mask is not None:
        assert prev_features is not None
        bwd_ol_kjt, bwd_nol_kjt = split_kjt_by_values_mask(
            prev_features,
            masks.backward_overlap_mask,
        )

    return ol_kjt, nol_kjt, bwd_ol_kjt, bwd_nol_kjt


@dataclass
class ForwardPartitionContext:
    """Forward partition result for one sharding group, produced by
    OverlapDistAwaitable on wait.

    Attributes:
        splits: per-rank OL/NOL split sizes for the embedding AllToAll.
        permute: permutation tensor that merges [ol, nol] embeddings
            back to the original (pre-split) order.
        ol_features: overlapped partition KJT (values present in prev batch).
        nol_features: nonoverlapped partition KJT (values not in prev batch).
    """

    splits: OverlapSplits
    permute: torch.Tensor
    ol_features: KeyedJaggedTensor
    nol_features: KeyedJaggedTensor


@dataclass
class BackwardPartitionContext:
    """Backward partition result for one sharding group.

    Produced by OverlapDistAwaitable on wait. Contains the previous
    batch's features re-split by overlap with the current batch, for
    gradient re-routing during backward. The pipeline sets these on
    PECAll2AllSeqInfo before calling loss.backward().

    Attributes:
        splits: per-rank OL/NOL split sizes for gradient AllToAll.
        ol_permute: indices into the gradient tensor for OL values,
            pre-composed with recat for rank-major AllToAll alignment.
        nol_permute: indices into the gradient tensor for NOL values,
            pre-composed with recat for rank-major AllToAll alignment.
        ol_features: overlapped partition KJT of the previous batch.
        nol_features: nonoverlapped partition KJT of the previous batch.
            Empty (zero values) when backward mask is all-True (last batch).
    """

    splits: OverlapSplits
    ol_permute: torch.Tensor
    nol_permute: torch.Tensor
    ol_features: KeyedJaggedTensor
    nol_features: KeyedJaggedTensor


OverlapDistOutput = Tuple[
    ForwardPartitionContext | None, BackwardPartitionContext | None
]


class OverlapDistAwaitable(LazyAwaitable[OverlapDistOutput]):
    """Awaitable that resolves mask AllToAlls and produces partition contexts.

    Created by overlap_dist with in-flight mask AllToAlls. On wait,
    completes the AllToAlls and computes splits + permutations from the
    received masks, returning (ForwardPartitionContext, BackwardPartitionContext).
    Either may be None: forward is None for last-batch finalization,
    backward is None for the first batch.
    """

    def __init__(
        self,
        handler: OverlapHandler,
        ol_features: KeyedJaggedTensor | None,
        nol_features: KeyedJaggedTensor | None,
        forward_mask_dist: MaskDistAwaitable | None,
        forward_sharding_ctx: SequenceShardingContext | None,
        backward_mask_dist: MaskDistAwaitable | None,
        backward_sharding_ctx: SequenceShardingContext | None,
        bwd_ol_features: KeyedJaggedTensor | None,
        bwd_nol_features: KeyedJaggedTensor | None,
    ) -> None:
        super().__init__()
        self._handler = handler

        # Forward fields (None for last-batch finalization)
        self._ol_features = ol_features
        self._nol_features = nol_features
        self._forward_mask_dist = forward_mask_dist
        self._forward_sharding_ctx = forward_sharding_ctx

        # Backward fields (None for first batch)
        self._backward_mask_dist = backward_mask_dist
        self._backward_sharding_ctx = backward_sharding_ctx
        self._bwd_ol_features = bwd_ol_features
        self._bwd_nol_features = bwd_nol_features

    def _wait_impl(self) -> OverlapDistOutput:
        forward_ctx = None
        backward_ctx = None

        if self._forward_mask_dist is not None:
            assert self._forward_sharding_ctx is not None
            assert self._ol_features is not None
            assert self._nol_features is not None

            fwd_post_dist_mask, fwd_pre_dist_mask = self._forward_mask_dist.wait()
            forward_ctx = ForwardPartitionContext(
                ol_features=self._ol_features,
                nol_features=self._nol_features,
                splits=self._handler.compute_splits(
                    fwd_post_dist_mask,
                    fwd_pre_dist_mask,
                    self._forward_sharding_ctx,
                ),
                permute=self._handler.compute_forward_permute(
                    fwd_post_dist_mask,
                    self._forward_sharding_ctx,
                ),
            )

        if self._backward_mask_dist is not None:
            assert self._backward_sharding_ctx is not None
            assert self._bwd_ol_features is not None
            assert self._bwd_nol_features is not None

            bwd_post_dist_mask, bwd_pre_dist_mask = self._backward_mask_dist.wait()
            bwd_ol_permute, bwd_nol_permute = self._handler.compute_backward_permutes(
                bwd_post_dist_mask,
                self._backward_sharding_ctx,
            )
            backward_ctx = BackwardPartitionContext(
                splits=self._handler.compute_splits(
                    bwd_post_dist_mask,
                    bwd_pre_dist_mask,
                    self._backward_sharding_ctx,
                ),
                ol_permute=bwd_ol_permute,
                nol_permute=bwd_nol_permute,
                ol_features=self._bwd_ol_features,
                nol_features=self._bwd_nol_features,
            )

        return forward_ctx, backward_ctx


@dataclass
class PECEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Per-batch context for ShardedPECEmbeddingCollection.

    Extends EmbeddingCollectionContext with PEC-specific state. Passed
    to all pipeline stages for a single batch.

    Attributes:
        remapped_kjt_values: Cached remapped values per sharding group,
            set by overlap_dist. Used by the next batch's overlap_dist
            to detect overlap without re-remapping.
        backward_ctx_per_group: PECAll2AllSeqInfo per sharding group,
            set by merge_partitioned_embeddings. The pipeline sets
            backward fields from the next batch's overlap_dist, then
            read by PECAll2AllSeqWait during loss.backward().
    """

    remapped_kjt_values: List[torch.Tensor] | None = None
    backward_ctx_per_group: List[PECAll2AllSeqInfo] | None = None


class PECEmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    """Merges overlapped and nonoverlapped embedding partitions.

    On wait, passes OL/NOL embeddings through PECAll2AllSeqWait (which merges
    via forward_permute and sets up the autograd graph for gradient re-split),
    then constructs Dict[str, JaggedTensor] output via construct_jagged_tensors.
    """

    def __init__(
        self,
        overlapped_awaitables: List[Awaitable[torch.Tensor]],
        nonoverlapped_awaitables: List[Awaitable[torch.Tensor]],
        backward_ctxs: List[PECAll2AllSeqInfo],
        features_per_sharding: List[KeyedJaggedTensor],
        embedding_names_per_sharding: List[List[str]],
        need_indices: bool,
        features_to_permute_indices: Dict[str, List[int]],
    ) -> None:
        super().__init__()
        # PEC-specific: partition awaitables and gradient intercept (per group)
        self._overlapped_awaitables = overlapped_awaitables
        self._nonoverlapped_awaitables = nonoverlapped_awaitables
        self._backward_ctxs = backward_ctxs

        # From EC: used by construct_jagged_tensors to build output
        self._features_per_sharding = features_per_sharding
        self._embedding_names_per_sharding = embedding_names_per_sharding
        self._need_indices = need_indices

        # CW-only: reorders column shards from rank-grouped to original order.
        # Empty dict for RW sharding (PEC currently only supports RW).
        self._features_to_permute_indices = features_to_permute_indices

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}

        for ol_aw, nol_aw, bwd_ctx, features, embedding_names in zip(
            self._overlapped_awaitables,
            self._nonoverlapped_awaitables,
            self._backward_ctxs,
            self._features_per_sharding,
            self._embedding_names_per_sharding,
        ):
            ol_embs = ol_aw.wait()
            nol_embs = nol_aw.wait()

            embeddings = PECAll2AllSeqWait.apply(bwd_ctx, ol_embs, nol_embs)

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
        self._overlap_handlers: List[OverlapHandler] = []
        for (
            sharding_type,
            sharding,
        ) in self._embedding_collection._sharding_type_to_sharding.items():
            self._overlap_handlers.append(
                create_overlap_handler(
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

    def overlap_dist(
        self,
        ctx: PECEmbeddingCollectionContext | None = None,
        dist_input: KJTList | None = None,
        prev_ctx: PECEmbeddingCollectionContext | None = None,
        prev_dist_input: KJTList | None = None,
    ) -> List[LazyAwaitable[OverlapDistOutput]]:
        """Detects and distributes overlap between batches.

        For each sharding group: remaps current values, detects overlap
        with the previous batch, starts mask AllToAlls (non-blocking),
        and splits features into OL/NOL partitions while AllToAlls are
        in flight. Returns LazyAwaitables that compute splits and
        permutations from the received masks on wait.

        Handles three batch positions based on which args are None:
        - First batch (ctx set, prev_ctx None): forward only, no backward
        - Normal batch (both set): forward + backward
        - Last batch finalization (ctx None, prev_ctx set): backward only

        The pipeline calls this during prefetch, then unpacks
        ForwardPartitionContext for compute_and_output_dist_in_partition
        and BackwardPartitionContext for gradient re-routing.

        Args:
            ctx: PEC context for current batch. None for last-batch
                finalization.
            dist_input: KJTList from input_dist (one KJT per sharding
                group). None for last-batch finalization.
            prev_ctx: Previous batch's PEC context (has remapped_kjt_values
                and sharding_contexts). None for first batch.
            prev_dist_input: Previous batch's features after input_dist.
                None for first batch.

        Returns:
            List of LazyAwaitable[OverlapDistOutput], one per sharding
            group. Each resolves to (ForwardPartitionContext | None,
            BackwardPartitionContext | None).
        """
        assert ctx is not None or prev_ctx is not None

        remapped_values = []
        awaitables: List[LazyAwaitable[OverlapDistOutput]] = []

        for i, handler in enumerate(self._overlap_handlers):
            current_remapped, sharding_ctx, features = None, None, None
            if ctx is not None:
                assert dist_input is not None
                current_remapped = handler.remap_kjt_values(dist_input[i])
                sharding_ctx = ctx.sharding_contexts[i]
                features = dist_input[i]
                remapped_values.append(current_remapped)

            prev_remapped, prev_sharding_ctx, prev_features = None, None, None
            if prev_ctx is not None:
                assert prev_ctx.remapped_kjt_values is not None
                assert prev_dist_input is not None
                prev_remapped = prev_ctx.remapped_kjt_values[i]
                prev_sharding_ctx = prev_ctx.sharding_contexts[i]
                prev_features = prev_dist_input[i]

            masks = handler.detect_overlap(current_remapped, prev_remapped)

            fwd_aw, bwd_aw = _mask_dist(
                handler=handler,
                features=features,
                sharding_ctx=sharding_ctx,
                prev_features=prev_features,
                prev_sharding_ctx=prev_sharding_ctx,
                masks=masks,
            )

            ol_kjt, nol_kjt, bwd_ol_kjt, bwd_nol_kjt = _split_kjt(
                features=features,
                prev_features=prev_features,
                masks=masks,
            )

            awaitables.append(
                OverlapDistAwaitable(
                    handler=handler,
                    ol_features=ol_kjt,
                    nol_features=nol_kjt,
                    forward_mask_dist=fwd_aw,
                    forward_sharding_ctx=sharding_ctx,
                    backward_mask_dist=bwd_aw,
                    backward_sharding_ctx=prev_sharding_ctx,
                    bwd_ol_features=bwd_ol_kjt,
                    bwd_nol_features=bwd_nol_kjt,
                )
            )

        if ctx is not None:
            ctx.remapped_kjt_values = remapped_values

        return awaitables

    def compute_and_output_dist_in_partition(
        self,
        ctx: PECEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
        overlap_splits: OverlapSplits,
        is_overlapped: bool,
    ) -> List[Awaitable[torch.Tensor]]:
        """Performs embedding lookup and output AllToAll for one partition.

        Looks up embeddings for the partition's features using the inner
        EC's lookup module, then AllToAll's the results back using
        partition-specific splits from ForwardPartitionContext.splits.

        Called twice per batch: once for OL (is_overlapped=True) and
        once for NOL (is_overlapped=False). The partition may be empty
        (e.g., first batch OL when no overlap exists) — empty lookups
        and zero-length AllToAll segments are handled correctly.

        Args:
            ctx: PEC context (sharding_contexts must be set from
                input_dist).
            features: partition features (OL or NOL KJT from
                ForwardPartitionContext).
            overlap_splits: per-rank OL/NOL split sizes from
                ForwardPartitionContext.splits.
            is_overlapped: True for overlapped partition (index 0 in
                splits), False for nonoverlapped (index 1).

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

            # Map overlap splits to embedding AllToAll context
            partition_ctx.input_splits = overlap_splits.input_splits[partition_idx]
            partition_ctx.output_splits = overlap_splits.output_splits[partition_idx]

            # In RW sharding, upt maps positions in the full (pre-split) batch —
            # wrong size for a partition. Reordering is handled by RW handler
            # if needed.
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
        forward_ctxs: List[ForwardPartitionContext],
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        """Creates a LazyAwaitable that merges OL/NOL embeddings on wait.

        On wait: passes OL/NOL embedding awaitables through
        PECAll2AllSeqWait (which merges via ForwardPartitionContext.permute
        and sets up the autograd graph for gradient re-split in backward),
        then constructs the final Dict[str, JaggedTensor] output.

        Also creates one PECAll2AllSeqInfo per sharding group (with only
        forward_permute set) and stores them on PEC context. The
        pipeline is responsible for setting backward fields (splits,
        permutes) from the next batch's overlap_dist result before
        calling loss.backward().

        Args:
            ctx: PEC context (sharding_contexts must be set from
                input_dist).
            overlapped_awaitables: from
                compute_and_output_dist_in_partition(is_overlapped=True).
            nonoverlapped_awaitables: from
                compute_and_output_dist_in_partition(is_overlapped=False).
            forward_ctxs: ForwardPartitionContexts from overlap_dist,
                one per sharding group.

        Returns:
            LazyAwaitable resolving to Dict[str, JaggedTensor].
        """
        ec = self._embedding_collection

        backward_ctxs = [
            PECAll2AllSeqInfo(
                forward_permute=fc.permute,
                pg=self._env.process_group,
            )
            for fc in forward_ctxs
        ]
        ctx.backward_ctx_per_group = backward_ctxs

        features_per_sharding = [
            sharding_ctx.features_before_input_dist
            for sharding_ctx in ctx.sharding_contexts
        ]
        return PECEmbeddingCollectionAwaitable(
            overlapped_awaitables=overlapped_awaitables,
            nonoverlapped_awaitables=nonoverlapped_awaitables,
            backward_ctxs=backward_ctxs,
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
