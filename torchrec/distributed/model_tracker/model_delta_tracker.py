#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging as logger
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import torch

from torch import nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.model_tracker.delta_store import DeltaStore
from torchrec.distributed.model_tracker.types import (
    DeltaRows,
    EmbdUpdateMode,
    TrackingMode,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

UPDATE_MODE_MAP: Dict[TrackingMode, EmbdUpdateMode] = {
    # Only IDs are tracked, no additional state is stored.
    TrackingMode.ID_ONLY: EmbdUpdateMode.NONE,
    # TrackingMode.EMBEDDING utilizes EmbdUpdateMode.FIRST to ensure that
    # the earliest embedding values are stored since the last checkpoint or snapshot.
    # This mode is used for computing topk delta rows, which is currently achieved by running (new_emb - old_emb).norm().topk().
    TrackingMode.EMBEDDING: EmbdUpdateMode.FIRST,
}

# Tracking is current only supported for ShardedEmbeddingCollection and ShardedEmbeddingBagCollection.
SUPPORTED_MODULES_TO_PREFIX = {
    ShardedEmbeddingCollection: ".embeddings",
    ShardedEmbeddingBagCollection: ".embedding_bags",
}


class ModelDeltaTracker:
    r"""

    ModelDeltaTracker provides a way to track and retrieve unique IDs for supported modules, along with optional support
    for tracking corresponding embeddings or states. This is useful for identifying and retrieving the latest delta or
    unique rows for a given model, which can help compute topk or to stream updated embeddings from predictors to trainers during
    online training. Unique IDs or states can be retrieved by calling the get_delta() method.

    Args:
        model (nn.Module): the model to track.
        consumers (List[str], optional): list of consumers to track. Each consumer will
            have its own batch offset index. Every get_delta and get_delta_ids invocation will
            only return the new values for the given consumer since last call.
        delete_on_read (bool, optional): whether to delete the tracked ids after all consumers have read them.
        auto_compact (bool, optional): Trigger compaction automatically during communication at each train cycle.
        When set false, compaction is triggered at get_delta() call. Default: False.
        mode (TrackingMode, optional): tracking mode to use from supported tracking modes. Default: TrackingMode.ID_ONLY.
        fqns_to_skip (Iterable[str], optional): list of FQNs to skip tracking. Default: None.

    """

    DEFAULT_CONSUMER: str = "default"

    def __init__(
        self,
        model: nn.Module,
        consumers: Optional[List[str]] = None,
        delete_on_read: bool = True,
        auto_compact: bool = False,
        mode: TrackingMode = TrackingMode.ID_ONLY,
        fqns_to_skip: Iterable[str] = (),
    ) -> None:
        self._model = model
        self._consumers: List[str] = consumers or [self.DEFAULT_CONSUMER]
        self._delete_on_read = delete_on_read
        self._auto_compact = auto_compact
        self._mode = mode
        self._fqn_to_feature_map: Dict[str, List[str]] = {}
        self._fqns_to_skip: Iterable[str] = fqns_to_skip

        # per_consumer_batch_idx is used to track the batch index for each consumer.
        # This is used to retrieve the delta values for a given consumer as well as
        # start_ids for compaction window.
        self.per_consumer_batch_idx: Dict[str, int] = {
            c: -1 for c in (consumers or [self.DEFAULT_CONSUMER])
        }
        self.curr_batch_idx: int = 0
        self.curr_compact_index: int = 0

        # from module FQN to ShardedEmbeddingCollection/ShardedEmbeddingBagCollection
        self.tracked_modules: Dict[str, nn.Module] = {}
        self.feature_to_fqn: Dict[str, str] = {}
        # Generate the mapping from FQN to feature names.
        self.fqn_to_feature_names()
        # Validate is the mode is supported for the given module and initialize tracker functions
        self._validate_and_init_tracker_fns()

        self.store: DeltaStore = DeltaStore(UPDATE_MODE_MAP[self._mode])

        # Mapping feature name to corresponding FQNs. This is used for retrieving
        # the FQN associated with a given feature name in record_lookup().
        for fqn, feature_names in self._fqn_to_feature_map.items():
            for feature_name in feature_names:
                if feature_name in self.feature_to_fqn:
                    logger.warning(
                        f"Duplicate feature name: {feature_name} in fqn {fqn}"
                    )
                    continue
                self.feature_to_fqn[feature_name] = fqn
        logger.info(f"feature_to_fqn: {self.feature_to_fqn}")

    def step(self) -> None:
        # Move batch index forward for all consumers.
        self.curr_batch_idx += 1

    def trigger_compaction(self) -> None:
        if self.curr_compact_index >= self.curr_batch_idx:
            # only trigger compaction once per iteration
            return

        # TODO: May need to revisit the compaction logic with multiple consmers.
        # At present we take the max per_consumer_batch_idx to ensure we only compact
        # newely received lookups

        # The trigger_compaction() function is expected to overlap with comms to hide
        # compaction compute overhead. Currently, we overlap compaction with odist
        # because ID tracking occurs during local embedding lookup, which takes place
        # before odist. This way, auto_compact always merges all past IDs tensors since
        # the last get_delta call into a single IDs tensor per FQN.
        #
        # For delete_on_read=True, get_delta() should delete up to per_consumer_batch_idx
        # (exclusive). So the compaction should start from per_consumer_batch_idx.
        #
        # For delete_on_read=False, get_delta() won't delete tensors, but it does advance
        # per_consumer_batch_idx accordingly, where all ids prior to per_consumer_batch_idx (exclusive)
        # should have been compacted into one tensor regardless of auto_compact=True/False.
        # Therefore, all future compactions should start from per_consumer_batch_idx.
        start_idx = max(self.per_consumer_batch_idx.values())
        end_idx = self.curr_batch_idx
        if start_idx < end_idx:
            self.compact(start_idx=start_idx, end_idx=end_idx)

            # Update the current compact index to the end index to avoid duplicate compaction.
            self.curr_compact_index = end_idx

    def record_lookup(self, kjt: KeyedJaggedTensor, states: torch.Tensor) -> None:
        """
        Records the IDs from a given KeyedJaggedTensor and their corresponding embeddings/parameter states.

        This method is run post-lookup, after the embedding lookup has been performed,
        as it needs access to both the input IDs and the resulting embeddings.

        This function processes the input KeyedJaggedTensor and records either just the IDs
        (in ID_ONLY mode) or both IDs and their corresponding embeddings (in EMBEDDING mode).

        Args:
            kjt (KeyedJaggedTensor): The KeyedJaggedTensor containing IDs to record.
            states (torch.Tensor): The embeddings or states corresponding to the IDs in the kjt.
        """

        # In ID_ONLY mode, we only track feature IDs received in the current batch.
        if self._mode == TrackingMode.ID_ONLY:
            self.record_ids(kjt)
        # In EMBEDDING mode, we track per feature IDs and corresponding embeddings received in the current batch.
        elif self._mode == TrackingMode.EMBEDDING:
            self.record_embeddings(kjt, states)

        else:
            raise NotImplementedError(f"Tracking mode {self._mode} is not supported")

    def record_ids(self, kjt: KeyedJaggedTensor) -> None:
        """
        Record Ids from a given KeyedJaggedTensor.

        Args:
            kjt (KeyedJaggedTensor): the KeyedJaggedTensor to record.
        """
        per_table_ids: Dict[str, List[torch.Tensor]] = {}
        for key in kjt.keys():
            table_fqn = self.feature_to_fqn[key]
            ids_list: List[torch.Tensor] = per_table_ids.get(table_fqn, [])
            ids_list.append(kjt[key].values())
            per_table_ids[table_fqn] = ids_list

        for table_fqn, ids_list in per_table_ids.items():
            self.store.append(
                batch_idx=self.curr_batch_idx,
                table_fqn=table_fqn,
                ids=torch.cat(ids_list),
                states=None,
            )

    def record_embeddings(
        self, kjt: KeyedJaggedTensor, embeddings: torch.Tensor
    ) -> None:
        """
        Record Ids along with Embeddings from a given KeyedJaggedTensor and embeddings.

        Args:
            kjt (KeyedJaggedTensor): the KeyedJaggedTensor to record.
            embeddings (torch.Tensor): the embeddings to record.
        """
        per_table_ids: Dict[str, List[torch.Tensor]] = {}
        per_table_emb: Dict[str, List[torch.Tensor]] = {}
        assert embeddings.numel() % kjt.values().numel() == 0, (
            f"ids and embeddings size mismatch, expect [{kjt.values().numel()} * emb_dim], "
            f"but got {embeddings.numel()}"
        )
        embeddings_2d = embeddings.view(kjt.values().numel(), -1)

        offset: int = 0
        for key in kjt.keys():
            table_fqn = self.feature_to_fqn[key]
            ids_list: List[torch.Tensor] = per_table_ids.get(table_fqn, [])
            emb_list: List[torch.Tensor] = per_table_emb.get(table_fqn, [])

            ids = kjt[key].values()
            ids_list.append(ids)
            emb_list.append(embeddings_2d[offset : offset + ids.numel()])
            offset += ids.numel()

            per_table_ids[table_fqn] = ids_list
            per_table_emb[table_fqn] = emb_list

        for table_fqn, ids_list in per_table_ids.items():
            self.store.append(
                batch_idx=self.curr_batch_idx,
                table_fqn=table_fqn,
                ids=torch.cat(ids_list),
                states=torch.cat(per_table_emb[table_fqn]),
            )

    def get_delta_ids(self, consumer: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Return a dictionary of hit local IDs for each sparse feature. Ids are
        first keyed by submodule FQN.

        Args:
            consumer (str, optional): The consumer to retrieve unique IDs for. If not specified, "default" is used as the default consumer.
        """
        per_table_delta_rows = self.get_delta(consumer)
        return {fqn: delta_rows.ids for fqn, delta_rows in per_table_delta_rows.items()}

    def get_delta(self, consumer: Optional[str] = None) -> Dict[str, DeltaRows]:
        """
        Return a dictionary of hit local IDs and parameter states / embeddings for each sparse feature. The Values are first keyed by submodule FQN.

        Args:
            consumer (str, optional): The consumer to retrieve delta values for. If not specified, "default" is used as the default consumer.
        """
        consumer = consumer or self.DEFAULT_CONSUMER
        assert (
            consumer in self.per_consumer_batch_idx
        ), f"consumer {consumer} not present in {self.per_consumer_batch_idx.values()}"

        index_end: int = self.curr_batch_idx + 1
        index_start = max(self.per_consumer_batch_idx.values())

        # In case of multiple consumers, it is possible that the previous consumer has already compact these indices
        # and index_start could be equal to index_end, in which case we should not compact again.
        if index_start < index_end:
            self.compact(index_start, index_end)
        tracker_rows = self.store.get_delta(
            from_idx=self.per_consumer_batch_idx[consumer]
        )
        self.per_consumer_batch_idx[consumer] = index_end
        if self._delete_on_read:
            self.store.delete(up_to_idx=min(self.per_consumer_batch_idx.values()))
        return tracker_rows

    def get_tracked_modules(self) -> Dict[str, nn.Module]:
        """
        Returns a dictionary of tracked modules.
        """
        return self.tracked_modules

    def fqn_to_feature_names(self) -> Dict[str, List[str]]:
        """
        Returns a mapping of FQN to feature names from all Supported Modules [EmbeddingCollection and EmbeddingBagCollection] present in the given model.
        """
        if (self._fqn_to_feature_map is not None) and len(self._fqn_to_feature_map) > 0:
            return self._fqn_to_feature_map

        fqn_to_feature_names: Dict[str, List[str]] = OrderedDict()
        for fqn, named_module in self._model.named_modules():
            # Skipping partial FQNs present in fqns_to_skip
            # TODO: Validate if we need to support more complex patterns for skipping fqns
            if type(named_module) in SUPPORTED_MODULES_TO_PREFIX:
                for table_name, config in named_module._table_name_to_config.items():
                    embedding_fqn = (
                        self._clean_fqn_fn(fqn)
                        + SUPPORTED_MODULES_TO_PREFIX[type(named_module)]
                        + f".{table_name}"
                    )

                    should_skip = False
                    for fqn_to_skip in self._fqns_to_skip:
                        if fqn_to_skip in embedding_fqn:
                            logger.info(
                                f"Skipping {fqn} because it is part of fqns_to_skip"
                            )
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    if embedding_fqn not in fqn_to_feature_names:
                        fqn_to_feature_names[embedding_fqn] = []
                    fqn_to_feature_names[embedding_fqn].extend(config.feature_names)
                    self.tracked_modules[embedding_fqn] = named_module
        self._fqn_to_feature_map = fqn_to_feature_names
        return fqn_to_feature_names

    def clear(self, consumer: Optional[str] = None) -> None:
        """
        Clear tracked IDs for a given consumer.

        Args:
            consumer (str, optional): The consumer to clear IDs/States for. If not specified, "default" is used as the default consumer.
        """
        # 1. If consumer is None, delete globally.
        if consumer is None:
            self.store.delete()
            return

        assert (
            consumer in self.per_consumer_batch_idx
        ), f"consumer {consumer} not found in {self.per_consumer_batch_idx.values()}"

        # 2. For single consumer, we can just delete all ids
        if len(self.per_consumer_batch_idx) == 1:
            self.store.delete()
            return

    def compact(self, start_idx: int, end_idx: int) -> None:
        """
        Compact tracked IDs for a given range of indices.

        Args:
            start_idx (int): Starting index for compaction.
            end_idx (int): Ending index for compaction.
        """
        self.store.compact(start_idx, end_idx)

    def _clean_fqn_fn(self, fqn: str) -> str:
        # strip FQN prefixes added by DMP and other TorchRec operations to match state dict FQN
        # handles both "_dmp_wrapped_module.module." and "module." prefixes
        prefixes_to_strip = ["_dmp_wrapped_module.module.", "module."]
        for prefix in prefixes_to_strip:
            if fqn.startswith(prefix):
                return fqn[len(prefix) :]
        return fqn

    def _validate_and_init_tracker_fns(self) -> None:
        "To validate the mode is supported for the given module"
        for module in self.tracked_modules.values():
            assert not (
                isinstance(module, ShardedEmbeddingBagCollection)
                and self._mode == TrackingMode.EMBEDDING
            ), "EBC's lookup returns pooled embeddings and currently, we do not support tracking raw embeddings."
            # register post lookup function
            # pyre-ignore[29]
            module.register_post_lookup_tracker_fn(self.record_lookup)
            # register auto compaction function at odist
            if self._auto_compact:
                # pyre-ignore[29]
                module.register_post_odist_tracker_fn(self.trigger_compaction)
