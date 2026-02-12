#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import logging
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torchrec.distributed.embedding_types import (
    KeyedJaggedTensor,
    ShardedEmbeddingTable,
)
from torchrec.distributed.mc_embedding import ShardedManagedCollisionEmbeddingCollection
from torchrec.distributed.mc_embeddingbag import (
    ShardedManagedCollisionEmbeddingBagCollection,
)
from torchrec.distributed.mc_modules import ShardedManagedCollisionCollection
from torchrec.distributed.model_tracker.delta_store import RawIdTrackerStore
from torchrec.distributed.model_tracker.model_delta_tracker import ModelDeltaTracker
from torchrec.distributed.model_tracker.types import UniqueRows

logger: logging.Logger = logging.getLogger(__name__)

SUPPORTED_TRACKING_MODULES = (ShardedManagedCollisionCollection,)

MANAGED_COLLISION_WRAPPER_MODULES = (
    ShardedManagedCollisionEmbeddingCollection,
    ShardedManagedCollisionEmbeddingBagCollection,
)


class RawIdTracker(ModelDeltaTracker):
    def __init__(
        self,
        model: nn.Module,
        delete_on_read: bool = True,
        fqns_to_skip: Iterable[str] = (),
    ) -> None:
        self._model = model
        self._consumers: Optional[List[str]] = None
        self._delete_on_read = delete_on_read
        self._fqn_to_feature_map: Dict[str, List[str]] = {}
        self._fqns_to_skip: Iterable[str] = fqns_to_skip

        self.curr_batch_idx: int = 0
        self.curr_compact_index: int = 0

        # from module FQN to SUPPORTED_TRACKING_MODULES
        self.tracked_modules: Dict[str, nn.Module] = {}
        self.table_to_fqn: Dict[str, str] = {}
        self.feature_to_fqn: Dict[str, str] = {}
        # Generate the mapping from FQN to feature names.
        self.fqn_to_feature_names()
        # Validate is the mode is supported for the given module and initialize tracker functions
        self._validate_and_init_tracker_fns()
        # init TBE tracker wrapper and register consumer ids
        self._init_tbe_tracker_wrapper(self._model)

        # per_consumer_batch_idx is used to track the batch index for each consumer.
        # This is used to retrieve the delta values for a given consumer as well as
        # start_ids for compaction window.

        # Note: For raw id tracking, this has to be assigned after the _init_tbe_tracker_wrapper()
        # call as _init_tbe_tracker_wrapper is setting up consumers for TBEs

        self.per_consumer_batch_idx: Dict[str, int] = {
            c: -1 for c in (self._consumers or [self.DEFAULT_CONSUMER])
        }

        self.store: RawIdTrackerStore = RawIdTrackerStore()

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

    def _should_skip_fqn(self, fqn: str) -> bool:
        split_fqn = fqn.split(".")
        # Skipping partial FQNs present in fqns_to_skip
        # TODO: Validate if we need to support more complex patterns for skipping fqns
        should_skip = False
        for fqn_to_skip in self._fqns_to_skip:
            if fqn_to_skip in split_fqn:
                logger.info(f"Skipping {fqn} because it is part of fqns_to_skip")
                should_skip = True
                break
        return should_skip

    def _should_track_table(
        self, embedding_tables: List[ShardedEmbeddingTable]
    ) -> bool:
        should_track = True
        for table_config in embedding_tables:
            for fqn_to_skip in self._fqns_to_skip:
                if fqn_to_skip in table_config.name:
                    should_track = False
                    break
        return should_track

    def fqn_to_feature_names(self) -> Dict[str, List[str]]:
        """
        Returns a mapping of FQN to feature names from all Supported Modules [EmbeddingCollection and EmbeddingBagCollection] present in the given model.
        """
        if (self._fqn_to_feature_map is not None) and len(self._fqn_to_feature_map) > 0:
            return self._fqn_to_feature_map

        table_to_feature_names: Dict[str, List[str]] = OrderedDict()
        for fqn, named_module in self._model.named_modules():
            if self._should_skip_fqn(fqn):
                continue
            # Using FQNs of the embedding and mapping them to features as state_dict() API uses these to key states.
            if isinstance(named_module, SUPPORTED_TRACKING_MODULES):
                should_track_module = True
                for table_name, config in named_module._table_name_to_config.items():
                    for fqn_to_skip in self._fqns_to_skip:
                        if fqn_to_skip in table_name:
                            should_track_module = False
                    logger.info(
                        f"Found {table_name} for {fqn} with features {config.feature_names} should_track_module: {should_track_module}"
                    )
                    table_to_feature_names[table_name] = config.feature_names
                if should_track_module:
                    self.tracked_modules[self._clean_fqn_fn(fqn)] = named_module
            for table_name in table_to_feature_names:
                # Using the split FQN to get the exact table name matching. Otherwise, checking "table_name in fqn"
                # will incorrectly match fqn with all the table names that have the same prefix
                split_fqn = fqn.split(".")
                if table_name in split_fqn:
                    embedding_fqn = self._clean_fqn_fn(fqn)
                    if table_name in self.table_to_fqn:
                        # Sanity check for validating that we don't have more then one table mapping to same fqn.
                        logger.warning(
                            f"Override {self.table_to_fqn[table_name]} with {embedding_fqn} for entry {table_name}"
                        )
                    self.table_to_fqn[table_name] = embedding_fqn
            logger.info(f"Table to fqn: {self.table_to_fqn}")
        flatten_names = [
            name for names in table_to_feature_names.values() for name in names
        ]
        # TODO: Validate if there is a better way to handle duplicate feature names.
        # Logging a warning if duplicate feature names are found across tables, but continue execution as this could be a valid case.
        if len(set(flatten_names)) != len(flatten_names):
            counts = Counter(flatten_names)
            duplicates = [item for item, count in counts.items() if count > 1]
            logger.warning(f"duplicate feature names found: {duplicates}")

        fqn_to_feature_names: Dict[str, List[str]] = OrderedDict()
        for table_name in table_to_feature_names:
            if table_name not in self.table_to_fqn:
                # This is likely unexpected, where we can't locate the FQN associated with this table.
                logger.warning(
                    f"Table {table_name} not found in {self.table_to_fqn}, skipping"
                )
                continue
            fqn_to_feature_names[self.table_to_fqn[table_name]] = (
                table_to_feature_names[table_name]
            )
        self._fqn_to_feature_map = fqn_to_feature_names
        return fqn_to_feature_names

    def record_lookup(
        self,
        kjt: KeyedJaggedTensor,
        states: torch.Tensor,
        emb_module: Optional[nn.Module] = None,
        raw_ids: Optional[torch.Tensor] = None,
        runtime_meta: Optional[torch.Tensor] = None,
    ) -> None:
        per_table_ids: Dict[str, List[torch.Tensor]] = {}
        per_table_raw_ids: Dict[str, List[torch.Tensor]] = {}
        per_table_runtime_meta: Dict[str, List[torch.Tensor]] = {}

        # Skip storing invalid input or raw ids, note that runtime_meta will only exist if raw_ids exists so we can return early
        if raw_ids is None:
            logger.debug("Skipping record_lookup: raw_ids is None")
            return

        if kjt.values().numel() == 0:
            logger.debug("Skipping record_lookup: kjt.values() is empty")
            return

        if not (raw_ids.numel() % kjt.values().numel() == 0):
            logger.warning(
                f"Skipping record_lookup. Raw_ids has invalid shape {raw_ids.shape}, expected multiple of {kjt.values().numel()}"
            )
            return

        # Skip storing if runtime_meta is provided but has invalid shape
        if runtime_meta is not None and not (
            runtime_meta.numel() % kjt.values().numel() == 0
        ):
            logger.warning(
                f"Skipping record_lookup. Runtime_meta has invalid shape {runtime_meta.shape}, expected multiple of {kjt.values().numel()}"
            )
            return

        raw_ids_2d = raw_ids.view(kjt.values().numel(), -1)
        runtime_meta_2d = None
        # It is possible that runtime_meta is None while raw_ids is not None so we will proceed
        if runtime_meta is not None:
            runtime_meta_2d = runtime_meta.view(kjt.values().numel(), -1)

        offset: int = 0
        for key in kjt.keys():
            table_fqn = self.table_to_fqn[key]
            ids_list: List[torch.Tensor] = per_table_ids.get(table_fqn, [])
            raw_ids_list: List[torch.Tensor] = per_table_raw_ids.get(table_fqn, [])
            runtime_meta_list: List[torch.Tensor] = per_table_runtime_meta.get(
                table_fqn, []
            )

            ids = kjt[key].values()
            ids_list.append(ids)
            raw_ids_list.append(raw_ids_2d[offset : offset + ids.numel()])
            if runtime_meta_2d is not None:
                runtime_meta_list.append(runtime_meta_2d[offset : offset + ids.numel()])
            offset += ids.numel()

            per_table_ids[table_fqn] = ids_list
            per_table_raw_ids[table_fqn] = raw_ids_list
            if runtime_meta_2d is not None:
                per_table_runtime_meta[table_fqn] = runtime_meta_list

        for table_fqn, ids_list in per_table_ids.items():
            self.store.append(
                batch_idx=self.curr_batch_idx,
                fqn=table_fqn,
                ids=torch.cat(ids_list),
                raw_ids=torch.cat(per_table_raw_ids[table_fqn]),
                runtime_meta=(
                    torch.cat(per_table_runtime_meta[table_fqn])
                    if table_fqn in per_table_runtime_meta
                    else None
                ),
            )

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
            if isinstance(module, SUPPORTED_TRACKING_MODULES):
                # register post lookup function
                module.register_post_lookup_tracker_fn(self.record_lookup)

    def _init_tbe_tracker_wrapper(self, module: nn.Module) -> None:
        for fqn, named_module in self._model.named_modules():
            if self._should_skip_fqn(fqn):
                continue
            if isinstance(named_module, MANAGED_COLLISION_WRAPPER_MODULES):
                for lookup in named_module._embedding_module._lookups:
                    for emb in lookup._emb_modules:
                        # Only initialize tracker for TBEs that contain tables we want to track
                        should_track_table = self._should_track_table(
                            emb._config.embedding_tables
                        )
                        if should_track_table:
                            emb.init_raw_id_tracker(
                                self.get_indexed_lookups,
                                self.delete,
                            )
                            if self._consumers is None:
                                self._consumers = []
                            self._consumers.append(emb._emb_module.uuid)

    def get_unique_ids(self, consumer: Optional[str] = None) -> Dict[str, torch.Tensor]:
        return {}

    def get_unique(
        self,
        consumer: Optional[str] = None,
        top_percentage: Optional[float] = 1.0,
        per_table_percentage: Optional[Dict[str, Tuple[float, str]]] = None,
        sorted_by_indices: Optional[bool] = True,
    ) -> Dict[str, UniqueRows]:
        return {}

    def clear(self, consumer: Optional[str] = None) -> None:
        pass

    def get_indexed_lookups(
        self,
        tables: List[str],
        consumer: Optional[str] = None,
    ) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        result: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        consumer = consumer or self.DEFAULT_CONSUMER
        assert (
            consumer in self.per_consumer_batch_idx
        ), f"consumer {consumer} not present in {self.per_consumer_batch_idx.values()}"

        index_end: int = self.curr_batch_idx + 1
        index_start = self.per_consumer_batch_idx[consumer]
        indexed_lookups = {}
        if index_start < index_end:
            self.per_consumer_batch_idx[consumer] = index_end
            indexed_lookups = self.store.get_indexed_lookups(index_start, index_end)

        for table in tables:
            raw_ids_list = []
            runtime_meta_list = []
            fqn = self.table_to_fqn[table]
            if fqn in indexed_lookups:
                for indexed_lookup in indexed_lookups[fqn]:
                    if indexed_lookup.raw_ids is not None:
                        raw_ids_list.append(indexed_lookup.raw_ids)
                    if indexed_lookup.runtime_meta is not None:
                        runtime_meta_list.append(indexed_lookup.runtime_meta)
                if (
                    raw_ids_list
                ):  # if raw_ids doesn't exist runtime_meta will not exist so no need to check for runtime_meta
                    result[table] = (raw_ids_list, runtime_meta_list)

        if self._delete_on_read:
            self.store.delete(up_to_idx=min(self.per_consumer_batch_idx.values()))

        return result

    def delete(self, up_to_idx: Optional[int]) -> None:
        self.store.delete(up_to_idx)
