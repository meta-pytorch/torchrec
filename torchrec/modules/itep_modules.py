#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import distributed as dist, nn
from torch.distributed._shard.metadata import ShardMetadata
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding_types import ShardedEmbeddingTable, ShardingType
from torchrec.distributed.types import Shard, ShardedTensor, ShardedTensorMetadata
from torchrec.modules.embedding_modules import reorder_inverse_indices
from torchrec.modules.pruning_logger import PruningLogger, PruningLoggerDefault
from torchrec.sparse.jagged_tensor import _pin_and_move, _to_offsets, KeyedJaggedTensor

try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:intraining_embedding_pruning_gpu"
    )
except OSError:
    pass

logger: logging.Logger = logging.getLogger(__name__)


class GenericITEPModule(nn.Module):
    """
    A generic module for applying In-Training Embedding Pruning (ITEP).
    This module can be hooked into the forward() of `EmbeddingBagCollection`.
    It will prune the embedding tables during training by applying a remapping transform
    to the embedding lookup indices.

    Args:
        table_name_to_unpruned_hash_sizes (Dict[str, int]): Map of table name to
            unpruned hash size.
        lookups (Optional[List[nn.Module]]): List of lookups in the EBC. Defaults to
            `None`.
        enable_pruning (Optional[bool]): Enable pruning or not. Defaults to `True`.
        pruning_interval (Optional[int]): Pruning interval. Defaults to `1001`.

    NOTE:
        The `lookups` argument is optional and is used in the sharded case. If not
        provided, the module will skip initialization for the dummy module.
        The `table_name_to_unpruned_hash_sizes` argument must not be empty. It is a map
        of table names to their unpruned hash sizes.

    Example::
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes={"table1": 1000, "table2": 2000},
            lookups=ShardedEmbeddingBagCollection._lookups,
            enable_pruning=True,
            pruning_interval=1001
        )
    """

    def __init__(
        self,
        table_name_to_unpruned_hash_sizes: Dict[str, int],
        lookups: Optional[List[nn.Module]] = None,
        enable_pruning: bool = True,
        pruning_interval: int = 1001,  # Default pruning interval 1001 iterations
        pg: Optional[dist.ProcessGroup] = None,
        table_name_to_sharding_type: Optional[Dict[str, str]] = None,
        pruning_logger_type: Type[PruningLogger] = PruningLoggerDefault,
    ) -> None:
        self.pruning_logger: Type[PruningLogger] = pruning_logger_type
        with self.pruning_logger.pruning_logger(
            event="ITEP_MODULE_INIT"
        ) as log_details:
            log_details.__setattr__("enable_pruning", enable_pruning)
            log_details.__setattr__("pruning_interval", pruning_interval)

            super(GenericITEPModule, self).__init__()

            if not table_name_to_sharding_type:
                table_name_to_sharding_type = {}

            # Construct in-training embedding pruning args
            self.enable_pruning: bool = enable_pruning
            self.rank_to_virtual_index_mapping: Dict[str, Dict[int, int]] = {}
            self.pruning_interval: int = pruning_interval
            self.lookups: List[nn.Module] = [] if not lookups else lookups
            self.table_name_to_unpruned_hash_sizes: Dict[str, int] = (
                table_name_to_unpruned_hash_sizes
            )
            self.table_name_to_sharding_type: Dict[str, str] = (
                table_name_to_sharding_type
            )

            # Map each feature to a physical address_lookup/row_util buffer
            self.feature_table_map: Dict[str, int] = {}
            self.table_name_to_idx: Dict[str, int] = {}
            self.buffer_offsets_list: List[int] = []
            self.idx_to_table_name: Dict[int, str] = {}
            # Prevent multi-pruning, after moving iteration counter to outside.
            self.last_pruned_iter: int = -1
            self.pg: Optional[dist.ProcessGroup] = pg

            if self.lookups:
                self.init_itep_state()
            else:
                logger.info(
                    "ITEP init: no lookups provided. Skipping init for dummy module."
                )

    def print_itep_eviction_stats(
        self,
        pruned_indices_offsets: torch.Tensor,
        pruned_indices_total_length: torch.Tensor,
        cur_iter: int,
    ) -> None:
        with self.pruning_logger.pruning_logger(event="ITEP_EVICTION") as log_details:
            table_name_to_eviction_ratio = {}
            buffer_idx_to_eviction_ratio = {}
            buffer_idx_to_sizes = {}

            num_buffers = len(self.buffer_offsets_list) - 1
            for buffer_idx in range(num_buffers):
                pruned_start = pruned_indices_offsets[buffer_idx]
                pruned_end = pruned_indices_offsets[buffer_idx + 1]
                pruned_length = pruned_end - pruned_start

                if pruned_length > 0:
                    start = self.buffer_offsets_list[buffer_idx]
                    end = self.buffer_offsets_list[buffer_idx + 1]
                    buffer_length = end - start
                    assert buffer_length > 0
                    eviction_ratio = pruned_length.item() / buffer_length
                    table_name_to_eviction_ratio[self.idx_to_table_name[buffer_idx]] = (
                        eviction_ratio
                    )
                    buffer_idx_to_eviction_ratio[buffer_idx] = eviction_ratio
                    buffer_idx_to_sizes[buffer_idx] = (
                        pruned_length.item(),
                        buffer_length,
                    )

            # Sort the mapping by eviction ratio in descending order
            sorted_mapping = dict(
                sorted(
                    table_name_to_eviction_ratio.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            logged_eviction_mapping = {}
            for idx in sorted_mapping.keys():
                try:
                    logged_eviction_mapping[self.reversed_feature_table_map[idx]] = (
                        sorted_mapping[idx]
                    )
                except KeyError:
                    # in dummy mode, we don't have the feature_table_map or reversed_feature_table_map
                    pass

            table_to_sizes_mapping = {}
            for idx in buffer_idx_to_sizes.keys():
                try:
                    table_to_sizes_mapping[self.reversed_feature_table_map[idx]] = (
                        buffer_idx_to_sizes[idx]
                    )
                except KeyError:
                    # in dummy mode, we don't have the feature_table_map or reversed_feature_table_map
                    pass

            # Print the sorted mapping
            logger.info(f"ITEP: table name to eviction ratio {sorted_mapping}")
            log_details.__setattr__("table_to_pruned_ratio", logged_eviction_mapping)

            # Calculate percentage of indiced updated/evicted during ITEP iter
            pruned_indices_ratio = (
                pruned_indices_total_length / self.buffer_offsets_list[-1]
                if self.buffer_offsets_list[-1] > 0
                else 0
            )
            logger.info(
                f"Performed ITEP in iter {cur_iter}, evicted {pruned_indices_total_length} ({pruned_indices_ratio:%}) indices."
            )

    def get_table_hash_sizes(self, table: ShardedEmbeddingTable) -> Tuple[int, int]:
        unpruned_hash_size = table.num_embeddings

        if table.name in self.table_name_to_unpruned_hash_sizes:
            unpruned_hash_size = self.table_name_to_unpruned_hash_sizes[table.name]
        else:
            # Tables are not pruned by ITEP if table.name not in table_name_to_unpruned_hash_sizes
            unpruned_hash_size = table.num_embeddings
            logger.info(
                f"ITEP: table {table.name} not pruned, because table name is not present in table_name_to_unpruned_hash_sizes."
            )

        return (table.num_embeddings, unpruned_hash_size)

    def create_itep_buffers(
        self,
        buffer_size: int,
        buffer_offsets: List[int],
        table_names: List[str],
        emb_sizes: List[int],
    ) -> None:
        """
        Registers ITEP specific buffers in a way that can be accessed by
        `torch.ops.fbgemm.init_address_lookup` and can be individually checkpointed.
        """
        # Buffers do not enter backward pass
        with torch.no_grad():
            # Don't use register_buffer for buffer_offsets and emb_sizes because they
            # may change as the sharding plan changes between preemption/resumption
            self.buffer_offsets = torch.tensor(
                buffer_offsets, dtype=torch.int64, device=self.current_device
            )
            self.emb_sizes = torch.tensor(
                emb_sizes, dtype=torch.int64, device=self.current_device
            )

            self.address_lookup = torch.zeros(
                buffer_size, dtype=torch.int64, device=self.current_device
            )
            self.row_util = torch.zeros(
                buffer_size, dtype=torch.float32, device=self.current_device
            )

            # Register buffers
            for idx, table_name in enumerate(table_names):
                self.register_buffer(
                    f"{table_name}_itp_address_lookup",
                    self.address_lookup[buffer_offsets[idx] : buffer_offsets[idx + 1]],
                )
                self.register_buffer(
                    f"{table_name}_itp_row_util",
                    self.row_util[buffer_offsets[idx] : buffer_offsets[idx + 1]],
                )

    def init_itep_state(self) -> None:
        idx = 0
        buffer_size = 0
        # Record address_lookup/row_util buffer lengths and offsets for each feature
        buffer_offsets: List[int] = [0]  # number of buffers + 1
        table_names: List[str] = []  # number of buffers + 1
        emb_sizes: List[int] = []  # Store embedding table sizes
        self.current_device = None

        # Iterate over all tables
        for lookup in self.lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            for emb in lookup._emb_modules:

                emb_tables: List[ShardedEmbeddingTable] = emb._config.embedding_tables
                for table in emb_tables:
                    # Skip if table was already added previously (if multiple shards assigned to same rank)
                    if table.name in self.table_name_to_idx:
                        continue

                    (
                        pruned_hash_size,
                        unpruned_hash_size,
                    ) = self.get_table_hash_sizes(table)

                    # Skip tables that are not pruned, aka pruned_hash_size == unpruned_hash_size.
                    if pruned_hash_size == unpruned_hash_size:
                        continue

                    logger.info(
                        f"ITEP: Pruning enabled for table {table.name} with features {table.feature_names}, pruned_hash_size {pruned_hash_size} vs unpruned_hash_size {unpruned_hash_size}"
                    )

                    # buffer size for address_lookup and row_util
                    buffer_size += unpruned_hash_size
                    buffer_offsets.append(buffer_size)
                    table_names.append(table.name)
                    emb_sizes.append(pruned_hash_size)

                    # Create feature to table mappings
                    for feature_name in table.feature_names:
                        self.feature_table_map[feature_name] = idx

                    # Create table_name to buffer idx mappings
                    self.table_name_to_idx[table.name] = idx
                    self.idx_to_table_name[idx] = table.name
                    idx += 1

                    # Check that all features have the same device
                    if (
                        table.local_metadata is not None
                        and table.local_metadata.placement is not None
                    ):
                        if self.current_device is None:
                            self.current_device = (
                                table.local_metadata.placement.device()
                            )
                        else:
                            assert (
                                self.current_device
                                == table.local_metadata.placement.device()
                            ), f"Device of table {table}: {table.local_metadata.placement.device()} does not match existing device: {self.current_device}"

        if self.current_device is None:
            self.current_device = torch.device("cuda")

        self.reversed_feature_table_map: Dict[int, str] = {
            idx: feature_name for feature_name, idx in self.feature_table_map.items()
        }
        self.buffer_offsets_list = buffer_offsets
        # Create buffers for address_lookup and row_util
        self.create_itep_buffers(
            buffer_size=buffer_size,
            buffer_offsets=buffer_offsets,
            table_names=table_names,
            emb_sizes=emb_sizes,
        )

        logger.info(
            f"ITEP: done init_state with feature_table_map {self.feature_table_map} and buffer_offsets {self.buffer_offsets_list}"
        )

        # initialize address_lookup
        torch.ops.fbgemm.init_address_lookup(
            self.address_lookup,
            self.buffer_offsets,
            self.emb_sizes,
        )

    def reset_weight_momentum(
        self,
        pruned_indices: torch.Tensor,
        pruned_indices_offsets: torch.Tensor,
    ) -> None:
        for lookup in self.lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            for emb in lookup._emb_modules:
                emb_tables: List[ShardedEmbeddingTable] = emb._config.embedding_tables

                logical_idx = 0
                logical_table_ids = []
                buffer_ids = []
                for table in emb_tables:
                    name = table.name
                    if name in self.table_name_to_idx:
                        buffer_idx = self.table_name_to_idx[name]
                        start = pruned_indices_offsets[buffer_idx]
                        end = pruned_indices_offsets[buffer_idx + 1]
                        length = end - start
                        if length > 0:
                            logical_table_ids.append(logical_idx)
                            buffer_ids.append(buffer_idx)
                    logical_idx += table.num_features()

                if len(logical_table_ids) > 0:
                    emb.emb_module.reset_embedding_weight_momentum(
                        pruned_indices,
                        pruned_indices_offsets,
                        torch.tensor(
                            logical_table_ids,
                            dtype=torch.int32,
                            requires_grad=False,
                        ),
                        torch.tensor(
                            buffer_ids, dtype=torch.int32, requires_grad=False
                        ),
                    )

    # Flush UVM cache after ITEP eviction to remove stale states
    def flush_uvm_cache(self) -> None:
        for lookup in self.lookups:
            while isinstance(lookup, DistributedDataParallel):
                lookup = lookup.module
            for emb in lookup._emb_modules:
                emb.emb_module.flush()
                emb.emb_module.reset_cache_states()

    def get_remap_info(self, features: KeyedJaggedTensor) -> List[torch.Tensor]:
        keys = features.keys()
        length_per_key = features.length_per_key()
        offset_per_key = features.offset_per_key()

        buffer_idx = []
        feature_lengths = []
        feature_offsets = []
        for i in range(len(keys)):
            key = keys[i]
            if key not in self.feature_table_map:
                continue
            buffer_idx.append(self.feature_table_map[key])
            feature_lengths.append(length_per_key[i])
            feature_offsets.append(offset_per_key[i])

        return [
            torch.tensor(buffer_idx, dtype=torch.int32, device=torch.device("cpu")),
            torch.tensor(
                feature_lengths, dtype=torch.int64, device=torch.device("cpu")
            ),
            torch.tensor(
                feature_offsets, dtype=torch.int64, device=torch.device("cpu")
            ),
        ]

    def get_full_values_list(self, features: KeyedJaggedTensor) -> List[torch.Tensor]:
        inverse_indices = features.inverse_indices()
        batch_size = inverse_indices[1].numel() // len(inverse_indices[0])
        keys = features.keys()
        if not all(key in self.feature_table_map for key in keys):
            keys = [key for key in keys if key in self.feature_table_map]
            key_indices = [features._key_indices()[key] for key in keys]
            features = features.permute(key_indices)
        indices = (
            inverse_indices[1]
            if keys == inverse_indices[0]
            else reorder_inverse_indices(inverse_indices, keys)
        )
        spk_tensor = _pin_and_move(
            torch.tensor(features.stride_per_key()), features.device()
        )
        offset_indices = (
            indices + _to_offsets(spk_tensor)[:-1].unsqueeze(-1)
        ).flatten()
        full_values, full_lengths = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            features.values(),
            features.lengths(),
            features.offsets(),
            offset_indices,
            features.lengths().numel(),
        )
        full_lpk = torch.sum(full_lengths.view(-1, batch_size), dim=1).tolist()
        return list(torch.split(full_values, full_lpk))

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
        cur_iter: int,
    ) -> KeyedJaggedTensor:
        """
        Args:
            sparse_features (KeyedJaggedTensor]): input embedding lookup indices to be
                remapped.
            cur_iter (int): iteration counter.

        Returns:
            KeyedJaggedTensor: remapped KJT

        NOTE:
            We use the same forward method for sharded and non-sharded case.
        """

        if not self.enable_pruning or not self.lookups:
            return sparse_features

        num_buffers = self.buffer_offsets.size(dim=0) - 1
        if num_buffers <= 0:
            return sparse_features

        start_pruning: bool = (
            (cur_iter < 10 and (cur_iter + 1) % 3 == 0)
            or (cur_iter < 100 and (cur_iter + 1) % 30 == 0)
            or (cur_iter < 1000 and (cur_iter + 1) % 300 == 0)
            or ((cur_iter + 1) % self.pruning_interval == 0)
        )
        if start_pruning and self.training and self.last_pruned_iter != cur_iter:
            # Pruning function outputs the indices that need weight/momentum reset
            # The indices order is by physical buffer
            (
                pruned_indices,
                pruned_indices_offsets,
                pruned_indices_total_length,
            ) = torch.ops.fbgemm.prune_embedding_tables(
                cur_iter,
                self.pruning_interval,
                self.address_lookup,
                self.row_util,
                self.buffer_offsets,
                self.emb_sizes,
            )
            # After pruning, reset weight and momentum of pruned indices
            if pruned_indices_total_length > 0 and cur_iter > self.pruning_interval:
                self.reset_weight_momentum(pruned_indices, pruned_indices_offsets)

            if pruned_indices_total_length > 0:
                # Flush UVM cache after every ITEP eviction (every pruning_interval iterations)
                self.flush_uvm_cache()
                logger.info(
                    f"ITEP: trying to flush UVM after ITEP eviction, {cur_iter=}"
                )

            self.last_pruned_iter = cur_iter

            # Print eviction stats
            self.print_itep_eviction_stats(
                pruned_indices_offsets, pruned_indices_total_length, cur_iter
            )

        (
            buffer_idx,
            feature_lengths,
            feature_offsets,
        ) = self.get_remap_info(sparse_features)

        update_util: bool = (
            (cur_iter < 10)
            or (cur_iter < 100 and (cur_iter + 1) % 19 == 0)
            or ((cur_iter + 1) % 39 == 0)
        )
        full_values_list = None
        if update_util and sparse_features.variable_stride_per_key():
            if sparse_features.inverse_indices_or_none() is not None:
                # full util update mode require reconstructing original input indicies from VBE input
                full_values_list = self.get_full_values_list(sparse_features)
            else:
                logger.info(
                    "Switching to deduped util updating mode due to features missing inverse indices. "
                    f"features {list(sparse_features.keys())=} with variable stride: {sparse_features.variable_stride_per_key()}"
                )

        remapped_values = torch.ops.fbgemm.remap_indices_update_utils(
            int(cur_iter),
            buffer_idx,
            feature_lengths,
            feature_offsets,
            sparse_features.values(),
            self.address_lookup,
            self.row_util,
            self.buffer_offsets,
            full_values_list=full_values_list,
            update_util=update_util,
        )

        sparse_features._values = remapped_values

        return sparse_features


class RowwiseShardedITEPModule(GenericITEPModule):
    def _get_local_metadata_idx(self, table: ShardedEmbeddingTable) -> int:
        for i, metadata in enumerate(table.global_metadata.shards_metadata):
            if metadata == table.local_metadata:
                return i
        return -1

    def _get_local_unpruned_hash_sizes_and_offsets(
        self, table: ShardedEmbeddingTable, sharding_type: str
    ) -> Tuple[List[int], List[int]]:
        """
        Returns a tuples of 2 lists: local_unpruned_hash_sizes, local_offsets. They are used
        to create itep buffers and set checkpoint local/global metadata.
        """
        num_devices = len(table.global_metadata.shards_metadata)

        if sharding_type == ShardingType.TABLE_ROW_WISE.value:
            i = 0
            for shard in table.global_metadata.shards_metadata:
                if table.name not in self.rank_to_virtual_index_mapping:
                    self.rank_to_virtual_index_mapping[table.name] = {}
                self.rank_to_virtual_index_mapping[table.name][
                    shard.placement.rank()
                ] = i
                i += 1

        global_hash_size = self.table_name_to_unpruned_hash_sizes[table.name]

        block_size: int = math.ceil(global_hash_size / num_devices)
        last_rank: int = global_hash_size // block_size
        # last_block_size: int = global_hash_size - block_size * last_rank
        shard_sizes: List[int] = []
        shard_offsets: List[int] = [0]

        for rank in range(num_devices):
            if (
                sharding_type == ShardingType.ROW_WISE.value
                or sharding_type == ShardingType.TABLE_ROW_WISE.value
            ):
                if rank < last_rank:
                    local_row: int = block_size
                elif rank == last_rank:
                    local_row: int = global_hash_size - block_size * last_rank
                else:
                    local_row: int = 0
            else:
                if rank <= last_rank:
                    local_row: int = block_size
                else:
                    local_row: int = 0
            shard_sizes.append(local_row)
        shard_offsets = [0]

        for i in range(num_devices - 1):
            shard_offsets.append(shard_sizes[i] + shard_offsets[i])

        return shard_sizes, shard_offsets

    def get_table_hash_sizes(
        self, table: ShardedEmbeddingTable, num_gpus: int = 8
    ) -> Tuple[int, int]:
        # calculate local unpruned and pruned hash sizes
        local_rows = table.local_rows
        assert (
            self.table_name_to_sharding_type is not None
            and len(self.table_name_to_sharding_type) > 0
        ), "No sharding type to feature mapping found"
        sharding_type = self.table_name_to_sharding_type[table.name]

        if table.name in self.table_name_to_unpruned_hash_sizes:
            (
                local_unpruned_shard_sizes,
                _,
            ) = self._get_local_unpruned_hash_sizes_and_offsets(table, sharding_type)

            if sharding_type == ShardingType.ROW_WISE.value:

                local_unpruned_rows = local_unpruned_shard_sizes[
                    table.local_metadata.placement.rank()
                ]
            else:

                local_unpruned_rows = local_unpruned_shard_sizes[
                    self.rank_to_virtual_index_mapping[table.name][
                        table.local_metadata.placement.rank()
                    ]
                ]

        else:
            # Tables are not pruned by ITEP if table.name not in table_name_to_unpruned_hash_sizes
            local_unpruned_rows = local_rows
            logger.info(
                f"ITEP: table {table.name} not pruned, because table name is not present in table_name_to_unpruned_hash_sizes."
            )

        return (local_rows, local_unpruned_rows)

    def get_buffer_param(self, buffer_params: torch.Tensor, idx: int) -> torch.Tensor:
        start = self.buffer_offsets_list[idx]
        end = self.buffer_offsets_list[idx + 1]
        length = end - start
        return buffer_params.detach()[start:end].view(length, 1)

    def get_itp_state_dict(
        self,
        embedding_tables: List[ShardedEmbeddingTable],
        params: torch.Tensor,
        pg: Optional[dist.ProcessGroup] = None,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        suffix: str = "",
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Any]:
        def get_param(params: torch.Tensor, idx: int) -> torch.Tensor:
            start = self.buffer_offsets_list[idx]
            end = self.buffer_offsets_list[idx + 1]
            length = end - start
            return params.detach()[start:end].view(length)

        def get_key_from_table_name_and_suffix(
            table_name: str, prefix: str, buffer_suffix: str
        ) -> str:
            return prefix + f"{table_name}{buffer_suffix}"

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        ckp_tables: List[str] = []
        skipped_tables: List[str] = []

        assert (
            self.table_name_to_sharding_type is not None
            and len(self.table_name_to_sharding_type) > 0
        ), "No sharding type to feature mapping found"

        for table in embedding_tables:
            # Skip tables that are not wrapped with itep
            if table.name not in self.table_name_to_idx.keys():
                skipped_tables.append(table.name)
                continue
            ckp_tables.append(table.name)
            # Create buffer key and param slice
            key = get_key_from_table_name_and_suffix(table.name, prefix, suffix)
            param_idx = self.table_name_to_idx[table.name]
            buffer_param: torch.Tensor = get_param(params, param_idx)
            sharding_type = self.table_name_to_sharding_type[table.name]

            # For inference there is no pg, all tensors are local
            if table.global_metadata is not None and pg is not None:
                # Get unpruned global and local hashsizes
                (
                    unpruned_row_sizes,
                    unpruned_row_offsets,
                ) = self._get_local_unpruned_hash_sizes_and_offsets(
                    table, sharding_type
                )
                global_unpruned_hash_size = self.table_name_to_unpruned_hash_sizes[
                    table.name
                ]
                # Build global shards metadata
                global_shards_metadata: List[ShardMetadata] = []
                for i, table_global_metadata in enumerate(
                    table.global_metadata.shards_metadata
                ):
                    shard_sizes = [unpruned_row_sizes[i]]
                    shard_offsets = [unpruned_row_offsets[i]]
                    placement = copy.deepcopy(table_global_metadata.placement)
                    global_shards_metadata.append(
                        ShardMetadata(
                            shard_sizes=shard_sizes,
                            shard_offsets=shard_offsets,
                            placement=placement,
                        )
                    )
                itp_global_metadata = ShardedTensorMetadata(
                    shards_metadata=global_shards_metadata,
                    size=torch.Size([global_unpruned_hash_size]),
                )
                itp_global_metadata.tensor_properties.dtype = dtype
                itp_global_metadata.tensor_properties.requires_grad = False
                # Build local shard metadata
                local_idx = self._get_local_metadata_idx(table)
                itp_local_medadata = global_shards_metadata[local_idx]

                destination[key] = (
                    ShardedTensor._init_from_local_shards_and_global_metadata(
                        local_shards=[Shard(buffer_param, itp_local_medadata)],
                        sharded_tensor_metadata=itp_global_metadata,
                        process_group=pg,
                    )
                )
            else:
                destination[key] = buffer_param

        logger.info(
            f"ITEP: get_itp_state_dict for {suffix}), got {ckp_tables}, skippped {skipped_tables}"
        )
        return destination

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        loaded_keys = []
        unexpected_keys = list(state_dict.keys())
        for key, dst_param in self.state_dict().items():
            if key in state_dict:
                src_param = state_dict[key]
                if isinstance(dst_param, ShardedTensor):
                    assert isinstance(src_param, ShardedTensor)
                    assert len(dst_param.local_shards()) == len(
                        src_param.local_shards()
                    )
                    for dst_local_shard, src_local_shard in zip(
                        dst_param.local_shards(), src_param.local_shards()
                    ):
                        assert (
                            dst_local_shard.metadata.shard_offsets
                            == src_local_shard.metadata.shard_offsets
                        )
                        assert (
                            dst_local_shard.metadata.shard_sizes
                            == src_local_shard.metadata.shard_sizes
                        )

                        dst_local_shard.tensor.detach().copy_(src_local_shard.tensor)
                else:
                    assert isinstance(src_param, torch.Tensor) and isinstance(
                        dst_param, torch.Tensor
                    )
                    dst_param.detach().copy_(src_param)
                unexpected_keys.remove(key)
                loaded_keys.append(key)
            else:
                missing_keys.append(key)

        logger.info(
            f"ITEP: load_state_dict, loaded {loaded_keys}, missed {missing_keys}, , unexpected {unexpected_keys}"
        )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for lookup in self.lookups:
            list_of_tables: List[ShardedEmbeddingTable] = []
            for emb_config in lookup.grouped_configs:
                list_of_tables.extend(emb_config.embedding_tables)

            destination = self.get_itp_state_dict(
                list_of_tables,
                self.address_lookup,
                self.pg,
                destination,
                prefix,
                suffix="_itp_address_lookup",
                dtype=torch.int64,
            )
            destination = self.get_itp_state_dict(
                list_of_tables,
                self.row_util,
                self.pg,
                destination,
                prefix,
                suffix="_itp_row_util",
                dtype=torch.float32,
            )
        return destination
