#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import copy
import inspect
import itertools
import logging
import tempfile
from dataclasses import dataclass
from math import sqrt
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BackendType,
    EvictionPolicy,
    KVZCHParams,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    PoolingMode,
    RESParams,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.ssd import ASSOC, SSDTableBatchedEmbeddingBags
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
from torch import nn
from torch.distributed._tensor import DTensor, Replicate, Shard as DTensorShard
from torchrec.distributed.comm import get_local_rank, get_node_group_size
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_kernel import (
    BaseEmbedding,
    create_virtual_sharded_tensors,
    create_virtual_table_local_metadata,
    get_state_dict,
)
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    DTensorMetadata,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardingType,
    ShardMetadata,
    TensorProperties,
)
from torchrec.distributed.utils import append_prefix, none_throws
from torchrec.modules.embedding_configs import (
    CountBasedEvictionPolicy,
    CountTimestampMixedEvictionPolicy,
    data_type_to_sparse_type,
    FeatureL2NormBasedEvictionPolicy,
    FeatureScoreBasedEvictionPolicy,
    NoEvictionPolicy,
    pooling_type_to_pooling_mode,
    TimestampBasedEvictionPolicy,
)
from torchrec.optim.fused import (
    EmptyFusedOptimizer,
    FusedOptimizer,
    FusedOptimizerModule,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

RES_ENABLED_TABLES_STR = "res_enabled_tables"
RES_STORE_SHARDS_STR = "res_store_shards"
ENABLE_RAW_EMBEDDING_STREAMING_STR = "enable_raw_embedding_streaming"


def _populate_res_params(config: GroupedEmbeddingConfig) -> Tuple[bool, RESParams]:
    # populate res_params, which is used for raw embedding streaming
    # here only populates the params available in fused_params and TBE configs
    res_params: RESParams = RESParams()
    fused_params = config.fused_params or {}
    # read and clean up the fused_params that are not in the constructor
    if RES_STORE_SHARDS_STR in fused_params:
        res_params.res_store_shards = fused_params.get(RES_STORE_SHARDS_STR)
        del fused_params[RES_STORE_SHARDS_STR]
    res_enabled_tables: Optional[List[str]] = None
    if RES_ENABLED_TABLES_STR in fused_params:
        res_enabled_tables = (
            fused_params.get(RES_ENABLED_TABLES_STR).split(",")
            if fused_params.get(RES_ENABLED_TABLES_STR) is not None
            else None
        )
        del fused_params[RES_ENABLED_TABLES_STR]
    enable_raw_embedding_streaming: Optional[bool] = None
    if ENABLE_RAW_EMBEDDING_STREAMING_STR in fused_params:
        enable_raw_embedding_streaming = fused_params.get(
            ENABLE_RAW_EMBEDDING_STREAMING_STR
        )

    if (
        enable_raw_embedding_streaming is None
        or enable_raw_embedding_streaming is False
    ):
        return (False, res_params)
    res_params.table_names = [table.name for table in config.embedding_tables]
    if res_enabled_tables is not None and len(res_enabled_tables) != 0:
        if len(set(res_enabled_tables) & set(res_params.table_names)) == 0:
            logger.info(
                f"No table is enabled for raw embedding streaming, "
                f"raw embedding streaming is disabled, {res_enabled_tables=} {res_params.table_names=}"
            )
            return (False, res_params)
    res_params.table_offsets = []
    for emb_tbl in config.embedding_tables:
        local_metadata = emb_tbl.local_metadata
        if (
            local_metadata is not None
            and local_metadata.shard_offsets is not None
            and len(local_metadata.shard_offsets) >= 1
        ):
            res_params.table_offsets.append(local_metadata.shard_offsets[0])
    return (enable_raw_embedding_streaming, res_params)


def _populate_ssd_tbe_params(config: GroupedEmbeddingConfig) -> Dict[str, Any]:
    """
    Construct SSD TBE params dict from config and fused params dict.
    """
    fused_params = config.fused_params or {}

    ssd_tbe_params: Dict[str, Any] = {}

    # drop the non-ssd tbe fused params
    ssd_tbe_signature = inspect.signature(
        SSDTableBatchedEmbeddingBags.__init__
    ).parameters.keys()
    invalid_keys: List[str] = []

    for key, value in fused_params.items():
        if key not in ssd_tbe_signature:
            invalid_keys.append(key)
        else:
            ssd_tbe_params[key] = value
    if len(invalid_keys) > 0:
        logger.warning(
            f"Dropping {invalid_keys} since they are not valid SSD TBE params."
        )

    # populate number cache sets, aka number of rows of the cache space
    if "cache_sets" not in ssd_tbe_params:
        cache_load_factor = fused_params.get("cache_load_factor")
        if cache_load_factor:
            cache_load_factor = fused_params.get("cache_load_factor")
            logger.info(
                f"Using cache load factor from fused params dict: {cache_load_factor}"
            )
        else:
            cache_load_factor = 0.2

        local_rows_sum: int = sum(table.local_rows for table in config.embedding_tables)
        ssd_tbe_params["cache_sets"] = max(
            int(cache_load_factor * local_rows_sum / ASSOC), 1
        )

    # populate init min and max
    if config.is_using_virtual_table:
        _generate_init_range_for_virtual_tables(ssd_tbe_params, config)

    if (
        "ssd_uniform_init_lower" not in ssd_tbe_params
        or "ssd_uniform_init_upper" not in ssd_tbe_params
    ):
        # Right now we do not support a per table init max and min. To use
        # per table init max and min, either we allow it in SSD TBE, or we
        # create one SSD TBE per table.
        # TODO: Solve the init problem
        mins = [table.get_weight_init_min() for table in config.embedding_tables]
        maxs = [table.get_weight_init_max() for table in config.embedding_tables]
        ssd_tbe_params["ssd_uniform_init_lower"] = sum(mins) / len(
            config.embedding_tables
        )
        ssd_tbe_params["ssd_uniform_init_upper"] = sum(maxs) / len(
            config.embedding_tables
        )

    if "ssd_storage_directory" not in ssd_tbe_params:
        ssd_tbe_params["ssd_storage_directory"] = tempfile.mkdtemp()
    else:
        if "@local_rank" in ssd_tbe_params["ssd_storage_directory"]:
            # assume we have initialized a process group already
            ssd_tbe_params["ssd_storage_directory"] = ssd_tbe_params[
                "ssd_storage_directory"
            ].replace("@local_rank", str(get_local_rank()))

    if "weights_precision" not in ssd_tbe_params:
        weights_precision = data_type_to_sparse_type(config.data_type)
        ssd_tbe_params["weights_precision"] = weights_precision

    if "max_l1_cache_size" in fused_params:
        l1_cache_size = fused_params.get("max_l1_cache_size") * 1024 * 1024
        max_dim: int = max(table.local_cols for table in config.embedding_tables)
        weight_precision_bytes = ssd_tbe_params["weights_precision"].bit_rate() / 8
        max_cache_sets = (
            l1_cache_size / ASSOC / weight_precision_bytes / max_dim
        )  # 100MB

        if ssd_tbe_params["cache_sets"] > int(max_cache_sets):
            logger.warning(
                f"cache_sets {ssd_tbe_params['cache_sets']} is larger than max_cache_sets {max_cache_sets} calculated "
                "by max_l1_cache_size, cap at max_cache_sets instead"
            )
            ssd_tbe_params["cache_sets"] = int(max_cache_sets)

    if "kvzch_eviction_trigger_mode" in fused_params and config.is_using_virtual_table:
        ssd_tbe_params["kvzch_eviction_trigger_mode"] = fused_params.get(
            "kvzch_eviction_trigger_mode"
        )

    ssd_tbe_params["table_names"] = [table.name for table in config.embedding_tables]

    enable_res, res_params = _populate_res_params(config)
    ssd_tbe_params["res_params"] = res_params
    ssd_tbe_params[ENABLE_RAW_EMBEDDING_STREAMING_STR] = enable_res

    return ssd_tbe_params


def _generate_init_range_for_virtual_tables(
    tbe_params: Dict[str, Any],
    config: GroupedEmbeddingConfig,
) -> None:
    """
    Generate uniform init range for zero collision TBE based
    """
    # populate init min and max
    if (
        "ssd_uniform_init_lower" not in tbe_params
        or "ssd_uniform_init_upper" not in tbe_params
    ):
        # Right now we do not support a per table init max and min. To use
        # per table init max and min, either we allow it in SSD TBE, or we
        # create one SSD TBE per table.
        weights_precision = data_type_to_sparse_type(config.data_type)

        # For Float32: use mathematically correct values, for Half: use safe range
        max_size = 4_000_000_000  # 4B virtual embeddings
        default_init_range = (
            (-sqrt(1 / max_size), sqrt(1 / max_size))
            if weights_precision.as_dtype() == torch.float32
            else (-0.001, 0.001)
        )

        def get_init_value(
            table_init_val: Optional[float], default_value: float
        ) -> float:
            return table_init_val if table_init_val is not None else default_value

        init_mins = [
            get_init_value(table.weight_init_min, default_init_range[0])
            for table in config.embedding_tables
        ]
        init_maxs = [
            get_init_value(table.weight_init_max, default_init_range[1])
            for table in config.embedding_tables
        ]

        num_tables = len(config.embedding_tables)
        tbe_params["ssd_uniform_init_lower"] = sum(init_mins) / num_tables
        tbe_params["ssd_uniform_init_upper"] = sum(init_maxs) / num_tables


def _populate_zero_collision_tbe_params(
    tbe_params: Dict[str, Any],
    sharded_local_buckets: List[Tuple[int, int, int]],
    config: GroupedEmbeddingConfig,
    backend_type: BackendType,
    embedding_cache_mode: bool = False,
) -> None:
    """
    Construct Zero Collision TBE params from config and fused params dict.
    """
    bucket_offsets: List[Tuple[int, int]] = [
        (offset_start, offset_end)
        for offset_start, offset_end, _ in sharded_local_buckets
    ]
    bucket_sizes: List[int] = [size for _, _, size in sharded_local_buckets]

    enabled = False
    meta_header_lens = [0] * len(config.embedding_tables)
    for i, table in enumerate(config.embedding_tables):
        # virtual_table_eviction_policy won't be None in reality: https://fburl.com/code/864a0w0f
        assert (
            table.virtual_table_eviction_policy is not None
        ), "virtual_table_eviction_policy for kvzch table should not be None"
        meta_header_lens[i] = table.virtual_table_eviction_policy.get_meta_header_len()
        if not isinstance(table.virtual_table_eviction_policy, NoEvictionPolicy):
            enabled = True
    if enabled:
        counter_thresholds = [0] * len(config.embedding_tables)
        ttls_in_mins = [0] * len(config.embedding_tables)
        counter_decay_rates = [0.0] * len(config.embedding_tables)
        feature_score_counter_decay_rates = [0.0] * len(config.embedding_tables)
        training_id_eviction_trigger_count = [0] * len(config.embedding_tables)
        training_id_keep_count = [0] * len(config.embedding_tables)
        l2_weight_thresholds = [0.0] * len(config.embedding_tables)
        eviction_strategy = -1
        table_names = [table.name for table in config.embedding_tables]
        l2_cache_size = tbe_params["l2_cache_size"]
        if "kvzch_eviction_trigger_mode" in tbe_params:
            eviction_trigger_mode = tbe_params["kvzch_eviction_trigger_mode"]
            tbe_params.pop("kvzch_eviction_trigger_mode")
        else:
            eviction_trigger_mode = 2  # 2 means mem_util based eviction
        for i, table in enumerate(config.embedding_tables):
            policy_t = table.virtual_table_eviction_policy
            if policy_t is not None:
                if isinstance(policy_t, CountBasedEvictionPolicy):
                    training_id_eviction_trigger_count[i] = (
                        policy_t.training_id_eviction_trigger_count
                    )
                    counter_thresholds[i] = policy_t.eviction_threshold
                    counter_decay_rates[i] = policy_t.decay_rate
                    if eviction_strategy == -1 or eviction_strategy == 1:
                        eviction_strategy = 1
                    else:
                        raise ValueError(
                            f"Do not support multiple eviction strategy in one tbe {eviction_strategy} and 1 for tables {table_names}"
                        )
                elif isinstance(policy_t, FeatureScoreBasedEvictionPolicy):
                    feature_score_counter_decay_rates[i] = policy_t.decay_rate
                    training_id_eviction_trigger_count[i] = (
                        policy_t.training_id_eviction_trigger_count
                    )
                    training_id_keep_count[i] = policy_t.training_id_keep_count
                    ttls_in_mins[i] = policy_t.eviction_ttl_mins
                    if eviction_strategy == -1 or eviction_strategy == 5:
                        eviction_strategy = 5
                    else:
                        raise ValueError(
                            f"Do not support multiple eviction strategy in one tbe {eviction_strategy} and 5 for tables {table_names}"
                        )
                elif isinstance(policy_t, TimestampBasedEvictionPolicy):
                    training_id_eviction_trigger_count[i] = (
                        policy_t.training_id_eviction_trigger_count
                    )
                    ttls_in_mins[i] = policy_t.eviction_ttl_mins
                    if eviction_strategy == -1 or eviction_strategy == 0:
                        eviction_strategy = 0
                    else:
                        raise ValueError(
                            f"Do not support multiple eviction strategy in one tbe {eviction_strategy} and 0 for tables {table_names}"
                        )
                elif isinstance(policy_t, FeatureL2NormBasedEvictionPolicy):
                    training_id_eviction_trigger_count[i] = (
                        policy_t.training_id_eviction_trigger_count
                    )
                    l2_weight_thresholds[i] = policy_t.eviction_threshold
                    if eviction_strategy == -1 or eviction_strategy == 3:
                        eviction_strategy = 3
                    else:
                        raise ValueError(
                            f"Do not support multiple eviction strategy in one tbe {eviction_strategy} and 3 for tables {table_names}"
                        )
                elif isinstance(policy_t, CountTimestampMixedEvictionPolicy):
                    training_id_eviction_trigger_count[i] = (
                        policy_t.training_id_eviction_trigger_count
                    )
                    counter_thresholds[i] = policy_t.eviction_threshold
                    counter_decay_rates[i] = policy_t.decay_rate
                    ttls_in_mins[i] = policy_t.eviction_ttl_mins
                    if eviction_strategy == -1 or eviction_strategy == 2:
                        eviction_strategy = 2
                    else:
                        raise ValueError(
                            f"Do not support multiple eviction strategy in one tbe {eviction_strategy} and 2 for tables {table_names}"
                        )
                else:
                    raise ValueError(
                        f"Unsupported eviction policy {policy_t} for table {table.name}"
                    )
        eviction_policy = EvictionPolicy(
            eviction_trigger_mode=eviction_trigger_mode,
            eviction_mem_threshold_gb=l2_cache_size,
            eviction_strategy=eviction_strategy,
            counter_thresholds=counter_thresholds,
            ttls_in_mins=ttls_in_mins,
            counter_decay_rates=counter_decay_rates,
            feature_score_counter_decay_rates=feature_score_counter_decay_rates,
            training_id_eviction_trigger_count=training_id_eviction_trigger_count,
            training_id_keep_count=training_id_keep_count,
            l2_weight_thresholds=l2_weight_thresholds,
            meta_header_lens=meta_header_lens,
        )
    else:
        eviction_policy = EvictionPolicy(meta_header_lens=meta_header_lens)

    embedding_cache_mode_ = (
        embedding_cache_mode
        if embedding_cache_mode
        else (
            config.fused_params.get("embedding_cache_mode", False)
            if config.fused_params
            else False
        )
    )
    tbe_params["kv_zch_params"] = KVZCHParams(
        bucket_offsets=bucket_offsets,
        bucket_sizes=bucket_sizes,
        enable_optimizer_offloading=True,
        backend_return_whole_row=(backend_type == BackendType.DRAM),
        eviction_policy=eviction_policy,
        embedding_cache_mode=embedding_cache_mode_,
    )


def _get_sharded_local_buckets_for_zero_collision(
    embedding_tables: List[ShardedEmbeddingTable],
    pg: Optional[dist.ProcessGroup] = None,
) -> List[Tuple[int, int, int]]:
    """
    utils to get bucket offset start, bucket offset end, bucket size based on embedding sharding spec
    """
    sharded_local_buckets: List[Tuple[int, int, int]] = []
    world_size = dist.get_world_size(pg)
    local_rank = dist.get_rank(pg)

    for table in embedding_tables:
        total_num_buckets = none_throws(table.total_num_buckets)
        assert (
            total_num_buckets % world_size == 0
        ), f"total_num_buckets={total_num_buckets} must be divisible by world_size={world_size}"
        assert (
            table.total_num_buckets
            and table.num_embeddings % table.total_num_buckets == 0
        ), f"Table size '{table.num_embeddings}' must be divisible by num_buckets '{table.total_num_buckets}'"
        bucket_offset_start = total_num_buckets // world_size * local_rank
        bucket_offset_end = min(
            total_num_buckets, total_num_buckets // world_size * (local_rank + 1)
        )
        bucket_size = (
            table.num_embeddings + total_num_buckets - 1
        ) // total_num_buckets
        sharded_local_buckets.append(
            (bucket_offset_start, bucket_offset_end, bucket_size)
        )
        logger.info(
            f"bucket_offset: {bucket_offset_start}:{bucket_offset_end}, bucket_size: {bucket_size} for table {table.name}"
        )
    return sharded_local_buckets


@dataclass
class ShardParams:
    optimizer_states: List[Optional[Tuple[torch.Tensor]]]
    optimizer_states_keys: List[torch.Tensor]
    local_metadata: List[ShardMetadata]
    global_metadata: ShardedTensorMetadata
    embedding_weights: List[torch.Tensor]
    dtensor_metadata: List[DTensorMetadata]


class KeyValueEmbeddingFusedOptimizer(FusedOptimizer):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SSDTableBatchedEmbeddingBags,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Fused optimizer for SSD TBE. Right now it only supports tuning learning
        rate.
        """
        self._emb_module: SSDTableBatchedEmbeddingBags = emb_module
        self._pg = pg

        # Initializing all required variables

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.get_learning_rate(),
        }
        params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}
        all_optimizer_states = emb_module.get_optimizer_state(None)
        table_to_shard_params: Dict[str, ShardParams] = {}

        # Changing weights location to CPU
        embedding_weights_by_table, _, _, _ = emb_module.split_embedding_weights()
        emb_tables_copy = copy.deepcopy(config.embedding_tables)
        for emb_table in emb_tables_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")

        # [Step 1] Create ShardParams for every embedding table
        for (
            table_config,
            optimizer_states,
            weight,
        ) in itertools.zip_longest(
            emb_tables_copy,
            all_optimizer_states,
            embedding_weights_by_table,
        ):
            # Creating a placeholder shardParam for every embedding table
            if table_config.name not in table_to_shard_params:
                table_to_shard_params[table_config.name] = ShardParams(
                    optimizer_states=[],
                    local_metadata=[],
                    embedding_weights=[],
                    dtensor_metadata=[],
                    global_metadata=ShardedTensorMetadata(),
                    optimizer_states_keys=[],
                )

            optimizer_state_values = None
            if optimizer_states:
                optimizer_state_values = tuple(optimizer_states.values())
                for optimizer_state_value in optimizer_state_values:
                    assert (
                        table_config.local_rows == optimizer_state_value.size(0)
                        or optimizer_state_value.nelement() == 1  # single value state
                    )
                # Saving the optimizer keys for every table
                table_to_shard_params[table_config.name].optimizer_states_keys.append(
                    optimizer_states.keys()
                )

            # Adding data to the shard params for every table
            table_to_shard_params[table_config.name].optimizer_states.append(
                optimizer_state_values
            )
            table_to_shard_params[table_config.name].local_metadata.append(
                table_config.local_metadata
            )
            table_to_shard_params[table_config.name].dtensor_metadata.append(
                table_config.dtensor_metadata
            )
            table_to_shard_params[table_config.name].embedding_weights.append(weight)
            table_to_shard_params[table_config.name].global_metadata = (
                table_config.global_metadata
            )

        # Loop through every table
        seen_tables = set()
        for table_config in emb_tables_copy:
            if table_config.name in seen_tables:
                continue
            seen_tables.add(table_config.name)
            shard_params: ShardParams = table_to_shard_params[table_config.name]

            local_weight_shards = []
            for local_weight, local_metadata in zip(
                shard_params.embedding_weights, shard_params.local_metadata
            ):
                # Creating a shard for every tensor -> this will have the tensor and its metadata
                local_weight_shards.append(Shard(local_weight, local_metadata))
                shard_params.global_metadata.tensor_properties.dtype = (
                    local_weight.dtype
                )
                shard_params.global_metadata.tensor_properties.requires_grad = (
                    local_weight.requires_grad
                )
            # Creating a Shard Tensor using all the above created shards
            weight = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards=local_weight_shards,
                sharded_tensor_metadata=shard_params.global_metadata,
                process_group=self._pg,
            )
            param_key = table_config.name + ".weight"

            # Saving the shard tensor
            state[weight] = {}
            param_group["params"].append(weight)
            params[param_key] = weight

            # Update sharding dimension and grid sharding for every embedding table
            self.sharding_dim: int = (
                1 if table_config.local_cols != table_config.embedding_dim else 0
            )

            self.is_grid_sharded: bool = (
                True
                if table_config.local_cols != table_config.embedding_dim
                and table_config.local_rows != table_config.num_embeddings
                else False
            )

            # Going through Optimizers
            if all(
                opt_state is not None for opt_state in shard_params.optimizer_states
            ):
                # Number of optimizers for the table
                num_states: int = min(
                    # pyre-ignore
                    [len(opt_state) for opt_state in shard_params.optimizer_states]
                )
                optimizer_state_keys = []
                if num_states > 0:
                    optimizer_state_keys = table_to_shard_params[
                        table_config.name
                    ].optimizer_states_keys

                for cur_state_idx in range(0, num_states):
                    if cur_state_idx == 0:
                        # for backward compatibility
                        # If only one optimizer state is present, we assume it is the momentum1 state
                        cur_state_key = "momentum1"
                    else:
                        cur_state_key = optimizer_state_keys[cur_state_idx]

                    # Creating the ShardedTensor for the optimizer weights for the table
                    state[weight][f"{table_config.name}.{cur_state_key}"] = (
                        self.get_sharded_optim_state(
                            cur_state_idx + 1, cur_state_key, shard_params, table_config
                        )
                    )

        logger.info("Completed initializing keyvalueembeddingfusedOptimizer")
        super().__init__(params, state, [param_group])

    def get_sharded_optim_state(
        self,
        momentum_idx: int,
        state_key: str,
        shard_params: ShardParams,
        table_config: ShardedEmbeddingTable,
    ) -> Union[ShardedTensor, DTensor]:

        momentum_local_shards: List[Shard] = []
        optimizer_sharded_tensor_metadata: ShardedTensorMetadata

        # Momentum idx is minimum 1
        # pyre-ignore [16]
        optim_state = shard_params.optimizer_states[0][momentum_idx - 1]
        if (
            optim_state.nelement() == 1 and state_key != "momentum1"
        ):  # special handling for backward compatibility, momentum1 is rowwise state for rowwise_adagrad
            # single value state: one value per table
            (
                table_shard_metadata_to_optimizer_shard_metadata,
                optimizer_sharded_tensor_metadata,
            ) = self.get_optimizer_single_value_shard_metadata_and_global_metadata(
                shard_params.global_metadata,
                optim_state,
            )
        elif optim_state.dim() == 1:
            # rowwise state: param.shape[0] == state.shape[0], state.shape[1] == 1
            (
                table_shard_metadata_to_optimizer_shard_metadata,
                optimizer_sharded_tensor_metadata,
            ) = self.get_optimizer_rowwise_shard_metadata_and_global_metadata(
                shard_params.global_metadata,
                optim_state,
                self.sharding_dim,
                self.is_grid_sharded,
            )
        else:
            # pointwise state: param.shape == state.shape
            (
                table_shard_metadata_to_optimizer_shard_metadata,
                optimizer_sharded_tensor_metadata,
            ) = self.get_optimizer_pointwise_shard_metadata_and_global_metadata(
                shard_params.global_metadata,
                optim_state,
            )

        for optimizer_state, table_shard_local_metadata in zip(
            shard_params.optimizer_states, shard_params.local_metadata
        ):
            local_optimizer_shard_metadata = (
                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_local_metadata
                ]
            )
            momentum_local_shards.append(
                Shard(
                    optimizer_state[momentum_idx - 1],
                    local_optimizer_shard_metadata,
                )
            )

        # Convert optimizer state to DTensor if enabled
        if (
            table_config.dtensor_metadata is not None
            and table_config.dtensor_metadata.mesh
        ):
            dtensor_metadata = table_config.dtensor_metadata
            # if rowwise state we do Shard(0), regardless of how the table is sharded
            if optim_state.dim() == 1:
                stride = (1,)
                placements = (
                    (Replicate(), DTensorShard(0))
                    if dtensor_metadata.mesh is not None
                    and dtensor_metadata.mesh.ndim == 2
                    else (DTensorShard(0),)
                )
            else:
                stride = dtensor_metadata.stride
                placements = dtensor_metadata.placements

            return DTensor.from_local(
                local_tensor=LocalShardsWrapper(
                    local_shards=[x.tensor for x in momentum_local_shards],
                    local_offsets=[  # pyre-ignore[6]
                        x.metadata.shard_offsets for x in momentum_local_shards
                    ],
                ),
                device_mesh=dtensor_metadata.mesh,
                placements=placements,
                shape=optimizer_sharded_tensor_metadata.size,
                stride=stride,
                run_check=False,
            )
        else:
            # TODO we should be creating this in SPMD fashion (e.g. init_from_local_shards), and let it derive global metadata.
            return ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards=momentum_local_shards,
                sharded_tensor_metadata=optimizer_sharded_tensor_metadata,
                process_group=self._pg,
            )

    def get_optimizer_single_value_shard_metadata_and_global_metadata(
        self,
        table_global_metadata: ShardedTensorMetadata,
        optimizer_state: torch.Tensor,
    ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
        table_global_shards_metadata: List[ShardMetadata] = (
            table_global_metadata.shards_metadata
        )

        table_shard_metadata_to_optimizer_shard_metadata = {}
        for offset, table_shard_metadata in enumerate(table_global_shards_metadata):

            # pyre-ignore [16]
            table_shard_metadata.placement._device = optimizer_state.device
            # Creating shardMetaData
            table_shard_metadata_to_optimizer_shard_metadata[table_shard_metadata] = (
                ShardMetadata(
                    shard_sizes=[1],  # single value optimizer state
                    shard_offsets=[offset],  # offset increases by 1 for each shard
                    placement=table_shard_metadata.placement,
                )
            )

        tensor_properties = TensorProperties(
            dtype=optimizer_state.dtype,
            layout=optimizer_state.layout,
            requires_grad=False,
        )
        # Creating ShardedTensorMetaData
        single_value_optimizer_st_metadata = ShardedTensorMetadata(
            shards_metadata=list(
                table_shard_metadata_to_optimizer_shard_metadata.values()
            ),
            size=torch.Size([len(table_global_shards_metadata)]),
            tensor_properties=tensor_properties,
        )

        return (
            table_shard_metadata_to_optimizer_shard_metadata,
            single_value_optimizer_st_metadata,
        )

    def get_optimizer_rowwise_shard_metadata_and_global_metadata(
        self,
        table_global_metadata: ShardedTensorMetadata,
        optimizer_state: torch.Tensor,
        sharding_dim: int,
        is_grid_sharded: bool = False,
    ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
        table_global_shards_metadata: List[ShardMetadata] = (
            table_global_metadata.shards_metadata
        )

        if sharding_dim == 1:
            # column-wise sharding
            # sort the metadata based on column offset and
            # we construct the momentum tensor in row-wise sharded way
            table_global_shards_metadata = sorted(
                table_global_shards_metadata,
                key=lambda shard: shard.shard_offsets[1],
            )

        table_shard_metadata_to_optimizer_shard_metadata = {}
        rolling_offset = 0
        for idx, table_shard_metadata in enumerate(table_global_shards_metadata):
            offset = table_shard_metadata.shard_offsets[0]

            if is_grid_sharded:
                # we use a rolling offset to calculate the current offset for shard to account for uneven row wise case for our shards
                offset = rolling_offset
                rolling_offset += table_shard_metadata.shard_sizes[0]
            elif sharding_dim == 1:
                # for column-wise sharding, we still create row-wise sharded metadata for optimizer
                # manually create a row-wise offset
                offset = idx * table_shard_metadata.shard_sizes[0]

            # pyre-ignore [16]
            table_shard_metadata.placement._device = optimizer_state.device
            table_shard_metadata_to_optimizer_shard_metadata[table_shard_metadata] = (
                ShardMetadata(
                    shard_sizes=[table_shard_metadata.shard_sizes[0]],
                    shard_offsets=[offset],
                    placement=table_shard_metadata.placement,
                )
            )

        tensor_properties = TensorProperties(
            dtype=optimizer_state.dtype,
            layout=optimizer_state.layout,
            requires_grad=False,
        )
        len_rw_shards = (
            len(table_shard_metadata_to_optimizer_shard_metadata)
            if sharding_dim == 1 and not is_grid_sharded
            else 1
        )
        # for grid sharding, the row dimension is replicated CW shard times
        grid_shard_nodes = (
            len(table_global_shards_metadata) // get_node_group_size()
            if is_grid_sharded
            else 1
        )
        rowwise_optimizer_st_metadata = ShardedTensorMetadata(
            shards_metadata=list(
                table_shard_metadata_to_optimizer_shard_metadata.values()
            ),
            size=torch.Size(
                [table_global_metadata.size[0] * len_rw_shards * grid_shard_nodes]
            ),
            tensor_properties=tensor_properties,
        )

        return (
            table_shard_metadata_to_optimizer_shard_metadata,
            rowwise_optimizer_st_metadata,
        )

    def get_optimizer_pointwise_shard_metadata_and_global_metadata(
        self,
        table_global_metadata: ShardedTensorMetadata,
        optimizer_state: torch.Tensor,
    ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
        table_global_shards_metadata: List[ShardMetadata] = (
            table_global_metadata.shards_metadata
        )

        table_shard_metadata_to_optimizer_shard_metadata = {}

        for table_shard_metadata in table_global_shards_metadata:
            # pyre-ignore [16]
            table_shard_metadata.placement._device = optimizer_state.device
            table_shard_metadata_to_optimizer_shard_metadata[table_shard_metadata] = (
                ShardMetadata(
                    shard_sizes=table_shard_metadata.shard_sizes,
                    shard_offsets=table_shard_metadata.shard_offsets,
                    placement=table_shard_metadata.placement,
                )
            )
        tensor_properties = TensorProperties(
            dtype=optimizer_state.dtype,
            layout=optimizer_state.layout,
            requires_grad=False,
        )
        pointwise_optimizer_st_metadata = ShardedTensorMetadata(
            shards_metadata=list(
                table_shard_metadata_to_optimizer_shard_metadata.values()
            ),
            size=table_global_metadata.size,
            tensor_properties=tensor_properties,
        )

        return (
            table_shard_metadata_to_optimizer_shard_metadata,
            pointwise_optimizer_st_metadata,
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])


class ZeroCollisionKeyValueEmbeddingFusedOptimizer(FusedOptimizer):
    def __init__(  # noqa C901
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SSDTableBatchedEmbeddingBags,
        sharded_embedding_weights_by_table: List[ShardedTensor],
        table_name_to_weight_count_per_rank: Dict[str, List[int]],
        sharded_embedding_weight_ids: Optional[List[ShardedTensor]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Implementation of a FusedOptimizer for KV ZCH computation kernel.
        The difference between this and KeyValueEmbeddingFusedOptimizer is that this optimizer
        support dynamic trained embedding weights and weights, instead of relying on fixed
        embedding table sizes.

        Args:
            config (GroupedEmbeddingConfig): the embedding config
            emb_module (SSDTableBatchedEmbeddingBags): the embedding module
            sharded_embedding_weights_by_table (List[ShardedTensor]): the sharded embedding weights
            table_name_to_weight_count_per_rank (Dict[str, List[int]]): the table name to weight count per rank
            sharded_embedding_weight_ids (Optional[List[ShardedTensor]]): the sharded embedding weight ids
            pg (Optional[dist.ProcessGroup]): the process group
        """
        self._emb_module: SSDTableBatchedEmbeddingBags = emb_module
        self._pg = pg
        self._my_rank: int = dist.get_rank(self._pg)
        self._config = config
        self._sharded_embedding_weight_ids: Optional[List[ShardedTensor]] = (
            sharded_embedding_weight_ids
        )
        self._table_name_to_weight_count_per_rank: Dict[str, List[int]] = (
            table_name_to_weight_count_per_rank
        )

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.get_learning_rate(),
        }

        params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}

        sorted_id_tensors = (
            [
                sharded_t._local_shards[0].tensor
                for sharded_t in self._sharded_embedding_weight_ids
            ]
            if self._sharded_embedding_weight_ids
            else None
        )

        all_optimizer_states = emb_module.get_optimizer_state(
            sorted_id_tensor=sorted_id_tensors,
        )

        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")

        for emb_config, sharded_weight, opt_state in zip(
            emb_table_config_copy,
            sharded_embedding_weights_by_table,
            all_optimizer_states,
        ):
            param_key = emb_config.name + ".weight"
            state[sharded_weight] = {}
            param_group["params"].append(sharded_weight)
            params[param_key] = sharded_weight
            for key, value in opt_state.items():
                opt_sharded_t = create_virtual_sharded_tensors(
                    [emb_config], [value], self._pg
                )[0]
                state[sharded_weight][f"{emb_config.name}.{key}"] = opt_sharded_t

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    def set_sharded_embedding_weight_ids(
        self, sharded_embedding_weight_ids: Optional[List[ShardedTensor]]
    ) -> None:
        self._sharded_embedding_weight_ids = sharded_embedding_weight_ids

    def _post_state_dict_hook(self, curr_state: Dict[str, Any]) -> None:
        logger.info("update optimizer state dict in state_dict_post_hook")
        embedding_weight_ids = (
            [
                sharded_t._local_shards[0].tensor
                for sharded_t in self._sharded_embedding_weight_ids
            ]
            if self._sharded_embedding_weight_ids is not None
            else None
        )
        all_optimizer_states = self._emb_module.get_optimizer_state(
            embedding_weight_ids,
            no_snapshot=False,
            should_flush=False,  # get embedding weights already flushed, no need to flush again here
        )
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")

        # The order of table_config is determined so put it as outer-loop for consistent traverse order across ranks
        for table_config, opt_states in zip(
            emb_table_config_copy,
            all_optimizer_states,
        ):
            for key, sharded_t_dict in curr_state.items():
                # update zero collision table's optimizer state
                if f".{table_config.name}.weight" in key:
                    for (_, opt_state_t), (sharded_t_k, sharded_t) in zip(
                        opt_states.items(), sharded_t_dict.items()
                    ):
                        logger.info(
                            f"update optimizer state for table {table_config.name} with state shape {opt_state_t.shape}, rank={self._my_rank}, weight_count_per_rank={self._table_name_to_weight_count_per_rank.get(table_config.name, None)}"
                        )
                        sharded_t.local_shards()[0].tensor = opt_state_t
                        create_virtual_table_local_metadata(
                            # pyre-ignore [6]
                            table_config.local_metadata,
                            opt_state_t,
                            self._my_rank,
                        )
                        for shard in sharded_t.local_shards():
                            shard.metadata = table_config.local_metadata
                        new_sharded_t = ShardedTensor._init_from_local_shards(
                            sharded_t.local_shards(),
                            None,
                            None,
                            process_group=self._pg,
                        )
                        sharded_t_dict[sharded_t_k] = new_sharded_t


class EmbeddingFusedOptimizer(FusedOptimizer):
    def __init__(  # noqa C901
        self,
        config: GroupedEmbeddingConfig,
        emb_module: SplitTableBatchedEmbeddingBagsCodegen,
        pg: Optional[dist.ProcessGroup] = None,
        create_for_table: Optional[str] = None,
        param_weight_for_table: Optional[nn.Parameter] = None,
        embedding_weights_by_table: Optional[List[torch.Tensor]] = None,
        all_optimizer_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """
        Implementation of a FusedOptimizer. Designed as a base class Embedding kernels

        create_for_table is an optional flag, which if passed in only creates the optimizer for a single table.
        This optimizer shares data with the broader optimizer (one per embedding kernel)
        and is used to share step and LR changes
        """
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = emb_module
        self._pg = pg

        @dataclass
        class ShardParams:
            optimizer_states: List[Optional[Tuple[torch.Tensor]]]
            local_metadata: List[ShardMetadata]
            embedding_weights: List[torch.Tensor]
            dtensor_metadata: List[DTensorMetadata]

        def get_optimizer_single_value_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
            table_global_shards_metadata: List[ShardMetadata] = (
                table_global_metadata.shards_metadata
            )

            table_shard_metadata_to_optimizer_shard_metadata = {}
            for offset, table_shard_metadata in enumerate(table_global_shards_metadata):
                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_metadata
                ] = ShardMetadata(
                    shard_sizes=[1],  # single value optimizer state
                    shard_offsets=[offset],  # offset increases by 1 for each shard
                    placement=table_shard_metadata.placement,
                )

            tensor_properties = TensorProperties(
                dtype=optimizer_state.dtype,
                layout=optimizer_state.layout,
                requires_grad=False,
            )
            single_value_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=torch.Size([len(table_global_shards_metadata)]),
                tensor_properties=tensor_properties,
            )

            return (
                table_shard_metadata_to_optimizer_shard_metadata,
                single_value_optimizer_st_metadata,
            )

        def get_optimizer_rowwise_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
            sharding_dim: int,
            is_grid_sharded: bool = False,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
            table_global_shards_metadata: List[ShardMetadata] = (
                table_global_metadata.shards_metadata
            )

            if sharding_dim == 1:
                # column-wise sharding
                # sort the metadata based on column offset and
                # we construct the momentum tensor in row-wise sharded way
                table_global_shards_metadata = sorted(
                    table_global_shards_metadata,
                    key=lambda shard: shard.shard_offsets[1],
                )

            table_shard_metadata_to_optimizer_shard_metadata = {}
            rolling_offset = 0
            for idx, table_shard_metadata in enumerate(table_global_shards_metadata):
                offset = table_shard_metadata.shard_offsets[0]

                if is_grid_sharded:
                    # we use a rolling offset to calculate the current offset for shard to account for uneven row wise case for our shards
                    offset = rolling_offset
                    rolling_offset += table_shard_metadata.shard_sizes[0]
                elif sharding_dim == 1:
                    # for column-wise sharding, we still create row-wise sharded metadata for optimizer
                    # manually create a row-wise offset
                    offset = idx * table_shard_metadata.shard_sizes[0]

                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_metadata
                ] = ShardMetadata(
                    shard_sizes=[table_shard_metadata.shard_sizes[0]],
                    shard_offsets=[offset],
                    placement=table_shard_metadata.placement,
                )

            tensor_properties = TensorProperties(
                dtype=optimizer_state.dtype,
                layout=optimizer_state.layout,
                requires_grad=False,
            )
            len_rw_shards = (
                len(table_shard_metadata_to_optimizer_shard_metadata)
                if sharding_dim == 1 and not is_grid_sharded
                else 1
            )
            # for grid sharding, the row dimension is replicated CW shard times
            grid_shard_nodes = (
                len(table_global_shards_metadata) // get_node_group_size()
                if is_grid_sharded
                else 1
            )
            rowwise_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=torch.Size(
                    [table_global_metadata.size[0] * len_rw_shards * grid_shard_nodes]
                ),
                tensor_properties=tensor_properties,
            )

            return (
                table_shard_metadata_to_optimizer_shard_metadata,
                rowwise_optimizer_st_metadata,
            )

        def get_optimizer_pointwise_shard_metadata_and_global_metadata(
            table_global_metadata: ShardedTensorMetadata,
            optimizer_state: torch.Tensor,
        ) -> Tuple[Dict[ShardMetadata, ShardMetadata], ShardedTensorMetadata]:
            table_global_shards_metadata: List[ShardMetadata] = (
                table_global_metadata.shards_metadata
            )

            table_shard_metadata_to_optimizer_shard_metadata = {}

            for table_shard_metadata in table_global_shards_metadata:
                table_shard_metadata_to_optimizer_shard_metadata[
                    table_shard_metadata
                ] = ShardMetadata(
                    shard_sizes=table_shard_metadata.shard_sizes,
                    shard_offsets=table_shard_metadata.shard_offsets,
                    placement=table_shard_metadata.placement,
                )
            tensor_properties = TensorProperties(
                dtype=optimizer_state.dtype,
                layout=optimizer_state.layout,
                requires_grad=False,
            )
            pointwise_optimizer_st_metadata = ShardedTensorMetadata(
                shards_metadata=list(
                    table_shard_metadata_to_optimizer_shard_metadata.values()
                ),
                size=table_global_metadata.size,
                tensor_properties=tensor_properties,
            )

            return (
                table_shard_metadata_to_optimizer_shard_metadata,
                pointwise_optimizer_st_metadata,
            )

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.get_learning_rate(),
        }

        params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}

        # Fused optimizers use buffers (they don't use autograd) and we want to make sure
        # that state_dict look identical to no-fused version.
        table_to_shard_params: Dict[str, ShardParams] = {}

        embedding_weights_by_table = (
            embedding_weights_by_table or emb_module.split_embedding_weights()
        )

        all_optimizer_states = all_optimizer_states or emb_module.get_optimizer_state()
        optimizer_states_keys_by_table: Dict[str, List[torch.Tensor]] = {}
        for (
            table_config,
            optimizer_states,
            weight,
        ) in itertools.zip_longest(
            config.embedding_tables,
            all_optimizer_states,
            embedding_weights_by_table,
        ):
            # When EmbeddingFusedOptimizer is created for composability, only create state
            if create_for_table is not None and create_for_table != table_config.name:
                continue
            if table_config.name not in table_to_shard_params:
                table_to_shard_params[table_config.name] = ShardParams(
                    optimizer_states=[],
                    local_metadata=[],
                    embedding_weights=[],
                    dtensor_metadata=[],
                )
            optimizer_state_values = None
            if optimizer_states:
                optimizer_state_values = tuple(optimizer_states.values())
                for optimizer_state_value in optimizer_state_values:
                    assert (
                        table_config.local_rows == optimizer_state_value.size(0)
                        or optimizer_state_value.nelement() == 1  # single value state
                    )
                optimizer_states_keys_by_table[table_config.name] = list(
                    optimizer_states.keys()
                )
            local_metadata = table_config.local_metadata

            table_to_shard_params[table_config.name].optimizer_states.append(
                optimizer_state_values
            )
            table_to_shard_params[table_config.name].local_metadata.append(
                local_metadata
            )
            table_to_shard_params[table_config.name].dtensor_metadata.append(
                table_config.dtensor_metadata
            )
            table_to_shard_params[table_config.name].embedding_weights.append(weight)

        seen_tables = set()
        for table_config in config.embedding_tables:
            if create_for_table is not None and create_for_table != table_config.name:
                continue
            if table_config.name in seen_tables:
                continue
            seen_tables.add(table_config.name)
            table_config_global_metadata: Optional[ShardedTensorMetadata] = (
                copy.deepcopy(table_config.global_metadata)
            )

            shard_params: ShardParams = table_to_shard_params[table_config.name]

            assert table_config_global_metadata is not None
            if create_for_table is None:
                local_weight_shards = []
                for local_weight, local_metadata in zip(
                    shard_params.embedding_weights, shard_params.local_metadata
                ):
                    local_weight_shards.append(Shard(local_weight, local_metadata))
                    table_config_global_metadata.tensor_properties.dtype = (
                        local_weight.dtype
                    )
                    table_config_global_metadata.tensor_properties.requires_grad = (
                        local_weight.requires_grad
                    )
                # TODO share this logic to create the same TableBatchedEmbeddingSlice in FusedModules below
                weight = ShardedTensor._init_from_local_shards_and_global_metadata(
                    local_shards=local_weight_shards,
                    sharded_tensor_metadata=table_config_global_metadata,
                    process_group=self._pg,
                )
                param_key = table_config.name + ".weight"
            else:
                assert (
                    param_weight_for_table is not None
                ), "param_weight_for_table cannot be None when using create_for_table"
                weight = param_weight_for_table
                param_key = ""

            state[weight] = {}
            param_group["params"].append(weight)
            params[param_key] = weight

            # Setting optimizer states
            sharding_dim: int = (
                1 if table_config.local_cols != table_config.embedding_dim else 0
            )

            is_grid_sharded: bool = (
                True
                if table_config.local_cols != table_config.embedding_dim
                and table_config.local_rows != table_config.num_embeddings
                else False
            )

            if all(
                opt_state is not None for opt_state in shard_params.optimizer_states
            ):
                # pyre-ignore
                def get_sharded_optim_state(
                    momentum_idx: int, state_key: str
                ) -> Union[ShardedTensor, DTensor]:
                    assert momentum_idx > 0
                    momentum_local_shards: List[Shard] = []
                    optimizer_sharded_tensor_metadata: ShardedTensorMetadata

                    # pyre-ignore [16]
                    optim_state = shard_params.optimizer_states[0][momentum_idx - 1]
                    if (
                        optim_state.nelement() == 1 and state_key != "momentum1"
                    ):  # special handling for backward compatibility, momentum1 is rowwise state for rowwise_adagrad
                        # single value state: one value per table
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_single_value_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            optim_state,
                        )
                    elif optim_state.dim() == 1:
                        # rowwise state: param.shape[0] == state.shape[0], state.shape[1] == 1
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_rowwise_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            optim_state,
                            sharding_dim,
                            is_grid_sharded,
                        )
                    else:
                        # pointwise state: param.shape == state.shape
                        (
                            table_shard_metadata_to_optimizer_shard_metadata,
                            optimizer_sharded_tensor_metadata,
                        ) = get_optimizer_pointwise_shard_metadata_and_global_metadata(
                            table_config.global_metadata,
                            optim_state,
                        )

                    for optimizer_state, table_shard_local_metadata in zip(
                        shard_params.optimizer_states, shard_params.local_metadata
                    ):
                        local_optimizer_shard_metadata = (
                            table_shard_metadata_to_optimizer_shard_metadata[
                                table_shard_local_metadata
                            ]
                        )
                        momentum_local_shards.append(
                            Shard(
                                optimizer_state[momentum_idx - 1],
                                local_optimizer_shard_metadata,
                            )
                        )

                    # Convert optimizer state to DTensor if enabled
                    if table_config.dtensor_metadata:
                        # if rowwise state we do Shard(0), regardless of how the table is sharded
                        if optim_state.dim() == 1:
                            stride = (1,)
                            placements = (
                                (Replicate(), DTensorShard(0))
                                if table_config.dtensor_metadata.mesh.ndim == 2
                                else (DTensorShard(0),)
                            )
                        else:
                            stride = table_config.dtensor_metadata.stride
                            placements = table_config.dtensor_metadata.placements

                        return DTensor.from_local(
                            local_tensor=LocalShardsWrapper(
                                local_shards=[x.tensor for x in momentum_local_shards],
                                local_offsets=[  # pyre-ignore[6]
                                    x.metadata.shard_offsets
                                    for x in momentum_local_shards
                                ],
                            ),
                            device_mesh=table_config.dtensor_metadata.mesh,
                            placements=placements,
                            shape=optimizer_sharded_tensor_metadata.size,
                            stride=stride,
                            run_check=False,
                        )
                    else:
                        # TODO we should be creating this in SPMD fashion (e.g. init_from_local_shards), and let it derive global metadata.
                        return ShardedTensor._init_from_local_shards_and_global_metadata(
                            local_shards=momentum_local_shards,
                            sharded_tensor_metadata=optimizer_sharded_tensor_metadata,
                            process_group=self._pg,
                        )

                num_states: int = min(
                    # pyre-ignore
                    [len(opt_state) for opt_state in shard_params.optimizer_states]
                )
                optimizer_state_keys = []
                if num_states > 0:
                    optimizer_state_keys = optimizer_states_keys_by_table[
                        table_config.name
                    ]
                for cur_state_idx in range(0, num_states):
                    if cur_state_idx == 0:
                        # for backward compatibility
                        cur_state_key = "momentum1"
                    else:
                        cur_state_key = optimizer_state_keys[cur_state_idx]

                    state[weight][f"{table_config.name}.{cur_state_key}"] = (
                        get_sharded_optim_state(cur_state_idx + 1, cur_state_key)
                    )

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    def set_optimizer_step(self, step: int) -> None:
        self._emb_module.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        self._emb_module.update_hyper_parameters(params_dict)


def _gen_named_parameters_by_table_ssd_pmt(
    emb_module: SSDTableBatchedEmbeddingBags,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
    pg: Optional[dist.ProcessGroup] = None,
) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Return an iterator over module parameters that are embedding tables, yielding both the table
    name as well as the parameter itself. The embedding table is in the form of
    PartiallyMaterializedTensor to support windowed access.
    """
    pmts, _, _, _ = emb_module.split_embedding_weights()
    for table_config, pmt in zip(config.embedding_tables, pmts):
        table_name = table_config.name
        emb_table = pmt
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got
        #  `Union[PartiallyMaterializedTensor, Tensor]`.
        weight: nn.Parameter = nn.Parameter(emb_table)
        # pyre-ignore
        weight._in_backward_optimizers = [EmptyFusedOptimizer()]
        yield (table_name, weight)


def _gen_named_parameters_by_table_fused(
    emb_module: SplitTableBatchedEmbeddingBagsCodegen,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
    pg: Optional[dist.ProcessGroup] = None,
) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
    # TODO: move logic to FBGEMM to avoid accessing fbgemm internals
    # Cache embedding_weights_by_table
    embedding_weights_by_table = emb_module.split_embedding_weights()
    # Cache all_optimizer_states
    all_optimizer_states = emb_module.get_optimizer_state()
    for t_idx, (rows, dim, location, _) in enumerate(emb_module.embedding_specs):
        table_name = config.embedding_tables[t_idx].name
        if table_name not in table_name_to_count:
            continue
        table_count = table_name_to_count.pop(table_name)
        if emb_module.weights_precision == SparseType.INT8:
            dim += emb_module.int8_emb_row_dim_offset
        # pyre-ignore [29]
        offset = emb_module.weights_physical_offsets[t_idx]
        weights: torch.Tensor
        if location == EmbeddingLocation.DEVICE.value:
            # pyre-fixme[9]: weights has type `Tensor`; used as `Union[Module, Tensor]`.
            weights = emb_module.weights_dev
        elif location == EmbeddingLocation.HOST.value:
            # pyre-fixme[9]: weights has type `Tensor`; used as `Union[Module, Tensor]`.
            weights = emb_module.weights_host
        else:
            # pyre-fixme[9]: weights has type `Tensor`; used as `Union[Module, Tensor]`.
            weights = emb_module.weights_uvm
        weight = TableBatchedEmbeddingSlice(
            data=weights,
            start_offset=offset,
            end_offset=offset + table_count * rows * dim,
            num_embeddings=-1,
            embedding_dim=dim,
        )
        # this reuses logic in EmbeddingFusedOptimizer but is per table
        # pyre-ignore
        weight._in_backward_optimizers = [
            EmbeddingFusedOptimizer(
                config=config,
                emb_module=emb_module,
                pg=pg,
                create_for_table=table_name,
                param_weight_for_table=weight,
                embedding_weights_by_table=embedding_weights_by_table,
                all_optimizer_states=all_optimizer_states,
            )
        ]
        yield (table_name, weight)


def _gen_named_parameters_by_table_dense(
    emb_module: DenseTableBatchedEmbeddingBagsCodegen,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
    # TODO: move logic to FBGEMM to avoid accessing fbgemm internals
    for t_idx, (rows, dim) in enumerate(emb_module.embedding_specs):
        table_name = config.embedding_tables[t_idx].name
        if table_name not in table_name_to_count:
            continue
        table_count = table_name_to_count.pop(table_name)
        offset = emb_module.weights_physical_offsets[t_idx]
        weight = TableBatchedEmbeddingSlice(
            data=emb_module.weights,
            start_offset=offset,
            end_offset=offset + table_count * rows * dim,
            num_embeddings=-1,
            embedding_dim=dim,
        )
        yield (table_name, weight)


SplitWeightType = TypeVar("SplitWeightType")


class BaseBatchedEmbedding(BaseEmbedding, Generic[SplitWeightType]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        self._pg = pg

        self._local_rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._embedding_dims: List[int] = []
        self._local_cols: List[int] = []
        self._row_offset: List[int] = []
        self._col_offset: List[int] = []
        self._feature_table_map: List[int] = []
        self.table_name_to_count: Dict[str, int] = {}
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = {}

        for idx, table_config in enumerate(self._config.embedding_tables):
            self._local_rows.append(table_config.local_rows)
            self._weight_init_mins.append(table_config.get_weight_init_min())
            self._weight_init_maxs.append(table_config.get_weight_init_max())
            self._num_embeddings.append(table_config.num_embeddings)
            self._embedding_dims.append(table_config.embedding_dim)
            self._row_offset.append(
                table_config.local_metadata.shard_offsets[0]
                if table_config.local_metadata
                and len(table_config.local_metadata.shard_offsets) > 0
                else 0
            )
            self._col_offset.append(
                table_config.local_metadata.shard_offsets[1]
                if table_config.local_metadata
                and len(table_config.local_metadata.shard_offsets) > 1
                else 0
            )
            self._local_cols.append(table_config.local_cols)
            self._feature_table_map.extend([idx] * table_config.num_features())
            if table_config.name not in self.table_name_to_count:
                self.table_name_to_count[table_config.name] = 0
            self.table_name_to_count[table_config.name] += 1

    def init_parameters(self) -> None:
        # initialize embedding weights
        assert len(self._num_embeddings) == len(self.split_embedding_weights())
        for rows, emb_dim, weight_init_min, weight_init_max, param in zip(
            self._local_rows,
            self._local_cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)  # pyre-ignore[16]
            if param.data.dtype in [  # pyre-ignore[16]
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]:
                tmp_param = torch.zeros(
                    param.shape, device=param.device  # pyre-ignore[16]
                )
                tmp_param.uniform_(weight_init_min, weight_init_max).to(
                    param.data.dtype
                )
                param.data.copy_(tmp_param)
            else:
                param.data.uniform_(
                    weight_init_min,
                    weight_init_max,
                )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        return self.emb_module(
            indices=features.values().long(),
            offsets=features.offsets().long(),
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        self.flush()
        return get_state_dict(
            self._config.embedding_tables,
            # pyre-ignore
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

    def split_embedding_weights(self) -> List[SplitWeightType]:
        return self.emb_module.split_embedding_weights()

    @property
    @abc.abstractmethod
    def emb_module(
        self,
    ) -> Union[
        DenseTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegen,
    ]: ...

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config

    def flush(self) -> None:
        pass

    def purge(self) -> None:
        pass

    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, param in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, param

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for name, param in self._param_per_table.items():
            yield name, param


class KeyValueEmbedding(BaseBatchedEmbedding[torch.Tensor], FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        assert (
            len(config.embedding_tables) > 0
        ), "Expected to see at least one table in SSD TBE, but found 0."
        assert (
            len({table.embedding_dim for table in config.embedding_tables}) == 1
        ), "Currently we expect all tables in SSD TBE to have the same embedding dimension."
        for table in config.embedding_tables:
            assert table.local_cols % 4 == 0, (
                f"table {table.name} has local_cols={table.local_cols} "
                "not divisible by 4. "
            )

        ssd_tbe_params = _populate_ssd_tbe_params(config)
        compute_kernel = config.embedding_tables[0].compute_kernel
        embedding_location = compute_kernel_to_embedding_location(compute_kernel)

        self._emb_module: SSDTableBatchedEmbeddingBags = SSDTableBatchedEmbeddingBags(
            embedding_specs=list(zip(self._local_rows, self._local_cols)),
            feature_table_map=self._feature_table_map,
            ssd_cache_location=embedding_location,
            pooling_mode=PoolingMode.NONE,
            **ssd_tbe_params,
        ).to(device)

        logger.info(
            f"tbe_unique_id:{self._emb_module.tbe_unique_id} => table name to count dict:{self.table_name_to_count}"
        )

        self._optim: KeyValueEmbeddingFusedOptimizer = KeyValueEmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, nn.Parameter] = dict(
            _gen_named_parameters_by_table_ssd_pmt(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        An advantage of SSD TBE is that we don't need to init weights. Hence skipping.
        """
        pass

    @property
    def emb_module(
        self,
    ) -> SSDTableBatchedEmbeddingBags:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        """
        SSD Embedding fuses backward with backward.
        """
        return self._optim

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            no_snapshot (bool): the tensors in the returned dict are
                PartiallyMaterializedTensors. this argument controls wether the
                PartiallyMaterializedTensor owns a RocksDB snapshot handle. True means the
                PartiallyMaterializedTensor doesn't have a RocksDB snapshot handle.  False means the
                PartiallyMaterializedTensor has a RocksDB snapshot handle
        """
        # in the case no_snapshot=False, a flush is required. we rely on the flush operation in
        # ShardedEmbeddingBagCollection._pre_state_dict_hook()

        emb_tables, _, _, _ = self.split_embedding_weights(no_snapshot=no_snapshot)
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")
        ret = get_state_dict(
            emb_table_config_copy,
            emb_tables,
            self._pg,
            destination,
            prefix,
        )
        return ret

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Only allowed ways to get state_dict.
        """
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            # pyre-ignore [6]
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    # pyre-ignore [15]
    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, PartiallyMaterializedTensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights()[0],
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    def get_named_split_embedding_weights_snapshot(self, prefix: str = "") -> Iterator[
        Tuple[
            str,
            Union[ShardedTensor, PartiallyMaterializedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
        ]
    ]:
        """
        Return an iterator over embedding tables, yielding both the table name as well as the embedding
        table itself. The embedding table is in the form of PartiallyMaterializedTensor with a valid
        RocksDB snapshot to support windowed access.
        optional ShardedTensor for weight_id, this won't be used here as this is non-kvzch
        optional ShardedTensor for bucket_cnt, this won't be used here as this is non-kvzch
        optional ShardedTensor for metadata, this won't be used here as this is non-kvzch
        """
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(no_snapshot=False)[0],
        ):
            key = append_prefix(prefix, f"{config.name}")
            yield key, tensor, None, None, None

    def flush(self) -> None:
        """
        Flush the embeddings in cache back to SSD. Should be pretty expensive.
        """
        self.emb_module.flush()

    def purge(self) -> None:
        """
        Reset the cache space. This is needed when we load state dict.
        """
        # TODO: move the following to SSD TBE.
        self.emb_module.lxu_cache_weights.zero_()
        self.emb_module.lxu_cache_state.fill_(-1)

    # Todo: [Raahul46]: Add a intermediate parent class between embedding and kv to support these functions
    def create_rocksdb_hard_link_snapshot(self) -> None:
        """
        Create a RocksDB checkpoint. This is needed before we call state_dict() for publish.
        """
        self.emb_module.create_rocksdb_hard_link_snapshot()

    # pyre-ignore [15]
    def split_embedding_weights(self, no_snapshot: bool = True) -> Tuple[
        List[PartiallyMaterializedTensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        # pyre-fixme[7]: Expected `Tuple[List[PartiallyMaterializedTensor],
        #  Optional[List[Tensor]], Optional[List[Tensor]]]` but got
        #  `Tuple[Union[List[PartiallyMaterializedTensor], List[Tensor]],
        #  Optional[List[Tensor]], Optional[List[Tensor]]]`.
        return self.emb_module.split_embedding_weights(no_snapshot)


class ZeroCollisionKeyValueEmbedding(
    BaseBatchedEmbedding[torch.Tensor], FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        backend_type: BackendType = BackendType.SSD,
        embedding_cache_mode: bool = False,
    ) -> None:
        super().__init__(config, pg, device)

        assert (
            len(config.embedding_tables) > 0
        ), "Expected to see at least one table in SSD TBE, but found 0."
        assert (
            len({table.embedding_dim for table in config.embedding_tables}) == 1
        ), "Currently we expect all tables in SSD TBE to have the same embedding dimension."
        assert (
            config.is_using_virtual_table
        ), "Try to create ZeroCollisionKeyValueEmbedding for non virtual tables"
        assert embedding_cache_mode == config.enable_embedding_update, (
            f"Embedding_cache kernel is {embedding_cache_mode} "
            f"but embedding config has enable_embedding_update {config.enable_embedding_update}"
        )
        for table in config.embedding_tables:
            assert table.local_cols % 4 == 0, (
                f"table {table.name} has local_cols={table.local_cols} "
                "not divisible by 4. "
            )

        ssd_tbe_params = _populate_ssd_tbe_params(config)
        self._bucket_spec: List[Tuple[int, int, int]] = (
            _get_sharded_local_buckets_for_zero_collision(
                self._config.embedding_tables, self._pg
            )
        )
        _populate_zero_collision_tbe_params(
            ssd_tbe_params,
            self._bucket_spec,
            config,
            backend_type,
            embedding_cache_mode,
        )
        self.embedding_cache_mode = embedding_cache_mode
        if ssd_tbe_params.get("kv_zch_params", None) is not None:
            self.embedding_cache_mode = ssd_tbe_params[
                "kv_zch_params"
            ].embedding_cache_mode
        compute_kernel = config.embedding_tables[0].compute_kernel
        embedding_location = compute_kernel_to_embedding_location(compute_kernel)

        # every split_embeding_weights call is expensive, since it iterates over all the elements in the backend kv db
        # use split weights result cache so that multiple calls in the same train iteration will only trigger once
        self._split_weights_res: Optional[
            Tuple[
                List[ShardedTensor],
                List[ShardedTensor],
                List[ShardedTensor],
                Optional[List[ShardedTensor]],
            ]
        ] = None

        self._emb_module: SSDTableBatchedEmbeddingBags = SSDTableBatchedEmbeddingBags(
            embedding_specs=list(zip(self._num_embeddings, self._local_cols)),
            feature_table_map=self._feature_table_map,
            ssd_cache_location=embedding_location,
            pooling_mode=PoolingMode.NONE,
            backend_type=backend_type,
            **ssd_tbe_params,
        ).to(device)

        logger.info(
            f"tbe_unique_id:{self._emb_module.tbe_unique_id} => table name to count dict:{self.table_name_to_count}"
        )
        self._table_name_to_weight_count_per_rank: Dict[str, List[int]] = {}
        self._init_sharded_split_embedding_weights()  # this will populate self._split_weights_res
        self._optim: ZeroCollisionKeyValueEmbeddingFusedOptimizer = (
            ZeroCollisionKeyValueEmbeddingFusedOptimizer(
                config,
                self._emb_module,
                # pyre-ignore[16]
                sharded_embedding_weights_by_table=self._split_weights_res[0],
                table_name_to_weight_count_per_rank=self._table_name_to_weight_count_per_rank,
                sharded_embedding_weight_ids=self._split_weights_res[1],
                pg=pg,
            )
        )
        self._param_per_table: Dict[str, nn.Parameter] = dict(
            _gen_named_parameters_by_table_ssd_pmt(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        An advantage of KV TBE is that we don't need to init weights. Hence skipping.
        """
        pass

    @property
    def emb_module(
        self,
    ) -> SSDTableBatchedEmbeddingBags:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        """
        SSD Embedding fuses backward with backward.
        """
        return self._optim

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            no_snapshot (bool): the tensors in the returned dict are
                PartiallyMaterializedTensors. this argument controls wether the
                PartiallyMaterializedTensor owns a RocksDB snapshot handle. True means the
                PartiallyMaterializedTensor doesn't have a RocksDB snapshot handle.  False means the
                PartiallyMaterializedTensor has a RocksDB snapshot handle
        """
        # in the case no_snapshot=False, a flush is required. we rely on the flush operation in
        # ShardedEmbeddingBagCollection._pre_state_dict_hook()

        emb_tables, _, _, _ = self.split_embedding_weights(no_snapshot=no_snapshot)
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")
        ret = get_state_dict(
            emb_table_config_copy,
            emb_tables,
            self._pg,
            destination,
            prefix,
        )
        return ret

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Only allowed ways to get state_dict.
        """
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            # pyre-ignore [6]
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    # pyre-ignore [15]
    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Union[PartiallyMaterializedTensor, torch.Tensor]]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights()[0],
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    # initialize sharded _split_weights_res if it's None
    # this method is used to generate sharded embedding weights once for all following state_dict
    # calls in checkpointing and publishing.
    # When training is resumed, the cached value will be reset to None and the value needs to be
    # rebuilt for next checkpointing and publishing, as the weight id, weight embedding will be updated
    # during training in backend k/v store.
    def _init_sharded_split_embedding_weights(
        self, prefix: str = "", force_regenerate: bool = False
    ) -> None:
        if not force_regenerate and self._split_weights_res is not None:
            return

        pmt_list, weight_ids_list, bucket_cnt_list, metadata_list = (
            self.split_embedding_weights(
                no_snapshot=False,
            )
        )
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            none_throws(
                none_throws(
                    emb_table.local_metadata,
                    f"local_metadata is None for emb_table: {emb_table.name}",
                ).placement,
                f"placement is None for local_metadata of emb table: {emb_table.name}",
            )._device = torch.device("cpu")

        pmt_sharded_t_list = create_virtual_sharded_tensors(
            emb_table_config_copy,
            pmt_list,
            self._pg,
            prefix,
            self._table_name_to_weight_count_per_rank,
        )
        weight_id_sharded_t_list = create_virtual_sharded_tensors(
            emb_table_config_copy,
            weight_ids_list,  # pyre-ignore [6]
            self._pg,
            prefix,
            self._table_name_to_weight_count_per_rank,
        )
        bucket_cnt_sharded_t_list = create_virtual_sharded_tensors(
            emb_table_config_copy,
            bucket_cnt_list,  # pyre-ignore [6]
            self._pg,
            prefix,
            self._table_name_to_weight_count_per_rank,
            use_param_size_as_rows=True,
        )
        metadata_sharded_t_list = None
        if metadata_list is not None:
            metadata_sharded_t_list = create_virtual_sharded_tensors(
                emb_table_config_copy,
                metadata_list,
                self._pg,
                prefix,
                self._table_name_to_weight_count_per_rank,
            )

        assert (
            len(pmt_list)
            == len(weight_ids_list)  # pyre-ignore
            == len(bucket_cnt_list)  # pyre-ignore
        )
        assert (
            len(pmt_sharded_t_list)
            == len(weight_id_sharded_t_list)
            == len(bucket_cnt_sharded_t_list)
        )
        if metadata_list is not None:
            assert metadata_sharded_t_list is not None
            assert len(pmt_list) == len(metadata_list)
            assert len(pmt_sharded_t_list) == len(metadata_sharded_t_list)

        self._split_weights_res = (
            pmt_sharded_t_list,
            weight_id_sharded_t_list,
            bucket_cnt_sharded_t_list,
            metadata_sharded_t_list,
        )

    def get_named_split_embedding_weights_snapshot(self, prefix: str = "") -> Iterator[
        Tuple[
            str,
            Union[ShardedTensor, PartiallyMaterializedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
        ]
    ]:
        """
        Return an iterator over embedding tables, for each table yielding
        table name,
        PMT for embedding table with a valid RocksDB snapshot to support tensor IO
        optional ShardedTensor for weight_id
        optional ShardedTensor for bucket_cnt
        optional ShardedTensor for metadata
        """
        self._init_sharded_split_embedding_weights()
        # pyre-ignore[16]
        self._optim.set_sharded_embedding_weight_ids(self._split_weights_res[1])

        pmt_sharded_t_list = self._split_weights_res[0]
        weight_id_sharded_t_list = self._split_weights_res[1]
        bucket_cnt_sharded_t_list = self._split_weights_res[2]
        metadata_sharded_t_list = self._split_weights_res[3]
        for table_idx, pmt_sharded_t in enumerate(pmt_sharded_t_list):
            table_config = self._config.embedding_tables[table_idx]
            key = append_prefix(prefix, f"{table_config.name}")
            metadata_sharded_t = None
            if metadata_sharded_t_list is not None:
                metadata_sharded_t = metadata_sharded_t_list[table_idx]

            yield key, pmt_sharded_t, weight_id_sharded_t_list[
                table_idx
            ], bucket_cnt_sharded_t_list[table_idx], metadata_sharded_t

    def flush(self) -> None:
        """
        Flush the embeddings in cache back to SSD. Should be pretty expensive.
        """
        self.emb_module.flush()

    def purge(self) -> None:
        """
        Reset the cache space. This is needed when we load state dict.
        """
        # TODO: move the following to SSD TBE.
        self.emb_module.lxu_cache_weights.zero_()
        self.emb_module.lxu_cache_state.fill_(-1)

    def create_rocksdb_hard_link_snapshot(self) -> None:
        """
        Create a RocksDB checkpoint. This is needed before we call state_dict() for publish.
        """
        self.emb_module.create_rocksdb_hard_link_snapshot()

    # pyre-ignore [15]
    def split_embedding_weights(
        self, no_snapshot: bool = True, should_flush: bool = False
    ) -> Tuple[
        Union[List[PartiallyMaterializedTensor], List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        return self.emb_module.split_embedding_weights(no_snapshot, should_flush)

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        # reset split weights during training
        self._split_weights_res = None
        self._optim.set_sharded_embedding_weight_ids(sharded_embedding_weight_ids=None)

        return self.emb_module(
            indices=features.values().long(),
            offsets=features.offsets().long(),
            weights=features.weights_or_none(),
        )


class ZeroCollisionEmbeddingCache(ZeroCollisionKeyValueEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        backend_type: BackendType = BackendType.SSD,
    ) -> None:
        super().__init__(
            config,
            pg,
            device,
            backend_type,
            True,  # embedding_cache_mode
        )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        # in the case of embedding_cache_mode, we don't need backward pass, so call forward in no_grad mode
        with torch.no_grad():
            return super().forward(features)

    def update(self, embeddings: KeyedJaggedTensor) -> None:
        """
        Update the embedding table with the new embeddings.
        """
        self.emb_module.direct_write_embedding(
            embeddings.values(), embeddings.offsets(), embeddings.weights()
        )


class BatchedFusedEmbedding(BaseBatchedEmbedding[torch.Tensor], FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)

        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision

        enable_res, res_params = _populate_res_params(config)
        fused_params[ENABLE_RAW_EMBEDDING_STREAMING_STR] = enable_res

        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=PoolingMode.NONE,
                weights_precision=weights_precision,
                device=device,
                table_names=[t.name for t in config.embedding_tables],
                embedding_shard_info=list(
                    zip(
                        self._num_embeddings,
                        self._embedding_dims,
                        self._row_offset,
                        self._col_offset,
                    )
                ),
                res_params=res_params,
                **fused_params,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self.init_parameters()

    @property
    def _param_per_table(self) -> Dict[str, TableBatchedEmbeddingSlice]:
        return dict(
            _gen_named_parameters_by_table_fused(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=self._pg,
            )
        )

    @_param_per_table.setter
    def _param_per_table(self, v: Dict[str, TableBatchedEmbeddingSlice]) -> None:
        self.__dict__["_param_per_table"] = v

    @property
    def emb_module(
        self,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        # TODO can delete this override once SEA is removed
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after SEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()

    def purge(self) -> None:
        self._emb_module.reset_cache_states()


class BatchedDenseEmbedding(BaseBatchedEmbedding[torch.Tensor]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        output_dtype = fused_params.get("output_dtype", SparseType.FP32)
        use_cpu: bool = (
            device is None
            or device.type == "cpu"
            or (not (torch.cuda.is_available() or torch.mtia.is_available()))
        )
        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=PoolingMode.NONE,
                use_cpu=use_cpu,
                weights_precision=weights_precision,
                output_dtype=output_dtype,
                use_mtia=device is not None and device.type == "mtia",
            )
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_dense(
                self._emb_module, self.table_name_to_count.copy(), self._config
            )
        )
        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> DenseTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join(
            [config.name for config in self._config.embedding_tables]
        )
        yield append_prefix(prefix, f"{combined_key}.weight"), cast(
            nn.Parameter, self._emb_module.weights
        )


class BaseBatchedEmbeddingBag(BaseEmbedding, Generic[SplitWeightType]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        self._pg = pg

        self._pooling: PoolingMode = pooling_type_to_pooling_mode(
            config.pooling, sharding_type  # pyre-ignore[6]
        )

        self._local_rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._embedding_dims: List[int] = []
        self._local_cols: List[int] = []
        self._row_offset: List[int] = []
        self._col_offset: List[int] = []
        self._feature_table_map: List[int] = []
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []
        self.table_name_to_count: Dict[str, int] = {}
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = {}

        for idx, table_config in enumerate(self._config.embedding_tables):
            self._local_rows.append(table_config.local_rows)
            self._weight_init_mins.append(table_config.get_weight_init_min())
            self._weight_init_maxs.append(table_config.get_weight_init_max())
            self._num_embeddings.append(table_config.num_embeddings)
            self._embedding_dims.append(table_config.embedding_dim)
            self._row_offset.append(
                table_config.local_metadata.shard_offsets[0]
                if table_config.local_metadata
                and len(table_config.local_metadata.shard_offsets) > 0
                else 0
            )
            self._col_offset.append(
                table_config.local_metadata.shard_offsets[1]
                if table_config.local_metadata
                and len(table_config.local_metadata.shard_offsets) > 1
                else 0
            )
            self._local_cols.append(table_config.local_cols)
            self._feature_table_map.extend([idx] * table_config.num_features())
            if table_config.name not in self.table_name_to_count:
                self.table_name_to_count[table_config.name] = 0
            self.table_name_to_count[table_config.name] += 1

    def init_parameters(self) -> None:
        # initialize embedding weights
        assert len(self._num_embeddings) == len(self.split_embedding_weights())
        for rows, emb_dim, weight_init_min, weight_init_max, param in zip(
            self._local_rows,
            self._local_cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)  # pyre-ignore[16]
            if param.data.dtype in [  # pyre-ignore[16]
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]:
                tmp_param = torch.zeros(
                    param.shape, device=param.device  # pyre-ignore[16]
                )
                tmp_param.uniform_(weight_init_min, weight_init_max).to(
                    param.data.dtype
                )
                param.data.copy_(tmp_param)
            else:
                param.data.uniform_(
                    weight_init_min,
                    weight_init_max,
                )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        weights = features.weights_or_none()
        if weights is not None and not torch.is_floating_point(weights):
            weights = None
        if features.variable_stride_per_key() and isinstance(
            self.emb_module,
            (
                SplitTableBatchedEmbeddingBagsCodegen,
                DenseTableBatchedEmbeddingBagsCodegen,
                SSDTableBatchedEmbeddingBags,
            ),
        ):
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                per_sample_weights=weights,
                batch_size_per_feature_per_rank=features.stride_per_key_per_rank(),
            )
        else:
            return self.emb_module(
                indices=features.values().long(),
                offsets=features.offsets().long(),
                per_sample_weights=weights,
            )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        self.flush()
        return get_state_dict(
            self._config.embedding_tables,
            # pyre-ignore
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

    def split_embedding_weights(self) -> List[SplitWeightType]:
        return self.emb_module.split_embedding_weights()

    @property
    @abc.abstractmethod
    def emb_module(
        self,
    ) -> Union[
        DenseTableBatchedEmbeddingBagsCodegen,
        SplitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegen,
    ]: ...

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config

    def flush(self) -> None:
        pass

    def purge(self) -> None:
        pass

    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    def named_parameters_by_table(
        self,
    ) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
        """
        Like named_parameters(), but yields table_name and embedding_weights which are wrapped in TableBatchedEmbeddingSlice.
        For a single table with multiple shards (i.e CW) these are combined into one table/weight.
        Used in composability.
        """
        for name, param in self._param_per_table.items():
            yield name, param


class KeyValueEmbeddingBag(BaseBatchedEmbeddingBag[torch.Tensor], FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        assert (
            len(config.embedding_tables) > 0
        ), "Expected to see at least one table in SSD TBE, but found 0."
        assert (
            len({table.embedding_dim for table in config.embedding_tables}) == 1
        ), "Currently we expect all tables in SSD TBE to have the same embedding dimension."
        for table in config.embedding_tables:
            assert table.local_cols % 4 == 0, (
                f"table {table.name} has local_cols={table.local_cols} "
                "not divisible by 4. "
            )

        ssd_tbe_params = _populate_ssd_tbe_params(config)
        compute_kernel = config.embedding_tables[0].compute_kernel
        embedding_location = compute_kernel_to_embedding_location(compute_kernel)

        self._emb_module: SSDTableBatchedEmbeddingBags = SSDTableBatchedEmbeddingBags(
            embedding_specs=list(zip(self._local_rows, self._local_cols)),
            feature_table_map=self._feature_table_map,
            ssd_cache_location=embedding_location,
            pooling_mode=self._pooling,
            **ssd_tbe_params,
        ).to(device)

        logger.info(
            f"tbe_unique_id:{self._emb_module.tbe_unique_id} => table name to count dict:{self.table_name_to_count}"
        )

        self._optim: KeyValueEmbeddingFusedOptimizer = KeyValueEmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self._param_per_table: Dict[str, nn.Parameter] = dict(
            _gen_named_parameters_by_table_ssd_pmt(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        An advantage of SSD TBE is that we don't need to init weights. Hence
        skipping.
        """
        pass

    @property
    def emb_module(
        self,
    ) -> SSDTableBatchedEmbeddingBags:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        """
        SSD Embedding fuses backward with backward.
        """
        return self._optim

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            no_snapshot (bool): the tensors in the returned dict are
                PartiallyMaterializedTensors. this argument controls wether the
                PartiallyMaterializedTensor owns a RocksDB snapshot handle. True means the
                PartiallyMaterializedTensor doesn't have a RocksDB snapshot handle.  False means the
                PartiallyMaterializedTensor has a RocksDB snapshot handle
        """
        # in the case no_snapshot=False, a flush is required. we rely on the flush operation in
        # ShardedEmbeddingBagCollection._pre_state_dict_hook()

        emb_tables, _, _, _ = self.split_embedding_weights(no_snapshot=no_snapshot)
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")
        ret = get_state_dict(
            emb_table_config_copy,
            emb_tables,
            self._pg,
            destination,
            prefix,
        )
        return ret

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Only allowed ways to get state_dict.
        """
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            # pyre-ignore [6]
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    # pyre-ignore [15]
    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, PartiallyMaterializedTensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights()[0],
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    def get_named_split_embedding_weights_snapshot(self, prefix: str = "") -> Iterator[
        Tuple[
            str,
            Union[ShardedTensor, PartiallyMaterializedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
        ]
    ]:
        """
        Return an iterator over embedding tables, yielding both the table name as well as the embedding
        table itself. The embedding table is in the form of PartiallyMaterializedTensor with a valid
        RocksDB snapshot to support windowed access.
        optional ShardedTensor for weight_id, this won't be used here as this is non-kvzch
        optional ShardedTensor for bucket_cnt, this won't be used here as this is non-kvzch
        optional ShardedTensor for metadata, this won't be used here as this is non-kvzch
        """
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights(no_snapshot=False)[0],
        ):
            key = append_prefix(prefix, f"{config.name}")
            yield key, tensor, None, None, None

    def flush(self) -> None:
        """
        Flush the embeddings in cache back to SSD. Should be pretty expensive.
        """
        self.emb_module.flush()

    def purge(self) -> None:
        """
        Reset the cache space. This is needed when we load state dict.
        """
        # TODO: move the following to SSD TBE.
        self.emb_module.lxu_cache_weights.zero_()
        self.emb_module.lxu_cache_state.fill_(-1)

    def create_rocksdb_hard_link_snapshot(self) -> None:
        """
        Create a RocksDB checkpoint. This is needed before we call state_dict() for publish.
        """
        self.emb_module.create_rocksdb_hard_link_snapshot()

    # pyre-ignore [15]
    def split_embedding_weights(self, no_snapshot: bool = True) -> Tuple[
        List[PartiallyMaterializedTensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        # pyre-fixme[7]: Expected `Tuple[List[PartiallyMaterializedTensor],
        #  Optional[List[Tensor]], Optional[List[Tensor]]]` but got
        #  `Tuple[Union[List[PartiallyMaterializedTensor], List[Tensor]],
        #  Optional[List[Tensor]], Optional[List[Tensor]]]`.
        return self.emb_module.split_embedding_weights(no_snapshot)


class ZeroCollisionKeyValueEmbeddingBag(
    BaseBatchedEmbeddingBag[torch.Tensor], FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
        backend_type: BackendType = BackendType.SSD,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        assert (
            len(config.embedding_tables) > 0
        ), "Expected to see at least one table in SSD TBE, but found 0."
        assert (
            len({table.embedding_dim for table in config.embedding_tables}) == 1
        ), "Currently we expect all tables in SSD TBE to have the same embedding dimension."
        assert (
            config.is_using_virtual_table
        ), "Try to create ZeroCollisionKeyValueEmbeddingBag for non virtual tables"

        for table in config.embedding_tables:
            assert table.local_cols % 4 == 0, (
                f"table {table.name} has local_cols={table.local_cols} "
                "not divisible by 4. "
            )

        ssd_tbe_params = _populate_ssd_tbe_params(config)
        self._bucket_spec: List[Tuple[int, int, int]] = (
            _get_sharded_local_buckets_for_zero_collision(
                self._config.embedding_tables, self._pg
            )
        )
        _populate_zero_collision_tbe_params(
            ssd_tbe_params, self._bucket_spec, config, backend_type
        )
        compute_kernel = config.embedding_tables[0].compute_kernel
        embedding_location = compute_kernel_to_embedding_location(compute_kernel)

        # every split_embeding_weights call is expensive, since it iterates over all the elements in the backend kv db
        # use split weights result cache so that multiple calls in the same train iteration will only trigger once
        self._split_weights_res: Optional[
            Tuple[
                List[ShardedTensor],
                List[ShardedTensor],
                List[ShardedTensor],
                Optional[List[ShardedTensor]],
            ]
        ] = None

        self._emb_module: SSDTableBatchedEmbeddingBags = SSDTableBatchedEmbeddingBags(
            embedding_specs=list(zip(self._num_embeddings, self._local_cols)),
            feature_table_map=self._feature_table_map,
            ssd_cache_location=embedding_location,
            pooling_mode=self._pooling,
            backend_type=backend_type,
            **ssd_tbe_params,
        ).to(device)

        logger.info(
            f"tbe_unique_id:{self._emb_module.tbe_unique_id} => table name to count dict:{self.table_name_to_count}"
        )
        self._table_name_to_weight_count_per_rank: Dict[str, List[int]] = {}
        self._init_sharded_split_embedding_weights()  # this will populate self._split_weights_res
        self._optim: ZeroCollisionKeyValueEmbeddingFusedOptimizer = (
            ZeroCollisionKeyValueEmbeddingFusedOptimizer(
                config,
                self._emb_module,
                # pyre-ignore[16]
                sharded_embedding_weights_by_table=self._split_weights_res[0],
                table_name_to_weight_count_per_rank=self._table_name_to_weight_count_per_rank,
                sharded_embedding_weight_ids=self._split_weights_res[1],
                pg=pg,
            )
        )
        self._param_per_table: Dict[str, nn.Parameter] = dict(
            _gen_named_parameters_by_table_ssd_pmt(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        An advantage of KV TBE is that we don't need to init weights. Hence skipping.
        """
        pass

    @property
    def emb_module(
        self,
    ) -> SSDTableBatchedEmbeddingBags:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        """
        SSD Embedding fuses backward with backward.
        """
        return self._optim

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        no_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            no_snapshot (bool): the tensors in the returned dict are
                PartiallyMaterializedTensors. this argument controls wether the
                PartiallyMaterializedTensor owns a RocksDB snapshot handle. True means the
                PartiallyMaterializedTensor doesn't have a RocksDB snapshot handle.  False means the
                PartiallyMaterializedTensor has a RocksDB snapshot handle
        """
        # in the case no_snapshot=False, a flush is required. we rely on the flush operation in
        # ShardedEmbeddingBagCollection._pre_state_dict_hook()

        emb_tables, _, _, _ = self.split_embedding_weights(no_snapshot=no_snapshot)
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            emb_table.local_metadata.placement._device = torch.device("cpu")
        ret = get_state_dict(
            emb_table_config_copy,
            emb_tables,
            self._pg,
            destination,
            prefix,
        )
        return ret

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Only allowed ways to get state_dict.
        """
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            # pyre-ignore [6]
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    # pyre-ignore [15]
    def named_split_embedding_weights(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Union[PartiallyMaterializedTensor, torch.Tensor]]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in BaseBatchedEmbedding.named_split_embedding_weights"
        for config, tensor in zip(
            self._config.embedding_tables,
            self.split_embedding_weights()[0],
        ):
            key = append_prefix(prefix, f"{config.name}.weight")
            yield key, tensor

    # initialize sharded _split_weights_res if it's None
    # this method is used to generate sharded embedding weights once for all following state_dict
    # calls in checkpointing and publishing.
    # When training is resumed, the cached value will be reset to None and the value needs to be
    # rebuilt for next checkpointing and publishing, as the weight id, weight embedding will be updated
    # during training in backend k/v store.
    def _init_sharded_split_embedding_weights(
        self, prefix: str = "", force_regenerate: bool = False
    ) -> None:
        if not force_regenerate and self._split_weights_res is not None:
            return

        pmt_list, weight_ids_list, bucket_cnt_list, metadata_list = (
            self.split_embedding_weights(
                no_snapshot=False,
            )
        )
        emb_table_config_copy = copy.deepcopy(self._config.embedding_tables)
        for emb_table in emb_table_config_copy:
            none_throws(
                none_throws(
                    emb_table.local_metadata,
                    f"local_metadata is None for emb_table: {emb_table.name}",
                ).placement,
                f"placement is None for local_metadata of emb table: {emb_table.name}",
            )._device = torch.device("cpu")

        pmt_sharded_t_list = create_virtual_sharded_tensors(
            emb_table_config_copy,
            pmt_list,
            self._pg,
            prefix,
            self._table_name_to_weight_count_per_rank,
        )
        weight_id_sharded_t_list = create_virtual_sharded_tensors(
            emb_table_config_copy,
            weight_ids_list,  # pyre-ignore [6]
            self._pg,
            prefix,
            self._table_name_to_weight_count_per_rank,
        )
        bucket_cnt_sharded_t_list = create_virtual_sharded_tensors(
            emb_table_config_copy,
            bucket_cnt_list,  # pyre-ignore [6]
            self._pg,
            prefix,
            self._table_name_to_weight_count_per_rank,
            use_param_size_as_rows=True,
        )
        metadata_sharded_t_list = None
        if metadata_list is not None:
            metadata_sharded_t_list = create_virtual_sharded_tensors(
                emb_table_config_copy,
                metadata_list,
                self._pg,
                prefix,
                self._table_name_to_weight_count_per_rank,
            )

        assert (
            len(pmt_list)
            == len(weight_ids_list)  # pyre-ignore
            == len(bucket_cnt_list)  # pyre-ignore
        )
        assert (
            len(pmt_sharded_t_list)
            == len(weight_id_sharded_t_list)
            == len(bucket_cnt_sharded_t_list)
        )
        if metadata_list is not None:
            assert metadata_sharded_t_list is not None
            assert len(pmt_list) == len(metadata_list)
            assert len(pmt_sharded_t_list) == len(metadata_sharded_t_list)

        self._split_weights_res = (
            pmt_sharded_t_list,
            weight_id_sharded_t_list,
            bucket_cnt_sharded_t_list,
            metadata_sharded_t_list,
        )

    def get_named_split_embedding_weights_snapshot(self, prefix: str = "") -> Iterator[
        Tuple[
            str,
            Union[ShardedTensor, PartiallyMaterializedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
            Optional[ShardedTensor],
        ]
    ]:
        """
        Return an iterator over embedding tables, for each table yielding
        table name,
        PMT for embedding table with a valid RocksDB snapshot to support tensor IO
        optional ShardedTensor for weight_id
        optional ShardedTensor for bucket_cnt
        optional ShardedTensor for metadata
        """
        self._init_sharded_split_embedding_weights()
        # pyre-ignore[16]
        self._optim.set_sharded_embedding_weight_ids(self._split_weights_res[1])

        pmt_sharded_t_list = self._split_weights_res[0]
        weight_id_sharded_t_list = self._split_weights_res[1]
        bucket_cnt_sharded_t_list = self._split_weights_res[2]
        metadata_sharded_t_list = self._split_weights_res[3]
        for table_idx, pmt_sharded_t in enumerate(pmt_sharded_t_list):
            table_config = self._config.embedding_tables[table_idx]
            key = append_prefix(prefix, f"{table_config.name}")
            metadata_sharded_t = None
            if metadata_sharded_t_list is not None:
                metadata_sharded_t = metadata_sharded_t_list[table_idx]

            yield key, pmt_sharded_t, weight_id_sharded_t_list[
                table_idx
            ], bucket_cnt_sharded_t_list[table_idx], metadata_sharded_t

    def flush(self) -> None:
        """
        Flush the embeddings in cache back to SSD. Should be pretty expensive.
        """
        self.emb_module.flush()

    def purge(self) -> None:
        """
        Reset the cache space. This is needed when we load state dict.
        """
        # TODO: move the following to SSD TBE.
        self.emb_module.lxu_cache_weights.zero_()
        self.emb_module.lxu_cache_state.fill_(-1)

    def create_rocksdb_hard_link_snapshot(self) -> None:
        """
        Create a RocksDB checkpoint. This is needed before we call state_dict() for publish.
        """
        self.emb_module.create_rocksdb_hard_link_snapshot()

    # pyre-ignore [15]
    def split_embedding_weights(
        self, no_snapshot: bool = True, should_flush: bool = False
    ) -> Tuple[
        Union[List[PartiallyMaterializedTensor], List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        return self.emb_module.split_embedding_weights(no_snapshot, should_flush)

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        # reset split weights during training
        self._split_weights_res = None
        self._optim.set_sharded_embedding_weight_ids(sharded_embedding_weight_ids=None)

        return super().forward(features)


class BatchedFusedEmbeddingBag(
    BaseBatchedEmbeddingBag[torch.Tensor], FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        managed: List[EmbeddingLocation] = []
        compute_devices: List[ComputeDevice] = []
        for table in config.embedding_tables:
            assert table.local_cols % 4 == 0, (
                f"table {table.name} has local_cols={table.local_cols} "
                "not divisible by 4. "
            )
            if device is not None and device.type == "cuda":
                compute_devices.append(ComputeDevice.CUDA)
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            elif device is not None and device.type == "mtia":
                compute_devices.append(ComputeDevice.MTIA)
                # Set EmbeddingLocation.HOST to make embedding op in FBGEMM choose CPU path.
                # But the tensor will still be created on MTIA with device type "mtia".
                managed.append(EmbeddingLocation.HOST)
            else:
                compute_devices.append(ComputeDevice.CPU)
                managed.append(EmbeddingLocation.HOST)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        if "cache_precision" not in fused_params:
            fused_params["cache_precision"] = weights_precision
            if weights_precision == SparseType.NFP8:
                fused_params["cache_precision"] = SparseType.FP16

        enable_res, res_params = _populate_res_params(config)
        fused_params[ENABLE_RAW_EMBEDDING_STREAMING_STR] = enable_res

        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=list(
                    zip(self._local_rows, self._local_cols, managed, compute_devices)
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                weights_precision=weights_precision,
                device=device,
                table_names=[t.name for t in config.embedding_tables],
                embedding_shard_info=list(
                    zip(
                        self._num_embeddings,
                        self._embedding_dims,
                        self._row_offset,
                        self._col_offset,
                    )
                ),
                res_params=res_params,
                **fused_params,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            config,
            self._emb_module,
            pg,
        )
        self.init_parameters()

    @property
    def _param_per_table(self) -> Dict[str, TableBatchedEmbeddingSlice]:
        return dict(
            _gen_named_parameters_by_table_fused(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=self._pg,
            )
        )

    @_param_per_table.setter
    def _param_per_table(self, v: Dict[str, TableBatchedEmbeddingSlice]) -> None:
        self.__dict__["_param_per_table"] = v

    @property
    def emb_module(
        self,
    ) -> SplitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        # TODO can delete this override once SEA is removed
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()

    def purge(self) -> None:
        self._emb_module.reset_cache_states()


class BatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag[torch.Tensor]):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        sharding_type: Optional[ShardingType] = None,
    ) -> None:
        super().__init__(config, pg, device, sharding_type)

        weights_precision = data_type_to_sparse_type(config.data_type)
        fused_params = config.fused_params or {}
        output_dtype = fused_params.get("output_dtype", SparseType.FP32)
        use_cpu: bool = (
            device is None
            or device.type == "cpu"
            or (not (torch.cuda.is_available() or torch.mtia.is_available()))
        )
        self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
            DenseTableBatchedEmbeddingBagsCodegen(
                list(zip(self._local_rows, self._local_cols)),
                feature_table_map=self._feature_table_map,
                pooling_mode=self._pooling,
                use_cpu=use_cpu,
                weights_precision=weights_precision,
                output_dtype=output_dtype,
                use_mtia=device is not None and device.type == "mtia",
            )
        )
        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_dense(
                self._emb_module, self.table_name_to_count.copy(), self._config
            )
        )
        self.init_parameters()

    @property
    def emb_module(
        self,
    ) -> DenseTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join(
            [config.name for config in self._config.embedding_tables]
        )
        yield append_prefix(prefix, f"{combined_key}.weight"), cast(
            nn.Parameter, self._emb_module.weights
        )
