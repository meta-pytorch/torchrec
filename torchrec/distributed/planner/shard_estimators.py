#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math
from math import ceil
from typing import cast, Dict, List, Optional, Tuple, Type

import torch
import torchrec.optim as trec_optim
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import (
    BATCHED_COPY_PERF_FACTOR,
    BIGINT_DTYPE,
    DP_ELEMENTWISE_KERNELS_PERF_FACTOR,
    FULL_BLOCK_EMB_DIM,
    HALF_BLOCK_PENALTY,
    kernel_bw_lookup,
    KV_CACHING_RATIO,
    NUM_POOLINGS,
    QUARTER_BLOCK_PENALTY,
    UVM_CACHING_RATIO,
    WEIGHTED_KERNEL_MULTIPLIER,
)
from torchrec.distributed.planner.types import (
    CollectiveType,
    GeneralizedCommsBandwidth,
    ParameterConstraints,
    Perf,
    PlannerError,
    ShardEstimator,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import prod, sharder_name
from torchrec.distributed.types import (
    CacheStatistics,
    CommOp,
    KeyValueParams,
    ModuleSharder,
    PipelineType,
    ShardingType,
)
from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS

from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface

logger: logging.Logger = logging.getLogger(__name__)


def _is_prefetch_pipelined(
    sharding_option: ShardingOption, sharder: ModuleSharder[nn.Module]
) -> bool:
    prefetch_pipeline = (
        sharding_option.cache_params.prefetch_pipeline
        if sharding_option.cache_params
        else None
    )
    # TODO: remove after deprecating fused_params in sharder
    if not prefetch_pipeline:
        prefetch_pipeline = (
            sharder.fused_params.get("prefetch_pipeline", False)  # pyre-ignore[16]
            if hasattr(sharder, "fused_params") and sharder.fused_params
            else False
        )
    return prefetch_pipeline


class EmbeddingPerfEstimator(ShardEstimator):
    """
    Embedding Wall Time Perf Estimator. This estimator estimates the wall time
    of a given sharding option.

    Args:
        topology (Topology): device topology.
        constraints (Optional[Dict[str, ParameterConstraints]]): parameter constraints.
        is_inference (bool): whether or not the estimator is used for inference.
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        is_inference: bool = False,
    ) -> None:
        self._topology = topology
        self._constraints = constraints
        self._is_inference = is_inference

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        """
        Estimates the wall time of a given sharding option.

        Args:
            sharding_options (List[ShardingOption]): list of sharding options.
            sharder_map (Optional[Dict[str, ModuleSharder[nn.Module]]]): sharder map.
        """
        if not sharder_map:
            assert not sharding_options, "sharder_map not provided for sharding_options"
            return
        num_feature_processors = 0
        for sharding_option in sharding_options:
            sharder_key = sharder_name(type(sharding_option.module[1]))
            sharder = sharder_map[sharder_key]

            caching_ratio = sharding_option.cache_load_factor
            # TODO: remove after deprecating fused_params in sharder
            if caching_ratio is None:
                caching_ratio = (
                    sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                    if hasattr(sharder, "fused_params") and sharder.fused_params
                    else None
                )

            num_poolings = get_num_poolings(self._constraints, sharding_option)
            batch_sizes = (
                cast(List[int], self._constraints[sharding_option.name].batch_sizes)
                if self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].batch_sizes
                else [sharding_option.batch_size] * sharding_option.num_inputs
            )

            assert (
                len(sharding_option.input_lengths)
                == len(num_poolings)
                == len(batch_sizes)
            ), "Provided `pooling_factors`, `num_poolings`, and `batch_sizes` constraints must match."

            module = sharding_option.module[1]

            # TODO remove this hack once feature processor is disaggregated
            has_feature_processor = False
            if (
                hasattr(module, "_feature_processor")
                and hasattr(module._feature_processor, "feature_processor_modules")
                and isinstance(
                    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                    #  attribute `feature_processor_modules`.
                    module._feature_processor.feature_processor_modules,
                    nn.ModuleDict,
                )
                and sharding_option.name
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `feature_processor_modules`.
                in module._feature_processor.feature_processor_modules.keys()
            ):
                has_feature_processor = True
                num_feature_processors += 1

            if isinstance(module, EmbeddingBagCollectionInterface):
                is_weighted = module.is_weighted()
            elif (
                self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].is_weighted
            ):
                is_weighted = self._constraints[sharding_option.name].is_weighted
            else:
                is_weighted = False

            # TODO remove this once migrate away from PEA
            is_weighted = is_weighted or has_feature_processor
            sharding_option.is_weighted = is_weighted

            table_data_type_size = sharding_option.tensor.element_size()
            (
                fwd_a2a_comm_data_type_size,
                bwd_a2a_comm_data_type_size,
                fwd_sr_comm_data_type_size,
                bwd_sr_comm_data_type_size,
            ) = _extract_comm_data_type_size(sharder, sharding_option)

            prefetch_pipeline = _is_prefetch_pipelined(sharding_option, sharder)

            # hardcoded as 8 bytes
            # input indices can be of int32, but in TBE they get converted to int64 anyway
            input_data_type_size = BIGINT_DTYPE
            output_data_type_size: float = (
                DATA_TYPE_NUM_BITS[sharding_option.output_dtype] / 8
                if sharding_option.output_dtype
                else sharding_option.tensor.element_size()
            )

            expected_cache_fetches = 0
            if (
                caching_ratio is not None
                and sharding_option.cache_params is not None
                and sharding_option.cache_params.stats is not None
                and sharding_option.compute_kernel
                == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
            ):
                _stats = sharding_option.cache_params.stats
                expected_cache_fetches = (
                    _stats.expected_miss_rate(caching_ratio) * _stats.expected_lookups
                )
                # Note, the above is not correct for data-parallel. stats.expected_lookups is
                # calculated by estimating the cardinality of a global batch size worth of data.
                # But for data-parallel, we need the calculate the cardinality of the local
                # input batch.  For now, we don't use cache stats with data parallel.

            shard_perfs = self.perf_func_emb_wall_time(
                shard_sizes=[shard.size for shard in sharding_option.shards],
                compute_kernel=sharding_option.compute_kernel,
                compute_device=self._topology.compute_device,
                sharding_type=sharding_option.sharding_type,
                batch_sizes=batch_sizes,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                input_data_type_size=input_data_type_size,
                table_data_type_size=table_data_type_size,
                output_data_type_size=output_data_type_size,
                fwd_a2a_comm_data_type_size=fwd_a2a_comm_data_type_size,
                bwd_a2a_comm_data_type_size=bwd_a2a_comm_data_type_size,
                fwd_sr_comm_data_type_size=fwd_sr_comm_data_type_size,
                bwd_sr_comm_data_type_size=bwd_sr_comm_data_type_size,
                num_poolings=num_poolings,
                hbm_mem_bw=self._topology.hbm_mem_bw,
                ddr_mem_bw=self._topology.ddr_mem_bw,
                hbm_to_ddr_mem_bw=self._topology.hbm_to_ddr_mem_bw,
                comms_bandwidths=self._topology.comms_bandwidths,
                bwd_compute_multiplier=self._topology.bwd_compute_multiplier,
                weighted_feature_bwd_compute_multiplier=self._topology.weighted_feature_bwd_compute_multiplier,
                is_pooled=sharding_option.is_pooled,
                is_weighted=is_weighted,
                is_inference=self._is_inference,
                caching_ratio=caching_ratio,
                prefetch_pipeline=prefetch_pipeline,
                expected_cache_fetches=expected_cache_fetches,
                uneven_sharding_perf_multiplier=self._topology.uneven_sharding_perf_multiplier,
            )

            for shard, perf in zip(sharding_option.shards, shard_perfs):
                shard.perf = perf

        logger.info(f"Total {num_feature_processors} feature processor.")

    @classmethod
    def perf_func_emb_wall_time(
        cls,
        shard_sizes: List[List[int]],
        compute_kernel: str,
        compute_device: str,
        sharding_type: str,
        batch_sizes: List[int],
        world_size: int,
        local_world_size: int,
        input_lengths: List[float],
        input_data_type_size: float,
        table_data_type_size: float,
        output_data_type_size: float,
        fwd_a2a_comm_data_type_size: float,
        bwd_a2a_comm_data_type_size: float,
        fwd_sr_comm_data_type_size: float,
        bwd_sr_comm_data_type_size: float,
        num_poolings: List[float],
        hbm_mem_bw: float,
        ddr_mem_bw: float,
        hbm_to_ddr_mem_bw: float,
        comms_bandwidths: GeneralizedCommsBandwidth,
        bwd_compute_multiplier: float,
        weighted_feature_bwd_compute_multiplier: float,
        is_pooled: bool,
        is_weighted: bool = False,
        caching_ratio: Optional[float] = None,
        is_inference: bool = False,
        prefetch_pipeline: bool = False,
        expected_cache_fetches: float = 0,
        uneven_sharding_perf_multiplier: float = 1.0,
    ) -> List[Perf]:
        """
        Attempts to model perfs as a function of relative wall times.

        Args:
            shard_sizes (List[List[int]]): the list of (local_rows, local_cols) of each
                shard.
            compute_kernel (str): compute kernel.
            compute_device (str): compute device.
            sharding_type (str): tw, rw, cw, twrw, dp.
            batch_sizes (List[int]): batch size for each input feature.
            world_size (int): the number of devices for all hosts.
            local_world_size (int): the number of the device for each host.
            input_lengths (List[float]): the list of the average number of lookups of each
                input query feature. Also referred to as pooling_mean, and it's equal to
                the pooling_factor * num_pooling.
            input_data_type_size (float): the data type size of the distributed
                data_parallel input.
            table_data_type_size (float): the data type size of the table.
            output_data_type_size (float): the data type size of the output embeddings.
            fwd_comm_data_type_size (float): the data type size of the distributed
                data_parallel input during forward communication.
            bwd_comm_data_type_size (float): the data type size of the distributed
                data_parallel input during backward communication.
            num_poolings (List[float]): number of poolings per sample, typically 1.0 for
                non-EBF use cases. In EBF use cases, this is the number of events per sample.
            hbm_mem_bw (float): the bandwidth of the device HBM.
            ddr_mem_bw (float): the bandwidth of the system DDR memory.
            hbm_to_ddr_bw (float): the bandwidth between device HBM and system DDR.
            intra_host_bw (float): the bandwidth within a single host like multiple threads.
            inter_host_bw (float): the bandwidth between two hosts like multiple machines.
            is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`), False
                if unpooled/sequential (ie. `Embedding`).
            is_weighted (bool = False): if the module is an EBC and is weighted, typically
                signifying an id score list feature.
            is_inference (bool = False): if planning for inference.
            caching_ratio (Optional[float] = None): cache ratio to determine the bandwidth
                of device.
            prefetch_pipeline (bool = False): whether prefetch pipeline is enabled.
            expected_cache_fetches (float): number of expected cache fetches across global batch
            uneven_sharding_perf_multiplier (float = 1.0): multiplier to account for uneven sharding perf

        Returns:
            List[float]: the list of perf for each shard.
        """

        shard_perfs = []
        device_bw = kernel_bw_lookup(
            compute_device,
            compute_kernel,
            hbm_mem_bw,
            ddr_mem_bw,
            hbm_to_ddr_mem_bw,
            caching_ratio,
            prefetch_pipeline,
        )
        if device_bw is None:
            raise PlannerError(
                f"No kernel bandwidth exists for this combo of compute device: {compute_device}, compute kernel: {compute_kernel}"
            )

        for hash_size, emb_dim in shard_sizes:
            if (
                sharding_type == ShardingType.TABLE_WISE.value
                or sharding_type == ShardingType.COLUMN_WISE.value
                or sharding_type == ShardingType.TABLE_COLUMN_WISE.value
            ):
                shard_perf = cls._get_tw_sharding_perf(
                    batch_sizes=batch_sizes,
                    world_size=world_size,
                    local_world_size=local_world_size,
                    input_lengths=input_lengths,
                    emb_dim=emb_dim,
                    input_data_type_size=input_data_type_size,
                    table_data_type_size=table_data_type_size,
                    output_data_type_size=output_data_type_size,
                    fwd_a2a_comm_data_type_size=fwd_a2a_comm_data_type_size,
                    bwd_a2a_comm_data_type_size=bwd_a2a_comm_data_type_size,
                    num_poolings=num_poolings,
                    hbm_to_ddr_mem_bw=hbm_to_ddr_mem_bw,
                    device_bw=device_bw,
                    comms_bandwidths=comms_bandwidths,
                    bwd_compute_multiplier=bwd_compute_multiplier,
                    weighted_feature_bwd_compute_multiplier=weighted_feature_bwd_compute_multiplier,
                    is_pooled=is_pooled,
                    is_weighted=is_weighted,
                    is_inference=is_inference,
                    expected_cache_fetches=expected_cache_fetches,
                )
            elif sharding_type == ShardingType.ROW_WISE.value:
                shard_perf = cls._get_rw_sharding_perf(
                    batch_sizes=batch_sizes,
                    world_size=world_size,
                    local_world_size=local_world_size,
                    input_lengths=input_lengths,
                    emb_dim=emb_dim,
                    input_data_type_size=input_data_type_size,
                    table_data_type_size=table_data_type_size,
                    output_data_type_size=output_data_type_size,
                    fwd_a2a_comm_data_type_size=fwd_a2a_comm_data_type_size,
                    bwd_a2a_comm_data_type_size=bwd_a2a_comm_data_type_size,
                    fwd_sr_comm_data_type_size=fwd_sr_comm_data_type_size,
                    bwd_sr_comm_data_type_size=bwd_sr_comm_data_type_size,
                    num_poolings=num_poolings,
                    hbm_to_ddr_mem_bw=hbm_to_ddr_mem_bw,
                    device_bw=device_bw,
                    comms_bandwidths=comms_bandwidths,
                    bwd_compute_multiplier=bwd_compute_multiplier,
                    weighted_feature_bwd_compute_multiplier=weighted_feature_bwd_compute_multiplier,
                    is_pooled=is_pooled,
                    is_weighted=is_weighted,
                    expected_cache_fetches=expected_cache_fetches,
                    is_inference=is_inference,
                )
            elif (
                sharding_type == ShardingType.TABLE_ROW_WISE.value
                or sharding_type == ShardingType.GRID_SHARD.value
            ):
                shard_perf = cls._get_twrw_sharding_perf(
                    batch_sizes=batch_sizes,
                    world_size=world_size,
                    local_world_size=local_world_size,
                    input_lengths=input_lengths,
                    emb_dim=emb_dim,
                    input_data_type_size=input_data_type_size,
                    table_data_type_size=table_data_type_size,
                    output_data_type_size=output_data_type_size,
                    fwd_a2a_comm_data_type_size=fwd_a2a_comm_data_type_size,
                    bwd_a2a_comm_data_type_size=bwd_a2a_comm_data_type_size,
                    fwd_sr_comm_data_type_size=fwd_sr_comm_data_type_size,
                    bwd_sr_comm_data_type_size=bwd_sr_comm_data_type_size,
                    num_poolings=num_poolings,
                    hbm_to_ddr_mem_bw=hbm_to_ddr_mem_bw,
                    device_bw=device_bw,
                    comms_bandwidths=comms_bandwidths,
                    bwd_compute_multiplier=bwd_compute_multiplier,
                    weighted_feature_bwd_compute_multiplier=weighted_feature_bwd_compute_multiplier,
                    is_pooled=is_pooled,
                    is_weighted=is_weighted,
                    expected_cache_fetches=expected_cache_fetches,
                )
            elif sharding_type == ShardingType.DATA_PARALLEL.value:
                shard_perf = cls._get_dp_sharding_perf(
                    batch_sizes=batch_sizes,
                    world_size=world_size,
                    local_world_size=local_world_size,
                    input_lengths=input_lengths,
                    grad_num_elem=hash_size * emb_dim,
                    emb_dim=emb_dim,
                    input_data_type_size=input_data_type_size,
                    table_data_type_size=table_data_type_size,
                    output_data_type_size=output_data_type_size,
                    num_poolings=num_poolings,
                    device_bw=device_bw,
                    comms_bandwidths=comms_bandwidths,
                    bwd_compute_multiplier=bwd_compute_multiplier,
                    weighted_feature_bwd_compute_multiplier=weighted_feature_bwd_compute_multiplier,
                    is_pooled=is_pooled,
                    is_weighted=is_weighted,
                )
            else:
                raise ValueError(
                    f"Unrecognized or unsupported sharding type provided: {sharding_type}"
                )
            shard_perfs.append(shard_perf)

        return shard_perfs

    @classmethod
    def _get_expected_cache_prefetch_time(
        cls,
        hbm_to_ddr_mem_bw: float,
        expected_cache_fetches: float,
        emb_dim: int,
        table_data_type_size: float,
    ) -> float:
        # TODO: validate cost model with empirical test
        prefetch_bytes = expected_cache_fetches * emb_dim * table_data_type_size
        return prefetch_bytes / hbm_to_ddr_mem_bw

    @classmethod
    def _get_tw_sharding_perf(
        cls,
        batch_sizes: List[int],
        world_size: int,
        local_world_size: int,
        input_lengths: List[float],
        emb_dim: int,
        input_data_type_size: float,
        table_data_type_size: float,
        output_data_type_size: float,
        fwd_a2a_comm_data_type_size: float,
        bwd_a2a_comm_data_type_size: float,
        num_poolings: List[float],
        hbm_to_ddr_mem_bw: float,
        device_bw: float,
        comms_bandwidths: GeneralizedCommsBandwidth,
        bwd_compute_multiplier: float,
        weighted_feature_bwd_compute_multiplier: float,
        is_pooled: bool,
        is_weighted: bool = False,
        is_inference: bool = False,
        expected_cache_fetches: float = 0,
    ) -> Perf:

        batch_inputs = sum(
            [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
        )
        batch_outputs = (
            sum([x * y for x, y in zip(num_poolings, batch_sizes)])
            if is_pooled
            else batch_inputs
        )

        input_read_size = math.ceil(batch_inputs * world_size * input_data_type_size)
        if is_weighted:
            input_read_size *= 2

        # minimum embedding dim is set to 32 due to kernel usage
        embedding_lookup_size = (
            batch_inputs * world_size * max(emb_dim, 32) * table_data_type_size
        )

        fwd_output_write_size = (
            batch_outputs * world_size * emb_dim * fwd_a2a_comm_data_type_size
        )
        bwd_output_write_size = (
            batch_outputs * world_size * emb_dim * bwd_a2a_comm_data_type_size
        )

        # embedding dim below 128 will reduce kernel efficency
        block_usage_penalty = 1
        if emb_dim < FULL_BLOCK_EMB_DIM:
            if emb_dim >= 64:
                block_usage_penalty = HALF_BLOCK_PENALTY
            else:  # emb_dim >= 32
                block_usage_penalty = QUARTER_BLOCK_PENALTY
        comms_bw = comms_bandwidths.get_bw(
            world_size=world_size,
            local_world_size=local_world_size,
            collective_type=CollectiveType.ALL_TO_ALL,
        )
        fwd_comms = fwd_output_write_size / comms_bw

        fwd_compute = (
            (input_read_size + embedding_lookup_size + fwd_output_write_size)
            * block_usage_penalty
            / device_bw
        )
        if is_inference:
            # only consider forward compute and comms for inference
            return Perf(
                fwd_compute=fwd_compute, fwd_comms=fwd_comms, bwd_compute=0, bwd_comms=0
            )

        bwd_comms = bwd_output_write_size / comms_bw

        bwd_grad_indice_weights_kernel = (
            fwd_compute * WEIGHTED_KERNEL_MULTIPLIER if is_weighted else 0
        )

        # includes fused optimizers
        bwd_compute = fwd_compute * bwd_compute_multiplier
        if is_weighted:
            bwd_compute = bwd_compute * weighted_feature_bwd_compute_multiplier

        prefetch_compute = cls._get_expected_cache_prefetch_time(
            hbm_to_ddr_mem_bw, expected_cache_fetches, emb_dim, table_data_type_size
        )

        # in order of model parallel execution, starting with:
        # BWD DP -> BWD MP ... FWD MP -> FWD DP
        return Perf(
            fwd_compute=fwd_compute,
            fwd_comms=fwd_comms,
            bwd_compute=bwd_compute + bwd_grad_indice_weights_kernel,
            bwd_comms=bwd_comms,
            prefetch_compute=prefetch_compute,
        )

    @classmethod
    def _get_rw_sharding_perf(
        cls,
        batch_sizes: List[int],
        world_size: int,
        local_world_size: int,
        input_lengths: List[float],
        emb_dim: int,
        input_data_type_size: float,
        table_data_type_size: float,
        output_data_type_size: float,
        fwd_a2a_comm_data_type_size: float,
        bwd_a2a_comm_data_type_size: float,
        fwd_sr_comm_data_type_size: float,
        bwd_sr_comm_data_type_size: float,
        num_poolings: List[float],
        hbm_to_ddr_mem_bw: float,
        device_bw: float,
        comms_bandwidths: GeneralizedCommsBandwidth,
        bwd_compute_multiplier: float,
        weighted_feature_bwd_compute_multiplier: float,
        is_pooled: bool,
        is_weighted: bool = False,
        expected_cache_fetches: float = 0,
        is_inference: bool = False,
    ) -> Perf:
        batch_inputs = (
            sum(
                [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
            )
            / world_size
        )
        batch_outputs = (
            sum([x * y for x, y in zip(num_poolings, batch_sizes)])
            if is_pooled
            else batch_inputs
        )

        input_read_size = math.ceil(batch_inputs * world_size * input_data_type_size)
        if is_weighted:
            input_read_size *= 2

        embedding_lookup_size = (
            batch_inputs * world_size * emb_dim * table_data_type_size
        )

        fwd_output_write_size = (
            batch_outputs * world_size * emb_dim * fwd_sr_comm_data_type_size
            if is_pooled
            else batch_outputs * world_size * emb_dim * fwd_a2a_comm_data_type_size
        )
        bwd_output_write_size = (
            batch_outputs * world_size * emb_dim * bwd_sr_comm_data_type_size
            if is_pooled
            else batch_outputs * world_size * emb_dim * bwd_a2a_comm_data_type_size
        )
        comms_bw = comms_bandwidths.get_bw(
            world_size=world_size,
            local_world_size=local_world_size,
            collective_type=CollectiveType.REDUCE_SCATTER,
        )

        fwd_comms = fwd_output_write_size / comms_bw

        fwd_compute = (
            input_read_size + embedding_lookup_size + fwd_output_write_size
        ) / device_bw

        if is_inference:
            # only consider forward compute and comms for inference
            return Perf(
                fwd_compute=fwd_compute, fwd_comms=fwd_comms, bwd_compute=0, bwd_comms=0
            )
        comms_bw = comms_bandwidths.get_bw(
            world_size=world_size,
            local_world_size=local_world_size,
            collective_type=CollectiveType.ALL_GATHER,
        )
        bwd_comms = bwd_output_write_size / comms_bw

        bwd_batched_copy = bwd_output_write_size * BATCHED_COPY_PERF_FACTOR / device_bw

        bwd_grad_indice_weights_kernel = (
            fwd_compute * WEIGHTED_KERNEL_MULTIPLIER if is_weighted else 0
        )

        bwd_compute = fwd_compute * bwd_compute_multiplier
        if is_weighted:
            bwd_compute = bwd_compute * weighted_feature_bwd_compute_multiplier

        # for row-wise, expected_cache_fetches per shard is / world_size
        prefetch_compute = cls._get_expected_cache_prefetch_time(
            hbm_to_ddr_mem_bw,
            expected_cache_fetches / world_size,
            emb_dim,
            table_data_type_size,
        )

        return Perf(
            fwd_compute=fwd_compute,
            fwd_comms=fwd_comms,
            bwd_compute=bwd_compute + bwd_grad_indice_weights_kernel,
            bwd_comms=bwd_comms + bwd_batched_copy,
            prefetch_compute=prefetch_compute,
        )

    @classmethod
    def _get_twrw_sharding_perf(
        cls,
        batch_sizes: List[int],
        world_size: int,
        local_world_size: int,
        input_lengths: List[float],
        emb_dim: int,
        input_data_type_size: float,
        table_data_type_size: float,
        output_data_type_size: float,
        fwd_a2a_comm_data_type_size: float,
        bwd_a2a_comm_data_type_size: float,
        fwd_sr_comm_data_type_size: float,
        bwd_sr_comm_data_type_size: float,
        num_poolings: List[float],
        hbm_to_ddr_mem_bw: float,
        device_bw: float,
        comms_bandwidths: GeneralizedCommsBandwidth,
        bwd_compute_multiplier: float,
        weighted_feature_bwd_compute_multiplier: float,
        is_pooled: bool,
        is_weighted: bool = False,
        expected_cache_fetches: float = 0,
    ) -> Perf:
        batch_inputs = (
            sum(
                [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
            )
            / local_world_size
        )
        batch_outputs = (
            sum([x * y for x, y in zip(num_poolings, batch_sizes)])
            if is_pooled
            else batch_inputs
        )

        input_read_size = math.ceil(batch_inputs * world_size * input_data_type_size)
        if is_weighted:
            input_read_size *= 2

        embedding_lookup_size = (
            batch_inputs * world_size * emb_dim * table_data_type_size
        )

        fwd_output_write_size = (
            batch_outputs * world_size * emb_dim * fwd_sr_comm_data_type_size
        )
        bwd_output_write_size = (
            batch_outputs * world_size * emb_dim * bwd_sr_comm_data_type_size
        )
        comms_bw = comms_bandwidths.get_bw(
            world_size=local_world_size,
            local_world_size=local_world_size,
            collective_type=CollectiveType.REDUCE_SCATTER,
        )

        # intra host comm
        fwd_comms = fwd_output_write_size / comms_bw

        # inter host comm
        if world_size > local_world_size:
            inter_host_fwd_output_write_size = (
                batch_outputs
                * (
                    world_size / local_world_size
                )  # this is the size of the procees group.
                * emb_dim
                * fwd_a2a_comm_data_type_size
            )
            comms_bw = comms_bandwidths.get_bw(
                world_size=int(world_size / local_world_size),
                local_world_size=1,
                collective_type=CollectiveType.ALL_TO_ALL,
            )
            fwd_comms += inter_host_fwd_output_write_size / comms_bw

        fwd_compute = (
            input_read_size + embedding_lookup_size + fwd_output_write_size
        ) / device_bw

        # intra host comm (i.e. all gather)
        comms_bw = comms_bandwidths.get_bw(
            world_size=local_world_size,
            local_world_size=local_world_size,
            collective_type=CollectiveType.ALL_GATHER,
        )
        bwd_comms = bwd_output_write_size / comms_bw

        # inter host comm (i.e. all to all)
        if world_size > local_world_size:
            inter_host_bwd_output_write_size = (
                batch_outputs
                * (
                    world_size / local_world_size
                )  # this is the size of the procress group.
                * emb_dim
                * bwd_a2a_comm_data_type_size
            )
            comms_bw = comms_bandwidths.get_bw(
                world_size=int(world_size / local_world_size),
                local_world_size=1,
                collective_type=CollectiveType.ALL_TO_ALL,
            )
            bwd_comms += inter_host_bwd_output_write_size / comms_bw

        bwd_grad_indice_weights_kernel = (
            fwd_compute * WEIGHTED_KERNEL_MULTIPLIER if is_weighted else 0
        )

        bwd_batched_copy = bwd_output_write_size * BATCHED_COPY_PERF_FACTOR / device_bw

        bwd_compute = fwd_compute * bwd_compute_multiplier
        if is_weighted:
            bwd_compute = bwd_compute * weighted_feature_bwd_compute_multiplier

        # for table-wise-row-wise or grid_shard, expected_cache_fetches per shard is / local_world_size
        prefetch_compute = cls._get_expected_cache_prefetch_time(
            hbm_to_ddr_mem_bw,
            expected_cache_fetches / local_world_size,
            emb_dim,
            table_data_type_size,
        )

        return Perf(
            fwd_compute=fwd_compute,
            fwd_comms=fwd_comms,
            bwd_compute=bwd_compute + bwd_grad_indice_weights_kernel,
            bwd_comms=bwd_comms + bwd_batched_copy,
            prefetch_compute=prefetch_compute,
        )

    @classmethod
    def _get_dp_sharding_perf(
        cls,
        batch_sizes: List[int],
        world_size: int,
        local_world_size: int,
        input_lengths: List[float],
        grad_num_elem: int,
        emb_dim: int,
        input_data_type_size: float,
        table_data_type_size: float,
        output_data_type_size: float,
        num_poolings: List[float],
        device_bw: float,
        comms_bandwidths: GeneralizedCommsBandwidth,
        bwd_compute_multiplier: float,
        weighted_feature_bwd_compute_multiplier: float,
        is_pooled: bool,
        is_weighted: bool = False,
    ) -> Perf:
        batch_inputs = sum(
            [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
        )
        batch_outputs = (
            sum([x * y for x, y in zip(num_poolings, batch_sizes)])
            if is_pooled
            else batch_inputs
        )

        input_read_size = math.ceil(batch_inputs * input_data_type_size)
        if is_weighted:
            input_read_size *= 2

        embedding_lookup_size = batch_inputs * emb_dim * table_data_type_size

        output_write_size = batch_outputs * emb_dim * table_data_type_size
        table_size = grad_num_elem * table_data_type_size

        fwd_compute = (
            input_read_size + embedding_lookup_size + output_write_size
        ) / device_bw

        num_nodes = min(world_size / local_world_size, 2)

        # all-reduce data transfer: https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf
        comms_bw = comms_bandwidths.get_bw(
            world_size=world_size,
            local_world_size=local_world_size,
            collective_type=CollectiveType.ALL_REDUCE,
        )
        all_reduce = table_size * (2 * num_nodes - 1) / num_nodes / comms_bw
        # inter host communication constraint
        if world_size > 2 * local_world_size:
            all_reduce *= 2

        # SGD + Fill + BUnary
        optimizer_kernels = table_size * DP_ELEMENTWISE_KERNELS_PERF_FACTOR / device_bw

        bwd_compute = fwd_compute * bwd_compute_multiplier
        if is_weighted:
            bwd_compute = bwd_compute * weighted_feature_bwd_compute_multiplier

        bwd_grad_indice_weights_kernel = (
            fwd_compute * WEIGHTED_KERNEL_MULTIPLIER if is_weighted else 0
        )

        # TODO(T170641643): we don't model prefetch_compute for data parallel yet, see
        # comment in perf_func_emb_wall_time() regarding expected_cache_fetches calculation.
        return Perf(
            fwd_compute=fwd_compute,
            fwd_comms=0,
            bwd_compute=bwd_compute + bwd_grad_indice_weights_kernel,
            bwd_comms=all_reduce + optimizer_kernels,
        )


def _extract_comm_data_type_size(
    sharder: ModuleSharder[nn.Module], sharding_option: ShardingOption
) -> Tuple[float, float, float, float]:
    table_data_type_size = sharding_option.tensor.element_size()

    fwd_a2a_comm_data_type_size = table_data_type_size
    bwd_a2a_comm_data_type_size = table_data_type_size
    fwd_sr_comm_data_type_size = table_data_type_size
    bwd_sr_comm_data_type_size = table_data_type_size

    if sharder.qcomm_codecs_registry is not None:
        qcomm_codecs_registry = sharder.qcomm_codecs_registry
        if (
            sharding_option.is_pooled
            and CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name in qcomm_codecs_registry
        ):
            codecs = sharder.qcomm_codecs_registry[
                CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name
            ]
            fwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.forward.quantized_dtype
            ).element_size()
            bwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.backward.quantized_dtype
            ).element_size()

        if (
            not sharding_option.is_pooled
            and CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name in qcomm_codecs_registry
        ):
            codecs = qcomm_codecs_registry[CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name]
            fwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.forward.quantized_dtype
            ).element_size()
            bwd_a2a_comm_data_type_size = torch.tensor(
                [], dtype=codecs.backward.quantized_dtype
            ).element_size()

        if (
            sharding_option.is_pooled
            and CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name in qcomm_codecs_registry
        ):
            codecs = qcomm_codecs_registry[CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name]
            fwd_sr_comm_data_type_size = torch.tensor(
                [], dtype=codecs.forward.quantized_dtype
            ).element_size()
            bwd_sr_comm_data_type_size = torch.tensor(
                [], dtype=codecs.backward.quantized_dtype
            ).element_size()

    return (
        fwd_a2a_comm_data_type_size,
        bwd_a2a_comm_data_type_size,
        fwd_sr_comm_data_type_size,
        bwd_sr_comm_data_type_size,
    )


class EmbeddingStorageEstimator(ShardEstimator):
    """
    Embedding Storage Usage Estimator

    Args:
        topology (Topology): device topology.
        constraints (Optional[Dict[str, ParameterConstraints]]): parameter constraints.
        pipeline_type (PipelineType): The type of pipeline, if any. Will determine the
            input replication factor during memory estimation.
        run_embedding_at_peak_memory (bool): If the embedding fwd/bwd will be execute when HBM
            usage is at peak. When set to TRUE, any temporary memory allocation during
            embedding forward/backward, as long as output sizes before output_dist will
            be counted towards HBM storage cost. Otherwise they won't since they'll be
            "hidden" by the real memory peak.

            Only take effect if pipeline_type is set for backward compatibility (not affecting
            models using old pipeline-agnostic formula)

            Default to false because this is typically false for RecSys since memory
            peak happens at the end of dense forwrad / beginning of dense backward instead.
        is_inference (bool): If the model is inference model. Default to False.
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        pipeline_type: PipelineType = PipelineType.NONE,
        run_embedding_at_peak_memory: bool = False,
        is_inference: bool = False,
    ) -> None:
        self._topology = topology
        self._constraints = constraints
        self._pipeline_type = pipeline_type
        self._run_embedding_at_peak_memory = run_embedding_at_peak_memory
        self._is_inference = is_inference

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        """
        Estimate the storage cost of each sharding option.

        Args:
            sharding_options (List[ShardingOption]): list of sharding options.
            sharder_map (Optional[Dict[str, ModuleSharder[nn.Module]]]): map from module
                type to sharder.
        """
        if not sharder_map:
            assert not sharding_options, "sharder_map not provided for sharding_options"
            return

        for sharding_option in sharding_options:
            sharder_key = sharder_name(type(sharding_option.module[1]))
            sharder = sharder_map[sharder_key]

            caching_ratio = sharding_option.cache_load_factor
            # TODO: remove after deprecating fused_params in sharder
            if caching_ratio is None:
                caching_ratio = (
                    sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                    if hasattr(sharder, "fused_params") and sharder.fused_params
                    else None
                )
            constraints: Optional[ParameterConstraints] = (
                self._constraints.get(sharding_option.name, None)
                if self._constraints
                else None
            )
            num_poolings = get_num_poolings(self._constraints, sharding_option)
            assert len(num_poolings) == sharding_option.num_inputs
            batch_sizes = (
                constraints.batch_sizes
                if constraints and constraints.batch_sizes
                else [sharding_option.batch_size] * sharding_option.num_inputs
            )

            key_value_params: Optional[KeyValueParams] = (
                constraints.key_value_params
                if constraints and constraints.key_value_params
                else None
            )
            kv_cache_load_factor: float = (
                sharder.fused_params.get("cache_load_factor", KV_CACHING_RATIO)
                if sharder.fused_params
                else KV_CACHING_RATIO
            )
            use_virtual_table: bool = (
                constraints.use_virtual_table if constraints else False
            )

            # hardcoded as 8 bytes
            # input indices can be of int32, but in TBE they get converted to int64 anyway
            input_data_type_size = BIGINT_DTYPE

            output_data_type_size: float = (
                DATA_TYPE_NUM_BITS[sharding_option.output_dtype] / 8
                if sharding_option.output_dtype
                else sharding_option.tensor.element_size()
            )

            mpp_conf = (
                sharding_option.cache_params.multipass_prefetch_config
                if sharding_option.cache_params
                else None
            )
            # TODO: remove after deprecating fused_params in sharder
            if mpp_conf is None:
                mpp_conf = (
                    sharder.fused_params.get("multipass_prefetch_config", None)
                    if hasattr(sharder, "fused_params") and sharder.fused_params
                    else None
                )
            shard_storages = calculate_shard_storages(
                sharder=sharder,
                sharding_type=sharding_option.sharding_type,
                tensor=sharding_option.tensor,
                compute_device=self._topology.compute_device,
                compute_kernel=sharding_option.compute_kernel,
                shard_sizes=[shard.size for shard in sharding_option.shards],
                batch_sizes=batch_sizes,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                num_poolings=num_poolings,
                caching_ratio=caching_ratio if caching_ratio else UVM_CACHING_RATIO,
                is_pooled=sharding_option.is_pooled,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                pipeline_type=self._pipeline_type,
                count_ephemeral_storage_cost=self._run_embedding_at_peak_memory,
                is_inference=self._is_inference,
                multipass_prefetch_max_pass=mpp_conf.num_passes if mpp_conf else None,
                key_value_params=key_value_params,
                kv_cache_load_factor=kv_cache_load_factor,
                use_virtual_table=use_virtual_table,
            )
            for shard, storage in zip(sharding_option.shards, shard_storages):
                shard.storage = storage


def calculate_pipeline_io_cost(
    input_size: int,
    output_size: int,
    prefetch_size: int,
    pipeline_type: PipelineType,
    multipass_prefetch_max_pass: Optional[int],
    count_ephemeral_storage_cost: bool = False,
    is_inference: bool = False,
) -> int:
    # These magical number comes from heuristical analysis of memory snapshot during
    # pipelining, and are subject to the actual implementation.
    #
    # Now it's static to make memory estimation more sane for UVM offloading,
    # we need to make this estimation more blackbox-based.
    if is_inference:
        return 0

    # Output size is considered ephemeral storage cost since they are temporarily
    # only during all2all and won't last long (e.g. from fwd to bwd)
    output_contribition_to_peak_memory = (
        output_size if count_ephemeral_storage_cost else 0
    )

    if pipeline_type == PipelineType.TRAIN_SPARSE_DIST:
        pipelining_hbm_input_factor = 2
        return (
            pipelining_hbm_input_factor * input_size
            + output_contribition_to_peak_memory
        )
    if pipeline_type == PipelineType.TRAIN_PREFETCH_SPARSE_DIST:
        multipass_prefetch_max_pass = multipass_prefetch_max_pass or 1
        pipelining_hbm_input_factor = 3
        prefetch_bursty_hbm_input_factor = 1 + 6 / multipass_prefetch_max_pass
        return (
            pipelining_hbm_input_factor * input_size
            + int(prefetch_bursty_hbm_input_factor * prefetch_size)
            + output_contribition_to_peak_memory
        )

    # Catch all case, for backward compatibility
    return input_size + output_size


def calculate_shard_storages(
    sharder: ModuleSharder[nn.Module],
    sharding_type: str,
    tensor: torch.Tensor,
    compute_device: str,
    compute_kernel: str,
    shard_sizes: List[List[int]],
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    num_poolings: List[float],
    caching_ratio: float,
    is_pooled: bool,
    input_data_type_size: float,
    output_data_type_size: float,
    pipeline_type: PipelineType = PipelineType.NONE,
    count_ephemeral_storage_cost: bool = False,
    is_inference: bool = False,
    multipass_prefetch_max_pass: Optional[int] = None,
    key_value_params: Optional[KeyValueParams] = None,
    kv_cache_load_factor: float = KV_CACHING_RATIO,
    use_virtual_table: bool = False,
) -> List[Storage]:
    """
    Calculates estimated storage sizes for each sharded tensor, comprised of input,
    output, tensor, gradient, and optimizer sizes.

    Args:
        sharder (ModuleSharder[nn.Module]): sharder for module that supports sharding.
        sharding_type (str): provided ShardingType value.
        tensor (torch.Tensor): tensor to be sharded.
        compute_device (str): compute device to be used.
        compute_kernel (str): compute kernel to be used.
        shard_sizes (List[List[int]]): list of dimensions of each sharded tensor.
        batch_sizes (List[int]): batch size for each input feature.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        input_lengths (List[float]): average input lengths synonymous with pooling
            factors.
        num_poolings (List[float]): average number of poolings per sample
            (typically 1.0).
        caching_ratio (float): ratio of HBM to DDR memory for UVM caching.
        is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`), False
            if unpooled/sequential (ie. `Embedding`).
        input_data_type_size (int): number of bytes of input data type.
        output_data_type_size (int): number of bytes of output data type.
        pipeline_type: PipelineType: pipeline type if for training.
        is_inference: bool, whether the model is for inference.
        key_value_params (Optional[KeyValueParams]): fused params for SSD/DRAM KV cache.

    Returns:
        List[Storage]: storage object for each device in topology.
    """
    input_sizes, output_sizes = _calculate_shard_io_sizes(
        sharding_type=sharding_type,
        batch_sizes=batch_sizes,
        world_size=world_size,
        local_world_size=local_world_size,
        input_lengths=input_lengths,
        emb_dim=tensor.shape[1],
        shard_sizes=shard_sizes,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
        num_poolings=num_poolings,
        is_pooled=is_pooled,
    )

    tensor_storage = sharder.storage_usage(tensor, compute_device, compute_kernel)
    hbm_storage: int = tensor_storage.get("hbm", 0)
    ddr_storage: int = tensor_storage.get("ddr", 0)

    table_cached = _is_table_cached(compute_kernel)
    if table_cached:
        hbm_storage = round(ddr_storage * caching_ratio)

    optimizer_class = getattr(tensor, "_optimizer_classes", [None])[0]

    hbm_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=hbm_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        optimizer_class=optimizer_class,
        is_inference=is_inference,
        clf=caching_ratio if table_cached else None,
    )
    ddr_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=ddr_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        optimizer_class=optimizer_class,
        is_inference=is_inference,
    )

    if (
        compute_kernel
        in {
            EmbeddingComputeKernel.KEY_VALUE.value,
            EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
            EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
        }
        or use_virtual_table
    ):
        # KVZCH does not have dedicated inference compute kernel, so we use use_virtual_table
        # to settup ddr_specific_sizes
        key_value_params = key_value_params or KeyValueParams(
            max_l1_cache_size=0, l2_cache_size=0
        )

        hbm_specific_sizes = [
            min(
                (key_value_params.max_l1_cache_size or 0) * 1024 * 1024,
                ceil(
                    tensor.shape[0]  # num_embeddings
                    * kv_cache_load_factor
                    * tensor.element_size()  # size of one column
                    * tensor.shape[1],  # number of columns in embedding
                ),
            )
            for _ in hbm_specific_sizes
        ]
        ddr_specific_sizes = [
            # TODO: revisit the logic for SSD virtual table
            0
            for _ in ddr_specific_sizes
        ]

    hbm_sizes: List[int] = [
        (
            hbm_specific_size
            + calculate_pipeline_io_cost(
                input_size=input_size,
                output_size=output_size,
                prefetch_size=input_size if table_cached else 0,
                pipeline_type=pipeline_type,
                multipass_prefetch_max_pass=multipass_prefetch_max_pass,
                count_ephemeral_storage_cost=count_ephemeral_storage_cost,
                is_inference=is_inference,
            )
            if compute_device in {"cuda", "mtia"}
            else 0
        )
        for input_size, output_size, hbm_specific_size in zip(
            input_sizes,
            output_sizes,
            hbm_specific_sizes,
        )
    ]
    ddr_sizes: List[int] = [
        (
            input_size + output_size + ddr_specific_size
            if compute_device == "cpu" and not is_inference
            else ddr_specific_size
        )
        for input_size, output_size, ddr_specific_size in zip(
            input_sizes,
            output_sizes,
            ddr_specific_sizes,
        )
    ]

    return [
        Storage(
            hbm=hbm_size,
            ddr=ddr_size,
        )
        for hbm_size, ddr_size in zip(hbm_sizes, ddr_sizes)
    ]


def _is_table_cached(
    compute_kernel: str,
) -> bool:
    if compute_kernel in {
        EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
        EmbeddingComputeKernel.KEY_VALUE.value,
        EmbeddingComputeKernel.SSD_VIRTUAL_TABLE.value,
        EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
    }:
        return True
    return False


def get_num_poolings(
    constraints: Optional[Dict[str, ParameterConstraints]], so: ShardingOption
) -> List[float]:
    # first priority is given for sharding_option.num_poolings,
    # otherwise Manifold planner configs will be overwritten by parameter constraints
    # default path will use constraints
    if so.num_poolings is not None:
        num_poolings = so.num_poolings
        if len(so.input_lengths) == len(num_poolings):
            return num_poolings

    # Second priority: use constraint-based num_poolings
    if constraints and constraints.get(so.name) and constraints[so.name].num_poolings:
        return cast(List[float], constraints[so.name].num_poolings)

    # Fallback: use default NUM_POOLINGS constant
    return [NUM_POOLINGS] * len(so.input_lengths)


def _calculate_shard_io_sizes(
    sharding_type: str,
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    shard_sizes: List[List[int]],
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return _calculate_dp_shard_io_sizes(
            batch_sizes=batch_sizes,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            num_shards=len(shard_sizes),
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return _calculate_tw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type in {
        ShardingType.COLUMN_WISE.value,
        ShardingType.TABLE_COLUMN_WISE.value,
    }:
        return _calculate_cw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _calculate_rw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif (
        sharding_type == ShardingType.TABLE_ROW_WISE.value
        or sharding_type == ShardingType.GRID_SHARD.value  # same as table row wise
    ):
        return _calculate_twrw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    else:
        raise ValueError(
            f"Unrecognized or unsupported sharding type provided: {sharding_type}"
        )


def _calculate_dp_shard_io_sizes(
    batch_sizes: List[int],
    input_lengths: List[float],
    emb_dim: int,
    num_shards: int,
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [math.ceil(batch_inputs * input_data_type_size)] * num_shards
    output_sizes = [
        math.ceil(batch_outputs * emb_dim * output_data_type_size)
    ] * num_shards

    return input_sizes, output_sizes


def _calculate_tw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [math.ceil(batch_inputs * world_size * input_data_type_size)]
    output_sizes = [
        math.ceil(batch_outputs * world_size * emb_dim * output_data_type_size)
    ]

    return input_sizes, output_sizes


def _calculate_cw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [math.ceil(batch_inputs * world_size * input_data_type_size)] * len(
        shard_sizes
    )
    output_sizes = [
        math.ceil(
            batch_outputs * world_size * shard_sizes[i][1] * output_data_type_size
        )
        for i in range(len(shard_sizes))
    ]

    return input_sizes, output_sizes


def _calculate_rw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = (
        sum([x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)])
        / world_size
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [
        (
            math.ceil(batch_inputs * world_size * input_data_type_size)
            if prod(shard) != 0
            else 0
        )
        for shard in shard_sizes
    ]
    output_sizes = [
        (
            math.ceil(
                batch_outputs * world_size * shard_sizes[i][1] * output_data_type_size
            )
            if prod(shard) != 0
            else 0
        )
        for i, shard in enumerate(shard_sizes)
    ]

    return input_sizes, output_sizes


def _calculate_twrw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = (
        sum([x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)])
        / local_world_size
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [
        (
            math.ceil(batch_inputs * world_size * input_data_type_size)
            if prod(shard) != 0
            else 0
        )
        for shard in shard_sizes
    ]
    output_sizes = [
        (
            math.ceil(
                batch_outputs * world_size * shard_sizes[i][1] * output_data_type_size
            )
            if prod(shard) != 0
            else 0
        )
        for i, shard in enumerate(shard_sizes)
    ]

    return input_sizes, output_sizes


def _calculate_storage_specific_sizes(
    storage: int,
    shape: torch.Size,
    shard_sizes: List[List[int]],
    sharding_type: str,
    optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
    is_inference: bool = False,
    clf: Optional[float] = None,
) -> List[int]:
    tensor_sizes: List[int] = _calculate_tensor_sizes(
        storage,
        shape,
        shard_sizes,
        sharding_type,
    )
    optimizer_sizes = _calculate_optimizer_sizes(
        tensor_sizes,
        optimizer_class,
        shape,
    )
    cache_aux_state_sizes: List[int] = _calculate_cache_aux_state_sizes(
        shard_sizes,
        clf,
    )

    return [
        (
            cache_state_size + tensor_size + optimizer_size
            if not is_inference
            else tensor_size
        )
        for cache_state_size, tensor_size, optimizer_size in zip(
            cache_aux_state_sizes, tensor_sizes, optimizer_sizes
        )
    ]


def _calculate_tensor_sizes(
    storage: int, shape: torch.Size, shard_sizes: List[List[int]], sharding_type: str
) -> List[int]:
    return [
        (
            math.ceil(storage * prod(size) / prod(shape))
            if sharding_type != ShardingType.DATA_PARALLEL.value
            else storage
        )
        for size in shard_sizes
    ]


# If a table has turned on UVM caching (meaning clf is not None), there'll be
# 4x of table hash size and 16x of cache slot size HBM storage cost dedicated to
# cache aux state (note that this is not the cache content itself)
def _calculate_cache_aux_state_sizes(
    shard_sizes: List[List[int]], clf: Optional[float]
) -> List[int]:
    if clf is None:
        return [0] * len(shard_sizes)
    return [math.ceil(size[0] * (4 + clf * 16)) for size in shard_sizes]


def _calculate_optimizer_sizes(
    tensor_sizes: List[int],
    optimizer_class: Optional[Type[torch.optim.Optimizer]],
    sharding_tensor_shape: torch.Size,
) -> List[int]:
    optimizer_multiplier: float = _get_optimizer_multipler(
        optimizer_class,
        sharding_tensor_shape,
    )
    optimizer_sizes: List[int] = [
        math.ceil(tensor_size * optimizer_multiplier) for tensor_size in tensor_sizes
    ]
    return optimizer_sizes


def _get_optimizer_multipler(
    optimizer_class: Optional[Type[torch.optim.Optimizer]],
    shape: torch.Size,
) -> float:
    if not optimizer_class:
        return 0.0
    if optimizer_class in [torch.optim.SGD, trec_optim.SGD]:
        return 0
    elif optimizer_class in [torch.optim.Adam, trec_optim.Adam]:
        return 2
    elif optimizer_class == trec_optim.RowWiseAdagrad:
        return 1 / shape[-1]
    else:
        return 1


class EmbeddingOffloadStats(CacheStatistics):
    """Computes cache statistics for uvm_fused_cache tables.

    Args:

    cachebility (float):
        The area-under-the-curve of miss-ratio curve.
    expected_lookups (float):
        The expected number of unique embedding ids per global batch.
    mrc_hist_counts (torch.Tensor):
        A 1d tensor (size n) holding a histogram of LRU miss ratio curve. Each bin
        represents 1/nth of possible LRU cache sizes (from load_factor 0 to load_factor
        1.0). The bin contains the number of expected LRU operations that could be
        handled without a cache miss if the LRU load_factor was at least that size.
    height (int):
        The height (num_embeddings) of the embedding table.
    """

    def __init__(
        self,
        cacheability: float,
        expected_lookups: int,
        mrc_hist_counts: torch.Tensor,
        height: int,
    ) -> None:
        self._cacheability = cacheability
        self._expected_lookups = expected_lookups
        self.height = height

        if mrc_hist_counts.dim() != 1:
            raise ValueError(f"expected 1d tensor, got {mrc_hist_counts.dim()}d")
        if mrc_hist_counts.size()[0] == 0:
            raise ValueError("expected non-empty tensor")

        self.hist: torch.Tensor = mrc_hist_counts
        self.bins: torch.Tensor = torch.linspace(0, height, len(mrc_hist_counts) + 1)

    @property
    def expected_lookups(self) -> int:
        return self._expected_lookups

    def expected_miss_rate(self, clf: float) -> float:
        cache_size = torch.tensor(clf * self.height)
        miss_rate = EmbeddingOffloadStats.estimate_cache_miss_rate(
            cache_sizes=cache_size, hist=self.hist, bins=self.bins
        )
        return miss_rate.item()

    @property
    def cacheability(self) -> float:
        return self._cacheability

    @staticmethod
    def estimate_cache_miss_rate(
        cache_sizes: torch.Tensor, hist: torch.Tensor, bins: torch.Tensor
    ) -> torch.Tensor:
        """Calculate estimated cache miss ratio for the proposed cache_sizes, given the MRC
        histogram.
        """
        ys = hist.cumsum(dim=0)
        if ys[-1] == 0:
            # feature has no usage data -> no cache misses
            return torch.zeros_like(cache_sizes, dtype=torch.float32)
        ys = ys / ys[-1]  # rescale [0,1]
        ys = 1 - ys  # make miss-ratio, not hit-ratio

        # torch.bucketize has slightly different semantics to np.digitize,
        # and np.digitize has a complex interface, read the docs carefully!
        # we're trying to reverse the ops of np.histogram, indices are one larger than
        # the insert positions, since with right=True, index returned such that x <
        # bins[index], so x 'lives' in hist[index-1]
        # A cache size of k will get hits for all stack distances of upto k-1 inclusive.
        larger_bin_indices = torch.bucketize(cache_sizes - 1, bins, right=True)
        # Augment ys to deal with torch.bucketize boundary conditions:
        #   values outside of bins range map to 0, or len(bins).
        # So we extend ys to populate sentinel values for these cases.  With the twist that
        # the left-hand sentinel we put on the right side of the array, as larger_bin_indices - 1
        # maps 0 -> -1, which pytorch maps to most right hand value.
        ys = torch.cat((ys, torch.tensor([0.0, 1.0])))
        return ys[larger_bin_indices - 1]
