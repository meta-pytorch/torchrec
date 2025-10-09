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
import statistics
from collections import defaultdict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from torch import nn

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import BIGINT_DTYPE, NUM_POOLINGS
from torchrec.distributed.planner.shard_estimators import (
    _calculate_shard_io_sizes,
    get_num_poolings,
)
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
    HeuristicalStorageReservation,
    InferenceStorageReservation,
)
from torchrec.distributed.planner.types import (
    CriticalPathEstimate,
    Enumerator,
    ParameterConstraints,
    Perf,
    PlanDebugStats,
    ShardingOption,
    Stats,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import (
    _find_imbalance_tables,
    bytes_to_gb,
    bytes_to_mb,
    sharder_name as get_sharder_name,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
    ShardingType,
)

logger: logging.Logger = logging.getLogger(__name__)


MIN_WIDTH = 90


def _normalize_float(p: List[float]) -> List[float]:
    p_total = sum(p)
    assert p_total > 0
    return [p_i / p_total for p_i in p]


def _normalize_int(p: List[int]) -> List[float]:
    p_total = sum(p)
    assert p_total > 0
    return [p_i * 1.0 / p_total for p_i in p]


def _total_variation(p: List[float]) -> float:
    k = len(p)
    assert k > 0
    return max(abs(pi - 1.0 / k) for pi in p)


def _total_distance(p: List[float]) -> float:
    k = len(p)
    assert k > 0
    return sum(abs(pi - 1.0 / k) for pi in p)


def _chi_sq_divergence(p: List[float]) -> float:
    k = len(p)
    assert k > 0
    return sum(abs(pi - 1.0 / k) ** 2.0 * k for pi in p)


def _kl_divergence(p: List[float]) -> float:
    k = len(p)
    assert k > 0
    return sum(pi * math.log(k * pi) for pi in p if pi > 0)


def _calc_max_chi_sq_divergence(N: int) -> float:
    # Upper bound for chi-sq divergence in our case given sample size of distribution (N)
    assert N > 0
    return (((N - 1) / N) ** 2.0) * N + (N - 1) * (1 / N)


def _calc_max_kl_divergence(N: int) -> float:
    # Upper bound for KL divergence in our case given sample size of distribution (N)
    assert N > 0
    return math.log(N)


def _normalized_kl_divergence(p: List[float]) -> float:
    k = len(p)
    assert k > 0
    # Max val can be 0 if world size is 1 (e.g. local run)
    max_val = _calc_max_kl_divergence(k)
    return _kl_divergence(p) / max_val if max_val > 0 else 0.0


def _normalized_chi_sq_divergence(p: List[float]) -> float:
    k = len(p)
    assert k > 0
    # Max val can be 0 if world size is 1 (e.g. local run)
    max_val = _calc_max_chi_sq_divergence(k)
    return _chi_sq_divergence(p) / max_val if max_val > 0 else 0.0


IMBALANCE_STAT_MEASURE: Dict[str, Tuple[Callable[..., float], Dict[str, Any]]] = {
    "Total Variation": (_total_variation, {}),
    "Total Distance": (_total_distance, {}),
    "Chi Divergence": (_normalized_chi_sq_divergence, {}),
    "KL Divergence": (_normalized_kl_divergence, {}),
}


class EmbeddingStats(Stats):
    """
    Stats for a sharding planner execution.
    """

    def __init__(self) -> None:
        self._width: int = MIN_WIDTH
        self._stats_table: List[str] = []

    def log(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        batch_size: int,
        storage_reservation: StorageReservation,
        num_proposals: int,
        num_plans: int,
        run_time: float,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        enumerator: Optional[Enumerator] = None,
        debug: bool = True,
        debug_stats: Optional[PlanDebugStats] = None,
    ) -> None:
        """
        Logs stats for a given sharding plan.

        Provides a tabular view of stats for the given sharding plan with per device
        storage usage (HBM and DDR), perf, input, output, and number/type of shards.

        Args:
            sharding_plan (ShardingPlan): sharding plan chosen by the planner.
            topology (Topology): device topology.
            batch_size (int): batch size.
            storage_reservation (StorageReservation): reserves storage for unsharded
                parts of the model
            num_proposals (int): number of proposals evaluated.
            num_plans (int): number of proposals successfully partitioned.
            run_time (float): time taken to find plan (in seconds).
            best_plan (List[ShardingOption]): plan with expected performance.
            constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter
                names to provided ParameterConstraints.
            debug (bool): whether to enable debug mode.
        """

        shard_by_fqn = {
            module_name + "." + param_name: value
            for module_name, param_dict in sharding_plan.plan.items()
            # pyre-ignore - this is a EmbeddingShardingPlan below
            for param_name, value in param_dict.items()
        }
        stats: Dict[int, Dict[str, Any]] = {
            rank: {"type": {}, "input_sizes": 0.0, "output_sizes": 0.0}
            for rank in range(topology.world_size)
        }

        used_sharding_types = set()
        compute_kernels_to_count = defaultdict(int)
        compute_kernels_to_storage = defaultdict(lambda: Storage(0, 0))

        reserved_hbm_percent, dense_storage, kjt_storage = _compute_storage(
            storage_reservation=storage_reservation
        )

        for sharding_option in best_plan:
            fqn = sharding_option.fqn

            compute_kernels_to_count[sharding_option.compute_kernel] += 1
            compute_kernels_to_storage[
                sharding_option.compute_kernel
            ] += sharding_option.total_storage

            # for shard in sharding_option.shards:
            # compute_kernels_to_storage[sharding_option.compute_kernel] += shard.hbm

            if shard_by_fqn.get(fqn) is None:
                continue
            shard: ParameterSharding = shard_by_fqn[fqn]

            ranks, input_sizes, output_sizes = self._get_shard_stats(
                shard=shard,
                sharding_option=sharding_option,
                world_size=topology.world_size,
                local_world_size=topology.local_world_size,
                constraints=constraints,
            )
            sharding_type_abbr = _get_sharding_type_abbr(shard.sharding_type)
            used_sharding_types.add(sharding_type_abbr)

            for i, rank in enumerate(ranks):
                count = stats[rank]["type"].get(sharding_type_abbr, 0)
                stats[rank]["type"][sharding_type_abbr] = count + 1
                stats[rank]["input_sizes"] += input_sizes[i]
                stats[rank]["output_sizes"] += output_sizes[i]

        sparse_hbm, used_hbm, used_ddr, perf = _compute_mem_usage_and_perf(
            topology=topology,
            best_plan=best_plan,
            dense_storage=dense_storage,
            kjt_storage=kjt_storage,
        )

        formatted_table = self._log_rank_mem_usage_and_perf(
            topology=topology,
            used_hbm=used_hbm,
            used_ddr=used_ddr,
            perf=perf,
            stats=stats,
            used_sharding_types=used_sharding_types,
            reserved_hbm_percent=reserved_hbm_percent,
        )

        if debug:
            formatted_param_table = self._log_sharding_plan(
                best_plan=best_plan,
                sharding_plan=sharding_plan,
                sharders=sharders,
                constraints=constraints,
            )

        self._stats_table.clear()
        self._stats_table.append("#" * self._width)
        header_text = "--- Planner Statistics ---"
        self._stats_table.append(f"#{header_text: ^{self._width-2}}#")

        iter_text = (
            f"--- Evaluated {num_proposals} proposal(s), "
            f"found {num_plans} possible plan(s), "
            f"ran for {run_time:.2f}s ---"
        )
        self._stats_table.append(f"#{iter_text: ^{self._width-2}}#")

        divider = "-" * (self._width - 4)
        self._stats_table.append(f"#{divider: ^{self._width-2}}#")

        if sharding_plan.plan:
            for row in formatted_table:
                self._stats_table.append(f"# {row: <{self._width-3}}#")

            perf_breakdown = "Perf: Total perf (Forward compute, Forward comms, Backward compute, Backward comms, Prefetch compute)"
            legend = (
                "Input: MB/iteration, Output: MB/iteration, Shards: number of tables"
            )
            hbm_info = "HBM: estimated peak memory usage for shards, dense tensors, and features (KJT)"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(f"# {perf_breakdown: <{self._width-3}}#")
            self._stats_table.append(f"# {legend: <{self._width-3}}#")
            self._stats_table.append(f"# {hbm_info: <{self._width-3}}#")

        if debug:
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(f"# {'Parameter Info:' : <{self._width-3}}#")
            for row in formatted_param_table:
                self._stats_table.append(f"# {row: <{self._width-3}}#")

        batch_size_text = f"Batch Size: {batch_size}"
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {batch_size_text : <{self._width-3}}#")

        if not sharding_plan.plan:
            rank_size_text = f"World Size: {topology.world_size}"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(f"# {rank_size_text : <{self._width-3}}#")

        self._log_compute_kernel_stats(
            compute_kernels_to_count, description="Compute Kernels Count"
        )
        self._log_compute_kernel_stats(
            {
                k: f"HBM: {round(bytes_to_gb(s.hbm),3)} GB, DDR: {round(bytes_to_gb(s.ddr),3)} GB"
                for k, s in compute_kernels_to_storage.items()
            },
            description="Compute Kernels Storage",
        )

        if debug:
            if sharding_plan.plan:
                # Plan imbalance stats for perf and storage
                self._log_plan_imbalance_stats(
                    perf,
                    used_hbm,
                    used_ddr,
                )

                # Max perf and HBM to help root cause imbalance
                self._log_max_perf_and_max_hbm(perf, sparse_hbm, used_hbm, best_plan)
            self._log_storage_reservation_stats(
                storage_reservation,
                topology,
                reserved_hbm_percent,
                dense_storage,
                kjt_storage,
            )
            if sharding_plan.plan:
                self._log_imbalance_tables(best_plan)

        self._stats_table.append("#" * self._width)

        for row in self._stats_table:
            logger.info(row)

    def _get_shard_stats(
        self,
        shard: ParameterSharding,
        sharding_option: ShardingOption,
        world_size: int,
        local_world_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Gets ranks, input sizes, and output sizes per shard.
        Input size is a function of pooling factor.
        Output size is a function of embedding dimension * number of features.

        Returns:
            ranks: list of ranks.
            input_sizes: input size per iter in MB across ranks for given shard.
            output_sizes: output size per iter in MB across ranks for given shard.
        """
        assert shard.ranks
        ranks = shard.ranks

        num_poolings = get_num_poolings(constraints, sharding_option)
        batch_sizes = (
            cast(List[int], constraints[sharding_option.name].batch_sizes)
            if constraints
            and constraints.get(sharding_option.name)
            and constraints[sharding_option.name].batch_sizes
            else [sharding_option.batch_size] * sharding_option.num_inputs
        )
        input_data_type_size = BIGINT_DTYPE
        output_data_type_size = sharding_option.tensor.element_size()

        input_sizes, output_sizes = _calculate_shard_io_sizes(
            sharding_type=sharding_option.sharding_type,
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=sharding_option.input_lengths,
            emb_dim=sharding_option.tensor.shape[1],
            shard_sizes=[shard.size for shard in sharding_option.shards],
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=sharding_option.is_pooled,
        )

        input_sizes = [bytes_to_mb(input_size) for input_size in input_sizes]
        output_sizes = [bytes_to_mb(output_size) for output_size in output_sizes]

        return ranks, input_sizes, output_sizes

    def _log_dist_imbalance_stats(
        self,
        normalized_dist: List[float],
    ) -> None:
        for name, (measure, kwargs) in IMBALANCE_STAT_MEASURE.items():
            result_txt = f"{name}: {measure(normalized_dist, **kwargs):.3f}"
            self._stats_table.append(f"# {result_txt : <{self._width-3}}#")

    def _log_plan_imbalance_stats(
        self, perf: List[Perf], used_hbm: List[int], used_ddr: List[int]
    ) -> None:
        imbalance_logged = False
        total_perfs = [perf_i.total for perf_i in perf]

        # Can extend with fwd/bwd perfs if needed
        perf_dists = [
            ("Total", total_perfs),
        ]

        for name, perf_dist in perf_dists:
            if sum(perf_dist) > 0:
                imbalance_logged = True
                self._stats_table.append(f"#{'' : ^{self._width-2}}#")
                self._stats_table.append(
                    f"# {name + ' Perf Imbalance Statistics' : <{self._width-3}}#"
                )
                normalized_perf_dist = _normalize_float(perf_dist)
                self._log_dist_imbalance_stats(normalized_perf_dist)

        if sum(used_hbm) > 0:
            imbalance_logged = True
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'HBM Imbalance Statistics' : <{self._width-3}}#"
            )
            normalized_used_hbm = _normalize_int(used_hbm)
            self._log_dist_imbalance_stats(normalized_used_hbm)

        if sum(used_ddr) > 0:
            imbalance_logged = True
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'DDR Imbalance Statistics' : <{self._width-3}}#"
            )
            normalized_used_ddr = _normalize_int(used_ddr)
            self._log_dist_imbalance_stats(normalized_used_ddr)

        if imbalance_logged:
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'Imbalance stats range 0-1, higher means more imbalanced' : <{self._width-3}}#"
            )

    def _log_max_perf_and_max_hbm(
        self,
        perfs: List[Perf],
        sparse_hbm: List[int],
        used_hbm: List[int],
        best_plan: List[ShardingOption],
    ) -> None:
        total_perfs = [perf.total for perf in perfs]

        max_total_perf_text = (
            f"Maximum of Total Perf: {_generate_max_text(total_perfs)}"
        )

        mean_total_perf = statistics.mean(total_perfs)
        mean_total_perf_text = f"Mean Total Perf: {round(mean_total_perf,3)} ms"

        max_total_perf = max(total_perfs)

        total_perf_delta_pct = 0.0
        if mean_total_perf > 0.0:
            total_perf_delta_pct = (
                (max_total_perf - mean_total_perf) / mean_total_perf * 100
            )

        total_perf_delta_text = (
            f"Max Total Perf is {total_perf_delta_pct:.3g}% greater than the mean"
        )

        max_fwd_compute_perf_text = f"Maximum of Forward Compute: {_generate_max_text([perf.fwd_compute for perf in perfs])}"
        max_fwd_comms_perf_text = f"Maximum of Forward Comms: {_generate_max_text([perf.fwd_comms for perf in perfs])}"
        max_bwd_compute_perf_text = f"Maximum of Backward Compute: {_generate_max_text([perf.bwd_compute for perf in perfs])}"
        max_bwd_comms_perf_text = f"Maximum of Backward Comms: {_generate_max_text([perf.bwd_comms for perf in perfs])}"
        max_prefetch_compute_perf_text = f"Maximum of Prefetch Compute: {_generate_max_text([perf.prefetch_compute for perf in perfs])}"

        sum_of_maxima = (
            max(perf.fwd_compute for perf in perfs)
            + max(perf.fwd_comms for perf in perfs)
            + max(perf.bwd_compute for perf in perfs)
            + max(perf.bwd_comms for perf in perfs)
            + max(perf.prefetch_compute for perf in perfs)
        )
        sum_of_maxima_text = f"Sum of Maxima: {round(sum_of_maxima, 3)} ms"

        critical_path_estimate = _calculate_critical_path(best_plan)

        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {max_total_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {mean_total_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {total_perf_delta_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_fwd_compute_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_fwd_comms_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_bwd_compute_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_bwd_comms_perf_text : <{self._width-3}}#")
        self._stats_table.append(
            f"# {max_prefetch_compute_perf_text : <{self._width-3}}#"
        )
        self._stats_table.append(f"# {sum_of_maxima_text : <{self._width-3}}#")

        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(
            f"# {'Estimated Sharding Distribution' : <{self._width-2}}#"
        )
        self._stats_table.append(
            f"# {'Sparse only Max HBM: '+_generate_rank_hbm_stats(sparse_hbm, max) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Sparse only Min HBM: '+_generate_rank_hbm_stats(sparse_hbm, min) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Max HBM: '+_generate_rank_hbm_stats(used_hbm, max) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Min HBM: '+_generate_rank_hbm_stats(used_hbm, min) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Mean HBM: '+_generate_rank_hbm_stats(used_hbm, statistics.mean) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Low Median HBM: '+_generate_rank_hbm_stats(used_hbm, statistics.median_low) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'High Median HBM: '+_generate_rank_hbm_stats(used_hbm, statistics.median_high) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Critical Path (comms): '+str(round(critical_path_estimate.comms_estimate, 3)) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Critical Path (compute): '+str(round(critical_path_estimate.comp_estimate, 3)) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Critical Path (comms + compute): '+str(round(critical_path_estimate.total(), 3)) : <{self._width-3}}#"
        )

        max_used_hbm = max(used_hbm)
        mean_used_hbm = statistics.mean(used_hbm)
        hbm_delta_pct = 0.0
        if mean_used_hbm > 0.0:
            hbm_delta_pct = (max_used_hbm - mean_used_hbm) / mean_used_hbm * 100
        hbm_delta_text = f"Max HBM is {hbm_delta_pct:.3g}% greater than the mean"
        self._stats_table.append(f"# {hbm_delta_text : <{self._width-3}}#")

        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        per_rank_hbm = copy.copy(used_hbm)
        NUM_PEAK_RANK = 5
        peak_memory_pressure = []

        top_hbm_usage_estimation = f"Top HBM Memory Usage Estimation: {round(bytes_to_gb(max(used_hbm)), 3)} GB"
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {top_hbm_usage_estimation : <{self._width-3}}#")

        for top in range(NUM_PEAK_RANK):
            if not per_rank_hbm:
                break
            max_hbm = max(per_rank_hbm)
            max_hbm_indices = [
                i
                for i in range(len(per_rank_hbm))
                if math.isclose(
                    bytes_to_mb(per_rank_hbm[i]), bytes_to_mb(max_hbm), abs_tol=1.0
                )
            ]
            rank_text = "ranks" if len(max_hbm_indices) > 1 else "rank"
            max_hbm_indices = _collapse_consecutive_ranks(max_hbm_indices)
            max_hbm_ranks = f"{rank_text} {','.join(max_hbm_indices)}"
            peak_memory_pressure.append(
                f"Top Tier #{top+1} Estimated Peak HBM Pressure: {round(bytes_to_gb(max_hbm), 3)} GB on {max_hbm_ranks}"
            )
            per_rank_hbm = [
                hbm
                for hbm in per_rank_hbm
                if not math.isclose(bytes_to_mb(hbm), bytes_to_mb(max_hbm), abs_tol=1.0)
            ]

        for peak_rank in reversed(peak_memory_pressure):
            self._stats_table.append(f"# {peak_rank : <{self._width-3}}#")

    def _log_storage_reservation_stats(
        self,
        storage_reservation: StorageReservation,
        topology: Topology,
        reserved_hbm_percent: float,
        dense_storage: Storage,
        kjt_storage: Storage,
    ) -> None:
        device_storage = topology.devices[0].storage
        usable_hbm = round(
            bytes_to_gb(int((1 - reserved_hbm_percent) * device_storage.hbm)), 3
        )
        reserved_hbm = round(
            bytes_to_gb(int(reserved_hbm_percent * device_storage.hbm)), 3
        )
        reserved_memory = f"HBM: {reserved_hbm} GB"
        reserved_hbm_percentage = f"Percent of Total HBM: {reserved_hbm_percent:.0%}"
        usable_ddr = round(bytes_to_gb(int(device_storage.ddr)), 3)
        usable_memory = f"HBM: {usable_hbm} GB, DDR: {usable_ddr} GB"
        usable_hbm_percentage = (
            f"Percent of Total HBM: {(1 - reserved_hbm_percent):.0%}"
        )
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {'Reserved Memory:' : <{self._width-3}}#")
        self._stats_table.append(f"#    {reserved_memory : <{self._width-6}}#")
        self._stats_table.append(f"#    {reserved_hbm_percentage : <{self._width-6}}#")
        self._stats_table.append(f"# {'Planning Memory:' : <{self._width-3}}#")
        self._stats_table.append(f"#    {usable_memory : <{self._width-6}}#")
        self._stats_table.append(f"#    {usable_hbm_percentage : <{self._width-6}}#")

        if isinstance(storage_reservation, HeuristicalStorageReservation):
            dense_hbm = round(bytes_to_gb(dense_storage.hbm), 3)
            dense_ddr = round(bytes_to_gb(dense_storage.ddr), 3)
            dense_storage_text = f"HBM: {dense_hbm} GB, DDR: {dense_ddr} GB"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'Dense Storage (per rank): ' : <{self._width-3}}#"
            )
            self._stats_table.append(f"#    {dense_storage_text : <{self._width-6}}#")

            kjt_hbm = round(bytes_to_gb(kjt_storage.hbm), 3)
            kjt_ddr = round(bytes_to_gb(kjt_storage.ddr), 3)
            kjt_storage_text = f"HBM: {kjt_hbm} GB, DDR: {kjt_ddr} GB"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'KJT Storage (per rank): ' : <{self._width-3}}#"
            )
            self._stats_table.append(f"#    {kjt_storage_text : <{self._width-6}}#")

    def _log_imbalance_tables(self, best_plan: List[ShardingOption]) -> None:
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        perf_imbalance_tables = _find_imbalance_tables(best_plan)
        hbm_imbalance_tables = _find_imbalance_tables(best_plan, target_imbalance="hbm")
        self._stats_table.append(
            f"# {'Top 5 Tables Causing Max Perf:' : <{self._width-3}}#"
        )
        for sharding_option in perf_imbalance_tables[0:5]:
            self._stats_table.append(f"#    {sharding_option.name : <{self._width-6}}#")
        self._stats_table.append(
            f"# {'Top 5 Tables Causing Max HBM:' : <{self._width-3}}#"
        )
        for sharding_option in hbm_imbalance_tables[0:5]:
            storage = sharding_option.shards[0].storage
            assert storage is not None  # linter friendly optional check

            rank_text = "ranks" if len(sharding_option.shards) > 1 else "rank"
            top_table = (
                f"{sharding_option.name}: {round(bytes_to_gb(storage.hbm),3)} GB on {rank_text} "
                f"{[shard.rank for shard in sharding_option.shards]}"
            )
            self._stats_table.append(f"#    {top_table : <{self._width-6}}#")

    def _log_compute_kernel_stats(
        self, compute_kernels_stats: Dict[str, Any], description: str
    ) -> None:
        compute_kernels_count = [
            f"{compute_kernel}: {count}"
            for compute_kernel, count in sorted(compute_kernels_stats.items())
        ]
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {description+':' : <{self._width-3}}#")
        for compute_kernel_count in compute_kernels_count:
            self._stats_table.append(f"#    {compute_kernel_count : <{self._width-6}}#")

    def _log_rank_mem_usage_and_perf(
        self,
        topology: Topology,
        used_ddr: List[int],
        used_hbm: List[int],
        perf: List[Perf],
        stats: Dict[int, Dict[str, Any]],
        used_sharding_types: Set[str],
        reserved_hbm_percent: float,
    ) -> List[str]:
        table: List[List[Union[str, int]]] = [
            [
                "Rank",
                "HBM (GB)",
                "DDR (GB)",
                "Perf (ms)",
                "Input (MB)",
                "Output (MB)",
                "Shards",
            ],
            [
                "------",
                "----------",
                "----------",
                "-----------",
                "------------",
                "-------------",
                "--------",
            ],
        ]

        for rank, device in enumerate(topology.devices):
            used_hbm_gb = bytes_to_gb(used_hbm[rank])
            used_hbm_ratio = (
                used_hbm[rank] / ((1 - reserved_hbm_percent) * device.storage.hbm)
                if (
                    topology.compute_device == "cuda"
                    or topology.compute_device == "mtia"
                )
                and ((1 - reserved_hbm_percent) * device.storage.hbm) != 0
                else 0
            )
            used_ddr_gb = bytes_to_gb(used_ddr[rank])
            used_ddr_ratio = (
                used_ddr[rank] / device.storage.ddr if device.storage.ddr > 0 else 0
            )
            for sharding_type in used_sharding_types:
                if sharding_type not in stats[rank]["type"]:
                    stats[rank]["type"][sharding_type] = 0

            rank_hbm = f"{round(used_hbm_gb, 3)} ({used_hbm_ratio:.0%})"
            rank_ddr = f"{round(used_ddr_gb, 3)} ({used_ddr_ratio:.0%})"
            rank_perf = _format_perf_breakdown(perf[rank])
            rank_input = f"{round(stats[rank]['input_sizes'], 2)}"
            rank_output = f"{round(stats[rank]['output_sizes'], 2)}"
            rank_shards = " ".join(
                f"{sharding_type}: {num_tables}"
                for sharding_type, num_tables in sorted(stats[rank]["type"].items())
            )
            table.append(
                [
                    rank,
                    rank_hbm,
                    rank_ddr,
                    rank_perf,
                    rank_input,
                    rank_output,
                    rank_shards,
                ]
            )
        formatted_table = _format_table(table)
        self._width = max(self._width, len(formatted_table[0]) + 8)
        return formatted_table

    def _log_sharding_plan(
        self,
        best_plan: List[ShardingOption],
        sharding_plan: ShardingPlan,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
    ) -> List[str]:
        def _get_embedding_dim(so: ShardingOption) -> str:
            embedding_dim = (
                f"{so.tensor.shape[1]} ({so.shards[0].size[1]})"
                if so.sharding_type == ShardingType.COLUMN_WISE.value
                or so.sharding_type == ShardingType.TABLE_COLUMN_WISE.value
                or so.sharding_type == ShardingType.GRID_SHARD.value
                else f"{so.tensor.shape[1]}"
            )
            return embedding_dim

        def _get_cache_load_factor(
            sharder: Optional[ModuleSharder[nn.Module]], so: ShardingOption
        ) -> str:
            sharder_cache_load_factor = (
                sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                if hasattr(sharder, "fused_params") and sharder.fused_params
                else None
            )
            cache_load_factor = "None"
            # Surfacing cache load factor does not make sense if not using uvm caching.
            if so.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING.value:
                cache_load_factor = str(
                    so.cache_load_factor
                    if so.cache_load_factor is not None
                    else sharder_cache_load_factor
                )
            return cache_load_factor

        param_table = [
            [
                "FQN",
                "Sharding",
                "Compute Kernel",
                "Perf (ms)",
                "Storage (HBM, DDR)",
                "Cache Load Factor",
                "Sum Pooling Factor",
                "Sum Num Poolings",
                "Num Indices",
                "Output",
                "Weighted",
                "Sharder",
                "Features",
                "Emb Dim (CW Dim)",
                "Hash Size",
                "Ranks",
            ],
            [
                "-----",  # FQN
                "----------",  # Sharding
                "----------------",  # Compute Kernel
                "-----------",  # Perf (ms)
                "--------------------",  # Storage (HBM, DDR)
                "-------------------",  # Cache Load Factor
                "--------------------",  # Sum Pooling Factor
                "------------------",  # Sum Num Poolings
                "-------------",  # Num Indices
                "--------",  # Output
                "----------",  # Weighted
                "---------",  # Sharder
                "----------",  # Features
                "------------------",  # Emb Dim (CW Dim)
                "-----------",  # Hash Size
                "-------",  # Ranks
            ],
        ]
        feat_batch_sizes = [
            (
                constraints[so.name].batch_sizes
                if constraints and constraints.get(so.name)
                else None
            )
            for so in best_plan
        ]

        sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            get_sharder_name(sharder.module_type): sharder
            # pyre-ignore - this is a ModuleSharder below
            for sharder in sharders
            if sharders
        }

        if include_batch_sizes := any(feat_batch_sizes):
            param_table[0].append("Batch Sizes")
            param_table[1].append("-------------")
        for i, so in enumerate(best_plan):
            ranks = sorted([cast(int, shard.rank) for shard in so.shards])
            ranks = _collapse_consecutive_ranks(ranks)

            so_perf = Perf(fwd_compute=0, fwd_comms=0, bwd_compute=0, bwd_comms=0)
            for shard in so.shards:
                so_perf += cast(Perf, shard.perf)

            shard_perfs = _format_perf_breakdown(so_perf)

            so_storage = Storage(hbm=0, ddr=0)
            for shard in so.shards:
                so_storage += cast(Storage, shard.storage)

            shard_storages = _format_storage_breakdown(so_storage)

            pooling_factor = str(round(sum(so.input_lengths), 3))
            num_poolings = get_num_poolings(constraints, so)
            num_indices = str(
                round(sum(x * y for x, y in zip(so.input_lengths, num_poolings)), 3)
            )
            num_poolings = str(round(sum(num_poolings), 3))
            output = "pooled" if so.is_pooled else "sequence"
            weighted = "weighted" if so.is_weighted else "unweighted"
            sharder = sharder_map.get(get_sharder_name(type(so.module[1])), None)
            sharder_name = type(sharder).__name__
            num_features = len(so.input_lengths)
            embedding_dim = _get_embedding_dim(so)
            cache_load_factor = _get_cache_load_factor(sharder, so)
            hash_size = so.tensor.shape[0]
            param_table.append(
                # pyre-ignore[6]
                [
                    so.fqn,
                    _get_sharding_type_abbr(so.sharding_type),
                    so.compute_kernel,
                    shard_perfs,
                    shard_storages,
                    cache_load_factor,
                    pooling_factor,
                    num_poolings,
                    num_indices,
                    output,
                    weighted,
                    sharder_name,
                    num_features,
                    embedding_dim,
                    hash_size,
                    ",".join(ranks) if sharding_plan.plan else "None",
                ]
            )
            if include_batch_sizes:
                bs = feat_batch_sizes[i]
                param_table[-1].append(_reduce_int_list(bs) if bs else "n/a")
        formatted_param_table = _format_table(param_table)  # pyre-ignore[6]
        self._width = max(self._width, len(formatted_param_table[0]) + 6)
        return formatted_param_table


def _generate_rank_hbm_stats(
    per_rank_hbm: List[int], func: Callable[[Iterable[float]], float]
) -> str:
    stats = round(func(per_rank_hbm))
    stats_indicies = [
        i
        for i in range(len(per_rank_hbm))
        if math.isclose(bytes_to_mb(per_rank_hbm[i]), bytes_to_mb(stats), abs_tol=1.0)
    ]
    rank_text = "ranks" if len(stats_indicies) > 1 else "rank"
    return f"{round(bytes_to_gb(stats), 3)} GB on {rank_text} {stats_indicies}"


def _generate_max_text(perfs: List[float]) -> str:
    max_perf = max(perfs)

    max_perf_indices = [i for i in range(len(perfs)) if perfs[i] == max_perf]
    rank_text = "ranks" if len(max_perf_indices) > 1 else "rank"
    max_perf_indices = _collapse_consecutive_ranks(max_perf_indices)
    max_perf_ranks = f"{rank_text} {','.join(max_perf_indices)}"

    return f"{round(max_perf, 3)} ms on {max_perf_ranks}"


def _get_sharding_type_abbr(sharding_type: str) -> str:
    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return "DP"
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return "TW"
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return "CW"
    elif sharding_type == ShardingType.ROW_WISE.value:
        return "RW"
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return "TWRW"
    elif sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
        return "TWCW"
    elif sharding_type == ShardingType.GRID_SHARD.value:
        return "GS"
    else:
        raise ValueError(
            f"Unrecognized or unsupported sharding type provided: {sharding_type}"
        )


def _format_perf_breakdown(perf: Perf) -> str:
    breakdown = [
        perf.fwd_compute,
        perf.fwd_comms,
        perf.bwd_compute,
        perf.bwd_comms,
        perf.prefetch_compute,
    ]
    breakdown_string = ",".join(
        [str(round(num)) if num >= 1 else round_to_one_sigfig(num) for num in breakdown]
    )

    return f"{str(round(perf.total, 3))} ({breakdown_string})"


def _compute_storage(
    storage_reservation: StorageReservation,
) -> Tuple[float, Storage, Storage]:
    reserved_hbm_percent = (
        storage_reservation._percentage
        if isinstance(
            storage_reservation,
            (
                FixedPercentageStorageReservation,
                HeuristicalStorageReservation,
                InferenceStorageReservation,
            ),
        )
        else 0.0
    )

    dense_storage = (
        storage_reservation._dense_storage
        if isinstance(
            storage_reservation,
            (HeuristicalStorageReservation, InferenceStorageReservation),
        )
        and storage_reservation._dense_storage is not None
        else Storage(0, 0)
    )
    assert dense_storage
    kjt_storage = (
        storage_reservation._kjt_storage
        if isinstance(
            storage_reservation,
            (HeuristicalStorageReservation, InferenceStorageReservation),
        )
        and storage_reservation._kjt_storage
        else Storage(0, 0)
    )
    assert kjt_storage
    return reserved_hbm_percent, dense_storage, kjt_storage


def _compute_mem_usage_and_perf(
    topology: Topology,
    best_plan: List[ShardingOption],
    dense_storage: Storage,
    kjt_storage: Storage,
) -> Tuple[List[int], List[int], List[int], List[Perf]]:
    sparse_hbm = [0] * topology.world_size
    used_ddr = [0] * topology.world_size
    perf = [
        Perf(fwd_compute=0, fwd_comms=0, bwd_compute=0, bwd_comms=0)
        for _ in range(topology.world_size)
    ]
    for sharding_option in best_plan:
        for shard in sharding_option.shards:
            shard_storage = cast(Storage, shard.storage)
            rank = cast(int, shard.rank)
            sparse_hbm[rank] += shard_storage.hbm
            used_ddr[rank] += shard_storage.ddr
            perf[rank] += cast(Perf, shard.perf)
    used_hbm = [hbm + dense_storage.hbm + kjt_storage.hbm for hbm in sparse_hbm]
    used_ddr = [ddr + dense_storage.ddr + kjt_storage.ddr for ddr in used_ddr]
    return sparse_hbm, used_hbm, used_ddr, perf


def _format_storage_breakdown(storage: Storage) -> str:
    storage_hbm = round(bytes_to_gb(storage.hbm), 3)
    storage_ddr = round(bytes_to_gb(storage.ddr), 3)
    return f"({storage_hbm} GB, {storage_ddr} GB)"


def round_to_one_sigfig(x: float) -> str:
    return f'{float(f"{x:.1g}"):g}'


def _format_table(table: List[List[Union[str, int]]]) -> List[str]:
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3) for i in range(len(table[0]))
    ]
    row_format = "".join(
        ["{:>" + str(longest_col) + "}" for longest_col in longest_cols]
    )
    return [row_format.format(*row) for row in table]


def _collapse_consecutive_ranks(ranks: List[int]) -> List[str]:
    if len(ranks) > 1 and ranks == list(range(min(ranks), max(ranks) + 1)):
        return [f"{min(ranks)}-{max(ranks)}"]
    else:
        return [str(rank) for rank in ranks]


def _reduce_int_list(input_list: List[int]) -> str:
    if len(input_list) == 0:
        return ""
    reduced = []
    count = 1
    prev_num = input_list[0]

    for num in input_list[1:]:
        if num == prev_num:
            count += 1
        else:
            if count > 1:
                reduced.append(f"{prev_num} * {count}")
            else:
                reduced.append(str(prev_num))
            prev_num = num
            count = 1

    # Handle the last number
    if count > 1:
        reduced.append(f"{prev_num}*{count}")
    else:
        reduced.append(str(prev_num))

    return ", ".join(reduced)


def _calculate_critical_path(best_plan: List[ShardingOption]) -> CriticalPathEstimate:
    """
    Calculates the critical path of the sharding plan. Makes the following assumptions:

        1. There is a synchronization point across the ranks after each of the 4 events: Fwd/Bwd x Comms/Comp.
        2. There are additional synchronization points during communication (both fwd & bwd) for each module <> sharding type combination.
            i. Communication operations for each shard from the same module <> sharding type group are executed sequentially.
            ii. Ranks need to synchronize before they can begin the communication operation for the next module <> sharding type group.
        3. There are additional synchronization points during computation (both fwd & bwd) at the rank level.
            i. Computation operations for each shard from the same module are executed sequentially.
            ii. Ranks need to synchronize before they can begin the next set of events.
    """
    comms_data = defaultdict(lambda: defaultdict(float))
    comp_data = defaultdict(lambda: defaultdict(float))
    for so in best_plan:
        module = so.module
        sharding_type = so.sharding_type
        for shard in so.shards:
            rank = cast(int, shard.rank)
            perf = cast(Perf, shard.perf)
            comms_data[(module, sharding_type, "fwd")][rank] += perf.fwd_comms
            comms_data[(module, sharding_type, "bwd")][rank] += perf.bwd_comms
            comp_data["fwd"][rank] += perf.fwd_compute
            comp_data["bwd"][rank] += perf.bwd_compute
    comms_rank_agg = {
        outer_key: max(inner_dict.values())
        for outer_key, inner_dict in comms_data.items()
    }
    rank_count = len({cast(int, shard.rank) for so in best_plan for shard in so.shards})
    sharding_types = list({so.sharding_type for so in best_plan})
    comms_estimate = sum(comms_rank_agg.values())
    comp_rank_agg = {
        outer_key: max(inner_dict.values())
        for outer_key, inner_dict in comp_data.items()
    }
    comp_estimate = sum(comp_rank_agg.values())

    return CriticalPathEstimate(comms_estimate, comp_estimate)


class NoopEmbeddingStats(Stats):
    """
    Noop Stats for a sharding planner execution.
    """

    def log(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        batch_size: int,
        storage_reservation: StorageReservation,
        num_proposals: int,
        num_plans: int,
        run_time: float,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        enumerator: Optional[Enumerator] = None,
        debug: bool = True,
        debug_stats: Optional[PlanDebugStats] = None,
    ) -> None:
        pass
