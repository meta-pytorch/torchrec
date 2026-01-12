#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import itertools
import logging
from collections import OrderedDict
from decimal import Decimal
from typing import Callable, cast, Dict, List, Optional, Set, Tuple, TypeVar, Union

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.types import (
    Enumerator,
    Perf,
    Proposer,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import bytes_to_gb, LuusJaakolaSearch, prod
from torchrec.distributed.types import CacheAlgorithm

logger: logging.Logger = logging.getLogger(__name__)

MAX_PROPOSALS: int = int(1e4)


class GreedyProposer(Proposer):
    """
    Proposes sharding plans in greedy fashion.

    Sorts sharding options for each shardable parameter by perf.
    On each iteration, finds parameter with largest current storage usage and tries its
    next sharding option.

    Args:
        use_depth (bool): When enabled, sharding_options of a fqn are sorted based on
            `max(shard.perf.total)`, otherwise sharding_options are sorted by
            `sum(shard.perf.total)`.
        threshold (Optional[int]): Threshold for early stopping. When specified, the
            proposer stops proposing when the proposals have consecutive worse perf_rating
            than best_perf_rating.
    """

    def __init__(self, use_depth: bool = True, threshold: Optional[int] = None) -> None:
        self._use_depth: bool = use_depth
        self._threshold: Optional[int] = threshold if threshold else None
        self._sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        self._current_proposal: Dict[str, int] = {}
        self._best_perf_rating: float = float("inf")
        self._num_inferior_perf: int = 0

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self._reset()
        for sharding_option in search_space:
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

        for sharding_options in self._sharding_options_by_fqn.values():
            sharding_options.sort(
                key=lambda x: _sharding_option_score(x, self._use_depth)
            )

        self._current_proposal = {
            fqn: 0 for fqn in self._sharding_options_by_fqn.keys()
        }

    def _reset(self) -> None:
        self._sharding_options_by_fqn = {}
        self._current_proposal = {}

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._current_proposal:
            return [
                self._sharding_options_by_fqn[fqn][index]
                for fqn, index in self._current_proposal.items()
            ]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        # When threshold is passed, observe the perf_rating trend. If the perf_rating
        # of the newly proposed plans have worse perf_rating, stop proposing.
        if self._threshold and perf_rating:
            self._num_inferior_perf += 1
            if perf_rating < self._best_perf_rating:
                self._best_perf_rating = perf_rating
                self._num_inferior_perf = 0
            # pyre-fixme [58]: `>` is not supported for operand types `int` and `Optional[int]`.
            if self._num_inferior_perf > self._threshold:
                self._current_proposal = {}
                return
        # static strategy, ignore feedback and just provide next proposal
        largest_fqn: Optional[str] = None
        largest_storage: Tuple[float, float, float, float] = (0, 0, 0, 0)
        for fqn, sharding_options in self._sharding_options_by_fqn.items():
            index = self._current_proposal[fqn]
            if index + 1 < len(sharding_options):
                sharding_option = sharding_options[index]
                current_storage = (
                    # pyre-fixme [16]: `Optional` has no attribute `hbm`
                    max([shard.storage.hbm for shard in sharding_option.shards]),
                    sum([shard.storage.hbm for shard in sharding_option.shards]),
                    # pyre-fixme [16]: `Optional` has no attribute `ddr`
                    max([shard.storage.ddr for shard in sharding_option.shards]),
                    sum([shard.storage.ddr for shard in sharding_option.shards]),
                )
                if current_storage > largest_storage:
                    largest_fqn = fqn
                    largest_storage = current_storage

        if largest_fqn is not None:
            self._current_proposal[largest_fqn] += 1
        else:
            self._current_proposal = {}


class UniformProposer(Proposer):
    """
    Proposes uniform sharding plans, plans that have the same sharding type for all
    sharding options.
    """

    def __init__(self, use_depth: bool = True) -> None:
        self._use_depth: bool = use_depth
        self._grouped_sharding_options: List[List[ShardingOption]] = []
        self._proposal_index: int = 0

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self._reset()
        all_fqns = set()
        sharding_options_by_type_and_fqn: Dict[str, Dict[str, List[ShardingOption]]] = (
            {}
        )

        for sharding_option in search_space:
            sharding_type = sharding_option.sharding_type
            fqn = sharding_option.fqn
            all_fqns.add(fqn)

            if sharding_type not in sharding_options_by_type_and_fqn:
                sharding_options_by_type_and_fqn[sharding_type] = {}
            if fqn not in sharding_options_by_type_and_fqn[sharding_type]:
                sharding_options_by_type_and_fqn[sharding_type][fqn] = []

            sharding_options_by_type_and_fqn[sharding_type][fqn].append(sharding_option)

        for sharding_options_by_fqn in sharding_options_by_type_and_fqn.values():
            for sharding_options in sharding_options_by_fqn.values():
                sharding_options.sort(
                    key=lambda x: _sharding_option_score(x, self._use_depth)
                )

        for sharding_options_by_fqn in sharding_options_by_type_and_fqn.values():
            if sharding_options_by_fqn.keys() == all_fqns:
                self._grouped_sharding_options.append(
                    [
                        sorted_sharding_options[0]
                        for sorted_sharding_options in sharding_options_by_fqn.values()
                    ]
                )

    def _reset(self) -> None:
        self._grouped_sharding_options = []
        self._proposal_index = 0

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._proposal_index < len(self._grouped_sharding_options):
            return self._grouped_sharding_options[self._proposal_index]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        self._proposal_index += 1


class GridSearchProposer(Proposer):
    def __init__(self, max_proposals: int = MAX_PROPOSALS) -> None:
        self._max_proposals: int = max_proposals
        self._sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        self._proposal_index: int = 0
        self._proposals: List[List[int]] = []

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self._reset()
        for sharding_option in search_space:
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

        for sharding_options in self._sharding_options_by_fqn.values():
            sharding_options.sort(key=lambda x: _sharding_option_score(x))

        total_proposals = prod(
            [
                len(sharding_options)
                for sharding_options in self._sharding_options_by_fqn.values()
            ]
        )
        if total_proposals > self._max_proposals:
            total_proposals = (
                "{:.2e}".format(Decimal(total_proposals))
                if total_proposals > 1e6
                else total_proposals
            )
            logger.info(
                "Skipping grid search proposer as there are too many proposals.\n"
                f"Total proposals to search: {total_proposals}\n"
                f"Max proposals allowed: {self._max_proposals}\n"
            )
            return
        sharding_options_by_fqn_indices = [
            range(len(sharding_options))
            for sharding_options in self._sharding_options_by_fqn.values()
        ]
        # pyre-fixme[8]: Attribute has type `List[List[int]]`; used as
        #  `List[Tuple[int]]`.
        self._proposals = list(itertools.product(*sharding_options_by_fqn_indices))

    def _reset(self) -> None:
        self._sharding_options_by_fqn = {}
        self._proposal_index = 0
        self._proposals = []

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._proposals and self._proposal_index < len(self._proposals):
            proposal_indices = self._proposals[self._proposal_index]
            return [
                sharding_options[index]
                for index, sharding_options in zip(
                    proposal_indices, self._sharding_options_by_fqn.values()
                )
            ]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        self._proposal_index += 1


def _bytes_to_float_bin(num_bytes: Union[float, int], bin_size: float) -> float:
    return float(num_bytes) / bin_size


class DynamicProgrammingProposer(Proposer):
    r"""Proposes sharding plans in dynamic programming fashion.

        The problem of the Embedding Sharding Plan can be framed as follows: Given
    :math:`M` tables and their corresponding :math:`N` Sharding Options, we need to
    select one sharding option for each table such that the total performance is
    minimized, while keeping the overall HBM constraint :math:`K` in check. This can
    be abstracted into the following mathematical formulation:

    Given a matrix :math:`A` of dimensions :math:`(M, N)` and another matrix :math:`B`
    of the same dimensions, let the elements of matrix :math:`A` be denoted as
    :math:`a_{i,j}` and the elements of matrix :math:`B` as :math:`b_{i,j}`. We aim
    to find a set of column indices :math:`\{ j_0, j_1, \ldots, j_{M-1} \}` such that
    the following conditions are satisfied:

    1. :math:`\sum_{i=0}^{M-1} a_{i,j_i} \leq K`, where :math:`K` is a float.
    2. :math:`\sum_{i=0}^{M-1} b_{i,j_i}` is minimized.

    This problem can be tackled using dynamic programming. First, discretize :math:`K`
    into :math:`K_i`, and denote the discretization function as :math:`f`.

    Define the state :math:`dp[i][f(k)]` to represent the minimum value of :math:`B`
    when considering the first :math:`i` rows and the total sum of :math:`A` is equal to
    the discretized value :math:`k`.

    The state transition can then be represented as:

    .. math::
        dp[i][f(k)] = \min_{j=0}^{N-1} \left( dp[i-1][f(k - A[i][j])] + B[i][j] \right)

    Since :math:`K` is the sum allocated across all HBM, simply satisfying that the
    total HBM in the plan equals :math:`K` does not guarantee that the allocation will
    fit on all cards. Therefore, it is essential to maintain all the states of the last
    layer of :math:`dp`. This allows us to propose different plans under varying total
    HBM constraints.

    Args:
        hbm_bins_per_device (int): hdm bins for dynamic programming precision.
    """

    def __init__(self, hbm_bins_per_device: int = 100) -> None:
        self._inited: bool = False
        self._hbm_bins_per_device: int = max(hbm_bins_per_device, 1)
        self._sharding_options_by_fqn: OrderedDict[str, List[ShardingOption]] = (
            OrderedDict()
        )
        # list of proposals with different total_hbm, a proposal is a list of indices of sharding_options
        self._proposal_list: List[List[int]] = []
        self._current_proposal: int = -1

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        """Load search space."""
        self._reset()
        # order the sharding_option by total_storage.hbm from low to high
        for sharding_option in sorted(search_space, key=lambda x: x.total_storage.hbm):
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

    def _reset(self) -> None:
        self._sharding_options_by_fqn = OrderedDict()
        self._proposal_list = []
        self._current_proposal = -1

    def propose(self) -> Optional[List[ShardingOption]]:
        """Propose a sharding plan."""
        if not self._inited:
            return [
                sharding_options[0]
                for sharding_options in self._sharding_options_by_fqn.values()
            ]
        elif self._current_proposal >= 0:
            proposal_index = self._proposal_list[self._current_proposal]
            return [
                self._sharding_options_by_fqn[fqn][index]
                for fqn, index in zip(
                    self._sharding_options_by_fqn.keys(), proposal_index
                )
            ]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        """Feedback last proposed plan."""
        if not self._inited:
            self._inited = True
            table_count = len(self._sharding_options_by_fqn)
            option_count = max([len(x) for x in self._sharding_options_by_fqn.values()])

            assert storage_constraint is not None
            # are we assuming the table will be evenly sharded on all devices?
            hbm_total = sum([x.storage.hbm for x in storage_constraint.devices])
            bin_count = self._hbm_bins_per_device * len(storage_constraint.devices)
            bin_size = float(hbm_total) / bin_count

            dp = [
                [(float("inf"), float("inf"))] * bin_count for _ in range(table_count)
            ]  # [table_id][hbm_bin][perf, hbm]

            backtrack = [
                [(-1, -1)] * bin_count for _ in range(table_count)
            ]  # [table_id][hbm_bin][opt_id, prev_hbm_bin]

            hbm_by_fqn = [
                [float("inf") for _ in range(option_count)] for _ in range(table_count)
            ]  # memory constraint lookup table: [table_id][sharding_option_id]
            perf_by_fqn = [
                [float("inf") for _ in range(option_count)] for _ in range(table_count)
            ]  # performance metrics lookup table: [table_id][sharding_option_id]

            # populate hbm and perf for each sharding option and table: A[table_id][sharding_option_id]
            for table_id, sharding_options in enumerate(
                self._sharding_options_by_fqn.values()
            ):
                for opt_id, sharding_option in enumerate(sharding_options):
                    hbm_by_fqn[table_id][opt_id] = _bytes_to_float_bin(
                        sharding_option.total_storage.hbm, bin_size
                    )
                    perf_by_fqn[table_id][opt_id] = sharding_option.total_perf

            table_0 = 0
            for opt_j in range(option_count):
                if hbm_by_fqn[0][opt_j] < bin_count:
                    hbm_i = int(hbm_by_fqn[0][opt_j])
                    # options are ordered in increasing order of hbm, we only want to consider
                    # a sharding option that has higher hbm and better perf (the smaller the better)
                    if dp[table_0][hbm_i][0] > perf_by_fqn[table_0][opt_j]:
                        dp[table_0][hbm_i] = (
                            perf_by_fqn[table_0][opt_j],
                            hbm_by_fqn[table_0][opt_j],
                        )
                        backtrack[table_0][hbm_i] = (opt_j, -1)

            # dp: table_count x option_count x bin_count
            for table_i in range(1, table_count):
                for opt_j in range(option_count):
                    for hbm in range(bin_count):
                        prev_perf, perv_hbm = dp[table_i - 1][hbm]
                        if prev_perf < float("inf"):
                            new_hbm = perv_hbm + hbm_by_fqn[table_i][opt_j]
                            if new_hbm < bin_count:
                                new_hbm_i = int(new_hbm)
                                new_perf = prev_perf + perf_by_fqn[table_i][opt_j]
                                if dp[table_i][new_hbm_i][0] > new_perf:
                                    dp[table_i][new_hbm_i] = (new_perf, new_hbm)
                                    backtrack[table_i][new_hbm_i] = (opt_j, hbm)
            self._proposal_list = []
            # fill in all the proposals, starting from highest hbm to lowest hbm
            for c in range(bin_count - 1, -1, -1):
                cur_opt_idx, cur_hbm_idx = backtrack[table_count - 1][c]
                if cur_opt_idx >= 0:
                    proposal_indices = [-1] * table_count
                    proposal_indices[table_count - 1] = cur_opt_idx
                    for i in range(table_count - 2, -1, -1):
                        proposal_indices[i], cur_hbm_idx = backtrack[i][cur_hbm_idx]
                    self._proposal_list.append(proposal_indices)
            if len(self._proposal_list) > 0:
                self._current_proposal = 0
        else:
            self._current_proposal += 1
            if self._current_proposal >= len(self._proposal_list):
                self._current_proposal = -1


_T = TypeVar("_T")


def _none_throws(x: Optional[_T]) -> _T:
    if x is None:
        raise AssertionError("unexpected None")
    return x


class EmbeddingOffloadScaleupProposer(Proposer):
    def __init__(self, use_depth: bool = True) -> None:
        self.use_depth: bool = use_depth
        self.enumerator: Optional[Enumerator] = None
        self.starting_proposal: List[ShardingOption] = []
        self.proposal: Optional[List[ShardingOption]] = None
        self.search: Optional[LuusJaakolaSearch] = None
        self.best_perf_rating: float = 1e99

    def _build_proposal_from_sharding_options(
        self,
        sharding_options_by_fqn: Dict[str, List[ShardingOption]],
    ) -> List[ShardingOption]:
        """
        Given a list of sharding options for each embedding table, selects which to include in the proposal.
        """
        # TODO(T206831945): Currently only uses 1 sharding option for proposal.
        # There is room for potentially exploring multiple options if done
        # carefully, e.g. traversing like GreedyProposer.
        proposal: List[ShardingOption] = []
        for table_sharding_options in sharding_options_by_fqn.values():
            if len(table_sharding_options) > 1:
                logger.warning(
                    f"EmbeddingOffloadScaleupProposer - ignored {len(table_sharding_options) - 1} sharding options for table {table_sharding_options[0].name} in proposal"
                )

            selected_option = next(
                (
                    sharding_option
                    for sharding_option in table_sharding_options
                    if sharding_option.compute_kernel
                    == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
                ),
                table_sharding_options[0],
            )
            proposal.append(selected_option)

            # Miss-ratio curves used for stats are modeled for an LRU cache, LFU cache would not work well with ScaleupProposer.
            if (
                selected_option.cache_params is not None
                and selected_option.cache_params.algorithm is not None
                and selected_option.cache_params.algorithm != CacheAlgorithm.LRU
            ):
                logger.error(
                    f"EmbeddingOffloadScaleupProposer - proposer only supports LRU cache algorithm, but {selected_option.cache_params.algorithm} is used for {selected_option}"
                )

        return proposal

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self.enumerator = enumerator
        sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        for sharding_option in search_space:
            sharding_options_by_fqn.setdefault(sharding_option.fqn, []).append(
                sharding_option
            )
        for sharding_options in sharding_options_by_fqn.values():
            sharding_options.sort(
                key=lambda x: _sharding_option_score(x, self.use_depth)
            )

        proposal: List[ShardingOption] = self._build_proposal_from_sharding_options(
            sharding_options_by_fqn
        )

        # deepcopy so it won't affect other proposers
        self.starting_proposal = copy.deepcopy(proposal)
        self.promote_high_prefetch_overheaad_table_to_hbm(
            self.enumerator, self.starting_proposal
        )
        self.proposal = copy.deepcopy(self.starting_proposal)

    @staticmethod
    def get_hbm_ceiling(
        starting_proposal: List[ShardingOption], enumerator: Enumerator
    ) -> int:
        """returns total amount of memory scaleup could use."""
        proposal = copy.deepcopy(starting_proposal)
        cache_tables = EmbeddingOffloadScaleupProposer.get_scalable_sharding_options(
            proposal
        )
        for sharding_option in cache_tables:
            if (
                sharding_option.compute_kernel
                == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
            ):
                assert sharding_option.cache_params  # appease pyre
                sharding_option.cache_params.load_factor = None
                sharding_option.compute_kernel = EmbeddingComputeKernel.FUSED.value
        enumerator.populate_estimates(cache_tables)
        return sum(sharding_option.total_storage.hbm for sharding_option in proposal)

    @staticmethod
    def promote_high_prefetch_overheaad_table_to_hbm(
        enumerator: Optional[Enumerator], proposal: List[ShardingOption]
    ) -> None:
        """
        Prefetch overhead is related to IO. When it's larger than saved memory from
        embedding offloading, we'd undo offloading and promote to HBM for better
        memory efficiency.

        This function will end up updating proposal.
        """
        if not enumerator:
            return
        what_if_hbm_proposal = copy.deepcopy(proposal)
        what_if_hbm_cached_tables = [
            sharding_option
            for sharding_option in what_if_hbm_proposal
            if sharding_option.compute_kernel
            == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        ]
        original_cached_tables = [
            sharding_option
            for sharding_option in proposal
            if sharding_option.compute_kernel
            == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        ]

        # Modify all cached tables in what_if_proposal to be HBM only
        for sharding_option in what_if_hbm_cached_tables:
            assert sharding_option.cache_params  # appease pyre
            sharding_option.cache_params.load_factor = None
            sharding_option.compute_kernel = EmbeddingComputeKernel.FUSED.value

        # appease pyre
        assert enumerator
        enumerator.populate_estimates(what_if_hbm_cached_tables)

        # Now what_if_hbm_proposal contain estimated storage for all HBM case. If
        # it's even smaller than offloaded case, we promote it to HBM
        promoted_count = 0
        saved_hbm = 0
        for so, original_so in zip(what_if_hbm_cached_tables, original_cached_tables):
            if so.total_storage.hbm < original_so.total_storage.hbm:
                promoted_count += 1
                saved_hbm += original_so.total_storage.hbm - so.total_storage.hbm
                assert original_so.cache_params  # appease pyre
                original_so.cache_params.load_factor = None
                original_so.compute_kernel = EmbeddingComputeKernel.FUSED.value

        if promoted_count > 0:
            logger.info(
                f"EmbeddingOffloadScaleupProposer - promoted {promoted_count} tables to HBM, because their IO size is larger than the table size itself, saving {saved_hbm // 1024 // 1024}MiB HBM"
            )

        # In the end, update the storage cost for new proposal

        # appease pyre
        assert enumerator
        enumerator.populate_estimates(original_cached_tables)

    def propose(self) -> Optional[List[ShardingOption]]:
        return self.proposal

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        if not self.enumerator or plan is None:
            self.proposal = None
            return

        hbm_used_previously = sum(
            sharding_option.total_storage.hbm for sharding_option in plan
        )

        if self.search is None:
            if not partitionable or storage_constraint is None:
                self.proposal = None
                return

            hbm_available = EmbeddingOffloadScaleupProposer.get_budget(
                plan, storage_constraint
            )
            # max scale up
            peak_budget_need = (
                EmbeddingOffloadScaleupProposer.get_hbm_ceiling(
                    plan, _none_throws(self.enumerator)
                )
                - hbm_used_previously
            )
            search_budget = min(hbm_available, peak_budget_need)

            logger.info(
                f"EmbeddingOffloadScaleupProposer - unscaled plan={round(bytes_to_gb(hbm_used_previously),2)} GB, cache scale up budget={round(bytes_to_gb(hbm_available), 2)} GB, peak scale up budget need={round(bytes_to_gb(peak_budget_need),2)} GB, exploring plans of size [{round(bytes_to_gb(hbm_used_previously), 2)}, {round(bytes_to_gb(hbm_used_previously + search_budget), 2)}] GB"
            )
            self.search = LuusJaakolaSearch(
                0, search_budget, max_iterations=16, left_cost=perf_rating
            )

        best = False
        if perf_rating is not None and perf_rating < self.best_perf_rating:
            self.best_perf_rating = perf_rating
            best = True

        logger.info(
            f"EmbeddingOffloadScaleupProposer - proposed size={bytes_to_gb(hbm_used_previously):.2f} GB, score={perf_rating}{' BEST' if best else ''}"
        )

        if not partitionable:
            # Focus our search on smaller plans by assuming plans larger than this
            # proposal will also fail to partition.
            starting_size = sum(
                sharding_option.total_storage.hbm
                for sharding_option in self.starting_proposal
            )
            new_budget = hbm_used_previously - starting_size
            self.search.shrink_right(new_budget)  # pyre-ignore

        assert self.search is not None  # keep pyre happy
        budget = self.search.next(perf_rating or 1e99)
        if budget is not None:
            budget = int(budget)
        self.proposal = EmbeddingOffloadScaleupProposer.next_plan(
            self.starting_proposal, budget, self.enumerator
        )

    @staticmethod
    def get_budget(proposal: List[ShardingOption], storage_constraint: Topology) -> int:
        """returns additional HBM budget available for GPU caches."""
        available_hbm = sum(device.storage.hbm for device in storage_constraint.devices)
        used_hbm = sum(
            sharding_option.total_storage.hbm for sharding_option in proposal
        )
        return available_hbm - used_hbm

    @staticmethod
    def get_scalable_sharding_options(
        proposal: List[ShardingOption],
    ) -> List[ShardingOption]:
        """Return the subset of tables that we can scale."""

        def none_to_zero(x: Optional[float]) -> float:
            return x if x is not None else 0.0

        return [
            sharding_option
            for sharding_option in proposal
            if sharding_option.compute_kernel
            == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
            and none_to_zero(
                EmbeddingOffloadScaleupProposer.get_cacheability(sharding_option)
            )
            * none_to_zero(
                EmbeddingOffloadScaleupProposer.get_expected_lookups(sharding_option)
            )
            * none_to_zero(sharding_option.cache_load_factor)
            > 0
        ]

    # Given an available budget of additional memory, and a provisional sharding plan,
    # attempt to use the budget wisely to scale up caches that would most benefit from it.
    @staticmethod
    def next_plan(
        starting_proposal: List[ShardingOption],
        budget: Optional[int],
        enumerator: Optional[Enumerator],
    ) -> Optional[List[ShardingOption]]:
        if budget is None or enumerator is None:
            return None

        def none_to_zero(x: Optional[float]) -> float:
            return x if x is not None else 0.0

        proposal = copy.deepcopy(starting_proposal)
        # This is the subset of tables that we can scale
        cache_tables = EmbeddingOffloadScaleupProposer.get_scalable_sharding_options(
            proposal
        )
        # Nothing to scale
        if len(cache_tables) == 0:
            return None

        size_model, fused_hbm_ceiling = (
            EmbeddingOffloadScaleupProposer.build_affine_storage_model(
                cache_tables, enumerator
            )
        )
        clfs = torch.tensor(
            [sharding_option.cache_load_factor for sharding_option in cache_tables]
        )
        # cooked_cacheability is cacheability scaled by the expected number of cache
        # lookups.

        cooked_cacheability = torch.tensor(
            [
                none_to_zero(
                    EmbeddingOffloadScaleupProposer.get_cacheability(sharding_option)
                )
                * none_to_zero(
                    EmbeddingOffloadScaleupProposer.get_expected_lookups(
                        sharding_option
                    )
                )
                for sharding_option in cache_tables
            ]
        )
        new_clfs = EmbeddingOffloadScaleupProposer.allocate_budget(
            model=size_model,
            fused_hbm_ceiling=fused_hbm_ceiling,
            clfs=clfs,
            budget=budget,
            allocation_priority=cooked_cacheability,
        )

        num_promoted = 0
        # apply new_clfs, promoting tables that made it to 1.0
        for sharding_option, clf in zip(cache_tables, new_clfs):
            clf = clf.item()  # tensor scalar -> scalar
            assert sharding_option.cache_params  # appease pyre
            sharding_option.cache_params.load_factor = clf
            if clf > 0.9999:  # tolerate float roundoff
                assert sharding_option.cache_params  # appease pyre
                sharding_option.cache_params.load_factor = None
                sharding_option.compute_kernel = EmbeddingComputeKernel.FUSED.value
                num_promoted += 1
        if num_promoted > 0:
            logger.info(
                f"EmbeddingOffloadScaleupProposer - Promoted {num_promoted} tables to HBM because cache size is similar to table size."
            )
        # recalculate cost estimates of modified tables
        enumerator.populate_estimates(cache_tables)
        return proposal

    @staticmethod
    def get_cacheability(sharding_option: ShardingOption) -> Optional[float]:
        # helper to appease pyre type checker, as cache_params is Optional it maybe None
        if (
            sharding_option.cache_params is None
            or sharding_option.cache_params.stats is None
        ):
            return None
        return sharding_option.cache_params.stats.cacheability

    @staticmethod
    def get_expected_lookups(sharding_option: ShardingOption) -> Optional[float]:
        # helper to appease pyre type checker, as cache_params is Optional it maybe None
        if (
            sharding_option.cache_params is None
            or sharding_option.cache_params.stats is None
        ):
            return None
        return sharding_option.cache_params.stats.expected_lookups

    # The relationship between clf and shard memory usage is non-linear due to non-clf
    # overheads like optimization stats and input/output storage. We model it as an
    # affine relationship: bytes = clf * A + B where B is fixed overhead independent of
    # CLF (e.g. input / output IO sizes and A is per cache-row overhead.
    @staticmethod
    def build_affine_storage_model(
        uvm_caching_sharding_options: List[ShardingOption], enumerator: Enumerator
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        plan: List[ShardingOption] = copy.deepcopy(uvm_caching_sharding_options)

        def set_clf(sharding_option: ShardingOption, clf: float) -> None:
            assert sharding_option.cache_params  # appease pyre
            sharding_option.cache_params.load_factor = clf

        def set_fused(sharding_option: ShardingOption) -> None:
            assert sharding_option.cache_params  # appease pyre
            sharding_option.cache_params.load_factor = None
            sharding_option.compute_kernel = EmbeddingComputeKernel.FUSED.value

        def compute_hbm_sizes(f: Callable[[ShardingOption], None]) -> torch.Tensor:
            for sharding_option in plan:
                f(sharding_option)
            enumerator.populate_estimates(plan)
            return torch.tensor(
                [sharding_option.total_storage.hbm for sharding_option in plan]
            )

        low_clf, high_clf = 0.1, 0.9
        low_hbms = compute_hbm_sizes(lambda so: set_clf(so, low_clf))
        high_hbms = compute_hbm_sizes(lambda so: set_clf(so, high_clf))
        fused_hbms = compute_hbm_sizes(set_fused)

        A = (high_hbms - low_hbms) / (high_clf - low_clf)
        B = low_hbms - A * low_clf
        caching_model = torch.stack((A, B), dim=1)  # Nx2 (a,b)
        return caching_model, fused_hbms

    @staticmethod
    def clf_to_bytes(
        model: torch.Tensor, clfs: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        # evaluate affine model AX + B
        return (model[:, 0] * clfs + model[:, 1]).to(torch.float64)

    # Given a model of an affine system, an existing configuration (clfs), available
    # budget, and an allocation policy, return new configuration that best uses the
    # available budget. We only add additional budget, we assume the existing
    # configuration is specifying a floor or minimum size.
    @staticmethod
    def allocate_budget(
        model: torch.Tensor,
        fused_hbm_ceiling: torch.Tensor,
        clfs: torch.Tensor,
        budget: int,
        allocation_priority: torch.Tensor,
    ) -> torch.Tensor:
        # min size is size of table at 0 CLF
        min_size_bytes = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 0)
        max_size_bytes = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1)
        table_size_bytes = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, clfs)
        cache_size_bytes = table_size_bytes - min_size_bytes
        max_cache_size_bytes = max_size_bytes - min_size_bytes

        # We have budget bytes to share across the tables in. We want to increase the
        # cache_size_bytes of each table in proportion to their allocation priority
        # fraction. If we raise the cache_size_bytes to beyond max_cache_size_bytes,
        # this is equivalent to reaching CLF=1.0, so we clip the memory to 1.0, and
        # reassign the released budget in a subsequent pass.
        num_pass = 0
        while budget > 1 and num_pass < 128:
            num_pass += 1
            # mask is False for tables at >= max_size, and True otherwise. This allows
            # us to remove tables that have already reached full size in one round from
            # being dealt more budget in subsequent rounds.
            mask = (min_size_bytes + cache_size_bytes) < max_size_bytes
            if mask.sum() == 0:
                break

            logger.debug(
                f"[allocate_budget] pass={num_pass}, budget={budget}, #cache_tables={mask.sum()}"
            )

            # switch to 64bit float to avoid rounding errors, as table cache sizes can
            # easily be > 2^24.
            masked_priority = (mask * allocation_priority).to(torch.float64)
            increase_ratio = masked_priority / torch.sum(masked_priority)
            proposed_increase_bytes = budget * increase_ratio
            new_cache_size_bytes = torch.minimum(
                cache_size_bytes + proposed_increase_bytes, max_cache_size_bytes
            )
            actual_increase_bytes = new_cache_size_bytes - cache_size_bytes

            budget -= torch.sum(actual_increase_bytes).item()  # pyre-ignore[58]
            cache_size_bytes = new_cache_size_bytes
            # TODO: consider trade off of using remaining budget to push >0.95 tables
            # to HBM vs spending that budget on improving hit rate on other tables in
            # next pass.

            # Is any table over the size we'd get if we promoted to HBM? (promotion can
            # be smaller if input size is large when using prefetch). If so, mark for
            # promotion and reclaim budget to use on remaining tables.
            promotes = mask & (min_size_bytes + cache_size_bytes > fused_hbm_ceiling)
            if promotes.sum() > 0:
                budget_reclaimed = torch.sum(
                    ((min_size_bytes + cache_size_bytes) - fused_hbm_ceiling)[promotes]
                ).item()
                logger.debug(
                    f"[allocate_budget] {promotes.sum()} tables exceeded ceiling, promoting to save {budget_reclaimed} bytes"
                )
                budget += budget_reclaimed  # pyre-ignore[58]
                # force these tables to 1.0 to ensure promotion
                cache_size_bytes[promotes] = max_cache_size_bytes[promotes]

        # cache_size_bytes are the new cache sizes we want to use. We convert them back
        # to clfs by dividing by max_cache_size_bytes, which has isolated the clf
        # portion of the table size from the fixed overheads.
        # convert 64bit values back to original clf precision
        return (cache_size_bytes / max_cache_size_bytes).to(clfs.dtype)


def _sharding_option_score(
    sharding_option: ShardingOption, use_depth: bool = True
) -> float:
    return (
        max([cast(Perf, shard.perf).total for shard in sharding_option.shards])
        if use_depth
        else sum([cast(Perf, shard.perf).total for shard in sharding_option.shards])
    )


def proposers_to_proposals_list(
    proposers_list: List[Proposer], search_space: List[ShardingOption]
) -> List[List[ShardingOption]]:
    """
    only works for static_feedback proposers (the path of proposals to check is independent of the performance of the proposals)
    """

    proposals_list = []

    proposal_cache: Set[Tuple[int, ...]] = set()

    for proposer in proposers_list:
        proposer.load(search_space=search_space)

    for proposer in proposers_list:
        proposal = proposer.propose()

        while proposal:
            proposal_key = tuple(sorted(map(hash, proposal)))
            proposer.feedback(partitionable=True)
            if proposal_key in proposal_cache:
                proposal = proposer.propose()
                continue

            proposals_list.append(proposal)
            proposal_cache.add(proposal_key)
            proposal = proposer.propose()

    return proposals_list
