#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import time
from functools import reduce
from time import perf_counter
from typing import Callable, cast, Dict, List, Optional, Tuple, Union

import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import invoke_on_rank_and_broadcast_result
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.constants import BATCH_SIZE, MAX_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import (
    GreedyPerfPartitioner,
    MemoryBalancedPartitioner,
)
from torchrec.distributed.planner.perf_models import NoopPerfModel
from torchrec.distributed.planner.proposers import (
    GreedyProposer,
    GridSearchProposer,
    UniformProposer,
)
from torchrec.distributed.planner.stats import EmbeddingStats
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    hash_planner_context_inputs,
    hash_planner_context_inputs_str,
    ParameterConstraints,
    Partitioner,
    PerfModel,
    PlanDebugStats,
    PlanLoader,
    PlannerError,
    PlannerErrorType,
    Proposer,
    ShardingOption,
    Stats,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import (
    bytes_to_gb,
    reset_shard_rank,
    storage_repr_in_gb,
)
from torchrec.distributed.sharding_plan import get_default_sharders, placement
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
    ShardingPlanner,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import get_device_type, none_throws
from torchrec.utils.emo_logger import DecisionCategory, log_emo_decision

try:
    # This is a safety measure against torch package issues for when
    # Torchrec is included in the inference side model code. We should
    # remove this once we are sure all model side packages have the required
    # dependencies
    from torchrec.distributed.logger import _torchrec_method_logger
except Exception:

    def _torchrec_method_logger(*args, **kwargs):
        """A no-op decorator that accepts any arguments."""

        def decorator(func):
            return func

        return decorator


logger: logging.Logger = logging.getLogger(__name__)


def to_sharding_plan(
    sharding_options: List[ShardingOption],
    topology: Topology,
) -> ShardingPlan:

    compute_device = topology.compute_device
    local_size = topology.local_world_size

    plan = {}
    for sharding_option in sharding_options:
        shards = sharding_option.shards
        sharding_type = sharding_option.sharding_type

        module_plan = plan.get(sharding_option.path, EmbeddingModuleShardingPlan())
        module_plan[sharding_option.name] = ParameterSharding(
            sharding_spec=(
                None
                if sharding_type == ShardingType.DATA_PARALLEL.value
                else EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_sizes=shard.size,
                            shard_offsets=shard.offset,
                            placement=placement(
                                compute_device, cast(int, shard.rank), local_size
                            ),
                        )
                        for shard in shards
                    ]
                )
            ),
            sharding_type=sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=[cast(int, shard.rank) for shard in shards],
            cache_params=sharding_option.cache_params,
            enforce_hbm=sharding_option.enforce_hbm,
            stochastic_rounding=sharding_option.stochastic_rounding,
            bounds_check_mode=sharding_option.bounds_check_mode,
            output_dtype=sharding_option.output_dtype,
            key_value_params=sharding_option.key_value_params,
        )
        plan[sharding_option.path] = module_plan
    # pyrefly: ignore[bad-argument-type]
    return ShardingPlan(plan)


def validate_rank_assignment(sharding_plan: ShardingPlan, topology: Topology) -> None:
    """
    Validates that all shards in the given sharding plan have valid rank assignments.

    This function iterates through each module and parameter in the provided sharding plan,
    checking that each shard's placement has a valid rank (i.e., not None, not negative, and
    less than the topology's world size). If any shard fails these checks, a PlannerError is raised.

    Args:
        sharding_plan (ShardingPlan): The sharding plan to validate.
        topology (Topology): The topology containing world size information.

    Raises:
        PlannerError: If any shard has an invalid rank assignment or if a sharding spec is missing.
    """
    for module_name, module_plan in sharding_plan.plan.items():
        # pyrefly: ignore[missing-attribute]
        for param_name, param_plan in module_plan.items():
            if param_plan.sharding_spec is not None:
                for shard in param_plan.sharding_spec.shards:
                    if shard.placement.rank() is None or shard.placement.rank() < 0:
                        msg = f"Rank is not assigned for shard {shard}"
                        logging.error(msg)
                        raise PlannerError(
                            error_type=PlannerErrorType.INVALID_RANK_ASSIGNMENT,
                            message=msg,
                        )
                    if shard.placement.rank() >= topology.world_size:
                        msg = f"Shard {shard} has rank {shard.placement.rank()} which is greater than world size {dist.get_world_size()}."
                        logging.error(msg)
                        raise PlannerError(
                            error_type=PlannerErrorType.INVALID_RANK_ASSIGNMENT,
                            message=msg,
                        )
            else:
                msg = f"Sharding spec not found for {module_name}.{param_name}"
                logging.warning(msg)


def extract_plan(
    search_space: List[ShardingOption],
    loaded_sharding_options: Dict[int, ShardingOption],
) -> List[ShardingOption]:

    new_search_space: List[ShardingOption] = []
    seen_hash_set = set()

    for so in search_space:

        # Validate that the storage hash is unique and isn't mapped to multiple sharding options
        if so.storage_hash() in seen_hash_set:
            raise PlannerError(
                error_type=PlannerErrorType.PLAN_LOADING_FAILED,
                message=f"Found a duplicate storage hash {so.storage_hash()} for FQNs {[so.fqn for so in search_space]}\n",
            )
        else:
            seen_hash_set.add(so.storage_hash())

        loaded_so = loaded_sharding_options.get(so.storage_hash())
        if loaded_so is not None:
            new_search_space.append(
                ShardingOption(
                    name=so.name,
                    tensor=so.tensor,
                    module=so.module,
                    input_lengths=so.input_lengths,
                    batch_size=so.batch_size,
                    compute_kernel=so.compute_kernel,
                    sharding_type=so.sharding_type,
                    partition_by=so.partition_by,
                    # We only need to update the shards from the loaded plan
                    shards=loaded_so.shards,
                    cache_params=so.cache_params,
                    enforce_hbm=so.enforce_hbm,
                    stochastic_rounding=so.stochastic_rounding,
                    bounds_check_mode=so.bounds_check_mode,
                    dependency=so.dependency,
                    is_pooled=so.is_pooled,
                    feature_names=so.feature_names,
                    output_dtype=so.output_dtype,
                    key_value_params=so.key_value_params,
                )
            )

    # Validate that populated search space is the same size as the enumerated search space
    if len(loaded_sharding_options) != len(new_search_space):
        raise PlannerError(
            error_type=PlannerErrorType.PLAN_LOADING_FAILED,
            message=f"Loaded sharding options from Storage, but not all search space is covered. Merged search space len {len(new_search_space)} != loaded Sharding options len {len(loaded_sharding_options)}\n",
        )
    return new_search_space


def _merge_plans(best_plans: List[ShardingPlan]) -> ShardingPlan:
    if len(best_plans) == 1:
        return best_plans[0]
    else:
        merged_plan = ShardingPlan({})
        for plan in best_plans:
            for name, ps in plan.plan.items():
                ps = cast(EmbeddingModuleShardingPlan, ps)
                if name not in merged_plan.plan:
                    merged_plan.plan[name] = ps
                else:
                    for k, v in ps.items():
                        cur_plan = cast(
                            EmbeddingModuleShardingPlan, merged_plan.plan[name]
                        )
                        if k not in cur_plan:
                            cur_plan[k] = v
                        else:
                            raise PlannerError(
                                "table can not be sharded between two device group"
                            )

        return merged_plan


class EmbeddingPlannerBase(ShardingPlanner):
    """
    Base class for embedding sharding planners that provides common initialization
    and shared functionality.

    Args:
        topology (Optional[Topology]): the topology of the current process group.
        batch_size (Optional[int]): the batch size of the model.
        enumerator (Optional[Enumerator]): the enumerator to use
        storage_reservation (Optional[StorageReservation]): the storage reservation to use
        stats (Optional[Union[Stats, List[Stats]]]): the stats to use
        constraints (Optional[Dict[str, ParameterConstraints]]): per table constraints
            for sharding.
        debug (bool): whether to print debug information.
        callbacks (Optional[List[Callable[[List[ShardingOption]], List[ShardingOption]]]):
            callback functions to apply to plans.
        timeout_seconds (Optional[int]): timeout for planning in seconds.
        heuristical_storage_reservation_percentage (float): percentage of storage to reserve for sparse archs.
    """

    def __init__(
        self,
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
        timeout_seconds: Optional[int] = None,
        heuristical_storage_reservation_percentage: float = 0.15,
    ) -> None:
        if topology is None:
            compute_device = get_device_type()
            topology = Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device=compute_device,
            )
        self._topology: Topology = topology
        self._batch_size: int = batch_size if batch_size else BATCH_SIZE
        self._constraints = constraints
        self._enumerator: Enumerator = (
            enumerator
            if enumerator
            else EmbeddingEnumerator(
                topology=topology,
                batch_size=self._batch_size,
                constraints=constraints,
            )
        )
        self._storage_reservation: StorageReservation = (
            storage_reservation
            if storage_reservation
            else HeuristicalStorageReservation(
                percentage=heuristical_storage_reservation_percentage
            )
        )

        if stats is not None:
            self._stats: List[Stats] = [stats] if not isinstance(stats, list) else stats
        else:
            self._stats = [EmbeddingStats()]

        self._debug = debug
        self._callbacks: List[
            Callable[[List[ShardingOption]], List[ShardingOption]]
        ] = ([] if callbacks is None else callbacks)
        if timeout_seconds is not None:
            assert timeout_seconds > 0, "Timeout must be positive"
        self._timeout_seconds = timeout_seconds

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast

        Args:
            module (nn.Module): the module to shard.
            sharders (Optional[List[ModuleSharder[nn.Module]]]): the sharders to use for sharding
            pg (Optional[dist.ProcessGroup]): the process group to use for collective operations

        Returns:
            ShardingPlan: the sharding plan for the module.
        """
        if pg is None:
            assert dist.is_initialized(), (
                "The default process group is not yet initialized. "
                "Please call torch.distributed.init_process_group() first before invoking this. "
                "If you are not within a distributed environment, use the single rank version plan() instead."
            )
            pg = none_throws(dist.GroupMember.WORLD)

        if sharders is None:
            sharders = get_default_sharders()
        return invoke_on_rank_and_broadcast_result(
            # pyrefly: ignore[bad-argument-type]
            pg,
            0,
            self.plan,
            module,
            sharders,
        )

    def hash_planner_context_inputs(self) -> int:
        """
        Generates a hash for all planner inputs except for partitioner, proposer, performance model, and stats.
        These are all the inputs needed to verify whether a previously generated sharding plan is still valid in a new context.

        Returns:
            Generates a hash capturing topology, batch size, enumerator, storage reservation, stats and constraints.
        """
        return hash_planner_context_inputs(
            self._topology,
            self._batch_size,
            self._enumerator,
            self._storage_reservation,
            self._constraints,
        )

    def hash_planner_context_inputs_str(self) -> str:
        """
        Generates a hash for all planner inputs except for partitioner, proposer, performance model, and stats.
        These are all the inputs needed to verify whether a previously generated sharding plan is still valid in a new context.

        Returns:
            Generates a hash capturing topology, batch size, enumerator, storage reservation, stats and constraints.
        """
        return hash_planner_context_inputs_str(
            self._topology,
            self._batch_size,
            self._enumerator,
            self._storage_reservation,
            self._constraints,
        )


class EmbeddingShardingPlanner(EmbeddingPlannerBase):
    """
    Provides an optimized sharding plan for a given module with shardable parameters
    according to the provided sharders, topology, and constraints.

    Args:
        topology (Optional[Topology]): the topology of the current process group.
        batch_size (Optional[int]): the batch size of the model.
        enumerator (Optional[Enumerator]): the enumerator to use
        storage_reservation (Optional[StorageReservation]): the storage reservation to use
        proposer (Optional[Union[Proposer, List[Proposer]]]): the proposer(s) to use
        partitioner (Optional[Partitioner]): the partitioner to use
        performance_model (Optional[PerfModel]): the performance model to use
        stats (Optional[Union[Stats, List[Stats]]]): the stats to use
        constraints (Optional[Dict[str, ParameterConstraints]]): per table constraints
            for sharding.
        debug (bool): whether to print debug information.

    Example::

        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
        planner = EmbeddingShardingPlanner()
        plan = planner.plan(
            module=ebc,
            sharders=[EmbeddingBagCollectionSharder()],
        )

    """

    @_torchrec_method_logger()
    def __init__(
        self,
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
        timeout_seconds: Optional[int] = None,
        plan_loader: Optional[PlanLoader] = None,
    ) -> None:
        super().__init__(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            stats=stats,
            constraints=constraints,
            debug=debug,
            callbacks=callbacks,
            timeout_seconds=timeout_seconds,
        )
        self._partitioner: Partitioner = (
            partitioner if partitioner else GreedyPerfPartitioner()
        )
        if proposer:
            self._proposers: List[Proposer] = (
                [proposer] if not isinstance(proposer, list) else proposer
            )
        else:
            self._proposers = [
                GridSearchProposer(),
                GreedyProposer(),
                GreedyProposer(use_depth=False),
                UniformProposer(),
            ]
        self._perf_model: PerfModel = (
            performance_model
            if performance_model
            else NoopPerfModel(topology=self._topology)
        )

        self.plan_loader = plan_loader

        self._num_proposals: int = 0
        self._num_plans: int = 0
        self._best_plan: Optional[List[ShardingOption]] = None

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast

        Args:
            module (nn.Module): the module to shard.
            sharders (Optional[List[ModuleSharder[nn.Module]]]): the sharders to use for sharding
            pg (Optional[dist.ProcessGroup]): the process group to use for collective operations

        Returns:
            ShardingPlan: the sharding plan for the module.
        """
        if pg is None:
            assert dist.is_initialized(), (
                "The default process group is not yet initialized. "
                "Please call torch.distributed.init_process_group() first before invoking this. "
                "If you are not within a distributed environment, use the single rank version plan() instead."
            )
            pg = none_throws(dist.GroupMember.WORLD)

        if sharders is None:
            sharders = get_default_sharders()
        return invoke_on_rank_and_broadcast_result(
            # pyrefly: ignore[bad-argument-type]
            pg,
            0,
            self.plan,
            module,
            sharders,
        )

    @_torchrec_method_logger()
    # pyrefly: ignore[bad-param-name-override]
    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Provides an optimized sharding plan for a given module with shardable parameters
        according to the provided sharders, topology, and constraints.

        Args:
            module (nn.Module): the module to shard.
            sharders (List[ModuleSharder[nn.Module]]): the sharders to use for sharding.

        Returns:
            ShardingPlan: the sharding plan for the module.
        """
        self._num_proposals = 0
        self._num_plans = 0
        start_time = perf_counter()
        best_plan = None
        lowest_storage = Storage(MAX_SIZE, MAX_SIZE, MAX_SIZE)
        last_planner_error: Optional[PlannerError] = None
        last_proposal: List[ShardingOption] = []
        best_perf_rating = MAX_SIZE

        storage_constraint: Topology = self._storage_reservation.reserve(
            topology=self._topology,
            batch_size=self._batch_size,
            module=module,
            sharders=sharders,
            constraints=self._constraints,
        )
        global_storage_capacity = reduce(
            lambda x, y: x + y,
            [device.storage for device in self._topology.devices],
        )
        storage_policy = self._storage_reservation.__class__.__name__
        storage_percentage = getattr(self._storage_reservation, "_percentage", None)
        log_emo_decision(
            DecisionCategory.PROPOSER,
            logging.INFO,
            f"Storage reservation applied: policy={storage_policy}, percentage={storage_percentage}, global_hbm_available={round(bytes_to_gb(global_storage_capacity.hbm), 3)}GB",
            storage_policy=storage_policy,
            storage_percentage=storage_percentage,
            global_hbm_available_gb=round(bytes_to_gb(global_storage_capacity.hbm), 3),
        )

        search_space = self._enumerator.enumerate(
            module=module,
            sharders=sharders,
        )
        if not search_space:
            # No shardable parameters
            return ShardingPlan({})

        loaded_sharding_options = None
        loaded_best_plan: List[ShardingOption] = []

        if self.plan_loader is not None:
            # validate plan before loading
            self._loader_plan_validation(
                current_planner_hash=self.hash_planner_context_inputs_str(),
                loaded_plan_hash=self.plan_loader.plan_context_hash(),
            )
            loaded_sharding_options = self.plan_loader.load()
            if loaded_sharding_options is not None:
                # Merging sharding options from loaded plan with enumerated search space
                loaded_best_plan = extract_plan(
                    search_space=search_space,
                    loaded_sharding_options=loaded_sharding_options,
                )

        # Loaded plan is validated successfully and can be used for generate the sharding plan, skipping new plan generation.
        if loaded_best_plan:
            logger.info(
                # pyrefly: ignore[missing-attribute]
                f"Loded sharding options from Storage with plan id: {self.plan_loader.get_plan_id()} skipping new plan generation"
            )
            best_plan = copy.deepcopy(loaded_best_plan)
        else:
            proposal_cache: Dict[
                Tuple[int, ...],
                Tuple[bool, Optional[List[ShardingOption]], Optional[float]],
            ] = {}

            for proposer in self._proposers:
                proposer.load(search_space=search_space, enumerator=self._enumerator)

            start = time.time()
            for proposer in self._proposers:
                proposal = proposer.propose()

                while proposal:
                    end = time.time()
                    elapsed = end - start
                    if self._timeout_seconds:
                        if elapsed > self._timeout_seconds:
                            logger.info(
                                f"Exceeded time limit of {self._timeout_seconds}s. Took {elapsed}s"
                            )
                            break
                    proposal_key = tuple(sorted(map(hash, proposal)))
                    if proposal_key in proposal_cache:
                        partitionable, plan, perf_rating = proposal_cache[proposal_key]
                        proposer.feedback(
                            partitionable=partitionable,
                            plan=plan,
                            perf_rating=perf_rating,
                            storage_constraint=storage_constraint,
                        )
                        proposal = proposer.propose()
                        continue

                    self._num_proposals += 1
                    try:
                        # plan is just proposal where shard.rank is populated
                        plan = self._partitioner.partition(
                            proposal=proposal,
                            storage_constraint=storage_constraint,
                        )
                        self._num_plans += 1
                        perf_rating = self._perf_model.rate(plan=plan)
                        if perf_rating < best_perf_rating:
                            best_perf_rating = perf_rating
                            best_plan = copy.deepcopy(plan)
                        proposal_cache[proposal_key] = (True, plan, perf_rating)
                        proposer.feedback(
                            partitionable=True,
                            plan=plan,
                            perf_rating=perf_rating,
                            storage_constraint=storage_constraint,
                        )
                    except PlannerError as planner_error:
                        last_planner_error = planner_error
                        # shallow copy of the proposal
                        last_proposal: List[ShardingOption] = copy.copy(proposal)
                        current_storage = cast(
                            Storage,
                            reduce(
                                lambda x, y: x + y,
                                [
                                    shard.storage
                                    for option in proposal
                                    for shard in option.shards
                                ],
                            ),
                        )
                        if current_storage < lowest_storage:
                            lowest_storage = current_storage
                        proposal_cache[proposal_key] = (False, proposal, None)
                        proposer.feedback(
                            partitionable=False,
                            plan=proposal,
                            storage_constraint=storage_constraint,
                        )

                    # clear shard.rank for each sharding_option
                    reset_shard_rank(proposal)
                    proposal = proposer.propose()

        if best_plan:
            for callback in self._callbacks:
                best_plan = callback(best_plan)

            self._best_plan = best_plan
            sharding_plan = to_sharding_plan(best_plan, self._topology)

            end_time = perf_counter()
            for stats in self._stats:
                stats.log(
                    sharding_plan=sharding_plan,
                    topology=self._topology,
                    batch_size=self._batch_size,
                    storage_reservation=self._storage_reservation,
                    num_proposals=self._num_proposals,
                    num_plans=self._num_plans,
                    run_time=end_time - start_time,
                    best_plan=best_plan,
                    constraints=self._constraints,
                    enumerator=self._enumerator,
                    sharders=sharders,
                    debug=self._debug,
                    debug_stats=PlanDebugStats(
                        planner_type=self.__class__.__name__,
                        timeout_seconds=self._timeout_seconds,
                    ),
                )

            validate_rank_assignment(sharding_plan, self._topology)
            return sharding_plan
        else:
            global_storage_capacity = reduce(
                lambda x, y: x + y,
                [device.storage for device in self._topology.devices],
            )
            global_storage_constraints = reduce(
                lambda x, y: x + y,
                [device.storage for device in storage_constraint.devices],
            )
            storage_reservation_solution = (
                (
                    f"\n\t  Storage reservation percentage: {self._storage_reservation._percentage}, "
                    f"\n\t  Per rank reservation for dense storage: {storage_repr_in_gb(self._storage_reservation._dense_storage)}, "
                    f"\n\t  Per rank reservation for kjt storage: {storage_repr_in_gb(self._storage_reservation._kjt_storage)}, "
                )
                if isinstance(self._storage_reservation, HeuristicalStorageReservation)
                # pyrefly: ignore[missing-attribute]
                else f"\n\t  Storage reservation percentage: {self._storage_reservation._percentage}, "
            )
            no_plan_solution = (
                f"Planner evaluated {self._num_proposals} proposals."
                "\nPossible solutions:"
                f"\n  1) Increase the number of devices ({self._topology.world_size})"
                f"\n  2) Reduce the model size ("
                f"\n\t  Global storage: {round(bytes_to_gb(global_storage_capacity.hbm), 3)} GB, "
                f"\n\t  Per rank hardware memory: {storage_repr_in_gb(self._topology.devices[0].storage)}, "
                f"{storage_reservation_solution}"
                f"\n\t  Global storage available for model parallel: {storage_repr_in_gb(global_storage_constraints)}, "
                f"\n\t  Global storage requirement for model parallel: {storage_repr_in_gb(lowest_storage)})"
                f"\n  3) Reduce local batch size ({self._batch_size})"
                "\n  4) Remove planner constraints that might be reducing search space or available storage\n"
            )
            last_planner_error_info = f"Last planner error: \n\t{last_planner_error}\n"

            # printout stats for no plan situation
            end_time = perf_counter()
            sharding_plan = ShardingPlan(plan={})
            # force all shards to have rank= -1
            for sharding_option in last_proposal:
                for shard in sharding_option.shards:
                    shard.rank = -1

            for stats in self._stats:
                stats.log(
                    sharding_plan=sharding_plan,
                    topology=self._topology,
                    batch_size=self._batch_size,
                    storage_reservation=self._storage_reservation,
                    num_proposals=self._num_proposals,
                    num_plans=self._num_plans,
                    run_time=end_time - start_time,
                    best_plan=last_proposal,
                    constraints=self._constraints,
                    sharders=sharders,
                    enumerator=self._enumerator,
                    debug=self._debug,
                )

            if not lowest_storage.fits_in(global_storage_constraints):
                raise PlannerError(
                    error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                    message="Unable to find a plan for this model because of insufficient storage. \n"
                    + no_plan_solution
                    + last_planner_error_info,
                )
            else:
                raise PlannerError(
                    error_type=PlannerErrorType.STRICT_CONSTRAINTS,
                    message="Unable to find a plan for this model because of the strict constraints. \n"
                    + no_plan_solution
                    + last_planner_error_info,
                )

    def _loader_plan_validation(
        self, current_planner_hash: str, loaded_plan_hash: Optional[str]
    ) -> None:
        """
        Validates that the current planner context hash matches the loaded plan context hash.

        Args:
            current_planner_hash (str): Hash from current planner context
            loaded_plan_hash (Optional[str]): Hash from loaded plan context

        Raises:
            PlannerError: If hashes don't match
        """
        if loaded_plan_hash is not None and current_planner_hash != loaded_plan_hash:
            # pyrefly: ignore[missing-attribute]
            plan_id = self.plan_loader.get_plan_id() if self.plan_loader else None
            error_msg = (
                f"Planner input context mismatch detected for {plan_id} and current planner set up:"
                f"\nCurrent planner hash: {current_planner_hash}, Loaded plan hash: {loaded_plan_hash}"
            )
            raise PlannerError(
                error_type=PlannerErrorType.PLANNER_INPUT_CONTEXT_MISMATCH,
                message="Unable to load, because of planner input mismatch - cannot validate this plan is the best plan for current context.. \n"
                + error_msg,
            )


class HeteroEmbeddingShardingPlanner(ShardingPlanner):
    """
    Provides an optimized sharding plan for a given module with shardable parameters
    according to the provided sharders, topology, and constraints.
    """

    def __init__(
        self,
        topology_groups: Optional[Dict[str, Topology]] = None,
        batch_size: Optional[int] = None,
        enumerators: Optional[Dict[str, Enumerator]] = None,
        storage_reservations: Optional[Dict[str, StorageReservation]] = None,
        proposers: Optional[Dict[str, Union[Proposer, List[Proposer]]]] = None,
        partitioners: Optional[Dict[str, Partitioner]] = None,
        performance_models: Optional[Dict[str, PerfModel]] = None,
        stats: Optional[Dict[str, Union[Stats, List[Stats]]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
    ) -> None:
        default_device = get_device_type()

        if topology_groups is None:
            topology_groups = {
                default_device: Topology(
                    local_world_size=get_local_size(),
                    world_size=dist.get_world_size(),
                    compute_device=default_device,
                )
            }
        self._topology_groups: Dict[str, Topology] = topology_groups
        self._batch_size: int = batch_size if batch_size else BATCH_SIZE
        self._constraints = constraints
        self._enumerators: Dict[str, Enumerator] = (
            enumerators
            if enumerators
            else {
                group: EmbeddingEnumerator(
                    topology=self._topology_groups[group],
                    batch_size=self._batch_size,
                    constraints=constraints,
                    use_exact_enumerate_order=True,
                )
                for group in self._topology_groups.keys()
            }
        )
        self._storage_reservations: Dict[str, StorageReservation] = (
            storage_reservations
            if storage_reservations
            else {
                group: HeuristicalStorageReservation(percentage=0.15)
                for group in self._topology_groups.keys()
            }
        )

        self._partitioners: Dict[str, Partitioner] = (
            partitioners
            if partitioners
            else {
                group: MemoryBalancedPartitioner()
                for group in self._topology_groups.keys()
            }
        )

        if proposers:
            # pyrefly: ignore[bad-assignment]
            self._proposers: Dict[str, List[Proposer]] = proposers
        else:
            self._proposers = {
                group: [
                    GridSearchProposer(),
                    GreedyProposer(),
                    GreedyProposer(use_depth=False),
                    UniformProposer(),
                ]
                for group in self._topology_groups.keys()
            }

        self._perf_models: Dict[str, PerfModel] = (
            performance_models
            if performance_models
            else {
                group: NoopPerfModel(topology=self._topology_groups[group])
                for group in self._topology_groups
            }
        )

        self._stats: Dict[str, List[Stats]] = {}

        if stats is not None:
            # pyrefly: ignore[bad-assignment]
            self._stats = stats
        else:
            self._stats = {
                group: [EmbeddingStats()] for group in self._topology_groups.keys()
            }

        self._debug = debug
        self._num_proposals: int = 0
        self._num_plans: int = 0
        self._best_plan: Optional[List[ShardingOption]] = None
        self._callbacks: List[
            Callable[[List[ShardingOption]], List[ShardingOption]]
        ] = ([] if callbacks is None else callbacks)

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        if pg is None:
            assert dist.is_initialized(), (
                "The default process group is not yet initialized. "
                "Please call torch.distributed.init_process_group() first before invoking this. "
                "If you are not within a distributed environment, use the single rank version plan() instead."
            )
            pg = none_throws(dist.GroupMember.WORLD)
        assert len(self._topology_groups) == 1, "Only single topology is supported"

        if sharders is None:
            sharders = get_default_sharders()
        return invoke_on_rank_and_broadcast_result(
            # pyrefly: ignore[bad-argument-type]
            pg,
            0,
            self.plan,
            module,
            sharders,
        )

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        best_plans: List[ShardingPlan] = []
        for group, topology in self._topology_groups.items():
            self._num_proposals = 0
            self._num_plans = 0
            start_time = perf_counter()
            best_plan = None
            lowest_storage = Storage(MAX_SIZE, MAX_SIZE, MAX_SIZE)
            last_planner_error: Optional[PlannerError] = None
            last_proposal: List[ShardingOption] = []
            best_perf_rating = MAX_SIZE

            storage_constraint: Topology = self._storage_reservations[group].reserve(
                topology=topology,
                batch_size=self._batch_size,
                module=module,
                sharders=sharders,
                constraints=self._constraints,
            )

            search_space = self._enumerators[group].enumerate(
                module=module,
                sharders=sharders,
            )

            # filter by device group
            search_space = [
                s_o
                for s_o in search_space
                # pyrefly: ignore[unsupported-operation]
                if self._constraints[s_o.name].device_group == group
            ]

            if not search_space:
                # No shardable parameters
                continue

            proposal_cache: Dict[
                Tuple[int, ...],
                Tuple[bool, Optional[List[ShardingOption]], Optional[float]],
            ] = {}

            for proposer in self._proposers[group]:
                proposer.load(
                    search_space=search_space, enumerator=self._enumerators[group]
                )

            for proposer in self._proposers[group]:
                proposal = proposer.propose()

                while proposal:
                    proposal_key = tuple(sorted(map(hash, proposal)))
                    if proposal_key in proposal_cache:
                        partitionable, plan, perf_rating = proposal_cache[proposal_key]
                        proposer.feedback(
                            partitionable=partitionable,
                            plan=plan,
                            perf_rating=perf_rating,
                            storage_constraint=storage_constraint,
                        )
                        proposal = proposer.propose()
                        continue

                    self._num_proposals += 1
                    try:
                        # plan is just proposal where shard.rank is populated
                        plan = self._partitioners[group].partition(
                            proposal=proposal,
                            storage_constraint=storage_constraint,
                        )
                        self._num_plans += 1
                        perf_rating = self._perf_models[group].rate(plan=plan)
                        if perf_rating < best_perf_rating:
                            best_perf_rating = perf_rating
                            best_plan = copy.deepcopy(plan)
                        proposal_cache[proposal_key] = (True, plan, perf_rating)
                        proposer.feedback(
                            partitionable=True,
                            plan=plan,
                            perf_rating=perf_rating,
                            storage_constraint=storage_constraint,
                        )
                    except PlannerError as planner_error:
                        last_planner_error = planner_error
                        # shallow copy of the proposal
                        last_proposal: List[ShardingOption] = copy.copy(proposal)
                        current_storage = cast(
                            Storage,
                            reduce(
                                lambda x, y: x + y,
                                [
                                    shard.storage
                                    for option in proposal
                                    for shard in option.shards
                                ],
                            ),
                        )
                        if current_storage < lowest_storage:
                            lowest_storage = current_storage
                        proposal_cache[proposal_key] = (False, proposal, None)
                        proposer.feedback(
                            partitionable=False,
                            plan=proposal,
                            storage_constraint=storage_constraint,
                        )

                    # clear shard.rank for each sharding_option
                    reset_shard_rank(proposal)
                    proposal = proposer.propose()

            if best_plan:
                for callback in self._callbacks:
                    best_plan = callback(best_plan)

                self._best_plan = best_plan
                sharding_plan = to_sharding_plan(
                    best_plan, self._topology_groups[group]
                )
                validate_rank_assignment(sharding_plan, self._topology_groups[group])
                best_plans.append(sharding_plan)

                end_time = perf_counter()
                for stats in self._stats[group]:
                    stats.log(
                        sharding_plan=sharding_plan,
                        topology=self._topology_groups[group],
                        batch_size=self._batch_size,
                        storage_reservation=self._storage_reservations[group],
                        num_proposals=self._num_proposals,
                        num_plans=self._num_plans,
                        run_time=end_time - start_time,
                        best_plan=best_plan,
                        constraints=self._constraints,
                        sharders=sharders,
                        debug=self._debug,
                    )
            else:
                global_storage_capacity = reduce(
                    lambda x, y: x + y,
                    [device.storage for device in self._topology_groups[group].devices],
                )
                global_storage_constraints = reduce(
                    lambda x, y: x + y,
                    [device.storage for device in storage_constraint.devices],
                )
                storage_reservation_solution = (
                    (
                        # pyrefly: ignore[missing-attribute]
                        f"\n\t  Storage reservation percentage: {self._storage_reservations[group]._percentage}, "
                        # pyrefly: ignore[missing-attribute]
                        f"\n\t  Per rank reservation for dense storage: {storage_repr_in_gb(self._storage_reservations[group]._dense_storage)}, "
                        # pyrefly: ignore[missing-attribute]
                        f"\n\t  Per rank reservation for kjt storage: {storage_repr_in_gb(self._storage_reservations[group]._kjt_storage)}, "
                    )
                    if isinstance(
                        self._storage_reservations[group], HeuristicalStorageReservation
                    )
                    # pyrefly: ignore[missing-attribute]
                    else f"\n\t  Storage reservation percentage: {self._storage_reservations[group]._percentage}, "
                )
                no_plan_solution = (
                    f"Planner evaluated {self._num_proposals} proposals for device group {group}."
                    "\nPossible solutions:"
                    f"\n  1) Increase the number of devices ({self._topology_groups[group].world_size})"
                    f"\n  2) Reduce the model size ("
                    f"\n\t  Global storage: {round(bytes_to_gb(global_storage_capacity.hbm), 3)} GB, "
                    f"\n\t  Per rank hardware memory: {storage_repr_in_gb(self._topology_groups[group].devices[0].storage)}, "
                    f"{storage_reservation_solution}"
                    f"\n\t  Global storage available for model parallel: {storage_repr_in_gb(global_storage_constraints)}, "
                    f"\n\t  Global storage requirement for model parallel: {storage_repr_in_gb(lowest_storage)})"
                    f"\n  3) Reduce local batch size ({self._batch_size})"
                    "\n  4) Remove planner constraints that might be reducing search space or available storage\n"
                )
                last_planner_error_info = (
                    f"Last planner error: \n\t{last_planner_error}\n"
                )

                # printout stats for no plan situation
                end_time = perf_counter()
                sharding_plan = ShardingPlan(plan={})
                # force all shards to have rank= -1
                for sharding_option in last_proposal:
                    for shard in sharding_option.shards:
                        shard.rank = -1

                for stats in self._stats[group]:
                    stats.log(
                        sharding_plan=sharding_plan,
                        topology=self._topology_groups[group],
                        batch_size=self._batch_size,
                        storage_reservation=self._storage_reservations[group],
                        num_proposals=self._num_proposals,
                        num_plans=self._num_plans,
                        run_time=end_time - start_time,
                        best_plan=last_proposal,
                        constraints=self._constraints,
                        sharders=sharders,
                        debug=self._debug,
                    )

                if not lowest_storage.fits_in(global_storage_constraints):
                    raise PlannerError(
                        error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                        message=f"Unable to find a plan for this model in device_group {group}  because of insufficient storage. \n"
                        + no_plan_solution
                        + last_planner_error_info,
                    )
                else:
                    raise PlannerError(
                        error_type=PlannerErrorType.STRICT_CONSTRAINTS,
                        message=f"Unable to find a plan for this model in device_group {group} because of the strict constraints. \n"
                        + no_plan_solution
                        + last_planner_error_info,
                    )

        sharding_plan = _merge_plans(best_plans)
        return sharding_plan
