#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""The process groups a Maglev stage runs on, and the rank layout behind them.

:class:`MaglevProcessGroups` is built once per job, before any sharding
component creates groups, and handed to
:class:`~torchrec.distributed.maglev.stage.StageWrapper`.
:func:`remap_plan_to_process_group` rewrites a sharding plan planned over one
stage's group into global ranks.
"""

from dataclasses import dataclass
from typing import cast, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchrec.distributed.types import ShardingPlan


def _count_stages(stage_size: int, world_size: Optional[int] = None) -> int:
    """How many contiguous stages of ``stage_size`` ranks ``world_size`` holds.

    ``world_size`` defaults to ``dist.get_world_size()``. Raises ``ValueError``
    if ``stage_size`` is not positive or ``world_size`` is not a whole number of
    stages.
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if stage_size <= 0:
        raise ValueError(f"stage_size must be positive, got {stage_size}")
    if world_size % stage_size:
        raise ValueError(
            f"world_size {world_size} is not a whole number of stages of "
            f"{stage_size} ranks"
        )
    return world_size // stage_size


def _build_stage_process_groups(
    stage_ranks: Sequence[Sequence[int]],
) -> List[dist.ProcessGroup]:
    """Create one data-parallel process group per Maglev stage.

    Every rank must call this collective for every stage in the same order.
    The returned list is index-aligned with ``stage_ranks``.
    """
    return [
        cast(dist.ProcessGroup, dist.new_group(ranks=list(ranks)))
        for ranks in stage_ranks
    ]


def _build_handoff_process_groups(
    stage_ranks: Sequence[Sequence[int]],
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create the activation and gradient handoff communicators, one per
    direction."""
    ranks = sorted(rank for stage in stage_ranks for rank in stage)
    activation_pg = cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
    gradient_pg = cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
    return activation_pg, gradient_pg


def _build_cascade_process_groups(
    stage_ranks: Sequence[Sequence[int]],
    backend: Optional[str] = None,
) -> List[dist.ProcessGroup]:
    """Create one same-position input-distribution group across all stages.

    Position ``p`` joins ``stage_ranks[0][p]``, ``stage_ranks[1][p]``, and so on.
    The default backend carries tensors; Gloo carries CPU metadata.
    """
    stage_size = len(stage_ranks[0]) if stage_ranks else 0
    return [
        cast(
            dist.ProcessGroup,
            dist.new_group(
                ranks=[stage[position] for stage in stage_ranks],
                backend=backend,
            ),
        )
        for position in range(stage_size)
    ]


@dataclass(frozen=True)
class MaglevProcessGroups:
    """The rank-local process groups and global rank layout for a Maglev stage.

    Device-mesh integrations use :meth:`from_device_mesh`. Standalone users such
    as the benchmark use :meth:`from_scratch`, which creates every group in a
    fixed collective order. Keeping the rank layout beside the groups lets
    :class:`~torchrec.distributed.maglev.stage.StageWrapper` support
    non-contiguous layouts without independently inferring the topology.

    Raises:
        ValueError: if ``stage_ranks`` is empty, ragged or repeats a rank, or if
            one communicator is passed for both handoff directions on more than
            one stage.
    """

    stage_ranks: Tuple[Tuple[int, ...], ...]
    stage_pg: dist.ProcessGroup
    handoff_pgs: Tuple[dist.ProcessGroup, dist.ProcessGroup]
    cascade_pg: dist.ProcessGroup
    cascade_gloo_pg: dist.ProcessGroup

    def __post_init__(self) -> None:
        normalized_ranks = self._normalize_stage_ranks(self.stage_ranks)
        object.__setattr__(self, "stage_ranks", normalized_ranks)
        if self.num_stages > 1 and self.handoff_pgs[0] is self.handoff_pgs[1]:
            raise ValueError(
                "the activation and gradient handoffs need two distinct process "
                f"groups on {self.num_stages} stages: one communicator carrying "
                "both directions deadlocks the pipeline"
            )

    @classmethod
    def from_device_mesh(
        cls,
        device_mesh: DeviceMesh,
        *,
        stage_mesh_dim_name: str,
        pipeline_mesh_dim_name: str,
    ) -> "MaglevProcessGroups":
        """Build Maglev process groups from a two-dimensional device mesh.

        The mesh remains the topology owner. Its stage-dimension group is reused
        for data parallelism, and its pipeline-dimension group is reused for
        accelerator input distribution. Maglev creates only its dedicated P2P
        handoff groups and the Gloo counterpart used for input metadata.

        Every rank must call this method with the same mesh.
        """
        mesh_dim_names = device_mesh.mesh_dim_names
        if device_mesh.ndim != 2 or mesh_dim_names is None:
            raise ValueError("Maglev requires a named two-dimensional device mesh")
        if (
            stage_mesh_dim_name not in mesh_dim_names
            or pipeline_mesh_dim_name not in mesh_dim_names
            or stage_mesh_dim_name == pipeline_mesh_dim_name
        ):
            raise ValueError(
                "stage and pipeline dimensions must be distinct dimensions of "
                f"the device mesh, got {mesh_dim_names}"
            )

        stage_dim = mesh_dim_names.index(stage_mesh_dim_name)
        pipeline_dim = mesh_dim_names.index(pipeline_mesh_dim_name)
        stage_ranks = tuple(
            tuple(int(rank) for rank in ranks)
            for ranks in device_mesh.mesh.permute(pipeline_dim, stage_dim).tolist()
        )
        _, position = cls.locate_rank(stage_ranks, dist.get_rank())
        handoff_pgs = _build_handoff_process_groups(stage_ranks)
        cascade_gloo_pgs = _build_cascade_process_groups(stage_ranks, backend="gloo")
        return cls(
            stage_ranks=stage_ranks,
            stage_pg=cast(
                dist.ProcessGroup,
                device_mesh[stage_mesh_dim_name].get_group(),
            ),
            handoff_pgs=handoff_pgs,
            cascade_pg=cast(
                dist.ProcessGroup,
                device_mesh[pipeline_mesh_dim_name].get_group(),
            ),
            cascade_gloo_pg=cascade_gloo_pgs[position],
        )

    @classmethod
    def from_scratch(
        cls,
        stage_size: int,
        num_stages: Optional[int] = None,
    ) -> "MaglevProcessGroups":
        """Create every Maglev process group for a contiguous rank layout.

        Stage ``i`` is the block of ``stage_size`` ranks starting at
        ``i * stage_size``. ``num_stages`` defaults to the world size over
        ``stage_size``, which must divide it.

        Process-group creation is collective. Every rank must call this method
        with identical arguments and before any sharding component creates
        groups. Stage, handoff, tensor-cascade, and Gloo metadata-cascade groups
        are created contiguously in that order.

        Example::

            # World size 8: four stages of two ranks.
            process_groups = MaglevProcessGroups.from_scratch(2)
        """
        if num_stages is None:
            num_stages = _count_stages(stage_size)
        if stage_size <= 0:
            raise ValueError(f"stage_size must be positive, got {stage_size}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        stage_ranks = tuple(
            tuple(range(stage * stage_size, (stage + 1) * stage_size))
            for stage in range(num_stages)
        )
        stage_pgs = _build_stage_process_groups(stage_ranks)
        handoff_pgs = _build_handoff_process_groups(stage_ranks)
        cascade_pgs = _build_cascade_process_groups(stage_ranks)
        cascade_gloo_pgs = _build_cascade_process_groups(stage_ranks, backend="gloo")
        stage_index, position = cls.locate_rank(stage_ranks, dist.get_rank())
        return cls(
            stage_ranks=stage_ranks,
            stage_pg=stage_pgs[stage_index],
            handoff_pgs=handoff_pgs,
            cascade_pg=cascade_pgs[position],
            cascade_gloo_pg=cascade_gloo_pgs[position],
        )

    @classmethod
    def locate_rank(
        cls,
        stage_ranks: Sequence[Sequence[int]],
        global_rank: int,
    ) -> Tuple[int, int]:
        """Return a rank's stage index and position within that stage."""
        for stage_index, ranks in enumerate(stage_ranks):
            if global_rank in ranks:
                return stage_index, ranks.index(global_rank)
        raise ValueError(f"rank {global_rank} is absent from the Maglev rank layout")

    @classmethod
    def _normalize_stage_ranks(
        cls,
        stage_ranks: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], ...]:
        normalized = tuple(tuple(ranks) for ranks in stage_ranks)
        if not normalized or not normalized[0]:
            raise ValueError("stage_ranks must contain at least one non-empty stage")
        stage_size = len(normalized[0])
        if any(len(ranks) != stage_size for ranks in normalized):
            raise ValueError("every Maglev stage must contain the same number of ranks")
        flattened = tuple(rank for ranks in normalized for rank in ranks)
        if len(set(flattened)) != len(flattened):
            raise ValueError("each rank must occur exactly once in stage_ranks")
        return normalized

    @property
    def stage_size(self) -> int:
        return len(self.stage_ranks[0])

    @property
    def num_stages(self) -> int:
        return len(self.stage_ranks)


def remap_plan_to_process_group(
    plan: ShardingPlan, pg: dist.ProcessGroup, device: torch.device
) -> None:
    """Rewrite a plan's shard placements from group-local to global, in place.

    A planner built over a sub-process-group topology emits *group-local* ranks
    and devices -- ``rank:0/cuda:0``, ``rank:1/cuda:1`` for a 2-rank group. Shard
    placements are interpreted against the *global* rank space, though:
    ``DistributedModelParallel`` maps ``placement.rank()`` back through the
    process group, and the shard tensor lives on the process's actual device. For
    a group whose ranks are global ``{2, 3}`` on ``cuda:{2, 3}`` the group-local
    plan is wrong twice over -- rank 0 is not in the group at all, and the device
    is ``cuda:0`` rather than ``cuda:2``.

    Call this on a plan produced by a planner whose ``Topology`` was sized to
    ``pg`` rather than to the world, before handing it to
    ``DistributedModelParallel``. Group-local rank ``r`` is remapped to global
    rank ``g = get_global_rank(pg, r)``, placed on ``cuda:{g % device_count}``.
    Deterministic given ``(plan, pg)``, so every rank in the group produces the
    same result.

    A no-op for non-CUDA devices, which carry no ordinal to correct.

    Args:
        plan: the sharding plan to rewrite, modified in place.
        pg: the process group the plan was planned over.
        device: the compute device; only ``cuda`` placements are remapped.

    .. note::
        Assumes one host -- global rank ``g`` is taken to be on local device
        ``g % torch.cuda.device_count()``.
    """
    if device.type != "cuda":
        return
    device_count = torch.cuda.device_count()
    for module_plan in plan.plan.values():
        # ModuleShardingPlan is dict-like at runtime (param name -> ParameterSharding).
        # pyrefly: ignore[missing-attribute]
        for param_sharding in module_plan.values():
            spec = getattr(param_sharding, "sharding_spec", None)
            if spec is None:
                continue
            for shard in spec.shards:
                g = dist.get_global_rank(pg, shard.placement.rank())
                dev = g % device_count if device_count else 0
                shard.placement = torch.distributed._remote_device(
                    f"rank:{g}/cuda:{dev}"
                )
