#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchrec.distributed.embedding import _log_sharding_plan as _log_ec_sharding_plan
from torchrec.distributed.embeddingbag import (
    _log_sharding_plan as _log_ebc_sharding_plan,
)
from torchrec.distributed.logging_utils import EventType
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingEnv,
    ShardingEnv2D,
    ShardingType,
)
from torchrec.distributed.utils import sharding_plan_fingerprint
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig

_EBC_HANDLER = "torchrec.distributed.embeddingbag.EventLoggingHandler"
_EC_HANDLER = "torchrec.distributed.embedding.EventLoggingHandler"

_EBC_FINGERPRINT = "ShardedEmbeddingBagCollection.sharding_plan_fingerprint"
_EBC_FULL_MAP = "ShardedEmbeddingBagCollection.table_names"
_EC_FINGERPRINT = "ShardedEmbeddingCollection.sharding_plan_fingerprint"
_EC_FULL_MAP = "ShardedEmbeddingCollection.table_names"

_EXPECTED_PLAN: Dict[str, str] = {
    "table_0": ShardingType.TABLE_WISE.value,
    "table_1": ShardingType.ROW_WISE.value,
}


class RecordingEventLoggingHandler:
    """Records emitted events instead of dropping them.

    The open-source EventLoggingHandler is a no-op, and the recording one lives
    under fb/ which open-source code cannot import, so there is no real handler
    for a test here to observe.

    log_event mirrors the real signature, so renaming a parameter on the real
    handler fails these tests instead of passing silently like a MagicMock.
    """

    def __init__(self) -> None:
        self.events: List[Tuple[str, Dict[str, str]]] = []

    def log_event(
        self,
        component: str,
        event_name: str,
        event_type: EventType,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        self.events.append((event_name, dict(metadata or {})))

    def event_names(self) -> List[str]:
        return [name for name, _ in self.events]

    def metadata_for(self, event_name: str) -> Dict[str, str]:
        for name, metadata in self.events:
            if name == event_name:
                return metadata
        raise AssertionError(f"{event_name} was not emitted, got {self.event_names()}")


def _ebc_configs() -> List[EmbeddingBagConfig]:
    return [
        EmbeddingBagConfig(
            name="table_0", embedding_dim=8, num_embeddings=16, feature_names=["f0"]
        ),
        EmbeddingBagConfig(
            name="table_1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
        ),
    ]


def _ec_configs() -> List[EmbeddingConfig]:
    return [
        EmbeddingConfig(
            name="table_0", embedding_dim=8, num_embeddings=16, feature_names=["f0"]
        ),
        EmbeddingConfig(
            name="table_1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
        ),
    ]


def _parameter_shardings() -> Dict[str, ParameterSharding]:
    return {
        "table_0": ParameterSharding(
            sharding_type=ShardingType.TABLE_WISE.value,
            compute_kernel="dense",
            ranks=[0],
        ),
        "table_1": ParameterSharding(
            sharding_type=ShardingType.ROW_WISE.value,
            compute_kernel="dense",
            ranks=[0],
        ),
    }


def _log_ebc_at_rank(rank: int) -> RecordingEventLoggingHandler:
    handler = RecordingEventLoggingHandler()
    with patch(_EBC_HANDLER, handler):
        _log_ebc_sharding_plan(
            ShardingEnv.from_local(world_size=4, rank=rank),
            _ebc_configs(),
            _parameter_shardings(),
        )
    return handler


def _log_ec_at_rank(rank: int) -> RecordingEventLoggingHandler:
    handler = RecordingEventLoggingHandler()
    with patch(_EC_HANDLER, handler):
        _log_ec_sharding_plan(
            ShardingEnv.from_local(world_size=4, rank=rank),
            _ec_configs(),
            _parameter_shardings(),
        )
    return handler


class ShardingPlanFingerprintTest(unittest.TestCase):
    """The fingerprint must be comparable across ranks and across processes."""

    def test_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(
            sharding_plan_fingerprint([("a", "row_wise"), ("b", "table_wise")]),
            sharding_plan_fingerprint([("b", "table_wise"), ("a", "row_wise")]),
        )

    def test_fingerprint_changes_with_sharding_type(self) -> None:
        self.assertNotEqual(
            sharding_plan_fingerprint([("a", "row_wise")]),
            sharding_plan_fingerprint([("a", "table_wise")]),
        )

    def test_fingerprint_changes_with_table_name(self) -> None:
        self.assertNotEqual(
            sharding_plan_fingerprint([("a", "row_wise")]),
            sharding_plan_fingerprint([("b", "row_wise")]),
        )

    def test_fingerprint_is_stable_across_calls(self) -> None:
        pairs = [("table_0", "table_wise"), ("table_1", "row_wise")]
        self.assertEqual(
            sharding_plan_fingerprint(pairs), sharding_plan_fingerprint(pairs)
        )


class ShardingPlanLoggingTest(unittest.TestCase):
    """Every rank emits the fingerprint. Only rank 0 emits the full map.

    These cases use a real ``ShardingEnv`` from ``ShardingEnv.from_local``, which
    needs no process group.
    """

    def test_ebc_rank_zero_emits_fingerprint_and_full_map(self) -> None:
        handler = _log_ebc_at_rank(0)
        self.assertEqual(handler.event_names(), [_EBC_FINGERPRINT, _EBC_FULL_MAP])

    def test_ebc_nonzero_rank_emits_fingerprint_only(self) -> None:
        for rank in (1, 3):
            with self.subTest(rank=rank):
                self.assertEqual(
                    _log_ebc_at_rank(rank).event_names(), [_EBC_FINGERPRINT]
                )

    def test_ebc_full_map_metadata_is_unchanged(self) -> None:
        handler = _log_ebc_at_rank(0)
        self.assertEqual(handler.metadata_for(_EBC_FULL_MAP), _EXPECTED_PLAN)

    def test_ebc_fingerprint_metadata(self) -> None:
        handler = _log_ebc_at_rank(2)
        self.assertEqual(
            handler.metadata_for(_EBC_FINGERPRINT),
            {
                "num_tables": "2",
                "sharding_type_counts": (
                    f"{ShardingType.ROW_WISE.value}:1,"
                    f"{ShardingType.TABLE_WISE.value}:1"
                ),
                "plan_fingerprint": sharding_plan_fingerprint(_EXPECTED_PLAN.items()),
            },
        )

    def test_ebc_fingerprint_matches_on_every_rank(self) -> None:
        fingerprints = {
            _log_ebc_at_rank(rank).metadata_for(_EBC_FINGERPRINT)["plan_fingerprint"]
            for rank in range(4)
        }
        self.assertEqual(len(fingerprints), 1)

    def test_ec_rank_zero_emits_fingerprint_and_full_map(self) -> None:
        handler = _log_ec_at_rank(0)
        self.assertEqual(handler.event_names(), [_EC_FINGERPRINT, _EC_FULL_MAP])

    def test_ec_nonzero_rank_emits_fingerprint_only(self) -> None:
        self.assertEqual(_log_ec_at_rank(2).event_names(), [_EC_FINGERPRINT])


def _test_2d_rank_gate(rank: int, world_size: int) -> None:
    with MultiProcessContext(rank, world_size, backend="gloo") as ctx:
        assert ctx.pg is not None
        # Mirrors DMPCollection._create_process_groups for local_size=2,
        # world_size=4. Shard groups are {0, 2} and {1, 3}, so global rank 1 is
        # rank 0 of its sharding group -- the case `env.rank == 0` gets wrong.
        mesh = DeviceMesh(
            device_type="cpu",
            mesh=[[0, 2], [1, 3]],
            mesh_dim_names=("replicate", "shard"),
        )
        env = ShardingEnv2D(
            sharding_pg=mesh.get_group(mesh_dim="shard"),
            replica_pg=mesh.get_group(mesh_dim="replicate"),
            global_pg=ctx.pg,
            device_mesh=mesh,
            node_group_size=2,
        )

        # Pin the premise, so a mesh-layout change fails loudly instead of
        # quietly making this test stop covering the 2D branch.
        if rank == 1:
            assert (
                env.rank == 0
            ), f"expected rank 1 to be rank 0 of its sharding group, got {env.rank}"
            assert (
                env.global_rank == 1
            ), f"expected global_rank 1, got {env.global_rank}"

        handler = RecordingEventLoggingHandler()
        with patch(_EBC_HANDLER, handler):
            _log_ebc_sharding_plan(env, _ebc_configs(), _parameter_shardings())

        expected = (
            [_EBC_FINGERPRINT, _EBC_FULL_MAP] if rank == 0 else [_EBC_FINGERPRINT]
        )
        assert handler.event_names() == expected, (
            f"rank {rank} (env.rank={env.rank}, global_rank={env.global_rank}): "
            f"expected {expected}, got {handler.event_names()}"
        )

        # The fingerprint is what keeps a non-zero rank observable, so every
        # rank must emit it with the same value.
        fingerprint = handler.metadata_for(_EBC_FINGERPRINT)["plan_fingerprint"]
        expected_fingerprint = sharding_plan_fingerprint(_EXPECTED_PLAN.items())
        assert (
            fingerprint == expected_fingerprint
        ), f"rank {rank}: fingerprint {fingerprint} != {expected_fingerprint}"


def _test_subgroup_rank_gate(rank: int, world_size: int) -> None:
    with MultiProcessContext(rank, world_size, backend="gloo"):
        # Tower sharding builds one env per node and pipeline-stage sharding one
        # per stage, both from a subgroup holding its own plan. Split {0,1} and
        # {2,3} to reproduce that. Group {2,3} has no global rank 0, so a
        # job-wide gate would record nothing for it.
        group_a = dist.new_group(ranks=[0, 1])
        group_b = dist.new_group(ranks=[2, 3])
        my_group = group_a if rank < 2 else group_b
        # new_group returns GroupMember.NON_GROUP_MEMBER for a rank outside the
        # group, so this also asserts we picked the group this rank belongs to.
        assert isinstance(
            my_group, dist.ProcessGroup
        ), f"rank {rank} is not a member of its own group"
        env = ShardingEnv.from_process_group(my_group)

        # Pin the premise, so this keeps covering the subgroup case.
        if rank == 2:
            assert (
                env.rank == 0
            ), f"expected global rank 2 to lead its subgroup, got env.rank={env.rank}"

        handler = RecordingEventLoggingHandler()
        with patch(_EBC_HANDLER, handler):
            _log_ebc_sharding_plan(env, _ebc_configs(), _parameter_shardings())

        expected = (
            [_EBC_FINGERPRINT, _EBC_FULL_MAP] if rank in (0, 2) else [_EBC_FINGERPRINT]
        )
        assert handler.event_names() == expected, (
            f"rank {rank} (env.rank={env.rank}): "
            f"expected {expected}, got {handler.event_names()}"
        )


class ShardingPlanLogging2DTest(MultiProcessTestBase):
    """Under 2D sharding the full-map gate keys on global_rank, not sub-group rank."""

    def test_2d_full_map_only_on_global_rank_zero(self) -> None:
        self._run_multi_process_test(
            callable=_test_2d_rank_gate,
            world_size=4,
        )


class ShardingPlanLoggingSubgroupTest(MultiProcessTestBase):
    """A plain ShardingEnv over a subgroup emits the full map once per subgroup."""

    def test_subgroup_full_map_once_per_group(self) -> None:
        self._run_multi_process_test(
            callable=_test_subgroup_rank_gate,
            world_size=4,
        )
