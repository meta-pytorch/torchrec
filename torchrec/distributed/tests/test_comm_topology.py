#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torchrec.distributed.comm as torchrec_comm
from torch._C._distributed_c10d import FakeProcessGroup
from torch.distributed.device_mesh import init_device_mesh
from torchrec.distributed.comm import (
    get_group_rank,
    get_local_rank,
    get_local_size,
    get_node_group_size,
    get_num_groups,
    intra_and_cross_node_pg,
    intra_and_cross_node_pg_2D,
    override_local_topology,
)
from torchrec.distributed.types import ShardingEnv2D


class LocalTopologyTest(unittest.TestCase):
    def test_override_applies_to_model_topology_without_changing_environment(
        self,
    ) -> None:
        physical_env = {"LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, physical_env, clear=True):
            with override_local_topology(8, 3):
                self.assertEqual(get_local_size(world_size=16), 8)
                self.assertEqual(get_local_rank(world_size=16, rank=11), 3)
                self.assertEqual(get_group_rank(world_size=16, rank=11), 1)
                self.assertEqual(get_num_groups(world_size=16), 2)
                self.assertEqual(dict(os.environ), physical_env)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    self.assertEqual(pool.submit(get_local_size, 16).result(), 8)
                    self.assertEqual(pool.submit(get_local_rank, 16, 11).result(), 3)
            self.assertEqual(get_local_size(world_size=16), 1)
            self.assertEqual(get_local_rank(world_size=16, rank=11), 0)

    def test_override_preserves_bounds_for_smaller_subgroups(self) -> None:
        with override_local_topology(8, 5):
            self.assertEqual(get_local_size(world_size=4), 4)
            self.assertEqual(get_local_rank(world_size=4, rank=13), 1)

    def test_override_falls_back_like_the_environment(self) -> None:
        with override_local_topology(8, 3):
            self.assertEqual(get_local_size(world_size=12), 12)

    def test_an_out_of_order_exit_raises_and_changes_nothing(self) -> None:
        outer = override_local_topology(8, 0)
        inner = override_local_topology(4, 0)
        outer.__enter__()
        inner.__enter__()
        with self.assertRaisesRegex(RuntimeError, "reverse order"):
            outer.__exit__(None, None, None)
        self.assertEqual(torchrec_comm._LOCAL_TOPOLOGY, (4, 0))
        inner.__exit__(None, None, None)
        self.assertEqual(torchrec_comm._LOCAL_TOPOLOGY, (8, 0))
        torchrec_comm._TOPOLOGY_SCOPES.clear()
        torchrec_comm._LOCAL_TOPOLOGY = None

    def test_exit_keeps_groups_shared_with_the_enclosing_scope(self) -> None:
        with patch(
            "torch.distributed.distributed_c10d.set_pytorch_distributed_envs_from_justknobs"
        ):
            dist.init_process_group(backend="fake", rank=0, world_size=8)
        self.addCleanup(dist.destroy_process_group)
        world = dist.GroupMember.WORLD
        with (
            patch("torchrec.distributed.comm.dist.destroy_process_group") as destroy,
            override_local_topology(8, 0),
        ):
            torchrec_comm._INTRA_PG = world
        destroy.assert_not_called()

    def test_exit_releases_every_group_even_if_one_destroy_fails(self) -> None:
        first = MagicMock(spec=dist.ProcessGroup)
        second = MagicMock(spec=dist.ProcessGroup)
        with (
            patch("torchrec.distributed.comm.dist.is_initialized", return_value=True),
            patch(
                "torchrec.distributed.comm.dist.destroy_process_group",
                side_effect=[RuntimeError("torn down"), None],
            ) as destroy,
        ):
            with override_local_topology(8, 0):
                torchrec_comm._INTRA_PG = first
                torchrec_comm._CROSS_PG = second
        self.assertEqual([c.args[0] for c in destroy.call_args_list], [first, second])

    def test_invalid_topology_is_rejected(self) -> None:
        for size, rank in ((0, 0), (-1, 0), (8, -1), (8, 8)):
            with (
                self.subTest(size=size, rank=rank),
                self.assertRaisesRegex(ValueError, "Local topology requires"),
                override_local_topology(size, rank),
            ):
                self.fail("Invalid topology was accepted")

    def test_nested_overrides_and_exceptions_restore_previous_topology(self) -> None:
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0"}):
            with override_local_topology(8, 3):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    with override_local_topology(4, 3):
                        self.assertEqual(get_local_size(world_size=16), 4)
                        raise RuntimeError("stop")
                self.assertEqual(get_local_size(world_size=16), 8)
                self.assertEqual(get_local_rank(world_size=16, rank=11), 3)
            self.assertEqual(get_local_size(world_size=16), 1)
            self.assertEqual(get_local_rank(world_size=16, rank=11), 0)

    def test_fake_subgroups_follow_virtual_host_topology_and_restore_caches(
        self,
    ) -> None:
        options = FakeProcessGroup.Options()
        options.simulate_uniform_ranks = True
        with patch(
            "torch.distributed.distributed_c10d.set_pytorch_distributed_envs_from_justknobs"
        ):
            dist.init_process_group(
                backend="fake", rank=11, world_size=16, pg_options=options
            )
        self.addCleanup(dist.destroy_process_group)

        with override_local_topology(8, 3):
            intra, cross = intra_and_cross_node_pg()
            assert intra is not None
            assert cross is not None
            self.assertEqual(dist.get_process_group_ranks(intra), list(range(8, 16)))
            self.assertEqual(dist.get_process_group_ranks(cross), [3, 11])
            for group, size in ((intra, 8), (cross, 2)):
                value = torch.tensor([2.0], device="cpu")
                dist.all_reduce(value, group=group)
                torch.testing.assert_close(
                    value, torch.tensor([2.0 * size], device="cpu")
                )
            with override_local_topology(4, 3):
                smaller_intra, larger_cross = intra_and_cross_node_pg()
                assert smaller_intra is not None
                assert larger_cross is not None
                self.assertEqual(
                    dist.get_process_group_ranks(smaller_intra), [8, 9, 10, 11]
                )
                self.assertEqual(
                    dist.get_process_group_ranks(larger_cross), [3, 7, 11, 15]
                )
            restored_intra, restored_cross = intra_and_cross_node_pg()
            self.assertIs(restored_intra, intra)
            self.assertIs(restored_cross, cross)

    def test_fake_2d_sparse_groups_inherit_virtual_local_size(self) -> None:
        options = FakeProcessGroup.Options()
        options.simulate_uniform_ranks = True
        with patch(
            "torch.distributed.distributed_c10d.set_pytorch_distributed_envs_from_justknobs"
        ):
            dist.init_process_group(
                backend="fake", rank=11, world_size=16, pg_options=options
            )
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (8, 2), mesh_dim_names=("shard", "replicate"))
        global_pg = dist.GroupMember.WORLD
        assert global_pg is not None
        env = ShardingEnv2D(
            sharding_pg=mesh.get_group("shard"),
            replica_pg=mesh.get_group("replicate"),
            global_pg=global_pg,
            device_mesh=mesh,
        )
        with patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0"}):
            with override_local_topology(4, 3):
                intra, cross = intra_and_cross_node_pg_2D(env)
                assert intra is not None
                assert cross is not None
                self.assertEqual(dist.get_process_group_ranks(intra), [9, 11, 13, 15])
                self.assertEqual(dist.get_process_group_ranks(cross), [3, 11])
                self.assertEqual(get_node_group_size(world_size=16), 4)
            self.assertEqual(get_node_group_size(world_size=16), 1)
