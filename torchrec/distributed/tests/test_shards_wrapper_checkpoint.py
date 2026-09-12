#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torchrec.distributed.embedding_kernel import get_state_dict
from torchrec.distributed.shards_wrapper import LocalShardsWrapper


def _dtensor_state_dict(
    mesh: DeviceMesh, wrapper: LocalShardsWrapper
) -> dict[str, Any]:
    embedding_table: Any = SimpleNamespace(
        name="table",
        use_virtual_table=False,
        compute_kernel="fused_triton",
        local_rows=5,
        local_cols=3,
        local_metadata=SimpleNamespace(shard_offsets=[0, 0]),
        dtensor_metadata=SimpleNamespace(
            mesh=mesh,
            placements=(Shard(0),),
            size=(5, 3),
            stride=(3, 1),
        ),
        global_metadata=None,
    )
    return get_state_dict(
        [embedding_table],
        [wrapper],
        pg=dist.group.WORLD,
    )


class LocalShardsWrapperCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            init_method=f"file://{self._directory.name}/rendezvous",
        )
        self.addCleanup(dist.destroy_process_group)

    def test_chunked_dtensor_state_dict_dcp_round_trip(self) -> None:
        expected = torch.arange(15, dtype=torch.float32).view(5, 3)
        source_wrapper = LocalShardsWrapper(
            local_shards=[expected[:2].clone(), expected[2:].clone()],
            local_offsets=[(0, 0), (2, 0)],
            logical_size=expected.size(),
        )
        mesh = DeviceMesh("cpu", [0])
        source_state = _dtensor_state_dict(mesh, source_wrapper)
        source_dtensor = cast(DTensor, source_state["table.weight"])

        self.assertIsInstance(source_dtensor, DTensor)
        save(
            source_state,
            storage_writer=FileSystemWriter(f"{self._directory.name}/checkpoint"),
        )

        destination_wrapper = LocalShardsWrapper(
            local_shards=[torch.zeros((1, 3)), torch.zeros((4, 3))],
            local_offsets=[(0, 0), (1, 0)],
            logical_size=expected.size(),
        )
        destination_state = _dtensor_state_dict(mesh, destination_wrapper)
        load(
            destination_state,
            storage_reader=FileSystemReader(f"{self._directory.name}/checkpoint"),
        )

        loaded = cast(DTensor, destination_state["table.weight"]).to_local()
        self.assertIsInstance(loaded, LocalShardsWrapper)
        loaded_wrapper = cast(LocalShardsWrapper, loaded)
        torch.testing.assert_close(
            torch.cat(loaded_wrapper.local_shards(), dim=0),
            expected,
        )
