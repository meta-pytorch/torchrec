#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import multiprocessing
import os
import unittest
from typing import Callable
from unittest import mock

import torch.distributed as dist
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR  # @manual
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
    is_leader,
    run_on_leader,
)
from torchrec.test_utils import get_free_port, seed_and_log


class CollectiveUtilsTest(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.WORLD_SIZE = 2

    def tearDown(self) -> None:
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _run_multi_process_test(
        self,
        world_size: int,
        backend: str,
        callable: Callable[[], None],
    ) -> None:
        processes = []
        ctx = multiprocessing.get_context("spawn")
        for rank in range(world_size):
            p = ctx.Process(
                target=callable,
                args=(
                    rank,
                    world_size,
                    backend,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    @classmethod
    def _test_is_leader(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(
            ranks=[0, 1],
            backend=backend,
        )
        if pg.rank() == 0:
            assert is_leader(pg, 0) is True
            assert is_leader(pg, 1) is False
        else:
            assert is_leader(pg, 1) is True
            assert is_leader(pg, 0) is False
        dist.destroy_process_group()

    def test_is_leader(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_is_leader,
        )

    @classmethod
    def _test_invoke_on_rank_and_broadcast_result(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(
            ranks=[0, 1],
            backend=backend,
        )

        func = mock.MagicMock()
        func.return_value = pg.rank()

        res = invoke_on_rank_and_broadcast_result(pg=pg, rank=0, func=func)
        assert res == 0, f"Expect res to be 0 (got {res})"

        if pg.rank() == 0:
            func.assert_called_once()
        else:
            func.assert_not_called()
        func.reset_mock()

        res = invoke_on_rank_and_broadcast_result(pg=pg, rank=1, func=func)
        assert res == 1, f"Expect res to be 1 (got {res})"

        if pg.rank() == 0:
            func.assert_not_called()
        else:
            func.assert_called_once()

        dist.destroy_process_group()

    def test_invoke_on_rank_and_broadcast_result(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_invoke_on_rank_and_broadcast_result,
        )

    @classmethod
    def _test_run_on_leader_decorator(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(
            ranks=[0, 1],
            backend=backend,
        )

        @run_on_leader(pg, 0)
        def _test_run_on_0(rank: int) -> int:
            return rank

        res = _test_run_on_0(pg.rank())
        assert res == 0

        @run_on_leader(pg, 1)
        def _test_run_on_1(rank: int) -> int:
            return rank

        res = _test_run_on_1(pg.rank())
        assert res == 1
        dist.destroy_process_group()

    def test_run_on_leader_decorator(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_run_on_leader_decorator,
        )
