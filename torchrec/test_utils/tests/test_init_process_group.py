#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from contextlib import contextmanager
from typing import Iterator

import torch.distributed as dist
from torchrec.test_utils import init_process_group_single_rank


@contextmanager
def _occupied_port() -> Iterator[int]:
    """Holds a port with a real TCPStore, so a second bind hits the conflict CI hits.

    A raw socket could bind a different address family and silently not collide.
    """
    store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
    try:
        yield store.port
    finally:
        del store


class InitProcessGroupSingleRankTest(unittest.TestCase):
    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_ignores_a_contested_master_port(self) -> None:
        with _occupied_port() as taken_port:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(taken_port)

            init_process_group_single_rank("gloo")

        self.assertTrue(dist.is_initialized())
        self.assertEqual(dist.get_world_size(), 1)
        self.assertEqual(dist.get_rank(), 0)

    def test_master_port_path_fails_on_a_contested_port(self) -> None:
        """Control: the env-based path this helper replaces still hits EADDRINUSE."""
        with _occupied_port() as taken_port:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(taken_port)

            with self.assertRaises(dist.DistNetworkError) as raised:
                dist.init_process_group("gloo", rank=0, world_size=1)

        self.assertIn("EADDRINUSE", str(raised.exception))
        self.assertFalse(dist.is_initialized())
