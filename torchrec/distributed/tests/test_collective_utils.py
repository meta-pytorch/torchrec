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
from typing import Callable, List
from unittest import mock

import torch
import torch.distributed as dist
import torchrec.distributed.collective_utils as cu
from hypothesis import given, settings, strategies as st
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR  # @manual
from torchrec.distributed.collective_utils import (
    _resolve_enablement_on_leader,
    create_on_rank_and_share_result,
    init_collective_validation,
    invoke_on_rank_and_broadcast_result,
    is_leader,
    run_on_leader,
    validate_collectives_enabled,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
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
        if backend == "nccl":
            torch.cuda.set_device(rank)
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(
            ranks=[0, 1],
            backend=backend,
        )
        assert isinstance(pg, dist.ProcessGroup)
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
        if backend == "nccl":
            torch.cuda.set_device(rank)
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(
            ranks=[0, 1],
            backend=backend,
        )
        assert isinstance(pg, dist.ProcessGroup)

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
        if backend == "nccl":
            torch.cuda.set_device(rank)
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(
            ranks=[0, 1],
            backend=backend,
        )
        assert isinstance(pg, dist.ProcessGroup)

        @run_on_leader(pg, 0)
        def _test_run_on_0(rank: int) -> int:
            return rank

        res = _test_run_on_0(pg.rank())  # pyrefly: ignore
        assert res == 0

        @run_on_leader(pg, 1)
        def _test_run_on_1(rank: int) -> int:
            return rank

        res = _test_run_on_1(pg.rank())  # pyrefly: ignore
        assert res == 1
        dist.destroy_process_group()

    def test_run_on_leader_decorator(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_run_on_leader_decorator,
        )

    @classmethod
    def _test_create_on_rank_and_share_result_single_tensor(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(ranks=list(range(world_size)), backend=backend)
        assert isinstance(pg, dist.ProcessGroup)

        shape = (64, 32)
        fill_value = 42.0

        result = create_on_rank_and_share_result(
            pg,
            0,
            creator=lambda: torch.full(
                shape, fill_value=fill_value, dtype=torch.float32
            ),
            extractor=lambda t: [t],
            constructor=lambda ts: ts[0],  # pyrefly: ignore[bad-argument-type]
        )

        assert result.shape == shape, f"Expected shape {shape}, got {result.shape}"
        assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
        assert torch.all(result == fill_value).item(), "Shared tensor data mismatch"

        if rank != 0:
            assert result.is_shared(), "Non-creator rank should map shared memory"

        dist.destroy_process_group()

    def test_create_on_rank_and_share_result_single_tensor(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_create_on_rank_and_share_result_single_tensor,
        )

    @classmethod
    def _test_create_on_rank_and_share_result_multiple_tensors(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(ranks=list(range(world_size)), backend=backend)
        assert isinstance(pg, dist.ProcessGroup)

        def _creator() -> List[torch.Tensor]:
            return [
                torch.full((8, 4), fill_value=1.0, dtype=torch.float32),
                torch.full((16,), fill_value=2.0, dtype=torch.float64),
                torch.full((3, 5, 7), fill_value=3.0, dtype=torch.float16),
            ]

        result = create_on_rank_and_share_result(
            pg,
            0,
            creator=_creator,
            extractor=lambda ts: ts,  # pyrefly: ignore[bad-argument-type]
            constructor=lambda ts: ts,  # pyrefly: ignore[bad-argument-type]
        )

        assert len(result) == 3, f"Expected 3 tensors, got {len(result)}"

        assert result[0].shape == (8, 4)
        assert result[0].dtype == torch.float32
        assert torch.all(result[0] == 1.0).item()

        assert result[1].shape == (16,)
        assert result[1].dtype == torch.float64
        assert torch.all(result[1] == 2.0).item()

        assert result[2].shape == (3, 5, 7)
        assert result[2].dtype == torch.float16
        assert torch.all(result[2] == 3.0).item()

        dist.destroy_process_group()

    def test_create_on_rank_and_share_result_multiple_tensors(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_create_on_rank_and_share_result_multiple_tensors,
        )

    @classmethod
    def _test_create_on_rank_and_share_result_repeated_calls(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        """Verifies the function is safe to call repeatedly (no stale state)."""
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(ranks=list(range(world_size)), backend=backend)
        assert isinstance(pg, dist.ProcessGroup)

        for i in range(3):
            fill_value = float(i + 1)
            result = create_on_rank_and_share_result(
                pg,
                0,
                creator=lambda fv=fill_value: torch.full(
                    (32, 16), fill_value=fv, dtype=torch.float32
                ),
                extractor=lambda t: [t],
                constructor=lambda ts: ts[0],  # pyrefly: ignore[bad-argument-type]
            )
            assert torch.all(
                result == fill_value
            ).item(), f"Iteration {i}: expected {fill_value}, got mismatched data"

        dist.destroy_process_group()

    def test_create_on_rank_and_share_result_repeated_calls(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_create_on_rank_and_share_result_repeated_calls,
        )

    @classmethod
    def _test_create_on_rank_and_share_result_non_zero_rank(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.new_group(ranks=list(range(world_size)), backend=backend)
        assert isinstance(pg, dist.ProcessGroup)

        result = create_on_rank_and_share_result(
            pg,
            1,
            creator=lambda: torch.full((16, 8), fill_value=99.0, dtype=torch.float32),
            extractor=lambda t: [t],
            constructor=lambda ts: ts[0],  # pyrefly: ignore[bad-argument-type]
        )

        assert result.shape == (16, 8)
        assert torch.all(result == 99.0).item(), "Data mismatch when creator is rank 1"

        dist.destroy_process_group()

    def test_create_on_rank_and_share_result_non_zero_rank(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_create_on_rank_and_share_result_non_zero_rank,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Not enough GPUs for NCCL test",
    )
    def test_is_leader_nccl(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_is_leader,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Not enough GPUs for NCCL test",
    )
    def test_invoke_on_rank_and_broadcast_result_nccl(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_invoke_on_rank_and_broadcast_result,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Not enough GPUs for NCCL test",
    )
    def test_run_on_leader_decorator_nccl(self) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_run_on_leader_decorator,
        )


class TestEnableCollectiveValidation(MultiProcessTestBase):
    def tearDown(self) -> None:
        cu._USE_COLLECTIVE_VALIDATION = False
        os.environ.pop("TORCHREC_VALIDATE_COLLECTIVES", None)
        cu._INITIALIZED = False
        super().tearDown()

    @staticmethod
    def _test_init_collective_validation(
        rank: int, world_size: int, backend: str, jk_enabled: bool
    ) -> None:
        with MultiProcessContext(rank, world_size, backend) as ctx:
            assert ctx.pg is not None

            with mock.patch(
                "torch._utils_internal.justknobs_check", return_value=jk_enabled
            ) as mock_jk:
                init_collective_validation(ctx.pg)

            if ctx.rank == 0:
                # Only rank 0 because of invoke_on_rank_and_broadcast_result
                mock_jk.assert_called_once_with(
                    "pytorch/torchrec:enable_collective_validation", default=False
                )
            assert cu._INITIALIZED is True

            if jk_enabled:
                assert cu._USE_COLLECTIVE_VALIDATION is True
                assert validate_collectives_enabled() is True
            else:
                assert cu._USE_COLLECTIVE_VALIDATION is False
                assert validate_collectives_enabled() is False

    @given(jk_enabled=st.booleans())
    @settings(deadline=None)
    def test_init_collective_validation_jk(self, jk_enabled: bool) -> None:
        world_size = 2
        self._run_multi_process_test(
            world_size=world_size,
            backend="gloo",
            jk_enabled=jk_enabled,
            callable=TestEnableCollectiveValidation._test_init_collective_validation,
        )

    def test_set_env_var_overrides(self) -> None:
        # Test setting env variable to false overrides jk
        os.environ["TORCHREC_VALIDATE_COLLECTIVES"] = "0"
        with mock.patch("torch._utils_internal.justknobs_check") as mock_jk:
            self.assertFalse(_resolve_enablement_on_leader())
            mock_jk.assert_not_called()

    def test_set_env_var_without_jk(self) -> None:
        # Test setting env variable to true enables without jk
        os.environ["TORCHREC_VALIDATE_COLLECTIVES"] = "1"
        with mock.patch("torch._utils_internal.justknobs_check") as mock_jk:
            self.assertTrue(_resolve_enablement_on_leader())
            mock_jk.assert_not_called()

    def test_default_is_false(self) -> None:
        os.environ.pop("TORCHREC_VALIDATE_COLLECTIVES", None)
        with mock.patch(
            "torch._utils_internal.justknobs_check", return_value=False
        ) as mock_jk:
            self.assertFalse(_resolve_enablement_on_leader())
            mock_jk.assert_called_once_with(
                "pytorch/torchrec:enable_collective_validation", default=False
            )

    def test_when_jk_fails(self) -> None:
        os.environ.pop("TORCHREC_VALIDATE_COLLECTIVES", None)
        with mock.patch(
            "torch._utils_internal.justknobs_check",
            side_effect=Exception("JustKnobs failed"),
        ):
            self.assertFalse(_resolve_enablement_on_leader())
