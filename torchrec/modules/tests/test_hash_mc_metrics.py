#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict
from unittest.mock import MagicMock, patch

import torch
from torchrec.modules.hash_mc_evictions import HashZchEvictionConfig
from torchrec.modules.hash_mc_metrics import (
    ConsoleScalarLoggerBackend,
    ScalarLogger,
    ScalarLoggerBackend,
)


class ConcreteBackend(ScalarLoggerBackend):
    """A minimal concrete backend for testing."""

    def __init__(self) -> None:
        self.reported: list[Dict[str, object]] = []

    def report(
        self,
        log_message: str,
        name: str,
        run_type: str,
        step: int,
        rate_metrics: Dict[str, float],
    ) -> None:
        self.reported.append(
            {
                "log_message": log_message,
                "name": name,
                "run_type": run_type,
                "step": step,
                "rate_metrics": dict(rate_metrics),
            }
        )


def _make_logger(
    backend: ScalarLoggerBackend,
    frequency: int = 1,
    start_bucket: int = 0,
    zch_size: int = 10,
    num_buckets_per_rank: int = 1,
    disable_fallback: bool = False,
) -> ScalarLogger:
    return ScalarLogger(
        name="test_table",
        zch_size=zch_size,
        frequency=frequency,
        start_bucket=start_bucket,
        num_buckets_per_rank=num_buckets_per_rank,
        num_reserved_slots_per_bucket=0,
        device=torch.device("cpu"),
        disable_fallback=disable_fallback,
        backend=backend,
    )


class ConsoleScalarLoggerBackendTest(unittest.TestCase):
    def test_report_logs_message(self) -> None:
        backend = ConsoleScalarLoggerBackend()
        with patch.object(backend, "_logger") as mock_logger:
            backend.report(
                log_message="test message",
                name="table",
                run_type="train",
                step=1,
                rate_metrics={"hit_rate": 0.5},
            )
            mock_logger.info.assert_called_once_with("test message")

    def test_file_logging_adds_handler(self) -> None:
        with patch("torchrec.modules.hash_mc_metrics.logging") as mock_logging:
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            mock_handler = MagicMock()
            mock_logging.FileHandler.return_value = mock_handler

            ConsoleScalarLoggerBackend(log_file_path="/tmp/test.log")

            mock_logging.FileHandler.assert_called_once_with("/tmp/test.log", mode="w")
            mock_logger.addHandler.assert_called_once_with(mock_handler)

    def test_no_file_handler_when_empty_path(self) -> None:
        with patch("torchrec.modules.hash_mc_metrics.logging") as mock_logging:
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger

            ConsoleScalarLoggerBackend(log_file_path="")

            mock_logging.FileHandler.assert_not_called()
            mock_logger.addHandler.assert_not_called()


class ScalarLoggerTest(unittest.TestCase):

    def test_update_accumulates_counters(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, disable_fallback=False)

        # 5 values, identities show 3 were already present (hits),
        # 2 slots were empty before (inserts), 0 collisions
        values = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        # identities_0: before remap — slots 0,1,2 have IDs matching values (hits),
        # slots 3,4 are empty (-1) → 2 inserts
        identities_0 = torch.tensor([[0], [1], [2], [-1], [-1]], dtype=torch.int64)
        # identities_1: after remap — all slots filled
        identities_1 = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.int64)
        remapped_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        hit_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        sl.update(
            identities_0=identities_0,
            identities_1=identities_1,
            values=values,
            remapped_ids=remapped_ids,
            hit_indices=hit_indices,
            evicted_emb_indices=None,
            metadata=None,
        )

        self.assertEqual(sl._total_cnt, 5)
        self.assertEqual(sl._hit_cnt, 3)
        self.assertEqual(sl._insert_cnt, 2)
        self.assertEqual(sl._collision_cnt, 0)

    def test_update_accumulates_across_calls(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, disable_fallback=False)

        values = torch.tensor([0, 1], dtype=torch.int64)
        identities_0 = torch.tensor([[0], [-1]], dtype=torch.int64)
        identities_1 = torch.tensor([[0], [1]], dtype=torch.int64)
        remapped_ids = torch.tensor([0, 1], dtype=torch.int64)
        hit_indices = torch.tensor([0, 1], dtype=torch.int64)

        sl.update(
            identities_0, identities_1, values, remapped_ids, hit_indices, None, None
        )
        sl.update(
            identities_0, identities_1, values, remapped_ids, hit_indices, None, None
        )

        self.assertEqual(sl._total_cnt, 4)
        self.assertEqual(sl._hit_cnt, 2)
        self.assertEqual(sl._insert_cnt, 2)

    def test_update_counts_evictions(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, disable_fallback=False)

        values = torch.tensor([0], dtype=torch.int64)
        identities_0 = torch.tensor([[0]], dtype=torch.int64)
        identities_1 = torch.tensor([[0]], dtype=torch.int64)
        remapped_ids = torch.tensor([0], dtype=torch.int64)
        hit_indices = torch.tensor([0], dtype=torch.int64)
        evicted = torch.tensor([5, 6, 5], dtype=torch.int64)  # 5 is duplicate
        metadata = torch.zeros(10, 1, dtype=torch.float32)

        sl.update(
            identities_0,
            identities_1,
            values,
            remapped_ids,
            hit_indices,
            evicted,
            metadata,
            eviction_config=HashZchEvictionConfig(features=[], single_ttl=10),
        )

        # 2 unique evictions (5 and 6)
        self.assertEqual(sl._eviction_cnt, 2)

    def test_should_report_respects_frequency(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, frequency=3)

        # Needs data to report
        sl._total_cnt = 10

        # Steps buffer starts at 1
        # Step 1: 1 % 3 != 0 → False
        self.assertFalse(sl.should_report())

        cast(torch.Tensor, sl._scalar_logger_steps).fill_(3)
        self.assertTrue(sl.should_report())

        cast(torch.Tensor, sl._scalar_logger_steps).fill_(4)
        self.assertFalse(sl.should_report())

        cast(torch.Tensor, sl._scalar_logger_steps).fill_(6)
        self.assertTrue(sl.should_report())

    def test_should_report_only_on_rank0(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, start_bucket=1)
        sl._total_cnt = 10
        self.assertFalse(sl.should_report())

    def test_should_report_requires_data(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend)
        # _total_cnt is 0 by default
        self.assertFalse(sl.should_report())

    def test_forward_calls_backend_report(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, zch_size=10)

        # Set up counters directly for controlled rate computation
        sl._total_cnt = 100
        sl._hit_cnt = 70
        sl._insert_cnt = 20
        sl._collision_cnt = 10
        sl._eviction_cnt = 5
        sl._opt_in_cnt = 50
        sl._sum_eviction_age = 25.0

        identities = torch.zeros(10, 1, dtype=torch.int64)
        # Fill 8 of 10 slots to get table_usage_ratio
        identities[0, 0] = -1
        identities[1, 0] = -1

        sl.forward("train", identities)

        self.assertEqual(len(backend.reported), 1)
        report = backend.reported[0]
        self.assertEqual(report["name"], "test_table")
        self.assertEqual(report["run_type"], "train")
        self.assertEqual(report["step"], 1)

        metrics = cast(Dict[str, float], report["rate_metrics"])
        self.assertAlmostEqual(metrics["hit_rate"], 0.7)
        self.assertAlmostEqual(metrics["insert_rate"], 0.2)
        self.assertAlmostEqual(metrics["collision_rate"], 0.1)
        self.assertAlmostEqual(metrics["eviction_rate"], 0.05)
        self.assertAlmostEqual(metrics["opt_in_rate"], 0.5)
        self.assertAlmostEqual(metrics["avg_eviction_age"], 5.0)
        self.assertAlmostEqual(metrics["table_usage_ratio"], 0.8)

    def test_forward_resets_counters_after_report(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend)

        sl._total_cnt = 100
        sl._hit_cnt = 70
        sl._insert_cnt = 20
        sl._collision_cnt = 10
        sl._eviction_cnt = 5
        sl._opt_in_cnt = 50
        sl._sum_eviction_age = 25.0

        identities = torch.zeros(10, 1, dtype=torch.int64)
        sl.forward("train", identities)

        self.assertEqual(sl._total_cnt, 0)
        self.assertEqual(sl._hit_cnt, 0)
        self.assertEqual(sl._insert_cnt, 0)
        self.assertEqual(sl._collision_cnt, 0)
        self.assertEqual(sl._eviction_cnt, 0)
        self.assertEqual(sl._opt_in_cnt, 0)
        self.assertAlmostEqual(sl._sum_eviction_age, 0.0)

    def test_forward_does_not_report_when_not_due(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, frequency=3)

        sl._total_cnt = 100
        identities = torch.zeros(10, 1, dtype=torch.int64)

        # Step 1: 1 % 3 != 0 → should not report
        sl.forward("train", identities)

        self.assertEqual(len(backend.reported), 0)
        # Counters should NOT be reset
        self.assertEqual(sl._total_cnt, 100)

    def test_forward_increments_steps(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, frequency=100)  # high frequency to avoid reporting

        identities = torch.zeros(10, 1, dtype=torch.int64)

        self.assertEqual(cast(torch.Tensor, sl._scalar_logger_steps).item(), 1)
        sl.forward("train", identities)
        self.assertEqual(cast(torch.Tensor, sl._scalar_logger_steps).item(), 2)
        sl.forward("train", identities)
        self.assertEqual(cast(torch.Tensor, sl._scalar_logger_steps).item(), 3)

    def test_eviction_age_computation(self) -> None:
        backend = ConcreteBackend()
        sl = _make_logger(backend, zch_size=10)

        values = torch.tensor([0], dtype=torch.int64)
        identities_0 = torch.tensor([[0]], dtype=torch.int64)
        identities_1 = torch.tensor([[0]], dtype=torch.int64)
        remapped_ids = torch.tensor([0], dtype=torch.int64)
        hit_indices = torch.tensor([0], dtype=torch.int64)
        evicted = torch.tensor([0, 1], dtype=torch.int64)

        # metadata stores the insertion hour for each slot
        metadata = torch.zeros(10, 1, dtype=torch.float32)
        metadata[0, 0] = 100.0  # slot 0 was inserted at hour 100
        metadata[1, 0] = 200.0  # slot 1 was inserted at hour 200

        eviction_config = HashZchEvictionConfig(features=[], single_ttl=10)

        with patch("torchrec.modules.hash_mc_metrics.time") as mock_time:
            mock_time.time.return_value = 360000  # hour 100
            sl.update(
                identities_0,
                identities_1,
                values,
                remapped_ids,
                hit_indices,
                evicted,
                metadata,
                eviction_config=eviction_config,
            )

        self.assertEqual(sl._eviction_cnt, 2)
        # sum_eviction_age = (cur_hour + ttl - metadata[0]) + (cur_hour + ttl - metadata[1])
        # cur_hour = 360000 / 3600 % (2^31 - 1) = 100
        # = (100 + 10 - 100) + (100 + 10 - 200) = 10 + (-90) = -80
        # The computation is sum, tested via the raw value
        self.assertAlmostEqual(sl._sum_eviction_age, -80.0, places=0)
