#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest
from typing import Any, Iterator, List, Tuple
from unittest.mock import Mock, patch

from torch.utils.data import IterDataPipe
from torchrec.datasets.utils import (
    idx_split_train_val,
    ParallelReadConcat,
    rand_split_train_val,
)


class _DummyDataReader(IterDataPipe):
    def __init__(self, num_rows: int, val: str = "") -> None:
        self.num_rows = num_rows
        self.val = val

    def __iter__(self) -> Iterator[Tuple[int, str]]:
        for idx in range(self.num_rows):
            yield idx, self.val


class TestLimit(unittest.TestCase):
    def test(self) -> None:
        datapipe = _DummyDataReader(100).limit(10)
        self.assertEqual(len(list(datapipe)), 10)


class TestIdxSplitTrainVal(unittest.TestCase):
    def test_even_split(self) -> None:
        datapipe = _DummyDataReader(int(1000))
        train_datapipe, val_datapipe = idx_split_train_val(datapipe, 0.5)
        self.assertEqual(len(list(train_datapipe)), 500)
        self.assertEqual(len(list(val_datapipe)), 500)

    def test_uneven_split(self) -> None:
        datapipe = _DummyDataReader(int(100000))
        train_datapipe, val_datapipe = idx_split_train_val(datapipe, 0.6)
        self.assertEqual(len(list(train_datapipe)), 100000 * 0.6)
        self.assertEqual(len(list(val_datapipe)), 100000 * 0.4)

    def test_invalid_train_perc(self) -> None:
        datapipe = _DummyDataReader(123)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = idx_split_train_val(datapipe, 0.0)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = idx_split_train_val(datapipe, 1.0)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = idx_split_train_val(datapipe, 10.2)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = idx_split_train_val(datapipe, -50.15)


class _FakeRandom(random.Random):
    def __init__(self, num_vals: int) -> None:
        super().__init__()
        self.num_vals = num_vals
        self.vals: List[float] = [val / num_vals for val in range(num_vals)]
        self.current_idx = 0

    def random(self) -> float:
        val = self.vals[self.current_idx]
        self.current_idx += 1
        return val

    def getstate(self) -> Tuple[Any, ...]:
        return (self.vals, self.current_idx)

    def setstate(self, state: Tuple[Any, ...]) -> None:
        self.vals, self.current_idx = state


class TestRandSplitTrainVal(unittest.TestCase):
    def test_deterministic_split(self) -> None:
        num_vals = 1000
        datapipe = _DummyDataReader(num_vals)
        with patch("random.Random", new=lambda a: _FakeRandom(num_vals)):
            train_datapipe, val_datapipe = rand_split_train_val(datapipe, 0.8)
            self.assertEqual(len(list(train_datapipe)), num_vals * 0.8)
            self.assertEqual(len(list(val_datapipe)), num_vals * 0.2)
            self.assertEqual(
                len(set(train_datapipe).intersection(set(val_datapipe))), 0
            )

    def test_rand_split(self) -> None:
        datapipe = _DummyDataReader(100000)
        train_datapipe, val_datapipe = rand_split_train_val(datapipe, 0.7)
        self.assertEqual(len(set(train_datapipe).intersection(set(val_datapipe))), 0)

    def test_invalid_train_perc(self) -> None:
        datapipe = _DummyDataReader(123)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = rand_split_train_val(datapipe, 0.0)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = rand_split_train_val(datapipe, 1.0)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = rand_split_train_val(datapipe, 10.2)
        with self.assertRaisesRegex(ValueError, "train_perc"):
            train_datapipe, val_datapipe = rand_split_train_val(datapipe, -50.15)


class TestParallelReadConcat(unittest.TestCase):
    def test_worker_assignment(self) -> None:
        datapipes = [_DummyDataReader(1000, str(idx)) for idx in range(10)]
        all_res = []
        num_workers = 4
        for idx in range(num_workers):
            with patch("torchrec.datasets.utils.get_worker_info") as get_worker_info:
                get_worker_info.return_value = Mock(id=idx, num_workers=num_workers)
                all_res += list(ParallelReadConcat(*datapipes))
        expected_res = []
        for dp in datapipes:
            expected_res += list(dp)
        self.assertEqual(all_res, expected_res)

    def test_no_workers(self) -> None:
        datapipes = [_DummyDataReader(1000, str(idx)) for idx in range(10)]
        with patch("torchrec.datasets.utils.get_worker_info") as get_worker_info:
            get_worker_info.return_value = None
            dp = ParallelReadConcat(*datapipes)
            self.assertEqual(len(list(dp)), 10000)

    def test_more_workers_than_dps(self) -> None:
        datapipes = [_DummyDataReader(1000, str(idx)) for idx in range(2)]
        with patch("torchrec.datasets.utils.get_worker_info") as get_worker_info:
            get_worker_info.return_value = Mock(id=2, num_workers=10)
            with self.assertRaises(ValueError):
                next(iter(ParallelReadConcat(*datapipes)))
