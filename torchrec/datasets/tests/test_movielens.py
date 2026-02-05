#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import csv
import os
import random
import tempfile
import unittest
from typing import Any, Callable, Dict, Generator, Iterable, List, Type, Union

from torchrec.datasets.movielens import movielens_20m, movielens_25m


class MovieLensTest(unittest.TestCase):
    RATINGS_FILENAME = "ratings.csv"
    MOVIES_FILENAME = "movies.csv"

    DEFAULT_RATINGS_COLUMN_NAMES: List[str] = [
        "userId",
        "movieId",
        "rating",
        "timestamp",
    ]
    DEFAULT_RATINGS_COLUMN_TYPES: List[Type[Union[float, int, str]]] = [
        int,
        int,
        float,
        int,
    ]
    DEFAULT_MOVIES_COLUMN_NAMES: List[str] = ["movieId", "title", "genres"]
    DEFAULT_MOVIES_COLUMN_TYPES: List[Type[Union[float, int, str]]] = [int, str, str]

    MOVIE_ID_RANGE = (0, 100)

    def _create_csv(self, filename: str, rows: Iterable[Any]) -> None:
        with open(filename, "w") as f:
            cf = csv.writer(f, delimiter=",")
            cf.writerows(rows)

    def _create_ratings_csv(self, filename: str, num_rows: int) -> None:
        self._create_csv(
            filename,
            [self.DEFAULT_RATINGS_COLUMN_NAMES]
            + [
                [
                    str(random.randint(0, 100)),
                    str(random.randint(*self.MOVIE_ID_RANGE)),
                    str(random.randint(0, 10) / 2),
                    str(random.randint(0, 100000)),
                ]
                for _ in range(num_rows)
            ],
        )

    def _create_movies_csv(self, filename: str, movie_ids: Iterable[str]) -> None:
        self._create_csv(
            filename,
            [self.DEFAULT_MOVIES_COLUMN_NAMES]
            + [
                [movie_id, "title", "action|adventure|comedy"] for movie_id in movie_ids
            ],
        )

    @contextlib.contextmanager
    def _create_root(self, ratings_row_count: int) -> Generator[str, None, None]:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_ratings_csv(
                os.path.join(tmpdir, self.RATINGS_FILENAME), ratings_row_count
            )
            self._create_movies_csv(
                os.path.join(tmpdir, self.MOVIES_FILENAME),
                movie_ids=[
                    str(movie_id) for movie_id in range(self.MOVIE_ID_RANGE[1] + 1)
                ],
            )
            yield tmpdir

    def _validate_sample(
        self,
        sample: Dict[str, Any],
        expected_column_names: List[str],
        expected_column_types: List[Type[Union[float, int, str]]],
    ) -> None:
        self.assertSetEqual(set(sample.keys()), set(expected_column_names))
        ordered_vals = [sample[column_name] for column_name in expected_column_names]
        for val, expected_type in zip(ordered_vals, expected_column_types):
            self.assertTrue(isinstance(val, expected_type))

    # pyre-ignore[24]
    def _test_ratings(self, dataset_fn: Callable) -> None:
        ratings_row_count = 100
        with self._create_root(ratings_row_count) as tmpdir:
            dataset = dataset_fn(tmpdir)
            for sample in dataset:
                self._validate_sample(
                    sample,
                    self.DEFAULT_RATINGS_COLUMN_NAMES,
                    self.DEFAULT_RATINGS_COLUMN_TYPES,
                )
            self.assertEqual(len(list(dataset)), ratings_row_count)

    # pyre-ignore[24]
    def _test_ratings_movies(self, dataset_fn: Callable) -> None:
        ratings_row_count = 200
        with self._create_root(ratings_row_count) as tmpdir:
            dataset = dataset_fn(tmpdir, include_movies_data=True)
            for sample in dataset:
                self._validate_sample(
                    sample,
                    self.DEFAULT_RATINGS_COLUMN_NAMES
                    + self.DEFAULT_MOVIES_COLUMN_NAMES[1:],
                    self.DEFAULT_RATINGS_COLUMN_TYPES
                    + self.DEFAULT_MOVIES_COLUMN_TYPES[1:],
                )
            self.assertEqual(len(list(dataset)), ratings_row_count)

    def test_20m_ratings(self) -> None:
        self._test_ratings(movielens_20m)

    def test_25m_ratings(self) -> None:
        self._test_ratings(movielens_25m)

    def test_20m_ratings_movies(self) -> None:
        self._test_ratings_movies(movielens_20m)

    def test_25m_ratings_movies(self) -> None:
        self._test_ratings_movies(movielens_25m)
