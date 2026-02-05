#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from typing import Any, Callable, Dict, List, Optional, Union

from torch.utils.data import IterDataPipe
from torchrec.datasets.utils import LoadFiles, ReadLinesFromCSV, safe_cast

RATINGS_FILENAME = "ratings.csv"
MOVIES_FILENAME = "movies.csv"

DEFAULT_RATINGS_COLUMN_NAMES: List[str] = ["userId", "movieId", "rating", "timestamp"]
DEFAULT_MOVIES_COLUMN_NAMES: List[str] = ["movieId", "title", "genres"]
DEFAULT_COLUMN_NAMES: List[str] = (
    DEFAULT_RATINGS_COLUMN_NAMES + DEFAULT_MOVIES_COLUMN_NAMES[1:]
)

COLUMN_TYPE_CASTERS: List[
    Callable[[Union[float, int, str]], Union[float, int, str]]
] = [
    lambda val: safe_cast(val, int, 0),
    lambda val: safe_cast(val, int, 0),
    lambda val: safe_cast(val, float, 0.0),
    lambda val: safe_cast(val, int, 0),
    lambda val: safe_cast(val, str, ""),
    lambda val: safe_cast(val, str, ""),
]


def _default_row_mapper(example: List[str]) -> Dict[str, Union[float, int, str]]:
    return {
        DEFAULT_COLUMN_NAMES[idx]: COLUMN_TYPE_CASTERS[idx](val)
        for idx, val in enumerate(example)
    }


def _join_with_movies(datapipe: IterDataPipe, root: str) -> IterDataPipe:
    movies_path = os.path.join(root, MOVIES_FILENAME)
    movies_datapipe = LoadFiles((movies_path,), mode="r")
    movies_datapipe = ReadLinesFromCSV(
        movies_datapipe,
        skip_first_line=True,
        delimiter=",",
    )
    movie_id_to_movie: Dict[str, List[str]] = {
        row[0]: row[1:] for row in movies_datapipe
    }

    def join_rating_movie(val: List[str]) -> List[str]:
        return val + movie_id_to_movie[val[1]]

    return datapipe.map(join_rating_movie)


def _movielens(
    root: str,
    *,
    include_movies_data: bool = False,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    ratings_path = os.path.join(root, RATINGS_FILENAME)
    datapipe = LoadFiles((ratings_path,), mode="r", **open_kw)
    datapipe = ReadLinesFromCSV(datapipe, skip_first_line=True, delimiter=",")

    if include_movies_data:
        datapipe = _join_with_movies(datapipe, root)
    if row_mapper:
        datapipe = datapipe.map(row_mapper)

    return datapipe


def movielens_20m(
    root: str,
    *,
    include_movies_data: bool = False,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    """`MovieLens 20M <https://grouplens.org/datasets/movielens/20m/>`_ Dataset
    Args:
        root (str): local path to root directory containing MovieLens 20M dataset files.
        include_movies_data (bool): if True, adds movies data to each line.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each split line.
        open_kw: options to pass to underlying invocation of iopath.common.file_io.PathManager.open.

    Example::

        datapipe = movielens_20m("/home/datasets/ml-20")
        datapipe = dp.iter.Batch(datapipe, 100)
        datapipe = dp.iter.Collate(datapipe)
        batch = next(iter(datapipe))
    """
    return _movielens(
        root,
        include_movies_data=include_movies_data,
        row_mapper=row_mapper,
        **open_kw,
    )


def movielens_25m(
    root: str,
    *,
    include_movies_data: bool = False,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    """`MovieLens 25M <https://grouplens.org/datasets/movielens/25m/>`_ Dataset
    Args:
        root (str): local path to root directory containing MovieLens 25M dataset files.
        include_movies_data (bool): if True, adds movies data to each line.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each split line.
        open_kw: options to pass to underlying invocation of iopath.common.file_io.PathManager.open.

    Example::

        datapipe = movielens_25m("/home/datasets/ml-25")
        datapipe = dp.iter.Batch(datapipe, 100)
        datapipe = dp.iter.Collate(datapipe)
        batch = next(iter(datapipe))
    """
    return _movielens(
        root,
        include_movies_data=include_movies_data,
        row_mapper=row_mapper,
        **open_kw,
    )
