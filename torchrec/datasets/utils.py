#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import csv
import math
import random
from dataclasses import dataclass
from functools import partial
from io import IOBase
from typing import Any, Callable, Iterable, Iterator, List, Sequence, Tuple, TypeVar

import torch
from iopath.common.file_io import PathManager, PathManagerFactory
from torch.utils.data import functional_datapipe, get_worker_info, IterDataPipe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable

PATH_MANAGER_KEY = "torchrec"


@dataclass
class Batch(Pipelineable):
    dense_features: torch.Tensor
    sparse_features: KeyedJaggedTensor
    labels: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            dense_features=self.dense_features.to(
                device=device, non_blocking=non_blocking
            ),
            sparse_features=self.sparse_features.to(
                device=device, non_blocking=non_blocking
            ),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.Stream) -> None:
        self.dense_features.record_stream(stream)
        self.sparse_features.record_stream(stream)
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            dense_features=self.dense_features.pin_memory(),
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
        )


class _IdxFilter(IterDataPipe):
    def __init__(
        self, datapipe: IterDataPipe, filter_fn: Callable[[int], bool]
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.filter_fn = filter_fn

    def __iter__(self) -> Iterator[Any]:
        for idx, data in enumerate(self.datapipe):
            if self.filter_fn(idx):
                yield data


def _default_key_fn(idx: int) -> int:
    return idx


def train_filter(
    key_fn: Callable[[int], int],
    train_perc: float,
    decimal_places: int,
    idx: int,
) -> bool:
    return (key_fn(idx) % 10**decimal_places) < round(train_perc * 10**decimal_places)


def val_filter(
    key_fn: Callable[[int], int],
    train_perc: float,
    decimal_places: int,
    idx: int,
) -> bool:
    return not train_filter(key_fn, train_perc, decimal_places, idx)


def idx_split_train_val(
    datapipe: IterDataPipe,
    train_perc: float,
    decimal_places: int = 2,
    key_fn: Callable[[int], int] = _default_key_fn,
) -> Tuple[IterDataPipe, IterDataPipe]:
    if not 0.0 < train_perc < 1.0:
        raise ValueError("train_perc must be in range (0.0, 1.0)")
    return (
        _IdxFilter(datapipe, partial(train_filter, key_fn, train_perc, decimal_places)),
        _IdxFilter(datapipe, partial(val_filter, key_fn, train_perc, decimal_places)),
    )


class _RandFilter(IterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe,
        filter_fn: Callable[[random.Random], bool],
        rand_gen: random.Random,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.filter_fn = filter_fn
        self.rand_gen = rand_gen
        self.rand_gen_init_state: Tuple[Any, ...] = rand_gen.getstate()

    def __iter__(self) -> Iterator[Any]:
        self.rand_gen.setstate(self.rand_gen_init_state)
        for data in self.datapipe:
            if self.filter_fn(self.rand_gen):
                yield data


def _rand_train_filter_fn(
    train_perc: float,
    rand_gen: random.Random,
) -> bool:
    return rand_gen.random() < train_perc


def _rand_val_filter_fn(train_perc: float, rand_gen: random.Random) -> bool:
    return not _rand_train_filter_fn(train_perc, rand_gen)


def rand_split_train_val(
    datapipe: IterDataPipe,
    train_perc: float,
    random_seed: int = 0,
) -> Tuple[IterDataPipe, IterDataPipe]:
    """Via uniform random sampling, generates two IterDataPipe instances representing
    disjoint train and val splits of the given IterDataPipe.

    Args:
        datapipe (IterDataPipe): datapipe to split.
        train_perc (float): value in range (0.0, 1.0) specifying target proportion of
            datapipe samples to include in train split. Note that the actual proportion
            is not guaranteed to match train_perc exactly.
        random_seed (int): determines split membership for a given sample
            and train_perc. Use the same value across calls to generate consistent splits.
    Example::

        datapipe = criteo_terabyte(
            ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        )
        train_datapipe, val_datapipe = rand_split_train_val(datapipe, 0.75)
        train_batch = next(iter(train_datapipe))
        val_batch = next(iter(val_datapipe))
    """
    if not 0.0 < train_perc < 1.0:
        raise ValueError("train_perc must be in range (0.0, 1.0)")

    return _RandFilter(
        datapipe, partial(_rand_train_filter_fn, train_perc), random.Random(random_seed)
    ), _RandFilter(
        datapipe, partial(_rand_val_filter_fn, train_perc), random.Random(random_seed)
    )


T = TypeVar("T")


def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    try:
        return dest_type(val)
    except ValueError:
        return default


@functional_datapipe("limit")
class Limit(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, limit: int) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.limit = limit

    def __iter__(self) -> Iterator[Any]:
        for idx, data in enumerate(self.datapipe):
            if idx >= self.limit:
                break
            yield data


class ReadLinesFromCSV(IterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[str, "IOBase"]],
        skip_first_line: bool = False,
        **kw,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.skip_first_line = skip_first_line
        self.kw = kw

    def __iter__(self) -> Iterator[List[str]]:
        for _, data in self.datapipe:
            reader = csv.reader(data, **self.kw)
            if self.skip_first_line:
                next(reader, None)
            for line in reader:
                yield line


class LoadFiles(IterDataPipe[Tuple[str, "IOBase"]]):
    """
    Taken and adapted from torch.utils.data.datapipes.iter.LoadFilesFromDisk

    TODO:
    Merge this back or replace this with something in core Datapipes lib
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "b",
        length: int = -1,
        path_manager_key: str = PATH_MANAGER_KEY,
        **open_kw,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.mode: str = mode
        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError("Invalid mode {}".format(mode))
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken
        self.length: int = length
        self.open_kw = open_kw
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)
        self.path_manager.set_strict_kwargs_checking(False)

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self):
        if self.mode in ("b", "t"):
            self.mode = "r" + self.mode
        for pathname in self.datapipe:
            if not isinstance(pathname, str):
                raise TypeError(
                    "Expected string type for pathname, but got {}".format(
                        type(pathname)
                    )
                )
            yield (
                pathname,
                self.path_manager.open(pathname, self.mode, **self.open_kw),
            )

    def __len__(self) -> int:
        if self.length == -1:
            raise NotImplementedError
        return self.length


def _default_dp_selector(
    datapipes: Sequence[IterDataPipe],
) -> Sequence[IterDataPipe]:
    worker_info = get_worker_info()
    if worker_info is None:
        return datapipes
    else:
        if worker_info.num_workers > len(datapipes):
            raise ValueError(
                f"Number of workers {worker_info.num_workers} exceeds"
                f"number of datapipes ({len(datapipes)})!"
            )
        offsets = [0]
        for num_workers in reversed(range(1, worker_info.num_workers + 1)):
            remaining_dps = len(datapipes) - offsets[-1]
            dps_to_assign = math.ceil(remaining_dps / num_workers)
            offsets.append(offsets[-1] + dps_to_assign)
        return datapipes[offsets[worker_info.id] : offsets[worker_info.id + 1]]


class ParallelReadConcat(IterDataPipe):
    r""":class:`ParallelReadConcat`.

    Iterable DataPipe that concatenates multiple Iterable DataPipes.
    When used with a DataLoader, assigns a subset of datapipes to each DataLoader worker
    to allow for parallel reading.
    Args:
        datapipes: IterDataPipe instances to read from.
        dp_selector: function that each DataLoader worker would use to determine the subset of datapipes
        to read from.
    Example::

        datapipes = [
            criteo_terabyte(
                (f"/home/local/datasets/criteo/shard_{idx}.tsv",),
            )
            .batch(100)
            .collate()
            for idx in range(4)
        ]
        dataloader = DataLoader(
            ParallelReadConcat(*datapipes), num_workers=4, batch_size=None
        )
    """

    def __init__(
        self,
        *datapipes: IterDataPipe,
        dp_selector: Callable[
            [Sequence[IterDataPipe]], Sequence[IterDataPipe]
        ] = _default_dp_selector,
    ) -> None:
        super().__init__()
        self.datapipes: Tuple[IterDataPipe, ...] = datapipes
        self.dp_selector = dp_selector

    def __iter__(self) -> Iterator[Any]:
        selected_dps = self.dp_selector(self.datapipes)
        for dp in selected_dps:
            for data in dp:
                yield data
