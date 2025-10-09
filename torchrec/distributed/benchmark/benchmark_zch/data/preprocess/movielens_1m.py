#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import os

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import kjt_batch_func
from generative_recommenders.dlrm_v3.datasets.utils import (
    maybe_truncate_seq,
    separate_uih_candidates,
)

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class Batch(Pipelineable):
    uih_features: KeyedJaggedTensor
    candidates_features: KeyedJaggedTensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            uih_features=self.uih_features.to(device=device, non_blocking=non_blocking),
            candidates_features=self.candidates_features.to(
                device=device, non_blocking=non_blocking
            ),
        )

    def record_stream(self, stream: torch.Stream) -> None:
        # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
        self.uih_features.record_stream(stream)
        # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
        self.candidates_features.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            uih_features=self.uih_features.pin_memory(),
            candidates_features=self.candidates_features.pin_memory(),
        )

    def get_dict(self) -> Dict[str, Any]:
        return {
            "uih_features": self.uih_features,
            "candidates_features": self.candidates_features,
        }


def collate_fn(
    batch_list: List[Batch],
) -> Batch:
    uih_features_kjt_list = []
    candidates_features_kjt_list = []
    for batch_data in batch_list:
        uih_features_kjt_list.append(batch_data.uih_features)
        candidates_features_kjt_list.append(batch_data.candidates_features)

    return Batch(
        uih_features=kjt_batch_func(uih_features_kjt_list),
        candidates_features=kjt_batch_func(candidates_features_kjt_list),
    )


class MovieLens1MDataset(Dataset):
    def __init__(
        self,
        configs: Dict[str, Any],
    ) -> None:
        super().__init__()
        # open the ratings file
        self.rating_frame: pd.DataFrame = pd.read_csv(
            os.path.join(configs["dataset_path"], "sasrec_format.csv"), delimiter=","
        )
        self._max_num_candidates: int = configs["max_num_candidates"]
        self._max_seq_len: int = configs["max_seq_len"]
        self._contextual_feature_to_max_length: Dict[str, int] = configs[
            "contextual_feature_to_max_length"
        ]
        self._max_uih_len: int = (
            self._max_seq_len
            - self._max_num_candidates
            - len(self._contextual_feature_to_max_length)
        )
        self._uih_keys: List[str] = configs["uih_keys"]
        self._candidates_keys: List[str] = configs["candidates_keys"]
        self.items_in_memory: Dict[int, Batch] = {}  # initialize the items in memory

    def __len__(self) -> int:
        return len(self.rating_frame)

    def load_item(self, idx: int) -> Batch:
        data = self.rating_frame.iloc[idx]
        movie_history_uih, movie_history_candidates = separate_uih_candidates(
            data.sequence_item_ids,
            candidates_max_seq_len=self._max_num_candidates,
        )
        movie_history_ratings_uih, _ = separate_uih_candidates(
            data.sequence_ratings,
            candidates_max_seq_len=self._max_num_candidates,
        )

        movie_timestamps_uih, _ = separate_uih_candidates(
            data.sequence_timestamps,
            candidates_max_seq_len=self._max_num_candidates,
        )

        assert len(movie_history_uih) == len(
            movie_timestamps_uih
        ), "history len differs from timestamp len."
        assert len(movie_history_uih) == len(
            movie_history_ratings_uih
        ), "history len differs from ratings len."

        movie_history_uih = maybe_truncate_seq(movie_history_uih, self._max_uih_len)
        movie_history_ratings_uih = maybe_truncate_seq(
            movie_history_ratings_uih, self._max_uih_len
        )
        movie_timestamps_uih = maybe_truncate_seq(
            movie_timestamps_uih, self._max_uih_len
        )

        uih_kjt_values: List[Union[float, int]] = []
        uih_kjt_lengths: List[int] = []
        for name, length in self._contextual_feature_to_max_length.items():
            uih_kjt_values.append(data[name])
            uih_kjt_lengths.append(length)

        uih_seq_len = len(movie_history_uih)
        movie_dummy_weights_uih = [0.0 for _ in range(uih_seq_len)]
        movie_dummy_watch_times_uih = [0.0 for _ in range(uih_seq_len)]
        uih_kjt_values.extend(
            movie_history_uih
            + movie_history_ratings_uih
            + movie_timestamps_uih
            + movie_dummy_weights_uih
            + movie_dummy_watch_times_uih
        )
        uih_kjt_lengths.extend(
            [
                len(movie_history_uih),
                len(movie_history_ratings_uih),
                len(movie_timestamps_uih),
                len(movie_dummy_weights_uih),
                len(movie_dummy_watch_times_uih),
            ]
        )

        dummy_query_time = max(movie_timestamps_uih)
        uih_features_kjt = KeyedJaggedTensor(
            keys=self._uih_keys,
            lengths=torch.tensor(uih_kjt_lengths).long(),
            values=torch.tensor(uih_kjt_values).long(),
        )

        candidates_kjt_lengths = self._max_num_candidates * torch.ones(
            len(self._candidates_keys)
        )
        candidates_kjt_values = (
            movie_history_candidates
            + [dummy_query_time] * self._max_num_candidates  # item_query_time
            + [1] * self._max_num_candidates  # item_dummy_weights
            + [1] * self._max_num_candidates  # item_dummy_watchtime
        )
        candidates_features_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=torch.tensor(candidates_kjt_lengths).long(),
            values=torch.tensor(candidates_kjt_values).long(),
        )

        batch = Batch(
            uih_features=uih_features_kjt,
            candidates_features=candidates_features_kjt,
        )
        return batch

    def get_item_count(self) -> int:
        assert self.rating_frame is not None
        return len(self.rating_frame)

    def unload_query_samples(self, sample_list: List[int]) -> None:
        self.items_in_memory = {}

    def iloc(self, idx: int) -> pd.DataFrame:
        assert self.rating_frame is not None
        return self.rating_frame.iloc[idx]

    def load_query_samples(self, sample_list: List[int]) -> None:
        max_num_candidates = self._max_num_candidates
        self.items_in_memory = {}
        for idx in sample_list:
            data = self.iloc(idx)
            if len(data.sequence_item_ids) <= max_num_candidates:
                continue
            sample = self.load_item(idx)
            self.items_in_memory[idx] = sample

    def get_sample(self, id: int) -> Batch:
        return self.items_in_memory[id]

    def __getitems__(self, indices: List[int]) -> List[Batch]:
        self.load_query_samples(indices)
        samples = [self.get_sample(i) for i in indices]
        self.unload_query_samples(indices)
        return samples

    def __getitem__(self, idx: int) -> Batch:
        self.load_query_samples([idx])
        sample = self.get_sample(idx)
        self.unload_query_samples([idx])
        return sample


def get_movielens_1m_dataloader(
    args: argparse.Namespace, configs: Dict[str, Any], stage: str = "train"
) -> DataLoader:
    dataset = MovieLens1MDataset(configs=configs)
    total_items = dataset.get_item_count()
    train_split_percentage = configs["train_split_percentage"]
    if stage == "train":
        train_size = round(train_split_percentage * total_items)
        dataset = torch.utils.data.Subset(dataset, range(train_size))
    elif stage == "val":
        train_size = round(train_split_percentage * total_items)
        val_size = round((1 - train_split_percentage) * total_items)
        dataset = torch.utils.data.Subset(
            dataset, range(train_size, train_size + val_size)
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=(
            configs["batch_size"] if args.batch_size is None else args.batch_size
        ),
        shuffle=args.shuffle_batches,
        collate_fn=collate_fn,
        prefetch_factor=configs["prefetch_factor"],
        num_workers=configs["num_workers"],
        sampler=DistributedSampler(dataset),
    )
    return dataloader
