#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
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
        # pyrefly: ignore[bad-argument-type]
        self.uih_features.record_stream(stream)
        # pyrefly: ignore[bad-argument-type]
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


def process_and_hash_x(
    x: Union[str, List[int], int], hash_size: int
) -> Union[str, List[int], int]:
    if isinstance(x, str):
        x = json.loads(x)
    if isinstance(x, list):
        return [x_i % hash_size for x_i in x]
    else:
        return x % hash_size


class KuaiRand27KDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        configs: Dict[str, Any],
    ) -> None:
        super().__init__()
        # open the seq_logs_frame file
        self.seq_logs_frame: pd.DataFrame = pd.read_csv(
            os.path.join(configs["dataset_path"], "processed_seqs_hashed.csv"),
            delimiter=",",
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

        for key in self.seq_logs_frame.columns:
            self.seq_logs_frame[key] = self.seq_logs_frame[key].apply(
                partial(
                    process_and_hash_x,
                    hash_size=(
                        args.input_hash_size
                        # if len(args.zch_method) > 0
                        # else args.num_embeddings
                    ),
                )
            )

        # define the items memory
        self.items_in_memory: Dict[int, Batch] = {}

    def __len__(self) -> int:
        return len(self.seq_logs_frame)

    def load_item(self, idx: int) -> Batch:
        data = self.seq_logs_frame.iloc[idx]
        video_history_uih, video_history_candidates = separate_uih_candidates(
            data.video_id,
            candidates_max_seq_len=self._max_num_candidates,
        )
        action_weights_uih, action_weights_candidates = separate_uih_candidates(
            data.action_weights,
            candidates_max_seq_len=self._max_num_candidates,
        )
        timestamps_uih, _ = separate_uih_candidates(
            data.time_ms,
            candidates_max_seq_len=self._max_num_candidates,
        )
        watch_time_uih, watch_time_candidates = separate_uih_candidates(
            data.play_time_ms,
            candidates_max_seq_len=self._max_num_candidates,
        )

        video_history_uih = maybe_truncate_seq(video_history_uih, self._max_uih_len)
        action_weights_uih = maybe_truncate_seq(action_weights_uih, self._max_uih_len)
        timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)
        watch_time_uih = maybe_truncate_seq(watch_time_uih, self._max_uih_len)

        uih_seq_len = len(video_history_uih)
        assert uih_seq_len == len(
            timestamps_uih
        ), "history len differs from timestamp len."
        assert uih_seq_len == len(
            action_weights_uih
        ), "history len differs from weights len."
        assert uih_seq_len == len(
            watch_time_uih
        ), "history len differs from watch time len."

        uih_kjt_values: List[Union[torch.Tensor, int]] = []
        uih_kjt_lengths: List[Union[torch.Tensor, int]] = []
        for name, length in self._contextual_feature_to_max_length.items():
            uih_kjt_values.append(data[name])
            uih_kjt_lengths.append(length)

        uih_kjt_values.extend(
            video_history_uih + timestamps_uih + action_weights_uih + watch_time_uih
        )

        uih_kjt_lengths.extend(
            [
                uih_seq_len
                for _ in range(
                    len(self._uih_keys) - len(self._contextual_feature_to_max_length)
                )
            ]
        )

        dummy_query_time = max(timestamps_uih)
        uih_features_kjt = KeyedJaggedTensor(
            keys=self._uih_keys,
            lengths=torch.tensor(uih_kjt_lengths).long(),
            values=torch.tensor(uih_kjt_values).long(),
        )

        candidates_kjt_lengths = self._max_num_candidates * torch.ones(
            len(self._candidates_keys)
        )
        candidates_kjt_values = (
            video_history_candidates
            + action_weights_candidates
            + watch_time_candidates
            + [dummy_query_time] * self._max_num_candidates
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
        assert self.seq_logs_frame is not None
        return len(self.seq_logs_frame)

    def unload_query_samples(self, sample_list: List[int]) -> None:
        self.items_in_memory = {}

    def iloc(self, idx: int) -> pd.DataFrame:
        assert self.seq_logs_frame is not None
        return self.seq_logs_frame.iloc[idx]

    def load_query_samples(self, sample_list: List[int]) -> None:
        max_num_candidates = self._max_num_candidates
        self.items_in_memory = {}
        for idx in sample_list:
            data = self.iloc(idx)
            if len(data.video_id) <= max_num_candidates:
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

    # pyrefly: ignore[bad-param-name-override]
    def __getitem__(self, idx: int) -> Batch:
        self.load_query_samples([idx])
        sample = self.get_sample(idx)
        self.unload_query_samples([idx])
        return sample


def get_kuairand_27k_dataloader(
    args: argparse.Namespace, configs: Dict[str, Any], stage: str = "train"
) -> DataLoader:
    dataset = KuaiRand27KDataset(args=args, configs=configs)
    total_items = dataset.get_item_count()
    train_split_percentage = configs["train_split_percentage"]
    train_size = round(train_split_percentage * total_items)
    if stage == "train":
        dataset = torch.utils.data.Subset(dataset, range(train_size))
    elif stage == "val":
        dataset = torch.utils.data.Subset(dataset, range(train_size, total_items))

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
