#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data as data_utils
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
)
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class ParametricDataset(Dataset):
    def __init__(
        self,
        binary_file_path: str,
        batch_size: int,
        drop_last_batch,
        **kwargs,
    ):
        self._batch_size = batch_size
        bytes_per_feature = {}
        for name in DEFAULT_INT_NAMES:
            bytes_per_feature[name] = np.dtype(np.float32).itemsize
        for name in DEFAULT_CAT_NAMES:
            bytes_per_feature[name] = np.dtype(np.int32).itemsize

        self._numerical_features_file = None
        self._label_file = None
        self._categorical_features_files = []

        self._numerical_bytes_per_batch = (
            bytes_per_feature[DEFAULT_INT_NAMES[0]]
            * len(DEFAULT_INT_NAMES)
            * batch_size
        )
        self._label_bytes_per_batch = np.dtype(np.float32).itemsize * batch_size
        self._categorical_bytes_per_batch = [
            bytes_per_feature[feature] * self._batch_size
            for feature in DEFAULT_CAT_NAMES
        ]
        # Load categorical
        for feature_name in DEFAULT_CAT_NAMES:
            path_to_open = os.path.join(binary_file_path, f"{feature_name}.bin")
            cat_file = os.open(path_to_open, os.O_RDONLY)
            bytes_per_batch = bytes_per_feature[feature_name] * self._batch_size
            batch_num_float = os.fstat(cat_file).st_size / bytes_per_batch
            self._categorical_features_files.append(cat_file)

        # Load numerical
        path_to_open = os.path.join(binary_file_path, "numerical.bin")
        self._numerical_features_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._numerical_features_file).st_size
            / self._numerical_bytes_per_batch
        )

        # Load label
        path_to_open = os.path.join(binary_file_path, "label.bin")
        self._label_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._label_file).st_size / self._label_bytes_per_batch
        )
        # number of batches means the ALL data for all ranks
        number_of_batches = (
            math.ceil(batch_num_float)
            if not drop_last_batch
            else math.floor(batch_num_float)
        )
        # for this data_loader, we should divide the num_batch by world_size
        self._num_entries = number_of_batches

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        """Numerical features are returned in the order they appear in the channel spec section
        For performance reasons, this is required to be the order they are saved in, as specified
        by the relevant chunk in source spec.
        Categorical features are returned in the order they appear in the channel spec section
        """

        if idx >= self._num_entries:
            raise IndexError()

        return self._get_item(idx)

    def _get_item(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features = self._get_categorical_features(idx)

        return numerical_features, categorical_features, click

    def _get_label(self, idx: int) -> torch.Tensor:
        raw_label_data = os.pread(
            self._label_file,
            self._label_bytes_per_batch,
            idx * self._label_bytes_per_batch,
        )
        array = np.frombuffer(raw_label_data, dtype=np.float32)
        return torch.from_numpy(array).to(torch.float32)

    def _get_numerical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._numerical_features_file is None:
            return None

        raw_numerical_data = os.pread(
            self._numerical_features_file,
            self._numerical_bytes_per_batch,
            idx * self._numerical_bytes_per_batch,
        )
        array = np.frombuffer(raw_numerical_data, dtype=np.float32)
        return (
            torch.from_numpy(array).to(torch.float32).view(-1, len(DEFAULT_INT_NAMES))
        )

    def _get_categorical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._categorical_features_files is None:
            return None
        categorical_features = []
        for cat_bytes, cat_file in zip(
            self._categorical_bytes_per_batch,
            self._categorical_features_files,
        ):
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            array = np.frombuffer(raw_cat_data, dtype=np.int32)
            tensor = torch.from_numpy(array).to(torch.long).view(-1)
            categorical_features.append(tensor)
        return torch.cat(categorical_features)


class NvtBinaryDataloader:
    def __init__(
        self,
        binary_file_path: str,
        batch_size: int = 2048,
        drop_last_batch: bool = True,  # the last batch may not contain enough data which breaks the size of KJT
    ) -> None:
        self.dataset = ParametricDataset(
            binary_file_path,
            batch_size,
            drop_last_batch,
        )
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def collate_fn(self, attr_dict):
        dense_features, sparse_features, labels = attr_dict
        return Batch(
            dense_features=dense_features,
            sparse_features=KeyedJaggedTensor(
                keys=DEFAULT_CAT_NAMES,
                values=sparse_features,
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=labels,
        )

    def get_dataloader(
        self,
        rank: int,
        world_size: int,
    ) -> data_utils.DataLoader:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = data_utils.DataLoader(
            self.dataset,
            batch_size=None,
            pin_memory=False,
            collate_fn=self.collate_fn,
            sampler=sampler,
            num_workers=0,
        )
        return dataloader
