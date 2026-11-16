#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler


class Bert4RecTrainDataset(data_utils.Dataset):
    def __init__(
        self,
        train_set: pd.DataFrame,
    ) -> None:
        self.train_set: pd.DataFrame = train_set

    def __len__(self) -> int:
        return len(self.train_set)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        row = self.train_set.iloc[index]
        return torch.LongTensor(row["seqs"]), torch.LongTensor(row["labels"])


class BertEvalDataset(data_utils.Dataset):
    def __init__(
        self,
        eval_set: pd.DataFrame,
    ) -> None:
        self.eval_set: pd.DataFrame = eval_set

    def __len__(self) -> int:
        return len(self.eval_set)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        row = self.eval_set.iloc[index]

        return (
            torch.LongTensor(row["seqs"]),
            torch.LongTensor(row["candidates"]),
            torch.LongTensor(row["labels"]),
        )


class Bert4RecDataloader:
    def __init__(
        self,
        dataset: Dict[str, Any],
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
    ) -> None:
        self.train: pd.DataFrame = dataset["train"]
        self.val: pd.DataFrame = dataset["val"]
        self.test: pd.DataFrame = dataset["test"]
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.test_batch_size: int = test_batch_size

    def get_pytorch_dataloaders(
        self,
        rank: int,
        world_size: int,
    ) -> Tuple[data_utils.DataLoader, data_utils.DataLoader, data_utils.DataLoader]:
        """
        Gets dataloaders based on current rank and the world_size

        Args:
            rank (int): the current rank
            world_size (int): the world size of the process group

        Returns:
            dataloaders (Tuple[data_utils.DataLoader, data_utils.DataLoader, data_utils.DataLoader]): dataloaders

        """
        train_loader = self._get_train_loader(rank, world_size)
        val_loader = self._get_val_loader(rank, world_size)
        test_loader = self._get_test_loader(rank, world_size)
        return train_loader, val_loader, test_loader

    def _get_train_loader(
        self,
        rank: int,
        world_size: int,
    ) -> data_utils.DataLoader:
        sampler = DistributedSampler(
            Bert4RecTrainDataset(
                self.train,
            ),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        dataloader = data_utils.DataLoader(
            Bert4RecTrainDataset(
                self.train,
            ),
            batch_size=self.train_batch_size,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader

    def _get_val_loader(
        self,
        rank: int,
        world_size: int,
    ) -> data_utils.DataLoader:
        sampler = DistributedSampler(
            Bert4RecTrainDataset(
                self.val,
            ),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = data_utils.DataLoader(
            BertEvalDataset(
                self.val,
            ),
            batch_size=self.val_batch_size,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader

    def _get_test_loader(
        self,
        rank: int,
        world_size: int,
    ) -> data_utils.DataLoader:
        sampler = DistributedSampler(
            Bert4RecTrainDataset(
                self.test,
            ),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = data_utils.DataLoader(
            BertEvalDataset(
                self.test,
            ),
            batch_size=self.test_batch_size,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader
