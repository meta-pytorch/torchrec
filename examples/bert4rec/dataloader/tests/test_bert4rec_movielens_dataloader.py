#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from ...data.bert4rec_movielens_datasets import Bert4RecPreprocsser, get_raw_dataframe
from ..bert4rec_movielens_dataloader import Bert4RecDataloader


class MainTest(unittest.TestCase):
    def test_random(self) -> None:
        min_rating = 4
        min_uc = 5
        min_sc = 0
        name = "random"
        max_len = 4
        mask_prob = 0.2
        random_user_count = 5
        random_item_count = 40
        random_size = 200
        dupe_factor = 1
        raw_data = get_raw_dataframe(
            name, random_user_count, random_item_count, random_size, min_rating, None
        )
        df = Bert4RecPreprocsser(
            raw_data,
            min_rating,
            min_uc,
            min_sc,
            name,
            max_len,
            mask_prob,
            dupe_factor,
        ).get_processed_dataframes()
        batch_size = 1
        bert4recDataloader = Bert4RecDataloader(df, batch_size, batch_size, batch_size)
        (
            train_loader,
            val_loader,
            test_loader,
        ) = bert4recDataloader.get_pytorch_dataloaders(rank=0, world_size=1)

        for seqs, labels in train_loader:
            self.assertEqual(seqs.size(0), batch_size)
            self.assertEqual(seqs.size(0), labels.size(0))
            self.assertEqual(seqs.size(1), labels.size(1))
            self.assertEqual(seqs.size(1), max_len)

            for i in range(seqs.size(0)):
                for j in range(seqs.size(1)):
                    # padding
                    if seqs[i][j].item() == 0:
                        self.assertEqual(labels[i][j].item(), 0)
                    # masked item. Label should be real item id
                    if seqs[i][j].item() == random_item_count + 1:
                        self.assertGreater(labels[i][j].item(), 0)

        for seqs, candidates, labels in val_loader:
            self.assertEqual(seqs.size(0), candidates.size(0))
            self.assertEqual(seqs.size(0), labels.size(0))
            self.assertEqual(candidates.size(1), labels.size(1))

        for seqs, candidates, labels in test_loader:
            self.assertEqual(seqs.size(0), candidates.size(0))
            self.assertEqual(seqs.size(0), labels.size(0))
            self.assertEqual(candidates.size(1), labels.size(1))
