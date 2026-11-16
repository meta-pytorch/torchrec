#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from ..bert4rec_movielens_datasets import Bert4RecPreprocsser, get_raw_dataframe


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
        train_set = df["train"]
        val_set = df["val"]
        test_set = df["test"]
        umap = df["umap"]
        smap = df["smap"]

        user_count = len(umap)
        item_count = len(smap)

        # check the size of the set should be as same as the user size
        self.assertEqual(len(val_set), user_count)
        self.assertEqual(len(test_set), user_count)
        self.assertEqual(len(umap), user_count)
        self.assertEqual(len(smap), item_count)

        # training set mask item check
        for index, row in train_set.iterrows():
            self.assertEqual(len(row["seqs"]), max_len)
            self.assertEqual(len(row["labels"]), len(row["seqs"]))
            for index, token in enumerate(row["seqs"]):
                if token == item_count + 1:
                    self.assertNotEqual(row["labels"][index], 0)

        for index, row in val_set.iterrows():
            self.assertEqual(len(row["seqs"]), max_len)
            self.assertEqual(row["seqs"][-1], item_count + 1)
            self.assertEqual(len(row["candidates"]), len(row["labels"]))
            for index, token in enumerate(row["candidates"]):
                if index > 0:
                    self.assertNotIn(token, row["seqs"])
            for index, label in enumerate(row["labels"]):
                self.assertEqual(label, 1 if index == 0 else 0)

        for index, row in test_set.iterrows():
            self.assertEqual(len(row["seqs"]), max_len)
            self.assertEqual(row["seqs"][-1], item_count + 1)
            self.assertEqual(len(row["candidates"]), len(row["labels"]))
            for index, token in enumerate(row["candidates"]):
                if index > 0:
                    self.assertNotIn(token, row["seqs"])
            for index, label in enumerate(row["labels"]):
                self.assertEqual(label, 1 if index == 0 else 0)
