#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get_dataframe_random(
    user_count: int = 50, item_count: int = 5000, size: int = 20000, min_rating: int = 2
) -> pd.DataFrame:
    uids = [random.choice(range(user_count)) for i in range(size)]
    sids = [random.choice(range(item_count)) for i in range(size)]
    ratings = [min_rating] * size
    timestamps = list(range(0, size))
    df = {"uid": uids, "sid": sids, "rating": ratings, "timestamp": timestamps}
    return pd.DataFrame(df)


def _get_dataframe_movielens(name: str, folder_path: Path) -> pd.DataFrame:
    if name == "ml-1m":
        file_path = folder_path.joinpath("ratings.dat")
        df = pd.read_csv(file_path, sep="::", header=None)
    elif name == "ml-20m":
        file_path = folder_path.joinpath("ratings.csv")
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Invalid name")
    return df


def get_raw_dataframe(
    name: str,
    user_count: int,
    item_count: int,
    size: int,
    min_rating: int,
    folder_path: Optional[str],
) -> pd.DataFrame:
    """
    Gets raw dataframe of both random and movielens

    Args:
        name (int): the random or movielens dataset name
        user_count (int): the random user count of the random set
        item_count (int): the random item count of the random set
        size (int): the random sample count of the random set
        min_rating (int): the minimum rating of the random set
        folder_path (Optional[str]): the path of the movielens dataset, None for random

    Returns:
        dataframe (pd.DataFrame): the raw dataframe

    """
    if name == "random":
        return _get_dataframe_random(user_count, item_count, size, min_rating)
    else:
        return _get_dataframe_movielens(
            name, Path(folder_path) if folder_path is not None else Path(name)
        )


class Bert4RecPreprocsser:
    """
    Bert4Rec data pre-processer to get the masked sequences for each user

    Args:
        raw_data (pd.DataFrame): the raw dataframe
        min_rating (int): minimum valid rating
        min_uc (int): minimum user ratings count
        min_sc (int): minimum item count for each valid use
        name (str): dataset for experiment, current support ml-1m, ml-20m, random
        max_len (int): max length of the Bert embedding dimensio
        mask_prob (float): probability of the mask
        dupe_factor (int): number of duplication while generating the random masked seqs

    Returns:
        Dict[str, Any]

    Example::

        raw_data = get_raw_dataframe(
            "random", 5, 40, 200, 4, None
        )
        df = Bert4RecPreprocsser(
            raw_data,
            4,
            5,
            0,
            "random",
            4,
            0.2,
            1,
        ).get_processed_dataframes()
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        min_rating: int,
        min_uc: int,
        min_sc: int,
        name: str,
        max_len: int,
        mask_prob: float,
        dupe_factor: int = 3,
    ) -> None:
        self.raw_data = raw_data
        assert min_uc >= 2, "Need at least 2 ratings per user for validation and test"
        self.min_rating = min_rating
        self.min_uc = min_uc

        self.min_sc = min_sc
        self.name = name

        self.max_len = max_len
        self.mask_prob = mask_prob
        self.dupe_factor = dupe_factor
        self.user_count = 0
        self.item_count = 0
        self.mask_token = 0

    def get_processed_dataframes(
        self,
    ) -> Dict[str, Any]:
        df = self._preprocess()
        return df

    def _preprocess(self) -> Dict[str, Any]:
        df = self._filter_ratings()
        df, umap, smap = self._densify_index(df)
        train, val, test = self._split_df(df, len(umap))
        df = {"train": train, "val": val, "test": test, "umap": umap, "smap": smap}
        self.user_count: int = len(df["umap"])
        self.item_count: int = len(df["smap"])
        self.mask_token: int = self.item_count + 1
        final_df = self._mask_and_labels(self._generate_negative_samples(df))
        return final_df

    def _filter_ratings(self) -> pd.DataFrame:
        df = self.raw_data
        df.columns = ["uid", "sid", "rating", "timestamp"]
        df = df[df["rating"] >= self.min_rating]
        print("Filtering triplets")
        if self.min_sc > 0:
            item_sizes = df.groupby("sid").size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df["sid"].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby("uid").size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df["uid"].isin(good_users)]

        return df

    def _densify_index(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
        print("Densifying index")
        umap = {u: i for i, u in enumerate(set(df["uid"]))}
        smap = {s: i + 1 for i, s in enumerate(set(df["sid"]))}
        df["uid"] = df["uid"].map(umap)
        df["sid"] = df["sid"].map(smap)
        return df, umap, smap

    def _split_df(
        self, df: pd.DataFrame, user_count: int
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
        # leave-one-out in the paper
        print("Splitting")
        user_group = df.groupby("uid")
        print(user_group)
        user2items = user_group.apply(
            lambda d: list(d.sort_values(by="timestamp")["sid"])
        )
        train, val, test = {}, {}, {}
        for user in range(user_count):
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            if len(val[user]) == 0:
                raise RuntimeError(
                    f"val set for user {user} is empty, consider increasing the data sample"
                )
            if len(test[user]) == 0:
                raise RuntimeError(
                    f"test set for user {user} is empty, consider increasing the data sample"
                )

        return train, val, test

    def _generate_negative_samples(
        self,
        df: Dict[str, Any],
    ) -> Dict[str, Any]:
        # follow the paper, no negative samples in training set
        # 100 negative samples in test set, 2 for random to save time
        test_set_sample_size = 100 if self.name != "random" else 10

        if self.name == "random":
            # use pure random to save time for random dataframe
            test_negative_samples = {}
            for user in range(self.user_count):
                seen = set(df["train"][user])
                seen.update(df["val"][user])
                seen.update(df["test"][user])
                test_negative_samples[user] = []
                for _ in range(test_set_sample_size):
                    # pyrefly: ignore[implicit-import]
                    t = np.random.randint(1, self.item_count + 1)
                    while t in seen:
                        # pyrefly: ignore[implicit-import]
                        t = np.random.randint(1, self.item_count + 1)
                    test_negative_samples[user].append(t)
        else:
            # use popularity random sampling align with paper
            popularity = Counter()
            for user in range(self.user_count):
                popularity.update(df["train"][user])
                popularity.update(df["val"][user])
                popularity.update(df["test"][user])
            (items_list, freq) = zip(*popularity.items())
            freq_sum = sum(freq)
            prob = [float(i) / freq_sum for i in freq]
            test_negative_samples = {}
            min_size = test_set_sample_size
            print("Sampling negative items")
            for user in range(self.user_count):
                seen = set(df["train"][user])
                seen.update(df["val"][user])
                seen.update(df["test"][user])
                samples = []
                while len(samples) < test_set_sample_size:
                    # pyrefly: ignore[implicit-import]
                    sampled_ids = np.random.choice(
                        items_list, test_set_sample_size * 2, replace=False, p=prob
                    )
                    sampled_ids = [x for x in sampled_ids if x not in seen]
                    samples.extend(sampled_ids[:])
                min_size = min_size if min_size < len(samples) else len(samples)
                test_negative_samples[user] = samples
            if min_size == 0:
                raise RuntimeError(
                    "we sampled 0 negative samples for a user, please increase the data size"
                )
            test_negative_samples = {
                key: value[:min_size] for key, value in test_negative_samples.items()
            }
        df["test_negative_samples"] = test_negative_samples
        return df

    def _generate_masked_train_set(
        self,
        train: Dict[int, List[int]],
        dupe_factor: int,
        need_padding: bool = True,
    ) -> pd.DataFrame:
        df = []
        for user, seq in train.items():
            sliding_step = (int)(0.1 * self.max_len)
            beg_idx = list(
                range(
                    len(seq) - self.max_len,
                    0,
                    -sliding_step if sliding_step != 0 else -1,
                )
            )
            beg_idx.append(0)
            seqs = [seq[i : i + self.max_len] for i in beg_idx[::-1]]
            for seq in seqs:
                for _ in range(dupe_factor):
                    tokens = []
                    labels = []
                    for s in seq:
                        prob = random.random()
                        if prob < self.mask_prob:
                            prob /= self.mask_prob

                            if prob < 0.8:
                                tokens.append(self.mask_token)
                            else:
                                tokens.append(random.randint(1, self.item_count))
                            labels.append(s)
                        else:
                            tokens.append(s)
                            labels.append(0)
                    if need_padding:
                        mask_len = self.max_len - len(tokens)
                        tokens = [0] * mask_len + tokens
                        labels = [0] * mask_len + labels
                    df.append([user, tokens, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "labels"])

    def _generate_labeled_eval_set(
        self,
        train: Dict[int, List[int]],
        eval: Dict[int, List[int]],
        negative_samples: Dict[int, List[int]],
        need_padding: bool = True,
    ) -> pd.DataFrame:
        df = []
        for user, seqs in train.items():
            answer = eval[user]
            negs = negative_samples[user]
            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)
            tokens = seqs
            tokens = tokens + [self.mask_token]
            tokens = tokens[-self.max_len :]
            if need_padding:
                padding_len = self.max_len - len(tokens)
                tokens = [0] * padding_len + tokens
            df.append([user, tokens, candidates, labels])
        return pd.DataFrame(df, columns=["user", "seqs", "candidates", "labels"])

    def _mask_and_labels(
        self,
        df: Dict[str, Any],
    ) -> Dict[str, Any]:
        masked_train_set = self._generate_masked_train_set(
            df["train"],
            dupe_factor=self.dupe_factor,
        )
        labled_val_set = self._generate_labeled_eval_set(
            df["train"],
            df["val"],
            df["test_negative_samples"],
        )
        train_with_valid = {
            key: df["train"].get(key, []) + df["val"].get(key, [])
            for key in set(list(df["train"].keys()) + list(df["val"].keys()))
        }
        labled_test_set = self._generate_labeled_eval_set(
            train_with_valid, df["test"], df["test_negative_samples"]
        )
        masked_df = {
            "train": masked_train_set,
            "val": labled_val_set,
            "test": labled_test_set,
            "umap": df["umap"],
            "smap": df["smap"],
        }
        return masked_df
