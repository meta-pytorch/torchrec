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
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from pyre_extensions import none_throws
from torchrec.datasets.criteo import CAT_FEATURE_COUNT, INT_FEATURE_COUNT


class CriteoTest(unittest.TestCase):
    """
    Superclass with helper functions for tests related to Criteo dataset. Helper
    functions include those that create mini mock/random datasets and some functions
    that validate samples returned by the dataset.
    """

    INT_FEATURE_COUNT = 13
    CAT_FEATURE_COUNT = 26

    LABEL_VAL_RANGE = (0, 1)
    INT_VAL_RANGE = (0, 100)
    CAT_VAL_RANGE = (0, 1000)

    @classmethod
    @contextlib.contextmanager
    def _create_dataset_tsv(
        cls,
        num_rows: int = 10,
        train: bool = True,
        filename: str = "criteo",
    ) -> Generator[str, None, None]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w") as f:
                rows = []
                for _ in range(num_rows):
                    row = []
                    if train:
                        row.append(str(random.randint(*cls.LABEL_VAL_RANGE)))
                    row += [
                        *(
                            str(random.randint(*cls.INT_VAL_RANGE))
                            for _ in range(cls.INT_FEATURE_COUNT)
                        ),
                        *(
                            (
                                "%x"
                                % abs(hash(str(random.randint(*cls.CAT_VAL_RANGE))))
                            ).zfill(8)[:8]
                            for _ in range(cls.CAT_FEATURE_COUNT)
                        ),
                    ]
                    rows.append(row)
                cf = csv.writer(f, delimiter="\t")
                cf.writerows(rows)
            yield path

    def _validate_sample(self, sample: Dict[str, Any], train: bool = True) -> None:
        if train:
            self.assertEqual(
                len(sample), self.INT_FEATURE_COUNT + self.CAT_FEATURE_COUNT + 1
            )
            label_val = sample["label"]
            self.assertTrue(
                self.LABEL_VAL_RANGE[0] <= label_val <= self.LABEL_VAL_RANGE[1]
            )
        else:
            self.assertEqual(
                len(sample), self.INT_FEATURE_COUNT + self.CAT_FEATURE_COUNT
            )
        for idx in range(self.INT_FEATURE_COUNT):
            int_val = sample[f"int_{idx}"]
            self.assertTrue(self.INT_VAL_RANGE[0] <= int_val <= self.INT_VAL_RANGE[1])
        for idx in range(self.CAT_FEATURE_COUNT):
            cat_val = int(sample[f"cat_{idx}"], 16)
            self.assertTrue(0 <= cat_val <= 16**8 - 1)

    @classmethod
    @contextlib.contextmanager
    def _create_dataset_npys(
        cls,
        num_rows: int = 10,
        filename: Optional[str] = "criteo",
        filenames: Optional[List[str]] = None,
        generate_dense: bool = True,
        generate_sparse: bool = True,
        generate_labels: bool = True,
        dense: Optional[np.ndarray] = None,
        sparse: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[str, ...], None, None]:
        with tempfile.TemporaryDirectory() as tmpdir:

            if filenames is None:
                # pyrefly: ignore[bad-assignment]
                filenames = [filename]

            paths = []
            # pyrefly: ignore[not-iterable]
            for filename in filenames:
                filename = none_throws(filename)

                if generate_dense:
                    # pyrefly: ignore[unsupported-operation]
                    dense_path = os.path.join(tmpdir, filename + "_dense.npy")
                    if dense is None:
                        # pyrefly: ignore[implicit-import]
                        dense = np.random.random((num_rows, INT_FEATURE_COUNT)).astype(
                            np.float32
                        )
                    np.save(dense_path, dense)
                    paths.append(dense_path)

                if generate_sparse:
                    # pyrefly: ignore[unsupported-operation]
                    sparse_path = os.path.join(tmpdir, filename + "_sparse.npy")
                    if sparse is None:
                        # pyrefly: ignore[implicit-import, no-matching-overload]
                        sparse = np.random.randint(
                            cls.CAT_VAL_RANGE[0],
                            cls.CAT_VAL_RANGE[1] + 1,
                            size=(num_rows, CAT_FEATURE_COUNT),
                            dtype=np.int32,
                        )
                    np.save(sparse_path, sparse)
                    paths.append(sparse_path)

                if generate_labels:
                    # pyrefly: ignore[unsupported-operation]
                    labels_path = os.path.join(tmpdir, filename + "_labels.npy")
                    if labels is None:
                        # pyrefly: ignore[implicit-import, no-matching-overload]
                        labels = np.random.randint(
                            cls.LABEL_VAL_RANGE[0],
                            cls.LABEL_VAL_RANGE[1] + 1,
                            size=(num_rows, 1),
                            dtype=np.int32,
                        )
                    np.save(labels_path, labels)
                    paths.append(labels_path)

            yield tuple(paths)
