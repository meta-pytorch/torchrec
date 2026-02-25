#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import math
import os
import random
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    BinaryCriteoUtils,
    CAT_FEATURE_COUNT,
    criteo_kaggle,
    criteo_terabyte,
    InMemoryBinaryCriteoIterDataPipe,
    INT_FEATURE_COUNT,
)
from torchrec.datasets.test_utils.criteo_test_utils import CriteoTest
from torchrec.datasets.utils import Batch


class CriteoTerabyteTest(CriteoTest):
    def test_single_file(self) -> None:
        with self._create_dataset_tsv() as dataset_pathname:
            dataset = criteo_terabyte((dataset_pathname,))
            for sample in dataset:
                self._validate_sample(sample)
            self.assertEqual(len(list(iter(dataset))), 10)

    def test_multiple_files(self) -> None:
        with contextlib.ExitStack() as stack:
            pathnames = [
                stack.enter_context(self._create_dataset_tsv()) for _ in range(3)
            ]
            dataset = criteo_terabyte(pathnames)
            for sample in dataset:
                self._validate_sample(sample)
            self.assertEqual(len(list(iter(dataset))), 30)


class CriteoKaggleTest(CriteoTest):
    def test_train_file(self) -> None:
        with self._create_dataset_tsv() as path:
            dataset = criteo_kaggle(path)
            for sample in dataset:
                self._validate_sample(sample)
            self.assertEqual(len(list(iter(dataset))), 10)

    def test_test_file(self) -> None:
        with self._create_dataset_tsv(train=False) as path:
            dataset = criteo_kaggle(path)
            for sample in dataset:
                self._validate_sample(sample, train=False)
            self.assertEqual(len(list(iter(dataset))), 10)


class CriteoDataLoaderTest(CriteoTest):
    def _validate_dataloader_sample(
        self,
        sample: Dict[str, List[Any]],
        batch_size: int,
        train: bool = True,
    ) -> None:
        unbatched_samples = [{} for _ in range(self._sample_len(sample))]
        for k, batched_values in sample.items():
            for idx, value in enumerate(batched_values):
                unbatched_samples[idx][k] = value
        for sample in unbatched_samples:
            self._validate_sample(sample, train=train)

    def _sample_len(
        self,
        sample: Dict[str, List[Any]],
    ) -> int:
        return len(next(iter(sample.values())))

    def _test_dataloader(
        self,
        num_workers: int = 0,
        batch_size: int = 1,
        num_tsvs: int = 1,
        num_rows_per_tsv: int = 10,
        train: bool = True,
    ) -> None:
        with contextlib.ExitStack() as stack:
            pathnames = [
                stack.enter_context(
                    self._create_dataset_tsv(num_rows=num_rows_per_tsv, train=train)
                )
                for _ in range(num_tsvs)
            ]
            dataset = criteo_terabyte(pathnames)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers
            )
            total_len = 0
            for sample in dataloader:
                sample_len = self._sample_len(sample)
                total_len += sample_len
                self._validate_dataloader_sample(
                    sample, batch_size=batch_size, train=train
                )
            self.assertEqual(total_len, len(list(iter(dataset))))

    def test_multiple_train_workers(self) -> None:
        self._test_dataloader(
            num_workers=4, batch_size=16, num_tsvs=5, num_rows_per_tsv=32
        )

    def test_fewer_tsvs_than_workers(self) -> None:
        self._test_dataloader(
            num_workers=2, batch_size=16, num_tsvs=1, num_rows_per_tsv=16
        )

    def test_single_worker(self) -> None:
        self._test_dataloader(batch_size=16, num_tsvs=2, num_rows_per_tsv=16)


class TestBinaryCriteoUtils(CriteoTest):
    def test_tsv_to_npys(self) -> None:
        num_rows = 10
        with self._create_dataset_tsv(num_rows=num_rows) as in_file:
            out_files = [tempfile.NamedTemporaryFile(delete=False) for _ in range(3)]
            for out_file in out_files:
                out_file.close()

            BinaryCriteoUtils.tsv_to_npys(
                in_file, out_files[0].name, out_files[1].name, out_files[2].name
            )

            dense = np.load(out_files[0].name)
            sparse = np.load(out_files[1].name)
            labels = np.load(out_files[2].name)

            self.assertEqual(dense.shape, (num_rows, INT_FEATURE_COUNT))
            self.assertEqual(dense.dtype, np.float32)
            self.assertEqual(sparse.shape, (num_rows, CAT_FEATURE_COUNT))
            self.assertEqual(sparse.dtype, np.int64)
            self.assertEqual(labels.shape, (num_rows, 1))
            self.assertEqual(labels.dtype, np.int32)

            for out_file in out_files:
                os.remove(out_file.name)

    def test_get_shape_from_npy(self) -> None:
        num_rows = 10
        with self._create_dataset_npys(num_rows=num_rows) as (
            dense_path,
            sparse_path,
            labels_path,
        ):
            dense_shape = BinaryCriteoUtils.get_shape_from_npy(dense_path)
            sparse_shape = BinaryCriteoUtils.get_shape_from_npy(sparse_path)
            labels_shape = BinaryCriteoUtils.get_shape_from_npy(labels_path)
            self.assertEqual(dense_shape, (num_rows, INT_FEATURE_COUNT))
            self.assertEqual(sparse_shape, (num_rows, CAT_FEATURE_COUNT))
            self.assertEqual(labels_shape, (num_rows, 1))

    def test_get_file_row_ranges_and_remainder(self) -> None:
        lengths = [14, 17, 20]
        world_size = 3
        expected = [{0: (0, 13), 1: (0, 2)}, {1: (3, 16), 2: (0, 2)}, {2: (3, 19)}]

        for i in range(world_size):
            self.assertEqual(
                (expected[i], 0),
                BinaryCriteoUtils.get_file_row_ranges_and_remainder(
                    lengths=lengths,
                    rank=i,
                    world_size=world_size,
                ),
            )

    def test_load_npy_range(self) -> None:
        num_rows = 10
        start_row = 2
        num_rows_to_select = 4
        with self._create_dataset_npys(
            num_rows=num_rows, generate_sparse=False, generate_labels=False
        ) as (dense_path,):
            full = np.load(dense_path)
            partial = BinaryCriteoUtils.load_npy_range(
                dense_path, start_row=start_row, num_rows=num_rows_to_select
            )
            # pyrefly: ignore[implicit-import]
            np.testing.assert_array_equal(
                full[start_row : start_row + num_rows_to_select], partial
            )

    def test_sparse_to_contiguous_ids(self) -> None:
        # Build the day .npy files. 3 days, 3 columns, 9 rows.
        unprocessed_data = [
            np.array([[10, 70, 10], [20, 80, 20], [30, 90, 30]]),  # day 0
            np.array([[20, 70, 40], [30, 80, 50], [40, 90, 60]]),  # day 1
            np.array([[20, 70, 70], [20, 80, 80], [30, 90, 90]]),  # day 2
        ]

        expected_data_no_freq_threshold = [
            np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]]),  # day 0
            np.array([[3, 2, 5], [4, 3, 6], [5, 4, 7]]),  # day 1
            np.array([[3, 2, 8], [3, 3, 9], [4, 4, 10]]),  # day 2
        ]
        self._validate_sparse_to_contiguous_preproc(
            unprocessed_data, expected_data_no_freq_threshold, 0, 3
        )

        expected_data_freq_threshold_2 = [
            np.array([[1, 2, 1], [2, 3, 1], [3, 4, 1]]),  # day 0
            np.array([[2, 2, 1], [3, 3, 1], [1, 4, 1]]),  # day 1
            np.array([[2, 2, 1], [2, 3, 1], [3, 4, 1]]),  # day 2
        ]
        self._validate_sparse_to_contiguous_preproc(
            unprocessed_data, expected_data_freq_threshold_2, 2, 3
        )

    def _validate_sparse_to_contiguous_preproc(
        self,
        unprocessed_data: List[np.ndarray],
        expected_data: List[np.ndarray],
        freq_threshold: int,
        columns: int,
    ) -> None:
        # Save the unprocessed data to temporary directory.
        temp_input_dir: str
        temp_output_dir: str
        with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
            input_files = []
            for i, data in enumerate(unprocessed_data):
                file = os.path.join(temp_input_dir, f"day_{i}_sparse.npy")
                input_files.append(file)
                np.save(file, data)

            BinaryCriteoUtils.sparse_to_contiguous(
                input_files, temp_output_dir, freq_threshold, columns
            )

            output_files = [
                os.path.join(temp_output_dir, f) for f in os.listdir(temp_output_dir)
            ]
            output_files.sort()
            for day, file in enumerate(output_files):
                processed_data = np.load(file)
                self.assertTrue(np.array_equal(expected_data[day], processed_data))

    def test_shuffle(self) -> None:
        """
        To ensure that the shuffle preserves the sanity of the input (no missing values), each row will
        be uniquely identifiable by the value in the labels column. Each row will have a unique sequence.
        The row ID will map to this sequence. The output map of row IDs to sequences must be the same as
        the input map of row IDs to sequences.
        """

        days: int = 3  # need type annotation to be captured in local function
        int_columns = 3
        cat_columns = 3

        temp_input_dir: str
        temp_output_dir: str
        with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
            dense_data = [  # 3 columns, 3 rows per day
                np.array(
                    [[i, i + 1, i + 2], [i + 3, i + 4, i + 5], [i + 6, i + 7, i + 8]]
                )
                for i in range(days)
            ]
            sparse_data = [
                np.array(
                    [[i, i + 1, i + 2], [i + 3, i + 4, i + 5], [i + 6, i + 7, i + 8]]
                )
                for i in range(days)
            ]
            labels_data = [np.array([[i], [i + 3], [i + 6]]) for i in range(3)]

            def save_data_list(data: List[np.ndarray], data_type: str) -> None:
                for day, data_ in enumerate(data):
                    file = os.path.join(temp_input_dir, f"day_{day}_{data_type}.npy")
                    np.save(file, data_)

            save_data_list(dense_data, "dense")
            save_data_list(sparse_data, "sparse")
            save_data_list(labels_data, "labels")

            rows_per_day = {0: 3, 1: 3}
            BinaryCriteoUtils.shuffle(
                temp_input_dir,
                temp_input_dir,
                temp_output_dir,
                rows_per_day,
                None,
                days,
                int_columns,
                cat_columns,
            )

            # The label is the row id in this test.
            def row_id_to_sequence(data_dir: str) -> Dict[int, List[int]]:
                id_to_sequence = {}
                for d in range(days):
                    label_data = np.load(os.path.join(data_dir, f"day_{d}_labels.npy"))
                    dense_data = np.load(os.path.join(data_dir, f"day_{d}_dense.npy"))
                    sparse_data = np.load(os.path.join(data_dir, f"day_{d}_sparse.npy"))

                    for row in range(len(label_data)):
                        label = label_data[row][0]
                        id_to_sequence[label] = [label]
                        id_to_sequence[label].extend(dense_data[row])
                        id_to_sequence[label].extend(sparse_data[row])

                return id_to_sequence

            self.assertEqual(
                row_id_to_sequence(temp_input_dir),
                row_id_to_sequence(temp_output_dir),
            )


class TestInMemoryBinaryCriteoIterDataPipe(CriteoTest):
    def _validate_batch(
        self, batch: Batch, batch_size: int, hashes: Optional[List[int]] = None
    ) -> None:
        self.assertEqual(
            tuple(batch.dense_features.size()), (batch_size, INT_FEATURE_COUNT)
        )
        self.assertEqual(
            tuple(batch.sparse_features.values().size()),
            (batch_size * CAT_FEATURE_COUNT,),
        )
        self.assertEqual(tuple(batch.labels.size()), (batch_size,))
        if hashes is not None:
            hashes_np = np.array(hashes).reshape((CAT_FEATURE_COUNT, 1))
            self.assertTrue(
                np.all(
                    batch.sparse_features.values().reshape(
                        (CAT_FEATURE_COUNT, batch_size)
                    )
                    < hashes_np
                )
            )

    def _test_dataset(
        self,
        rows_per_file: List[int],
        batch_size: int,
        world_size: int,
        stage: str = "train",
    ) -> None:
        with contextlib.ExitStack() as stack:
            num_rows = sum(rows_per_file)
            if stage == "train":
                dense, sparse, labels = None, None, None
            else:
                dense = np.mgrid[0:num_rows, 0:INT_FEATURE_COUNT][0]
                sparse = np.mgrid[0:num_rows, 0:CAT_FEATURE_COUNT][0]
                labels = np.ones((num_rows, 1))
            files = [
                stack.enter_context(
                    self._create_dataset_npys(
                        num_rows=num_rows, dense=dense, sparse=sparse, labels=labels
                    )
                )
                for num_rows in rows_per_file
            ]
            hashes = [i + 1 for i in range(CAT_FEATURE_COUNT)]

            if stage == "train":
                dataset_start = 0
                dataset_len = num_rows
            elif stage == "val":
                dataset_start = 0
                dataset_len = num_rows // 2 + num_rows % 2
            else:
                dataset_start = num_rows // 2 + num_rows % 2
                dataset_len = num_rows // 2

            lens = []
            remainder = dataset_len % (world_size * batch_size)
            total_samples_count = 0
            for rank in range(world_size):
                end_batch_size = remainder // world_size
                num_batches = math.ceil(dataset_len / (world_size * batch_size))
                if 0 < remainder < world_size:
                    num_batches -= 1
                    end_batch_size = batch_size
                if rank < (remainder % world_size):
                    end_batch_size += 1
                datapipe = InMemoryBinaryCriteoIterDataPipe(
                    stage=stage,
                    dense_paths=[f[0] for f in files],
                    sparse_paths=[f[1] for f in files],
                    labels_paths=[f[2] for f in files],
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                    hashes=hashes,
                )
                datapipe_len = len(datapipe)
                self.assertEqual(datapipe_len, num_batches)

                len_ = 0
                for batch in datapipe:
                    if stage in ["val", "test"] and len_ == 0 and rank == 0:
                        self.assertEqual(
                            batch.dense_features[0, 0].item(),
                            dataset_start,
                        )
                    if len_ < num_batches - 1 or end_batch_size == 0:
                        self._validate_batch(batch, batch_size=batch_size)
                    else:
                        self._validate_batch(batch, batch_size=end_batch_size)
                    len_ += 1
                    total_samples_count += batch.dense_features.shape[0]

                # Check that dataset __len__ matches true length.
                self.assertEqual(datapipe_len, len_)
                lens.append(len_)

            # Ensure all ranks return the same number of batches.
            self.assertEqual(len(set(lens)), 1)
            # Ensure the number of samples read match the number of samples in the dataset
            self.assertEqual(dataset_len, total_samples_count)

    def _test_in_memory_training_set_shuffle(
        self,
        rows_per_file: List[int],
        batch_size: int,
        world_size: int,
        random_seed: int = 0,
    ) -> None:
        with contextlib.ExitStack() as stack:
            num_rows_csr = np.cumsum([0] + rows_per_file)
            dense, sparse, labels = [], [], []
            for i, _ in enumerate(rows_per_file):
                start = num_rows_csr[i]
                end = num_rows_csr[i + 1]
                dense.append(np.mgrid[start:end, 0:INT_FEATURE_COUNT][0])
                sparse.append(np.mgrid[start:end, 0:CAT_FEATURE_COUNT][0])
                labels.append(np.mgrid[start:end, 0:1][0])
            files = [
                stack.enter_context(
                    self._create_dataset_npys(
                        num_rows=num_rows,
                        dense=dense[i],
                        sparse=sparse[i],
                        labels=labels[i],
                    )
                )
                for i, num_rows in enumerate(rows_per_file)
            ]
            hashes = [i + 1 for i in range(CAT_FEATURE_COUNT)]
            dataset_len = num_rows = sum(rows_per_file)
            remainder = dataset_len % world_size
            rows_per_rank = dataset_len // world_size
            rows_per_rank = np.array([rows_per_rank for _ in range(world_size)])
            rows_per_rank[:remainder] += 1
            lens = []
            total_samples_count = 0
            for rank in range(world_size):
                shuffle_dataset_len = rows_per_rank[rank]
                num_batches = math.ceil(shuffle_dataset_len / batch_size)
                end_batch_size = shuffle_dataset_len % batch_size

                datapipe = InMemoryBinaryCriteoIterDataPipe(
                    stage="train",
                    dense_paths=[f[0] for f in files],
                    sparse_paths=[f[1] for f in files],
                    labels_paths=[f[2] for f in files],
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                    shuffle_training_set=True,
                    shuffle_training_set_random_seed=random_seed,
                    hashes=hashes,
                )
                datapipe_len = len(datapipe)
                self.assertEqual(datapipe_len, num_batches)

                # pyrefly: ignore [bad-argument-type, implicit-import]
                np.random.seed(random_seed)
                # pyrefly: ignore[implicit-import]
                permutation_arr = np.random.permutation(num_rows)
                src_arr = np.arange(num_rows)
                target_shuffled_arr = np.empty(num_rows)
                target_shuffled_arr[permutation_arr] = src_arr
                target_shuffled_arr_ = np.array(
                    [b for a, b in sorted(zip(permutation_arr, src_arr))]
                )
                # pyrefly: ignore[implicit-import]
                np.testing.assert_array_equal(target_shuffled_arr, target_shuffled_arr_)
                len_ = 0
                for batch in datapipe:
                    if len_ < num_batches - 1 or end_batch_size == 0:
                        self._validate_batch(batch, batch_size=batch_size)
                    else:
                        self._validate_batch(batch, batch_size=end_batch_size)
                    len_ += 1
                    total_samples_count += batch.dense_features.shape[0]

                # Check that dataset __len__ matches true length.
                self.assertEqual(datapipe_len, len_)
                lens.append(len_)

            # Ensure all ranks return the same number of batches.
            self.assertEqual(len(set(lens)), 1)
            # Ensure the number of samples read match the number of samples in the dataset
            self.assertEqual(dataset_len, total_samples_count)

    def test_dataset_small_files(self) -> None:
        self._test_dataset([1] * 20, 4, 2)

    def test_dataset_random_sized_files(self) -> None:
        random.seed(0)
        self._test_dataset([random.randint(1, 100) for _ in range(100)], 16, 3)
        # Test case where the global batch size does not evenly divide
        # dataset_len but the local batch size does.
        self._test_dataset([352], batch_size=32, world_size=8)

    def test_dataset_train_val_and_test_sets(self) -> None:
        for stage in ["train", "val", "test"]:
            # Test cases where batch_size evenly divides dataset_len.
            self._test_dataset([100], batch_size=1, world_size=2, stage=stage)
            self._test_dataset([101], batch_size=1, world_size=2, stage=stage)
            # Test cases where the first and only batch is an incomplete batch.
            self._test_dataset([100], batch_size=32, world_size=8, stage=stage)
            self._test_dataset([101], batch_size=32, world_size=8, stage=stage)
            # Test cases where batches are full size followed by a last batch that is incomplete.
            self._test_dataset([10000], batch_size=128, world_size=8, stage=stage)
            self._test_dataset([10001], batch_size=128, world_size=8, stage=stage)

    def test_in_memory_training_set_shuffle_driver(self) -> None:
        self._test_in_memory_training_set_shuffle([100] * 10, 32, 4, random_seed=0)
        self._test_in_memory_training_set_shuffle([100] * 10, 32, 4, random_seed=100)
        self._test_in_memory_training_set_shuffle([10000], 128, 8, random_seed=0)
        self._test_in_memory_training_set_shuffle([10000], 128, 8, random_seed=100)
