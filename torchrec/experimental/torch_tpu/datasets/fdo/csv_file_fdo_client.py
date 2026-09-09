#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An FDO client implementation that uses CSV files as storage."""

import collections
import csv
import glob
import os
from typing import Dict

from absl import logging
from torchrec.experimental.torch_tpu.datasets.fdo import fdo_client


class CSVFileFDOClient(fdo_client.BaseFileFDOClient):
    """FDO client that writes stats to a file in .csv format."""

    _FILE_EXTENSION = "csv"

    def publish(self) -> None:
        """Publishes locally accumulated stats to a CSV file in base_dir."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._generate_file_name()
        logging.info("Writing FDO CSV stats to %s", file_name)
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["table_name", "max_ids", "max_unique_ids", "dropped_count"]
            )
            all_tables = set(self._observed_max_ids.keys())
            for t_name in sorted(all_tables):
                writer.writerow(
                    [
                        t_name,
                        self._observed_max_ids[t_name],
                        self._observed_max_unique_ids[t_name],
                        self._dropped_counts[t_name],
                    ]
                )

    def load(self) -> fdo_client.KeyedSparseCoreInputStats:
        """Loads state of local FDO client from disk and returns aggregated stats."""
        files_glob = os.fspath(
            self._base_dir / f"{self._FILE_NAME}*.{self._FILE_EXTENSION}"
        )
        files = self._get_latest_files_by_process(glob.glob(files_glob))
        if not files:
            raise FileNotFoundError(f"No stats files found in {files_glob}")

        max_ids: Dict[str, int] = collections.defaultdict(int)
        max_unique_ids: Dict[str, int] = collections.defaultdict(int)
        dropped_counts: Dict[str, int] = collections.defaultdict(int)

        for file_name in files:
            logging.info("Reading FDO CSV stats from %s", file_name)
            with open(file_name, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_name = row["table_name"]
                    max_ids[t_name] = max(max_ids[t_name], int(row["max_ids"]))
                    max_unique_ids[t_name] = max(
                        max_unique_ids[t_name], int(row["max_unique_ids"])
                    )
                    dropped_counts[t_name] += int(row["dropped_count"])

        table_stats = {}
        for t_name in max_ids:
            table_stats[t_name] = fdo_client.SparseCoreInputStats(
                dropped_count=dropped_counts[t_name],
                observed_max_ids=max_ids[t_name],
                observed_max_unique_ids=max_unique_ids[t_name],
            )

        return fdo_client.KeyedSparseCoreInputStats(table_stats)
