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

"""Abstract interface for FDO client in Torch TPU Embedding."""

import abc
import collections
import dataclasses
import itertools
import os
import re
import time
from typing import Any, Dict, List

from etils import epath


@dataclasses.dataclass
class SparseCoreInputStats:
  """Container for preprocessor diagnostic statistics for a single table."""

  dropped_count: int = 0
  observed_max_ids: int = 0
  observed_max_unique_ids: int = 0


class KeyedSparseCoreInputStats:
  """Container for preprocessor diagnostic statistics across all tables."""

  def __init__(
      self,
      table_stats: Dict[str, SparseCoreInputStats],
  ):
    """Initializes the container with a mapping from table names to diagnostic statistics.

    Args:
      table_stats: Dictionary mapping table name string to its diagnostic
        statistics.
    """
    self.table_stats = table_stats


class FDOClient(abc.ABC):
  """Abstract interface for FDO client in Torch TPU Embedding.

  This class defines the interface for a per-process client that interacts with
  the FDO system. An implementation of this class should define how the FDO
  stats are recorded and published to a storage location (disk, database, etc.).
  The load method should return the current aggregated stats across all
  processes.
  """

  @abc.abstractmethod
  def record(
      self,
      data: KeyedSparseCoreInputStats,
  ) -> None:
    """Records the raw stats to process-local memory.

    Args:
      data: Input preprocessing diagnostic statistics to be recorded.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def publish(self) -> None:
    """Publishes stats to the storage location."""
    raise NotImplementedError

  @abc.abstractmethod
  def load(self) -> Any:
    """Loads state of local FDO client and returns aggregated stats.

    Returns:
      Aggregated stats across processes.
    """
    raise NotImplementedError


class BaseFileFDOClient(FDOClient):
  """Base class for file-based FDO clients."""

  _FILE_NAME = "fdo_stats"
  _FILE_EXTENSION: str = ""

  def __init__(
      self,
      base_dir: epath.PathLike,
      process_id: int = 0,
      initial_stats: KeyedSparseCoreInputStats | None = None,
  ):
    self._base_dir = epath.Path(base_dir)
    self._process_id = process_id
    self._observed_max_ids: Dict[str, int] = collections.defaultdict(int)
    self._observed_max_unique_ids: Dict[str, int] = collections.defaultdict(int)
    self._dropped_counts: Dict[str, int] = collections.defaultdict(int)

    if initial_stats is not None:
      self.record(initial_stats)

  def record(
      self,
      data: KeyedSparseCoreInputStats,
  ) -> None:
    """Records stats per process."""
    for table_name, stats in data.table_stats.items():
      self._observed_max_ids[table_name] = max(
          self._observed_max_ids[table_name], stats.observed_max_ids
      )
      self._observed_max_unique_ids[table_name] = max(
          self._observed_max_unique_ids[table_name],
          stats.observed_max_unique_ids,
      )
      self._dropped_counts[table_name] += stats.dropped_count

  def _generate_file_name(self) -> str:
    filename = f"{self._FILE_NAME}_{self._process_id}_{time.time_ns()}.{self._FILE_EXTENSION}"
    return os.fspath(self._base_dir / filename)

  def _get_latest_files_by_process(self, files: List[str]) -> List[str]:
    if not files:
      return []
    pattern = rf"{self._FILE_NAME}_(\d+)_(\d+)\.{self._FILE_EXTENSION}"
    file_groups = []
    for file in files:
      match = re.search(pattern, os.path.basename(file))
      if match:
        file_groups.append((match.group(1), int(match.group(2)), file))
    if not file_groups:
      return []
    file_groups = sorted(file_groups, reverse=True)
    latest_files = []
    for _, file_group in itertools.groupby(file_groups, key=lambda x: x[0]):
      _, _, file_name = next(file_group)
      latest_files.append(file_name)
    return latest_files
