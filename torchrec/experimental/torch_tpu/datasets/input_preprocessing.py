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

"""Input preprocessing utilities for TPU SparseCore."""

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from etils import epath
import torch
from torch import distributed as dist
from torchrec.experimental.torch_tpu.datasets import pybind_input_preprocessing
from torchrec.experimental.torch_tpu.datasets.fdo import csv_file_fdo_client
from torchrec.experimental.torch_tpu.datasets.fdo import fdo_client
from torchrec.experimental.torch_tpu.datasets.fdo.fdo_client import KeyedSparseCoreInputStats
from torchrec.experimental.torch_tpu.datasets.fdo.fdo_client import SparseCoreInputStats
from torchrec.experimental.torch_tpu.modules.embedding_configs import SparseCoreEmbeddingConfig
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_configs import PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclasses.dataclass
class SparseCorePreprocessedInput:
  """Container for preprocessed TPU tensors for a single table."""

  row_pointers: torch.Tensor
  embedding_ids: torch.Tensor
  sample_ids: torch.Tensor
  gains: torch.Tensor
  lengths: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
  actual_num_ids: Dict[str, int] = dataclasses.field(default_factory=dict)
  stats: Optional[SparseCoreInputStats] = None

  def to(
      self, device: torch.device, non_blocking: bool = False
  ) -> "SparseCorePreprocessedInput":
    """Moves all preprocessed tensors and lengths dictionaries to the target device.

    Args:
      device: The target PyTorch device (e.g., TPU or CPU).
      non_blocking: Whether to perform asynchronous copy if in pinned memory.

    Returns:
      A new SparseCorePreprocessedInput container on the target device.
    """
    new_lengths = {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in self.lengths.items()
    }
    return SparseCorePreprocessedInput(
        row_pointers=self.row_pointers.to(device, non_blocking=non_blocking),
        embedding_ids=self.embedding_ids.to(device, non_blocking=non_blocking),
        sample_ids=self.sample_ids.to(device, non_blocking=non_blocking),
        gains=self.gains.to(device, non_blocking=non_blocking),
        lengths=new_lengths,
        actual_num_ids=self.actual_num_ids,
        stats=self.stats,
    )

  def record_stream(self, stream: torch.Stream) -> None:
    """Records the CUDA/TPU stream usage for all underlying tensors to prevent premature recycling.

    Args:
      stream: The execution stream to record against.
    """
    self.row_pointers.record_stream(stream)
    self.embedding_ids.record_stream(stream)
    self.sample_ids.record_stream(stream)
    self.gains.record_stream(stream)
    for v in self.lengths.values():
      v.record_stream(stream)

  def pin_memory(self) -> "SparseCorePreprocessedInput":
    """Pins page-locked memory for all underlying CPU tensors for faster asynchronous device transfer.

    Returns:
      A new SparseCorePreprocessedInput container backed by pinned CPU memory.
    """
    new_lengths = {k: v.pin_memory() for k, v in self.lengths.items()}
    return SparseCorePreprocessedInput(
        row_pointers=self.row_pointers.pin_memory(),
        embedding_ids=self.embedding_ids.pin_memory(),
        sample_ids=self.sample_ids.pin_memory(),
        gains=self.gains.pin_memory(),
        lengths=new_lengths,
        actual_num_ids=self.actual_num_ids,
        stats=self.stats,
    )


class KeyedSparseCorePreprocessedInput:
  """Container for preprocessed TPU tensors used by SparseCore custom collections."""

  def __init__(
      self,
      table_tensors: Dict[str, SparseCorePreprocessedInput],
      stats: Optional[KeyedSparseCoreInputStats] = None,
  ):
    """Initializes the container with a mapping from table names to preprocessed tensor bundles.

    Args:
      table_tensors: Dictionary mapping table name string to its preprocessed
        input tensors.
      stats: Optional preprocessor diagnostic statistics across all tables.
    """
    self.table_tensors = table_tensors
    self.stats = stats

  def to(
      self, device: torch.device, non_blocking: bool = False
  ) -> "KeyedSparseCorePreprocessedInput":
    """Moves all table preprocessed tensor bundles to the specified target device.

    Args:
      device: The target PyTorch device (e.g., TPU or CPU).
      non_blocking: Whether to perform asynchronous transfer if possible.

    Returns:
      A new KeyedSparseCorePreprocessedInput container on the target device.
    """
    new_tensors = {
        table_name: table_inputs.to(device, non_blocking=non_blocking)
        for table_name, table_inputs in self.table_tensors.items()
    }
    return KeyedSparseCorePreprocessedInput(new_tensors, stats=self.stats)

  def record_stream(self, stream: torch.Stream) -> None:
    """Records stream usage across all table preprocessed tensor bundles.

    Args:
      stream: The execution stream to record against.
    """
    for table_inputs in self.table_tensors.values():
      table_inputs.record_stream(stream)

  def pin_memory(self) -> "KeyedSparseCorePreprocessedInput":
    """Pins page-locked memory across all table preprocessed tensor bundles.

    Returns:
      A new KeyedSparseCorePreprocessedInput container backed by pinned CPU
      memory.
    """
    new_tensors = {
        table_name: table_inputs.pin_memory()
        for table_name, table_inputs in self.table_tensors.items()
    }
    return KeyedSparseCorePreprocessedInput(new_tensors, stats=self.stats)


class SparseCoreInputPreprocessor:
  """CPU Helper to preprocess KeyedJaggedTensor into KeyedSparseCorePreprocessedInput."""

  def __init__(
      self,
      tables: List[SparseCoreEmbeddingConfig],
      batch_size: int,
      global_device_count: int = 1,
      num_sc_per_device: int = 2,
      allow_id_dropping: bool = False,
      fdo_client: Optional[fdo_client.FDOClient] = None,
      fdo_dir: Optional[epath.PathLike] = None,
  ) -> None:
    """Initializes the CPU input preprocessor and its stateful PyBind11 C++ backend.

    Args:
      tables: List of embedding table configurations defining sharding and
        pooling.
      batch_size: Process-local batch size per training step.
      global_device_count: Total number of TPU devices across all distributed
        hosts.
      num_sc_per_device: Number of SparseCore virtual partitions per TPU chip.
      allow_id_dropping: Whether to drop out-of-capacity embedding IDs without
        raising errors.
      fdo_client: Optional pre-constructed FDO client to record statistics.
      fdo_dir: Optional directory path to automatically create a
        CSVFileFDOClient seeded with initial table configurations.
    """
    self._batch_size = batch_size
    self._local_device_count = 1  # Always 1 for PyTorch TPU.
    self._global_device_count = global_device_count
    self._num_sc_per_device = num_sc_per_device
    self._last_stats: Optional[KeyedSparseCoreInputStats] = None

    for table in tables:
      if (
          isinstance(table.config, EmbeddingConfig)
          and table.max_seq_len is None
      ):
        raise ValueError(
            "max_seq_len must be provided for EmbeddingConfig table"
            f" {table.name}"
        )
    self._tables = tables

    if fdo_client is None and fdo_dir is not None:
      rank = dist.get_rank() if dist.is_initialized() else 0
      initial_stats = KeyedSparseCoreInputStats({
          t.name: SparseCoreInputStats(
              dropped_count=0,
              observed_max_ids=t.max_ids_per_partition,
              observed_max_unique_ids=t.max_unique_ids_per_partition,
          )
          for t in tables
      })
      fdo_client = csv_file_fdo_client.CSVFileFDOClient(
          fdo_dir, process_id=rank, initial_stats=initial_stats
      )

    self._fdo_client = fdo_client

    # Determine if we are in unpooled (EC) mode
    self._is_unpooled = isinstance(self._tables[0].config, EmbeddingConfig)
    self._allow_id_dropping = allow_id_dropping

    self._rebuild_backend()

  def _rebuild_backend(self) -> None:
    """Rebuilds metadata and re-instantiates C++ preprocessor backend."""
    self._tables_metadata = self._build_tables_metadata(
        self._tables,
        self._batch_size,
        self._global_device_count,
    )
    self._backend = pybind_input_preprocessing.SparseCorePreprocessorBackend(
        self._tables_metadata,
        self._batch_size,
        self._local_device_count,
        self._global_device_count,
        self._num_sc_per_device,
        self._allow_id_dropping,
    )

  @property
  def fdo_client(self) -> Optional[fdo_client.FDOClient]:
    """Returns the attached FDO client if present."""
    return self._fdo_client

  @property
  def last_stats(self) -> Optional[KeyedSparseCoreInputStats]:
    """Returns the FDO statistics of the most recently preprocessed batch."""
    return self._last_stats

  def publish_fdo(self) -> None:
    """Publishes FDO statistics to storage if FDO client is configured."""
    if self._fdo_client is not None:
      self._fdo_client.publish()

  def update_preprocessing_parameters(
      self,
      updated_params: Optional[KeyedSparseCoreInputStats] = None,
      scale_factor: float = 1.2,
  ) -> bool:
    """Updates table capacities in-place from loaded FDO stats and rebuilds backend.

    Follows the JAX update_preprocessing_parameters naming convention.

    Args:
      updated_params: Aggregated FDO statistics. If None, loads from the
        attached FDO client.
      scale_factor: Multiplicative safety headroom factor (default 1.2 = +20%).

    Returns:
      True if any table configuration was updated, False otherwise.
    """
    if updated_params is None:
      if self._fdo_client is None:
        raise ValueError("No FDO client configured on preprocessor.")
      updated_params = self._fdo_client.load()

    changed = False
    for table in self._tables:
      if table.name in updated_params.table_stats:
        t_stats = updated_params.table_stats[table.name]
        updated_max_ids = max(
            table.max_ids_per_partition,
            int(math.ceil(t_stats.observed_max_ids * scale_factor)),
        )
        updated_max_unique_ids = max(
            table.max_unique_ids_per_partition,
            int(math.ceil(t_stats.observed_max_unique_ids * scale_factor)),
        )
        if (
            updated_max_ids > table.max_ids_per_partition
            or updated_max_unique_ids > table.max_unique_ids_per_partition
        ):
          table.max_ids_per_partition = updated_max_ids
          table.max_unique_ids_per_partition = updated_max_unique_ids
          changed = True

    if changed:
      self._rebuild_backend()
    return changed

  def _build_tables_metadata(
      self,
      tables: List[SparseCoreEmbeddingConfig],
      batch_size: int,
      global_device_count: int = 1,
  ) -> List[Dict[str, Any]]:
    """Builds lightweight table and feature metadata dictionaries for C++ backend initialization.

    Args:
      tables: List of embedding table configurations.
      batch_size: Process-local batch size.
      global_device_count: Total number of TPU chips globally.

    Returns:
      A list of dictionaries containing table names, partition ID capacities,
      and feature sharding offsets.
    """
    tables_metadata = []
    for table in tables:
      features = []
      row_offset = 0
      for feature_name in table.feature_names:
        if isinstance(table.config, EmbeddingConfig):
          max_seq_len_table = table.max_seq_len
          assert max_seq_len_table is not None
          # For EC, fictional batch size is max_ids = batch_size * max_seq_len
          seq_batch_size = batch_size * max_seq_len_table
          batch_size_global = seq_batch_size * global_device_count
          feat_batch_size = seq_batch_size
          combiner = "sum"
        elif isinstance(table.config, EmbeddingBagConfig):
          batch_size_global = batch_size * global_device_count
          feat_batch_size = batch_size
          combiner = "sum" if table.pooling == PoolingType.SUM else "mean"
        else:
          raise TypeError(f"Unsupported table type: {type(table.config)}")

        features.append({
            "name": feature_name,
            "row_offset": row_offset,
            "col_offset": 0,
            "col_shift": 0,
            "batch_size": feat_batch_size,
            "combiner": combiner,
            "max_col_id": table.config.num_embeddings,
        })
        row_offset += batch_size_global

      tables_metadata.append({
          "name": table.name,
          "max_ids_per_partition": table.max_ids_per_partition,
          "max_unique_ids_per_partition": table.max_unique_ids_per_partition,
          "suggested_coo_buffer_size_per_device": (
              table.suggested_coo_buffer_size_per_device
          ),
          "features": features,
      })
    return tables_metadata

  def __call__(
      self, features: KeyedJaggedTensor
  ) -> KeyedSparseCorePreprocessedInput:
    """Preprocesses a batch of sparse features on CPU into CSR-wrapped COO tensors for SparseCore.

    Args:
      features: Input KeyedJaggedTensor containing sparse feature IDs and
        offsets.

    Returns:
      A KeyedSparseCorePreprocessedInput container ready for device transfer.
    """
    # This runs on CPU.

    input_indices = {}
    input_offsets = {}
    feature_dict = features.to_dict()

    if self._is_unpooled:
      table_lengths_dict = {}
      table_actual_num_ids_dict = {}

      for table in self._tables:
        input_indices[table.name] = []
        input_offsets[table.name] = []
        max_seq_len_table = table.max_seq_len
        assert max_seq_len_table is not None
        seq_batch_size = self._batch_size * max_seq_len_table
        lengths_dict = {}
        actual_num_ids_dict = {}
        for feature_name in table.feature_names:
          f = feature_dict[feature_name]
          indices = f.values().to("cpu", torch.int32)
          lengths = f.lengths().to("cpu")
          lengths_dict[feature_name] = lengths

          actual_num_ids = indices.numel()
          actual_num_ids_dict[feature_name] = actual_num_ids

          # Pad indices to max_ids
          padded_indices = torch.zeros(seq_batch_size, dtype=torch.int32)
          padded_indices[:actual_num_ids] = indices

          # Build offsets: [0, 1, 2, ..., N, N, N, ..., N] of length max_ids + 1
          padded_offsets = torch.empty(seq_batch_size + 1, dtype=torch.int32)
          padded_offsets[: actual_num_ids + 1] = torch.arange(
              0, actual_num_ids + 1, dtype=torch.int32
          )
          padded_offsets[actual_num_ids + 1 :] = actual_num_ids

          input_indices[table.name].append(padded_indices)
          input_offsets[table.name].append(padded_offsets)

        table_lengths_dict[table.name] = lengths_dict
        table_actual_num_ids_dict[table.name] = actual_num_ids_dict

      # Call official TPU preprocess op (CPU) via PyBind
      res = self._backend.preprocess(
          input_indices,
          input_offsets,
      )

      table_tensors = {}
      table_stats = {}
      for name, t in res.items():
        table_stats[name] = SparseCoreInputStats(
            dropped_count=int(t["dropped_count"].item()),
            observed_max_ids=int(t["observed_max_ids"].item()),
            observed_max_unique_ids=int(t["observed_max_unique_ids"].item()),
        )
        table_tensors[name] = SparseCorePreprocessedInput(
            row_pointers=t["row_pointers"],
            embedding_ids=t["embedding_ids"],
            sample_ids=t["sample_ids"],
            gains=t["gains"],
            lengths=table_lengths_dict.get(name, {}),
            actual_num_ids=table_actual_num_ids_dict.get(name, {}),
            stats=table_stats[name],
        )

      stats = KeyedSparseCoreInputStats(table_stats)
      self._last_stats = stats
      if self._fdo_client is not None:
        self._fdo_client.record(stats)
      return KeyedSparseCorePreprocessedInput(table_tensors, stats=stats)

    else:  # EBC mode
      for table in self._tables:
        input_indices[table.name] = []
        input_offsets[table.name] = []
        for feature_name in table.feature_names:
          f = feature_dict[feature_name]
          input_indices[table.name].append(f.values().to("cpu", torch.int32))
          input_offsets[table.name].append(f.offsets().to("cpu", torch.int32))

      res = self._backend.preprocess(
          input_indices,
          input_offsets,
      )

      table_tensors = {}
      table_stats = {}
      for name, t in res.items():
        table_stats[name] = SparseCoreInputStats(
            dropped_count=int(t["dropped_count"].item()),
            observed_max_ids=int(t["observed_max_ids"].item()),
            observed_max_unique_ids=int(t["observed_max_unique_ids"].item()),
        )
        table_tensors[name] = SparseCorePreprocessedInput(
            row_pointers=t["row_pointers"],
            embedding_ids=t["embedding_ids"],
            sample_ids=t["sample_ids"],
            gains=t["gains"],
            stats=table_stats[name],
        )

      stats = KeyedSparseCoreInputStats(table_stats)
      self._last_stats = stats
      if self._fdo_client is not None:
        self._fdo_client.record(stats)
      return KeyedSparseCorePreprocessedInput(table_tensors, stats=stats)
