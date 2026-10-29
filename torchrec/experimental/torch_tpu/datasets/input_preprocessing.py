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

"""Input preprocessing utilities for TPU SparseCore."""

import dataclasses
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from etils import epath
from torch import distributed as dist
from torchrec.experimental.torch_tpu.datasets import pybind_input_preprocessing
from torchrec.experimental.torch_tpu.datasets.fdo import csv_file_fdo_client, fdo_client
from torchrec.experimental.torch_tpu.datasets.fdo.fdo_client import (
    KeyedSparseCoreInputStats,
    SparseCoreInputStats,
)
from torchrec.experimental.torch_tpu.modules import utils
from torchrec.experimental.torch_tpu.modules.embedding_configs import (
    SparseCoreEmbeddingConfig,
    StackedSparseCoreEmbeddingConfig,
)
from torchrec.experimental.torch_tpu.modules.table_stacking import (
    auto_stack_tables,
    prepare_tables_for_stacking,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

MinibatchingMode = pybind_input_preprocessing.MinibatchingMode


def _round_up_bucket(val: int, min_bucket: int = 32) -> int:
    """Rounds `val` up to the next power-of-2 bucket (at least `min_bucket`)."""
    return max(min_bucket, 1 << (max(1, val) - 1).bit_length())


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
    enable_minibatching: bool = False
    max_ids_per_partition: Optional[int] = None
    max_unique_ids_per_partition: Optional[int] = None

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
            k: v.to(device, non_blocking=non_blocking) for k, v in self.lengths.items()
        }
        return SparseCorePreprocessedInput(
            row_pointers=self.row_pointers.to(device, non_blocking=non_blocking),
            embedding_ids=self.embedding_ids.to(device, non_blocking=non_blocking),
            sample_ids=self.sample_ids.to(device, non_blocking=non_blocking),
            gains=self.gains.to(device, non_blocking=non_blocking),
            lengths=new_lengths,
            actual_num_ids=self.actual_num_ids,
            stats=self.stats,
            enable_minibatching=self.enable_minibatching,
            max_ids_per_partition=self.max_ids_per_partition,
            max_unique_ids_per_partition=self.max_unique_ids_per_partition,
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
            enable_minibatching=self.enable_minibatching,
            max_ids_per_partition=self.max_ids_per_partition,
            max_unique_ids_per_partition=self.max_unique_ids_per_partition,
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
    """CPU Helper to preprocess KeyedJaggedTensor into KeyedSparseCorePreprocessedInput.

    Note:
      When `online_fdo=True`, preprocessing should run in the main process or a
      thread-based `PrefetchDataLoader` rather than a multi-process
      `DataLoader(num_workers > 0)` so that in-memory FDO state is shared across
      batches.
    """

    def __init__(
        self,
        tables: List[SparseCoreEmbeddingConfig],
        batch_size: int,
        global_device_count: int = 1,
        num_sc_per_device: int = 2,
        sc_simd_width: Optional[int] = None,
        hbm_word_size_in_4b: Optional[int] = None,
        allow_id_dropping: bool = False,
        fdo_client: Optional[fdo_client.FDOClient] = None,
        fdo_dir: Optional[epath.PathLike] = None,
        auto_stack: bool = False,
        activation_mem_bytes_limit: int = 6 * 1024 * 1024,
        minibatching_mode: Union[MinibatchingMode, str] = MinibatchingMode.DISABLED,
        online_fdo: bool = False,
        fdo_headroom: float = 1.5,
    ) -> None:
        """Initializes the CPU input preprocessor and its stateful PyBind11 C++ backend.

        Args:
          tables: List of embedding table configurations defining sharding and
            pooling.
          batch_size: Process-local batch size per training step.
          global_device_count: Total number of TPU devices across all distributed
            hosts.
          num_sc_per_device: Number of SparseCores per TPU device.
          sc_simd_width: SIMD width of each SparseCore. Defaults to the
            `SC_SIMD_WIDTH` environment variable or the SIMD width of the attached
            TPU, so it must be given on hosts without a TPU with SparseCores.
          hbm_word_size_in_4b: HBM word size in 4-byte units.
          allow_id_dropping: Whether to drop out-of-capacity embedding IDs without
            raising errors.
          fdo_client: Optional pre-constructed FDO client to record statistics.
          fdo_dir: Optional directory path to automatically create a
            CSVFileFDOClient seeded with initial table configurations.
          auto_stack: Whether to automatically group compatible tables for stacking.
          activation_mem_bytes_limit: Memory limit in bytes for activation buffers
            per SparseCore partition.
          minibatching_mode: Minibatching mode (`MinibatchingMode.DISABLED`,
            `MinibatchingMode.HOST`, `MinibatchingMode.DEVICE`, or string
            `"disabled"`, `"host"`, `"device"`).
          online_fdo: Whether to enable zero-config online dynamic FDO. On single
            device (`global_device_count == 1`), automatically steps up to the next
            power-of-2 bucket and re-preprocesses the batch on CPU whenever IDs are
            dropped. On multi-device (`global_device_count > 1`), records running
            peak stats and requires calling `sync_online_fdo()` at synchronized step
            or epoch boundaries across all ranks so all devices step up to identical
            shapes simultaneously.
          fdo_headroom: Multiplicative safety headroom factor when `online_fdo=True`
            (default 1.5, must be >= 1.0).
        """
        if fdo_headroom < 1.0:
            raise ValueError(f"fdo_headroom must be >= 1.0, got {fdo_headroom}")

        if isinstance(minibatching_mode, str):
            mode_map = {
                "disabled": MinibatchingMode.DISABLED,
                "host": MinibatchingMode.HOST,
                "device": MinibatchingMode.DEVICE,
            }
            key = minibatching_mode.lower()
            if key not in mode_map:
                raise ValueError(
                    f"Unsupported minibatching_mode: {minibatching_mode!r}. Expected"
                    " one of 'disabled', 'host', or 'device'."
                )
            self._minibatching_mode = mode_map[key]
        else:
            self._minibatching_mode = minibatching_mode
        self._batch_size = batch_size
        self._local_device_count = 1  # Always 1 for PyTorch TPU.
        self._global_device_count = global_device_count
        self._num_sc_per_device = num_sc_per_device
        self._sc_simd_width = utils.get_sc_simd_width(sc_simd_width)
        self._hbm_word_size_in_4b = utils.get_hbm_word_size_in_4b(hbm_word_size_in_4b)
        self._last_stats: Optional[KeyedSparseCoreInputStats] = None
        self._online_fdo = online_fdo
        self._fdo_headroom = fdo_headroom
        self._running_peak_stats: Dict[str, SparseCoreInputStats] = {}
        self._warned_multihost_online_fdo = False

        for table in tables:
            if (
                table.max_ids_per_partition <= 0
                or table.max_unique_ids_per_partition <= 0
            ):
                raise ValueError(
                    f"max_ids_per_partition ({table.max_ids_per_partition}) and"
                    " max_unique_ids_per_partition"
                    f" ({table.max_unique_ids_per_partition}) must be > 0 for table"
                    f" {table.name}"
                )
            if isinstance(table.config, EmbeddingConfig) and table.max_seq_len is None:
                raise ValueError(
                    "max_seq_len must be provided for EmbeddingConfig table"
                    f" {table.name}"
                )
        if auto_stack:
            auto_stack_tables(
                tables,
                global_device_count=global_device_count,
                num_sc_per_device=num_sc_per_device,
                batch_size=batch_size,
                activation_mem_bytes_limit=activation_mem_bytes_limit,
            )
        self._tables = tables
        self._stacked_configs = prepare_tables_for_stacking(
            tables,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
        )

        if fdo_client is None and fdo_dir is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            initial_stats = KeyedSparseCoreInputStats(
                {
                    stack_config.stack_name: SparseCoreInputStats(
                        dropped_count=0,
                        observed_max_ids=stack_config.max_ids_per_partition,
                        observed_max_unique_ids=stack_config.max_unique_ids_per_partition,
                    )
                    for stack_config in self._stacked_configs
                }
            )
            fdo_client = csv_file_fdo_client.CSVFileFDOClient(
                fdo_dir, process_id=rank, initial_stats=initial_stats
            )

        self._fdo_client = fdo_client

        # Determine if we are in unpooled (EC) mode
        self._is_unpooled = isinstance(self._tables[0].config, EmbeddingConfig)
        self._allow_id_dropping = allow_id_dropping or online_fdo

        self._rebuild_backend()

    def _rebuild_backend(self) -> None:
        """Rebuilds metadata and re-instantiates C++ preprocessor backend."""
        self._tables_metadata = self._build_tables_metadata(
            self._stacked_configs,
            self._batch_size,
            self._global_device_count,
        )
        self._backend = pybind_input_preprocessing.SparseCorePreprocessorBackend(
            self._tables_metadata,
            self._batch_size,
            self._local_device_count,
            self._global_device_count,
            self._num_sc_per_device,
            self._sc_simd_width,
            self._hbm_word_size_in_4b,
            self._allow_id_dropping,
            self._minibatching_mode,
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

    def _update_running_peak_stats(self, stats: KeyedSparseCoreInputStats) -> None:
        """Updates in-memory running peak statistics per stack."""
        for name, s in stats.table_stats.items():
            prev = self._running_peak_stats.get(name)
            if prev is None:
                self._running_peak_stats[name] = s
            else:
                self._running_peak_stats[name] = SparseCoreInputStats(
                    dropped_count=prev.dropped_count + s.dropped_count,
                    observed_max_ids=max(prev.observed_max_ids, s.observed_max_ids),
                    observed_max_unique_ids=max(
                        prev.observed_max_unique_ids, s.observed_max_unique_ids
                    ),
                )

    def sync_online_fdo(self, device: Optional[torch.device] = None) -> bool:
        """Synchronizes running peak FDO stats across ranks and steps up to common power-of-2 buckets.

        Args:
          device: Optional device on which to allocate the reduction tensor when
            `torch.distributed` is initialized. If None, selects CPU for `gloo` and
            the active TPU device otherwise.

        Returns:
          True if any table or stack configuration was updated, False otherwise.
        """
        if dist.is_initialized() and self._global_device_count > 1:
            if device is None:
                if dist.get_backend() == "gloo":
                    target_device = torch.device("cpu")
                else:
                    target_device = utils.get_device()
            else:
                target_device = device

            rows = []
            for stack_config in self._stacked_configs:
                peak = self._running_peak_stats.get(stack_config.stack_name)
                rows.append(
                    [
                        peak.dropped_count if peak is not None else 0,
                        peak.observed_max_ids if peak is not None else 0,
                        peak.observed_max_unique_ids if peak is not None else 0,
                    ]
                )
            stats_tensor = torch.tensor(rows, dtype=torch.int64, device=target_device)
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.MAX)
            synced_rows = stats_tensor.cpu().tolist()
            synced_stats = KeyedSparseCoreInputStats(
                {
                    stack_config.stack_name: SparseCoreInputStats(
                        dropped_count=int(row[0]),
                        observed_max_ids=int(row[1]),
                        observed_max_unique_ids=int(row[2]),
                    )
                    for stack_config, row in zip(self._stacked_configs, synced_rows)
                }
            )
        else:
            if not self._running_peak_stats:
                return False
            synced_stats = KeyedSparseCoreInputStats(dict(self._running_peak_stats))

        return self.update_preprocessing_parameters(
            synced_stats, scale_factor=self._fdo_headroom, bucket_step_up=True
        )

    def update_preprocessing_parameters(
        self,
        updated_params: Optional[KeyedSparseCoreInputStats] = None,
        scale_factor: float = 1.2,
        bucket_step_up: bool = False,
    ) -> bool:
        """Updates table capacities in-place from loaded FDO stats and rebuilds backend.

        Follows the JAX update_preprocessing_parameters naming convention.

        Args:
          updated_params: Aggregated FDO statistics. If None, loads from the
            attached FDO client.
          scale_factor: Multiplicative safety headroom factor (default 1.2 = +20%,
            must be >= 1.0).
          bucket_step_up: Whether to round updated capacities up to the next
            power-of-2 bucket (minimum 32) to bound recompilations.

        Returns:
          True if any table configuration was updated, False otherwise.
        """
        if scale_factor < 1.0:
            raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")

        if updated_params is None:
            if self._fdo_client is None:
                raise ValueError("No FDO client configured on preprocessor.")
            updated_params = self._fdo_client.load()

        changed = False
        for stack_config in self._stacked_configs:
            t_stats = updated_params.table_stats.get(stack_config.stack_name)
            if t_stats is None:
                matching_stats = [
                    updated_params.table_stats[t.name]
                    for t in stack_config.tables
                    if t.name in updated_params.table_stats
                ]
                if matching_stats:
                    t_stats = SparseCoreInputStats(
                        dropped_count=sum(s.dropped_count for s in matching_stats),
                        observed_max_ids=max(
                            s.observed_max_ids for s in matching_stats
                        ),
                        observed_max_unique_ids=max(
                            s.observed_max_unique_ids for s in matching_stats
                        ),
                    )
            if t_stats is None:
                continue

            raw_max_ids = int(math.ceil(t_stats.observed_max_ids * scale_factor))
            raw_max_unique_ids = int(
                math.ceil(t_stats.observed_max_unique_ids * scale_factor)
            )
            if bucket_step_up:
                raw_max_ids = _round_up_bucket(raw_max_ids)
                raw_max_unique_ids = _round_up_bucket(raw_max_unique_ids)

            current_max_ids = stack_config.max_ids_per_partition
            current_max_unique_ids = stack_config.max_unique_ids_per_partition
            updated_max_ids = max(current_max_ids, raw_max_ids)
            updated_max_unique_ids = max(current_max_unique_ids, raw_max_unique_ids)

            limits_increased = (
                updated_max_ids > current_max_ids
                or updated_max_unique_ids > current_max_unique_ids
            )
            if stack_config.suggested_coo_buffer_size_per_device is not None and (
                limits_increased or t_stats.dropped_count > 0
            ):
                word_size = self._hbm_word_size_in_4b
                num_scs = self._global_device_count * self._num_sc_per_device
                max_ids_rounded_up = (
                    (updated_max_ids + word_size - 1) // word_size
                ) * word_size
                num_buckets = (
                    64 if self._minibatching_mode != MinibatchingMode.DISABLED else 1
                )
                theoretical_max = (
                    max_ids_rounded_up * self._num_sc_per_device * num_scs * num_buckets
                )
                alignment = word_size * self._num_sc_per_device
                old_rounded = max(
                    word_size,
                    ((current_max_ids + word_size - 1) // word_size) * word_size,
                )
                buf_scale = max(
                    max_ids_rounded_up / old_rounded,
                    scale_factor if t_stats.dropped_count > 0 else 1.0,
                )
                scaled_buf = int(
                    math.ceil(
                        stack_config.suggested_coo_buffer_size_per_device * buf_scale
                    )
                )
                if bucket_step_up and t_stats.dropped_count > 0:
                    scaled_buf = _round_up_bucket(scaled_buf)
                scaled_buf = ((scaled_buf + alignment - 1) // alignment) * alignment
                min_device_buf = (
                    max_ids_rounded_up * self._num_sc_per_device
                    if limits_increased
                    else 0
                )
                new_buf = min(
                    theoretical_max,
                    max(
                        stack_config.suggested_coo_buffer_size_per_device,
                        scaled_buf,
                        min_device_buf,
                    ),
                )
                if new_buf > stack_config.suggested_coo_buffer_size_per_device:
                    stack_config.suggested_coo_buffer_size_per_device = new_buf
                    for table in stack_config.tables:
                        if table.suggested_coo_buffer_size_per_device is not None:
                            table.suggested_coo_buffer_size_per_device = new_buf
                    changed = True

            if limits_increased:
                stack_config.max_ids_per_partition = updated_max_ids
                stack_config.max_unique_ids_per_partition = updated_max_unique_ids
                for table in stack_config.tables:
                    table.max_ids_per_partition = updated_max_ids
                    table.max_unique_ids_per_partition = updated_max_unique_ids
                changed = True

        if changed:
            self._rebuild_backend()
        return changed

    def _build_tables_metadata(
        self,
        stacked_configs: List[StackedSparseCoreEmbeddingConfig],
        batch_size: int,
        global_device_count: int = 1,
    ) -> List[Dict[str, Any]]:
        """Builds lightweight table and feature metadata dictionaries for C++ backend initialization."""
        tables_metadata = []
        num_sparsecores = global_device_count * self._num_sc_per_device

        for stack_config in stacked_configs:
            features = []
            row_offset = 0
            for table in stack_config.tables:
                col_offset = stack_config.get_col_offset(table.name, num_sparsecores)
                col_shift = stack_config.get_shard_rotation(table.name)

                for feature_name in table.feature_names:
                    if isinstance(table.config, EmbeddingConfig):
                        max_seq_len_table = table.max_seq_len
                        assert max_seq_len_table is not None
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

                    features.append(
                        {
                            "name": feature_name,
                            "row_offset": row_offset,
                            "col_offset": col_offset,
                            "col_shift": col_shift,
                            "batch_size": feat_batch_size,
                            "combiner": combiner,
                            "max_col_id": table.config.num_embeddings,
                        }
                    )
                    row_offset += batch_size_global

            tables_metadata.append(
                {
                    "name": stack_config.stack_name,
                    "max_ids_per_partition": stack_config.max_ids_per_partition,
                    "max_unique_ids_per_partition": (
                        stack_config.max_unique_ids_per_partition
                    ),
                    "suggested_coo_buffer_size_per_device": (
                        stack_config.suggested_coo_buffer_size_per_device or 0
                    ),
                    "features": features,
                }
            )
        return tables_metadata

    def __call__(self, features: KeyedJaggedTensor) -> KeyedSparseCorePreprocessedInput:
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
        table_lengths_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        table_actual_num_ids_dict: Dict[str, Dict[str, int]] = {}

        if self._is_unpooled:
            for stack_config in self._stacked_configs:
                input_indices[stack_config.stack_name] = []
                input_offsets[stack_config.stack_name] = []
                lengths_dict = {}
                actual_num_ids_dict = {}
                for table in stack_config.tables:
                    max_seq_len_table = table.max_seq_len
                    assert max_seq_len_table is not None
                    seq_batch_size = self._batch_size * max_seq_len_table
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
                        padded_offsets = torch.empty(
                            seq_batch_size + 1, dtype=torch.int32
                        )
                        padded_offsets[: actual_num_ids + 1] = torch.arange(
                            0, actual_num_ids + 1, dtype=torch.int32
                        )
                        padded_offsets[actual_num_ids + 1 :] = actual_num_ids

                        input_indices[stack_config.stack_name].append(padded_indices)
                        input_offsets[stack_config.stack_name].append(padded_offsets)

                table_lengths_dict[stack_config.stack_name] = lengths_dict
                table_actual_num_ids_dict[stack_config.stack_name] = actual_num_ids_dict
        else:  # EBC mode
            for stack_config in self._stacked_configs:
                input_indices[stack_config.stack_name] = []
                input_offsets[stack_config.stack_name] = []
                for table in stack_config.tables:
                    for feature_name in table.feature_names:
                        f = feature_dict[feature_name]
                        input_indices[stack_config.stack_name].append(
                            f.values().to("cpu", torch.int32)
                        )
                        input_offsets[stack_config.stack_name].append(
                            f.offsets().to("cpu", torch.int32)
                        )

        res = self._backend.preprocess(
            input_indices,
            input_offsets,
        )

        table_stats = {
            name: SparseCoreInputStats(
                dropped_count=int(t["dropped_count"].item()),
                observed_max_ids=int(t["observed_max_ids"].item()),
                observed_max_unique_ids=int(t["observed_max_unique_ids"].item()),
            )
            for name, t in res.items()
        }
        stats = KeyedSparseCoreInputStats(table_stats)

        if self._online_fdo:
            if self._global_device_count == 1:
                if any(s.dropped_count > 0 for s in stats.table_stats.values()):
                    if self.update_preprocessing_parameters(
                        stats, scale_factor=self._fdo_headroom, bucket_step_up=True
                    ):
                        res = self._backend.preprocess(
                            input_indices,
                            input_offsets,
                        )
                        table_stats = {
                            name: SparseCoreInputStats(
                                dropped_count=int(t["dropped_count"].item()),
                                observed_max_ids=int(t["observed_max_ids"].item()),
                                observed_max_unique_ids=int(
                                    t["observed_max_unique_ids"].item()
                                ),
                            )
                            for name, t in res.items()
                        }
                        stats = KeyedSparseCoreInputStats(table_stats)
            elif (
                any(s.dropped_count > 0 for s in stats.table_stats.values())
                and not self._warned_multihost_online_fdo
            ):
                self._warned_multihost_online_fdo = True
                logging.warning(
                    "SparseCore IDs were dropped with online_fdo=True and"
                    " global_device_count=%d. Per-batch automatic step-up is disabled"
                    " in multi-device mode to prevent SPMD collective shape"
                    " divergence; invoke preprocessor.sync_online_fdo() at a"
                    " synchronized step boundary across all ranks.",
                    self._global_device_count,
                )

        self._update_running_peak_stats(stats)

        enable_minibatching = self._minibatching_mode != MinibatchingMode.DISABLED
        stack_configs_by_name = {sc.stack_name: sc for sc in self._stacked_configs}
        table_tensors = {}
        for name, t in res.items():
            stack_cfg = stack_configs_by_name.get(name)
            table_tensors[name] = SparseCorePreprocessedInput(
                row_pointers=t["row_pointers"],
                embedding_ids=t["embedding_ids"],
                sample_ids=t["sample_ids"],
                gains=t["gains"],
                lengths=table_lengths_dict.get(name, {}),
                actual_num_ids=table_actual_num_ids_dict.get(name, {}),
                stats=table_stats[name],
                enable_minibatching=enable_minibatching,
                max_ids_per_partition=(
                    stack_cfg.max_ids_per_partition if stack_cfg is not None else None
                ),
                max_unique_ids_per_partition=(
                    stack_cfg.max_unique_ids_per_partition
                    if stack_cfg is not None
                    else None
                ),
            )

        self._last_stats = stats
        if self._fdo_client is not None:
            self._fdo_client.record(stats)
        return KeyedSparseCorePreprocessedInput(table_tensors, stats=stats)
