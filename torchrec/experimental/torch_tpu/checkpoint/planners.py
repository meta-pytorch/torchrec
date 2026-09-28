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

"""PyTorch DCP Save and Load Planners for PyTorch TPU Embedding."""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    default_planner,
    metadata as metadata_mod,
    planner as planner_mod,
)
from torchrec.experimental.torch_tpu.checkpoint import utils
from torchrec.experimental.torch_tpu.modules.embedding_configs import (
    StackedSparseCoreEmbeddingConfig,
)

__all__ = [
    "SparseCoreSavePlanner",
    "SparseCoreLoadPlanner",
]


def _extract_key_and_prefix(fqn: str) -> tuple[str, str]:
    """Extracts the table/stack key and module prefix from a full parameter name."""
    for prefix in ("embedding_bags", "embeddings"):
        marker = f"{prefix}."
        if marker in fqn and fqn.endswith(".weight"):
            key = fqn.split(marker, 1)[1].rsplit(".weight", 1)[0]
            return key, prefix

    for prefix in ("accumulators", "momentums", "velocities"):
        marker = f"{prefix}."
        if marker in fqn:
            key = fqn.split(marker, 1)[1]
            return key, prefix

    return fqn, ""


def _is_embedding_param(fqn: str) -> bool:
    """Checks if a parameter name corresponds to an embedding table or optimizer state."""
    _, prefix = _extract_key_and_prefix(fqn)
    return bool(prefix)


class SparseCoreSavePlanner(default_planner.DefaultSavePlanner):
    """Custom PyTorch DCP SavePlanner that injects SparseCore topology and table stacking metadata into DCP global metadata.

    Enables cross-topology loading and CPU inference unsharding by preserving
    original sharding parameters (world_size, num_sc_per_device, stacked_configs,
    etc.) in `metadata.planner_data["sparsecore"]`.
    """

    def __init__(
        self,
        num_sc_per_device: int = 2,
        stacked_configs: Optional[
            Union[Sequence[StackedSparseCoreEmbeddingConfig], dict[str, Any]]
        ] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_sc_per_device = num_sc_per_device
        self.stacked_configs = stacked_configs

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        storage_meta: Optional[Any] = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.stacked_configs is None:
            for obj in state_dict.values():
                if hasattr(obj, "_stacked_configs"):
                    self.stacked_configs = obj._stacked_configs
                    break
        super().set_up_planner(
            state_dict, storage_meta=storage_meta, is_coordinator=is_coordinator
        )

    def create_global_plan(
        self, all_plans: list[planner_mod.SavePlan]
    ) -> tuple[list[planner_mod.SavePlan], metadata_mod.Metadata]:
        global_plan, metadata = super().create_global_plan(all_plans)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        sparsecore_metadata: dict[str, Any] = {
            "world_size": world_size,
            "num_sc_per_device": self.num_sc_per_device,
        }
        if self.stacked_configs is not None:
            sparsecore_metadata["stacked_configs"] = utils.serialize_stacked_configs(
                self.stacked_configs
            )
        loaded_planner_data = metadata.planner_data or {}
        if isinstance(loaded_planner_data, dict):
            loaded_planner_data["sparsecore"] = sparsecore_metadata
            metadata = dataclasses.replace(metadata, planner_data=loaded_planner_data)
        else:
            logging.warning(
                "planner_data is not a dict, skipping metadata injection."
            )
        return global_plan, metadata


class SparseCoreLoadPlanner(default_planner.DefaultLoadPlanner):
    """Custom PyTorch DCP LoadPlanner that supports cross-topology loading and CPU inference unsharding.

    - Cross-Topology: Allows loading checkpoints onto topologies with different
      number of ranks or SparseCores. Transparently un-MOD-shards (and unstacks)
      using original topology and re-shards/stacks for target topology.
    - CPU Inference (unshard_for_cpu=True): Allows loading MOD-sharded (and
      stacked) checkpoints directly onto sequential CPU buffers for inference
      with standard TorchRec modules.
    """

    def __init__(
        self,
        num_sc_per_device: int = 2,
        unshard_for_cpu: bool = False,
        stacked_configs: Optional[
            Union[Sequence[StackedSparseCoreEmbeddingConfig], dict[str, Any]]
        ] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_sc_per_device = num_sc_per_device
        self.unshard_for_cpu = unshard_for_cpu
        self.stacked_configs = stacked_configs
        self.old_topology: Optional[dict[str, Any]] = None
        self.old_stacked_configs: dict[str, StackedSparseCoreEmbeddingConfig] = {}
        self.cross_topology = False
        self._cpu_buffers: dict[str, torch.Tensor] = {}
        self._target_tensors: dict[str, Any] = {}
        self._target_table_mapping: dict[str, list[tuple[str, str, Any]]] = {}
        self._metadata: Optional[Any] = None
        self._loaded_chunks: dict[str, int] = {}
        self._expected_local_chunks: dict[str, int] = {}

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Optional[Any] = None,
        is_coordinator: bool = False,
    ) -> None:
        self._metadata = metadata
        self._loaded_chunks = {}
        self._expected_local_chunks = {}
        self._cpu_buffers = {}
        self._target_tensors = {}
        self._target_table_mapping = {}
        self.old_stacked_configs = {}

        if self.stacked_configs is None:
            for obj in state_dict.values():
                if hasattr(obj, "_stacked_configs"):
                    self.stacked_configs = obj._stacked_configs
                    break

        if metadata and metadata.planner_data:
            if isinstance(metadata.planner_data, dict):
                self.old_topology = metadata.planner_data.get("sparsecore", None)
                if self.old_topology:
                    logging.info("Read native SparseCore metadata: %s", self.old_topology)
                    if "stacked_configs" in self.old_topology:
                        stacked_meta = self.old_topology["stacked_configs"]
                        # Handle both list and dict formats for stacked configs.
                        if isinstance(stacked_meta, dict):
                            for _, sc_data in stacked_meta.items():
                                if isinstance(sc_data, dict):
                                    sc = StackedSparseCoreEmbeddingConfig.from_dict(sc_data)
                                    self.old_stacked_configs[sc.stack_name] = sc
                        elif isinstance(stacked_meta, list):
                            for sc_data in stacked_meta:
                                if isinstance(sc_data, dict):
                                    sc = StackedSparseCoreEmbeddingConfig.from_dict(sc_data)
                                    self.old_stacked_configs[sc.stack_name] = sc
            else:
                logging.warning(
                    "planner_data is not a dict, cannot read sparsecore metadata."
                )

        # If stacked_configs is not None, use it instead of metadata.
        if self.stacked_configs:
            configs = (
                self.stacked_configs.values()
                if isinstance(self.stacked_configs, dict)
                else self.stacked_configs
            )
            for sc in configs:
                if isinstance(sc, dict):
                    sc = StackedSparseCoreEmbeddingConfig.from_dict(sc)
                if sc.stack_name not in self.old_stacked_configs:
                    self.old_stacked_configs[sc.stack_name] = sc

        if self.unshard_for_cpu:
            modified_sd = dict(state_dict)
            if metadata and hasattr(metadata, "state_dict_metadata"):
                for fqn, meta in metadata.state_dict_metadata.items():
                    if not _is_embedding_param(fqn) or not fqn.endswith(".weight"):
                        continue

                    key, prefix = _extract_key_and_prefix(fqn)

                    if key in self.old_stacked_configs:
                        # Build mapping from table name to stack config.
                        stack_config = self.old_stacked_configs[key]
                        matching_tables = []
                        for t in stack_config.tables:
                            # Build candidate keys for matching tables in state_dict.
                            prefix_with_dot = (
                                f"{prefix}."
                                if prefix and not prefix.endswith(".")
                                else prefix
                            )
                            full_prefix = (
                                fqn[: -len(f"{key}.weight")]
                                if fqn.endswith(f"{key}.weight")
                                else prefix_with_dot
                            )
                            candidate_keys = [
                                f"{full_prefix}{t.name}.weight",
                                f"{prefix_with_dot}{t.name}.weight",
                                f"{t.name}.weight",
                                t.name,
                            ]
                            for ck in candidate_keys:
                                if ck in state_dict:
                                    matching_tables.append((t.name, ck, state_dict[ck]))
                                    break

                        # Allocate buffers and create mappings for matching tables.
                        if matching_tables:
                            self._target_table_mapping[fqn] = matching_tables
                            for _, ck, _ in matching_tables:
                                if ck in modified_sd:
                                    del modified_sd[ck]
                            cpu_tensor = torch.empty(
                                meta.size,
                                dtype=meta.properties.dtype,
                                layout=meta.properties.layout,
                                device="cpu",
                            )
                            self._cpu_buffers[fqn] = cpu_tensor
                            modified_sd[fqn] = cpu_tensor
                        elif fqn in state_dict:
                            self._target_tensors[fqn] = state_dict[fqn]
                            cpu_tensor = torch.empty(
                                meta.size,
                                dtype=meta.properties.dtype,
                                layout=meta.properties.layout,
                                device="cpu",
                            )
                            self._cpu_buffers[fqn] = cpu_tensor
                            modified_sd[fqn] = cpu_tensor
                    elif fqn in state_dict:
                        # Fallback if no stacked config is found.
                        self._target_tensors[fqn] = state_dict[fqn]
                        target_obj = state_dict[fqn]
                        if target_obj.shape != meta.size:
                            cpu_tensor = torch.empty(
                                meta.size,
                                dtype=meta.properties.dtype,
                                layout=meta.properties.layout,
                                device="cpu",
                            )
                            self._cpu_buffers[fqn] = cpu_tensor
                            modified_sd[fqn] = cpu_tensor

            super().set_up_planner(modified_sd, metadata, is_coordinator)
            return

        self.cross_topology = False
        if self.old_topology:
            try:
                curr_world_size = dist.get_world_size() if dist.is_initialized() else 1
                curr_num_sc = self.num_sc_per_device
                if (
                    self.old_topology["world_size"] != curr_world_size
                    or self.old_topology["num_sc_per_device"] != curr_num_sc
                ):
                    self.cross_topology = True
                    logging.info(
                        "SparseCoreLoadPlanner: Topology mismatch detected! Old:"
                        " world_size=%d, num_sc=%d; Current: world_size=%d, num_sc=%d",
                        self.old_topology["world_size"],
                        self.old_topology["num_sc_per_device"],
                        curr_world_size,
                        curr_num_sc,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Failed to check topology compatibility: %s", e)

        if (
            self.cross_topology
            and metadata
            and hasattr(metadata, "state_dict_metadata")
        ):
            modified_sd = dict(state_dict)
            for fqn, obj in state_dict.items():
                if _is_embedding_param(fqn) and fqn in metadata.state_dict_metadata:
                    meta = metadata.state_dict_metadata[fqn]
                    self._target_tensors[fqn] = obj
                    cpu_tensor = torch.empty(
                        meta.size,
                        dtype=meta.properties.dtype,
                        layout=meta.properties.layout,
                        device="cpu",
                    )
                    self._cpu_buffers[fqn] = cpu_tensor
                    modified_sd[fqn] = cpu_tensor
            super().set_up_planner(modified_sd, metadata, is_coordinator)
        else:
            super().set_up_planner(state_dict, metadata, is_coordinator)

    def create_local_plan(self) -> planner_mod.LoadPlan:
        """Overrides LoadPlan to track expected local chunks per FQN."""
        plan = super().create_local_plan()
        self._expected_local_chunks = {}
        for item in plan.items:
            fqn = item.dest_index.fqn
            self._expected_local_chunks[fqn] = (
                self._expected_local_chunks.get(fqn, 0) + 1
            )
        return plan

    def resolve_tensor(self, read_item: planner_mod.ReadItem) -> torch.Tensor:
        fqn = read_item.dest_index.fqn
        if (
            self.cross_topology or self.unshard_for_cpu
        ) and fqn in self._cpu_buffers:
            return self.transform_tensor(read_item, self._cpu_buffers[fqn])

        return super().resolve_tensor(read_item)

    def commit_tensor(
        self, read_item: planner_mod.ReadItem, tensor: torch.Tensor
    ) -> None:
        super().commit_tensor(read_item, tensor)
        fqn = read_item.dest_index.fqn

        if not _is_embedding_param(fqn):
            return

        # Only process if post-processing is needed
        if not (
            self.unshard_for_cpu
            or (fqn in self._cpu_buffers and self.cross_topology)
        ):
            return

        self._loaded_chunks[fqn] = self._loaded_chunks.get(fqn, 0) + 1

        expected_chunks = self._expected_local_chunks.get(fqn, 1)
        if self._loaded_chunks[fqn] < expected_chunks:
            return

        # Common extraction for both paths
        assert self.old_topology is not None
        old_world_size = self.old_topology.get("world_size", 1)
        old_num_sc = self.old_topology.get("num_sc_per_device", 2)
        old_num_shards = old_world_size * old_num_sc

        assert self._metadata is not None
        assert fqn in self._metadata.state_dict_metadata
        md = self._metadata.state_dict_metadata[fqn]
        vocab_size = md.size[0]
        embedding_dim = md.size[1]

        key, _ = _extract_key_and_prefix(fqn)
        old_stacked_configs = (
            self.old_stacked_configs
            if self.old_stacked_configs
            else (
                self.old_topology.get("stacked_configs", {})
                if self.old_topology
                else {}
            )
        )

        table_to_stack_name: dict[str, str] = {}
        for s_name, s_cfg in old_stacked_configs.items():
            cfg_tables = (
                s_cfg.get("tables", [])
                if isinstance(s_cfg, dict)
                else getattr(s_cfg, "tables", [])
            )
            for t in cfg_tables:
                t_name = t.get("name") if isinstance(t, dict) else t.name
                if t_name is not None:
                    table_to_stack_name[t_name] = s_name

        if self.unshard_for_cpu:
            # CPU Inference Path: In-place unshard the loaded buffer
            target = (
                self._cpu_buffers[fqn]
                if fqn in self._cpu_buffers
                else self.lookup_tensor(read_item.dest_index)
            )
            dest = (
                self._target_tensors[fqn] if fqn in self._target_tensors else target
            )

            if key in old_stacked_configs:
                stack_config = old_stacked_configs[key]
                unstacked_dict = utils.unstack_and_unshard_global_tensor(
                    target,
                    stack_config,
                    num_shards=old_num_shards,
                )
                if fqn in self._target_table_mapping:
                    for t_name, _, target_tensor in self._target_table_mapping[fqn]:
                        if t_name in unstacked_dict:
                            t_w = unstacked_dict[t_name]
                            target_tensor.data[: t_w.size(0), : t_w.size(1)].copy_(t_w)
                else:
                    # Target is stacked on CPU: re-pack as 1-shard sequential stacked tensor
                    cpu_stacked = utils.stack_and_shard_global_tensor(
                        unstacked_dict,
                        stack_config,
                        num_shards=1,
                    )
                    dest.data[: cpu_stacked.size(0), : cpu_stacked.size(1)].copy_(
                        cpu_stacked
                    )
            elif key in table_to_stack_name:
                s_name = table_to_stack_name[key]
                stack_config = old_stacked_configs[s_name]
                unstacked_dict = utils.unstack_and_unshard_global_tensor(
                    target,
                    stack_config,
                    num_shards=old_num_shards,
                )
                if key in unstacked_dict:
                    t_w = unstacked_dict[key]
                    dest.data[: t_w.size(0), : t_w.size(1)].copy_(t_w)
            else:
                unsharded = utils.reverse_mod_shard(
                    target,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    num_shards=old_num_shards,
                )
                rows = min(dest.size(0), unsharded.size(0))
                cols = min(dest.size(1), unsharded.size(1))
                dest.data[:rows, :cols].copy_(unsharded[:rows, :cols])

            if fqn in self._cpu_buffers:
                del self._cpu_buffers[fqn]
            return

        # Cross Topology Path
        full_cpu_tensor = self._cpu_buffers[fqn]
        target_param = self._target_tensors[fqn]

        # 1. Unshard and unstack from old topology
        if key in old_stacked_configs:
            unstacked_tables = utils.unstack_and_unshard_global_tensor(
                full_cpu_tensor,
                old_stacked_configs[key],
                num_shards=old_num_shards,
            )
        elif key in table_to_stack_name:
            s_name = table_to_stack_name[key]
            unstacked_tables = utils.unstack_and_unshard_global_tensor(
                full_cpu_tensor,
                old_stacked_configs[s_name],
                num_shards=old_num_shards,
            )
        else:
            sequential_cpu = utils.reverse_mod_shard(
                full_cpu_tensor,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_shards=old_num_shards,
            )
            unstacked_tables = {key: sequential_cpu}

        # 2. Reshard and stack for target topology
        curr_world_size = dist.get_world_size() if dist.is_initialized() else 1
        curr_num_sc = self.num_sc_per_device
        curr_num_shards = curr_world_size * curr_num_sc

        target_stack_config = None
        if self.stacked_configs is not None:
            norm_stacked = utils.serialize_stacked_configs(self.stacked_configs)
            target_stack_config = norm_stacked.get(key, None)
        if target_stack_config is None and key in old_stacked_configs:
            target_stack_config = old_stacked_configs[key]

        if target_stack_config is not None:
            target_sharded_cpu = utils.stack_and_shard_global_tensor(
                unstacked_tables,
                target_stack_config,
                num_shards=curr_num_shards,
            )
        else:
            if key not in unstacked_tables:
                raise KeyError(
                    f"Parameter '{key}' not found in unstacked tables {list(unstacked_tables.keys())}."
                )
            target_padded_vocab = (
                target_param.size(0)
                if hasattr(target_param, "to_local")
                else target_param.size(0) * curr_world_size
            )
            t_w = unstacked_tables[key]
            target_padded_cpu = torch.zeros(
                (target_padded_vocab, t_w.size(1)), dtype=full_cpu_tensor.dtype
            )
            copy_limit = min(t_w.size(0), target_padded_vocab)
            target_padded_cpu[:copy_limit, :] = t_w[:copy_limit, :]
            target_sharded_cpu = utils.mod_shard(
                target_padded_cpu,
                num_shards=curr_num_shards,
            )

        # 3. Extract local slice and copy to TPU
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rows = (
            target_param.to_local().size(0)
            if hasattr(target_param, "to_local")
            else target_param.size(0)
        )
        local_slice = target_sharded_cpu[
            rank * local_rows : (rank + 1) * local_rows
        ]

        logging.info("SparseCoreLoadPlanner: Copying slice to TPU for %s", fqn)
        with torch.no_grad():
            if hasattr(target_param, "to_local"):
                target_param.to_local().copy_(local_slice.to(target_param.device))
            else:
                target_param.copy_(local_slice.to(target_param.device))

        logging.info("SparseCoreLoadPlanner: Finished processing %s", fqn)
        del self._cpu_buffers[fqn]
