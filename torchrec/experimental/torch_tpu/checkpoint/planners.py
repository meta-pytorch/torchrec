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
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    default_planner,
    metadata as metadata_mod,
    planner as planner_mod,
)
from torchrec.experimental.torch_tpu.checkpoint import utils

__all__ = [
    "SparseCoreSavePlanner",
    "SparseCoreLoadPlanner",
]


class SparseCoreSavePlanner(default_planner.DefaultSavePlanner):
    """Custom PyTorch DCP SavePlanner that injects SparseCore topology metadata into DCP global metadata.

    Enables cross-topology loading and CPU inference unsharding by preserving
    original sharding parameters (world_size, num_sc_per_device, etc.) in
    `metadata.planner_data["sparsecore"]`.
    """

    def __init__(
        self,
        num_sc_per_device: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_sc_per_device = num_sc_per_device

    def create_global_plan(
        self, all_plans: list[planner_mod.SavePlan]
    ) -> tuple[list[planner_mod.SavePlan], metadata_mod.Metadata]:
        global_plan, metadata = super().create_global_plan(all_plans)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        sparsecore_metadata = {
            "world_size": world_size,
            "num_sc_per_device": self.num_sc_per_device,
        }
        loaded_planner_data = metadata.planner_data or {}
        if isinstance(loaded_planner_data, dict):
            loaded_planner_data["sparsecore"] = sparsecore_metadata
            metadata = dataclasses.replace(metadata, planner_data=loaded_planner_data)
        else:
            logging.warning("planner_data is not a dict, skipping metadata injection.")
        return global_plan, metadata


class SparseCoreLoadPlanner(default_planner.DefaultLoadPlanner):
    """Custom PyTorch DCP LoadPlanner that supports cross-topology loading and CPU inference unsharding.

    - Cross-Topology: Allows loading checkpoints onto topologies with different
      number of ranks or SparseCores. Transparently un-MOD-shards using original
      topology and re-shards for target topology.
    - CPU Inference (unshard_for_cpu=True): Allows loading MOD-sharded checkpoints
      directly onto sequential CPU buffers for inference with standard TorchRec
      modules.
    """

    def __init__(
        self,
        num_sc_per_device: int = 2,
        unshard_for_cpu: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_sc_per_device = num_sc_per_device
        self.unshard_for_cpu = unshard_for_cpu
        self.old_topology: Optional[dict[str, Any]] = None
        self.cross_topology = False
        self._cpu_buffers: dict[str, torch.Tensor] = {}
        self._metadata: Optional[Any] = None
        self._loaded_chunks: dict[str, int] = {}

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Optional[Any] = None,
        is_coordinator: bool = False,
    ) -> None:
        self._metadata = metadata
        self._loaded_chunks = {}
        if metadata and metadata.planner_data:
            if isinstance(metadata.planner_data, dict):
                self.old_topology = metadata.planner_data.get("sparsecore", None)
                if self.old_topology:
                    logging.info(
                        "Read native SparseCore metadata: %s", self.old_topology
                    )
            else:
                logging.warning(
                    "planner_data is not a dict, cannot read sparsecore metadata."
                )

        if self.unshard_for_cpu:
            super().set_up_planner(state_dict, metadata, is_coordinator)
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

        self._target_tensors: dict[str, Any] = {}
        if (
            self.cross_topology
            and metadata
            and hasattr(metadata, "state_dict_metadata")
        ):
            modified_sd = dict(state_dict)
            for fqn, obj in state_dict.items():
                is_embedding_weight = (
                    "embedding_bags." in fqn or "embeddings." in fqn
                ) and fqn.endswith(".weight")

                is_embedding_state = any(
                    x in fqn
                    for x in [
                        "accumulators.",
                        "momentums.",
                        "velocities.",
                    ]
                )
                if (
                    is_embedding_weight or is_embedding_state
                ) and fqn in metadata.state_dict_metadata:
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

    def resolve_tensor(self, read_item: planner_mod.ReadItem) -> torch.Tensor:
        fqn = read_item.dest_index.fqn
        if (self.cross_topology or self.unshard_for_cpu) and fqn in self._cpu_buffers:
            return self.transform_tensor(read_item, self._cpu_buffers[fqn])

        return super().resolve_tensor(read_item)

    def commit_tensor(
        self, read_item: planner_mod.ReadItem, tensor: torch.Tensor
    ) -> None:
        super().commit_tensor(read_item, tensor)
        fqn = read_item.dest_index.fqn

        is_embedding_weight = (
            "embedding_bags." in fqn or "embeddings." in fqn
        ) and fqn.endswith(".weight")

        is_embedding_state = any(
            x in fqn
            for x in [
                "accumulators.",
                "momentums.",
                "velocities.",
            ]
        )
        is_embedding = is_embedding_weight or is_embedding_state

        if not is_embedding:
            return

        # Only process if post-processing is needed
        if not (
            self.unshard_for_cpu or (fqn in self._cpu_buffers and self.cross_topology)
        ):
            return

        self._loaded_chunks[fqn] = self._loaded_chunks.get(fqn, 0) + 1

        metadata = self._metadata
        total_chunks = 1
        if metadata is not None and fqn in metadata.state_dict_metadata:
            md = metadata.state_dict_metadata[fqn]
            if hasattr(md, "chunks"):
                total_chunks = len(md.chunks)

        if self._loaded_chunks[fqn] < total_chunks:
            return

        # Common extraction for both paths
        assert self.old_topology is not None
        old_world_size = self.old_topology.get("world_size", 1)
        old_num_sc = self.old_topology.get("num_sc_per_device", 2)
        old_num_shards = old_world_size * old_num_sc

        assert metadata is not None
        assert fqn in metadata.state_dict_metadata
        md = metadata.state_dict_metadata[fqn]
        vocab_size = md.size[0]
        embedding_dim = md.size[1]

        if self.unshard_for_cpu:
            # CPU Inference Path: In-place unshard the loaded buffer
            target = self.lookup_tensor(read_item.dest_index)

            unsharded = utils.reverse_mod_shard(
                target,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_shards=old_num_shards,
            )
            # Copy into target, ensuring size match by slicing target to vocab_size
            target.data[:vocab_size, :].copy_(unsharded)
            return

        # Cross Topology Path
        full_cpu_tensor = self._cpu_buffers[fqn]
        target_param = self._target_tensors[fqn]

        # 1. Unshard from old topology
        sequential_cpu = utils.reverse_mod_shard(
            full_cpu_tensor,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_shards=old_num_shards,
        )

        # 2. Reshard for current topology
        curr_world_size = dist.get_world_size() if dist.is_initialized() else 1
        curr_num_sc = self.num_sc_per_device
        curr_num_shards = curr_world_size * curr_num_sc

        target_padded_vocab = target_param.size(0)

        target_padded_cpu = torch.zeros(
            (target_padded_vocab, embedding_dim), dtype=full_cpu_tensor.dtype
        )
        # Handle potential padding in sequential_cpu
        copy_limit = min(sequential_cpu.size(0), target_padded_vocab)
        target_padded_cpu[:copy_limit, :] = sequential_cpu[:copy_limit, :]

        # Apply target MOD sharding
        target_sharded_cpu = utils.mod_shard(
            target_padded_cpu,
            num_shards=curr_num_shards,
        )

        # Extract local slice and copy to TPU
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rows = target_padded_vocab // curr_world_size
        local_slice = target_sharded_cpu[rank * local_rows : (rank + 1) * local_rows]

        logging.info("SparseCoreLoadPlanner: Copying slice to TPU for %s", fqn)
        with torch.no_grad():
            if hasattr(target_param, "to_local"):
                target_param.to_local().copy_(local_slice.to(target_param.device))
            else:
                target_param.copy_(local_slice.to(target_param.device))

        logging.info("SparseCoreLoadPlanner: Finished processing %s", fqn)
        del self._cpu_buffers[fqn]
