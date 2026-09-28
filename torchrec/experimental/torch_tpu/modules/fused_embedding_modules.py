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

"""Custom Embedding Collections using SparseCore on TPU."""

import abc
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.distributed.tensor as dt
from torch import nn
from torchrec.experimental.torch_tpu.datasets import input_preprocessing
from torchrec.experimental.torch_tpu.modules import custom_ops, table_stacking
from torchrec.experimental.torch_tpu.modules.embedding_configs import (
    SparseCoreEmbeddingConfig,
    StackedSparseCoreEmbeddingConfig,
)

# import torch_tpu._internal.device_utils.annotations as tpu_annotations
from torchrec.modules import embedding_configs
from torchrec.optim import fused, keyed
from torchrec.sparse import jagged_tensor

KeyedOptimizer = keyed.KeyedOptimizer
FusedOptimizer = fused.FusedOptimizer
FusedOptimizerModule = fused.FusedOptimizerModule

KeyedSparseCorePreprocessedInput = input_preprocessing.KeyedSparseCorePreprocessedInput
EmbeddingConfig = embedding_configs.EmbeddingConfig
KeyedTensor = jagged_tensor.KeyedTensor
JaggedTensor = jagged_tensor.JaggedTensor
EmbeddingBagConfig = embedding_configs.EmbeddingBagConfig

_DISPATCH_TABLE = {
    torch.optim.SGD: custom_ops.sparse_dense_matmul_sgd_fwd,
    torch.optim.Adagrad: custom_ops.sparse_dense_matmul_adagrad_fwd,
    torch.optim.Adam: custom_ops.sparse_dense_matmul_adam_fwd,
}

_DISPATCH_TABLE_BWD = {
    torch.optim.SGD: custom_ops.sparse_dense_matmul_sgd_bwd,
    torch.optim.Adagrad: custom_ops.sparse_dense_matmul_adagrad_bwd,
    torch.optim.Adam: custom_ops.sparse_dense_matmul_adam_bwd,
}


class SparseCoreFusedOptimizer(FusedOptimizer):
    """FusedOptimizer replacement using TPU SparseCore SparseDenseMatmul."""

    def __init__(self, emb_module) -> None:
        self._emb_module = emb_module
        params = {}
        state = {}
        for stack_config in emb_module._stacked_configs:
            name = stack_config.stack_name

            if (
                hasattr(emb_module, "embedding_bags")
                and name in emb_module.embedding_bags
            ):
                weight = emb_module.embedding_bags[name].weight
            elif hasattr(emb_module, "embeddings") and name in emb_module.embeddings:
                weight = emb_module.embeddings[name].weight
            elif (
                hasattr(emb_module, "embedding_tables")
                and name in emb_module.embedding_tables
            ):
                weight = emb_module.embedding_tables[name]
            else:
                continue
            param_key = f"{name}.weight"
            params[param_key] = weight
            state[weight] = {}
            # Adagrad state
            if name in emb_module.accumulators:
                state[weight][f"{param_key}.accumulator"] = emb_module.accumulators[
                    name
                ]
            # Adam states
            if name in emb_module.momentums:
                state[weight][f"{param_key}.momentum"] = emb_module.momentums[name]
            if name in emb_module.velocities:
                state[weight][f"{param_key}.velocity"] = emb_module.velocities[name]
        initial_lr = emb_module.get_initial_lr()
        super().__init__(
            params, state, [{"params": list(params.values()), "lr": initial_lr}]
        )
        self._last_synced_lr: Optional[float] = initial_lr

    def _sync_lr(self) -> None:
        param_groups = list(self.param_groups)
        if param_groups:
            new_lr = float(param_groups[0]["lr"])
            if self._last_synced_lr is None or new_lr != self._last_synced_lr:
                self._emb_module.set_learning_rate(new_lr)
                self._last_synced_lr = new_lr

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._sync_lr()

    def step(self, closure: Any = None) -> None:
        self._sync_lr()


def run_sparse_dense_matmul(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    learning_rate: torch.Tensor,
    batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    optimizer_type: type[torch.optim.Optimizer],
    table_name: str,
    accumulator: Optional[torch.Tensor] = None,
    momentum: Optional[torch.Tensor] = None,
    velocity: Optional[torch.Tensor] = None,
    epsilon: float = 1e-10,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.Tensor:
    """Runs SparseCore SparseDenseMatmul with the given optimizer.

    Args:
      row_pointers: Row pointers tensor for CSR layout.
      embedding_ids: Embedding table indices.
      sample_ids: Batch sample IDs for each ID.
      gains: Feature gains tensor.
      embedding_table: Local table parameter tensor.
      learning_rate: Tensor learning rate.
      batch_size: Device batch size.
      max_ids_per_partition: Static partition size for IDs.
      max_unique_ids_per_partition: Static partition size for unique IDs.
      optimizer_type: Optimizer class (SGD, Adagrad, Adam).
      table_name: Name of the table / stack for debugging and custom op naming.
      accumulator: Optional Adagrad accumulator tensor.
      momentum: Optional Adam momentum tensor.
      velocity: Optional Adam velocity tensor.
      epsilon: Numerical stability constant for Adagrad / Adam.
      beta1: Adam exponential decay rate for first moment estimates.
      beta2: Adam exponential decay rate for second moment estimates.

    Returns:
      Output activation tensor from SparseDenseMatmul lookup.

    Raises:
      ValueError: If optimizer_type is unsupported or required optimizer state
        tensors are None.
    """
    if optimizer_type not in _DISPATCH_TABLE:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. Only 'sgd', 'adagrad', and"
            " 'adam' are supported."
        )
    op = _DISPATCH_TABLE[optimizer_type]
    if optimizer_type == torch.optim.SGD:
        return op(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table,
            learning_rate,
            batch_size,
            max_ids_per_partition,
            max_unique_ids_per_partition,
            table_name,
        )
    elif optimizer_type == torch.optim.Adagrad:
        if accumulator is None:
            raise ValueError("accumulator must be provided for Adagrad")
        return op(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table,
            accumulator,
            learning_rate,
            epsilon,
            batch_size,
            max_ids_per_partition,
            max_unique_ids_per_partition,
            table_name,
        )
    elif optimizer_type == torch.optim.Adam:
        if momentum is None:
            raise ValueError("momentum must be provided for Adam")
        if velocity is None:
            raise ValueError("velocity must be provided for Adam")
        return op(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table,
            momentum,
            velocity,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            batch_size,
            max_ids_per_partition,
            max_unique_ids_per_partition,
            table_name,
        )


def run_sparse_dense_matmul_bwd(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    learning_rate: torch.Tensor,
    batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    optimizer_type: type[torch.optim.Optimizer],
    table_name: str,
    accumulator: Optional[torch.Tensor] = None,
    momentum: Optional[torch.Tensor] = None,
    velocity: Optional[torch.Tensor] = None,
    epsilon: float = 1e-10,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> None:
    """Runs SparseCore backward pass and in-place optimizer update."""
    if optimizer_type not in _DISPATCH_TABLE_BWD:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. Only 'sgd', 'adagrad', and"
            " 'adam' are supported."
        )
    op = _DISPATCH_TABLE_BWD[optimizer_type]
    if optimizer_type == torch.optim.SGD:
        op(
            grad_output,
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table,
            learning_rate,
            batch_size,
            max_ids_per_partition,
            max_unique_ids_per_partition,
            table_name,
        )
    elif optimizer_type == torch.optim.Adagrad:
        if accumulator is None:
            raise ValueError("accumulator must be provided for Adagrad")
        op(
            grad_output,
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table,
            accumulator,
            learning_rate,
            epsilon,
            batch_size,
            max_ids_per_partition,
            max_unique_ids_per_partition,
            table_name,
        )
    elif optimizer_type == torch.optim.Adam:
        if momentum is None:
            raise ValueError("momentum must be provided for Adam")
        if velocity is None:
            raise ValueError("velocity must be provided for Adam")
        op(
            grad_output,
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table,
            momentum,
            velocity,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            batch_size,
            max_ids_per_partition,
            max_unique_ids_per_partition,
            table_name,
        )


class SparseCoreEmbeddingBagCollectionInterface(abc.ABC, nn.Module):
    """Interface for `SparseCoreEmbeddingBagCollection`."""

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedSparseCorePreprocessedInput,
    ) -> KeyedTensor:
        pass

    @abc.abstractmethod
    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        pass


class SparseCoreEmbeddingCollectionInterface(abc.ABC, nn.Module):
    """Interface for `SparseCoreEmbeddingCollection`."""

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedSparseCorePreprocessedInput,
    ) -> Dict[str, JaggedTensor]:
        pass

    @abc.abstractmethod
    def embedding_configs(
        self,
    ) -> List[EmbeddingConfig]:
        pass


class _SparseCoreFusedEmbeddingBase(FusedOptimizerModule):
    """Shared base for SparseCoreFused{EmbeddingBag,Embedding}Collection.

    Holds all common initialization (weights, optimizer states, learning rates,
    TPU layout, sync) and utility methods. Subclasses provide only `forward()`
    and their config accessor.
    """

    def __init__(
        self,
        tables: List[SparseCoreEmbeddingConfig],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        weight_dict_name: str,
        batch_size: int = 1,
        global_device_count: int = 1,
        num_sc_per_device: int = 2,
        auto_stack: bool = False,
        activation_mem_bytes_limit: int = 6 * 1024 * 1024,
        use_custom_embedding_formatting: bool = True,
    ) -> None:
        super().__init__()
        self._tables = tables
        self._optimizer_type = optimizer_type
        self._use_custom_embedding_formatting = use_custom_embedding_formatting

        # Normalize optimizer kwargs to canonical keys (lr, eps, beta1, beta2).
        optimizer_kwargs = (
            dict(optimizer_kwargs) if optimizer_kwargs is not None else {}
        )
        if "learning_rate" in optimizer_kwargs:
            optimizer_kwargs["lr"] = optimizer_kwargs.pop("learning_rate")

        if "epsilon" in optimizer_kwargs:
            optimizer_kwargs["eps"] = optimizer_kwargs.pop("epsilon")

        if "betas" in optimizer_kwargs:
            optimizer_kwargs["beta1"], optimizer_kwargs["beta2"] = (
                optimizer_kwargs.pop("betas")
            )

        self._optimizer_kwargs = optimizer_kwargs

        self._global_device_count = global_device_count
        self._num_sc_per_device = num_sc_per_device
        if batch_size % num_sc_per_device != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_sc_per_device"
                f" ({num_sc_per_device})."
            )
        self._batch_size = batch_size
        self._weight_dict_name = weight_dict_name

        self._device = torch.device("tpu")
        self._device_mesh = None
        if dist.is_initialized() and self._global_device_count > 1:
            self._device_mesh = dt.init_device_mesh(
                "tpu", (self._global_device_count,)
            )

        self.learning_rates = nn.ParameterDict()

        if auto_stack:
            table_stacking.auto_stack_tables(
                tables,
                global_device_count=global_device_count,
                num_sc_per_device=num_sc_per_device,
                batch_size=batch_size,
                activation_mem_bytes_limit=activation_mem_bytes_limit,
            )
        self._stacked_configs = table_stacking.prepare_tables_for_stacking(
            tables,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
        )

        for stack_config in self._stacked_configs:
            self.learning_rates[stack_config.stack_name] = nn.Parameter(
                torch.tensor(
                    self._optimizer_kwargs.get("lr", 0.01),
                    dtype=torch.float32,
                    device=self._device,
                ),
                requires_grad=False,
            )

        weight_modules = nn.ModuleDict()
        setattr(self, weight_dict_name, weight_modules)

        self.accumulators = nn.ParameterDict()
        self.momentums = nn.ParameterDict()
        self.velocities = nn.ParameterDict()
        self._table_configs: Dict[str, SparseCoreEmbeddingConfig] = {}
        self._feature_to_table_config: Dict[str, SparseCoreEmbeddingConfig] = {}
        self._feature_to_stack_config: Dict[
            str, StackedSparseCoreEmbeddingConfig
        ] = {}
        self._stack_unstack_metadata: Dict[
            str, tuple[list[tuple[str, SparseCoreEmbeddingConfig]], list[int]]
        ] = {}

        for stack_config in self._stacked_configs:
            table_feature_list = []
            per_feature_dims = []
            for config in stack_config.tables:
                self._table_configs[config.name] = config
                for feature_name in config.feature_names:
                    self._feature_to_table_config[feature_name] = config
                    self._feature_to_stack_config[feature_name] = stack_config
                    table_feature_list.append((feature_name, config))
                    per_feature_dims.append(config.embedding_dim)

            self._stack_unstack_metadata[stack_config.stack_name] = (
                table_feature_list,
                per_feature_dims,
            )

            num_embeddings = stack_config.stack_num_embeddings
            embedding_dim = stack_config.stack_embedding_dim

            stack_shape = (num_embeddings // self._global_device_count, embedding_dim)

            table_weights = {}
            for config in stack_config.tables:
                local_vocab = (
                    stack_config.padded_vocab_sizes[config.name]
                    // self._global_device_count
                )
                local_dim = config.embedding_dim
                t_w = torch.empty((local_vocab, local_dim), dtype=torch.float32)
                if config.init_fn:
                    config.init_fn(t_w)
                else:
                    nn.init.uniform_(t_w, -0.01, 0.01)
                table_weights[config.name] = t_w

            sharded_stacked = table_stacking.stack_and_shard_tables(
                table_weights,
                [stack_config],
                global_device_count=self._global_device_count,
                num_sc_per_device=self._num_sc_per_device,
            )[stack_config.stack_name]

            embedding_table_device = sharded_stacked.to(self._device)

            if self._device_mesh is not None:
                sharded_weight = dt.DTensor.from_local(
                    embedding_table_device,
                    device_mesh=self._device_mesh,
                    placements=[dt.Shard(0)],
                )
                weight_param = nn.Parameter(sharded_weight)
            else:
                weight_param = nn.Parameter(embedding_table_device)

            weight_param._in_backward_optimizers = [self._optimizer_type]

            weight_modules[stack_config.stack_name] = torch.nn.Module()
            weight_modules[stack_config.stack_name].register_parameter(
                "weight", weight_param
            )

            if self._optimizer_type == torch.optim.SGD:
                pass
            elif self._optimizer_type == torch.optim.Adagrad:
                accumulator = torch.full(
                    stack_shape,
                    self._optimizer_kwargs.get("initial_accumulator_value", 0.0),
                    dtype=torch.float32,
                    device=self._device,
                )
                if self._device_mesh is not None:
                    sharded_accumulator = dt.DTensor.from_local(
                        accumulator,
                        device_mesh=self._device_mesh,
                        placements=[dt.Shard(0)],
                    )
                    self.accumulators[stack_config.stack_name] = nn.Parameter(
                        sharded_accumulator
                    )
                else:
                    self.accumulators[stack_config.stack_name] = nn.Parameter(accumulator)
            elif self._optimizer_type == torch.optim.Adam:
                momentum = torch.zeros(
                    stack_shape, dtype=torch.float32, device=self._device
                )
                velocity = torch.zeros(
                    stack_shape, dtype=torch.float32, device=self._device
                )
                if self._device_mesh is not None:
                    sharded_momentum = dt.DTensor.from_local(
                        momentum,
                        device_mesh=self._device_mesh,
                        placements=[dt.Shard(0)],
                    )
                    sharded_velocity = dt.DTensor.from_local(
                        velocity,
                        device_mesh=self._device_mesh,
                        placements=[dt.Shard(0)],
                    )
                    self.momentums[stack_config.stack_name] = nn.Parameter(
                        sharded_momentum
                    )
                    self.velocities[stack_config.stack_name] = nn.Parameter(
                        sharded_velocity
                    )
                else:
                    self.momentums[stack_config.stack_name] = nn.Parameter(momentum)
                    self.velocities[stack_config.stack_name] = nn.Parameter(velocity)
            else:
                raise ValueError(
                    f"Unsupported optimizer: {self._optimizer_type}. Only 'sgd',"
                    " 'adagrad', and 'adam' are supported."
                )

        # Synchronize TPU devices to ensure all tables are initialized
        # before the first forward pass.
        if hasattr(torch, "accelerator") and self._device.type == "tpu":
            torch.accelerator.synchronize()

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return SparseCoreFusedOptimizer(self)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stacked_configs(self) -> List[StackedSparseCoreEmbeddingConfig]:
        return self._stacked_configs

    def optimizer_type(self) -> Type[torch.optim.Optimizer]:
        return self._optimizer_type

    def optimizer_kwargs(self) -> Dict[str, Any]:
        return self._optimizer_kwargs

    def get_initial_lr(self) -> float:
        for tensor in self.learning_rates.values():
            return tensor.item()
        lr = self._optimizer_kwargs.get("lr")
        if lr is not None:
            return float(lr)
        return 0.01

    def set_learning_rate(self, lr: float) -> None:
        for name in self.learning_rates:
            self.learning_rates[name].fill_(lr)

    def _shard_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        num_shards = self._global_device_count * self._num_sc_per_device
        return torch.cat(
            [tensor[i::num_shards] for i in range(num_shards)], dim=0
        ).to(self._device)

    def _unstack_activation(
        self,
        out: torch.Tensor,
        num_features: int,
        feature_batch_size: int,
    ) -> torch.Tensor:
        """Unstacks and uninterleaves activations across SparseCore cores.

        Args:
          out: The output tensor from sparse-dense matmul of shape
            [feature_batch_size * num_features, embedding_dim].
          num_features: Number of features stacked on this table.
          feature_batch_size: Batch size per feature (process-local batch size for
            EBC, or batch_size * max_seq_len for EC).

        Returns:
          Tensor of shape [num_features, feature_batch_size, embedding_dim].
        """
        if feature_batch_size % self._num_sc_per_device != 0:
            raise ValueError(
                f"feature_batch_size ({feature_batch_size}) must be divisible by"
                f" num_sc_per_device ({self._num_sc_per_device})"
            )
        per_sc_batch_size = feature_batch_size // self._num_sc_per_device
        return (
            out.contiguous()
            .view(self._num_sc_per_device, num_features, per_sc_batch_size, -1)
            .transpose(0, 1)
            .reshape(num_features, feature_batch_size, -1)
        )

    def _unstack_and_slice_features(
        self,
        out: torch.Tensor,
        stack_config: StackedSparseCoreEmbeddingConfig,
        feature_batch_size: int,
    ) -> List[tuple[str, torch.Tensor, SparseCoreEmbeddingConfig]]:
        """Unstacks, uninterleaves, and unpads embedding feature activations.

        Args:
          out: The output tensor from sparse-dense matmul of shape
            [feature_batch_size * total_features, stack_embedding_dim].
          stack_config: Configuration of the stacked table.
          feature_batch_size: Batch size per feature (process-local batch size for
            EBC, or batch_size * max_seq_len for EC).

        Returns:
          List of (feature_name, unpadded_feature_tensor, table_config) tuples.
        """
        table_feature_list, per_feature_dims = self._stack_unstack_metadata[
            stack_config.stack_name
        ]
        if self._use_custom_embedding_formatting and out.device.type == "tpu":
            per_feature_batch_sizes = [feature_batch_size] * len(table_feature_list)
            unstacked_tensors = torch.ops.tpu.sparse_dense_matmul_activation_unstack(
                out,
                per_feature_batch_sizes,
                per_feature_dims,
            )
            return [
                (feat_name, feat_emb, tbl)
                for (feat_name, tbl), feat_emb in zip(
                    table_feature_list, unstacked_tensors
                )
            ]

        unstacked = self._unstack_activation(
            out, stack_config.total_features, feature_batch_size
        )
        results = []
        feat_idx = 0
        for table in stack_config.tables:
            for feature_name in table.feature_names:
                feat_emb = unstacked[feat_idx]
                if stack_config.stack_embedding_dim > table.embedding_dim:
                    feat_emb = feat_emb[:, : table.embedding_dim]
                results.append((feature_name, feat_emb, table))
                feat_idx += 1
        return results

    def _run_table_lookup(
        self,
        stack_config: StackedSparseCoreEmbeddingConfig,
        features: KeyedSparseCorePreprocessedInput,
        weight: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Runs sparse-dense matmul for a single stacked table."""
        stack_inputs = features.table_tensors[stack_config.stack_name]

        # Unwrap DTensors to local tensors for custom ops.
        local_weight = (
            weight.to_local() if isinstance(weight, dt.DTensor) else weight
        )

        accumulator = self.accumulators.get(stack_config.stack_name, None)
        local_accumulator = (
            accumulator.to_local()
            if isinstance(accumulator, dt.DTensor)
            else accumulator
        )

        momentum = self.momentums.get(stack_config.stack_name, None)
        local_momentum = (
            momentum.to_local() if isinstance(momentum, dt.DTensor) else momentum
        )

        velocity = self.velocities.get(stack_config.stack_name, None)
        local_velocity = (
            velocity.to_local() if isinstance(velocity, dt.DTensor) else velocity
        )

        return run_sparse_dense_matmul(
            stack_inputs.row_pointers,
            stack_inputs.embedding_ids,
            stack_inputs.sample_ids,
            stack_inputs.gains,
            local_weight,
            self.learning_rates[stack_config.stack_name],
            batch_size,
            stack_config.max_ids_per_partition,
            stack_config.max_unique_ids_per_partition,
            self._optimizer_type,
            stack_config.stack_name,
            local_accumulator,
            local_momentum,
            local_velocity,
            self._optimizer_kwargs.get("eps", 1e-10),
            self._optimizer_kwargs.get("beta1", 0.9),
            self._optimizer_kwargs.get("beta2", 0.999),
        )

    @property
    def table_weights(self) -> Dict[str, torch.Tensor]:
        """Returns dictionary mapping stack names to current stacked table weights."""
        weight_dict = getattr(self, self._weight_dict_name)
        return {name: mod.weight for name, mod in weight_dict.items()}

    def get_unstacked_table_weights(self) -> Dict[str, torch.Tensor]:
        """Returns individual unstacked and unsharded table weights."""
        weight_dict = getattr(self, self._weight_dict_name)
        stacked_weights = {
            name: (
                mod.weight.to_local()
                if hasattr(mod.weight, "to_local")
                else mod.weight
            )
            for name, mod in weight_dict.items()
        }
        return table_stacking.unshard_and_unstack_tables(
            stacked_weights,
            self._stacked_configs,
            global_device_count=self._global_device_count,
            num_sc_per_device=self._num_sc_per_device,
        )

    def set_unstacked_table_weights(
        self, table_weights: Dict[str, torch.Tensor]
    ) -> None:
        """Sets individual table weights by stacking and sharding them into the parameters."""
        device_table_weights = {
            name: w.to(self._device) for name, w in table_weights.items()
        }
        stacked_weights = table_stacking.stack_and_shard_tables(
            device_table_weights,
            self._stacked_configs,
            global_device_count=self._global_device_count,
            num_sc_per_device=self._num_sc_per_device,
        )
        weight_dict = getattr(self, self._weight_dict_name)
        for stack_name, w in stacked_weights.items():
            with torch.no_grad():
                param = weight_dict[stack_name].weight
                if hasattr(param, "to_local"):
                    param.to_local().copy_(w.to(self._device))
                else:
                    param.copy_(w.to(self._device))

    def unstacked_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns state dict containing individual table weights on CPU for checkpointing."""
        return {
            name: w.to("cpu")
            for name, w in self.get_unstacked_table_weights().items()
        }

    def load_unstacked_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Loads individual table weights from state dict into stacked parameters."""
        self.set_unstacked_table_weights(state_dict)

    def _stack_gradient(
        self,
        grad: torch.Tensor,
        num_features: int,
        feature_batch_size: int,
    ) -> torch.Tensor:
        """Stacks and interleaves feature gradients across SparseCore cores.

        Inverse of `_unstack_activation`.

        Args:
          grad: Gradients of shape [num_features, feature_batch_size,
            embedding_dim].
          num_features: Number of features stacked on this table.
          feature_batch_size: Batch size per feature.

        Returns:
          Stacked gradient tensor of shape [feature_batch_size * num_features,
          embedding_dim].
        """
        if feature_batch_size % self._num_sc_per_device != 0:
            raise ValueError(
                f"feature_batch_size ({feature_batch_size}) must be divisible by"
                f" num_sc_per_device ({self._num_sc_per_device})"
            )
        per_sc_batch_size = feature_batch_size // self._num_sc_per_device
        return (
            grad.contiguous()
            .view(num_features, self._num_sc_per_device, per_sc_batch_size, -1)
            .transpose(0, 1)
            .reshape(feature_batch_size * num_features, -1)
        )

    def _get_table_weight(self, stack_name: str) -> torch.Tensor:
        return getattr(self, self._weight_dict_name)[stack_name].weight

    def _get_batch_size_per_feature(
        self, stack_config: StackedSparseCoreEmbeddingConfig
    ) -> int:
        del stack_config
        return self._batch_size

    def sc_forward(
        self,
        features: KeyedSparseCorePreprocessedInput,
        embedding_tables: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Runs SparseCore forward lookups and returns raw stacked activations per stack.

        Args:
          features: Preprocessed sparse input features.
          embedding_tables: Optional dictionary mapping stack names to embedding
            tables. If provided, table lookups use these tensors instead of the
            module parameters (functional lookup).

        Returns:
          Dictionary mapping stack names to raw stacked activation tensors.
        """
        raw_outputs = {}
        for stack_config in self._stacked_configs:
            if (
                embedding_tables is not None
                and stack_config.stack_name in embedding_tables
            ):
                embedding_table = embedding_tables[stack_config.stack_name]
            else:
                embedding_table = self._get_table_weight(stack_config.stack_name)
            total_batch_size = (
                self._get_batch_size_per_feature(stack_config)
                * stack_config.total_features
            )
            raw_outputs[stack_config.stack_name] = self._run_table_lookup(
                stack_config, features, embedding_table, total_batch_size
            )
        return raw_outputs

    def _update_param_state(
        self,
        param_dict: Union[nn.ParameterDict, Dict[str, torch.Tensor]],
        key: str,
        old_tensor: Optional[torch.Tensor],
        new_local_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Wraps local TPU output as DTensor and updates parameter/buffer dictionary."""
        if old_tensor is not None and (
            isinstance(old_tensor, dt.DTensor) or hasattr(old_tensor, "device_mesh")
        ):
            updated = dt.DTensor.from_local(
                new_local_tensor,
                device_mesh=old_tensor.device_mesh,
                placements=old_tensor.placements,
            )
        else:
            updated = new_local_tensor

        if torch.compiler.is_compiling() and isinstance(
            param_dict, nn.ParameterDict
        ):
            param_dict._parameters[key] = updated
        elif isinstance(param_dict, nn.ParameterDict):
            param_dict[key] = nn.Parameter(
                updated,
                requires_grad=getattr(old_tensor, "requires_grad", False),
            )
        else:
            param_dict[key] = updated
        return updated

    def sc_backward(
        self,
        features: KeyedSparseCorePreprocessedInput,
        raw_gradients: Dict[str, torch.Tensor],
        embedding_tables: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Runs functional SparseCore backward pass returning updated embedding tables.

        Args:
          features: Preprocessed sparse input features.
          raw_gradients: Raw stacked gradients per stack table.
          embedding_tables: Optional dictionary mapping stack names to input
            embedding tables. If None, uses current module table weights.

        Returns:
          Dictionary mapping stack names to updated embedding tables.

        Raises:
          RuntimeError: If required optimizer states (e.g. accumulator for Adagrad,
            or momentum/velocity for Adam) are uninitialized.
        """
        if embedding_tables is None:
            embedding_tables = self.table_weights

        updated_tables = dict(embedding_tables)
        for stack_config in self._stacked_configs:
            stack_name = stack_config.stack_name
            if stack_name not in raw_gradients:
                continue
            grad_output = raw_gradients[stack_name]
            table_inputs = features.table_tensors[stack_name]
            table = embedding_tables.get(
                stack_name, self._get_table_weight(stack_name)
            )
            local_weight = (
                table.to_local() if isinstance(table, dt.DTensor) else table
            )

            accumulator = self.accumulators.get(stack_name, None)
            local_accumulator = (
                accumulator.to_local()
                if isinstance(accumulator, dt.DTensor)
                else accumulator
            )

            momentum = self.momentums.get(stack_name, None)
            local_momentum = (
                momentum.to_local() if isinstance(momentum, dt.DTensor) else momentum
            )

            velocity = self.velocities.get(stack_name, None)
            local_velocity = (
                velocity.to_local() if isinstance(velocity, dt.DTensor) else velocity
            )

            total_batch_size = (
                self._get_batch_size_per_feature(stack_config)
                * stack_config.total_features
            )

            lr = self.learning_rates[stack_name]
            max_ids = stack_config.max_ids_per_partition
            max_unique_ids = stack_config.max_unique_ids_per_partition

            if self._optimizer_type == torch.optim.SGD:
                new_table = torch.ops.tpu.sparse_dense_matmul_grad_with_sgd(
                    table_inputs.row_pointers,
                    table_inputs.embedding_ids,
                    table_inputs.sample_ids,
                    table_inputs.gains,
                    local_weight,
                    grad_output.contiguous(),
                    lr,
                    device_batch_size=total_batch_size,
                    max_ids_per_partition=max_ids,
                    max_unique_ids_per_partition=max_unique_ids,
                    computation_name=f"sgd_{stack_name}",
                )
            elif self._optimizer_type == torch.optim.Adagrad:
                if local_accumulator is None:
                    raise RuntimeError(
                        f"local_accumulator for stack {stack_name} is None."
                    )
                eps = self._optimizer_kwargs.get("eps", 1e-10)
                new_table, new_accumulator = (
                    torch.ops.tpu.sparse_dense_matmul_grad_with_adagrad(
                        table_inputs.row_pointers,
                        table_inputs.embedding_ids,
                        table_inputs.sample_ids,
                        table_inputs.gains,
                        local_weight,
                        local_accumulator,
                        grad_output.contiguous(),
                        lr,
                        eps,
                        device_batch_size=total_batch_size,
                        max_ids_per_partition=max_ids,
                        max_unique_ids_per_partition=max_unique_ids,
                        computation_name=f"adagrad_{stack_name}",
                    )
                )
                self._update_param_state(
                    self.accumulators, stack_name, accumulator, new_accumulator
                )
            elif self._optimizer_type == torch.optim.Adam:
                if local_momentum is None or local_velocity is None:
                    raise RuntimeError(
                        f"local_momentum or local_velocity for stack {stack_name} is"
                        " None."
                    )
                eps = self._optimizer_kwargs.get("eps", 1e-10)
                b1 = self._optimizer_kwargs.get("beta1", 0.9)
                b2 = self._optimizer_kwargs.get("beta2", 0.999)
                new_table, new_momentum, new_velocity = (
                    torch.ops.tpu.sparse_dense_matmul_grad_with_adam(
                        table_inputs.row_pointers,
                        table_inputs.embedding_ids,
                        table_inputs.sample_ids,
                        table_inputs.gains,
                        local_weight,
                        local_momentum,
                        local_velocity,
                        grad_output.contiguous(),
                        lr,
                        b1,
                        b2,
                        eps,
                        device_batch_size=total_batch_size,
                        max_ids_per_partition=max_ids,
                        max_unique_ids_per_partition=max_unique_ids,
                        computation_name=f"adam_{stack_name}",
                    )
                )
                self._update_param_state(
                    self.momentums, stack_name, momentum, new_momentum
                )
                self._update_param_state(
                    self.velocities, stack_name, velocity, new_velocity
                )
            else:
                raise ValueError(f"Unsupported optimizer {self._optimizer_type}")

            self._update_param_state(updated_tables, stack_name, table, new_table)

        return updated_tables

    def stack_gradients(
        self,
        unstacked_gradients: KeyedTensor | Dict[str, torch.Tensor] | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stacks and interleaves feature gradients per stack table."""
        if isinstance(unstacked_gradients, KeyedTensor):
            unstacked_gradients = unstacked_gradients.values()

        if isinstance(unstacked_gradients, torch.Tensor):
            all_features = []
            for stack_config in self._stacked_configs:
                for table in stack_config.tables:
                    all_features.extend(table.feature_names)
            if len(all_features) == 1:
                grads_dict = {all_features[0]: unstacked_gradients}
            else:
                grads_dict = {}
                offset = 0
                for stack_config in self._stacked_configs:
                    for table in stack_config.tables:
                        for f_name in table.feature_names:
                            grads_dict[f_name] = unstacked_gradients[
                                ..., offset : offset + table.embedding_dim
                            ]
                            offset += table.embedding_dim
            unstacked_gradients = grads_dict

        if isinstance(unstacked_gradients, dict):
            sample_grad = next(
                (g for g in unstacked_gradients.values() if g is not None),
                None,
            )
            dev = sample_grad.device if sample_grad is not None else self._device
            dtype = sample_grad.dtype if sample_grad is not None else torch.float32
        else:
            dev = unstacked_gradients.device
            dtype = unstacked_gradients.dtype

        raw_gradients = {}
        for stack_config in self._stacked_configs:
            feature_batch_size = self._get_batch_size_per_feature(stack_config)
            num_features = stack_config.total_features
            stack_dim = stack_config.stack_embedding_dim
            feature_grads = []
            for table in stack_config.tables:
                table_dim = table.embedding_dim
                for feature_name in table.feature_names:
                    if (
                        isinstance(unstacked_gradients, dict)
                        and feature_name in unstacked_gradients
                        and unstacked_gradients[feature_name] is not None
                    ):
                        f_grad = unstacked_gradients[feature_name]
                        if f_grad.shape[0] < feature_batch_size:
                            f_grad = torch.nn.functional.pad(
                                f_grad, (0, 0, 0, feature_batch_size - f_grad.shape[0])
                            )
                        if stack_dim > table_dim:
                            f_grad = torch.nn.functional.pad(
                                f_grad, (0, stack_dim - table_dim)
                            )
                        feature_grads.append(f_grad)
                    else:
                        feature_grads.append(
                            torch.zeros(
                                (feature_batch_size, stack_dim),
                                dtype=dtype,
                                device=dev,
                            )
                        )
            stacked_f_grads = torch.stack(feature_grads, dim=0)
            raw_gradients[stack_config.stack_name] = self._stack_gradient(
                stacked_f_grads, num_features, feature_batch_size
            )
        return raw_gradients


class SparseCoreFusedEmbeddingBagCollection(
    SparseCoreEmbeddingBagCollectionInterface, _SparseCoreFusedEmbeddingBase
):
    """EmbeddingBagCollection replacement using TPU SparseCore SparseDenseMatmul with preprocessed inputs."""

    def __init__(
        self,
        tables: List[SparseCoreEmbeddingConfig],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        batch_size: int = 1,
        global_device_count: int = 1,
        num_sc_per_device: int = 2,
        auto_stack: bool = False,
        activation_mem_bytes_limit: int = 6 * 1024 * 1024,
        use_custom_embedding_formatting: bool = True,
    ) -> None:
        super().__init__(
            tables=tables,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            weight_dict_name="embedding_bags",
            batch_size=batch_size,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            auto_stack=auto_stack,
            activation_mem_bytes_limit=activation_mem_bytes_limit,
            use_custom_embedding_formatting=use_custom_embedding_formatting,
        )

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        configs: List[EmbeddingBagConfig] = []
        for t in self._tables:
            config = t.config
            if isinstance(config, EmbeddingBagConfig):
                configs.append(config)
        return configs

    def forward(self, features: KeyedSparseCorePreprocessedInput) -> KeyedTensor:
        return self.unstack_activations(self.sc_forward(features), features)

    def unstack_activations(
        self,
        raw_activations: Dict[str, torch.Tensor],
        features: Optional[Any] = None,
    ) -> KeyedTensor:
        """Unstacks and uninterleaves raw activations across tables into a KeyedTensor."""
        del features  # Unused in EmbeddingBagCollection
        pooled_embeddings = []
        embedding_names = []
        for stack_config in self._stacked_configs:
            out = raw_activations[stack_config.stack_name]
            for feature_name, feat_emb, _ in self._unstack_and_slice_features(
                out, stack_config, self._batch_size
            ):
                pooled_embeddings.append(feat_emb)
                embedding_names.append(feature_name)

        concat_values = torch.cat(pooled_embeddings, dim=1)
        length_per_key = [
            self._feature_to_table_config[name].embedding_dim
            for name in embedding_names
        ]
        return KeyedTensor(
            keys=embedding_names,
            values=concat_values,
            length_per_key=length_per_key,
        )


class SparseCoreFusedEmbeddingCollection(
    SparseCoreEmbeddingCollectionInterface, _SparseCoreFusedEmbeddingBase
):
    """EmbeddingCollection replacement using TPU SparseCore SparseDenseMatmul with preprocessed inputs."""

    def __init__(
        self,
        tables: List[SparseCoreEmbeddingConfig],
        optimizer_type: Type[torch.optim.Optimizer] = torch.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        global_device_count: int = 1,
        num_sc_per_device: int = 2,
        auto_stack: bool = False,
        activation_mem_bytes_limit: int = 6 * 1024 * 1024,
        use_custom_embedding_formatting: bool = True,
    ) -> None:
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.01}
        else:
            optimizer_kwargs = dict(optimizer_kwargs)

        for table in tables:
            if (
                isinstance(table.config, EmbeddingConfig)
                and table.max_seq_len is None
            ):
                raise ValueError(
                    "max_seq_len must be provided for EmbeddingConfig table"
                    f" {table.name}"
                )

        super().__init__(
            tables=tables,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            weight_dict_name="embeddings",
            batch_size=batch_size,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            auto_stack=auto_stack,
            activation_mem_bytes_limit=activation_mem_bytes_limit,
            use_custom_embedding_formatting=use_custom_embedding_formatting,
        )

    def embedding_configs(self) -> List[EmbeddingConfig]:
        configs: List[EmbeddingConfig] = []
        for t in self._tables:
            config = t.config
            if isinstance(config, EmbeddingConfig):
                configs.append(config)
        return configs

    def _get_batch_size_per_feature(
        self, stack_config: StackedSparseCoreEmbeddingConfig
    ) -> int:
        multiplier = (
            stack_config.max_seq_len if stack_config.max_seq_len is not None else 1
        )
        return self._batch_size * multiplier

    def forward(
        self, features: KeyedSparseCorePreprocessedInput
    ) -> Dict[str, JaggedTensor]:
        return self.unstack_activations(self.sc_forward(features), features)

    def unstack_activations(
        self,
        raw_activations: Dict[str, torch.Tensor] | torch.Tensor,
        features: Optional[
            KeyedSparseCorePreprocessedInput | Dict[str, torch.Tensor]
        ] = None,
    ) -> Dict[str, JaggedTensor]:
        """Unstacks raw activations into a dictionary of JaggedTensors."""
        if not isinstance(raw_activations, dict) and isinstance(features, dict):
            raw_activations, features = features, raw_activations

        if raw_activations is None:
            raise ValueError("raw_activations must be provided.")
        output_dict = {}
        for stack_config in self._stacked_configs:
            if stack_config.max_seq_len is None:
                raise ValueError(
                    "max_seq_len must be specified for sequence table"
                    f" {stack_config.stack_name}"
                )
            seq_batch_size = self._batch_size * stack_config.max_seq_len
            out = raw_activations[stack_config.stack_name]
            stack_inputs = (
                features.table_tensors[stack_config.stack_name]
                if isinstance(features, KeyedSparseCorePreprocessedInput)
                else None
            )

            for feature_name, feat_emb, _ in self._unstack_and_slice_features(
                out, stack_config, seq_batch_size
            ):
                if (
                    stack_inputs is not None
                    and feature_name in stack_inputs.lengths
                    and feature_name in stack_inputs.actual_num_ids
                ):
                    lengths = stack_inputs.lengths[feature_name]
                    actual_num_ids = stack_inputs.actual_num_ids[feature_name]
                    output_dict[feature_name] = JaggedTensor(
                        values=feat_emb[:actual_num_ids],
                        lengths=lengths,
                    )
                else:
                    lengths = torch.full(
                        (self._batch_size,),
                        stack_config.max_seq_len,
                        dtype=torch.int32,
                        device=feat_emb.device,
                    )
                    output_dict[feature_name] = JaggedTensor(
                        values=feat_emb,
                        lengths=lengths,
                    )
        return output_dict

    def stack_gradients(
        self,
        unstacked_gradients: Dict[str, torch.Tensor] | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stacks and interleaves feature gradients per stack table."""
        if isinstance(unstacked_gradients, torch.Tensor):
            all_features = [
                f
                for stack_config in self._stacked_configs
                for table in stack_config.tables
                for f in table.feature_names
            ]
            if len(all_features) == 1:
                grads_dict = {all_features[0]: unstacked_gradients}
            else:
                grads_dict = {}
                offset = 0
                for stack_config in self._stacked_configs:
                    for table in stack_config.tables:
                        for f_name in table.feature_names:
                            grads_dict[f_name] = unstacked_gradients[
                                ..., offset : offset + table.embedding_dim
                            ]
                            offset += table.embedding_dim
            unstacked_gradients = grads_dict

        if isinstance(unstacked_gradients, dict):
            sample_grad = next(
                (g for g in unstacked_gradients.values() if g is not None),
                None,
            )
            if isinstance(sample_grad, JaggedTensor):
                dev = sample_grad.values().device
                dtype = sample_grad.values().dtype
            elif sample_grad is not None:
                dev = sample_grad.device
                dtype = sample_grad.dtype
            else:
                dev = self._device
                dtype = torch.float32
        elif isinstance(unstacked_gradients, JaggedTensor):
            dev = unstacked_gradients.values().device
            dtype = unstacked_gradients.values().dtype
        else:
            dev = unstacked_gradients.device
            dtype = unstacked_gradients.dtype

        raw_gradients = {}
        for stack_config in self._stacked_configs:
            if stack_config.max_seq_len is None:
                raise ValueError(
                    "max_seq_len must be specified for sequence table"
                    f" {stack_config.stack_name}"
                )
            max_seq = stack_config.max_seq_len
            seq_batch_size = self._batch_size * max_seq
            num_features = stack_config.total_features
            stack_dim = stack_config.stack_embedding_dim
            feature_grads = []
            for table in stack_config.tables:
                table_dim = table.embedding_dim
                for feature_name in table.feature_names:
                    if (
                        feature_name in unstacked_gradients
                        and unstacked_gradients[feature_name] is not None
                    ):
                        f_grad = unstacked_gradients[feature_name]
                        if isinstance(f_grad, JaggedTensor):
                            vals = f_grad.values()
                            if vals.shape[0] < seq_batch_size:
                                f_grad = torch.nn.functional.pad(
                                    vals, (0, 0, 0, seq_batch_size - vals.shape[0])
                                )
                            else:
                                f_grad = vals[:seq_batch_size]
                        elif f_grad.dim() == 3:
                            if f_grad.shape[1] < max_seq:
                                f_grad = torch.nn.functional.pad(
                                    f_grad, (0, 0, 0, max_seq - f_grad.shape[1])
                                )
                            f_grad = f_grad.reshape(-1, f_grad.shape[-1])
                        elif f_grad.shape[0] < seq_batch_size:
                            f_grad = torch.nn.functional.pad(
                                f_grad, (0, 0, 0, seq_batch_size - f_grad.shape[0])
                            )
                        if stack_dim > table_dim:
                            f_grad = torch.nn.functional.pad(
                                f_grad, (0, stack_dim - table_dim)
                            )
                        feature_grads.append(f_grad)
                    else:
                        feature_grads.append(
                            torch.zeros(
                                (seq_batch_size, stack_dim),
                                dtype=dtype,
                                device=dev,
                            )
                        )
            stacked_f_grads = torch.stack(feature_grads, dim=0)
            raw_gradients[stack_config.stack_name] = self._stack_gradient(
                stacked_f_grads, num_features, seq_batch_size
            )
        return raw_gradients
