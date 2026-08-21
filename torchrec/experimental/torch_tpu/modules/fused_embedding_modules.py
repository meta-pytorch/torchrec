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
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
import torch.distributed as dist
import torch.distributed.tensor as dt
from torchrec.experimental.torch_tpu.datasets import input_preprocessing
from torchrec.experimental.torch_tpu.modules import custom_ops
from torchrec.experimental.torch_tpu.modules.embedding_configs import SparseCoreEmbeddingConfig
# import torch_tpu._internal.device_utils.annotations as tpu_annotations
from torchrec.modules import embedding_configs
from torchrec.optim import fused
from torchrec.optim import keyed
from torchrec.sparse import jagged_tensor

KeyedOptimizer = keyed.KeyedOptimizer
FusedOptimizer = fused.FusedOptimizer
FusedOptimizerModule = fused.FusedOptimizerModule

KeyedSparseCorePreprocessedInput = (
    input_preprocessing.KeyedSparseCorePreprocessedInput
)
EmbeddingConfig = embedding_configs.EmbeddingConfig
KeyedTensor = jagged_tensor.KeyedTensor
JaggedTensor = jagged_tensor.JaggedTensor
EmbeddingBagConfig = embedding_configs.EmbeddingBagConfig

_DISPATCH_TABLE = {
    torch.optim.SGD: custom_ops.sparse_dense_matmul_sgd_fwd,
    torch.optim.Adagrad: custom_ops.sparse_dense_matmul_adagrad_fwd,
    torch.optim.Adam: custom_ops.sparse_dense_matmul_adam_fwd,
}

def _round_up_to_multiple(x: int, y: int) -> int:
  return ((x + y - 1) // y) * y

class SparseCoreFusedOptimizer(FusedOptimizer):
  """FusedOptimizer replacement using TPU SparseCore SparseDenseMatmul."""

  def __init__(self, emb_module) -> None:
    self._emb_module = emb_module
    params = {}
    state = {}
    for name in emb_module._table_configs:
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

  def _sync_lr(self) -> None:
    param_groups = list(self.param_groups)
    if param_groups:
      new_lr = float(param_groups[0]["lr"])
      self._emb_module.set_learning_rate(new_lr)

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
  """Runs SparseCore SparseDenseMatmul with the given optimizer."""
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
    assert accumulator is not None, "accumulator must be provided for Adagrad"
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
    assert momentum is not None, "momentum must be provided for Adam"
    assert velocity is not None, "velocity must be provided for Adam"
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
  ) -> None:
    super().__init__()
    self._tables = tables
    self._optimizer_type = optimizer_type

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

    # Always use TPU device for embedding tables.
    self._device = torch.device("tpu")
    self._device_mesh = None
    if dist.is_initialized() and self._global_device_count > 1:
      self._device_mesh = dt.init_device_mesh(
          "tpu", (self._global_device_count,)
      )

    self.learning_rates = nn.ParameterDict()

    for config in tables:
      self.learning_rates[config.name] = nn.Parameter(
          torch.tensor(
              self._optimizer_kwargs.get("lr", 0.01),
              dtype=torch.float32,
              device=self._device,
          ),
          requires_grad=False,
      )

    # Weight modules dict — aliased as `embedding_bags` or `embeddings`
    # by each subclass for backward compatibility.
    weight_modules = nn.ModuleDict()
    setattr(self, weight_dict_name, weight_modules)

    # Optimizer states.
    self.accumulators = nn.ParameterDict()
    self.momentums = nn.ParameterDict()
    self.velocities = nn.ParameterDict()
    self._table_configs: Dict[str, SparseCoreEmbeddingConfig] = {}
    self._feature_to_table_config: Dict[str, SparseCoreEmbeddingConfig] = {}

    # layout = tpu_annotations.TpuLayout(minor_to_major=[0, 1], tiles=[[8, 128]])
    for config in tables:
      self._table_configs[config.name] = config
      for feature_name in config.feature_names:
        self._feature_to_table_config[feature_name] = config

      # Initialize weights assuming MOD sharding across SC devices.
      # round up num_embeddings to be divisible by global_device_count
      num_embeddings = _round_up_to_multiple(
          config.num_embeddings,
          self._global_device_count * self._num_sc_per_device * 8,
      )
      embedding_dim = _round_up_to_multiple(config.embedding_dim, 8)

      table_shape = (num_embeddings // self._global_device_count, embedding_dim)
      embedding_table = torch.empty(table_shape, dtype=torch.float32)
      if config.init_fn:
        config.init_fn(embedding_table)
      else:
        nn.init.uniform_(embedding_table, -0.01, 0.01)

      # Clone to TPU with T8 layout.
      # with tpu_annotations.LayoutContext(layout):
      embedding_table_device = embedding_table.to(self._device)

      if self._device_mesh is not None:
        sharded_weight = dt.DTensor.from_local(
            embedding_table_device,
            device_mesh=self._device_mesh,
            placements=[dt.Shard(0)],
        )
        weight_param = nn.Parameter(sharded_weight)
      else:
        weight_param = nn.Parameter(embedding_table_device)

      weight_modules[config.name] = torch.nn.Module()
      weight_modules[config.name].register_parameter("weight", weight_param)

      if self._optimizer_type == torch.optim.SGD:
        pass
      elif self._optimizer_type == torch.optim.Adagrad:
        accumulator = torch.full(
            table_shape,
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
          self.accumulators[config.name] = nn.Parameter(sharded_accumulator)
        else:
          self.accumulators[config.name] = nn.Parameter(accumulator)
      elif self._optimizer_type == torch.optim.Adam:
        momentum = torch.zeros(
            table_shape, dtype=torch.float32, device=self._device
        )
        velocity = torch.zeros(
            table_shape, dtype=torch.float32, device=self._device
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
          self.momentums[config.name] = nn.Parameter(sharded_momentum)
          self.velocities[config.name] = nn.Parameter(sharded_velocity)
        else:
          self.momentums[config.name] = nn.Parameter(momentum)
          self.velocities[config.name] = nn.Parameter(velocity)

    # Synchronize TPU devices to ensure all tables are initialized
    # before the first forward pass.
    if hasattr(torch, "tpu") and self._device.type == "tpu":
      torch.tpu.synchronize()

  @property
  def fused_optimizer(self) -> KeyedOptimizer:
    return SparseCoreFusedOptimizer(self)

  @property
  def device(self) -> torch.device:
    return self._device

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
    for name in self._table_configs:
      if name in self.learning_rates:
        self.learning_rates[name].copy_(
            torch.tensor(lr, dtype=torch.float32, device=self._device)
        )

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
    assert feature_batch_size % self._num_sc_per_device == 0, (
        f"feature_batch_size ({feature_batch_size}) must be divisible by"
        f" num_sc_per_device ({self._num_sc_per_device})"
    )
    per_sc_batch_size = feature_batch_size // self._num_sc_per_device
    return (
        out.view(self._num_sc_per_device, num_features, per_sc_batch_size, -1)
        .transpose(0, 1)
        .reshape(num_features, feature_batch_size, -1)
    )

  def _run_table_lookup(
      self,
      table: SparseCoreEmbeddingConfig,
      features: KeyedSparseCorePreprocessedInput,
      weight: torch.Tensor,
      batch_size: int,
  ) -> torch.Tensor:
    """Runs sparse-dense matmul for a single table."""
    table_inputs = features.table_tensors[table.name]

    # Unwrap DTensors to local tensors for custom ops.
    local_weight = (
        weight.to_local() if isinstance(weight, dt.DTensor) else weight
    )

    accumulator = self.accumulators._parameters.get(table.name, None)
    local_accumulator = (
        accumulator.to_local()
        if isinstance(accumulator, dt.DTensor)
        else accumulator
    )

    momentum = self.momentums._parameters.get(table.name, None)
    local_momentum = (
        momentum.to_local() if isinstance(momentum, dt.DTensor) else momentum
    )

    velocity = self.velocities._parameters.get(table.name, None)
    local_velocity = (
        velocity.to_local() if isinstance(velocity, dt.DTensor) else velocity
    )

    return run_sparse_dense_matmul(
        table_inputs.row_pointers,
        table_inputs.embedding_ids,
        table_inputs.sample_ids,
        table_inputs.gains,
        local_weight,
        self.learning_rates[table.name],
        batch_size,
        table.max_ids_per_partition,
        table.max_unique_ids_per_partition,
        self._optimizer_type,
        table.name,
        local_accumulator,
        local_momentum,
        local_velocity,
        self._optimizer_kwargs.get("eps", 1e-10),
        self._optimizer_kwargs.get("beta1", 0.9),
        self._optimizer_kwargs.get("beta2", 0.999),
    )


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
  ) -> None:
    super().__init__(
        tables=tables,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        weight_dict_name="embedding_bags",
        batch_size=batch_size,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
    )

  def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
    configs: List[EmbeddingBagConfig] = []
    for t in self._tables:
      config = t.config
      if isinstance(config, EmbeddingBagConfig):
        configs.append(config)
    return configs

  def forward(self, features: KeyedSparseCorePreprocessedInput) -> KeyedTensor:
    pooled_embeddings = []
    embedding_names = []

    for table in self._tables:
      embedding_table = self.embedding_bags[table.name].weight
      num_features = len(table.feature_names)
      total_batch_size = self._batch_size * num_features

      out = self._run_table_lookup(
          table, features, embedding_table, total_batch_size
      )

      # Unstack and uninterleave features across SparseCore cores.
      unstacked = self._unstack_activation(out, num_features, self._batch_size)
      for f_idx, feature_name in enumerate(table.feature_names):
        pooled_embeddings.append(unstacked[f_idx][:, : table.embedding_dim])
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
    )

  def embedding_configs(self) -> List[EmbeddingConfig]:
    configs: List[EmbeddingConfig] = []
    for t in self._tables:
      config = t.config
      if isinstance(config, EmbeddingConfig):
        configs.append(config)
    return configs

  def forward(
      self, features: KeyedSparseCorePreprocessedInput
  ) -> Dict[str, JaggedTensor]:
    output_dict = {}
    for table in self._tables:
      embedding_table = self.embeddings[table.name].weight

      max_seq_len_table = table.max_seq_len
      assert max_seq_len_table is not None
      seq_batch_size = self._batch_size * max_seq_len_table
      num_features = len(table.feature_names)
      total_batch_size = seq_batch_size * num_features

      out = self._run_table_lookup(
          table, features, embedding_table, total_batch_size
      )

      # out has shape [seq_batch_size * num_features, D].
      # Unstack and uninterleave features across SparseCore cores, then slice
      # back to the actual length.
      unstacked = self._unstack_activation(out, num_features, seq_batch_size)
      table_inputs = features.table_tensors[table.name]

      for f_idx, feature_name in enumerate(table.feature_names):
        lengths = table_inputs.lengths[feature_name]
        actual_num_ids = table_inputs.actual_num_ids[feature_name]
        feature_out = unstacked[f_idx][:actual_num_ids, : table.embedding_dim]
        output_dict[feature_name] = JaggedTensor(
            values=feature_out,
            lengths=lengths,
        )

    return output_dict
