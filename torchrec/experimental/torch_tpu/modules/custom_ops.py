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

"""Custom torch.library operators for SparseCore embedding lookups and optimizer backward passes."""

import torch


# --- SGD Custom Ops ---
@torch.library.custom_op(
    "torchrec_torch_tpu::sparse_dense_matmul_sgd_fwd", mutates_args=()
)
def sparse_dense_matmul_sgd_fwd(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    learning_rate: torch.Tensor,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
) -> torch.Tensor:
  """Executes the forward pass of sparse-dense matrix multiplication for SGD."""
  return torch.ops.tpu.sparse_dense_matmul(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  )


@torch.library.custom_op(
    "torchrec_torch_tpu::sparse_dense_matmul_sgd_bwd",
    mutates_args=("embedding_table",),
)
def sparse_dense_matmul_sgd_bwd(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    learning_rate: torch.Tensor,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
) -> None:
  """Executes the backward pass and inplace SGD weight update."""
  updated_table = torch.ops.tpu.sparse_dense_matmul_grad_with_sgd(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      grad_output.contiguous(),
      learning_rate,
      device_batch_size=batch_size,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      computation_name=f"sgd_{table_name}",
  )
  embedding_table.copy_(updated_table)
  return None


@sparse_dense_matmul_sgd_fwd.register_fake
def _sgd_fwd_fake(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    learning_rate: torch.Tensor,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
):
  """Fake implementation for SGD forward pass during tracing."""
  return torch.empty(
      (batch_size, embedding_table.size(1)),
      dtype=embedding_table.dtype,
      device=embedding_table.device,
  )


@sparse_dense_matmul_sgd_bwd.register_fake
def _sgd_bwd_fake(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    learning_rate: torch.Tensor,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
):
  """Fake implementation for SGD backward pass during tracing."""
  embedding_table.add_(0)
  return None


def _sgd_setup_context(ctx, inputs, output):
  """Saves tensors and hyperparameters to autograd context for SGD backward."""
  (
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
  ) = inputs
  ctx.save_for_backward(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      learning_rate,
  )
  ctx.batch_size = batch_size
  ctx.max_ids_per_partition = max_ids_per_partition
  ctx.max_unique_ids_per_partition = max_unique_ids_per_partition
  ctx.table_name = table_name


def _sgd_backward(ctx, grad_output):
  """Autograd backward function for SGD sparse-dense matrix multiplication."""
  (
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      learning_rate,
  ) = ctx.saved_tensors
  sparse_dense_matmul_sgd_bwd(
      grad_output,
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      learning_rate,
      ctx.batch_size,
      ctx.max_ids_per_partition,
      ctx.max_unique_ids_per_partition,
      ctx.table_name,
  )
  return (None,) * 10


sparse_dense_matmul_sgd_fwd.register_autograd(
    _sgd_backward, setup_context=_sgd_setup_context
)


# --- Adagrad Custom Ops ---
@torch.library.custom_op(
    "torchrec_torch_tpu::sparse_dense_matmul_adagrad_fwd", mutates_args=()
)
def sparse_dense_matmul_adagrad_fwd(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    accumulator: torch.Tensor,
    learning_rate: torch.Tensor,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
) -> torch.Tensor:
  """Executes the forward pass of sparse-dense matrix multiplication for Adagrad."""
  return torch.ops.tpu.sparse_dense_matmul(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  )


@torch.library.custom_op(
    "torchrec_torch_tpu::sparse_dense_matmul_adagrad_bwd",
    mutates_args=("embedding_table", "accumulator"),
)
def sparse_dense_matmul_adagrad_bwd(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    accumulator: torch.Tensor,
    learning_rate: torch.Tensor,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
) -> None:
  """Executes the backward pass and inplace Adagrad weight and accumulator update."""
  updated_table, updated_accumulator = (
      torch.ops.tpu.sparse_dense_matmul_grad_with_adagrad(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          embedding_table,
          accumulator,
          grad_output.contiguous(),
          learning_rate,
          epsilon,
          device_batch_size=batch_size,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
          computation_name=f"adagrad_{table_name}",
      )
  )
  embedding_table.copy_(updated_table)
  accumulator.copy_(updated_accumulator)
  return None


@sparse_dense_matmul_adagrad_fwd.register_fake
def _adagrad_fwd_fake(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    accumulator: torch.Tensor,
    learning_rate: torch.Tensor,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
):
  """Fake implementation for Adagrad forward pass during tracing."""
  return torch.empty(
      (batch_size, embedding_table.size(1)),
      dtype=embedding_table.dtype,
      device=embedding_table.device,
  )


@sparse_dense_matmul_adagrad_bwd.register_fake
def _adagrad_bwd_fake(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    accumulator: torch.Tensor,
    learning_rate: torch.Tensor,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
):
  """Fake implementation for Adagrad backward pass during tracing."""
  embedding_table.add_(0)
  accumulator.add_(0)
  return None


def _adagrad_setup_context(ctx, inputs, output):
  """Saves tensors and hyperparameters to autograd context for Adagrad backward."""
  (
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
  ) = inputs
  ctx.save_for_backward(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      accumulator,
      learning_rate,
  )
  ctx.epsilon = epsilon
  ctx.batch_size = batch_size
  ctx.max_ids_per_partition = max_ids_per_partition
  ctx.max_unique_ids_per_partition = max_unique_ids_per_partition
  ctx.table_name = table_name


def _adagrad_backward(ctx, grad_output):
  """Autograd backward function for Adagrad sparse-dense matrix multiplication."""
  (
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      accumulator,
      learning_rate,
  ) = ctx.saved_tensors
  sparse_dense_matmul_adagrad_bwd(
      grad_output,
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      accumulator,
      learning_rate,
      ctx.epsilon,
      ctx.batch_size,
      ctx.max_ids_per_partition,
      ctx.max_unique_ids_per_partition,
      ctx.table_name,
  )
  return (None,) * 12


sparse_dense_matmul_adagrad_fwd.register_autograd(
    _adagrad_backward, setup_context=_adagrad_setup_context
)


# --- Adam Custom Ops ---
@torch.library.custom_op(
    "torchrec_torch_tpu::sparse_dense_matmul_adam_fwd", mutates_args=()
)
def sparse_dense_matmul_adam_fwd(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    momentum: torch.Tensor,
    velocity: torch.Tensor,
    learning_rate: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
) -> torch.Tensor:
  """Executes the forward pass of sparse-dense matrix multiplication for Adam."""
  return torch.ops.tpu.sparse_dense_matmul(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      batch_size,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  )


@torch.library.custom_op(
    "torchrec_torch_tpu::sparse_dense_matmul_adam_bwd",
    mutates_args=("embedding_table", "momentum", "velocity"),
)
def sparse_dense_matmul_adam_bwd(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    momentum: torch.Tensor,
    velocity: torch.Tensor,
    learning_rate: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
) -> None:
  """Executes the backward pass and inplace Adam weight, momentum, and velocity update."""
  updated_table, updated_momentum, updated_velocity = (
      torch.ops.tpu.sparse_dense_matmul_grad_with_adam(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          embedding_table,
          momentum,
          velocity,
          grad_output.contiguous(),
          learning_rate,
          beta1,
          beta2,
          epsilon,
          device_batch_size=batch_size,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
          computation_name=f"adam_{table_name}",
      )
  )
  embedding_table.copy_(updated_table)
  momentum.copy_(updated_momentum)
  velocity.copy_(updated_velocity)
  return None


@sparse_dense_matmul_adam_fwd.register_fake
def _adam_fwd_fake(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    momentum: torch.Tensor,
    velocity: torch.Tensor,
    learning_rate: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
):
  """Fake implementation for Adam forward pass during tracing."""
  return torch.empty(
      (batch_size, embedding_table.size(1)),
      dtype=embedding_table.dtype,
      device=embedding_table.device,
  )


@sparse_dense_matmul_adam_bwd.register_fake
def _adam_bwd_fake(
    grad_output: torch.Tensor,
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    embedding_table: torch.Tensor,
    momentum: torch.Tensor,
    velocity: torch.Tensor,
    learning_rate: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    batch_size: int,
    max_ids_per_partition: int = 256,
    max_unique_ids_per_partition: int = 256,
    table_name: str = "table",
):
  """Fake implementation for Adam backward pass during tracing."""
  embedding_table.add_(0)
  momentum.add_(0)
  velocity.add_(0)
  return None


def _adam_setup_context(ctx, inputs, output):
  """Saves tensors and hyperparameters to autograd context for Adam backward."""
  (
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
  ) = inputs
  ctx.save_for_backward(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      momentum,
      velocity,
      learning_rate,
  )
  ctx.beta1 = beta1
  ctx.beta2 = beta2
  ctx.epsilon = epsilon
  ctx.batch_size = batch_size
  ctx.max_ids_per_partition = max_ids_per_partition
  ctx.max_unique_ids_per_partition = max_unique_ids_per_partition
  ctx.table_name = table_name


def _adam_backward(ctx, grad_output):
  """Autograd backward function for Adam sparse-dense matrix multiplication."""
  (
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      momentum,
      velocity,
      learning_rate,
  ) = ctx.saved_tensors
  sparse_dense_matmul_adam_bwd(
      grad_output,
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      embedding_table,
      momentum,
      velocity,
      learning_rate,
      ctx.beta1,
      ctx.beta2,
      ctx.epsilon,
      ctx.batch_size,
      ctx.max_ids_per_partition,
      ctx.max_unique_ids_per_partition,
      ctx.table_name,
  )
  return (None,) * 15


sparse_dense_matmul_adam_fwd.register_autograd(
    _adam_backward, setup_context=_adam_setup_context
)
