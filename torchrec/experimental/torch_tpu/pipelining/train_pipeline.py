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

"""High-level SparseCore Train Pipeline for PyTorch models."""

import inspect
import logging
import os
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import distributed as dist, nn
from torchrec.experimental.torch_tpu.datasets.input_preprocessing import (
    KeyedSparseCorePreprocessedInput,
)
from torchrec.experimental.torch_tpu.modules import optimizers
from torchrec.experimental.torch_tpu.modules.fused_embedding_modules import (
    SparseCoreFusedEmbeddingBagCollection,
    SparseCoreFusedEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedTensor

_SPMD_SAFE_METADATA_KEY = "spmd_safe"


def _detach_structure(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach()
    if isinstance(val, dict):
        return {k: _detach_structure(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_detach_structure(v) for v in val)
    if hasattr(val, "detach"):
        return val.detach()
    return val


class _ModelDenseWrapper(nn.Module):
    """Wrapper that exposes dense forward pass as standard nn.Module forward for functional_call."""

    def __init__(self, model: nn.Module, forward_fn: Callable[..., Any]):
        super().__init__()
        self._model = model
        self._forward_fn = forward_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._forward_fn(*args, **kwargs)


def _adapt_dense_optimizer(
    optimizer: Optional[Union[torch.optim.Optimizer, optimizers.Optimizer]],
) -> Optional[optimizers.Optimizer]:
    """Adapts a PyTorch optimizer or functional optimizer to functional Optimizer."""
    if optimizer is None:
        return None
    if isinstance(optimizer, optimizers.Optimizer):
        return optimizer
    if isinstance(optimizer, optimizers.KeyedStatelessOptimizer):
        return optimizer.functional_optimizer
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return optimizers.ReferenceAdamw(
            lr=optimizer.defaults.get("lr", 1e-3),
            betas=optimizer.defaults.get("betas", (0.9, 0.999)),
            eps=optimizer.defaults.get("eps", 1e-8),
            weight_decay=optimizer.defaults.get("weight_decay", 0.0),
        )
    if isinstance(optimizer, torch.optim.SGD):
        return optimizers.ReferenceSgd(
            lr=optimizer.defaults.get("lr", 1e-3),
            momentum=optimizer.defaults.get("momentum", 0.0),
            weight_decay=optimizer.defaults.get("weight_decay", 0.0),
        )
    if isinstance(optimizer, torch.optim.Adagrad):
        return optimizers.ReferenceAdagrad(
            lr=optimizer.defaults.get("lr", 1e-3),
            lr_decay=optimizer.defaults.get("lr_decay", 0.0),
            weight_decay=optimizer.defaults.get("weight_decay", 0.0),
            initial_accumulator_value=optimizer.defaults.get(
                "initial_accumulator_value", 0.0
            ),
            eps=optimizer.defaults.get("eps", 1e-10),
        )
    return None


class SparseCoreTrainPipeline:
    """3-stage SparseCore/TensorCore training pipeline wrapper for PyTorch.

    This pipeline overlaps 3 training stages across consecutive batches:
      - Stage 1: SparseCore Forward (embedding lookup) on Batch t+2
      - Stage 2: TensorCore Forward + Backward (dense layers & loss) on Batch t+1
      - Stage 3: SparseCore Backward (table weights update) on Batch t

    This wrapper encapsulates:
      - Stage callback creation (`sc_fwd_stage`, `tc_stage`, `sc_bwd_stage`).
      - Raw activation unstacking and gradient stacking on TensorCore.
      - Autograd boundaries (`detach().requires_grad_()`).
      - Pipeline state machine (warmup fill, steady state, and end-of-epoch
      drain).
      - Distributed all-reduce of dense gradients and optimizer stepping.
      - Full-graph `torch.compile` invocation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Union[torch.optim.Optimizer, optimizers.Optimizer]] = None,
        criterion: Optional[Callable[..., torch.Tensor]] = None,
        *,
        embedding_layer: Optional[nn.Module] = None,
        dense_forward_fn: Optional[Callable[..., torch.Tensor]] = None,
        compile: bool = True,
        **compile_kwargs: Any,
    ):
        """Initializes the SparseCoreTrainPipeline.

        Args:
          model: The user PyTorch model.
          optimizer: Dense parameters optimizer. If provided, `optimizer.step()` and
            `optimizer.zero_grad()` will be called when valid outputs are produced.
          criterion: Loss function callable (e.g. `nn.CrossEntropyLoss()`).
          embedding_layer: The SparseCore embedding collection layer. If None,
            automatically discovered from `model.embedding_layer` or submodules.
          dense_forward_fn: Callable executing the dense layers from unstacked
            embedding activations. If None, automatically detected from
            `model.dense_forward`.
          compile: If True, compiles the steady-state 3-stage pipeline step using
            `torch.compile(self._steady_state_step, **compile_options)`.
          **compile_kwargs: Additional keyword arguments forwarded to
            `torch.compile` (e.g. `backend`, `fullgraph`, `dynamic`). Defaults to
            `fullgraph=True`, `dynamic=False`.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.compile = compile

        # Resolve embedding layer
        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        elif hasattr(model, "embedding_layer"):
            self.embedding_layer = getattr(model, "embedding_layer")
        else:
            candidates = [
                m
                for m in model.modules()
                if isinstance(
                    m,
                    (
                        SparseCoreFusedEmbeddingBagCollection,
                        SparseCoreFusedEmbeddingCollection,
                    ),
                )
            ]
            if candidates:
                self.embedding_layer = candidates[0]
            elif hasattr(model, "sc_forward"):
                self.embedding_layer = model
            else:
                raise ValueError(
                    "Could not automatically detect embedding_layer on model. "
                    "Please pass embedding_layer explicitly."
                )

        self._unstack_fn = getattr(self.embedding_layer, "unstack_activations", None)

        # Resolve dense forward callable
        if dense_forward_fn is not None:
            self.dense_forward_fn = dense_forward_fn
        elif hasattr(model, "dense_forward"):
            self.dense_forward_fn = getattr(model, "dense_forward")
        elif model != self.embedding_layer:
            self.dense_forward_fn = model
        else:
            raise ValueError(
                "Could not detect dense_forward on model. Please define a"
                " `dense_forward` method on your model or pass `dense_forward_fn`."
            )

        # Determine whether dense_forward_fn accepts dense features (e.g. (embeddings, dense_features))
        sig = inspect.signature(self.dense_forward_fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self._dense_forward_takes_dense = len(params) >= 2

        # Collect dense parameters (all model parameters not belonging to embedding_layer)
        emb_params = (
            set(self.embedding_layer.parameters())
            if self.embedding_layer != model
            else set((self._get_embedding_tables() or {}).values())
        )
        self.dense_params_dict: Dict[str, nn.Parameter] = {
            name: p
            for name, p in model.named_parameters()
            if p not in emb_params and p.requires_grad
        }
        self.dense_params = list(self.dense_params_dict.values())
        self.dense_param_names = list(self.dense_params_dict.keys())

        # Create dense module wrapper for functional calls
        self._dense_module = _ModelDenseWrapper(model, self.dense_forward_fn)

        # Initialize functional dense optimizer
        self.optimizer = optimizer
        self.dense_optimizer: Optional[optimizers.Optimizer] = _adapt_dense_optimizer(
            optimizer
        )
        if self.dense_optimizer is not None and self.dense_params_dict:
            self.dense_param_group: Optional[Any] = (
                self.dense_optimizer.init_param_group(self.dense_params_dict)
            )
        else:
            self.dense_param_group = None

        # Steady-state pipeline compilation
        if compile:
            os.environ.setdefault(
                "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "0"
            )
            if hasattr(torch._dynamo.config, "trace_autograd_ops"):
                torch._dynamo.config.trace_autograd_ops = True
            compile_options: Dict[str, Any] = {
                "fullgraph": True,
                "dynamic": False,
            }
            compile_options.update(compile_kwargs)
            logging.info(
                "Compiling steady-state SparseCore/TensorCore pipeline kernel"
                " (compile_options=%s)...",
                compile_options,
            )
            self._steady_state_fn = torch.compile(
                self._steady_state_step,
                **compile_options,
            )
        else:
            self._steady_state_fn = self._steady_state_step

        self.reset()

    def _get_embedding_tables(self) -> Optional[Dict[str, torch.Tensor]]:
        """Returns the current embedding tables dictionary if available."""
        if hasattr(self.embedding_layer, "table_weights"):
            tb = getattr(self.embedding_layer, "table_weights")
            if isinstance(tb, dict):
                return {k: v for k, v in tb.items()}
        if hasattr(self.embedding_layer, "_stacked_configs") and hasattr(
            self.embedding_layer, "_get_table_weight"
        ):
            return {
                sc.stack_name: self.embedding_layer._get_table_weight(sc.stack_name)
                for sc in self.embedding_layer._stacked_configs
            }
        if hasattr(self.embedding_layer, "table"):
            return {"table": getattr(self.embedding_layer, "table")}
        return None

    def reset(
        self,
        preserve_embedding_tables: bool = False,
        preserve_dense_params: bool = False,
    ) -> None:
        """Resets the pipeline state for a new epoch or training run."""
        self._is_draining = False
        self._drain_iter = None
        self._step_count = 0
        # Steady state pipeline buffers
        self._sparse_inputs_t = None
        self._embedding_gradients_t = None
        self._embedding_activations_t1 = None
        self._dense_inputs_t1 = None
        self._sparse_features_t1 = None
        if not preserve_embedding_tables or self._embedding_tables is None:
            self._embedding_tables = self._get_embedding_tables()
        if (
            not preserve_dense_params
            and self.dense_optimizer is not None
            and self.dense_params_dict
        ):
            self.dense_param_group = self.dense_optimizer.init_param_group(
                self.dense_params_dict
            )

    def sync_model_weights(self) -> None:
        """Copies trained dense parameters from dense_param_group back into model parameters."""
        if self.dense_param_group is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.dense_param_group.params:
                        param.copy_(self.dense_param_group.params[name])
        for opt_state_dict_name in ("accumulators", "momentums", "velocities"):
            opt_dict = getattr(self.embedding_layer, opt_state_dict_name, None)
            if isinstance(opt_dict, nn.ParameterDict):
                for k, v in list(opt_dict._parameters.items()):
                    if v is not None and not isinstance(v, nn.Parameter):
                        opt_dict._parameters[k] = nn.Parameter(v, requires_grad=False)

    def _update_dense_params(self, new_pg: Optional[Any]) -> None:
        """Updates dense parameter group or performs eager optimizer step."""
        if new_pg is not None:
            self.dense_param_group = new_pg
        else:
            self._optimizer_step()
        if not self.compile:
            self.sync_model_weights()

    def _sc_fwd_stage(
        self,
        features: Any,
        embedding_tables: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Any, None]:
        raw_acts = self.embedding_layer.sc_forward(
            features, embedding_tables=embedding_tables
        )
        detached_acts = _detach_structure(raw_acts)
        return detached_acts, None

    def _tc_stage(
        self,
        embedding_activations: Any,
        dense_inputs: Any,
        dense_param_group: Optional[Any] = None,
        sc_fwd_aux: Optional[Any] = None,
    ) -> Tuple[Any, torch.Tensor, Optional[Any]]:
        """Stage 2: TensorCore Forward, Backward, All-Reduce, and Dense Optimizer."""
        del sc_fwd_aux
        dense_in = dense_inputs

        # Unstack raw activations on TensorCore
        if self._unstack_fn is not None:
            unstacked = self._unstack_fn(embedding_activations)
        else:
            unstacked = embedding_activations

        # Enable gradient tracking on activations for autograd
        if isinstance(unstacked, KeyedTensor):
            act_values = unstacked.values().detach().requires_grad_(True)
            emb_var = KeyedTensor(
                keys=unstacked.keys(),
                values=act_values,
                length_per_key=unstacked.length_per_key(),
            )
            act_tensors = [act_values]
            reconstruct_grads = lambda grad_list: grad_list[0]
        elif isinstance(unstacked, dict):
            emb_var = {}
            act_tensors = []
            keys_order = []
            for k, v in unstacked.items():
                if isinstance(v, JaggedTensor):
                    act_t = v.values().detach().requires_grad_(True)
                    act_tensors.append(act_t)
                    keys_order.append(k)
                    emb_var[k] = JaggedTensor(
                        values=act_t,
                        lengths=v.lengths(),
                        weights=v.weights_or_none(),
                    )
                elif isinstance(v, torch.Tensor):
                    act_t = v.detach().requires_grad_(True)
                    act_tensors.append(act_t)
                    keys_order.append(k)
                    emb_var[k] = act_t
                else:
                    emb_var[k] = v
            reconstruct_grads = lambda grad_list: {
                k: g for k, g in zip(keys_order, grad_list)
            }
        elif isinstance(unstacked, torch.Tensor):
            act_values = unstacked.detach().requires_grad_(True)
            emb_var = act_values
            act_tensors = [act_values]
            reconstruct_grads = lambda grad_list: grad_list[0]
        else:
            emb_var = unstacked
            act_tensors = []
            reconstruct_grads = lambda grad_list: None

        # Resolve dense features and loss target
        if isinstance(dense_in, (tuple, list)) and len(dense_in) == 2:
            # Format (dense_features, target) for models like DLRM
            dense_feat, target = dense_in
            if self._dense_forward_takes_dense:
                forward_args = (emb_var, dense_feat)
            else:
                forward_args = (emb_var,)
            loss_target = target
        elif (
            isinstance(dense_in, dict)
            and "dense_features" in dense_in
            and "labels" in dense_in
        ):
            if self._dense_forward_takes_dense:
                forward_args = (emb_var, dense_in["dense_features"])
            else:
                forward_args = (emb_var,)
            loss_target = dense_in["labels"]
        else:
            forward_args = (emb_var,)
            loss_target = dense_in

        # Prepare dense parameters for forward
        if dense_param_group is not None:
            params_to_use = {
                k: v.detach().requires_grad_(True)
                for k, v in dense_param_group.params.items()
            }
            wrapper_params = {"_model." + k: v for k, v in params_to_use.items()}
            logits = torch.func.functional_call(
                self._dense_module,
                wrapper_params,
                forward_args,
                strict=False,
                tie_weights=False,
            )
        else:
            params_to_use = None
            logits = self.dense_forward_fn(*forward_args)

        # Compute Loss
        if self.criterion is not None:
            if isinstance(loss_target, (tuple, list)):
                loss = self.criterion(logits, *loss_target)
            elif isinstance(loss_target, dict):
                loss = self.criterion(logits, **loss_target)
            else:
                loss = self.criterion(logits, loss_target)
        else:
            loss = logits

        # Functional Backward via torch.autograd.grad
        if params_to_use is not None:
            dense_target_tensors = list(params_to_use.values())
            inputs_to_grad = act_tensors + dense_target_tensors
        else:
            inputs_to_grad = act_tensors + self.dense_params

        if inputs_to_grad:
            all_grads = torch.autograd.grad(
                outputs=loss,
                inputs=inputs_to_grad,
                retain_graph=False,
                allow_unused=True,
            )
            act_grads = all_grads[: len(act_tensors)]
            dense_grads = all_grads[len(act_tensors) :]
            grads = reconstruct_grads(act_grads)
        else:
            dense_grads = []
            grads = None

        # In-graph distributed all-reduce and dense optimizer step
        if dense_param_group is not None and self.dense_optimizer is not None:
            dense_grads_dict = dict(zip(dense_param_group.params.keys(), dense_grads))

            if dist.is_initialized() and dist.get_world_size() > 1:
                world_size = float(dist.get_world_size())
                for k, g in dense_grads_dict.items():
                    if g is not None:
                        is_1d = g.ndim == 1
                        if is_1d:
                            g = g.unsqueeze(0)
                        dist.all_reduce(g, op=dist.ReduceOp.SUM)
                        g = g / world_size
                        if is_1d:
                            g = g.squeeze(0)
                        dense_grads_dict[k] = g
            new_dense_param_group = self.dense_optimizer.step(
                dense_param_group, dense_grads_dict
            )
            new_dense_param_group.params = {
                k: v.detach() for k, v in new_dense_param_group.params.items()
            }
        else:
            new_dense_param_group = None
            if dense_grads:
                for param, grad in zip(self.dense_params, dense_grads):
                    if grad is not None:
                        param.grad = grad

        # Extract Gradients and Stack for SparseCore Backward dispatch
        if hasattr(self.embedding_layer, "stack_gradients") and grads is not None:
            raw_gradients = self.embedding_layer.stack_gradients(grads)
        else:
            raw_gradients = grads

        detached_grads = _detach_structure(raw_gradients)
        return detached_grads, loss.detach(), new_dense_param_group

    def _sc_bwd_stage(
        self,
        features: Any,
        embedding_gradients: Any,
        embedding_tables: Optional[Dict[str, torch.Tensor]] = None,
        tc_aux: Optional[Any] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        del tc_aux
        if embedding_gradients is None or not hasattr(
            self.embedding_layer, "sc_backward"
        ):
            return None
        return self.embedding_layer.sc_backward(
            features,
            embedding_gradients,
            embedding_tables=embedding_tables,
        )

    def _optimizer_step(self) -> None:
        """Performs all-reduce across distributed ranks and steps the dense optimizer."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            for param in self.dense_params:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= dist.get_world_size()

        if isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _wrap_batch(self, batch: Any) -> Tuple[Any, Any]:
        """Extracts (sparse_inputs, dense_inputs) from a user batch."""
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                # Format (dense_features, labels, sparse_features) common in DLRM
                dense_feat, labels, sparse_feat = batch
                return sparse_feat, (dense_feat, labels)
            if len(batch) >= 2:
                return batch[0], batch[1]
            return batch[0], None
        if isinstance(batch, dict):
            sparse_in = batch.get("sparse_inputs", batch.get("features", None))
            if "dense_inputs" in batch:
                dense_in = batch["dense_inputs"]
            elif "dense_features" in batch and "labels" in batch:
                dense_in = (batch["dense_features"], batch["labels"])
            elif "dense_features" in batch:
                dense_in = batch["dense_features"]
            else:
                dense_in = batch.get("labels", None)
            return sparse_in, dense_in
        if hasattr(batch, "sparse_inputs") and hasattr(batch, "dense_inputs"):
            return batch.sparse_inputs, batch.dense_inputs
        return batch, None

    def _steady_state_step(
        self,
        sparse_inputs_t: Any,
        embedding_gradients_t: Any,
        embedding_tables_t: Optional[Dict[str, torch.Tensor]],
        sparse_inputs_t2: Any,
        embedding_activations_t1: Any,
        dense_inputs_t1: Any,
        dense_param_group_t1: Optional[Any] = None,
    ) -> Tuple[
        Any,
        Any,
        torch.Tensor,
        Optional[Dict[str, torch.Tensor]],
        Optional[Any],
    ]:
        """Pure tensor 3-stage steady-state kernel (Zero Graph Break).

        Stages run concurrently on hardware:
          - SparseCore Backward: Batch t (functional update returning new_tables)
          - SparseCore Forward:  Batch t+2 (functional lookup using current tables)
          - TensorCore Fwd+Bwd+AllReduce+Optimizer: Batch t+1

        Args:
          sparse_inputs_t: Sparse inputs for batch t.
          embedding_gradients_t: Embedding gradients for batch t.
          embedding_tables_t: Current embedding tables for batch t.
          sparse_inputs_t2: Sparse inputs for batch t+2.
          embedding_activations_t1: Embedding activations for batch t+1.
          dense_inputs_t1: Dense inputs for batch t+1.
          dense_param_group_t1: Optional dense parameter group for batch t+1.

        Returns:
          Tuple of (acts_t2, grads_t1, loss_t1, new_tables, new_dense_param_group).
        """
        with torch.fx.traceback.annotate({_SPMD_SAFE_METADATA_KEY: True}):
            # SparseCore Backward on Batch t (functional)
            new_tables = self._sc_bwd_stage(
                sparse_inputs_t,
                embedding_gradients_t,
                embedding_tables=embedding_tables_t,
            )

            # TensorCore Forward + Backward + AllReduce + Optimizer on Batch t+1
            raw_grads_t1, loss_t1, new_dense_param_group = self._tc_stage(
                embedding_activations=embedding_activations_t1,
                dense_inputs=dense_inputs_t1,
                dense_param_group=dense_param_group_t1,
            )

            # SparseCore Forward on Batch t+2 (using current or updated tables)
            raw_acts_t2, _ = self._sc_fwd_stage(
                sparse_inputs_t2,
                embedding_tables=(
                    new_tables if new_tables is not None else embedding_tables_t
                ),
            )
            return (
                raw_acts_t2,
                raw_grads_t1,
                loss_t1,
                new_tables,
                new_dense_param_group,
            )

    def step(
        self,
        sparse_inputs: KeyedSparseCorePreprocessedInput,
        dense_inputs: Any,
    ) -> Optional[torch.Tensor]:
        """Manual step API for feeding batches explicitly step-by-step.

        Args:
          sparse_inputs: Preprocessed sparse input features.
          dense_inputs: Dense tensors / labels for the batch.

        Returns:
          Loss tensor if TC executed for a previous batch, or None during warmup.
        """
        if self._step_count == 0:
            # Warmup step 0: SC FWD on batch 0
            acts0, _ = self._sc_fwd_stage(
                sparse_inputs, embedding_tables=self._embedding_tables
            )
            self._embedding_activations_t1 = acts0
            self._dense_inputs_t1 = dense_inputs
            self._sparse_features_t1 = sparse_inputs
            self._step_count = 1
            return None

        if self._step_count == 1:
            # Warmup step 1: SC FWD on batch 1, TC on batch 0
            acts1, _ = self._sc_fwd_stage(
                sparse_inputs, embedding_tables=self._embedding_tables
            )
            grads0, loss0, new_pg0 = self._tc_stage(
                self._embedding_activations_t1,
                self._dense_inputs_t1,
                dense_param_group=self.dense_param_group,
            )
            self._update_dense_params(new_pg0)

            self._sparse_inputs_t = self._sparse_features_t1
            self._embedding_gradients_t = grads0
            self._embedding_activations_t1 = acts1
            self._dense_inputs_t1 = dense_inputs
            self._sparse_features_t1 = sparse_inputs
            self._step_count = 2
            return loss0

        # Steady-state step
        acts_t2, grads_t1, loss_t1, new_tables, new_pg = self._steady_state_fn(
            self._sparse_inputs_t,
            self._embedding_gradients_t,
            self._embedding_tables,
            sparse_inputs,
            self._embedding_activations_t1,
            self._dense_inputs_t1,
            self.dense_param_group,
        )
        self._update_dense_params(new_pg)

        if new_tables is not None:
            self._embedding_tables = new_tables

        self._sparse_inputs_t = self._sparse_features_t1
        self._embedding_gradients_t = grads_t1
        self._embedding_activations_t1 = acts_t2
        self._dense_inputs_t1 = dense_inputs
        self._sparse_features_t1 = sparse_inputs
        self._step_count += 1
        return loss_t1

    def drain(self) -> Iterator[torch.Tensor]:
        """Flushes remaining in-flight pipeline stages (TC and SC BWD) at the end of an epoch.

        Yields:
          Loss tensor for in-flight batches processed during draining.
        """
        if self._step_count == 0:
            return

        # Drain step 1: TC on last batch + SC BWD on batch N-2 (if applicable)
        if self._embedding_activations_t1 is not None:
            new_tables = self._sc_bwd_stage(
                self._sparse_inputs_t,
                self._embedding_gradients_t,
                embedding_tables=self._embedding_tables,
            )
            if new_tables is not None:
                self._embedding_tables = new_tables

            grads_last, loss_last, new_pg_last = self._tc_stage(
                self._embedding_activations_t1,
                self._dense_inputs_t1,
                dense_param_group=self.dense_param_group,
            )
            self._update_dense_params(new_pg_last)
            self._sparse_inputs_t = self._sparse_features_t1
            self._embedding_gradients_t = grads_last
            yield loss_last

        # Drain step 2: SC BWD on last batch
        if (
            self._sparse_inputs_t is not None
            and self._embedding_gradients_t is not None
        ):
            new_tables = self._sc_bwd_stage(
                self._sparse_inputs_t,
                self._embedding_gradients_t,
                embedding_tables=self._embedding_tables,
            )
            if new_tables is not None:
                self._embedding_tables = new_tables

        self.sync_model_weights()
        self.reset(preserve_embedding_tables=True, preserve_dense_params=True)

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Executes one training step on hardware.

        Args:
          dataloader_iter: Iterator over the training dataloader.

        Returns:
          Loss tensor if TC executed for a previous batch, or None during warmup
          step 0.

        Raises:
          StopIteration: When all input batches and pipeline drain stages are
          finished.
        """
        if not self._is_draining:
            batch = next(dataloader_iter, None)
            if batch is not None:
                sparse, dense = self._wrap_batch(batch)
                return self.step(sparse, dense)
            self._is_draining = True
            self._drain_iter = self.drain()

        if self._drain_iter is None:
            raise StopIteration
        return next(self._drain_iter)

    def iterate(
        self, dataloader_or_iter: Union[Iterable[Any], Iterator[Any]]
    ) -> Iterator[torch.Tensor]:
        """Iterates through all batches in the dataloader yielding valid losses."""
        self.reset(preserve_embedding_tables=True, preserve_dense_params=True)
        for batch in dataloader_or_iter:
            sparse, dense = self._wrap_batch(batch)
            loss = self.step(sparse, dense)
            if loss is not None:
                yield loss

        yield from self.drain()
