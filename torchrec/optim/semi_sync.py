#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Collection, Dict, Mapping, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.optim.optimizer import ParamsT
from torchrec.optim.keyed import KeyedOptimizer

logger: logging.Logger = logging.getLogger(__name__)

SEMI_SYNC_GLOBAL_OPTIM_KEY: str = "semi_sync_global_optim"
SEMI_SYNC_LOCAL_OPTIM_KEY: str = "semi_sync_local_optim"

SEMI_SYNC_GLOBAL_STATE_KEY: str = "semi_sync_global_"
SEMI_SYNC_LOCAL_STATE_KEY: str = "semi_sync_local_"

# Step counter keys for special parameters
SEMISYNC_GLOBAL_STEP_COUNTER_KEY: str = "__semisync_global_step_counter__"
SEMISYNC_LOCAL_STEP_COUNTER_KEY: str = "__semisync_local_step_counter__"


class SemisyncOptimizer(KeyedOptimizer):
    """
    Semi-Synchronous Optimizer. Implements semi synchronous training for cross-regional
    distributed training for large scale recommendation models.

    This optimizer:
    1. Takes a local optimizer (e.g., Shampoo or Adam wrapped in KeyedOptimizer)
    2. Takes a global optimizer (e.g., DiLoCo)
    3. Performs local steps on the local optimizer
    4. Periodically performs global aggregation steps using the logic of global optimizer
    5. Exposes combined state from both optimizers through KeyedOptimizer interface

    Args:
        global_params (ParamsT): Global parameters for SemisyncOptimizer
        optimizer (KeyedOptimizer): Local optimizer (typically Shampoo or Adam, wrapped in KeyedOptimizer)
        global_optimizer (KeyedOptimizer): Global optimizer (such as DiLoCo, also wrapper in KeyedOptimizer)
        num_local_steps (int): Number of local steps before global sync
        semi_sync_worker_shard_group (Optional[dist.ProcessGroup]): Process group for metrics logging
        offload_global_model (bool): Offload global model parameters onto CPU (Default: False)
        non_blocking (bool): GPU <-> CPU communication is non blocking (Default: False)
    """

    def __init__(
        self,
        global_params: ParamsT,
        optimizer: KeyedOptimizer,
        global_optimizer: KeyedOptimizer,
        num_local_steps: int = 16,
        semi_sync_worker_shard_group: Optional[dist.ProcessGroup] = None,
        offload_global_model: bool = False,
        non_blocking: bool = False,
    ) -> None:
        # Store the optimizers
        self._optimizer: KeyedOptimizer = optimizer
        self._global_optimizer: KeyedOptimizer = global_optimizer

        assert global_params is not None, "global params must be provided"
        # Determine parameters for global optimizer
        global_params_list = list(global_params)
        self._worker_model_params: list[torch.Tensor] = (
            # pyre-ignore
            [param for pgroup in global_params_list for param in pgroup["params"]]
            if isinstance(global_params_list[0], dict)
            else global_params_list
        )

        # Semi-sync configuration

        self._num_local_steps: int = num_local_steps
        self._local_step_counter: torch.Tensor = torch.tensor(
            0, dtype=torch.int64, device="cpu"
        )
        self._global_step_counter: torch.Tensor = torch.tensor(
            0, dtype=torch.int64, device="cpu"
        )

        # Store process groups info for metrics logging
        self._worker_shard_group: Optional[dist.ProcessGroup] = (
            semi_sync_worker_shard_group
        )
        self._offload_global_model: bool = offload_global_model
        self._non_blocking: bool = non_blocking
        self.defaults: Dict[str, Any] = {"_save_param_groups": False}

        logger.info(
            "Instantiated SemisyncOptimizer with: "
            f"num_local_steps={self._num_local_steps}, "
            f"worker_model_params_count={len(self._worker_model_params)}, "
            f"offload_global_model={offload_global_model}, "
            f"non_blocking={non_blocking}, "
            f"self.defaults={self.defaults}, "
            f"local_optimizer={type(self._optimizer)}, "
            f"global_optimizer={type(self._global_optimizer)}"
        )

    @property
    def param_groups(self) -> Collection[Mapping[str, Any]]:
        """
        Combine param_groups from both local and global optimizers.
        """
        return [
            param_group
            for opt in [self._optimizer, self._global_optimizer]
            for param_group in opt.param_groups
        ]

    @property
    def params(self) -> Mapping[str, Union[torch.Tensor, ShardedTensor]]:
        """
        Combine params from both local and global optimizers.
        If param_key already exists, verify that local and global point to the same parameter tensor.
        """
        ret = dict(self._optimizer.params)

        # Add global params with collision check
        for param_key, param in self._global_optimizer.params.items():
            if param_key in ret:
                assert (
                    ret[param_key] is param
                ), f"Parameter key '{param_key}' exists in both optimizers but points to different tensors"
            else:
                ret[param_key] = param

        # Add step counters as special parameters
        ret[SEMISYNC_GLOBAL_STEP_COUNTER_KEY] = self._global_step_counter
        ret[SEMISYNC_LOCAL_STEP_COUNTER_KEY] = self._local_step_counter

        return ret

    @property
    def state(self) -> Mapping[torch.Tensor, Any]:
        """
        Combine state from both local and global optimizers.
            - For tensors that exist in both optimizers, prefix the state keys to avoid conflicts.
            - Step counters are embedded into every global parameter's state.
        """
        # Start with prefixed local states
        ret = {
            param: {
                f"{SEMI_SYNC_LOCAL_STATE_KEY}{key}": value
                for key, value in local_state.items()
            }
            for param, local_state in self._optimizer.state.items()
        }

        # Add prefixed global states and step counters
        for param, global_state in self._global_optimizer.state.items():
            prefixed_global_state = {
                f"{SEMI_SYNC_GLOBAL_STATE_KEY}{key}": value
                for key, value in global_state.items()
            }

            if param in ret:
                ret[param].update(prefixed_global_state)
            else:
                ret[param] = prefixed_global_state

            # Add step counters to all global params
            ret[param].update(
                {
                    SEMISYNC_GLOBAL_STEP_COUNTER_KEY: self._global_step_counter,
                    SEMISYNC_LOCAL_STEP_COUNTER_KEY: self._local_step_counter,
                }
            )

        return ret

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        """
        Perform semi-sync optimization step:
            1. Always perform local optimizer step
            2. At every num_local_steps, perform global optimizer step

        TODO:
        See more details in D80683853, v8
            - metrics: add _metrics_logger for global grad and pseudo-gradient statistics
            - cpu cache: add GlobalModelParamsCache with non_blocking_transfer for memory efficiency
            - gradient clipping: add gradient clipping for global optimizer if needed
        """
        self._local_step_counter.add_(1)
        trigger_global_step = self._local_step_counter.item() > 0 and (
            self._local_step_counter.item() % self._num_local_steps == 0
        )

        # Perform local optimizer step (delegate to the local optimizer)
        self._optimizer.step(closure)

        # Perform global model sync every num_local_steps
        if trigger_global_step:
            self._global_step_counter.add_(1)

            # Step 0: Release the gradient buffer to reduce memory consumption
            self._global_optimizer.zero_grad()

            # Step 1: perform global optimizer step
            # pyre-ignore
            self._global_optimizer._optimizer.global_step(
                self._local_step_counter, closure
            )

            logger.info(
                f"Finished global optimizer step {self._global_step_counter.item()} "
                f"(after {self._local_step_counter.item()} local steps)"
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)
        self._global_optimizer.zero_grad(set_to_none=set_to_none)

    def post_load_state_dict(self) -> None:
        """
        Called after KeyedOptimizer.load_state_dict() completes.
        This is where we separate the prefixed combined state back to individual optimizers.
        """
        logger.info(
            "SemisyncOptimizer: post_load_state_dict called - separating prefixed combined state"
        )

        # Extract step counters from any param states that contain them
        combined_state = dict(self.state)
        self._post_load_state_dict_step_counter(combined_state)

        # Separate states using dictionary comprehensions
        local_tensors = set(self._optimizer.state.keys())
        global_tensors = set(self._global_optimizer.state.keys())

        local_state = {
            param: self._extract_prefixed_state(param_state, SEMI_SYNC_LOCAL_STATE_KEY)
            for param, param_state in combined_state.items()
            if param in local_tensors
            and any(k.startswith(SEMI_SYNC_LOCAL_STATE_KEY) for k in param_state)
        }

        global_state = {
            param: self._extract_prefixed_state(param_state, SEMI_SYNC_GLOBAL_STATE_KEY)
            for param, param_state in combined_state.items()
            if param in global_tensors
            and any(k.startswith(SEMI_SYNC_GLOBAL_STATE_KEY) for k in param_state)
        }

        # Update optimizer states
        for opt, state, name in [
            (self._optimizer, local_state, "local"),
            (self._global_optimizer, global_state, "global"),
        ]:
            if state:
                opt.state.clear()  # pyre-ignore
                opt.state.update(state)  # pyre-ignore
                logger.info(
                    f"SemisyncOptimizer: Set state on {name} optimizer for {len(state)} parameters"
                )

        # Call post_load_state_dict on individual optimizers if they support it
        for opt in [self._optimizer, self._global_optimizer]:
            if hasattr(opt, "post_load_state_dict"):
                opt.post_load_state_dict()

    def save_param_groups(self, save: bool) -> None:
        self.defaults["_save_param_groups"] = save
        self._optimizer.save_param_groups(save)
        self._global_optimizer.save_param_groups(save)

    def __repr__(self) -> str:
        ret = []
        ret.append(f"{SEMI_SYNC_LOCAL_OPTIM_KEY}: {self._optimizer.__repr__()}")
        ret.append(f"{SEMI_SYNC_GLOBAL_OPTIM_KEY}: {self._global_optimizer.__repr__()}")
        return ", ".join(ret)

    def set_optimizer_step(self, step: int) -> None:
        for opt in [self._optimizer, self._global_optimizer]:
            if hasattr(opt, "set_optimizer_step"):
                # pyre-ignore [16]: Undefined attribute [16]: `KeyedOptimizer` has no attribute `set_optimizer_step`.
                opt.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:

        for opt in [self._optimizer, self._global_optimizer]:
            if hasattr(opt, "update_hyper_parameters"):
                # pyre-ignore [16]: Undefined attribute [16]: `KeyedOptimizer` has no attribute `update_hyper_parameters`.
                opt.update_hyper_parameters(params_dict)

    @staticmethod
    def _extract_prefixed_state(
        param_state: Dict[str, Any], prefix: str
    ) -> Dict[str, Any]:
        """
        Extract state keys with a specific prefix and remove the prefix.

        Args:
            param_state: Parameter state dictionary
            prefix: Prefix to extract and remove

        Returns:
            Dictionary with prefix removed from matching keys
        """
        return {
            key[len(prefix) :]: value
            for key, value in param_state.items()
            if key.startswith(prefix)
        }

    def _post_load_state_dict_step_counter(
        self,
        combined_state: Dict[torch.Tensor, Any],
    ) -> None:
        """Extract step counters from any param states that contain them."""
        found = {"global": False, "local": False}

        for param_state in combined_state.values():
            if not found["global"] and SEMISYNC_GLOBAL_STEP_COUNTER_KEY in param_state:
                self._global_step_counter = param_state[
                    SEMISYNC_GLOBAL_STEP_COUNTER_KEY
                ]
                found["global"] = True
            if not found["local"] and SEMISYNC_LOCAL_STEP_COUNTER_KEY in param_state:
                self._local_step_counter = param_state[SEMISYNC_LOCAL_STEP_COUNTER_KEY]
                found["local"] = True
            if all(found.values()):
                break

        missing = [k for k, v in found.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing {' and '.join(missing)} step counter(s) in checkpoint."
            )
