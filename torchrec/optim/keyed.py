#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import torch
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor

OptimizerFactory = Callable[[List[Union[torch.Tensor, ShardedTensor]]], optim.Optimizer]


class KeyedOptimizer(optim.Optimizer):
    """
    Takes a dict of parameters and exposes state_dict by parameter key.

    This implementation is much stricter than the one in torch.Optimizer:
    it requires implementations to fully initialize their state during first optimization iteration,
    and it prohibits loading an empty state into already initialized KeyedOptimizer and vise versa.

    It also doesn't expose param_groups in state_dict() by default
    Old behavior can be switch on by setting save_param_groups flag.
    The reason is that during distributed training not all parameters are present on all ranks
    and we identify param_group by its parameters.
    In addition to that, param_groups are typically re-set during training initialization,
    so it makes little sense to save them as a part of the state to begin with.
    """

    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        state: Mapping[Any, Any],
        param_groups: Collection[Mapping[str, Any]],
    ) -> None:
        torch._C._log_api_usage_once(f"torchrec.optim.{self.__class__.__name__}")

        # TODO: remove these and call super().__init__()
        # super().__init__ calls add_param_group, which we've explicitly marked as not implemented.
        # However, we need to ensure that all Optimizer member variables are created.
        # pyre-ignore
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        # pyre-ignore
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

        self.state: Mapping[Any, Any] = state
        self.param_groups: Collection[Mapping[str, Any]] = param_groups
        self.params = params
        self.defaults: Dict[str, Any] = {"_save_param_groups": False}

        params_set = set(params.values())
        non_param_state_keys = [key for key in self.state if key not in params_set]
        if len(non_param_state_keys) > 0:
            raise ValueError(
                "All state keys must be params. The following keys are not: {}.".format(
                    non_param_state_keys
                )
            )

    @staticmethod
    def _extract_state_dict_content(
        input_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Converts nested dictionary with objects with state dict functionality.

        Args:
            input_dict (Dict[str, Any]): Nested dictionary containing objects with
                state dict functionality.

        Output:
            output_dict (Dict[str, Any]): Nested dictionary where the terminal values
                cannot have state dict functionality.

        """
        result = {}
        for k, v in input_dict.items():
            if isinstance(v, dict):
                result[k] = KeyedOptimizer._extract_state_dict_content(v)
            elif hasattr(v, "state_dict") and callable(v.state_dict):
                result[k] = v.state_dict()
            else:
                result[k] = v
        return result

    @staticmethod
    def _update_param_state_dict_object(
        current_param_state_dict: Dict[str, Any],
        param_state_dict_to_load: Dict[str, Any],
        parent_keys: List[Union[str, int, float, bool, None]],
    ) -> None:
        # Import at function level to avoid circular dependency.
        from torchrec.distributed.shards_wrapper import LocalShardsWrapper

        for k, v in current_param_state_dict.items():
            new_v = param_state_dict_to_load[k]
            parent_keys.append(k)

            if isinstance(v, dict):
                KeyedOptimizer._update_param_state_dict_object(
                    v,
                    new_v,
                    parent_keys,
                )
            elif hasattr(v, "load_state_dict") and callable(v.load_state_dict):
                v.load_state_dict(new_v)
            elif isinstance(v, ShardedTensor):
                assert isinstance(new_v, ShardedTensor)
                num_shards = len(v.local_shards())
                num_new_shards = len(new_v.local_shards())
                if num_shards != num_new_shards:
                    raise ValueError(
                        f"Different number of shards {num_shards} vs {num_new_shards} for the path of {json.dumps(parent_keys)}"
                    )
                for shard, new_shard in zip(v.local_shards(), new_v.local_shards()):
                    # CW sharding can create different size tensors
                    if shard.tensor.shape != new_shard.tensor.shape:
                        # Resize the tensor to match the new shape
                        shard.tensor.resize_(new_shard.tensor.shape)
                    shard.tensor.detach().copy_(new_shard.tensor)
            elif isinstance(v, DTensor):
                assert isinstance(new_v, DTensor)
                if isinstance(v.to_local(), LocalShardsWrapper):
                    assert isinstance(new_v.to_local(), LocalShardsWrapper)
                    num_shards = len(v.to_local().local_shards())  # pyre-ignore[16]
                    num_new_shards = len(new_v.to_local().local_shards())
                    if num_shards != num_new_shards:
                        raise ValueError(
                            f"Different number of shards {num_shards} vs {num_new_shards} for the path of {json.dumps(parent_keys)}"
                        )
                    for shard, new_shard in zip(
                        v.to_local().local_shards(), new_v.to_local().local_shards()
                    ):
                        shard.detach().copy_(new_shard)
                else:
                    assert isinstance(new_v.to_local(), torch.Tensor)
                    v.detach().copy_(new_v)
            elif isinstance(v, torch.Tensor):
                v.detach().copy_(new_v)
            else:
                current_param_state_dict[k] = deepcopy(new_v)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returned state and param_groups will contain parameter keys
        instead of parameter indices in torch.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.

        Can also handle classes and supported data structures that follow the PyTorch stateful
        protocol.
        """

        param_groups = self.param_groups
        params = self.params
        param_to_key = {param: key for key, param in params.items()}

        ret_state = {
            param_to_key[param]: self._extract_state_dict_content(param_state)
            for param, param_state in self.state.items()
        }

        ret_groups = []
        for group in param_groups:
            param_keys = []
            for param in group["params"]:
                param_keys.append(param_to_key[param])
            ret_group = {"params": sorted(param_keys)}
            for k, v in group.items():
                if k != "params":
                    ret_group[k] = deepcopy(v)
            ret_groups.append(ret_group)

        ret: Dict[str, object] = {"state": ret_state}
        if self.defaults["_save_param_groups"]:
            ret["param_groups"] = ret_groups
        return ret

    def post_load_state_dict(self) -> None:
        pass

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        This implementation is much stricter than the one in torch.Optimizer:
        it requires implementations to fully initialize their state during first optimization iteration,
        and it prohibits loading an empty state into already initialized KeyedOptimizer and vise versa.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves usability
            * avoid state duplication by directly copying into state tensors, e.g.
              optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard if needed
              optimizer.load_state_dict(sd)  # replace param_groups
        """

        new_state = state_dict["state"]
        state = self.state
        params = self.params

        # Load state
        if len(state) != len(new_state):
            raise ValueError(
                f"Different parameter count: {len(state)} vs {len(new_state)}"
            )
        for param_key, param in params.items():
            if param not in state:
                continue
            if param_key not in new_state:
                raise ValueError(f"Parameter {param_key} not found")
            if len(state[param]) != len(new_state[param_key]):
                raise ValueError(
                    f"Different state size: {len(state[param])} vs {len(new_state[param_key])}"
                )

            KeyedOptimizer._update_param_state_dict_object(
                current_param_state_dict=state[param],
                param_state_dict_to_load=new_state[param_key],
                parent_keys=[param_key],
            )

        # Load param_groups.
        if self.defaults["_save_param_groups"]:
            new_param_groups = state_dict["param_groups"]
            param_groups = self.param_groups

            if len(param_groups) != len(new_param_groups):
                raise ValueError(
                    f"Different param_groups count: {len(param_groups)} vs {len(new_param_groups)}"
                )
            param_to_key = {param: key for key, param in params.items()}
            group_map = {}
            for group in param_groups:
                param_keys = []
                for param in group["params"]:
                    param_keys.append(param_to_key[param])
                group_map["/".join(sorted(param_keys))] = group
            new_group_map = {}
            for new_group in new_param_groups:
                param_keys = []
                for param_key in new_group["params"]:
                    param_keys.append(param_key)
                new_group_map["/".join(sorted(param_keys))] = new_group
            for group_key, group in group_map.items():
                if group_key not in new_group_map:
                    raise ValueError(f"Group {group_key} not found")
                new_group = new_group_map[group_key]
                if len(group) != len(new_group):
                    raise ValueError(
                        f"Different param_group size: {len(group)} vs {len(new_group)}"
                    )
                for k in group:
                    if k not in new_group:
                        raise ValueError(
                            f"Group key {k} not found for group {group_key}"
                        )
                    if k != "params":
                        group[k] = deepcopy(new_group[k])

        self.post_load_state_dict()

    def add_param_group(self, param_group: Any) -> None:
        raise NotImplementedError()

    def init_state(
        self,
        sparse_grad_parameter_names: Optional[Set[str]] = None,
    ) -> None:
        """
        Runs a dummy optimizer step, which allows to initialize
        optimizer state, which is typically lazy.
        This allows us to do in-place loading of optimizer state from a checkpoint.
        """
        for key, param in self.params.items():
            if param.requires_grad:
                t = torch.zeros_like(param)
                if (
                    sparse_grad_parameter_names is not None
                    and key in sparse_grad_parameter_names
                ):
                    t = t.to_sparse()
                param.grad = torch.autograd.Variable(t)
        self.step(closure=None)

    def save_param_groups(self, save: bool) -> None:
        self.defaults["_save_param_groups"] = save

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__


class CombinedOptimizer(KeyedOptimizer):
    """
    Combines multiple KeyedOptimizers into one.

    Meant to combine different optimizers for different submodules
    """

    def __init__(
        self, optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]]
    ) -> None:
        self.defaults: Dict[str, Any] = {}
        # Append empty optimizer key if not passed.
        self._optims: List[Tuple[str, KeyedOptimizer]] = []
        for key_value in optims:
            if isinstance(key_value, KeyedOptimizer):
                key_value = ("", key_value)
            self._optims.append(key_value)

        all_keys: Set[str] = set()
        self.defaults["_save_param_groups"] = (
            False
            if len(self._optims) == 0
            else self._optims[0][1].defaults["_save_param_groups"]
        )
        for opt_key, opt in self._optims:
            assert (
                self.defaults["_save_param_groups"]
                == opt.defaults["_save_param_groups"]
            )

            for param_key in opt.params.keys():
                new_param = CombinedOptimizer.prepend_opt_key(param_key, opt_key)
                if new_param in all_keys:
                    raise ValueError(f"Duplicate param key {new_param}")
                all_keys.add(new_param)

        # pyre-ignore
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        # pyre-ignore
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

        self._patch_step_function()

    def __repr__(self) -> str:
        ret = []
        for key, opt in self._optims:
            ret.append(f"{key}: {opt.__repr__()}")
        return ",".join(ret)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for _, opt in self._optims:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        for _, opt in self._optims:
            opt.step(closure=closure)

    @property
    def optimizers(self) -> List[Tuple[str, KeyedOptimizer]]:
        return self._optims

    @staticmethod
    def prepend_opt_key(name: str, opt_key: str) -> str:
        if not name:
            return opt_key
        return opt_key + ("." if opt_key else "") + name

    @property
    def param_groups(self) -> Collection[Mapping[str, Any]]:
        return [
            param_group for _, opt in self._optims for param_group in opt.param_groups
        ]

    @property
    def params(self) -> Mapping[str, Union[torch.Tensor, ShardedTensor]]:
        ret = {}
        for opt_key, opt in self._optims:
            for param_key, param in opt.params.items():
                ret[CombinedOptimizer.prepend_opt_key(param_key, opt_key)] = param
        return ret

    @property
    def state(self) -> Mapping[torch.Tensor, Any]:
        ret = {}
        for _, opt in self._optims:
            for param, state in opt.state.items():
                ret[param] = state
        return ret

    def post_load_state_dict(self) -> None:
        for _, opt in self._optims:
            opt.post_load_state_dict()

    def save_param_groups(self, save: bool) -> None:
        self.defaults["_save_param_groups"] = save
        for _, opt in self._optims:
            opt.save_param_groups(save)

    def set_optimizer_step(self, step: int) -> None:
        for _, opt in self._optims:
            if hasattr(opt, "set_optimizer_step"):
                # pyre-ignore [16]: Undefined attribute [16]: `KeyedOptimizer` has no attribute `set_optimizer_step`.
                opt.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        for _, opt in self._optims:
            if hasattr(opt, "update_hyper_parameters"):
                # pyre-ignore [16].
                opt.update_hyper_parameters(params_dict)


class KeyedOptimizerWrapper(KeyedOptimizer):
    """
    Takes a dict of parameters and exposes state_dict by parameter key.

    Convenience wrapper to take in optim_factory callable to create KeyedOptimizer
    """

    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        optim_factory: OptimizerFactory,
        pass_params_dict: bool = False,
    ) -> None:

        pass_params = params if pass_params_dict else list(params.values())
        # pyre-fixme[6]: For 1st argument expected `List[Union[Tensor,
        #  ShardedTensor]]` but got `Union[List[Union[Tensor, ShardedTensor]],
        #  Mapping[str, Union[Tensor, ShardedTensor]]]`.
        self._optimizer: optim.Optimizer = optim_factory(pass_params)
        super().__init__(params, self._optimizer.state, self._optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad()

    def step(self, closure: Any = None) -> None:
        self._optimizer.step(closure=closure)

    def set_optimizer_step(self, step: int) -> None:
        if hasattr(self._optimizer, "set_optimizer_step"):
            # pyre-ignore [16].
            self._optimizer.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        if hasattr(self._optimizer, "update_hyper_parameters"):
            # pyre-ignore [16].
            self._optimizer.update_hyper_parameters(params_dict)


class OptimizerWrapper(KeyedOptimizer):
    """
    Wrapper which takes in a KeyedOptimizer and is a KeyedOptimizer

    Subclass for Optimizers like GradientClippingOptimizer and WarmupOptimizer
    """

    def __init__(self, optimizer: KeyedOptimizer) -> None:
        self._optimizer = optimizer
        self.params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = optimizer.params
        self.state: Mapping[Any, Any] = optimizer.state
        self.param_groups: Collection[Mapping[str, Any]] = optimizer.param_groups
        self.defaults: Dict[str, Any] = {"_save_param_groups": False}

    def __repr__(self) -> str:
        return self._optimizer.__repr__()

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        self._optimizer.step(closure=closure)

    def add_param_group(self, param_group: Any) -> None:
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        return self._optimizer.state_dict()

    def post_load_state_dict(self) -> None:
        self._optimizer.post_load_state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)
        # Reassign references because self._optimizer receives new state and param_group
        # references after load_state_dict.
        self.state = self._optimizer.state
        self.param_groups = self._optimizer.param_groups

        self.post_load_state_dict()

    def save_param_groups(self, save: bool) -> None:
        self._optimizer.save_param_groups(save)

    def set_optimizer_step(self, step: int) -> None:
        if hasattr(self._optimizer, "set_optimizer_step"):
            # pyre-ignore [16].
            self._optimizer.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        if hasattr(self._optimizer, "update_hyper_parameters"):
            # pyre-ignore [16].
            self._optimizer.update_hyper_parameters(params_dict)
