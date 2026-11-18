#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import os
import threading
import weakref
from contextlib import contextmanager
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor
from torch.optim import optimizer as torch_optimizer

OptimizerFactory = Callable[[List[Union[torch.Tensor, ShardedTensor]]], optim.Optimizer]

_logger: logging.Logger = logging.getLogger()

# ONEFLOW_OPT_ADAMW_STREAMS splits eligible fused AdamW work across streams.
_OPT_STREAM_BIG_NUMEL: int = 1 << 26
# ONEFLOW_OPT_STREAMS_CHECK checks this many engaged calls per process.
_OPT_STREAMS_CHECK_CALLS: int = 3


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
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

        # pyrefly: ignore[bad-override]
        self.state: Mapping[Any, Any] = state
        # pyrefly: ignore[bad-override]
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
                # pyrefly: ignore[unsupported-operation]
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
                    # pyrefly: ignore[missing-attribute]
                    num_shards = len(v.to_local().local_shards())
                    # pyrefly: ignore[missing-attribute]
                    num_new_shards = len(new_v.to_local().local_shards())
                    if num_shards != num_new_shards:
                        raise ValueError(
                            f"Different number of shards {num_shards} vs {num_new_shards} for the path of {json.dumps(parent_keys)}"
                        )
                    for shard, new_shard in zip(
                        # pyrefly: ignore[missing-attribute]
                        v.to_local().local_shards(),
                        # pyrefly: ignore[missing-attribute]
                        new_v.to_local().local_shards(),
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
        metadata = self.get_checkpoint_metadata()
        if metadata:
            ret["checkpoint_metadata"] = metadata
        return ret

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """Optional optimizer-specific compatibility data, outside parameter state.

        Implementations must also override validate_checkpoint_metadata. Values
        must be checkpoint-serializable; use rank-qualified Tensor leaves for
        rank-local data that must survive distributed checkpoint deduplication.
        Compatibility metadata must be available before the first optimizer step.
        The empty default preserves the existing keyed checkpoint format.
        """
        return {}

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Validate before copying any parameter state, without changing state."""
        if metadata:
            raise ValueError("Unexpected optimizer checkpoint metadata")

    def post_load_state_dict(self) -> None:
        """Finish restoring optimizer state.

        Wrapper composition may call this more than once per load, so
        implementations must be idempotent.
        """
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

        self.validate_checkpoint_metadata(state_dict.get("checkpoint_metadata", {}))
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
                        # pyrefly: ignore[unsupported-operation]
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

        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
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

    # pyrefly: ignore[bad-override]
    def step(self, closure: Any = None) -> None:
        _step_optimizers(self._optims, closure)

    @property
    def optimizers(self) -> List[Tuple[str, KeyedOptimizer]]:
        return self._optims

    @staticmethod
    def prepend_opt_key(name: str, opt_key: str) -> str:
        if not name:
            return opt_key
        return opt_key + ("." if opt_key else "") + name

    @property
    # pyrefly: ignore[bad-override]
    def param_groups(self) -> Collection[Mapping[str, Any]]:
        return [
            param_group for _, opt in self._optims for param_group in opt.param_groups
        ]

    @property
    # pyrefly: ignore[bad-override]
    def params(self) -> Mapping[str, Union[torch.Tensor, ShardedTensor]]:
        ret = {}
        for opt_key, opt in self._optims:
            for param_key, param in opt.params.items():
                ret[CombinedOptimizer.prepend_opt_key(param_key, opt_key)] = param
        return ret

    @property
    # pyrefly: ignore[bad-override]
    def state(self) -> Mapping[torch.Tensor, Any]:
        ret: Dict[torch.Tensor, Any] = {}
        for _, opt in self._optims:
            ret.update(opt.state)
        return ret

    def post_load_state_dict(self) -> None:
        for _, opt in self._optims:
            opt.post_load_state_dict()

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        # Include position as well as prefix: anonymous optimizers may share an
        # empty prefix while owning disjoint parameter names.
        result = {}
        for index, (key, opt) in enumerate(self._optims):
            metadata = opt.get_checkpoint_metadata()
            if metadata:
                result[f"{index}:{key}"] = metadata
        return result

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        if set(metadata) != set(self.get_checkpoint_metadata()):
            raise ValueError("Optimizer checkpoint metadata does not match components")
        # Validate every child before KeyedOptimizer copies state for any child.
        for index, (key, opt) in enumerate(self._optims):
            opt.validate_checkpoint_metadata(metadata.get(f"{index}:{key}", {}))

    def save_param_groups(self, save: bool) -> None:
        self.defaults["_save_param_groups"] = save
        for _, opt in self._optims:
            opt.save_param_groups(save)

    def set_optimizer_step(self, step: int) -> None:
        for _, opt in self._optims:
            if hasattr(opt, "set_optimizer_step"):
                opt.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        for _, opt in self._optims:
            if hasattr(opt, "update_hyper_parameters"):
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
        # pyrefly: ignore[bad-argument-type]
        self._optimizer: optim.Optimizer = optim_factory(pass_params)
        super().__init__(params, self._optimizer.state, self._optimizer.param_groups)

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        get_metadata = getattr(self._optimizer, "get_checkpoint_metadata", None)
        if get_metadata is None:
            return {}
        metadata = get_metadata()
        if not metadata:
            return {}
        param_to_key = {param: key for key, param in self.params.items()}
        # Ordering matters for an optimizer whose ownership is assigned by
        # parameter index, even when the parameter shapes are identical.
        keys = [
            [param_to_key[p] for p in group["params"]] for group in self.param_groups
        ]
        encoded_keys = json.dumps(keys, ensure_ascii=True).encode("utf-8")
        return {
            "optimizer": metadata,
            "parameter_keys": torch.tensor(
                list(encoded_keys), dtype=torch.uint8, device="cpu"
            ),
        }

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        expected = self.get_checkpoint_metadata()
        if not expected:
            return super().validate_checkpoint_metadata(metadata)
        if set(metadata) != set(expected):
            raise ValueError("Missing or incompatible optimizer checkpoint metadata")
        keys = metadata["parameter_keys"]
        if (
            not isinstance(keys, torch.Tensor)
            or keys.dtype != torch.uint8
            or not torch.equal(keys.cpu(), expected["parameter_keys"])
        ):
            raise ValueError("Optimizer checkpoint parameter names or order changed")
        validate = getattr(self._optimizer, "validate_checkpoint_metadata", None)
        if validate is None:
            raise ValueError("Optimizer checkpoint metadata has no validator")
        validate(metadata["optimizer"])

    def post_load_state_dict(self) -> None:
        if not hasattr(self._optimizer, "get_checkpoint_metadata"):
            return
        post_load = getattr(self._optimizer, "post_load_state_dict", None)
        if post_load is not None:
            post_load()

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad()

    # pyrefly: ignore[bad-override]
    def step(self, closure: Any = None) -> None:
        _opt_streams_reach(self._optimizer)
        self._optimizer.step(closure=closure)

    def set_optimizer_step(self, step: int) -> None:
        if hasattr(self._optimizer, "set_optimizer_step"):
            self._optimizer.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        if hasattr(self._optimizer, "update_hyper_parameters"):
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

    # pyrefly: ignore[bad-override]
    def step(self, closure: Any = None) -> None:
        self._optimizer.step(closure=closure)

    def add_param_group(self, param_group: Any) -> None:
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        return self._optimizer.state_dict()

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return self._optimizer.get_checkpoint_metadata()

    def validate_checkpoint_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._optimizer.validate_checkpoint_metadata(metadata)

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
            self._optimizer.set_optimizer_step(step)

    def update_hyper_parameters(self, params_dict: Dict[str, Any]) -> None:
        if hasattr(self._optimizer, "update_hyper_parameters"):
            self._optimizer.update_hyper_parameters(params_dict)


class _OptStreamOps:
    """Stream operations of ONEFLOW_OPT_ADAMW_STREAMS (CUDA and HIP). Tests replace
    ``_OPT_STREAM_OPS``."""

    def supports(self, device: torch.device) -> bool:
        return device.type == "cuda"

    def current(self, device: torch.device) -> Any:
        return torch.cuda.current_stream(device)

    def new(self, device: torch.device) -> Any:
        return torch.cuda.Stream(device=device)

    def use(self, stream: Any) -> Any:
        return torch.cuda.stream(stream)

    def wait(self, waiter: Any, producer: Any) -> None:
        waiter.wait_stream(producer)

    def record(self, tensor: torch.Tensor, stream: Any) -> None:
        tensor.record_stream(stream)

    def name(self, stream: Any) -> str:
        return f"stream_id={stream.stream_id} handle=0x{stream.cuda_stream:x}"

    def defer(self, flag: torch.Tensor, stream: Any) -> Tuple[torch.Tensor, Any]:
        """Copies the 0-dim tensor ``flag``, made on ``stream`` (the current
        stream of its device), to pinned host memory without waiting for it.
        Returns the host tensor and an event that completes after the
        copy."""
        host = torch.empty((), dtype=flag.dtype, pin_memory=True)
        host.copy_(flag, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        return host, event

    def done(self, event: Any) -> bool:
        return event.query()


_OPT_STREAM_OPS: _OptStreamOps = _OptStreamOps()
_OPT_SIDE_STREAMS: Dict[Tuple[str, str, int], Any] = {}
_OPT_STREAMS_LOGGED: Set[str] = set()
_OPT_STREAMS_TLS: threading.local = threading.local()
_OPT_STREAM_CLIPPING = "torchrec.optim.clipping.GradientClippingOptimizer"
_OPT_STREAM_WARMUP = "torchrec.optim.warmup.WarmupOptimizer"
# Wrapper attributes that hold a tensor made in step() and read after it.
# Wrapper attributes that only steer logging: the ONEFLOW_OPT_SMALL_STREAM
# CHECK replay keeps their values from the side-stream step.
_OPT_STREAM_NUMEL: "weakref.WeakKeyDictionary[optim.Optimizer, Tuple[int, int]]" = (
    weakref.WeakKeyDictionary()
)
_OPT_STREAM_PLAIN: Tuple[type, ...] = (torch.Tensor, torch.nn.Parameter)
_OPT_STREAM_BITS: Dict[torch.dtype, torch.dtype] = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}
# ONEFLOW_OPT_SMALL_STREAM process state: first checks left, latched off,
# last logged plan, engaged steps.
# Deferred ONEFLOW_OPT_STREAMS_CHECK verdicts not read yet: (process state of
# the flag, flag, mismatch description, pinned host bool, event of its copy).
_OPT_STREAMS_PENDING: List[Tuple[Dict[str, Any], str, str, torch.Tensor, Any]] = []
# (key, optimizer, chain from the optimizer down to the torch optimizer)


def _opt_stream_ops() -> _OptStreamOps:
    return _OPT_STREAM_OPS


def _opt_side_stream(device: torch.device, role: str, slot: int) -> Any:
    """The side stream ``slot`` of ``role`` on ``device``, made on first use."""
    key = (str(device), role, slot)
    stream = _OPT_SIDE_STREAMS.get(key)
    if stream is None:
        stream = _OPT_STREAM_OPS.new(device)
        _OPT_SIDE_STREAMS[key] = stream
    return stream


def _opt_streams_log_once(key: str, level: int, message: Callable[[], str]) -> None:
    """Logs ``message()`` the first time ``key`` is seen in this process."""
    if key not in _OPT_STREAMS_LOGGED:
        _OPT_STREAMS_LOGGED.add(key)
        _logger.log(level, message())


def _opt_streams_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _opt_streams_same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality; equal NaN payloads compare equal."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    bits = _OPT_STREAM_BITS.get(a.dtype)
    if bits is None:
        return torch.equal(a, b)
    return torch.equal(a.view(bits), b.view(bits))


def _opt_streams_same_all(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Bitwise equality of every pair (equal NaN payloads compare equal) as
    one 0-dim bool tensor, computed without a host sync; None if a pair
    differs in dtype or shape."""
    flags: List[torch.Tensor] = []
    for a, b in pairs:
        if a.dtype != b.dtype or a.shape != b.shape:
            return None
        bits = _OPT_STREAM_BITS.get(a.dtype)
        if bits is not None:
            a, b = a.view(bits), b.view(bits)
        flags.append(torch.eq(a, b).all())
    if not flags:
        return torch.ones((), dtype=torch.bool)
    return torch.stack(flags).all()


def _opt_streams_every() -> int:
    """ONEFLOW_OPT_STREAMS_CHECK_EVERY=N (read per call): after its first
    checks, a flag is checked again on every N-th engaged call, and the
    verdict is read later (``_opt_streams_poll``). Unset or 0: never."""
    raw = os.environ.get("ONEFLOW_OPT_STREAMS_CHECK_EVERY", "0")
    try:
        every = int(raw)
    except ValueError:
        every = -1
    if every < 0:
        _opt_streams_log_once(
            "every:" + raw,
            logging.WARNING,
            lambda: f"ONEFLOW_OPT_STREAMS_CHECK_EVERY={raw!r} is not a "
            "non-negative integer; no periodic checks",
        )
        return 0
    return every


def _opt_streams_check_mode(checks_left: int, count: int) -> str:
    """ONEFLOW_OPT_STREAMS_CHECK=1 (read per call): "sync" while a flag has
    first checks left, then "deferred" on every
    ONEFLOW_OPT_STREAMS_CHECK_EVERY-th engaged call (``count`` counts them
    from 1); "" for no check."""
    if os.environ.get("ONEFLOW_OPT_STREAMS_CHECK", "1") != "1":
        return ""
    if checks_left > 0:
        return "sync"
    every = _opt_streams_every()
    return "deferred" if every > 0 and count % every == 0 else ""


def _opt_streams_checked(
    state: Dict[str, Any],
    flag: str,
    mode: str,
    ok: Optional[torch.Tensor],
    mismatch: str,
    stream: Any,
    matched: Callable[[], str],
) -> None:
    """The verdict of a ONEFLOW_OPT_STREAMS_CHECK call of ``flag``, whose
    process state ``state`` holds its first "checks" left. A "sync" check
    reads ``ok`` now (a host sync) and logs ``matched()`` after the last
    first check; a "deferred" check queues ``ok`` for ``_opt_streams_poll``.
    A mismatch is fatal (``_opt_streams_fatal``)."""
    if ok is None:
        _opt_streams_fatal(state, flag, mismatch)
    if mode == "deferred":
        host, event = _OPT_STREAM_OPS.defer(ok, stream)
        _OPT_STREAMS_PENDING.append((state, flag, mismatch, host, event))
        _opt_streams_log_once(
            "every:on:" + flag,
            logging.INFO,
            lambda: f"ONEFLOW_OPT_STREAMS_CHECK_EVERY: {flag} is checked again "
            f"every {_opt_streams_every()} engaged calls; the verdicts are read "
            "later without a host sync",
        )
        return
    if not bool(ok):
        _opt_streams_fatal(state, flag, mismatch)
    state["checks"] -= 1
    if state["checks"] == 0:
        _logger.info(f"ONEFLOW_OPT_STREAMS_CHECK: {matched()}")


def _opt_streams_poll() -> None:
    """Reads the verdicts of deferred checks whose host copies completed
    (``event.query()``, no host sync); a mismatch is fatal."""
    while _OPT_STREAMS_PENDING and _OPT_STREAM_OPS.done(_OPT_STREAMS_PENDING[0][4]):
        state, flag, mismatch, host, _ = _OPT_STREAMS_PENDING.pop(0)
        if not bool(host):
            _opt_streams_fatal(state, flag, f"{mismatch} in a deferred check")


def _opt_streams_fatal(state: Dict[str, Any], flag: str, what: str) -> NoReturn:
    """A ONEFLOW_OPT_STREAMS_CHECK mismatch of a stream flag. The inputs are
    the same on every rank, so a result that differs from the stock one on
    some ranks only means that the stream ordering failed there: rather
    than letting this rank go on with a path that may race, latch ``flag``
    off and raise (no collective)."""
    state["off"] = True
    message = (
        f"ONEFLOW_OPT_STREAMS_CHECK: rank {_opt_streams_rank()}: {what}; {flag} "
        "is not bitwise safe on this rank, stopping"
    )
    _logger.error(message)
    raise RuntimeError(message)


def _opt_streams_reach(inner: Any) -> None:
    """KeyedOptimizerWrapper.step: one-time warnings when
    ONEFLOW_OPT_SMALL_STREAM=1 or ONEFLOW_OPT_ADAMW_STREAMS=1 cannot act on
    ``inner`` because its step bypasses their hook points
    (CombinedOptimizer.step, GradientClippingOptimizer.step)."""
    tls = _OPT_STREAMS_TLS
    if os.environ.get("ONEFLOW_OPT_SMALL_STREAM", "0") == "1" and not getattr(
        tls, "combined", 0
    ):
        _opt_streams_log_once(
            "small:reach",
            logging.WARNING,
            lambda: "ONEFLOW_OPT_SMALL_STREAM=1 has no effect on a "
            f"{type(inner).__qualname__} stepped outside CombinedOptimizer.step; "
            "it steps on the current stream",
        )
    if (
        os.environ.get("ONEFLOW_OPT_ADAMW_STREAMS", "0") == "1"
        and not getattr(tls, "clipping", 0)
        and _opt_stream_numel(inner) >= _OPT_STREAM_BIG_NUMEL
    ):
        _opt_streams_log_once(
            "adamw:reach",
            logging.WARNING,
            lambda: "ONEFLOW_OPT_ADAMW_STREAMS=1 has no effect on a "
            f"{type(inner).__qualname__} with {_opt_stream_numel(inner)} elements "
            "that is not wrapped by GradientClippingOptimizer; its fused AdamW "
            "runs unsplit",
        )


def _opt_stream_chain(opt: Any) -> Tuple[Optional[str], List[Any]]:
    """Unwraps ``opt`` to a torch Adam or AdamW with ``fused=True``. Returns
    a decline reason or None, and the objects from ``opt`` down to the torch
    optimizer. Wrappers match by exact class: GradientClippingOptimizer
    without sharded params and WarmupOptimizer add host bookkeeping and plain
    torch ops on the current stream, and no collectives."""
    chain: List[Any] = []
    o = opt
    while type(o) is not KeyedOptimizerWrapper:
        name = f"{type(o).__module__}.{type(o).__qualname__}"
        if name == _OPT_STREAM_CLIPPING:
            if getattr(o, "_sharded_params", None):
                return "clipping sharded params runs collectives", chain
        elif name != _OPT_STREAM_WARMUP:
            return f"{name} is not a known stream-safe wrapper", chain
        chain.append(o)
        o = o._optimizer
    inner = o._optimizer
    chain += [o, inner]
    if type(inner) not in (optim.AdamW, optim.Adam) or not all(
        group.get("fused") for group in inner.param_groups
    ):
        return f"{type(inner).__qualname__} is not a fused torch Adam(W)", chain
    return None, chain


def _opt_stream_numel(inner: optim.Optimizer) -> int:
    """Elements of the params of ``inner``, cached while its param count
    stays the same."""
    count = sum(len(group["params"]) for group in inner.param_groups)
    cached = _OPT_STREAM_NUMEL.get(inner)
    if cached is None or cached[0] != count:
        numel = sum(p.numel() for group in inner.param_groups for p in group["params"])
        cached = (count, numel)
        _OPT_STREAM_NUMEL[inner] = cached
    return cached[1]


_OPT_STREAM_ESCAPED: Tuple[str, ...] = ("_last_total_grad_norm",)


_OPT_STREAM_LOG_ONLY: Tuple[str, ...] = ("_clip_flat_logged",)


_SMALL_STREAM: Dict[str, Any] = {
    "checks": _OPT_STREAMS_CHECK_CALLS,
    "off": False,
    "plan": None,
    "steps": 0,
}


_SmallPlan = List[Tuple[str, Any, List[Any]]]


def _opt_stream_safe_hooks() -> Tuple[Any, ...]:
    """Global optimizer step hooks that only do host bookkeeping, so they
    may run with any current stream: torch.profiler's step counter for
    on-demand tracing (registered at import when KINETO_USE_DAEMON or
    KINETO_FORCE_OPTIMIZER_HOOK is set)."""
    hook = getattr(getattr(torch, "profiler", None), "_optimizer_post_hook", None)
    return () if hook is None else (hook,)


def _opt_stream_global_hooks() -> List[str]:
    """The names of the registered global optimizer step hooks other than
    ``_opt_stream_safe_hooks``."""
    safe = _opt_stream_safe_hooks()
    names = []
    for attr in ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks"):
        for hook in (getattr(torch_optimizer, attr, None) or {}).values():
            if not any(hook is s for s in safe):
                names.append(
                    f"{getattr(hook, '__module__', None)}."
                    f"{getattr(hook, '__qualname__', type(hook).__qualname__)}"
                )
    return names


@contextmanager
def _opt_stream_skip_safe_hooks() -> Iterator[None]:
    """Runs the block without the global hooks of ``_opt_stream_safe_hooks``:
    the ONEFLOW_OPT_SMALL_STREAM CHECK replay repeats steps that ran them
    already. They are put back in their places afterwards."""
    hooks: Any = getattr(torch_optimizer, "_global_optimizer_post_hooks", None)
    safe = _opt_stream_safe_hooks()
    saved = list(hooks.items()) if hooks else []
    removed = {key for key, hook in saved if any(hook is s for s in safe)}
    for key in removed:
        del hooks[key]
    try:
        yield
    finally:
        if removed:
            now = dict(hooks)
            hooks.clear()
            for key, hook in saved:
                if key in removed or key in now:
                    hooks[key] = now.get(key, hook)
            for key, hook in now.items():
                hooks.setdefault(key, hook)


def _opt_stream_escaped(chain: List[Any], device: torch.device) -> List[torch.Tensor]:
    """The tensors of ``_OPT_STREAM_ESCAPED`` on ``device`` in ``chain``."""
    tensors = []
    for o in chain:
        for name in _OPT_STREAM_ESCAPED:
            t = o.__dict__.get(name)
            if isinstance(t, torch.Tensor) and t.device == device:
                tensors.append(t)
    return tensors


def _small_stream_tensors(inner: optim.Optimizer) -> List[torch.Tensor]:
    """The params with grads of ``inner``, their grads and their state
    tensors."""
    tensors = []
    for group in inner.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                tensors += [p, p.grad]
                tensors += [t for t in inner.state[p].values() if torch.is_tensor(t)]
    return tensors


def _small_stream_ready(
    inner: optim.Optimizer,
) -> Tuple[Optional[str], Optional[torch.device]]:
    """ONEFLOW_OPT_SMALL_STREAM: why ``inner`` cannot step on a side stream
    now (or None), and its device. Lazy state init stays on the current
    stream, so the side stream never allocates optimizer state."""
    if inner._optimizer_step_pre_hooks or inner._optimizer_step_post_hooks:
        return "its torch optimizer has step hooks", None
    device: Optional[torch.device] = None
    for group in inner.param_groups:
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if (
                type(p) not in _OPT_STREAM_PLAIN
                or type(grad) is not torch.Tensor
                or grad.is_sparse
            ):
                return "a param or grad is not a plain dense tensor", None
            if device is not None and p.device != device:
                return "its params are on several devices", None
            device = p.device
            if not inner.state.get(p):
                return "its state is not initialized yet", None
    if device is None:
        return "no param has a grad", None
    if not _OPT_STREAM_OPS.supports(device):
        return f"{device} is not a supported device", None
    return None, device


def _small_stream_decline(closure: Any) -> str:
    if closure is not None:
        return "a closure was passed"
    if torch.compiler.is_compiling():
        return "torch.compile is tracing"
    if _SMALL_STREAM["off"]:
        return "latched off after a ONEFLOW_OPT_STREAMS_CHECK mismatch"
    hooks = _opt_stream_global_hooks()
    if hooks:
        return f"global optimizer step hooks are registered: {', '.join(hooks)}"
    return ""


def _small_stream_plan(
    optims: List[Tuple[str, Any]], closure: Any
) -> Tuple[str, _SmallPlan, List[str], Optional[torch.device]]:
    """ONEFLOW_OPT_SMALL_STREAM: a decline reason, the optimizers to step on
    the side stream, notes on the others and the device."""
    reason = _small_stream_decline(closure)
    if reason:
        return reason, [], [], None
    side: _SmallPlan = []
    kept: List[str] = []
    devices: Set[torch.device] = set()
    for key, opt in optims:
        reason, chain = _opt_stream_chain(opt)
        if not reason and _opt_stream_numel(chain[-1]) >= _OPT_STREAM_BIG_NUMEL:
            reason = f"at least {_OPT_STREAM_BIG_NUMEL} elements"
        device = None
        if not reason:
            reason, device = _small_stream_ready(chain[-1])
        if reason:
            kept.append(f"{key or '<unnamed>'} ({reason})")
        else:
            assert device is not None
            side.append((key, opt, chain))
            devices.add(device)
    if not side:
        return "no optimizer can use it: " + "; ".join(kept), [], kept, None
    if len(devices) != 1:
        return "the small optimizers are on several devices", [], kept, None
    return "", side, kept, devices.pop()


def _step_optimizers(optims: List[Tuple[str, Any]], closure: Any) -> None:
    """CombinedOptimizer.step. With ONEFLOW_OPT_SMALL_STREAM=1 (read per
    call) the optimizers with fewer than ``_OPT_STREAM_BIG_NUMEL`` elements
    step on a side stream (``_small_stream_step``)."""
    if os.environ.get("ONEFLOW_OPT_SMALL_STREAM", "0") != "1":
        for _, opt in optims:
            opt.step(closure=closure)
        return
    tls = _OPT_STREAMS_TLS
    tls.combined = getattr(tls, "combined", 0) + 1
    try:
        if getattr(tls, "active", False) or not _small_stream_step(optims, closure):
            for _, opt in optims:
                opt.step(closure=closure)
    finally:
        tls.combined -= 1


def _small_stream_step(optims: List[Tuple[str, Any]], closure: Any) -> bool:
    """ONEFLOW_OPT_SMALL_STREAM: steps the optimizers in their usual order;
    the small ones run on a side stream that first waits for the current
    stream (grads and everything before this step), and the current stream
    waits for the side stream before anything after this step. Tensors that
    the small optimizers keep for later readers are recorded on the current
    stream for the allocator. Returns False, after a one-time warning, if
    declined."""
    _opt_streams_poll()
    reason, side, kept, device = _small_stream_plan(optims, closure)
    if reason or device is None:
        _opt_streams_log_once(
            "small:" + reason,
            logging.WARNING,
            lambda: f"ONEFLOW_OPT_SMALL_STREAM=1 declined: {reason}; every "
            "optimizer steps on the current stream",
        )
        return False
    ops = _OPT_STREAM_OPS
    cur = ops.current(device)
    stream = _opt_side_stream(device, "small", 0)
    _small_stream_log_plan(side, kept, device, cur, stream)
    _SMALL_STREAM["steps"] += 1
    mode = _opt_streams_check_mode(_SMALL_STREAM["checks"], _SMALL_STREAM["steps"])
    replay = _SmallStreamReplay([c for _, _, c in side]) if mode else None
    _small_stream_run(optims, {id(opt) for _, opt, _ in side}, stream, cur, closure)
    for _, _, chain in side:
        for t in _opt_stream_escaped(chain, device):
            ops.record(t, cur)
    if replay is not None:
        _opt_streams_checked(
            _SMALL_STREAM,
            "ONEFLOW_OPT_SMALL_STREAM",
            mode,
            replay.compare(side, device),
            "the side-stream step differs from the serial step (kept the serial "
            "result)",
            cur,
            lambda: "ONEFLOW_OPT_SMALL_STREAM matched the serial step bitwise "
            f"(params, grads, state, grad norms of {len(side)} optimizers) in "
            f"{_OPT_STREAMS_CHECK_CALLS} steps",
        )
    return True


def _small_stream_run(
    optims: List[Tuple[str, Any]],
    side_ids: Set[int],
    stream: Any,
    cur: Any,
    closure: Any,
) -> None:
    """Steps ``optims`` in order, those in ``side_ids`` on ``stream``,
    between ``stream`` waiting for ``cur`` and ``cur`` waiting for
    ``stream``."""
    ops = _OPT_STREAM_OPS
    ops.wait(stream, cur)
    _OPT_STREAMS_TLS.active = True
    try:
        for _, opt in optims:
            if id(opt) in side_ids:
                with ops.use(stream):
                    opt.step(closure=closure)
            else:
                opt.step(closure=closure)
    finally:
        _OPT_STREAMS_TLS.active = False
        ops.wait(cur, stream)


def _small_stream_log_plan(
    side: _SmallPlan, kept: List[str], device: torch.device, cur: Any, stream: Any
) -> None:
    """Logs the ONEFLOW_OPT_SMALL_STREAM plan whenever it changes."""
    plan = (tuple(key for key, _, _ in side), tuple(kept))
    if plan == _SMALL_STREAM["plan"]:
        return
    _SMALL_STREAM["plan"] = plan
    ops = _OPT_STREAM_OPS
    _logger.info(
        f"ONEFLOW_OPT_SMALL_STREAM: {len(side)} optimizers "
        f"({', '.join(key or '<unnamed>' for key, _, _ in side)}; "
        f"{sum(_opt_stream_numel(c[-1]) for _, _, c in side)} elements) step on "
        f"side stream {ops.name(stream)} of {device}; the current stream "
        f"{ops.name(cur)} keeps {'; '.join(kept) or 'nothing'}"
    )


class _SmallStreamReplay:
    """ONEFLOW_OPT_STREAMS_CHECK for ONEFLOW_OPT_SMALL_STREAM. Snapshots the
    side-stream optimizers (tensors and host state) before the step; after
    it, restores the snapshot, replays their step serially on the current
    stream and compares bitwise. The replay is the stock step and stays, so
    every rank keeps the stock result whatever the verdict. Rank-local, no
    collectives."""

    def __init__(self, chains: List[List[Any]]) -> None:
        with torch.no_grad():
            self._tensors: List[Tuple[torch.Tensor, torch.Tensor]] = [
                (t, t.clone()) for c in chains for t in _small_stream_tensors(c[-1])
            ]
        self._hosts: List[Tuple[Any, Dict[str, Any]]] = [
            (o, dict(o.__dict__)) for c in chains for o in c
        ]
        self._states: List[Tuple[Any, Dict[Any, Any], Dict[Any, Dict[str, Any]]]] = [
            (
                c[-1].state,
                dict(c[-1].state),
                {k: dict(v) for k, v in c[-1].state.items() if isinstance(v, dict)},
            )
            for c in chains
        ]
        self._groups: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
            (group, dict(group)) for c in chains for group in c[-1].param_groups
        ]

    def _restore(self) -> None:
        with torch.no_grad():
            for live, saved in self._tensors:
                live.copy_(saved)
        for o, saved in self._hosts:
            logged = {k: o.__dict__[k] for k in _OPT_STREAM_LOG_ONLY if k in o.__dict__}
            o.__dict__.clear()
            o.__dict__.update(saved)
            o.__dict__.update(logged)
        for state, top, entries in self._states:
            state.clear()
            state.update(top)
            for key, saved in entries.items():
                state[key].clear()
                state[key].update(saved)
        for group, saved in self._groups:
            group.clear()
            group.update(saved)

    def compare(self, side: _SmallPlan, device: torch.device) -> Optional[torch.Tensor]:
        """Replays the step of ``side`` serially from the snapshot, without
        the global hooks of ``_opt_stream_skip_safe_hooks`` (they ran in the
        side-stream step), and returns the bitwise verdict of the side-stream
        results against the replay (``_opt_streams_same_all``)."""
        with torch.no_grad():
            got = [live.clone() for live, _ in self._tensors]
        got_escaped = [t for _, _, c in side for t in _opt_stream_escaped(c, device)]
        self._restore()
        with _opt_stream_skip_safe_hooks():
            for _, opt, _ in side:
                opt.step(closure=None)
        want_escaped = [t for _, _, c in side for t in _opt_stream_escaped(c, device)]
        if len(got_escaped) != len(want_escaped):
            return None
        return _opt_streams_same_all(
            list(zip(got_escaped, want_escaped))
            + [(live, g) for (live, _), g in zip(self._tensors, got)]
        )
