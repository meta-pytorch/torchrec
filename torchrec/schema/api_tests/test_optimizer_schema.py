#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import Any, Collection, List, Mapping, Optional, Set, Tuple, Union

import torch
from torch import optim
from torchrec.distributed.types import ShardedTensor
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.schema.utils import is_signature_compatible


class StableKeyedOptimizer(optim.Optimizer):
    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        # pyre-ignore [2]
        state: Mapping[Any, Any],
        param_groups: Collection[Mapping[str, Any]],
    ) -> None:
        pass

    def init_state(
        self,
        sparse_grad_parameter_names: Optional[Set[str]] = None,
    ) -> None:
        pass

    def save_param_groups(self, save: bool) -> None:
        pass

    # pyre-ignore [2]
    def add_param_group(self, param_group: Any) -> None:
        pass

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass


class StableCombinedOptimizer(KeyedOptimizer):
    def __init__(
        self,
        optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]],
    ) -> None:
        pass

    @property
    def optimizers(self) -> List[Tuple[str, StableKeyedOptimizer]]:
        return []

    @staticmethod
    def prepend_opt_key(name: str, opt_key: str) -> str:
        return ""

    @property
    def param_groups(self) -> Collection[Mapping[str, Any]]:
        return []

    @property
    def params(self) -> Mapping[str, Union[torch.Tensor, ShardedTensor]]:
        return {}

    def post_load_state_dict(self) -> None:
        pass

    def save_param_groups(self, save: bool) -> None:
        pass

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass


class TestOptimizerSchema(unittest.TestCase):
    def test_keyed_optimizer(self) -> None:
        stable_keyed_optimizer_funcs = inspect.getmembers(
            StableKeyedOptimizer, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_keyed_optimizer_funcs:
            self.assertTrue(getattr(KeyedOptimizer, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(KeyedOptimizer, func_name)),
                )
            )

    def test_combined_optimizer(self) -> None:
        stable_combined_optimizer_funcs = inspect.getmembers(
            StableCombinedOptimizer, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_combined_optimizer_funcs:
            self.assertTrue(getattr(CombinedOptimizer, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(CombinedOptimizer, func_name)),
                )
            )
