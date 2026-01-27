#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Optional
from unittest.mock import MagicMock

import parameterized
import torch
from torch import nn
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.tracing import (
    _get_leaf_module_names,
    ArgInfo,
    ArgInfoStepFactory,
    CallArgs,
    NodeArgsHelper,
    Tracer,
)
from torchrec.distributed.types import NullShardedModuleContext, ShardedModule
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestNodeArg(unittest.TestCase):

    @parameterized.parameterized.expand(
        [
            (
                CallArgs(
                    args=[],
                    kwargs={
                        "id_list_features": ArgInfo(steps=[ArgInfoStepFactory.noop()]),
                        # Empty attrs to ignore any attr based logic.
                        "id_score_list_features": ArgInfo(
                            steps=[ArgInfoStepFactory.noop()]
                        ),
                    },
                ),
                0,
                ["id_list_features", "id_score_list_features"],
            ),
            (
                CallArgs(
                    args=[
                        # Empty attrs to ignore any attr based logic.
                        ArgInfo(steps=[ArgInfoStepFactory.noop()]),
                        ArgInfo(steps=[]),
                    ],
                    kwargs={},
                ),
                2,
                [],
            ),
            (
                CallArgs(
                    args=[
                        # Empty attrs to ignore any attr based logic.
                        ArgInfo(
                            steps=[ArgInfoStepFactory.noop()],
                        )
                    ],
                    kwargs={"id_score_list_features": ArgInfo(steps=[])},
                ),
                1,
                ["id_score_list_features"],
            ),
        ]
    )
    def test_build_args_kwargs(
        self,
        fwd_args: CallArgs,
        args_len: int,
        kwarges_keys: List[str],
    ) -> None:
        args, kwargs = fwd_args.build_args_kwargs("initial_input")
        self.assertEqual(len(args), args_len)
        self.assertEqual(list(kwargs.keys()), kwarges_keys)

    def test_get_node_args_helper_call_module_kjt(self) -> None:
        graph = torch.fx.Graph()
        kjt_args = []

        kjt_args.append(
            torch.fx.Node(graph, "values", "placeholder", "torch.Tensor", (), {})
        )
        kjt_args.append(
            torch.fx.Node(graph, "lengths", "placeholder", "torch.Tensor", (), {})
        )
        kjt_args.append(
            torch.fx.Node(
                graph, "weights", "call_module", "PositionWeightedModule", (), {}
            )
        )

        kjt_node = torch.fx.Node(
            graph,
            "keyed_jagged_tensor",
            "call_function",
            KeyedJaggedTensor,
            tuple(kjt_args),
            {},
        )

        node_args_helper = NodeArgsHelper(MagicMock(), TrainPipelineContext(), False)

        _, num_found = node_args_helper.get_node_args(kjt_node)

        # Weights is call_module node, so we should only find 2 args unmodified
        self.assertEqual(num_found, len(kjt_args) - 1)


class DummyShardedModule(
    ShardedModule[torch.Tensor, torch.Tensor, torch.Tensor, NullShardedModuleContext]
):
    def __init__(self, alpha: float = 1) -> None:
        super().__init__()
        self.alpha = alpha

    # pyre-ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x

    # pyre-ignore
    def compute(self) -> torch.Tensor:
        return torch.empty(0)

    def create_context(self) -> NullShardedModuleContext:
        return NullShardedModuleContext()

    # pyre-ignore
    def input_dist(self, ctx: NullShardedModuleContext):
        pass

    # pyre-ignore
    def output_dist(self):
        pass

    # pyre-ignore
    def unsharded_module_type(self):
        pass


class DummyUmbrellaModule(nn.Module):
    def __init__(self, m1: nn.Module, m2: nn.Module) -> None:
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m1(x) + self.m2(x)


class DummyNestedModule(nn.Module):
    def __init__(self, layer: int = 0) -> None:
        super().__init__()
        self.layer = layer
        self.inner: Optional[nn.Module] = (
            DummyNestedModule(layer - 1) if layer > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = 0 if self.inner is None else self.inner(x)
        return inner + 10**self.layer


class TestFxTracer(unittest.TestCase):
    @classmethod
    def _generate_sharded_model(cls) -> nn.Module:
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = DummyNestedModule(3)
                self.umbrella1 = DummyUmbrellaModule(
                    DummyNestedModule(2), DummyShardedModule()
                )
                self.umbrella2 = DummyUmbrellaModule(
                    DummyNestedModule(3), DummyShardedModule()
                )
                self.umbrella3 = DummyUmbrellaModule(
                    DummyNestedModule(4), DummyNestedModule(5)
                )
                self.umbrella4 = DummyUmbrellaModule(
                    DummyNestedModule(6), DummyNestedModule(7)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (
                    # umbrella2 and umbrella4 are not directly
                    # called in this forward function
                    self.nested(x)
                    + self.umbrella1(x)
                    + self.umbrella2.m1(x)
                    + self.umbrella2.m2(x)
                    + self.umbrella3(x)
                    + self.umbrella4.m1(x)
                    + self.umbrella4.m2(x)
                )

        return MyModel()

    def test_get_leaf_module_names(self) -> None:
        model = self._generate_sharded_model()
        leaf_modules = _get_leaf_module_names(model)
        self.assertSetEqual(
            set(leaf_modules),  # umbrella1.m2 and umbrella2.m2 are `ShardedModule`s
            {"nested", "umbrella1.m1", "umbrella2.m1", "umbrella3", "umbrella4"},
        )

    def test_top_level_tracer(self) -> None:
        model = self._generate_sharded_model()
        concrete_args = {}
        tracer = Tracer(
            leaf_modules=_get_leaf_module_names(model), extend_leaf_fqn=True
        )
        graph = tracer.trace(model, concrete_args=concrete_args)
        targets = {node.target for node in graph.nodes if node.op == "call_module"}
        self.assertSetEqual(
            targets,
            {
                "nested",
                "umbrella1.m1",
                "umbrella1.m2",
                "umbrella2.m1",
                "umbrella2.m2",
                "umbrella3",
                "umbrella4.m1",  # umbrella4 is not called in model.forward
                "umbrella4.m2",  # so umbrella4 is not a leaf module
            },
        )
