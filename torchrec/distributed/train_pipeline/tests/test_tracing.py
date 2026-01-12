#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List
from unittest.mock import MagicMock

import parameterized
import torch
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.tracing import (
    ArgInfo,
    ArgInfoStepFactory,
    CallArgs,
    NodeArgsHelper,
)
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
