#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Callable, List, Union

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torch import nn
from torchrec.fx import symbolic_trace
from torchrec.modules.mlp import MLP, Perceptron


class TestMLP(unittest.TestCase):
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    @settings(deadline=None)
    def test_perceptron_single_channel(
        self,
        has_bias: bool,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3

        input_dims: List[int] = [40, 30, 20, 10]
        input_tensors: List[torch.Tensor] = [
            torch.randn(batch_size, input_dims[0]),  # Task 1
            torch.randn(batch_size, input_dims[1]),  # Task 2
            torch.randn(batch_size, input_dims[2]),  # Task 3
            torch.randn(batch_size, input_dims[3]),  # Task 4
        ]

        perceptron_layer_size = 16
        num_tasks = 4

        perceptron_for_tasks = [
            Perceptron(
                input_dims[i],
                perceptron_layer_size,
                bias=has_bias,
                activation=activation,
            )
            for i in range(num_tasks)
        ]

        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        for i in range(num_tasks):
            perceptron_for_tasks[i](
                torch.randn(dry_run_batch_size, input_tensors[i].shape[-1])
            )

        output_tensors = []
        expected_output_tensors = []
        for i in range(len(input_tensors)):
            output_tensors.append(perceptron_for_tasks[i](input_tensors[i]))
            expected_output_tensors.append(
                perceptron_for_tasks[i]._activation_fn(
                    perceptron_for_tasks[i]._linear(input_tensors[i])
                )
            )

        for i in range(len(output_tensors)):
            self.assertEqual(
                list(output_tensors[i].shape), [batch_size, perceptron_layer_size]
            )
            self.assertTrue(
                torch.allclose(output_tensors[i], expected_output_tensors[i])
            )

    def test_fx_script_Perceptron(self) -> None:
        batch_size = 1
        in_features = 3
        out_features = 5
        m = Perceptron(in_features, out_features)

        # Dry-run to initialize lazy module.
        m(torch.randn(batch_size, in_features))

        gm = symbolic_trace(m)
        torch.jit.script(gm)

    def test_fx_script_MLP(self) -> None:
        in_features = 3
        layer_sizes = [16, 8, 4]
        m = MLP(in_features, layer_sizes)

        gm = symbolic_trace(m)
        torch.jit.script(gm)
