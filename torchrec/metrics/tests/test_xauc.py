#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import sysconfig
import unittest
from typing import Dict
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo import is_dynamo_supported
from torchrec.metrics.metrics_config import DefaultTaskInfo, RecComputeMode, RecTaskInfo
from torchrec.metrics.xauc import compute_error_sum, get_xauc_states, XAUCMetric


WORLD_SIZE = 4
BATCH_SIZE = 10


def _is_free_threaded() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def generate_model_output() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "labels": torch.tensor([[0.2, 0.1, 0.3, 0.5, 0.25]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]]),
        "expected_xauc": torch.tensor([0.6667]),
    }


class XAUCMetricTest(unittest.TestCase):
    def test_xauc(self) -> None:
        cases = [
            (
                torch.tensor([[0.2, 0.5, 0.5, 0.9, 0.1]]),
                torch.tensor([[True, False, False, True, True]]),
                torch.ones(1, 5),
                torch.tensor([3 / 10]),
            ),
            (
                torch.tensor([[0.2, 0.5, 0.5, 0.9, 0.1, torch.nan, 0.3]]),
                torch.tensor([[3.0, 1.0, 1.0, 4.0, 2.0, 2.0, torch.nan]]),
                torch.tensor([[1.0, 2.0, 1.0, 1.0, 0.5, 3.0, 2.0]]),
                torch.tensor([7 / 45]),
            ),
            (
                torch.tensor([[0.0, 0.0, 0.0]]),
                torch.tensor([[torch.nan, 0.0, 0.0]]),
                torch.ones(1, 3),
                torch.tensor([1 / 3]),
            ),
            (
                torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.5, 0.5, 0.9, 0.1]]),
                torch.tensor([[0.2, 0.1, 0.3, 0.5, 0.25], [3.0, 1.0, 2.0, 4.0, 2.0]]),
                torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
                torch.tensor([4 / 6, 5 / 10]),
            ),
        ]
        for predictions, labels, weights, expected_metric in cases:
            with self.subTest(expected_metric=expected_metric):
                tasks = [
                    RecTaskInfo(name=f"task_{head}") for head in range(len(predictions))
                ]
                xauc = XAUCMetric(
                    world_size=WORLD_SIZE,
                    my_rank=0,
                    batch_size=BATCH_SIZE,
                    tasks=tasks,
                    compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
                )
                task_names = [task.name for task in tasks]
                xauc.update(
                    predictions=dict(zip(task_names, predictions)),
                    labels=dict(zip(task_names, labels)),
                    weights=dict(zip(task_names, weights)),
                )
                metric = xauc.compute()
                actual_metric = torch.stack(
                    [metric[f"xauc-{task.name}|lifetime_xauc"] for task in tasks]
                )

                self.assertFalse(
                    compute_error_sum(
                        labels, predictions.requires_grad_(), weights
                    ).requires_grad
                )
                torch.testing.assert_close(
                    actual_metric, expected_metric, check_dtype=False
                )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @unittest.skipIf(
        _is_free_threaded(),
        "torch.compile segfaults on free-threaded Python, "
        "see https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276",
    )
    def test_xauc_compile(self) -> None:
        xauc = XAUCMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyrefly: ignore[bad-argument-type]
            enable_pt2_compile=False,
            window_size=200,
        )

        model_output = generate_model_output()
        fullgraph_compile = functools.partial(torch.compile, fullgraph=True)
        # torchmetrics.Metric.update() increments the integer ``_update_count``
        # nn.Module attribute on every call. By default torch.compile specializes
        # on integer module attributes, so each update sees a new value and
        # triggers a recompile, exceeding ``recompile_limit`` after a few
        # iterations (the test was failing with FailOnRecompileLimitHit under
        # fullgraph=True). Treat integer module attributes as dynamic so the
        # changing _update_count no longer forces recompiles, and assert that the
        # recompile limit is never hit so this stays a regression guard.
        with patch.object(torch, "compile", fullgraph_compile), patch.object(
            torch._dynamo.config, "allow_unspec_int_on_nn_module", True
        ), patch.object(torch._dynamo.config, "fail_on_recompile_limit_hit", True):
            xauc_compile = XAUCMetric(
                world_size=WORLD_SIZE,
                my_rank=0,
                batch_size=BATCH_SIZE,
                tasks=[DefaultTaskInfo],
                # pyrefly: ignore[bad-argument-type]
                enable_pt2_compile=True,
                window_size=200,
            )
            for _ in range(10):
                xauc_compile.update(
                    predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
                    labels={DefaultTaskInfo.name: model_output["labels"][0]},
                    weights={DefaultTaskInfo.name: model_output["weights"][0]},
                    _log_tensors=False,  # required for fullgraph=True
                )
                xauc.update(
                    predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
                    labels={DefaultTaskInfo.name: model_output["labels"][0]},
                    weights={DefaultTaskInfo.name: model_output["weights"][0]},
                )
        compile_out = xauc_compile.compute()
        eager_out = xauc.compute()
        for prefix in ("lifetime_xauc", "window_xauc"):
            key = f"xauc-{DefaultTaskInfo.name}|{prefix}"
            torch.testing.assert_close(
                compile_out[key],
                eager_out[key],
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                equal_nan=True,
                msg=f"[{prefix}] Compiled: {compile_out[key]}, Eager: {eager_out[key]}",
            )

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    @unittest.skipIf(
        _is_free_threaded(),
        "torch.compile segfaults on free-threaded Python, "
        "see https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276",
    )
    def test_xauc_states_compile_dynamic(self) -> None:
        compiled_get_xauc_states = torch.compile(
            get_xauc_states, fullgraph=True, dynamic=True
        )
        with patch.object(torch._dynamo.config, "error_on_recompile", True):
            for num_samples in (5, 7):
                inputs = (
                    torch.rand(2, num_samples),
                    torch.rand(2, num_samples),
                    torch.ones(2, num_samples),
                )
                torch.testing.assert_close(
                    compiled_get_xauc_states(*inputs), get_xauc_states(*inputs)
                )
