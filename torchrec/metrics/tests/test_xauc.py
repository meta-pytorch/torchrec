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
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rec_metric import WindowBuffer
from torchrec.metrics.xauc import XAUCMetric


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
        xauc = XAUCMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
        )

        model_output = generate_model_output()
        xauc.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
        )
        metric = xauc.compute()
        actual_metric = metric[f"xauc-{DefaultTaskInfo.name}|lifetime_xauc"]
        expected_metric = model_output["expected_xauc"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    @unittest.skipIf(
        _is_free_threaded(),
        "torch.compile segfaults on free-threaded Python, "
        "see https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276",
    )
    def test_xauc_compile(self) -> None:
        """
        Covers three torch.compile-related aspects of windowed xauc:
        1. Numerics match between compiled and eager xauc.
        2. No graph breaks (fullgraph=True).
        3. Guard against corruption as past seen with window buffer aliasing.
        """
        window_size = 40  # sized small enough to test eviction
        n_iters = 15

        xauc = XAUCMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyrefly: ignore[bad-argument-type]
            enable_pt2_compile=False,
            window_size=window_size,
        )

        model_output = generate_model_output()
        fullgraph_compile = functools.partial(torch.compile, fullgraph=True)
        with patch.object(torch, "compile", fullgraph_compile):
            xauc_compile = XAUCMetric(
                world_size=WORLD_SIZE,
                my_rank=0,
                batch_size=BATCH_SIZE,
                tasks=[DefaultTaskInfo],
                # pyrefly: ignore[bad-argument-type]
                enable_pt2_compile=True,
                window_size=window_size,
            )
            for _ in range(n_iters):
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

        # Guard against the inductor buffer-reuse corruption
        # Now more implementation-agnostic: any defense that prevents
        # production corruption -- in-op clone, slab+copy_, an inductor-side
        # never_reuse annotation, storage handoff -- passes this check.
        sentinel = -1.0e9
        max_buffer_count = 4
        buf = WindowBuffer(max_size=10**9, max_buffer_count=max_buffer_count)

        @torch.compile(fullgraph=True)
        def step(window_state: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            curr_state = source + 1.0
            buf.aggregate_state(window_state, curr_state, size=1)
            return torch.full_like(curr_state, sentinel)

        window_state = torch.zeros(3, dtype=torch.float64)
        snapshots: list[torch.Tensor] = []
        for i in range(max_buffer_count):
            source = torch.tensor(
                [i + 0.1 - 1.0, i + 0.2 - 1.0, i + 0.3 - 1.0], dtype=torch.float64
            )
            snapshots.append(source + 1.0)
            step(window_state, source)
        for idx, expected in enumerate(snapshots):
            torch.testing.assert_close(
                buf.buffers[idx],
                expected,
                atol=1e-6,
                rtol=1e-6,
                msg=(
                    f"WindowBuffer entry #{idx} corrupted by Inductor " "buffer reuse."
                ),
            )
