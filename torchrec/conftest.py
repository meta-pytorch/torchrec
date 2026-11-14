#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""pytest configuration for the OSS test run (``python -m pytest torchrec``); the
Buck test targets do not load it.

PyTorch builds for Python 3.14+ ship without TorchScript, so ``torch.jit.script`` and
``torch.jit.trace`` do not exist there. Tests that script or trace a module cannot run
on those builds: report them as skipped, not failed. Only that AttributeError is
converted, and only when TorchScript is missing; every other failure still fails.
"""

from typing import Generator

import pytest
import torch

_NO_TORCHSCRIPT: bool = not hasattr(torch.jit, "script")
_MISSING_JIT = "module 'torch.jit' has no attribute"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, None]:
    outcome = yield
    exc = call.excinfo.value if call.excinfo is not None else None
    if _NO_TORCHSCRIPT and isinstance(exc, AttributeError) and _MISSING_JIT in str(exc):
        report = outcome.get_result()
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1] or 0,
            f"TorchScript is not available in this PyTorch build: {exc}",
        )
