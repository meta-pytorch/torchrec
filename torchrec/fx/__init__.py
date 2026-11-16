#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Tracer

Custom FX tracer for torchrec

See `Torch.FX documentation <https://pytorch.org/docs/stable/fx.html>`_
"""

from torchrec.fx.tracer import symbolic_trace, Tracer  # noqa
