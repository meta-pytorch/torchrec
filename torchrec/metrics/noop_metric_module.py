#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
from torchrec.metrics.metric_module import MetricValue, RecMetricModule


class NoOpMetricModule(RecMetricModule):
    """
    A no-op implementation of RecMetricModule for when metrics
    computation is disabled.
    """

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def _update_rec_metrics(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        pass

    def update(self, model_out: Dict[str, torch.Tensor], **kwargs: Any) -> None:
        pass

    def should_compute(self) -> bool:
        return False

    def compute(self) -> Dict[str, MetricValue]:
        return {}

    def local_compute(self) -> Dict[str, MetricValue]:
        return {}

    def sync(self) -> None:
        pass

    def unsync(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_required_inputs(self) -> Optional[List[str]]:
        return None

    def get_pre_compute_states(
        self, pg: Optional[Union[dist.ProcessGroup, DeviceMesh]] = None
    ) -> Dict[str, Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]]:
        return {}

    def load_pre_compute_states(
        self,
        source: Dict[
            str, Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]
        ],
    ) -> None:
        pass

    def shutdown(self) -> None:
        pass

    # pyrefly: ignore[bad-override]
    def async_compute(self, future: Future[Dict[str, MetricValue]]) -> None:
        pass
