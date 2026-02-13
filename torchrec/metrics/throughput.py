#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
import logging
import math
import time
from collections import deque, OrderedDict
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.distributed.utils import none_throws
from torchrec.metrics.metrics_config import BatchSizeStage
from torchrec.metrics.metrics_namespace import (
    compose_metric_key,
    MetricName,
    MetricNamespace,
    MetricPrefix,
)

MAX_WINDOW_TS: int = 2 * 60 * 60  # 2 hours

logger: logging.Logger = logging.getLogger(__name__)


class ThroughputMetric(nn.Module):
    """
    The module to calculate throughput. Throughput is defined as the trained examples
    across all ranks per second. For example, if the batch size on each rank is 512
    and there are 32 ranks, throughput is 512 * 32 / time_to_train_one_step.

    Args:
        batch_size (int): batch size for the trainer
        world_size (int): the number of trainers
        window_seconds (int): Throughput use time-based window for window_throughput. This
                              argument specify the window size in seconds.
        warmup_steps (int): the number of warmup batches. No Throughput will be calculated
                            before the warmup batches count reached.

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example::

        throughput = ThroughputMetric(
                      batch_size=128,
                      world_size=4,
                      window_seconds=100,
                      warmup_steps=100
                  )
    """

    _namespace: MetricNamespace = MetricNamespace.THROUGHPUT
    _metric_name: MetricName = MetricName.THROUGHPUT
    _window_seconds: int
    _warmup_steps: int
    _window_time_lapse_buffer: Deque[float]
    _window_time_lapse: float
    _previous_ts: float
    _lifetime_throughput_key: str
    _window_throughput_key: str
    _attempt_throughput_key: str
    _total_examples_key: str
    _attempt_examples_key: str
    _batch_size_key: str
    _steps: int

    def __init__(
        self,
        *,
        batch_size: int,
        world_size: int,
        window_seconds: int,
        warmup_steps: int = 100,
        batch_size_stages: Optional[List[BatchSizeStage]] = None,
    ) -> None:
        super().__init__()
        if window_seconds < 1:
            raise ValueError(
                "window_seconds must be at least 1 to give window throughput "
                "the minimum time window"
            )
        if warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be at least 1 to give throughput a "
                "reasonable begin time."
            )

        if window_seconds > MAX_WINDOW_TS:
            logger.warning(
                f"window_seconds is greater than {MAX_WINDOW_TS}, capping to {MAX_WINDOW_TS} to make sure window_qps is not staled"
            )
            window_seconds = MAX_WINDOW_TS

        self._batch_size = batch_size
        self._world_size = world_size
        self._window_seconds = window_seconds
        self._warmup_steps = warmup_steps
        self._batch_size_stages: Optional[List[BatchSizeStage]] = copy.deepcopy(
            batch_size_stages
        )

        if self._batch_size_stages is not None:
            # Keep track of the number of batches if using batch_size_stages
            self._num_batch: int = 0

        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)
        self.register_state_dict_post_hook(self.state_dict_hook)

        self.register_buffer("total_examples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("warmup_examples", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "time_lapse_after_warmup", torch.tensor(0, dtype=torch.double)
        )

        self.register_buffer(
            "attempt_examples", torch.tensor(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "attempt_warmup_examples",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "attempt_time_lapse_after_warmup",
            torch.tensor(0, dtype=torch.double),
            persistent=False,
        )

        self._window_time_lapse_buffer = deque(maxlen=MAX_WINDOW_TS)
        self._window_time_lapse = 0
        self._previous_ts = 0

        self._lifetime_throughput_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            self._metric_name,
            MetricPrefix.LIFETIME,
        )
        self._window_throughput_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            self._metric_name,
            MetricPrefix.WINDOW,
        )
        self._attempt_throughput_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            self._metric_name,
            MetricPrefix.ATTEMPT,
        )
        self._total_examples_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            MetricName.TOTAL_EXAMPLES,
        )
        self._attempt_examples_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            MetricName.ATTEMPT_EXAMPLES,
        )
        self._batch_size_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            MetricName.BATCH_SIZE,
        )
        self._steps = 0

    def _get_batch_size(self) -> int:
        # No batch size stages, use the default batch size
        if not self._batch_size_stages:
            return self._batch_size

        # Get batch size from batch_size_stages
        assert self._num_batch is not None, "num_batch should not be None"
        batch_size_stages = none_throws(self._batch_size_stages)
        while self._batch_size_stages:
            stage = self._batch_size_stages[0]
            # Reach the last stage
            if stage.max_iters is None:
                assert len(batch_size_stages) == 1
                return stage.batch_size
            # This stage finished
            if stage.max_iters < self._num_batch:
                batch_size_stages.pop(0)
                # Move to the next stage
                continue
            # In this stage
            return stage.batch_size
        raise AssertionError("Unreachable, batch_size_stages should always has 1 item")

    def _batch_examples(self) -> int:
        return self._get_batch_size() * self._world_size

    def _check_window(self) -> None:
        while self._window_time_lapse > self._window_seconds:
            self._window_time_lapse -= self._window_time_lapse_buffer.popleft()

    def update(self) -> None:
        ts = time.monotonic()
        self._steps += 1
        if self._batch_size_stages is not None:
            self._num_batch += 1
        batch_examples = self._batch_examples()
        # pyrefly: ignore [bad-argument-type, unsupported-operation]
        self.total_examples += batch_examples
        # pyrefly: ignore [bad-argument-type, unsupported-operation]
        self.attempt_examples += batch_examples

        if self._steps <= self._warmup_steps:
            # pyrefly: ignore [bad-argument-type, unsupported-operation]
            self.warmup_examples += batch_examples
            # pyrefly: ignore [bad-argument-type, unsupported-operation]
            self.attempt_warmup_examples += batch_examples
            if self._steps == self._warmup_steps:
                self._previous_ts = ts
        else:
            time_lapse = ts - self._previous_ts
            # pyrefly: ignore [bad-argument-type, unsupported-operation]
            self.time_lapse_after_warmup += time_lapse
            # pyrefly: ignore [bad-argument-type, unsupported-operation]
            self.attempt_time_lapse_after_warmup += time_lapse
            self._window_time_lapse += time_lapse
            self._window_time_lapse_buffer.append(time_lapse)
            self._check_window()
            self._previous_ts = ts

    def compute(self) -> Dict[str, torch.Tensor]:
        ret = {
            self._total_examples_key: self.total_examples,
            self._attempt_examples_key: self.attempt_examples,
        }
        if self._steps > self._warmup_steps and (
            # pyrefly: ignore[not-callable]
            not math.isclose(self.time_lapse_after_warmup.item(), 0)
        ):
            # pyrefly: ignore[unsupported-operation]
            lifetime_throughput = (
                # pyrefly: ignore[unsupported-operation]
                self.total_examples
                - self.warmup_examples
            ) / self.time_lapse_after_warmup
            # pyrefly: ignore[unsupported-operation]
            attempt_throughput = (
                # pyrefly: ignore[unsupported-operation]
                self.attempt_examples
                - self.attempt_warmup_examples
            ) / self.attempt_time_lapse_after_warmup
            if not math.isclose(self._window_time_lapse, 0):
                window_throughput = (
                    len(self._window_time_lapse_buffer)
                    * self._batch_examples()
                    / self._window_time_lapse
                )
            else:
                window_throughput = 0.0
            if not math.isclose(lifetime_throughput.item(), 0):
                ret.update(
                    {
                        self._lifetime_throughput_key: lifetime_throughput.clone().detach(),
                        self._window_throughput_key: torch.tensor(
                            window_throughput, dtype=torch.double
                        ),
                    }
                )
            if not math.isclose(attempt_throughput.item(), 0):
                ret.update(
                    {
                        self._attempt_throughput_key: attempt_throughput.clone().detach(),
                    }
                )
        # If using batch_size_stages, also report the current batch size
        # that was used for the throughput calculation
        if self._batch_size_stages is not None:
            ret.update(
                {
                    self._batch_size_key: torch.tensor(
                        self._get_batch_size(), dtype=torch.int32
                    ),
                }
            )

        # pyrefly: ignore[bad-return]
        return ret

    @staticmethod
    def state_dict_hook(
        module: nn.Module,
        state_dict: OrderedDict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
    ) -> None:
        if module._batch_size_stages is not None:
            # Save the number of batches used for the throughput calculation to the state dict
            num_batch_key = f"{prefix}num_batch"
            state_dict[num_batch_key] = torch.tensor(
                module._num_batch, dtype=torch.long
            )

    def load_state_dict_hook(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        key = f"{prefix}num_batch"
        if key in state_dict:
            # If present, pop the number of batches used for the throughput calculation from the state dict
            num_batch_tensor = state_dict.pop(key)
            # Apply the number of batches to the module if using batch_size_stages
            if self._batch_size_stages is not None:
                self._num_batch = int(num_batch_tensor.item())
