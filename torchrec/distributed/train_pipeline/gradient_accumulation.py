#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Gradient Accumulation support for TorchRec Train Pipelines.

This module provides:
1. GradientAccumulationConfig - Configuration dataclass for GA settings
2. GradientAccumulationWrapper - Wrapper that adds GA to any TrainPipeline
"""

import contextlib
from dataclasses import dataclass
from typing import Any, ContextManager, Generic, Iterator, Optional, TYPE_CHECKING

import torch
from torchrec.distributed.train_pipeline.pipeline_context import In, Out

if TYPE_CHECKING:
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline


@dataclass
class GradientAccumulationConfig:
    """
    Configuration for gradient accumulation.

    Attributes:
        is_enabled: Whether gradient accumulation is enabled.
        num_steps: Number of micro-batches to accumulate before optimizer step.
        num_warmup_steps: Number of warmup steps where all iterations sync.
    """

    is_enabled: bool = False
    num_steps: int = 1
    num_warmup_steps: int = 1

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.num_warmup_steps < 1:
            raise ValueError(
                f"num_warmup_steps must be >= 1, got {self.num_warmup_steps}. "
                "At least 1 warmup step is required for DDP static_graph compatibility."
            )
        # Auto-enable if num_steps > 1
        if self.num_steps > 1 and not self.is_enabled:
            self.is_enabled = True


class _GAOptimizerWrapper:
    """
    Internal optimizer wrapper that intercepts zero_grad() and step() calls.

    This wrapper controls when the actual optimizer step is executed based on
    the accumulation schedule.

    The wrapper uses a _needs_zero_grad flag to ensure proper timing of
    zero_grad calls regardless of pipeline execution order.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: GradientAccumulationConfig,
    ) -> None:
        self._optimizer = optimizer
        self._config = config
        self._current_step: int = 0
        self._needs_zero_grad: bool = True

    def _should_step(self) -> bool:
        """Returns True if optimizer.step() should actually execute."""
        return (self._current_step + 1) % self._config.num_steps == 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Intercepts zero_grad to only clear gradients at accumulation boundaries.

        Uses the _needs_zero_grad flag to ensure proper timing regardless of
        when zero_grad is called in the pipeline execution order.
        """
        if self._needs_zero_grad:
            self._optimizer.zero_grad(set_to_none=set_to_none)
            self._needs_zero_grad = False

    def step(self, *args: Any, **kwargs: Any) -> None:
        """
        Intercepts step to execute only at accumulation boundaries.
        """
        if self._should_step():
            self._optimizer.step(*args, **kwargs)
            self._needs_zero_grad = True

    def advance_step(self) -> None:
        """Advances the internal step counter."""
        self._current_step += 1

    def reset(self) -> None:
        """Resets the internal step counter and zero_grad flag."""
        self._current_step = 0
        self._needs_zero_grad = True

    def set_step(self, step: int) -> None:
        """Sets the internal step counter. Use this instead of directly modifying _current_step."""
        self._current_step = step

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the wrapped optimizer."""
        return getattr(self._optimizer, name)


class GradientAccumulationWrapper(Generic[In, Out]):
    """
    Wrapper that adds gradient accumulation to any TrainPipeline.

    This wrapper:
    - Intercepts the optimizer to control zero_grad and step timing
    - Manages no_sync context for DDP to skip gradient synchronization

    Example:
        >>> config = GradientAccumulationConfig(is_enabled=True, num_steps=4)
        >>> pipeline = TrainPipelineSparseDist(model, optimizer, device)
        >>> wrapped = GradientAccumulationWrapper(pipeline, optimizer, model, config)
        >>> for batch in dataloader:
        >>>     loss = wrapped.progress(iter([batch]))
    """

    def __init__(
        self,
        pipeline: "TrainPipeline[In, Out]",
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        config: GradientAccumulationConfig,
    ) -> None:
        self._pipeline = pipeline
        self._model = model
        self._config = config
        self._optimizer_wrapper = _GAOptimizerWrapper(optimizer, config)

        # Only replace optimizer in pipeline when GA is enabled
        # This avoids unintended side effects when GA is disabled
        if config.is_enabled and hasattr(pipeline, "_optimizer"):
            # pyre-ignore[16]: pipeline may not have _optimizer
            pipeline._optimizer = self._optimizer_wrapper

    def _should_sync_grad(self, is_last_batch: bool = False) -> bool:
        """
        Determines if gradient synchronization should happen.

        Returns True on the last step of accumulation, if warmup is not complete,
        or on the very first step (required for DDP static_graph compatibility,
        see https://fb.workplace.com/groups/1922750938494298/permalink/25911539665113154/).
        """
        if is_last_batch:
            return True

        # Always sync on the first step. DDP with static_graph=True requires
        # gradient synchronization on the first iteration to initialize its
        # internal state. Using no_sync() on the first step causes an error
        # in Reducer::finalize_backward() because prepare_for_backward() was
        # never called. This check is intentionally separate from warmup to
        # make the requirement explicit.
        if self.current_step == 0:
            return True

        # During warmup, always sync
        if self.current_step < self._config.num_warmup_steps:
            return True

        # Sync on the last step of each accumulation cycle
        return (self.current_step + 1) % self._config.num_steps == 0

    def _get_no_sync_context(self) -> ContextManager[None]:
        """
        Returns the no_sync context manager for the model, or nullcontext if unavailable.
        """
        model = self._model

        # Check for DMP-wrapped DDP
        if hasattr(model, "_dmp_wrapped_module"):
            # pyre-ignore[16]: model may not have _dmp_wrapped_module
            dmp = model._dmp_wrapped_module
            if hasattr(dmp, "no_sync"):
                return dmp.no_sync()

        # Check for direct DDP
        if hasattr(model, "no_sync"):
            # pyre-ignore[29]: model is typed as nn.Module; no_sync exists on DDP subclasses
            return model.no_sync()

        return contextlib.nullcontext()

    def _flush_accumulated_gradients(self, steps_accumulated: int) -> bool:
        """
        Force a gradient sync and optimizer step for any remaining gradients.

        Args:
            steps_accumulated: Number of micro-batches accumulated so far.
                This is passed explicitly to ensure consistent behavior regardless
                of when flush is called (before or after _advance_state).

        Returns:
            True if gradients were flushed, False if no flush was needed.
        """
        remaining = steps_accumulated % self._config.num_steps
        if remaining > 0:
            # Step, zero gradients to prevent stale state, and reset flag
            self._optimizer_wrapper._optimizer.step()
            self._optimizer_wrapper._optimizer.zero_grad(set_to_none=True)
            self._optimizer_wrapper._needs_zero_grad = False
            return True
        return False

    def _advance_state(self) -> None:
        """Advances internal state after each progress call."""
        self._optimizer_wrapper.advance_step()

    def progress(
        self, dataloader_iter: Iterator[In], is_last_batch: Optional[bool] = None
    ) -> Out:
        """
        Runs one step of the training pipeline with gradient accumulation.

        Args:
            dataloader_iter: Iterator providing input batches.
            is_last_batch: Optional flag to indicate this is the last batch.
                When True, forces gradient sync and optimizer step.
                When None (default), relies on StopIteration for detection.

        Returns:
            Output from the wrapped pipeline's progress call.

        Raises:
            StopIteration: When the dataloader is exhausted. Flushes any
                remaining accumulated gradients before raising.
        """
        if not self._config.is_enabled:
            return self._pipeline.progress(dataloader_iter)

        should_sync = self._should_sync_grad(is_last_batch=is_last_batch or False)
        ctx: ContextManager[None] = (
            contextlib.nullcontext() if should_sync else self._get_no_sync_context()
        )

        try:
            with ctx:
                result = self._pipeline.progress(dataloader_iter)
        except StopIteration:
            # When StopIteration is raised, pipeline.progress() had no batch
            # to process — no forward, backward, or optimizer step happened in
            # this call. In TrainPipelineSparseDist, StopIteration is raised at
            # the top of progress() when self.batches is empty (all prefetched
            # batches were already fully processed in prior calls). Therefore
            # current_step accurately reflects the number of completed batches
            # and we should NOT add +1.
            self._flush_accumulated_gradients(self.current_step)
            raise

        self._advance_state()

        # If user explicitly marked this as last batch, flush remaining gradients
        # Use current_step which now includes this batch after _advance_state()
        if is_last_batch:
            self._flush_accumulated_gradients(self.current_step)

        return result

    def reset(self) -> None:
        """Resets the wrapper and underlying pipeline state."""
        self._optimizer_wrapper.reset()
        if hasattr(self._pipeline, "reset"):
            self._pipeline.reset()

    @property
    def optimizer_wrapper(self) -> _GAOptimizerWrapper:
        """Returns the optimizer wrapper for testing/inspection."""
        return self._optimizer_wrapper

    @property
    def current_step(self) -> int:
        """Returns the current step count (single source of truth from optimizer wrapper)."""
        return self._optimizer_wrapper._current_step

    def set_step(self, step: int) -> None:
        """
        Sets the current step counter.

        Use this method instead of directly manipulating internal state
        to ensure proper synchronization.
        """
        self._optimizer_wrapper.set_step(step)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped pipeline."""
        # This is called when the attribute is not found on the wrapper itself.
        # We delegate to the wrapped pipeline to support attributes like 'metrics'.
        return getattr(self._pipeline, name)
