#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Gradient accumulation for TorchRec training pipelines."""

import contextlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    cast,
    ContextManager,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TYPE_CHECKING,
)

import torch
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.train_pipeline.bucket_view_zeroing import BucketViewZeroing
from torchrec.distributed.train_pipeline.pipeline_context import In, Out

if TYPE_CHECKING:
    from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline


class GAWindowObserver(Protocol):
    """Receive accumulation boundaries before each inner ``progress()`` call.

    Pipeline operations that bypass the wrapped optimizer use this callback to share its
    boundary decision. The callback is not invoked when accumulation is disabled.
    """

    def __call__(self, *, should_step: bool, at_window_start: bool) -> None: ...


class _GAPipelineHooks(Protocol):
    """Methods used by the wrapper but not declared by ``TrainPipeline``."""

    def attach(
        self, model: Optional[torch.nn.Module] = None, *args: Any, **kwargs: Any
    ) -> Any: ...


logger: logging.Logger = logging.getLogger(__name__)


def _ga_abort_all_process_groups(reason: str) -> None:
    """Abort process groups before raising on a distributed boundary.

    Without the abort, other ranks may block in their next collective until the watchdog
    times out. The abort is best-effort and is skipped for a single process.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if torch.distributed.get_world_size() <= 1:
        return
    logger.error(
        f"[gradient_accumulation] aborting all process groups before raising: {reason}"
    )
    try:
        torch.distributed.distributed_c10d._abort_process_group(None)
    except Exception:
        # Report that other ranks may remain blocked if the abort fails.
        logger.exception(
            "[gradient_accumulation] could not abort process groups; other ranks may "
            "remain blocked after this rank raises"
        )


class PartialWindowPolicy(Enum):
    """Action when input ends before an accumulation window is complete.

    ``STEP`` applies the partial gradients. With multiple ranks, those gradients were not
    synchronized. ``RAISE`` rejects the partial window and first aborts distributed
    process groups. ``DISCARD`` clears the gradients and is safe only when every rank
    exhausts together.
    """

    STEP = "step"
    RAISE = "raise"
    DISCARD = "discard"


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation.

    Attributes:
        is_enabled: Whether gradient accumulation is enabled.
        num_steps: Number of micro-batches to accumulate before optimizer step.
        num_warmup_steps: Number of initial micro-batches that synchronize gradients.
            The default synchronizes only the first micro-batch, as required by DDP
            ``static_graph`` initialization.
    """

    is_enabled: bool = False
    num_steps: int = 1
    num_warmup_steps: int = 1
    # Preserve DDP bucket aliases by zeroing eligible dense gradients in place. An unused
    # parameter retains a zero gradient instead of None, which can affect optimizers that
    # distinguish those states.
    accumulate_into_buckets: bool = False

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.num_warmup_steps < 1:
            raise ValueError(
                f"num_warmup_steps must be >= 1, got {self.num_warmup_steps}. "
                "At least 1 warmup step is required for DDP static_graph compatibility."
            )
        if self.num_steps > 1 and not self.is_enabled:
            self.is_enabled = True


class _GAOptimizerWrapper:
    """Control optimizer clearing and stepping across accumulation windows."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: GradientAccumulationConfig,
    ) -> None:
        self._optimizer = optimizer
        self._config = config
        # Use one stored window size for every boundary calculation.
        self._k: int = config.num_steps
        self._current_step: int = 0
        # Track where the current window starts without resetting the global counter.
        self._window_base: int = 0
        self._needs_zero_grad: bool = True
        # Allow an explicit final partial batch to step before the scheduled boundary.
        self._force_step: bool = False
        # The owning wrapper provides model topology for bucket-view zeroing.
        self._ga_wrapper: Optional["GradientAccumulationWrapper[Any, Any]"] = None

    @property
    def micro_batches_into_window(self) -> int:
        """Number of processed micro-batches in the current window."""
        return (self._current_step - self._window_base) % self._k

    def _should_step(self) -> bool:
        """Return whether the next optimizer step should run."""
        return (self.micro_batches_into_window + 1) % self._k == 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients once at the start of each accumulation window.

        Bucket-view gradients are zeroed in place when ``accumulate_into_buckets`` is
        enabled.
        """
        if not self._needs_zero_grad:
            return
        if self._config.accumulate_into_buckets and self._ga_wrapper is not None:
            self._ga_wrapper._window_start_zero_grad()
        else:
            self._optimizer.zero_grad(set_to_none=set_to_none)
        self._needs_zero_grad = False

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Run the optimizer at a scheduled or explicitly forced boundary."""
        if self._should_step() or self._force_step:
            self._optimizer.step(*args, **kwargs)
            self._needs_zero_grad = True

    def advance_step(self) -> None:
        """Advance the global micro-batch counter."""
        self._current_step += 1

    def reset(self) -> None:
        """Reset the counter and gradient-clear state."""
        self._current_step = 0
        self._window_base = 0
        self._needs_zero_grad = True
        self._force_step = False

    def realign_window(self) -> None:
        """Start a new window without resetting the global micro-batch counter."""
        self._window_base = self._current_step

    def set_step(self, step: int) -> None:
        """Set the micro-batch counter and reset the window origin.

        Resetting the origin restores the configured boundary schedule after a previous
        realignment. Negative values remain accepted for backward compatibility.
        """
        self._current_step = step
        self._window_base = 0

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the wrapped optimizer."""
        return getattr(self._optimizer, name)


class GradientAccumulationWrapper(Generic[In, Out]):
    """Add gradient accumulation to a ``TrainPipeline``.

    The wrapper controls optimizer boundaries and suppresses distributed gradient
    synchronization between those boundaries.
    """

    def __init__(
        self,
        pipeline: "TrainPipeline[In, Out]",
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        config: GradientAccumulationConfig,
        partial_window_policy: PartialWindowPolicy = PartialWindowPolicy.STEP,
        window_observer: Optional[GAWindowObserver] = None,
    ) -> None:
        self._pipeline = pipeline
        self._model = model
        self._config = config
        # Share accumulation boundaries with pipeline operations that bypass this optimizer.
        self._window_observer = window_observer
        self._partial_window_policy = partial_window_policy
        self._optimizer_wrapper = _GAOptimizerWrapper(optimizer, config)
        self._optimizer_wrapper._ga_wrapper = self
        self._cached_ddp_modules: list[Any] | None = None
        # Track gradients from an incomplete window for exhaustion handling.
        self._pending_uncommitted: bool = False
        # Set after a synchronized backward establishes DDP bucket-view aliases.
        self._bucket_views_ready: bool = False
        self._bucket_view_zeroing: BucketViewZeroing = BucketViewZeroing()

        # Only replace the pipeline's optimizer when GA is enabled.
        if config.is_enabled:
            if not hasattr(pipeline, "_optimizer"):
                raise RuntimeError(
                    f"{type(pipeline).__name__} has no _optimizer attribute required "
                    "for gradient accumulation"
                )
            if isinstance(getattr(pipeline, "_optimizer", None), _GAOptimizerWrapper):
                raise RuntimeError(
                    f"{type(pipeline).__name__} already uses a gradient-accumulation "
                    "optimizer; wrap each pipeline only once"
                )
            # pyrefly: ignore[missing-attribute]: pipeline may not have _optimizer
            pipeline._optimizer = self._optimizer_wrapper

    def _should_sync_grad(self, is_last_batch: bool = False) -> bool:
        """Return whether this batch should synchronize distributed gradients."""
        if is_last_batch:
            return True

        # DDP static_graph requires a synchronized first backward to initialize the reducer.
        if self.current_step == 0:
            return True

        if self.current_step < self._config.num_warmup_steps:
            return True

        # Use the optimizer wrapper's boundary calculation after any realignment.
        return self._optimizer_wrapper._should_step()

    def _get_no_sync_context(self) -> ContextManager[None]:
        """Enter ``no_sync()`` on each registered DDP module.

        DDP modules stored in plain Python containers are not registered and therefore
        continue synchronizing every micro-batch.
        """
        return self._compose_no_sync_contexts()

    def _get_ddp_modules(self) -> list[Any]:
        """Cache modules whose ``no_sync()`` context this wrapper manages.

        The root may be DDP, FSDP, or another wrapper with ``no_sync()``. Descendants must
        be ``DistributedDataParallel`` instances.
        """
        if self._cached_ddp_modules is not None:
            return self._cached_ddp_modules

        ddp_modules: list[Any] = []
        model = self._model

        # Use the wrapped module as the discovery root when available.
        root: torch.nn.Module = model
        if hasattr(model, "_dmp_wrapped_module"):
            dmp_wrapped = model._dmp_wrapped_module
            if isinstance(dmp_wrapped, torch.nn.Module):
                root = dmp_wrapped
            elif hasattr(dmp_wrapped, "no_sync"):
                ddp_modules.append(dmp_wrapped)

        # The root may be DDP, FSDP, or another wrapper that supports ``no_sync``.
        if hasattr(root, "no_sync"):
            ddp_modules.append(root)

        # Collect registered DDP descendants only.
        if hasattr(root, "modules"):
            for module in root.modules():
                if module is not root and isinstance(module, DistributedDataParallel):
                    ddp_modules.append(module)

        self._cached_ddp_modules = ddp_modules
        return ddp_modules

    @contextlib.contextmanager
    # pyre-ignore[3]: Return type must be annotated
    def _compose_no_sync_contexts(self):
        """Enter every cached ``no_sync()`` context using ``ExitStack``."""
        ddp_modules = self._get_ddp_modules()

        if not ddp_modules:
            yield
            return

        with contextlib.ExitStack() as stack:
            for ddp in ddp_modules:
                stack.enter_context(ddp.no_sync())
            yield

    def _window_start_zero_grad(self) -> None:
        """Clear gradients without dropping DDP bucket-view aliases.

        Before aliases exist, or when no gradients are eligible, this uses the default
        ``set_to_none`` behavior.
        """
        optimizer = self._optimizer_wrapper._optimizer
        if not self._bucket_views_ready:
            optimizer.zero_grad(set_to_none=True)
            return
        targets = self._bucket_view_zeroing.collect(self._get_ddp_modules(), optimizer)
        if not targets:
            optimizer.zero_grad(set_to_none=True)
            return
        self._bucket_view_zeroing.zero(optimizer, targets)

    def _flush_accumulated_gradients(self, steps_accumulated: int) -> bool:
        """Handle an incomplete window at exhaustion and report whether it stepped."""
        remaining = steps_accumulated % self.num_micro_batches_per_step
        if remaining > 0:
            if self._partial_window_policy is PartialWindowPolicy.DISCARD:
                self._optimizer_wrapper._needs_zero_grad = True
                self._optimizer_wrapper.zero_grad(set_to_none=True)
                logger.warning(
                    "Discarded a partial gradient-accumulation window "
                    "(steps_accumulated=%d, num_steps=%d, remaining=%d) under "
                    "PartialWindowPolicy.DISCARD. Cleared %d micro-batch(es) without an "
                    "optimizer step. A later collective may time out if ranks did not "
                    "exhaust together.",
                    steps_accumulated,
                    self.num_micro_batches_per_step,
                    remaining,
                    remaining,
                )
                return False
            multi_rank = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if self._partial_window_policy is PartialWindowPolicy.RAISE:
                if multi_rank:
                    # Abort first so other ranks do not wait in their next collective.
                    _ga_abort_all_process_groups(
                        f"partial final window at world_size>1 "
                        f"(steps_accumulated={steps_accumulated}, remaining={remaining})"
                    )
                    raise RuntimeError(
                        "cannot apply a partial gradient-accumulation window "
                        f"(steps_accumulated={steps_accumulated}, num_steps="
                        f"{self.num_micro_batches_per_step}, remaining={remaining}) across "
                        "multiple ranks because its gradients were not synchronized. "
                        "Use a divisible batch count, pass is_last_batch=True for a "
                        "synchronized final step, or use DISCARD only when all ranks "
                        "exhaust together."
                    )
                raise RuntimeError(
                    "partial gradient-accumulation window violates "
                    "PartialWindowPolicy.RAISE "
                    f"(steps_accumulated={steps_accumulated}, num_steps="
                    f"{self.num_micro_batches_per_step}, remaining={remaining}). Use a "
                    "divisible batch count or select PartialWindowPolicy.STEP."
                )
            if multi_rank:
                logger.warning(
                    "Applying a partial gradient-accumulation window across multiple ranks "
                    "(steps_accumulated=%d, num_steps=%d, remaining=%d) under "
                    "PartialWindowPolicy.STEP. The %d micro-batch(es) were not synchronized, "
                    "so model replicas will diverge. Pass is_last_batch=True for a "
                    "synchronized final step, or use DISCARD when all ranks exhaust "
                    "together.",
                    steps_accumulated,
                    self.num_micro_batches_per_step,
                    remaining,
                    remaining,
                )
            self._optimizer_wrapper._optimizer.step()
            self._optimizer_wrapper._needs_zero_grad = True
            self._optimizer_wrapper.zero_grad(set_to_none=True)
            return True
        return False

    def _advance_state(self) -> None:
        """Advances internal state after each progress call."""
        self._optimizer_wrapper.advance_step()

    def progress(
        self, dataloader_iter: Iterator[In], is_last_batch: Optional[bool] = None
    ) -> Out:
        """Run one pipeline step with gradient accumulation.

        ``is_last_batch=True`` synchronizes and applies a final partial window. On input
        exhaustion, any remaining gradients follow ``PartialWindowPolicy`` before
        ``StopIteration`` is raised.
        """
        if not self._config.is_enabled:
            return self._pipeline.progress(dataloader_iter)

        should_sync = self._should_sync_grad(is_last_batch=is_last_batch or False)
        # Reject an explicit partial window before the optimizer runs.
        if (
            is_last_batch
            and self._partial_window_policy is PartialWindowPolicy.RAISE
            and getattr(self._model, "training", True)
            and not self._optimizer_wrapper._should_step()
        ):
            steps_accumulated = self.micro_batches_into_window + 1
            remaining = steps_accumulated % self.num_micro_batches_per_step
            _ga_abort_all_process_groups(
                f"partial final window via is_last_batch under PartialWindowPolicy.RAISE "
                f"(steps_accumulated={steps_accumulated}, remaining={remaining})"
            )
            raise RuntimeError(
                "partial gradient-accumulation window violates "
                "PartialWindowPolicy.RAISE "
                f"(steps_accumulated={steps_accumulated}, num_steps="
                f"{self.num_micro_batches_per_step}, remaining={remaining}). Use a "
                "divisible batch count or select PartialWindowPolicy.STEP."
            )
        # Publish the optimizer boundary before pipeline operations run.
        should_step = self._optimizer_wrapper._should_step() or bool(is_last_batch)
        # An explicit final batch may step before the scheduled boundary.
        self._optimizer_wrapper._force_step = bool(is_last_batch)
        # The counter advances after progress(), so zero marks the window start.
        at_window_start = (
            self.micro_batches_into_window % self.num_micro_batches_per_step
        ) == 0
        if self._window_observer is not None and getattr(self._model, "training", True):
            self._window_observer(
                should_step=should_step, at_window_start=at_window_start
            )
        ctx: ContextManager[None] = (
            contextlib.nullcontext() if should_sync else self._get_no_sync_context()
        )

        try:
            with ctx:
                result = self._pipeline.progress(dataloader_iter)
        except StopIteration:
            # No batch was processed, so current_step already reflects completed batches.
            if getattr(self._model, "training", True):
                if self._pending_uncommitted:
                    self._flush_accumulated_gradients(self.micro_batches_into_window)
                    self._pending_uncommitted = False
                # Start the next training phase with a new accumulation window.
                self._optimizer_wrapper.realign_window()
            raise

        # Split optimizers bypass _GAOptimizerWrapper.step(), so update its clear state here.
        if getattr(self._model, "training", True):
            if should_step:
                self._optimizer_wrapper._needs_zero_grad = True
            self._pending_uncommitted = not should_step
            # Evaluation must not advance the training accumulation window.
            self._advance_state()
            if is_last_batch:
                self._optimizer_wrapper.realign_window()
            if should_sync and self._config.accumulate_into_buckets:
                self._bucket_views_ready = True

        return result

    def reset(self, drop_partial: bool = False) -> None:
        """Resets the wrapper and underlying pipeline state.

        Bucket-view readiness is retained because the model and DDP instances do not
        change across reset.
        """
        # A forced final step may end between boundaries without leaving pending gradients.
        if self._pending_uncommitted and not drop_partial:
            # @lint-ignore FIXIT AllRaisesAreAIExceptions
            raise RuntimeError(
                "cannot reset with uncommitted gradients ("
                f"current_step={self.current_step}, "
                f"num_steps={self.num_micro_batches_per_step}). "
                "Complete the current window or pass drop_partial=True to discard it."
            )
        self._optimizer_wrapper.reset()
        self._pending_uncommitted = False
        if hasattr(self._pipeline, "reset"):
            self._pipeline.reset()

    def attach(
        self, model: Optional[torch.nn.Module] = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Delegate attachment while requiring the original model.

        A different model needs a new wrapper because the optimizer and cached DDP topology
        belong to the original model.
        """
        if model is not None and model is not self._model:
            raise RuntimeError(
                "attach() cannot replace the model used by GradientAccumulationWrapper; "
                "construct a new wrapper for the new model and optimizer"
            )
        if not hasattr(  # @lint-ignore FIXIT [AvoidHasattrEverywhere] the wrapped pipeline is an unconstrained TrainPipeline; attach() is optional on it
            self._pipeline, "attach"
        ):
            raise RuntimeError(
                f"{type(self._pipeline).__name__} does not provide attach(), which is "
                "required by GradientAccumulationWrapper"
            )
        return self._ga_hooks.attach(model, *args, **kwargs)

    @property
    def _ga_hooks(self) -> _GAPipelineHooks:
        """Return the wrapped pipeline with its optional hook methods typed."""
        return cast(_GAPipelineHooks, self._pipeline)

    @property
    def optimizer_wrapper(self) -> _GAOptimizerWrapper:
        """Returns the optimizer wrapper for testing/inspection."""
        return self._optimizer_wrapper

    @property
    def current_step(self) -> int:
        """Returns the current step count (single source of truth from optimizer wrapper)."""
        return self._optimizer_wrapper._current_step

    @property
    def micro_batches_into_window(self) -> int:
        """Number of processed micro-batches in the current accumulation window."""
        return self._optimizer_wrapper.micro_batches_into_window

    @property
    def num_micro_batches_per_step(self) -> int:
        """Number of micro-batches accumulated per optimizer step."""
        return self._optimizer_wrapper._k

    def set_step(self, step: int) -> None:
        """
        Sets the current step counter.

        Use this method instead of directly manipulating internal state
        to ensure proper synchronization.
        """
        self._optimizer_wrapper.set_step(step)

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped pipeline."""
        return getattr(self._pipeline, name)
