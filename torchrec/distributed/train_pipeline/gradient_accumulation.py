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
    """Notified once per micro-batch, immediately before the inner ``progress()``.

    For pipelines whose sparse/dense sub-steps or grad-clip bypass the GA-wrapped
    optimizer and so must gate on the same boundary the wrapper uses. Supplied explicitly
    at wrapper construction: the wrapper never writes state onto the pipeline it wraps.
    Keyword-only so the two booleans cannot be transposed.

    Not called when GA is disabled -- that path is a pure pass-through, and the observer's
    owner keeps whatever default it initialised.
    """

    def __call__(self, *, should_step: bool, at_window_start: bool) -> None: ...


class _GAPipelineHooks(Protocol):
    """The surface the GA wrapper drives on the pipeline it wraps.

    ``TrainPipeline`` does not declare ``attach``, so naming the surface lets the one call
    site be a plain attribute access instead of ``getattr``.
    """

    def attach(
        self, model: Optional[torch.nn.Module] = None, *args: Any, **kwargs: Any
    ) -> Any: ...


logger: logging.Logger = logging.getLogger(__name__)


def _ga_abort_all_process_groups(reason: str) -> None:
    """Tear down every process group before a rank-local raise on a collective boundary.

    Peers would otherwise block in the next collective until the NCCL watchdog fires, with
    the timeout masking the real cause. Best-effort and NCCL-only; elsewhere the abort may
    fail and the raise still stands. Not gated on ``get_backend()``, which reports only the
    default group while the abort covers all of them. No-op at ``world_size <= 1``.
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
        # `_abort_process_group(None)` raises AssertionError when there is no default
        # group. Log loudly: peers were NOT torn down, so fail-closed does not hold.
        logger.exception(
            "[gradient_accumulation] _abort_process_group FAILED -- peers were NOT torn "
            "down; the raise that follows is rank-local and may strand them"
        )


class PartialWindowPolicy(Enum):
    """Caller policy for a partial (r < K) final window.

    - ``STEP`` (default): step it, warning that the un-reduced tail diverges
      replicas above one rank. Invisible to metrics and checkpoint progress
      (``progress()`` re-raises first); ``is_last_batch=True`` is reduced instead.
    - ``RAISE``: fail closed, aborting the process groups first above one rank.
      Also fences the explicit ``is_last_batch`` path; no other policy does.
    - ``DISCARD``: drop rank-locally on exhaustion. Needs out-of-band agreement
      that every rank exhausted; unlocks consume-all (``num_batches < 0``).
    """

    STEP = "step"
    RAISE = "raise"
    DISCARD = "discard"


@dataclass
class GradientAccumulationConfig:
    """
    Configuration for gradient accumulation.

    Attributes:
        is_enabled: Whether gradient accumulation is enabled.
        num_steps: Number of micro-batches to accumulate before optimizer step.
            The pipeline provider builds this independently of the K the train
            module derives, and the two are asserted equal -- so this is not the
            operator-facing setting.
        num_warmup_steps: Number of warmup MICRO-steps (NOT optimizer steps)
            during which every iteration syncs gradients. Counted against
            the global micro counter (current_step), so with num_steps=K a value
            of W means the first W micro-batches sync (~the first ceil(W/K)
            windows). Default 1 (only the very first micro).
    """

    is_enabled: bool = False
    num_steps: int = 1
    num_warmup_steps: int = 1
    # Zero dense grads in place at window-start instead of set_to_none=True, preserving the
    # DDP gradient_as_bucket_view alias so autograd accumulates into the persistent reduction
    # bucket rather than allocating a standalone dense .grad set held across the K-micro
    # window (~1x dense-grad-size of extra HBM). Semantic delta vs OFF: a dense param holding
    # a live bucket-view grad at window start but unused through the window keeps a present
    # zero instead of None, so a dense optimizer sees participate-on-zero rather than skip.
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
        # The live K, and the single source of truth for every boundary computation.
        # ``config.num_steps`` is only the seed.
        self._k: int = config.num_steps
        self._current_step: int = 0
        # Anchor of the current K-window. Every boundary is relative to this rather than to
        # ``_current_step``, so ``realign_window`` restarts a window without breaking the
        # counter's monotonic contract or replaying ``num_warmup_steps``.
        self._window_base: int = 0
        self._needs_zero_grad: bool = True
        # One-shot: when True, step() fires even off the schedule boundary, forcing an
        # in-band optimizer step on an explicit final partial window (r<K).
        self._force_step: bool = False
        # Backref to the owning wrapper, set just after construction. Under
        # ``accumulate_into_buckets`` the window-start zero delegates there because it needs
        # the model's DDP topology; ``None`` standalone (unit tests) takes the plain path.
        self._ga_wrapper: Optional["GradientAccumulationWrapper[Any, Any]"] = None

    @property
    def micro_batches_into_window(self) -> int:
        """Micro-batches accumulated into the current window, bounded to ``[0, num_steps)``.

        Bounded because every consumer reduces mod ``num_steps`` anyway, and because the raw
        span reads far outside ``[0, K)`` after a re-anchor while describing a window that
        holds only its residue.
        """
        return (self._current_step - self._window_base) % self._k

    def _should_step(self) -> bool:
        """Returns True if optimizer.step() should actually execute."""
        return (self.micro_batches_into_window + 1) % self._k == 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients only at accumulation boundaries, tracked by ``_needs_zero_grad``
        so timing does not depend on where the pipeline calls zero_grad. Under
        ``accumulate_into_buckets`` the window-start zero goes to
        ``GradientAccumulationWrapper._window_start_zero_grad``, which preserves the DDP
        ``gradient_as_bucket_view`` alias so autograd accumulates into the persistent
        reduction bucket instead of a standalone dense ``.grad`` set held across the window.
        """
        if not self._needs_zero_grad:
            return
        if self._config.accumulate_into_buckets and self._ga_wrapper is not None:
            # ``set_to_none`` is deliberately not forwarded: the selective protocol fixes its
            # own per-leaf semantics -- tree-clear for non-targets, in-place zero for targets.
            self._ga_wrapper._window_start_zero_grad()
        else:
            self._optimizer.zero_grad(set_to_none=set_to_none)
        self._needs_zero_grad = False

    def step(self, *args: Any, **kwargs: Any) -> None:
        """
        Intercepts step to execute only at accumulation boundaries, or when a
        one-shot forced step is requested (an explicit last batch whose partial
        window is not on the schedule boundary -- see
        GradientAccumulationWrapper.progress).
        """
        if self._should_step() or self._force_step:
            self._optimizer.step(*args, **kwargs)
            self._needs_zero_grad = True

    def advance_step(self) -> None:
        """Advances the internal step counter."""
        self._current_step += 1

    def reset(self) -> None:
        """Resets the internal step counter and zero_grad flag."""
        self._current_step = 0
        self._window_base = 0
        self._needs_zero_grad = True
        self._force_step = False

    def realign_window(self) -> None:
        """Re-anchor the K-window at the current micro counter.

        Called on iterator exhaustion: a phase that consumed a non-multiple of K would
        otherwise leave the counter off-modulo and shift every subsequent window boundary.
        Moving the anchor rather than zeroing ``_current_step`` keeps ``current_step``
        monotonic and keeps ``num_warmup_steps`` counted globally, so warmup is not replayed.
        """
        self._window_base = self._current_step

    def set_step(self, step: int) -> None:
        """Set the internal step counter, and drop the window anchor back to 0.

        Callers place the wrapper at a specific point in the accumulation cycle and expect
        plain ``(step + 1) % K`` semantics; without the anchor reset a preceding
        ``realign_window()`` would leave a stale base. ``step`` is not validated: a negative
        value yields a negative ``current_step``, as it did before the anchor existed.
        """
        self._current_step = step
        self._window_base = 0

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
        partial_window_policy: PartialWindowPolicy = PartialWindowPolicy.STEP,
        window_observer: Optional[GAWindowObserver] = None,
    ) -> None:
        self._pipeline = pipeline
        self._model = model
        self._config = config
        # Boundary signal for a pipeline whose sub-steps bypass the wrapped optimizer. Opt-in
        # rather than inferred, so a rename is a type error not a silent every-micro step.
        self._window_observer = window_observer
        # Caller policy for a partial (r < K) final window, read in
        # _flush_accumulated_gradients and honoured at every world size.
        self._partial_window_policy = partial_window_policy
        self._optimizer_wrapper = _GAOptimizerWrapper(optimizer, config)
        self._optimizer_wrapper._ga_wrapper = self
        self._cached_ddp_modules: list[Any] | None = None
        # True after a non-boundary micro left accumulated-but-un-stepped gradients;
        # gates the StopIteration flush so an in-band-committed window is not re-stepped.
        self._pending_uncommitted: bool = False
        # True once a synchronized training backward has established the DDP
        # ``gradient_as_bucket_view`` aliases. Until then the window-start zero uses a plain
        # ``set_to_none`` clear, which also avoids the post-alias-only detach-on-view crash.
        self._bucket_views_ready: bool = False
        self._bucket_view_zeroing: BucketViewZeroing = BucketViewZeroing()

        # Only replace the pipeline's optimizer when GA is enabled.
        if config.is_enabled:
            if not hasattr(pipeline, "_optimizer"):
                # Necessary, not sufficient: a pipeline can expose ``_optimizer``, accept this
                # replacement, and still step a separately captured reference.
                raise RuntimeError(
                    "Gradient accumulation is enabled but the wrapped pipeline "
                    f"{type(pipeline).__name__} exposes no `_optimizer` attribute, so the "
                    "GA optimizer wrapper cannot be injected and optimizer-step gating to "
                    f"one step per {config.num_steps}-micro window cannot be guaranteed. "
                    "Use a pipeline that exposes `_optimizer` (e.g. "
                    "TrainPipelineSparseDist), or disable gradient accumulation."
                )
            if isinstance(getattr(pipeline, "_optimizer", None), _GAOptimizerWrapper):
                # Nesting the gates under-steps silently rather than crashing.
                raise RuntimeError(
                    "Gradient accumulation is enabled but the wrapped pipeline "
                    f"{type(pipeline).__name__} already has a GA-wrapped `_optimizer`, so "
                    "this pipeline is being wrapped a second time. Nesting the wrappers "
                    "would gate the optimizer to one step per "
                    f"{config.num_steps}**2 micro-batches instead of per "
                    f"{config.num_steps}. Wrap each pipeline exactly once."
                )
            # pyrefly: ignore[missing-attribute]: pipeline may not have _optimizer
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

        # Always sync on the first step: DDP with static_graph=True requires gradient
        # synchronization on the first iteration to initialize its internal state, and
        # no_sync() there fails in Reducer::finalize_backward(). Kept separate from the
        # warmup check so the requirement stays explicit.
        if self.current_step == 0:
            return True

        # During warmup, always sync
        if self.current_step < self._config.num_warmup_steps:
            return True

        # Delegated, not recomputed -- a second copy of the boundary predicate drifts. Unlike
        # the two guards above this is window-relative, so it moves with a re-anchor.
        return self._optimizer_wrapper._should_step()

    def _get_no_sync_context(self) -> ContextManager[None]:
        """Composite ``no_sync()`` over EVERY ``DistributedDataParallel`` in the model tree,
        not just the outermost one. A DDP held in a plain Python list is not in ``_modules``,
        is not found by the ``root.modules()`` walk, and all-reduces every micro-batch --
        correct, but it forgoes the no_sync saving. ``PlainListDDPDiscoveryContractTest``
        guards that.
        """
        return self._compose_no_sync_contexts()

    def _get_ddp_modules(self) -> list[Any]:
        """Discover and cache all modules that need ``no_sync()``.

        The module tree is static after construction, so the walk runs once. Detection is
        deliberately asymmetric: the root is added on ``hasattr(no_sync)`` alone -- it is
        known to be the top-level parallel wrapper, DDP or FSDP or custom -- while
        descendants are arbitrary and need a strict ``DistributedDataParallel`` check.
        """
        if self._cached_ddp_modules is not None:
            return self._cached_ddp_modules

        ddp_modules: list[Any] = []
        model = self._model

        # Unwrap DMP to find the real module tree root.
        root: torch.nn.Module = model
        if hasattr(model, "_dmp_wrapped_module"):
            dmp_wrapped = model._dmp_wrapped_module
            if isinstance(dmp_wrapped, torch.nn.Module):
                root = dmp_wrapped
            elif hasattr(dmp_wrapped, "no_sync"):
                ddp_modules.append(dmp_wrapped)

        # Collect the root if it supports no_sync (broad check: DDP, FSDP, or a custom
        # parallel wrapper).
        if hasattr(root, "no_sync"):
            ddp_modules.append(root)

        # Walk descendants for REGISTERED inner DDPs (present in ``_modules``), strict
        # isinstance so FSDP and other no_sync-bearing modules are not collected. The
        # hasattr guard is needed because root may not be an nn.Module.
        if hasattr(root, "modules"):
            for module in root.modules():
                if module is not root and isinstance(module, DistributedDataParallel):
                    ddp_modules.append(module)

        self._cached_ddp_modules = ddp_modules
        return ddp_modules

    @contextlib.contextmanager
    # pyre-ignore[3]: Return type must be annotated
    def _compose_no_sync_contexts(self):
        """
        Enter ``no_sync()`` on every cached DDP module via ``ExitStack``.

        Contexts are torn down in the correct order even if an exception
        occurs.
        """
        ddp_modules = self._get_ddp_modules()

        if not ddp_modules:
            yield
            return

        with contextlib.ExitStack() as stack:
            for ddp in ddp_modules:
                stack.enter_context(ddp.no_sync())
            yield

    def _window_start_zero_grad(self) -> None:
        """Window-start zero that preserves the DDP ``gradient_as_bucket_view`` alias.

        Only the WHEN lives here. Before the views exist (first window), and whenever no
        leaf is eligible, this is a plain ``set_to_none`` clear -- inert, and identical to
        ``accumulate_into_buckets`` being off.
        """
        optimizer = self._optimizer_wrapper._optimizer
        if not self._bucket_views_ready:
            # First window / views not yet established: plain None clear, which also avoids
            # the detach-on-view crash that only exists post-alias.
            optimizer.zero_grad(set_to_none=True)
            return
        targets = self._bucket_view_zeroing.collect(self._get_ddp_modules(), optimizer)
        if not targets:
            # No live bucket-view DDP targets (e.g. promoted tables on the fp32-grad manual
            # reducer; or a custom/FSDP root; or grads not yet present).
            optimizer.zero_grad(set_to_none=True)
            return
        self._bucket_view_zeroing.zero(optimizer, targets)

    def _flush_accumulated_gradients(self, steps_accumulated: int) -> bool:
        """Resolve an exhaustion-time partial window per ``PartialWindowPolicy``.

        ``steps_accumulated`` is explicit so the result does not depend on whether flush
        runs before or after ``_advance_state``. True only when the window was stepped.
        """
        remaining = steps_accumulated % self.num_micro_batches_per_step
        if remaining > 0:
            if self._partial_window_policy is PartialWindowPolicy.DISCARD:
                # Rank-local drop: no step, no collective, no abort.
                self._optimizer_wrapper._needs_zero_grad = True
                # Re-arm first: zero_grad() early-returns while _needs_zero_grad is False.
                # Through the wrapper, so accumulate_into_buckets alias routing survives.
                self._optimizer_wrapper.zero_grad(set_to_none=True)
                logger.warning(
                    "Gradient accumulation discarded a partial final window "
                    "(steps_accumulated=%d, num_steps=%d, remaining=%d) under "
                    "PartialWindowPolicy.DISCARD: %d micro-batch(es) of accumulated "
                    "gradients were zeroed without an optimizer step. Expected at the end "
                    "of a consume-all phase. If a downstream collective later times out, "
                    "suspect an ASYMMETRIC exhaustion (one rank's reader errored) rather "
                    "than a clean end-of-data.",
                    steps_accumulated,
                    self.num_micro_batches_per_step,
                    remaining,
                    remaining,
                )
                # No reset() (it would replay GA/DDP warmup on every new iterator) and no
                # realign_window() (the caller does that right after this returns).
                return False
            multi_rank = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if self._partial_window_policy is PartialWindowPolicy.RAISE:
                if multi_rank:
                    # Abort before raising: a rank-local raise strands peers in the next
                    # collective until the watchdog fires, burying the real cause.
                    _ga_abort_all_process_groups(
                        f"partial final window at world_size>1 "
                        f"(steps_accumulated={steps_accumulated}, remaining={remaining})"
                    )
                    raise RuntimeError(
                        "Gradient accumulation reached a partial final window "
                        f"(steps_accumulated={steps_accumulated}, num_steps="
                        f"{self.num_micro_batches_per_step}, remaining={remaining}) that was never "
                        "committed in-band. Flushing would step un-reduced rank-local "
                        "gradients (replica divergence), so the caller's selected "
                        "PartialWindowPolicy.RAISE fails closed instead. "
                        "The total reader-batch count is not a whole multiple of "
                        "num_micro_batches_per_step: either it never was, or a checkpoint "
                        "resume left a remainder that no config-time guard can catch. "
                        "Remedies: make the total reader batches a whole multiple of K; or, "
                        "if every rank is known to exhaust on the same batch, select "
                        "PartialWindowPolicy.DISCARD to drop the partial window rank-locally; "
                        "or warm-start without restoring the saved training progress so "
                        "training restarts at step 0."
                    )
                raise RuntimeError(
                    "Gradient accumulation reached a partial final window "
                    f"(steps_accumulated={steps_accumulated}, num_steps="
                    f"{self.num_micro_batches_per_step}, remaining={remaining}) at world_size <= 1 "
                    "and the caller selected PartialWindowPolicy.RAISE. The local grads "
                    "would be a correct single-process step, but this caller requires the "
                    "total reader batch count to be a whole multiple of "
                    "num_micro_batches_per_step. Make it a whole multiple of K, or select "
                    "PartialWindowPolicy.STEP to take the sanctioned local step."
                )
            # PartialWindowPolicy.STEP: in-band step, at every world size.
            if multi_rank:
                logger.warning(
                    "Gradient accumulation stepped a partial final window at world_size>1 "
                    "(steps_accumulated=%d, num_steps=%d, remaining=%d) under "
                    "PartialWindowPolicy.STEP. Those %d micro-batch(es) ran under "
                    "no_sync(), so every rank applied its own un-reduced gradients and the "
                    "replicas diverge from here. Select DISCARD when all ranks exhaust "
                    "together, or RAISE to fail closed. The reduced alternative is "
                    "is_last_batch=True through progress(), which commits the short window "
                    "through a synchronized step.",
                    steps_accumulated,
                    self.num_micro_batches_per_step,
                    remaining,
                    remaining,
                )
            self._optimizer_wrapper._optimizer.step()
            # Re-arm first, then zero through the wrapper, exactly as DISCARD does: a raw
            # set_to_none would drop the accumulate_into_buckets alias.
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

        ``is_last_batch=True`` forces the gradient sync and the optimizer step on an
        off-boundary final window; ``None`` relies on ``StopIteration`` instead. Raises
        ``StopIteration`` when the dataloader is exhausted, after flushing any remaining
        accumulated gradients.
        """
        if not self._config.is_enabled:
            # Pass-through: no window, so no boundary to signal.
            return self._pipeline.progress(dataloader_iter)

        should_sync = self._should_sync_grad(is_last_batch=is_last_batch or False)
        # is_last_batch commits an off-boundary window in-band via the force-step armed
        # below, bypassing _flush_accumulated_gradients and so PartialWindowPolicy. Fail
        # closed here, before arming it. Training-only; a FULL window is not partial.
        if (
            is_last_batch
            and self._partial_window_policy is PartialWindowPolicy.RAISE
            and getattr(self._model, "training", True)
            and not self._optimizer_wrapper._should_step()
        ):
            steps_accumulated = self.micro_batches_into_window + 1
            remaining = steps_accumulated % self.num_micro_batches_per_step
            # Gated with the RAISE branch it belongs to.
            _ga_abort_all_process_groups(
                f"partial final window via is_last_batch under PartialWindowPolicy.RAISE "
                f"(steps_accumulated={steps_accumulated}, remaining={remaining})"
            )
            raise RuntimeError(
                "Gradient accumulation reached a partial final window "
                f"(steps_accumulated={steps_accumulated}, num_steps="
                f"{self.num_micro_batches_per_step}, remaining={remaining}) via an explicit "
                "is_last_batch commit and the caller selected PartialWindowPolicy.RAISE. "
                "The in-band step would be synchronized (replica-safe), but this caller "
                "requires the total reader batch count to be a whole multiple of "
                "num_micro_batches_per_step. Make it a whole multiple of K, or select "
                "PartialWindowPolicy.STEP to take the sanctioned synchronized final-window "
                "step."
            )
        # Publish the GA CONSUME BOUNDARY onto the inner pipeline BEFORE progress() so
        # split-optimizer sub-steps (sparse/dense) + grad-clip that BYPASS the GA-wrapped
        # optimizer gate on the SAME boundary the wrapped optimizer uses. That boundary is
        # _should_step(), NOT _should_sync_grad() -- the latter force-True's on step-0 and
        # warmup, where the optimizer does not step, and drives DDP no_sync only.
        should_step = self._optimizer_wrapper._should_step() or bool(is_last_batch)
        # One-shot: an explicit last batch forces an in-band step off the schedule boundary.
        # Assigned every progress() so it never leaks into a later window.
        self._optimizer_wrapper._force_step = bool(is_last_batch)
        # Not yet advanced, so == 0 marks the first micro of each window and stays correct
        # across a re-anchor. The FP-param own-grad-bucket path zeroes its buffer there.
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
            # No batch was processed (TrainPipelineSparseDist raises at the top of progress()
            # on an empty prefetch queue), so current_step already reflects completed micros
            # -- do NOT add +1. Training-only: an eval interlude must not flush or step.
            if getattr(self._model, "training", True):
                if self._pending_uncommitted:
                    self._flush_accumulated_gradients(self.micro_batches_into_window)
                    # Committed (single-process step + zero) or raised (distributed). Clear
                    # so a later reset() sees clean state instead of raising.
                    self._pending_uncommitted = False
                # Re-anchor at the exhaustion point: a phase consuming a non-multiple of K
                # would otherwise shift every later window boundary. Inside the training
                # gate so an eval interlude cannot touch accumulation state.
                self._optimizer_wrapper.realign_window()
            raise

        # Split-optimizer modes step their child optimizers directly and never call
        # _GAOptimizerWrapper.step(), the only other place that re-arms _needs_zero_grad.
        # Without this the next zero_grad() no-ops and dense grads LEAK across windows.
        if getattr(self._model, "training", True):
            if should_step:
                self._optimizer_wrapper._needs_zero_grad = True
            # So a later StopIteration knows a flush is needed, and an is_last_batch window
            # already committed in-band is not double-stepped by a raw flush.
            self._pending_uncommitted = not should_step
            # Training-only: eval reuses this entry point, and advancing there would desync
            # the K-micro window boundaries for the resumed training.
            self._advance_state()
            # Re-anchor for the same reason the StopIteration path does. After
            # _advance_state(), so the committed micro is counted before the anchor moves.
            if is_last_batch:
                self._optimizer_wrapper.realign_window()
            # should_sync True means progress() ran under nullcontext, so the DDP reducer
            # finalized and (re)aliased each managed dense grad.
            if should_sync and self._config.accumulate_into_buckets:
                self._bucket_views_ready = True

        return result

    def reset(self, drop_partial: bool = False) -> None:
        """Resets the wrapper and underlying pipeline state.

        ``_bucket_views_ready`` is intentionally NOT cleared: reset() runs at epoch or
        dataloader boundaries where the same model and DDP instances stay alive, and
        ``attach()`` rejects a model swap, so preserved readiness can never be stale.
        """
        # Tests semantic dirtiness rather than current_step % K, because an explicit
        # is_last_batch can commit a clean partial window off the modulo boundary.
        if self._pending_uncommitted and not drop_partial:
            # @lint-ignore FIXIT AllRaisesAreAIExceptions
            raise RuntimeError(
                "GradientAccumulationWrapper.reset() called with an open partial window "
                "(accumulated, un-stepped gradients left by a non-boundary micro; "
                f"current_step={self.current_step}, "
                f"num_steps={self.num_micro_batches_per_step}). "
                "Resetting now would silently drop those gradients and desync the K-micro "
                "window. Complete the window (reach the K-th micro) before reset, or pass "
                "drop_partial=True to intentionally discard the partial window."
            )
        self._optimizer_wrapper.reset()
        self._pending_uncommitted = False
        if hasattr(self._pipeline, "reset"):
            self._pipeline.reset()

    def attach(
        self, model: Optional[torch.nn.Module] = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Reject a model swap; otherwise delegate.

        This class is not a ``TrainPipeline`` subclass, so ``__getattr__`` would forward
        ``attach(new_model)`` to the inner pipeline while the wrapped optimizer stayed bound
        to the original model's parameters. A swap needs a fresh wrapper. ``*args`` is
        forwarded because ``TrainPipelineSparseDist.attach`` takes ``sparse_dist`` first.
        """
        if model is not None and model is not self._model:
            raise RuntimeError(
                "GradientAccumulationWrapper does not support swapping the model via "
                "attach(): the wrapped optimizer stays bound to the original model's "
                "parameters and the selective bucket-view zero caches the original DDP "
                "topology. Construct a new GradientAccumulationWrapper with the new model "
                "and its optimizer instead."
            )
        if not hasattr(  # @lint-ignore FIXIT [AvoidHasattrEverywhere] the wrapped pipeline is an unconstrained TrainPipeline; attach() is optional on it
            self._pipeline, "attach"
        ):
            raise RuntimeError(
                "GradientAccumulationWrapper requires the wrapped pipeline to provide "
                f"attach(); {type(self._pipeline).__name__} does not. Wrap a pipeline "
                "that implements the TrainPipeline attach protocol."
            )
        return self._ga_hooks.attach(model, *args, **kwargs)

    @property
    def _ga_hooks(self) -> _GAPipelineHooks:
        """The wrapped pipeline seen through the GA hook surface.

        ``TrainPipeline`` does not declare ``attach``, so the cast is what lets the one
        call site be a plain attribute access.
        """
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
        """Micro-batches accumulated into the current window, bounded to ``[0, K)``.

        This -- not ``current_step`` -- is what the optimizer-step, grad-sync and
        window-start boundaries key on.
        """
        return self._optimizer_wrapper.micro_batches_into_window

    @property
    def num_micro_batches_per_step(self) -> int:
        """Number of micro-batches (K) accumulated per optimizer step.

        Public accessor for the wrapper's configured K so a Layer-1 trainer loop can
        assert its own K matches the wrapper's without reaching into private config.
        """
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
