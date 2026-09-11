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
    Final,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TYPE_CHECKING,
)

import torch
from torch.nn.parallel import DistributedDataParallel
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


# Capability marker for the APS config-time eval-GA guard
# (aps_models/ads/common/train_pipeline.py), which fail-fasts when GA and scheduled
# in-trainer eval are both on and this is False. Not a config knob: no torchrec reader,
# never assigned at runtime. It records that _advance_state is gated on model.training, so
# an eval interlude reusing the GA progress() path cannot desync the K-micro window.
# If that gating is reverted set this False -- do NOT delete it, which breaks the APS
# import instead of activating the guard.
GA_EVAL_COUNTER_SAFE: Final[bool] = True

logger: logging.Logger = logging.getLogger(__name__)


def _ga_abort_all_process_groups(reason: str) -> None:
    """Tear down every process group before a rank-LOCAL raise on a collective-bearing
    boundary.

    The partial-window guards raise on the rank that detects the problem. At
    ``world_size > 1`` its peers would otherwise block in the next collective until the
    NCCL watchdog fires ~30 minutes later, with the timeout masking the real cause.

    Best-effort, and effective only on NCCL: ``_abort_process_group`` is experimental,
    documented NCCL-only, and expects ``TORCH_NCCL_ASYNC_ERROR_HANDLING=0``. On other
    backends the abort may fail and peers are not torn down; the raise still stands.

    Not gated on the backend: ``get_backend()`` reports only the default PG while
    ``_abort_process_group(None)`` aborts all of them, and a composite
    ``cpu:gloo,cuda:nccl`` config does not equal ``"nccl"`` -- a gate would skip the abort
    on hybrid jobs that need it.

    A no-op at ``world_size <= 1`` (no peers to strand), so call sites need no guard.
    A failure to abort must never mask the raise that follows.
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
        # `_abort_process_group(None)` resolves `pg = group or GroupMember.WORLD` and raises
        # AssertionError when that is None. Log loudly rather than silently: at this point the
        # peers were NOT torn down and the fail-closed guarantee does not hold. See the SPL
        # boundary-hardening journal, NR-10.
        logger.exception(
            "[gradient_accumulation] _abort_process_group FAILED -- peers were NOT torn "
            "down; the raise that follows is rank-local and may strand them"
        )


class PartialWindowPolicy(Enum):
    """Caller policy for a partial (r < K) gradient-accumulation window.

    A partial final window means the total micro-batch count was not a whole multiple of
    ``num_steps`` (K).

    ``STEP`` and ``RAISE`` ONLY select what happens at world_size <= 1, where the local
    grads ARE the full window (no replicas to diverge). At world_size > 1 both of them
    fail-closed (abort the process groups, then raise): stepping un-reduced rank-local
    grads without a cross-rank reduce would diverge replicas.

      - ``STEP`` (default; preserves the historical single-process behavior): take the
        sanctioned in-band optimizer step + zero_grad on the partial window.
      - ``RAISE``: fail-closed even at world_size <= 1 so a caller that requires K-divisible
        data (e.g. APS) is told loudly instead of silently training on a smaller final batch.

    ``DISCARD`` is different in kind: it applies at EVERY world size, and is the only
    policy that makes a partial window a normal, non-fatal ending.

      - ``DISCARD``: on ITERATOR EXHAUSTION, throw the partial window's accumulated
        gradients away rank-locally -- no optimizer step, no collective, no process-group
        abort -- and let the caller's ``StopIteration`` propagate. Correct ONLY for callers
        that have established cross-rank agreement on exhaustion out of band (so every rank
        discards together, keeping replicas identical) and that accept losing up to K-1
        micro-batches at the end of a phase. This is what unlocks consume-all
        (``num_batches < 0``) under GA.

    SCOPE OF ``DISCARD`` -- exhaustion only, by design. An explicit ``is_last_batch=True``
    partial window does NOT discard: it still commits IN-BAND via the forced step in
    ``progress()``, exactly as under ``STEP``. That is deliberate and is the better outcome,
    not an oversight -- ``is_last_batch`` forces ``should_sync``, so the backward runs
    outside ``no_sync`` and the partial window's gradients ARE cross-rank reduced. The
    result is replica-safe at any world size AND keeps the data, whereas discarding would
    throw away up to K-1 micro-batches for no safety gain. ``DISCARD`` exists precisely for
    the case where that information is NOT available (nobody knows which batch is last until
    the iterator is already empty). ``RAISE`` is the one policy that also fences the explicit
    path, because a RAISE caller is asserting K-divisibility rather than asking for a
    best-effort ending.
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
        num_warmup_steps: Number of warmup MICRO-steps (NOT logical/optimizer
            steps) during which every iteration syncs gradients. Counted against
            the global micro counter (current_step), so with num_steps=K a value
            of W means the first W micro-batches sync (~the first ceil(W/K)
            logical windows). Default 1 (only the very first micro).
    """

    is_enabled: bool = False
    num_steps: int = 1
    num_warmup_steps: int = 1
    # Zero dense grads in place at window-start instead of set_to_none=True, preserving
    # the DDP gradient_as_bucket_view alias so autograd accumulates into the persistent
    # reduction bucket rather than allocating a standalone dense .grad set held across
    # the K-micro window (~1x dense-grad-size of extra HBM). Callers that construct this
    # dataclass directly get OFF; the APS layer resolves it from a tri-state config (see
    # aps_models.ads.common.gradient_accumulation.config.resolve_accumulate_into_buckets,
    # which derives ON at K>1). Semantic delta vs OFF: a dense param holding a live
    # bucket-view grad at window start but unused through the window keeps a present zero
    # instead of None, so a dense optimizer sees participate-on-zero rather than skip.
    accumulate_into_buckets: bool = False

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
        # Anchor of the CURRENT K-window within the global micro counter. The
        # optimizer-step / grad-sync / window-start boundaries are all computed relative
        # to this anchor (steps_in_window), not to _current_step, so that re-anchoring on
        # an iterator exhaustion (realign_window) restarts the window without disturbing
        # the counter's monotonic contract or replaying num_warmup_steps.
        self._window_base: int = 0
        self._needs_zero_grad: bool = True
        # One-shot: when True, step() fires even off the schedule boundary. Used to
        # force an in-band optimizer step on an explicit final partial window (r<K).
        self._force_step: bool = False
        # Backref to the owning GradientAccumulationWrapper, set right after
        # construction. When accumulate_into_buckets is on, zero_grad delegates the
        # window-start zero to the outer wrapper's selective bucket-view-preserving
        # protocol (it needs the model's DDP topology). None when this optimizer
        # wrapper is used standalone (unit tests) -> plain None-path zero.
        self._ga_wrapper: Optional["GradientAccumulationWrapper[Any, Any]"] = None

    @property
    def steps_in_window(self) -> int:
        """Micro-batches consumed since the latest window anchor.

        NOT bounded to ``[0, num_steps)``: the anchor moves only on ``realign_window()``
        (iterator exhaustion / explicit last batch), ``set_step()`` and ``reset()``, so
        within one continuous phase this grows without bound exactly like
        ``_current_step``. Only its residue mod ``num_steps`` is meaningful.
        """
        return self._current_step - self._window_base

    def _should_step(self) -> bool:
        """Returns True if optimizer.step() should actually execute."""
        return (self.steps_in_window + 1) % self._config.num_steps == 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Intercepts zero_grad to only clear gradients at accumulation boundaries.

        Uses the _needs_zero_grad flag to ensure proper timing regardless of
        when zero_grad is called in the pipeline execution order.

        When ``accumulate_into_buckets`` is set (and the owning wrapper is present),
        the window-start zero is delegated to
        ``GradientAccumulationWrapper._window_start_zero_grad``, which preserves the
        DDP ``gradient_as_bucket_view`` alias so autograd accumulates into the
        persistent reduction bucket instead of allocating a standalone dense ``.grad``
        set held across the K-micro no_sync window. Otherwise (feature off, or the
        wrapper is used standalone) it delegates to the real optimizer's zero_grad.
        """
        if not self._needs_zero_grad:
            return
        if self._config.accumulate_into_buckets and self._ga_wrapper is not None:
            # The incoming set_to_none is intentionally NOT forwarded: the selective
            # protocol has fixed semantics -- the first-window / not-ready path and every
            # non-target leaf use set_to_none=True (the real-optimizer tree-clear), while
            # ready bucket-view targets are retained and zeroed in place (alias-preserving).
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

        Called when a data iterator is exhausted: a phase that consumed a non-multiple of
        K micro-batches would otherwise leave the free-running counter off-modulo and
        shift every SUBSEQUENT window's boundary off the trainer's logical step. Moving
        the anchor (rather than zeroing ``_current_step``) keeps ``current_step``
        monotonic for its public accessor AND keeps ``num_warmup_steps`` counted against
        the global counter, so warmup is not replayed on each new iterator.
        """
        self._window_base = self._current_step

    def set_step(self, step: int) -> None:
        """Sets the internal step counter. Use this instead of directly modifying _current_step.

        Also drops the window anchor back to 0: callers place the wrapper at a specific
        point in the accumulation cycle and expect plain ``(step + 1) % K`` semantics on
        the raw value. Without this, a preceding realign_window() would leave a stale base
        and make steps_in_window disagree with the step the caller just set.

        No validation is added here (this method's contract predates the anchor): a
        negative ``step`` still yields a negative ``steps_in_window``, exactly as it
        already yielded a negative ``current_step``.
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
        # Boundary signal for a pipeline whose sub-steps bypass the wrapped optimizer.
        # Explicit rather than inferred: a pipeline that needs it opts in here, so a
        # rename is a type error instead of a silent fall back to stepping every micro.
        self._window_observer = window_observer
        # Caller policy for a partial (r < K) final window at world_size <= 1 (read in
        # _flush_accumulated_gradients). world_size > 1 ALWAYS fail-closes regardless.
        self._partial_window_policy = partial_window_policy
        self._optimizer_wrapper = _GAOptimizerWrapper(optimizer, config)
        # Backref so the optimizer wrapper's zero_grad can delegate the selective
        # window-start zero (accumulate_into_buckets) back here, where the model's
        # DDP topology is discoverable.
        self._optimizer_wrapper._ga_wrapper = self
        self._cached_ddp_modules: list[Any] | None = None
        # True after a non-boundary micro left accumulated-but-un-stepped gradients;
        # gates the StopIteration flush so an in-band-committed window is not re-stepped.
        self._pending_uncommitted: bool = False
        # Component 2 (selective retain readiness). True once a synchronized training
        # backward has established the DDP gradient_as_bucket_view aliases. Gates the
        # selective window-start zero so the FIRST window -- before any views exist --
        # uses a plain set_to_none clear (also clears APF dummy grads and avoids the
        # detach-on-view crash that only exists post-alias). Intentionally survives
        # reset() (same model + DDP instances remain alive; the module tree is static post
        # construction -- see _get_ddp_modules). attach() rejects a model swap, so a live
        # wrapper's DDP topology + readiness can never become stale.
        self._bucket_views_ready: bool = False
        # Warn-once keys for leaves excluded from the alias-preserving window-start zero.
        # See _warn_bucket_view_skip.
        self._warned_bucket_view_skips: set[str] = set()

        # Only replace optimizer in pipeline when GA is enabled
        # This avoids unintended side effects when GA is disabled
        if config.is_enabled:
            if not hasattr(pipeline, "_optimizer"):
                # Fail-closed on exactly ONE hole: a pipeline that exposes no _optimizer
                # at all. Previously the injection was silently skipped, so GA quietly
                # provided no optimizer-step gating whatsoever while the caller believed
                # gradients were accumulating over K.
                #
                # SCOPE: hasattr("_optimizer") is necessary, NOT sufficient. A pipeline can
                # expose _optimizer, accept this replacement, and still step a separately
                # captured optimizer reference. Closing that requires a GA capability /
                # rebind contract on TrainPipeline; this guard does not attempt it. Nor does
                # it claim to know what an _optimizer-less pipeline does instead -- only
                # that this wrapper's gating injection point is absent.
                raise RuntimeError(
                    "Gradient accumulation is enabled but the wrapped pipeline "
                    f"{type(pipeline).__name__} exposes no `_optimizer` attribute, so the "
                    "GA optimizer wrapper cannot be injected and optimizer-step gating to "
                    f"one step per {config.num_steps}-micro window cannot be guaranteed. "
                    "Use a pipeline that exposes `_optimizer` (e.g. "
                    "TrainPipelineSparseDist), or disable gradient accumulation."
                )
            if isinstance(getattr(pipeline, "_optimizer", None), _GAOptimizerWrapper):
                # Already GA-wrapped. Replacing again would nest the gates: the outer
                # wrapper's step() would reach the inner one, which gates again, so the
                # real optimizer would step once per K**2 micro-batches instead of once
                # per K -- silent under-stepping, not a crash. There is no unwrap path,
                # so no legitimate caller reaches this.
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

        # Sync on the last step of each accumulation cycle. Window-relative (unlike the
        # two job-lifetime guards above): an iterator exhaustion re-anchors the window, and
        # the grad-sync boundary must move with it or DDP would sync on the wrong micro.
        return (self.steps_in_window + 1) % self._config.num_steps == 0

    def _get_no_sync_context(self) -> ContextManager[None]:
        """
        Returns a composite ``no_sync`` context manager that suppresses gradient
        synchronization on **all** ``DistributedDataParallel`` modules in the
        model tree, not just the outermost one.

        Some sharded submodules wrap their DATA_PARALLEL lookups in their own internal
        DDP instances. Inner DDPs that are REGISTERED submodules (in ``_modules``) are
        discovered by the ``root.modules()`` walk in ``_get_ddp_modules`` and entered via
        ``no_sync()`` here. NOTE: ``ShardedVariableLengthEmbeddingArch`` stores its lookup
        DDPs in a PLAIN Python list (``self._lookups``), NOT an ``nn.ModuleList``, so those
        DDPs are NOT in ``_modules`` and are NOT discovered -- they all-reduce EVERY micro
        (numerically correct under GA: mean-all-reduce is linear + idempotent on an
        already-reduced tensor; no standalone-grad duplication, but no no_sync comm saving
        either). A plain-list -> ``nn.ModuleList`` migration would silently change this;
        test_ga_bucket_view_alias guards the plain-list-not-discovered contract.

        This method uses a cached list of DDP modules (computed once on first
        call) and composes their ``no_sync()`` contexts with
        ``contextlib.ExitStack`` so that a single ``with ctx:`` block
        suppresses gradient sync everywhere.
        """
        return self._compose_no_sync_contexts()

    def _get_ddp_modules(self) -> list[Any]:
        """
        Discover and cache all modules that need ``no_sync()``.

        The module tree is static after model construction, so this walk
        is performed once and the result is reused on every subsequent
        non-sync step — avoiding an O(num_modules) traversal in the
        training-loop hot path.

        After unwrapping ``DistributedModelParallel`` (if present), the method
        uses a two-tier detection strategy:

          1. **Root** (the outer wrapper) is added if it has ``no_sync``,
             regardless of its concrete type. This broad check covers DDP,
             FSDP, and any custom parallel wrapper.
          2. **Descendants** are added only if they are ``isinstance`` of
             ``DistributedDataParallel``. This strict check prevents
             accidentally entering ``no_sync`` on non-DDP modules (e.g.
             nested FSDP) that may have different ``no_sync`` semantics.

        The asymmetry is intentional: the root is *known* to be the
        top-level parallel wrapper (set by ``DistributedModelParallel``),
        while descendants are arbitrary submodules that need positive
        identification.
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

        # Collect the root module if it supports no_sync (broad check:
        # covers DDP, FSDP, or any custom parallel wrapper).
        if hasattr(root, "no_sync"):
            ddp_modules.append(root)

        # Walk descendants for any REGISTERED inner DDP instances (present in
        # ``_modules``). Uses a strict isinstance check — only actual DDP modules are
        # collected, not FSDP or other modules that happen to have no_sync. NOTE:
        # ``ShardedVariableLengthEmbeddingArch`` keeps its lookup DDPs in a PLAIN list
        # (not ``_modules``), so they are NOT collected here and reduce every micro
        # (see the _get_no_sync_context docstring).
        # The hasattr guard is needed because root may not be an
        # nn.Module (e.g. when _dmp_wrapped_module is a non-Module
        # wrapper object and model itself lacks modules()).
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

    def _collect_bucket_view_targets(  # noqa: C901 — the branches ARE the eligibility contract for in-place bucket-view zeroing; splitting them would scatter a correctness gate
        self,
    ) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        """Params (+ their live bucket-view grad) eligible for in-place window-start zero.

        A target is a parameter that:
          (a) is managed by a real ``DistributedDataParallel`` that the GA wrapper
              enters via ``no_sync()``;
          (b) that DDP has ``gradient_as_bucket_view=True``;
          (c) currently has a live grad (not None) that is a bucket-view alias
              (``grad._base is not None``) with a strided (dense) layout;
          (d) is owned by the wrapped optimizer (H1.1 ownership intersection below), so
              the OFF-path ``optimizer.zero_grad`` would have cleared it -- ON must not
              diverge from OFF for a DDP-reduced-but-not-optimizer-owned leaf.

        NOTE (heuristic, H1.2): ``grad._base is not None`` is a CONSERVATIVE,
        NON-AUTHORITATIVE check -- it confirms the grad is SOME view, NOT authoritatively
        the reducer's live reduction-bucket view. The reducer's live bucket views are
        C++-internal and not exposed to Python (``reducer.cpp``; ``init.cpp``), so a
        reducer bucket-view cannot be distinguished from an arbitrary other view here.
        Under ``gradient_as_bucket_view=True`` the managed dense grad IS the bucket-view
        in the common case; we rely on that + the ownership + no_sync-DDP gating and do
        NOT claim to reject an arbitrary non-bucket view.

        Skipped (fall to the None path via the tree-clear):
          - ``grad is None`` (permanently grad-less dense head -> MUST stay None so the
            dense optimizer keeps skipping it: no None->zero state corruption);
          - sparse / non-strided grad (not a dense bucket-view);
          - params of a non-DDP root (DMP/custom/FSDP) or a non-bucket-view DDP;
          - fp32-grad DP params (never in any ``DDP._module_parameters`` -> not seen).

        DEGRADED (excluded from the target list + warn-once) on unsafe / unknown states,
        rather than raised:
          - a real DDP with no ``_module_parameters`` (no authoritative ownership list;
            never broaden to ``module.parameters()``, which ignores the ignore-list);
          - a param reduced by two DISTINCT DDP reducers (a grad can alias only one
            reduction bucket, so an in-place zero could target the wrong one);
          - a no_sync'd real DDP with ``find_unused_parameters=True`` (the per-iteration
            used-parameter set is dynamic, so a dense grad zeroed in place one window may
            be absent (None) the next -> None-vs-zero divergence for the dense optimizer
            state);
          - a live dense grad on an optimizer-owned bucket-view DDP param whose grad is
            NOT a view (``grad._base is None``): unexpected standalone-grad state for a
            ``gradient_as_bucket_view=True`` DDP (heuristic per the NOTE above; catches
            the common "bucket-view was dropped" case, not an authoritative assertion).

        WHY EXCLUSION AND NOT A RAISE. A non-target is not hidden below, so it takes the
        real ``optimizer.zero_grad(set_to_none=True)`` tree-clear -- byte-identical to the
        flag-OFF path. The cost of degrading is forgone HBM reclaim for those params,
        which is precisely the default-off behaviour, i.e. strictly no worse. Two sibling
        branches in this same loop (non-bucket-view, find_unused_parameters) already do
        exactly this, and an empty target list short-circuits to the same clear. Raising
        instead would convert a working GA run into a hard crash the moment
        ``accumulate_into_buckets`` becomes the K>1 default -- trading a memory
        optimisation for an availability regression. The skip is WARNED rather than
        silent, because someone must not believe they are getting reclaim they are not.
        """
        real_ddps: list[DistributedDataParallel] = [
            m for m in self._get_ddp_modules() if isinstance(m, DistributedDataParallel)
        ]

        # Pass 1 (ownership): map param-identity -> owning DDP across ALL real DDPs.
        # A grad can alias only one reducer bucket, so a param
        # reduced by two DISTINCT DDP reducers is an unsupported double-owner config ->
        # exclude that param. _module_parameters (excludes DDP-ignored MP/FSDP params) is
        # the authoritative ownership list; its absence on a real DDP is an unknown state
        # -> exclude that whole DDP (never broaden to module.parameters(), which ignores
        # the ignore-list). Both exclusions leave the affected leaves on the plain
        # set_to_none path -- see the docstring for why that is inert.
        owner_of: dict[int, DistributedDataParallel] = {}
        skipped_ddp_ids: set[int] = set()
        excluded_param_ids: set[int] = set()
        for ddp in real_ddps:
            module_params = getattr(ddp, "_module_parameters", None)
            if module_params is None:
                skipped_ddp_ids.add(id(ddp))
                self._warn_bucket_view_skip(
                    "missing_module_parameters",
                    f"{type(ddp).__name__} exposes no authoritative _module_parameters "
                    "ownership list",
                )
                continue
            for p in module_params:
                if not p.requires_grad:
                    continue
                prev = owner_of.get(id(p))
                if prev is not None and prev is not ddp:
                    excluded_param_ids.add(id(p))
                    self._warn_bucket_view_skip(
                        "double_ddp_ownership",
                        f"a parameter of shape {tuple(p.shape)} is reduced by two "
                        "distinct DistributedDataParallel reducers",
                    )
                    continue
                owner_of[id(p)] = ddp

        # H1.1 ownership intersection: the wrapped optimizer's owned param id-set. The
        # OFF path clears ONLY optimizer-owned param_groups (optimizer.zero_grad), so an
        # in-place window-start zero must touch ONLY params the optimizer owns -- a
        # DDP-reduced-but-not-optimizer-owned param must fall to the plain path (left
        # accumulating), else ON would zero a grad OFF leaves present -> a None-vs-present
        # divergence for that leaf. Duck-typed on ``param_groups`` (torchrec must NOT
        # import the apf/aps CombinedOptimizer -- layering; the apf CombinedOptimizer
        # aggregates its sub-optimizers' param_groups). Empty owned-set -> no targets ->
        # plain path everywhere (inert, identical to OFF).
        owned_param_ids: set[int] = set()
        wrapped_optimizer = self._optimizer_wrapper._optimizer
        for group in getattr(wrapped_optimizer, "param_groups", None) or []:
            if isinstance(group, dict):
                for p in group.get("params", []):
                    owned_param_ids.add(id(p))

        # Pass 2: select in-place-zero targets from the no_sync'd bucket-view DDPs.
        targets: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for ddp in real_ddps:
            if id(ddp) in skipped_ddp_ids:
                # Excluded in pass 1 (no ownership list). Already warned.
                continue
            if not getattr(ddp, "gradient_as_bucket_view", False):
                # Grads are not reduction-bucket aliases -> the None path is correct, and
                # there is no standalone duplicate to reclaim. Checked BEFORE the
                # find_unused_parameters warn: otherwise a DDP with nothing to reclaim still
                # logs "their HBM is NOT reclaimed", which is alarming and untrue. Not
                # reachable on APS (it hardcodes gradient_as_bucket_view=True) but generic
                # torchrec callers do hit it.
                continue
            if getattr(ddp, "find_unused_parameters", False):
                self._warn_bucket_view_skip(
                    "find_unused_parameters",
                    f"{type(ddp).__name__} uses find_unused_parameters=True, so its "
                    "used-parameter set is dynamic per micro-batch",
                )
                continue
            # _module_parameters guaranteed present (pass 1).
            for p in ddp._module_parameters:
                if not p.requires_grad:
                    continue
                if id(p) in excluded_param_ids:
                    # Double-owned (pass 1). Already warned.
                    continue
                if id(p) not in owned_param_ids:
                    # DDP-reduced but NOT wrapped-optimizer-owned (H1.1): OFF never clears
                    # it, so ON must not zero it either -> plain path (leave accumulating).
                    continue
                grad = p.grad
                if grad is None:
                    # Grad-less dense head: stays None (dense optimizer skips it).
                    continue
                if grad.is_sparse or grad.layout != torch.strided:
                    # Not a dense bucket-view alias.
                    continue
                if getattr(grad, "_base", None) is None:
                    self._warn_bucket_view_skip(
                        "standalone_grad",
                        f"a parameter of shape {tuple(p.shape)} on a "
                        "gradient_as_bucket_view=True DDP holds a STANDALONE grad "
                        "(grad._base is None) -- the bucket-view alias was dropped",
                    )
                    continue
                targets.append((p, grad))
        return targets

    def _warn_bucket_view_skip(self, reason: str, detail: str) -> None:
        """Warn once per REASON that something was excluded from the in-place window-start
        zero, and therefore keeps the flag-OFF ``set_to_none`` path.

        Keyed per reason rather than per param so a model with hundreds of affected
        leaves logs once, not hundreds of times -- and rank 0 only, because every reason
        here is a property of the DDP topology, which is rank-uniform by construction.

        Not silent: the exclusion is correct and inert, but it means the forgone HBM
        reclaim for those leaves is real, and someone reading only the config would
        otherwise believe they were getting it.
        """
        if reason in self._warned_bucket_view_skips:
            return
        self._warned_bucket_view_skips.add(reason)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return
        logger.warning(
            "[accumulate_into_buckets] excluding leaves from the alias-preserving "
            "window-start zero (%s): %s. They fall back to "
            "optimizer.zero_grad(set_to_none=True) -- numerically identical to "
            "accumulate_into_buckets=False, but their dense-gradient HBM is NOT "
            "reclaimed across the K-micro window.",
            reason,
            detail,
        )

    def _window_start_zero_grad(self) -> None:
        """Window-start zero for ``accumulate_into_buckets`` (Component 1).

        Selective retain/hide/tree-clear/restore so the DDP ``gradient_as_bucket_view``
        alias survives the window-start zero (autograd then accumulates into the
        persistent reduction bucket instead of allocating a standalone dense ``.grad``
        set across the K-micro no_sync window) WITHOUT bypassing the real optimizer
        tree's ``zero_grad`` (which propagates fused-embedding LR and applies correct
        None semantics to Shampoo dense / fp32-grad DP / every non-target leaf).

          1. Until bucket-views are established (first window), plain set_to_none clear.
          2. Collect targets = live bucket-view grads owned by a GA-no_sync'd real DDP.
          3. Hide: ``p.grad = None`` for targets only (so tree-clear no-ops on them).
          4. Tree-clear: ``self._optimizer.zero_grad(set_to_none=True)`` -- fused-LR
             propagation + None semantics for every non-target leaf.
          5. Restore + in-place zero: reinstate the saved bucket-view and ``grad.zero_()``
             (view-safe: no ``detach_()`` on a view).
        """
        optimizer = self._optimizer_wrapper._optimizer
        if not self._bucket_views_ready:
            # First window / views not yet established: plain None clear (also clears
            # APF dummy grads; avoids the detach-on-view crash that only exists
            # post-alias). Numerically identical to the None path.
            optimizer.zero_grad(set_to_none=True)
            return
        targets = self._collect_bucket_view_targets()
        if not targets:
            # No live bucket-view DDP targets (e.g. default AFOC, where promoted tables
            # use the fp32-grad manual reducer; or a custom/FSDP root; or grads not yet
            # present): plain None path -- inert, identical to accumulate_into_buckets off.
            optimizer.zero_grad(set_to_none=True)
            return
        # Hide the target grads so the real optimizer tree-clear cannot touch them, then
        # ALWAYS restore the saved bucket-view aliases (finally) even if the tree-clear
        # raises -- otherwise an exception would leave targets at None and drop their
        # aliases (HBM reclaim would silently stop and the next backward would allocate a
        # fresh standalone grad). Zero in place only on the success path (after restore).
        for p, _grad in targets:
            p.grad = None
        try:
            # Tree-clear the real optimizer: fused-embedding LR propagation + None
            # semantics for Shampoo dense / fp32-grad DP / all non-target leaves (targets
            # are None -> no-op on them).
            optimizer.zero_grad(set_to_none=True)
        finally:
            # Restore the bucket-view alias (preserves it so the next no_sync backward
            # accumulates into the persistent reduction bucket).
            for p, grad in targets:
                p.grad = grad
        # Zero the restored aliases in place (view-safe; only reached on success).
        # H4.1 (Phase-P2 conditional): coalesce the per-target ``grad.zero_()`` memsets
        # into one ``torch._foreach_zero_`` per ``(device, dtype)`` group -- Phase P found
        # a MATERIAL ~2% per-window QPS cost (CMF -2.32%) from launching N separate memset
        # kernels (one per dense param) each window. Grouped writes are storage-safe: DDP
        # reduction-bucket slices share storage but do NOT overlap
        # (reducer.cpp:1170-1193,1293-1349). Rebuild the group lists from the FRESH
        # ``targets`` every window -- never cache the live tensor list (a static_graph
        # bucket rebuild repoints grads to new views). NOTE: ``_foreach_zero_`` does not
        # bump per-tensor version counters (unlike ``Tensor.zero_()``, ForeachUnaryOp.cu);
        # these grads are DDP reduction-bucket views WRITTEN by the reducer, not tensors
        # autograd READS for backward, so the version counter is immaterial here -- the
        # alias-preservation + next-window freshness is covered by the bucket-view
        # state-machine + foreach coalescing tests.
        zero_groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for _p, grad in targets:
            zero_groups.setdefault((grad.device, grad.dtype), []).append(grad)
        for grad_list in zero_groups.values():
            torch._foreach_zero_(grad_list)

    def _flush_accumulated_gradients(self, steps_accumulated: int) -> bool:
        """
        Force a gradient sync and optimizer step for any remaining gradients.

        Args:
            steps_accumulated: Number of micro-batches accumulated so far.
                This is passed explicitly to ensure consistent behavior regardless
                of when flush is called (before or after _advance_state).

        Returns:
            True if a partial window was committed with an optimizer step.
            False if no flush was needed (complete window) OR the partial window was
            discarded under ``PartialWindowPolicy.DISCARD`` (no step is taken, so nothing
            is flushed). The sole caller ignores this value.
        """
        remaining = steps_accumulated % self._config.num_steps
        if remaining > 0:
            if self._partial_window_policy is PartialWindowPolicy.DISCARD:
                # DISCARD: drop the partial window rank-locally. No optimizer step, no
                # collective, and deliberately NOT _ga_abort_all_process_groups -- under
                # this policy an exhaustion-time partial window is the NORMAL ending of a
                # consume-all phase, not a fault, and every rank reaches it together
                # (the caller established cross-rank exhaustion agreement out of band).
                #
                # Placed AHEAD of the world-size block on purpose, and this placement is
                # load-bearing: every policy check below is written as `is RAISE / else
                # STEP` with no exhaustive match, so a DISCARD that fell through would
                # take a silent local un-reduced optimizer step -- corruption, not a
                # crash. Early-returning here also leaves the deliberately-unconditional
                # abort in the world_size > 1 block byte-identical for STEP and RAISE.
                #
                # Re-arm BEFORE zeroing. zero_grad() early-returns when _needs_zero_grad
                # is False, which is exactly the mid-window state we are in, so calling it
                # bare is a guaranteed no-op and the discarded gradients would survive
                # into the next window. Same idiom the boundary-commit path uses. Go
                # through the wrapper (not the raw optimizer) so the
                # accumulate_into_buckets bucket-view alias routing is preserved.
                self._optimizer_wrapper._needs_zero_grad = True
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
                    self._config.num_steps,
                    remaining,
                    remaining,
                )
                # Deliberately no reset() (it zeroes _current_step/_window_base and would
                # replay GA/DDP warmup on every new iterator) and no realign_window()
                # (the caller already does that right after this returns).
                return False
            # A raw flush steps the RANK-LOCAL accumulated grads of an incomplete window
            # WITHOUT a cross-rank reduce. In a distributed (world_size > 1) run that
            # diverges replicas, so FAIL-CLOSED (raise) rather than silently corrupt. The
            # supported divisible-N path never reaches here (full windows -> remaining==0),
            # and the APS config-time guards (train_pipeline.py) block a statically-partial
            # N; but a CHECKPOINT RESUME can still produce a runtime-partial remaining
            # window (the data loader subtracts a logical-step count from a reader-micro
            # total with no xK conversion), which the static guards do not catch -- this
            # runtime guard closes that hole. (An explicit is_last_batch partial window
            # commits IN-BAND under a synchronized step and never sets _pending_uncommitted,
            # so it never reaches this flush.)
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                # A3: fires BEFORE policy evaluation and is deliberately unconditional, so
                # the abort must apply under BOTH PartialWindowPolicy values -- gating it on
                # RAISE would preserve the hang for STEP callers.
                _ga_abort_all_process_groups(
                    f"partial final window at world_size>1 "
                    f"(steps_accumulated={steps_accumulated}, remaining={remaining})"
                )
                raise RuntimeError(
                    "Gradient accumulation reached a partial final window "
                    f"(steps_accumulated={steps_accumulated}, num_steps="
                    f"{self._config.num_steps}, remaining={remaining}) that was never "
                    "committed in-band. Flushing would step un-reduced rank-local "
                    "gradients (replica divergence). This usually means a checkpoint "
                    "resume left a reader-batch count that is not a whole multiple of "
                    "num_micro_batches_per_step. Make the total reader batches a whole "
                    "multiple of K, and when warm-starting from a seed set "
                    "++checkpoint.skip_loading_training_progress_checkpoint_on_start=true "
                    "so training restarts at step 0."
                )
            # Single-process (world_size <= 1): the local grads ARE the full window (no
            # replicas to diverge), so an in-band step is numerically correct. Whether a
            # partial world_size <= 1 window is TOLERATED is a caller policy
            # (PartialWindowPolicy, selected at wrapper construction):
            #   - RAISE: fail-closed even here, so a caller that requires K-divisible data
            #     (e.g. APS, which already fail-closes a static N % K != 0 at config time)
            #     is told loudly rather than silently stepping a smaller final batch -- e.g.
            #     a checkpoint-resume remainder the static guards cannot catch.
            #   - STEP (default): take the sanctioned local step + reset (historical W<=1
            #     behavior).
            if self._partial_window_policy is PartialWindowPolicy.RAISE:
                raise RuntimeError(
                    "Gradient accumulation reached a partial final window "
                    f"(steps_accumulated={steps_accumulated}, num_steps="
                    f"{self._config.num_steps}, remaining={remaining}) at world_size <= 1 "
                    "and the caller selected PartialWindowPolicy.RAISE. The local grads "
                    "would be a correct single-process step, but this caller requires the "
                    "total reader batch count to be a whole multiple of "
                    "num_micro_batches_per_step. Make it a whole multiple of K, or select "
                    "PartialWindowPolicy.STEP to take the sanctioned local step."
                )
            # PartialWindowPolicy.STEP: sanctioned in-band step + reset. Plain set_to_none
            # (post-step boundary; no bucket-view alias to preserve after a real step).
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
            # Pass-through: no window, so no boundary to signal.
            return self._pipeline.progress(dataloader_iter)

        should_sync = self._should_sync_grad(is_last_batch=is_last_batch or False)
        # VETO fix (policy-contract completeness): is_last_batch commits an OFF-BOUNDARY
        # partial (r < K) window in-band via the SYNCHRONIZED force-step armed below
        # (should_sync is True for is_last_batch, so the backward runs under nullcontext and
        # the partial window's grads ARE cross-rank reduced -> replica-safe at any
        # world_size). But that in-band commit BYPASSES _flush_accumulated_gradients and thus
        # PartialWindowPolicy: a caller that selected RAISE (e.g. APS) could otherwise
        # silently force-step a partial window here instead of raising. Fail-closed on a
        # partial window BEFORE arming the force-step / publishing the boundary / calling
        # pipeline.progress() so no partial step occurs (mirrors the flush path). Gated on
        # model.training (like the flush guard and the boundary-commit block below) so an
        # eval interlude never spuriously raises. A FULL window via is_last_batch (already on
        # the schedule boundary => _should_step() True) is NOT partial and is unaffected;
        # PartialWindowPolicy.STEP keeps the synchronized force-step unchanged.
        #
        # DISCARD deliberately does NOT fence here, and this asymmetry with RAISE is the
        # point: is_last_batch makes should_sync True, so the partial window's grads are
        # cross-rank REDUCED and the in-band step is replica-safe at any world_size. Taking
        # it keeps up to K-1 micro-batches that a discard would throw away, for no safety
        # gain. DISCARD governs the EXHAUSTION path, where that information does not exist.
        # (No non-test caller in aps_models/, apf/ or torchrec/ passes is_last_batch=True
        # today, so this is a contract statement, not a live behavior difference.)
        if (
            is_last_batch
            and self._partial_window_policy is PartialWindowPolicy.RAISE
            and getattr(self._model, "training", True)
            and not self._optimizer_wrapper._should_step()
        ):
            steps_accumulated = self.steps_in_window + 1
            remaining = steps_accumulated % self._config.num_steps
            # A3: this branch only exists under RAISE, so the abort is gated with it. At
            # world_size <= 1 the helper is a no-op, which is the correct behavior -- there
            # are no peers to strand.
            _ga_abort_all_process_groups(
                f"partial final window via is_last_batch under PartialWindowPolicy.RAISE "
                f"(steps_accumulated={steps_accumulated}, remaining={remaining})"
            )
            raise RuntimeError(
                "Gradient accumulation reached a partial final window "
                f"(steps_accumulated={steps_accumulated}, num_steps="
                f"{self._config.num_steps}, remaining={remaining}) via an explicit "
                "is_last_batch commit and the caller selected PartialWindowPolicy.RAISE. "
                "The in-band step would be synchronized (replica-safe), but this caller "
                "requires the total reader batch count to be a whole multiple of "
                "num_micro_batches_per_step. Make it a whole multiple of K, or select "
                "PartialWindowPolicy.STEP to take the sanctioned synchronized final-window "
                "step."
            )
        # Publish the GA CONSUME BOUNDARY onto the inner pipeline BEFORE progress()
        # so split-optimizer sub-steps (sparse/dense) + grad-clip that BYPASS the
        # GA-wrapped optimizer gate on the SAME boundary the wrapped optimizer uses.
        # NOTE: the consume boundary is _should_step() (the optimizer-step boundary),
        # NOT _should_sync_grad() which force-True's on step-0/warmup where the
        # optimizer does NOT step (that predicate drives DDP no_sync only).
        should_step = self._optimizer_wrapper._should_step() or bool(is_last_batch)
        # Partial-window support (one-shot): on an explicit last batch, force the wrapped
        # optimizer to step in-band even when the schedule boundary (_should_step) is not
        # reached (a final r<K window). Split modes already step via the observer below;
        # this makes DEFAULT mode's wrapper.step() also fire.
        # Assigned every progress() so it never leaks into a later window.
        self._optimizer_wrapper._force_step = bool(is_last_batch)
        # steps_in_window is the micro counter relative to the current window's anchor
        # (not yet advanced), so == 0 marks the first micro of each window -- and stays
        # correct after an iterator exhaustion re-anchors it. The FP-param own-grad-bucket
        # path zeroes its coalesced buffer there, so the K micro grads accumulate before
        # the boundary all-reduce.
        at_window_start = (self.steps_in_window % self._config.num_steps) == 0
        # Signal BEFORE progress() so sub-steps that bypass the wrapped optimizer gate on
        # the same boundary it uses.
        if self._window_observer is not None:
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
            # When StopIteration is raised, pipeline.progress() had no batch
            # to process — no forward, backward, or optimizer step happened in
            # this call. In TrainPipelineSparseDist, StopIteration is raised at
            # the top of progress() when self.batches is empty (all prefetched
            # batches were already fully processed in prior calls). Therefore
            # current_step reflects completed batches (do NOT add +1). Flush only if
            # this rank has un-stepped accumulated gradients (a partial window not
            # committed in-band). Partial windows are now FAIL-CLOSED at APS config time:
            # train_pipeline.py hard-fails GA with a non-divisible bounded num_batches
            # (N%K!=0) AND with consume-all num_batches=-1 (R1-lite frontier fail-fasts),
            # so under the supported divisible-N path only full windows reach here ->
            # _pending_uncommitted is False -> this no-ops. (A raw flush of a partial
            # window would step un-reduced rank-local grads = replica divergence,
            # distributed only; that frontier stays blocked at config time.)
            # Gate on model.training: an eval interlude must never raw-flush / step the
            # optimizer. _pending_uncommitted is only set inside the training-mode block
            # below, so at a clean window boundary it is already False; the guard also
            # closes the mid-window-eval edge (pending grads from an incomplete training
            # window followed by an eval whose iterator is exhausted).
            if getattr(self._model, "training", True):
                if self._pending_uncommitted:
                    self._flush_accumulated_gradients(self.steps_in_window)
                    # The flush committed the partial window (single-process step + zero) or
                    # raised (distributed). On the committed path clear the pending flag so a
                    # subsequent reset() at the epoch/sample boundary sees a clean state (S2
                    # guard) rather than raising on an already-handled window -- e.g. the
                    # benchmark harness that calls reset() once per sample after a non-K-
                    # divisible iter count.
                    self._pending_uncommitted = False
                # Re-anchor the K-window at the exhaustion point. A phase that consumed a
                # non-multiple of K micros leaves the free-running counter off-modulo, which
                # would shift EVERY subsequent window's boundary off the trainer's logical
                # step once a new iterator starts. Runs on a clean-boundary exhaustion too
                # (cheap, and keeps the anchor equal to the counter). Inside the training
                # gate on purpose: an eval interlude must not touch GA state
                # (test_eval_stop_iteration_does_not_flush), and a mid-window training
                # window must survive an eval whose iterator exhausts.
                self._optimizer_wrapper.realign_window()
            raise

        # Boundary-commit (cross-window grad-zeroing fix). The APS split-optimizer
        # pipeline modes step their child optimizers DIRECTLY (train_pipeline.py
        # dispatch to _step_optimizer_embedding_lookup_fwd / _step_optimizer_fp_allreduce)
        # and NEVER call _GAOptimizerWrapper.step() — the only other place that resets
        # _needs_zero_grad back to True. The sole grad zeroer is the wrapped
        # self._optimizer.zero_grad() at the top of the next progress(), gated by
        # _needs_zero_grad. Without this reset that zero_grad() no-ops from window 2
        # onward and the dense (DDP) + DP-promoted Mechanism-B dense gradients LEAK
        # across logical windows. Reset on the consume boundary for BOTH split and
        # default modes (idempotent in default mode, where wrapper.step() already set
        # it). Keyed on should_step (the optimizer-step boundary), NOT should_sync
        # (which is True on warmup / step 0 without an optimizer step).
        #
        # Gated on model.training: the APS optimizer-step + zero_grad paths are all
        # training-only (train_pipeline.py step dispatch ~1465-1470 + zero_grad ~1392),
        # so in eval no step/zero happens and there is no grad state to manage. This also
        # ensures should_step (an INTENT flag) only commits the boundary when a real
        # optimizer step could actually have occurred.
        if getattr(self._model, "training", True):
            if should_step:
                self._optimizer_wrapper._needs_zero_grad = True
            # Track whether this micro left un-stepped accumulated gradients so a later
            # StopIteration knows whether a flush is actually needed -- AND an explicit
            # is_last_batch boundary (committed in-band above via the forced step) is NOT
            # double-stepped by a raw flush. should_step True => committed this micro.
            self._pending_uncommitted = not should_step
            # Advance the GA micro-step counter ONLY during training. Eval reuses the same
            # train_step -> progress() dispatch (ads_rec_train_factory keys the dispatch on
            # _ga_config, NOT train/eval), so advancing _current_step during an eval
            # interlude would desync the K-micro window boundaries (should_step /
            # should_sync / at_window_start all key on current_step) for the resumed
            # training step. Eval sets module.eval() (TrainLoopScheduler.set_eval_mode),
            # so gating the advance on model.training freezes the counter across eval.
            self._advance_state()
            # An explicit last batch closes the phase, so re-anchor the K-window here for
            # the same reason the StopIteration path does: the next iterator must start a
            # fresh window even though this one ended off-modulo. APS never passes
            # is_last_batch (_train_step_with_local_ga calls progress(data_iter) bare);
            # this is for generic torchrec callers. Ordering: after _advance_state(), so
            # the committed micro is counted before the anchor moves.
            if is_last_batch:
                self._optimizer_wrapper.realign_window()
            # Component 2: mark the DDP gradient_as_bucket_view aliases established once a
            # synchronized training backward has completed. should_sync True => the
            # pipeline.progress() above ran under nullcontext (not no_sync), so DDP's
            # reducer finalized and (re)aliased each managed dense grad to its
            # reduction-bucket view. The selective in-place window-start zero then engages
            # only from the NEXT window start (>= window 1, whose first micro is no_sync);
            # window 0's start zero already used set_to_none (views not yet ready). The
            # one-time static_graph bucket rebuild re-points each grad view exactly once and
            # may fire during a later window's forward (not necessarily at the window-0
            # boundary); the reducer COPIES the current (zeroed) grad into the new
            # bucket-view and repoints p.grad, so the alias survives the rebuild regardless
            # of timing. Targets are collected fresh every window and stay bucket-views
            # (empirically verified in test_ga_bucket_view_alias: grad._base is set at
            # every micro under the flag).
            if should_sync and self._config.accumulate_into_buckets:
                self._bucket_views_ready = True

        return result

    def reset(self, drop_partial: bool = False) -> None:
        """Resets the wrapper and underlying pipeline state.

        ``_bucket_views_ready`` is intentionally NOT cleared: reset() runs at
        epoch/dataloader boundaries where the SAME model and DDP instances (and their
        established reduction-bucket views) remain alive, so readiness carries over. The
        model/DDP topology cannot change under a live wrapper -- ``attach()`` rejects a
        model swap -- so preserved readiness can never become stale.
        """
        # Clean-boundary guard (S2): a reset() with an OPEN partial window
        # (_pending_uncommitted True -- a non-boundary micro left accumulated,
        # un-stepped, un-reduced gradients) would silently DROP those gradients (the
        # optimizer step never happens) and desync the K-micro window. Test semantic
        # dirtiness, not current_step % K: an explicit is_last_batch can commit a clean
        # partial window off the modulo boundary. APF checkpoints after a full logical
        # step, so a normal epoch/dataloader-boundary reset has no pending window.
        if self._pending_uncommitted and not drop_partial:
            # @lint-ignore FIXIT AllRaisesAreAIExceptions AllRaisesAreAPSExceptions
            raise RuntimeError(
                "GradientAccumulationWrapper.reset() called with an open partial window "
                "(accumulated, un-stepped gradients left by a non-boundary micro; "
                f"current_step={self.current_step}, num_steps={self._config.num_steps}). "
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
        ``attach(new_model)`` to the inner pipeline while the wrapper's optimizer stayed
        bound to the original model's parameters and the window-start zero targeted a
        stale DDP set. A swap therefore needs a fresh wrapper; same-model re-attach and
        ``attach(None)`` delegate unchanged.

        ``*args`` is forwarded because ``TrainPipelineSparseDist.attach`` takes
        ``sparse_dist`` positionally.
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
        call site be a plain attribute access. Pure view, no check: availability is
        enforced at that call site.
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
    def steps_in_window(self) -> int:
        """Micro-batches consumed since the latest window anchor.

        This -- not ``current_step`` -- is what the optimizer-step, grad-sync and
        window-start boundaries key on, so it is the value a Layer-1 trainer loop must
        check when asserting that a logical step begins on a window boundary.
        ``current_step`` stays globally monotonic and can legitimately be off-modulo
        after an iterator exhaustion re-anchors the window.

        NOT bounded to ``[0, num_micro_batches_per_step)`` -- see
        ``_GAOptimizerWrapper.steps_in_window``; only its residue mod K is meaningful. In
        particular ``_GAOptimizerWrapper.set_step(n)`` drops the anchor back to 0, so this
        then reads ``n`` verbatim -- including a negative value for a negative ``n``,
        exactly as ``current_step`` already did.
        """
        return self._optimizer_wrapper.steps_in_window

    @property
    def num_micro_batches_per_step(self) -> int:
        """Number of micro-batches (K) accumulated per optimizer step.

        Public accessor for the wrapper's configured K so a Layer-1 trainer loop can
        assert its own K matches the wrapper's without reaching into private config.
        """
        return self._config.num_steps

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
