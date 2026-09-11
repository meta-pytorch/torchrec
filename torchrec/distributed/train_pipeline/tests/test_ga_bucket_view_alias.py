#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Citrine missing_for_each_optimizer is suppressed file-wide below. These are CPU unit
# tests whose whole purpose is to pin the optimizer's INTERACTION with DDP bucket views
# (gradient_as_bucket_view aliasing, the selective window-start zero, and zero_grad
# interception). `foreach=True` swaps SGD's internal implementation to the multi-tensor
# kernels, which is exactly the behaviour under test, and the perf benefit the rule is
# chasing is nil on a CPU unit test.
# @lint-ignore-every CITRINE

"""Real-DDP bucket-view alias state-machine + degradation tests for the
``accumulate_into_buckets`` selective window-start zero (gradient_accumulation.py C1/C2).

Builds a REAL ``DistributedDataParallel`` with ``gradient_as_bucket_view=True`` over a
single-process gloo group so the ACTUAL bucket-view alias lifecycle is exercised (the
mocked-DDP tests in test_gradient_accumulation.py cannot). Uses a bias-free ``Linear`` fed
constant ones with ``lr=0`` so every micro contributes an identical all-ones grad -> the
per-micro accumulation COUNT is ``grad.max()``, which lets us assert both (a) the alias
invariant (bucket-view vs standalone) and (b) grad freshness (each window restarts at 1x,
not K+1x -- i.e. ``grad.zero_()`` actually zeroed).

Core invariant: across a K-micro window whose first micro runs under ``no_sync()``, the ON
path keeps the dense grad a BUCKET-VIEW (``grad._base is not None``) every micro (no
standalone HBM duplication), while the OFF path allocates a STANDALONE grad at each
window-start no_sync micro. A permanently grad-less head stays ``None`` on both paths.

NOTE on identity: DDP performs a ONE-TIME static_graph bucket rebuild that re-points each
grad to a new bucket-view object exactly once; the invariant is "still a bucket-view", not
"same object across windows".
"""

import inspect
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any, cast, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pyre_extensions import none_throws
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline


class _ConstGradModel(nn.Module):
    """bias-free ``Linear`` (grad_W == outer(d_out, x)); with x=ones and lr=0 every micro
    contributes an identical all-ones grad, so ``active.weight.grad.max()`` == the number of
    micro-grads accumulated since the last zero. ``unused`` is forward-reachable but its
    output is discarded (not in the loss) -> permanently grad-less (mirrors AFOC task heads
    that static_graph=True tolerates)."""

    def __init__(self) -> None:
        super().__init__()
        self.active = nn.Linear(4, 4, bias=False)
        self.unused = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.unused(x)  # forward-reachable, discarded -> grad-less
        return self.active(x)


class _DDPForwardPipeline(TrainPipeline[Any, torch.Tensor]):
    """zero_grad -> forward -> backward -> step through the (GA-wrapper-replaced) optimizer.
    The GA wrapper supplies the no_sync context and gates the actual step/zero at window
    boundaries."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        self._optimizer.zero_grad()
        out = self._model(batch)
        loss = out.sum()
        loss.backward()
        self._optimizer.step()
        return loss

    def attach(
        self, model: Optional[nn.Module] = None, sparse_dist: bool = True, **kwargs: Any
    ) -> None:
        # Mirrors TrainPipelineSparseDist.attach, which takes `sparse_dist` POSITIONALLY.
        if model is not None:
            self._model = model
        self.last_sparse_dist: bool = sparse_dist


def _init_single_process_group() -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{tmp.name}", world_size=1, rank=0
    )
    return tmp.name


def _probe(p: nn.Parameter) -> Tuple[bool, bool, Optional[int], int]:
    """(has_grad, is_bucket_view, base_data_ptr, accumulation_count). Holds NO tensor
    reference beyond this call (avoids perturbing AccumulateGrad)."""
    g = p.grad
    if g is None:
        return (False, False, None, 0)
    base = getattr(g, "_base", None)
    return (
        True,
        base is not None,
        base.data_ptr() if base is not None else None,
        int(round(float(g.detach().max()))),
    )


@dataclass
class _Trace:
    active: List[Tuple[bool, bool, Optional[int], int]] = field(default_factory=list)
    unused_has_grad: List[bool] = field(default_factory=list)
    ready_initial: bool = False
    ready_after_first: bool = False
    ready_after_reset: bool = False
    tree_clears: int = 0


class _PGTestBase(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self._init_file: str = _init_single_process_group()

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            os.unlink(self._init_file)
        except OSError:
            pass

    def _make_ddp(self, model: nn.Module, **kwargs: Any) -> DistributedDataParallel:
        return DistributedDataParallel(model, process_group=dist.group.WORLD, **kwargs)


class BucketViewAliasStateMachineTest(_PGTestBase):
    def _drive(self, k: int, accumulate_into_buckets: bool, num_micros: int) -> _Trace:
        model = _ConstGradModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.0)  # lr=0 -> grads constant

        tree_clear_true = [0]
        orig_zero_grad = optimizer.zero_grad

        def _spy_zero_grad(*args: Any, **kwargs: Any) -> None:
            if kwargs.get("set_to_none") is True:
                tree_clear_true[0] += 1
            return orig_zero_grad(*args, **kwargs)

        optimizer.zero_grad = _spy_zero_grad

        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=k,
            num_warmup_steps=1,
            accumulate_into_buckets=accumulate_into_buckets,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp, config)

        tr = _Trace()
        tr.ready_initial = ga._bucket_views_ready
        # Pre-seed an APF-style dummy grad on the active param: micro 0's first-window
        # set_to_none must drop it, so micro 0's count is 1 (not 2).
        model.active.weight.grad = torch.ones_like(model.active.weight)

        batch = torch.ones(1, 4)
        for i in range(num_micros):
            ga.progress(iter([batch]))
            if i == 0:
                tr.ready_after_first = ga._bucket_views_ready
            tr.active.append(_probe(cast(nn.Parameter, model.active.weight)))
            tr.unused_has_grad.append(model.unused.weight.grad is not None)
        ga.reset()
        tr.ready_after_reset = ga._bucket_views_ready
        tr.tree_clears = tree_clear_true[0]
        return tr

    def _assert_state_machine(self, k: int) -> None:
        num_micros = 3 * k  # 3 windows
        num_windows = 3
        on = self._drive(k, True, num_micros)
        off = self._drive(k, False, num_micros)
        # Per-micro accumulation count restarts each window: 1..k repeating.
        expected_counts = [(i % k) + 1 for i in range(num_micros)]

        # --- ON: every micro is a bucket-view (no standalone) with FRESH grad values.
        for i in range(num_micros):
            has, view, _base, count = on.active[i]
            self.assertTrue(
                has and view, f"[K={k}] ON micro {i} not a bucket-view: {on.active[i]}"
            )
            self.assertEqual(
                count,
                expected_counts[i],
                f"[K={k}] ON micro {i} accumulation count {count} != {expected_counts[i]} "
                "(grad.zero_() freshness broken -> would leak across windows)",
            )
        # ON: base_ptr changes EXACTLY ONCE (the one-time static_graph rebuild), then stable.
        on_bases = [b for (_h, _v, b, _c) in on.active]
        transitions = sum(1 for a, b in zip(on_bases, on_bases[1:]) if a != b)
        self.assertEqual(
            transitions,
            1,
            f"[K={k}] expected exactly one bucket rebuild re-point: {on_bases}",
        )

        # --- OFF (bug reproduction): steady-state (window >= 1) no_sync window-start micros
        # allocate a STANDALONE grad; the synced boundary re-aliases to a bucket-view.
        for i in range(k, num_micros):
            has, view, _base, count = off.active[i]
            is_boundary = (i % k) == (k - 1)
            if is_boundary:
                self.assertTrue(
                    view,
                    f"[K={k}] OFF boundary micro {i} should re-alias: {off.active[i]}",
                )
            else:
                self.assertTrue(
                    has and not view,
                    f"[K={k}] OFF non-boundary micro {i} should be STANDALONE (the bug): "
                    f"{off.active[i]}",
                )
            self.assertEqual(
                count, expected_counts[i], f"[K={k}] OFF micro {i} count off"
            )

        # --- Grad-less head stays None on BOTH paths (Distributed-Shampoo-safe).
        self.assertFalse(
            any(on.unused_has_grad), f"[K={k}] ON: grad-less head got a grad"
        )
        self.assertFalse(
            any(off.unused_has_grad), f"[K={k}] OFF: grad-less head got a grad"
        )

        # --- Readiness lifecycle.
        self.assertFalse(
            on.ready_initial, "readiness should be False before any progress"
        )
        self.assertTrue(
            on.ready_after_first, "readiness should be True after first synced backward"
        )
        self.assertTrue(
            on.ready_after_reset, "readiness should survive reset() (same DDP)"
        )

        # --- Exactly one real-optimizer tree-clear per window (fused-LR carrier; the real
        # KeyedOptimizer.zero_grad propagates fused-embedding LR recursively).
        self.assertEqual(
            on.tree_clears, num_windows, f"[K={k}] expected one tree-clear per window"
        )

    def test_alias_state_machine_k2(self) -> None:
        self._assert_state_machine(k=2)

    def test_alias_state_machine_k4(self) -> None:
        self._assert_state_machine(k=4)


class BucketViewDegradationTest(_PGTestBase):
    """The four unsafe/unknown states DEGRADE (exclude + warn once), they do not raise.

    Raising here would be the wrong trade: ``accumulate_into_buckets`` is derived from
    ``K > 1``, so every GA job reaches this code, and a raise on any of these four states
    would convert a working run into a hard crash to protect a memory optimisation.

    Exclusion is provably inert, and that is what these tests assert. A param that is not
    a target is not hidden, so it takes the real ``optimizer.zero_grad(set_to_none=True)``
    tree-clear -- i.e. its grad ends up ``None``, byte-identical to the flag-OFF path.
    Asserting merely "did not raise" would not distinguish that from silently zeroing a
    grad the OFF path would have cleared, which is the divergence being guarded against.
    So every case asserts ``grad is None`` afterwards, and that the skip was WARNED rather
    than silent (a silent skip would let an operator believe they were getting reclaim
    they are not).
    """

    def _wrap_ready(
        self, ddp_owner: nn.Module
    ) -> GradientAccumulationWrapper[Any, torch.Tensor]:
        optimizer = torch.optim.SGD(ddp_owner.parameters(), lr=0.1)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(ddp_owner, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp_owner, config)
        ga._bucket_views_ready = True  # force reaching target collection
        return ga

    def _zero_and_capture_warning(
        self, ga: GradientAccumulationWrapper[Any, torch.Tensor], reason: str
    ) -> None:
        """Run the window-start zero; assert it did not raise and warned once, by reason."""
        with self.assertLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", level="WARNING"
        ) as logs:
            ga._window_start_zero_grad()
        self.assertIn(reason, "".join(logs.output))

    def test_find_unused_parameters_degrades(self) -> None:
        model = _ConstGradModel()
        ddp = self._make_ddp(
            model,
            gradient_as_bucket_view=True,
            static_graph=False,
            find_unused_parameters=True,
        )
        ga = self._wrap_ready(ddp)
        model.active.weight.grad = torch.ones_like(model.active.weight)
        self._zero_and_capture_warning(ga, "find_unused_parameters")
        self.assertIsNone(
            model.active.weight.grad,
            "the excluded DDP's params must take the OFF (set_to_none) path",
        )

    def test_shared_param_two_ddps_degrades(self) -> None:
        # A single Parameter reduced by TWO distinct DDP reducers (double ownership).
        shared = nn.Linear(4, 4, bias=False)

        class _Leaf(nn.Module):
            def __init__(self, s: nn.Module) -> None:
                super().__init__()
                self.s = s

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.s(x)

        ddp_a = self._make_ddp(
            _Leaf(shared), gradient_as_bucket_view=True, static_graph=True
        )
        ddp_b = self._make_ddp(
            _Leaf(shared), gradient_as_bucket_view=True, static_graph=True
        )

        class _Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a: nn.Module = ddp_a
                self.b: nn.Module = ddp_b

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.a(x) + self.b(x)

        root = _Root()
        ga = self._wrap_ready(root)
        shared.weight.grad = torch.ones_like(shared.weight)
        self._zero_and_capture_warning(ga, "double_ddp_ownership")
        self.assertIsNone(
            shared.weight.grad,
            "the double-owned param must take the OFF (set_to_none) path",
        )

    def test_missing_module_parameters_degrades(self) -> None:
        model = _ConstGradModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        ga = self._wrap_ready(ddp)
        model.active.weight.grad = torch.ones_like(model.active.weight)
        # Simulate an unknown DDP variant lacking the authoritative ownership list.
        del ddp._module_parameters
        self._zero_and_capture_warning(ga, "missing_module_parameters")
        self.assertIsNone(
            model.active.weight.grad,
            "the whole DDP is excluded, so its params take the OFF path",
        )

    def test_owned_bucket_view_param_with_standalone_grad_degrades(self) -> None:
        # An optimizer-owned param of a gradient_as_bucket_view DDP whose grad is
        # STANDALONE (grad._base is None) is an unknown alias state (the bucket-view
        # was dropped) -> exclude it rather than zero a non-bucket-view in place.
        model = _ConstGradModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        ga = self._wrap_ready(ddp)  # optimizer over ddp.parameters(); ready forced True
        model.active.weight.grad = torch.ones_like(model.active.weight)  # standalone
        self.assertIsNone(
            getattr(model.active.weight.grad, "_base", None),
            "precondition: the injected grad must be standalone (_base is None)",
        )
        self._zero_and_capture_warning(ga, "standalone_grad")
        self.assertIsNone(
            model.active.weight.grad,
            "the standalone-grad param must take the OFF (set_to_none) path",
        )

    def test_warning_is_emitted_once_per_reason(self) -> None:
        """A model with the condition on every window must not log on every window."""
        model = _ConstGradModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        ga = self._wrap_ready(ddp)
        model.active.weight.grad = torch.ones_like(model.active.weight)
        self._zero_and_capture_warning(ga, "standalone_grad")
        model.active.weight.grad = torch.ones_like(model.active.weight)
        with self.assertNoLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", level="WARNING"
        ):
            ga._window_start_zero_grad()


class AttachRejectTest(_PGTestBase):
    def _wrap(self, model: nn.Module) -> GradientAccumulationWrapper[Any, torch.Tensor]:
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        return GradientAccumulationWrapper(pipeline, optimizer, ddp, config)

    def test_attach_distinct_model_rejected(self) -> None:
        ga = self._wrap(_ConstGradModel())
        with self.assertRaisesRegex(
            RuntimeError, "does not support swapping the model"
        ):
            ga.attach(_ConstGradModel())

    def test_attach_same_model_and_none_delegate(self) -> None:
        model = _ConstGradModel()
        ga = self._wrap(model)
        calls: List[Any] = []
        # `attach` is not on the TrainPipeline ABC; the concrete pipelines
        # under test provide it.
        orig = cast(Any, ga._pipeline).attach

        def _spy(m: Optional[nn.Module] = None, **kw: Any) -> None:
            calls.append(m)
            return orig(m, **kw)

        # pyre-ignore[8]: instance spy
        ga._pipeline.attach = _spy
        # Same model + None must NOT raise AND must DELEGATE to the inner pipeline.
        ga.attach(ga._model)
        ga.attach(None)
        self.assertEqual(
            len(calls),
            2,
            "attach() did not delegate same-model / None calls to the inner pipeline",
        )

    def test_attach_forwards_positional_sparse_dist(self) -> None:
        """``TrainPipelineSparseDist.attach`` takes ``sparse_dist`` positionally, so a
        ``(model, **kwargs)`` wrapper signature would raise TypeError and narrow the
        public protocol."""
        model = _ConstGradModel()
        ga = self._wrap(model)
        ga.attach(ga._model, False)
        self.assertIs(cast(Any, ga._pipeline).last_sparse_dist, False)
        ga.attach(ga._model, True)
        self.assertIs(cast(Any, ga._pipeline).last_sparse_dist, True)


class PlainListDDPDiscoveryContractTest(_PGTestBase):
    """Locks the VLE reduction-cadence contract: a DDP stored in a PLAIN Python list is NOT
    discovered by ``_get_ddp_modules`` (so it never enters no_sync -> reduces every micro),
    while a REGISTERED inner DDP IS. Also a source tripwire tied to the REAL VLE class so a
    future ``_lookups`` -> ``nn.ModuleList`` migration is caught."""

    def test_registered_vs_plain_list_ddp_discovery(self) -> None:
        reg_ddp = self._make_ddp(
            nn.Linear(4, 4), gradient_as_bucket_view=True, static_graph=True
        )
        list_ddp = self._make_ddp(
            nn.Linear(4, 4), gradient_as_bucket_view=True, static_graph=True
        )

        class _Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(4, 4)
                self.registered_ddp: nn.Module = reg_ddp  # registered submodule
                self._lookups: List[nn.Module] = [list_ddp]  # plain list (VLE pattern)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.dense(x)

        root = _Root()
        optimizer = torch.optim.SGD(root.dense.parameters(), lr=0.1)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(root, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, root, config)
        discovered = ga._get_ddp_modules()
        self.assertIn(reg_ddp, discovered, "registered inner DDP should be discovered")
        self.assertNotIn(
            list_ddp,
            discovered,
            "plain-list DDP (VLE _lookups pattern) must NOT be discovered (reduces every "
            "micro; a ModuleList migration would silently change this)",
        )

    def test_real_vle_uses_plain_list_not_modulelist(self) -> None:
        """Source tripwire on the ACTUAL ShardedVariableLengthEmbeddingArch: its ``_lookups``
        must remain a plain ``list`` (undiscovered by no_sync). If this fails, VLE was
        migrated to a registered container and now enters no_sync -- the every-micro-reduction
        docs (gradient_accumulation._get_no_sync_context) + the VLE path-coverage scope MUST
        be revisited."""
        try:
            from torchrec.fb.ads.distributed.variable_length_embedding_arch import (
                ShardedVariableLengthEmbeddingArch,
            )
        except Exception as e:  # pragma: no cover - dep availability guard
            self.skipTest(f"VLE class not importable in this target: {e}")
        src = inspect.getsource(ShardedVariableLengthEmbeddingArch.__init__)
        self.assertIn(
            "self._lookups: List[nn.Module] = []",
            src,
            "VLE._lookups is no longer a plain list -- revisit no_sync discovery + docs",
        )
        self.assertNotIn(
            "self._lookups: nn.ModuleList",
            src,
            "VLE._lookups migrated to nn.ModuleList -> now discoverable by no_sync; revisit",
        )


# ──────────────────────────────────────────────────────────────────────────────
# RISK #1 (report §3 D11d): the None-vs-present-zero Shampoo divergence surface.
#
# ``set_to_none`` (OFF) leaves a grad-less param's grad None -> Distributed Shampoo's
# ``is_invalid_grad = grad is None or grad.numel()==0`` SKIPS it. ``accumulate_into_
# buckets`` (ON) zeros the reduction-bucket-view IN PLACE -> a param that was active
# last window keeps a PRESENT-ZERO grad -> Shampoo PARTICIPATES (present-zero). The
# divergence is possible IFF a dense param is ACTIVE in one window and INACTIVE in a
# later one. AFOC cannot exercise it (static_graph=True + find_unused=False => every
# dense param is always used), which is WHY the Phase-B per-window instrumentation
# reporting NO active->inactive transition is the load-bearing inertness proof. This
# test constructs the counterfactual EXPLICITLY (a param present-then-inactive) and,
# MULTI-PROCESS (world_size=2, so the present-zero grad is really all-reduced), shows
# the OFF-vs-ON participation divergence + that an always-active param does NOT diverge.
# ──────────────────────────────────────────────────────────────────────────────


class _ParticipateOnPresentZeroOptimizer(torch.optim.Optimizer):
    """Mirrors Distributed Shampoo's participation rule
    (``is_invalid_grad = grad is None or grad.numel()==0``): a present-zero grad
    PARTICIPATES (state updated), a None/empty grad is SKIPPED. Records per-param
    participation so the test can observe the OFF-vs-ON divergence."""

    def __init__(self, params: Any) -> None:
        super().__init__(params, {"lr": 0.0})
        self.participated: dict[str, int] = {}
        self.skipped: dict[str, int] = {}
        self._names: dict[int, str] = {}

    def name_params(self, named: List[Tuple[str, nn.Parameter]]) -> None:
        for n, p in named:
            self._names[id(p)] = n

    # pyre-ignore[14]: matches torch.optim.Optimizer.step signature loosely.
    def step(self, closure: Any = None) -> None:
        _ = closure
        for group in self.param_groups:
            for p in group["params"]:
                nm = self._names.get(id(p), f"id{id(p)}")
                g = p.grad
                invalid = g is None or g.numel() == 0
                if invalid:
                    self.skipped[nm] = self.skipped.get(nm, 0) + 1
                else:
                    self.participated[nm] = self.participated.get(nm, 0) + 1
                    st = self.state[p]
                    st["updates"] = st.get("updates", 0) + 1


class _TwoParamModel(nn.Module):
    """``active`` (always in the loss) + ``intermittent`` (in the loss only when the
    forward flag is set) — both bias-free Linears fed constant ones."""

    def __init__(self) -> None:
        super().__init__()
        self.active = nn.Linear(4, 4, bias=False)
        self.intermittent = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.active(x) + self.intermittent(x)


def _active_inactive_worker(
    rank: int,
    world_size: int,
    init_file: str,
    accumulate_into_buckets: bool,
    out_dir: str,
) -> None:
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        torch.manual_seed(0)
        k = 2
        model = _TwoParamModel()
        ddp = DistributedDataParallel(
            model,
            process_group=dist.group.WORLD,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        optimizer = _ParticipateOnPresentZeroOptimizer(ddp.parameters())
        optimizer.name_params(list(ddp.module.named_parameters()))
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=k,
            num_warmup_steps=1,
            accumulate_into_buckets=accumulate_into_buckets,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp, config)

        # --- Window 1: BOTH params active (real fwd/bwd/step through the GA flow),
        # establishing present bucket-view grads + readiness. ---
        batch = torch.ones(2, 4)
        for _ in range(k):
            ga.progress(iter([batch]))

        # Isolate the WINDOW-2 divergence: reset participation counters after window 1
        # (both params were active in window 1 -> both participated on both paths).
        optimizer.participated.clear()
        optimizer.skipped.clear()

        # --- Window 2 START zero via the GA-WRAPPER path (exactly what the pipeline
        # calls each window start -- `pipeline._optimizer` is the _GAOptimizerWrapper):
        # ON  -> _window_start_zero_grad -> intermittent.grad = present-zero (in place)
        # OFF -> optimizer.zero_grad(set_to_none=True) -> intermittent.grad = None
        pipeline._optimizer.zero_grad()

        # --- Window 2 is INACTIVE for `intermittent` (no window-2 backward touches it);
        # `active` STAYS active (a real window-2 grad). AFOC can't do this drop-out under
        # static_graph=True/find_unused=False, so we construct it directly. ---
        with torch.no_grad():
            model.active.weight.grad = torch.ones_like(model.active.weight)

        # --- Window 2 optimizer step: the participate-on-present-zero optimizer records
        # each param as participated (present grad) or skipped (None). ---
        optimizer.step()

        blob = {
            "aib": accumulate_into_buckets,
            "participated": dict(optimizer.participated),
            "skipped": dict(optimizer.skipped),
        }
        torch.save(blob, os.path.join(out_dir, f"rank_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ActiveInactiveShampooDivergenceTest(unittest.TestCase):
    """RISK #1: an active->inactive dense param yields DIVERGENT Shampoo participation
    OFF (skip, grad None) vs ON (participate, grad present-zero); an always-active param
    does NOT. Multi-process (world_size=2). Motivates the Phase-B per-window
    active->inactive instrumentation as the load-bearing AFOC inertness proof."""

    def _run(self, accumulate_into_buckets: bool) -> List[dict]:
        world_size = 2
        with tempfile.TemporaryDirectory() as out_dir:
            init = tempfile.NamedTemporaryFile(delete=False)
            init.close()
            try:
                mp.spawn(
                    _active_inactive_worker,
                    args=(world_size, init.name, accumulate_into_buckets, out_dir),
                    nprocs=world_size,
                    join=True,
                )
                return [
                    torch.load(
                        os.path.join(out_dir, f"rank_{r}.pt"), weights_only=False
                    )
                    for r in range(world_size)
                ]
            finally:
                try:
                    os.unlink(init.name)
                except OSError:
                    pass

    def test_active_inactive_participation_divergence(self) -> None:
        on_blobs = self._run(accumulate_into_buckets=True)
        off_blobs = self._run(accumulate_into_buckets=False)
        for rank, blob in enumerate(on_blobs):
            # ON: intermittent kept a present-zero grad in window 2 -> PARTICIPATES.
            self.assertGreaterEqual(
                blob["participated"].get("intermittent.weight", 0),
                1,
                f"[ON rank{rank}] intermittent (active->inactive) should PARTICIPATE "
                f"(present-zero grad): {blob}",
            )
            self.assertEqual(
                blob["skipped"].get("intermittent.weight", 0),
                0,
                f"[ON rank{rank}] intermittent should NOT be skipped under ON: {blob}",
            )
        for rank, blob in enumerate(off_blobs):
            # OFF: intermittent went None in window 2 -> SKIPPED (Shampoo drops it).
            self.assertGreaterEqual(
                blob["skipped"].get("intermittent.weight", 0),
                1,
                f"[OFF rank{rank}] intermittent (active->inactive) should be SKIPPED "
                f"(grad None): {blob}",
            )
            self.assertEqual(
                blob["participated"].get("intermittent.weight", 0),
                0,
                f"[OFF rank{rank}] intermittent should NOT participate under OFF: {blob}",
            )
        # --- The ALWAYS-active param participates on BOTH paths (NO divergence): this is
        # the AFOC regime (every dense param always used) => the fix is inert there. ---
        for blob in on_blobs + off_blobs:
            self.assertGreaterEqual(
                blob["participated"].get("active.weight", 0),
                1,
                f"always-active param must participate on both paths: {blob}",
            )
            self.assertEqual(
                blob["skipped"].get("active.weight", 0),
                0,
                f"always-active param must never be skipped: {blob}",
            )


# ──────────────────────────────────────────────────────────────────────────────
# H3 additions (session 9): coverage gaps not exercised above — the H1.1
# optimizer-ownership intersection (the key pre-fail/post-pass), the
# exception-restore (finally) alias preservation, and the fused-LR tree-clear on a
# non-target owned leaf. (The non-view grad._base-is-None guard is in
# BucketViewFailClosedTest; RISK#1, find_unused, double-owner + missing
# _module_parameters are covered above.)
# ──────────────────────────────────────────────────────────────────────────────


class _OwnedSubsetModel(nn.Module):
    """``owned`` + ``unowned``, both bias-free Linears fed constant ones -> after a
    synced DDP backward BOTH carry an all-ones bucket-view grad. The wrapped optimizer
    is built over ``owned`` ONLY, so ``unowned`` is DDP-reduced but NOT
    optimizer-owned."""

    def __init__(self) -> None:
        super().__init__()
        self.owned = nn.Linear(4, 4, bias=False)
        self.unowned = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.owned(x) + self.unowned(x)


class OwnershipIntersectionTest(_PGTestBase):
    """H1.1: a param that is DDP-reduced (in ``_module_parameters``, requires_grad,
    live bucket-view) but NOT in the wrapped optimizer's ``param_groups`` MUST be
    EXCLUDED from ``_collect_bucket_view_targets`` so it falls to the plain path and ON
    does not diverge from OFF (whose ``optimizer.zero_grad`` only clears
    optimizer-owned param_groups). Pre-H1.1 ON zeroed EVERY selected bucket-view target
    in place -> an un-owned DDP grad was zeroed under ON yet left accumulating under OFF
    (a None-vs-present divergence for that leaf). This asserts the post-fix exclusion.
    """

    def test_ddp_reduced_but_not_optimizer_owned_is_excluded(self) -> None:
        model = _OwnedSubsetModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        # Optimizer owns ONLY `owned` -> `unowned` is reduced-but-unowned.
        optimizer = torch.optim.SGD(model.owned.parameters(), lr=0.0)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp, config)

        # One synced window establishes bucket-view aliases + readiness for BOTH params.
        batch = torch.ones(2, 4)
        ga.progress(iter([batch]))
        self.assertTrue(ga._bucket_views_ready)
        # Precondition: BOTH params carry a live bucket-view grad (so the ONLY reason
        # `unowned` is excluded is the ownership intersection, not a missing grad).
        self.assertIsNotNone(model.owned.weight.grad)
        self.assertIsNotNone(model.unowned.weight.grad)
        self.assertIsNotNone(getattr(model.unowned.weight.grad, "_base", None))

        target_ids = {id(p) for p, _g in ga._collect_bucket_view_targets()}
        self.assertIn(
            id(model.owned.weight),
            target_ids,
            "optimizer-owned bucket-view param must be an in-place-zero target",
        )
        self.assertNotIn(
            id(model.unowned.weight),
            target_ids,
            "DDP-reduced-but-NOT-optimizer-owned param must be EXCLUDED (H1.1) so ON "
            "does not zero a grad OFF leaves accumulating",
        )


class BucketViewExceptionRestoreTest(_PGTestBase):
    """If the real-optimizer tree-clear raises, the ``finally`` MUST restore the saved
    bucket-view aliases (targets NOT left at None -> HBM reclaim keeps working and the
    next backward accumulates into the persistent bucket) and MUST NOT zero them (the
    in-place ``grad.zero_()`` runs only on the success path, after restore)."""

    def test_tree_clear_exception_restores_and_does_not_zero_aliases(self) -> None:
        model = _ConstGradModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.0)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp, config)
        ga.progress(iter([torch.ones(1, 4)]))  # establish aliases + readiness
        self.assertTrue(ga._bucket_views_ready)
        targets = ga._collect_bucket_view_targets()
        self.assertTrue(targets, "need >=1 bucket-view target for the restore test")
        saved = [
            (p, p.grad, float(none_throws(p.grad).detach().max())) for p, _g in targets
        ]

        def _boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("boom-tree-clear")

        ga._optimizer_wrapper._optimizer.zero_grad = _boom
        with self.assertRaisesRegex(RuntimeError, "boom-tree-clear"):
            ga._window_start_zero_grad()

        for p, saved_grad, val_before in saved:
            self.assertIsNotNone(
                p.grad,
                "target grad left at None after a tree-clear exception (alias dropped)",
            )
            self.assertIs(
                p.grad, saved_grad, "target grad not restored to the saved bucket-view"
            )
            self.assertIsNotNone(
                getattr(p.grad, "_base", None), "restored grad is not a bucket-view"
            )
            self.assertEqual(
                float(p.grad.detach().max()),
                val_before,
                "target grad was zeroed despite the exception (zero() must run only on "
                "the success path, after restore)",
            )


class FusedLRTreeClearTest(_PGTestBase):
    """The window-start zero HIDES only the bucket-view targets, then runs the REAL
    optimizer tree-clear (the fused-embedding-LR carrier + correct None semantics) which
    set_to_none-clears every NON-target owned leaf. Shows a non-target owned leaf's grad
    IS cleared while the target's bucket-view alias is preserved + zeroed in place."""

    def test_non_target_leaf_cleared_while_target_alias_preserved(self) -> None:
        model = _ConstGradModel()
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        # A plain dense leaf OUTSIDE the DDP, owned by the optimizer -> a non-target.
        extra = nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.SGD(list(ddp.parameters()) + [extra], lr=0.0)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp, config)
        ga.progress(iter([torch.ones(1, 4)]))  # establish target alias + readiness
        self.assertTrue(ga._bucket_views_ready)
        extra.grad = torch.ones(4)  # non-target owned leaf carries a standalone grad
        target = model.active.weight
        self.assertIsNotNone(target.grad)
        self.assertIsNotNone(getattr(target.grad, "_base", None))

        ga._window_start_zero_grad()

        self.assertIsNone(
            extra.grad,
            "non-target owned leaf grad must be cleared by the real tree-clear",
        )
        self.assertIsNotNone(target.grad, "target bucket-view alias was dropped")
        self.assertIsNotNone(
            getattr(target.grad, "_base", None), "target is no longer a bucket-view"
        )
        self.assertEqual(
            float(target.grad.detach().abs().max()),
            0.0,
            "target bucket-view grad was not zeroed in place",
        )


class ForeachZeroCoalescingTest(_PGTestBase):
    """H4.1: the window-start in-place zero coalesces the per-target ``grad.zero_()``
    into ``torch._foreach_zero_`` per (device,dtype). With MULTIPLE owned bucket-view
    targets: every target is zeroed, aliases are preserved (still bucket-views), and a
    subsequent synced window accumulates freshly (the foreach in-place zero took AND did
    not break the next backward via the version-counter it skips). The multi-window
    freshness/version behavior is also covered end-to-end by
    BucketViewAliasStateMachineTest (which now exercises the foreach path)."""

    def test_foreach_zero_multi_target_preserves_aliases(self) -> None:
        model = _TwoParamModel()  # active + intermittent, both in the loss -> 2 grads
        ddp = self._make_ddp(model, gradient_as_bucket_view=True, static_graph=True)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.0)  # own BOTH params
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=2,
            num_warmup_steps=1,
            accumulate_into_buckets=True,
        )
        pipeline = _DDPForwardPipeline(ddp, optimizer)
        ga = GradientAccumulationWrapper(pipeline, optimizer, ddp, config)
        ga.progress(iter([torch.ones(2, 4)]))  # synced window 0 -> bucket-views ready
        self.assertTrue(ga._bucket_views_ready)
        targets = ga._collect_bucket_view_targets()
        self.assertGreaterEqual(
            len(targets), 2, "need >=2 owned bucket-view targets to exercise grouping"
        )

        ga._window_start_zero_grad()  # foreach-coalesced in-place zero

        for p, _g in targets:
            self.assertIsNotNone(p.grad, "target alias dropped by the foreach path")
            self.assertIsNotNone(
                getattr(p.grad, "_base", None),
                "target no longer a bucket-view after foreach zero",
            )
            self.assertEqual(
                float(p.grad.detach().abs().max()),
                0.0,
                "target grad not zeroed by torch._foreach_zero_",
            )
