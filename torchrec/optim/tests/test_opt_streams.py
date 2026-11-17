#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import threading
import unittest
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torchrec.optim import clipping, keyed
from torchrec.optim.clipping import GradientClipping, GradientClippingOptimizer
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.warmup import WarmupOptimizer, WarmupPolicy, WarmupStage

# Sizes for the tests: optimizers with at least _BIG elements are "big",
# tensors are cut at multiples of _ALIGN, calls below _MIN are not split.
_BIG = 4096
_ALIGN = 64
_MIN = 512
_OFF: Dict[str, str] = {
    "ONEFLOW_OPT_ADAMW_STREAMS": "0",
    "ONEFLOW_OPT_SMALL_STREAM": "0",
    "ONEFLOW_OPT_STREAMS_CHECK": "0",
    "ONEFLOW_CLIP_FLAT": "0",
    "ONEFLOW_CLIP_FLAT_SEGMENTS": "0",
}
_BIG_SHAPES: List[Tuple[int, ...]] = [
    (37, 64),
    (5,),
    (1000,),
    (3, 7, 11),
    (4000,),
    (130,),
    (64, 64),
]
_HEAD_SHAPES: List[Tuple[int, ...]] = [(16, 8), (8,), (33,)]


def _env(**flags: bool) -> Dict[str, str]:
    env = dict(_OFF)
    names = {
        "adamw": "ONEFLOW_OPT_ADAMW_STREAMS",
        "small": "ONEFLOW_OPT_SMALL_STREAM",
        "check": "ONEFLOW_OPT_STREAMS_CHECK",
        "flat": "ONEFLOW_CLIP_FLAT",
        "segments": "ONEFLOW_CLIP_FLAT_SEGMENTS",
    }
    for name, on in flags.items():
        env[names[name]] = "1" if on else "0"
    return env


# The tests set every flag they need; flags exported by the caller must not
# leak into their reference steps.
_HERMETIC_ENV = mock.patch.dict(os.environ)


def setUpModule() -> None:
    _HERMETIC_ENV.start()
    prefixes = ("ONEFLOW_OPT_", "ONEFLOW_CLIP_")
    for name in [k for k in os.environ if k.startswith(prefixes)]:
        del os.environ[name]


def tearDownModule() -> None:
    _HERMETIC_ENV.stop()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _fresh_state() -> List[Any]:
    """Patches of the process state of the flags, fresh for each test."""
    return [
        mock.patch.object(keyed, "_OPT_SIDE_STREAMS", {}, create=True),
        mock.patch.object(keyed, "_OPT_STREAMS_LOGGED", set(), create=True),
        mock.patch.object(keyed, "_OPT_STREAMS_PENDING", [], create=True),
        mock.patch.object(
            keyed,
            "_SMALL_STREAM",
            {"checks": 3, "off": False, "plan": None, "steps": 0},
            create=True,
        ),
        mock.patch.object(
            clipping,
            "_ADAMW_STREAMS",
            {"checks": 3, "off": False, "plans": {}, "calls": 0, "coverage": {}},
            create=True,
        ),
    ]


class _Stream:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOps:
    """CPU stand-in for ``keyed._OptStreamOps``. Logs waits, records and
    deferred verdicts, and tracks the current stream so that ``_OpStreams``
    can tag every op with it. Deferred copies complete when ``ready``."""

    def __init__(self) -> None:
        self.main = _Stream("main")
        self.cur: _Stream = self.main
        self.log: List[Tuple[Any, ...]] = []
        self.created = 0
        self.supported = True
        self.ready = True

    def supports(self, device: torch.device) -> bool:
        return self.supported and device.type == "cpu"

    def current(self, device: torch.device) -> _Stream:
        return self.cur

    def new(self, device: torch.device) -> _Stream:
        self.created += 1
        return _Stream(f"side{self.created}")

    @contextmanager
    def use(self, stream: _Stream) -> Iterator[None]:
        previous = self.cur
        self.cur = stream
        try:
            yield
        finally:
            self.cur = previous

    def wait(self, waiter: _Stream, producer: _Stream) -> None:
        self.log.append(("wait", waiter.name, producer.name))

    def record(self, tensor: torch.Tensor, stream: _Stream) -> None:
        self.log.append(("record", stream.name, id(tensor)))

    def name(self, stream: _Stream) -> str:
        return stream.name

    def defer(self, flag: torch.Tensor, stream: _Stream) -> Tuple[torch.Tensor, Any]:
        self.log.append(("defer", stream.name))
        return flag.clone(), object()

    def done(self, event: Any) -> bool:
        return self.ready


class _OpStreams(TorchDispatchMode):
    """Logs (op, stream) for every aten op."""

    def __init__(self, ops: _FakeOps) -> None:
        super().__init__()
        self.ops = ops

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: Any = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self.ops.log.append(("op", func.overloadpacket.__name__, self.ops.cur.name))
        return func(*args, **(kwargs or {}))


def _params(
    shapes: List[Tuple[int, ...]], seed: int, dtype: torch.dtype = torch.float32
) -> List[torch.nn.Parameter]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=gen).to(dtype)) for s in shapes]


def _keyed(name: str, params: List[torch.nn.Parameter], **kwargs: Any) -> Any:
    kwargs.setdefault("fused", True)
    cls = kwargs.pop("cls", torch.optim.AdamW)
    lr = kwargs.pop("lr", 1e-3)
    return KeyedOptimizerWrapper(
        {f"{name}.{i}": p for i, p in enumerate(params)},
        lambda ps: cls(
            ps, lr=lr, betas=(0.9, 0.99), eps=1e-5, weight_decay=1e-3, **kwargs
        ),
    )


def _clipped(
    name: str,
    params: List[torch.nn.Parameter],
    lr: Any = 1e-3,
    warmup: bool = True,
    **kwargs: Any,
) -> GradientClippingOptimizer:
    """GradientClippingOptimizer(WarmupOptimizer(KeyedOptimizerWrapper(AdamW)))
    as built for the dense groups."""
    opt: Any = _keyed(name, params, lr=lr, **kwargs)
    if warmup:
        opt = WarmupOptimizer(
            opt,
            [
                WarmupStage(policy=WarmupPolicy.LINEAR, max_iters=3, value=0.25),
                WarmupStage(policy=WarmupPolicy.CONSTANT, max_iters=1 << 30, value=0.7),
            ],
            lr=float(lr),
            param_name=f"{name}.__warmup",
        )
    return GradientClippingOptimizer(opt, GradientClipping.NORM, max_gradient=1.0)


def _set_grads(params: List[torch.nn.Parameter], step: int) -> None:
    for j, p in enumerate(params):
        gen = torch.Generator().manual_seed(1000 * step + j)
        scale = 3.0 if step % 2 else 0.003
        p.grad = (torch.randn(p.shape, generator=gen) * scale).to(p.dtype)


def _set_bucket_grads(params: List[torch.nn.Parameter], step: int) -> None:
    """Grads as views of shared fp32 buffers, like DDP gradient_as_bucket_view."""
    _set_grads(params, step)
    fp32 = [p for p in params if p.dtype == torch.float32]
    half = len(fp32) // 2
    for chunk in (fp32[:half], fp32[half:]):
        # pyrefly: ignore [missing-attribute]
        buf = torch.cat([p.grad.reshape(-1) for p in chunk])
        offset = 0
        for p in chunk:
            p.grad = buf[offset : offset + p.numel()].view(p.shape)
            offset += p.numel()


class _Model:
    """Two small head groups, an optimizer that cannot use the side stream
    and a big default group with fp32 and fp16 params, in a
    CombinedOptimizer like the job's."""

    def __init__(self, seed: int = 0, buckets: bool = False) -> None:
        self.buckets = buckets
        self.head_a = _params(_HEAD_SHAPES, seed + 1)
        self.head_b = _params(_HEAD_SHAPES[:2], seed + 2)
        self.other = _params([(7,)], seed + 3)
        self.big = _params(_BIG_SHAPES, seed + 4) + _params(
            [(300,), (400,)], seed + 5, torch.float16
        )
        self.groups: Dict[str, GradientClippingOptimizer] = {
            "head_a": _clipped("head_a", self.head_a),
            "head_b": _clipped("head_b", self.head_b),
            "default": _clipped("default", self.big),
        }
        self.nested = CombinedOptimizer(
            [("other", _clipped("other", self.other, fused=False, foreach=True))]
        )
        self.opt = CombinedOptimizer(
            [
                ("head_a", self.groups["head_a"]),
                ("nested", self.nested),
                ("head_b", self.groups["head_b"]),
                ("default", self.groups["default"]),
            ]
        )

    def all_params(self) -> List[torch.nn.Parameter]:
        return self.head_a + self.head_b + self.other + self.big

    def step(self, step: int, env: Dict[str, str]) -> None:
        (_set_bucket_grads if self.buckets else _set_grads)(self.all_params(), step)
        with mock.patch.dict(os.environ, env):
            self.opt.step()

    def tensors(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for p in self.all_params():
            # pyrefly: ignore [bad-assignment]
            out += [p, p.grad]
        for state in self.opt.state.values():
            out += [t for t in state.values() if isinstance(t, torch.Tensor)]
        for g in self.groups.values():
            norm = g.last_total_grad_norm
            assert norm is not None
            out.append(norm)
        return out

    def host_state(self) -> List[Any]:
        return [
            (
                g._step_num,
                [
                    {k: v for k, v in pg.items() if k != "params"}
                    for pg in g.param_groups
                ],
            )
            for g in self.groups.values()
        ]


def _wrong_on_side(ops: _FakeOps) -> Callable[..., None]:
    """torch._fused_adamw_ with a 1.5x learning rate off the main stream."""
    original = torch._fused_adamw_

    def wrong(*args: Any, **kwargs: Any) -> None:
        if ops.cur.name != "main":
            kwargs = dict(kwargs, lr=kwargs["lr"] * 1.5)
        original(*args, **kwargs)

    return wrong


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.ops = _FakeOps()
        # create=True lets the tests run, and fail, on code without the flags.
        for patch in _fresh_state() + [
            mock.patch.object(keyed, "_OPT_STREAM_OPS", self.ops, create=True),
            mock.patch.object(keyed, "_OPT_STREAM_BIG_NUMEL", _BIG, create=True),
            mock.patch.object(clipping, "_OPT_STREAM_BIG_NUMEL", _BIG, create=True),
            mock.patch.object(clipping, "_ADAMW_SPLIT_ALIGN", _ALIGN, create=True),
            mock.patch.object(clipping, "_ADAMW_SPLIT_MIN_NUMEL", _MIN, create=True),
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def assert_same(self, ref: _Model, alt: _Model) -> None:
        a, b = ref.tensors(), alt.tensors()
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(x.dtype, y.dtype)
            self.assertTrue(torch.equal(_bits(x), _bits(y)))
        self.assertEqual(ref.host_state(), alt.host_state())

    def run_pair(
        self,
        envs: List[Dict[str, str]],
        ref_env: Optional[Dict[str, str]] = None,
        buckets: bool = False,
        before: Optional[Callable[[int, _Model], None]] = None,
    ) -> Tuple[_Model, _Model]:
        """Steps a reference with ``ref_env`` and a twin with ``envs[i]`` in
        step i + 1 after a first step without the flags, and checks they
        stay bitwise equal."""
        ref, alt = _Model(buckets=buckets), _Model(buckets=buckets)
        ref.step(0, ref_env or _OFF)
        alt.step(0, ref_env or _OFF)
        for i, env in enumerate(envs):
            if before is not None:
                before(i + 1, ref)
                before(i + 1, alt)
            ref.step(i + 1, ref_env or _OFF)
            with _OpStreams(self.ops):
                alt.step(i + 1, env)
            self.assert_same(ref, alt)
        return ref, alt

    def side(self, role: str, slot: int = 0) -> str:
        return keyed._OPT_SIDE_STREAMS[("cpu", role, slot)].name

    def ops_on(self, stream: str, op: Optional[str] = None) -> List[int]:
        return [
            i
            for i, e in enumerate(self.ops.log)
            if e[0] == "op" and e[2] == stream and (op is None or e[1] == op)
        ]

    def index(self, entry: Tuple[Any, ...]) -> int:
        return self.ops.log.index(entry)

    def assert_joined(self, stream: str, sections: int) -> None:
        """Every op on ``stream`` runs in a section that the side stream opens
        by waiting for the main stream and that the main stream closes by
        waiting for the side stream."""
        is_open = False
        opened = 0
        ops = 0
        for entry in self.ops.log:
            if entry == ("wait", stream, "main"):
                self.assertFalse(is_open)
                is_open, ops = True, 0
                opened += 1
            elif entry == ("wait", "main", stream):
                self.assertTrue(is_open)
                self.assertGreater(ops, 0)
                is_open = False
            elif entry[0] == "op" and entry[2] == stream:
                self.assertTrue(is_open)
                ops += 1
        self.assertFalse(is_open)
        self.assertEqual(opened, sections)


class TestSameBits(unittest.TestCase):
    def test_bitwise_not_close(self) -> None:
        def same_all(a: torch.Tensor, b: torch.Tensor) -> bool:
            verdict = keyed._opt_streams_same_all([(a, b)])
            return verdict is not None and bool(verdict)

        for same in (keyed._opt_streams_same_bits, same_all):
            x = torch.tensor([1.0, -0.0, float("nan")])
            self.assertTrue(same(x, x.clone()))
            self.assertFalse(same(x, torch.tensor([1.0, 0.0, float("nan")])))
            self.assertFalse(
                same(
                    x,
                    torch.tensor(
                        [
                            float(torch.tensor(1.0).nextafter(torch.tensor(2.0))),
                            -0.0,
                            float("nan"),
                        ]
                    ),
                )
            )
            nan2 = x.clone()
            nan2.view(torch.int32)[2] += 1  # another NaN payload
            self.assertFalse(same(x, nan2))
            self.assertTrue(same(x.half(), x.half().clone()))
            self.assertFalse(same(x, x.double()))
            self.assertFalse(same(x, x[:2]))

    def test_same_all_is_one_device_flag(self) -> None:
        x = torch.arange(6.0)
        verdict = keyed._opt_streams_same_all([(x, x.clone()), (x[:2], x[:2] + 0)])
        assert verdict is not None
        self.assertEqual((verdict.dim(), verdict.dtype), (0, torch.bool))
        self.assertTrue(bool(verdict))
        y = x.clone()
        y[5] = 7.0
        verdict = keyed._opt_streams_same_all([(x[:2], x[:2]), (x, y)])
        assert verdict is not None
        self.assertFalse(bool(verdict))
        self.assertIsNone(keyed._opt_streams_same_all([(x, x[:3])]))


class TestAdamWPlan(unittest.TestCase):
    def check(
        self, numels: List[int], cuttable: List[bool], k: int, align: int
    ) -> None:
        parts = clipping._adamw_plan(numels, cuttable, k, align)
        self.assertTrue(1 <= len(parts) <= k)
        self.assertTrue(all(parts))
        covered: List[Tuple[int, int, int]] = []
        for part in parts:
            for seg in part:
                if seg[0] == "run":
                    covered += [(i, 0, numels[i]) for i in range(seg[1], seg[2])]
                else:
                    self.assertTrue(cuttable[seg[1]])
                    self.assertEqual(seg[2] % align, 0)
                    self.assertLess(seg[2], seg[3])
                    covered.append(tuple(seg[1:]))
        # Every element once, in order.
        expected_next = (0, 0)
        for i, start, stop in covered:
            if start == 0:
                self.assertEqual(expected_next, (i, 0))
            else:
                self.assertEqual(expected_next, (i, start))
            expected_next = (i + 1, 0) if stop == numels[i] else (i, stop)
        self.assertEqual(expected_next, (len(numels), 0))
        self.check_pieces(parts, numels, align)

    def check_pieces(self, parts: Any, numels: List[int], align: int) -> None:
        """The check pieces of a plan: both chunks around every cut boundary
        in the ranges on either side, whole chunks of their slices."""
        with mock.patch.object(clipping, "_ADAMW_SPLIT_ALIGN", align):
            pieces, boundaries, _ = clipping._adamw_check_pieces(
                {"parts": parts, "numels": numels}
            )
        self.assertEqual(len(pieces), len(parts))
        cuts = [
            (r, seg) for r, part in enumerate(parts) for seg in part if seg[0] == "cut"
        ]
        self.assertEqual(boundaries, sum(1 for _, seg in cuts if seg[2] > 0))
        for r, (_, i, start, stop) in cuts:
            if start > 0:
                self.assertIn((i, start, min(start + align, stop)), pieces[r])
                self.assertIn((i, start - align, start), pieces[r - 1])
        for r, mine in enumerate(pieces):
            for i, start, stop in mine:
                if stop < 0:
                    continue
                seg = next(s for s in parts[r] if s[0] == "cut" and s[1] == i)
                self.assertTrue(seg[2] <= start < stop <= seg[3])
                self.assertEqual((start - seg[2]) % align, 0)
                self.assertTrue(stop - start == align or stop == numels[i])

    def test_random_plans(self) -> None:
        gen = torch.Generator().manual_seed(0)
        for _ in range(300):
            n = int(torch.randint(1, 9, (), generator=gen))
            numels = torch.randint(1, 3000, (n,), generator=gen).tolist()
            cuttable = (torch.rand(n, generator=gen) < 0.7).tolist()
            k = int(torch.randint(1, 6, (), generator=gen))
            self.check(numels, cuttable, k, 64)

    def test_balanced_cut(self) -> None:
        parts = clipping._adamw_plan([1000, 1000, 1000], [True] * 3, 3, 64)
        self.assertEqual(
            parts,
            [[("run", 0, 1)], [("run", 1, 2)], [("run", 2, 3)]],
        )
        parts = clipping._adamw_plan([3000], [True], 3, 64)
        self.assertEqual(
            parts,
            [[("cut", 0, 0, 960)], [("cut", 0, 960, 1920)], [("cut", 0, 1920, 3000)]],
        )
        # An atomic tensor goes whole into the current range.
        parts = clipping._adamw_plan([100, 3000, 100], [True, False, True], 3, 64)
        self.assertEqual(parts, [[("run", 0, 2)], [("run", 2, 3)]])

    def test_check_pieces(self) -> None:
        def pieces(numels: List[int], cuttable: List[bool], k: int) -> Any:
            parts = clipping._adamw_plan(numels, cuttable, k, 64)
            return clipping._adamw_check_pieces({"parts": parts, "numels": numels})

        with mock.patch.object(clipping, "_ADAMW_SPLIT_ALIGN", 64):
            self.assertEqual(
                pieces([3000], [True], 3),
                (
                    [
                        [(0, 896, 960)],
                        [(0, 960, 1024), (0, 1856, 1920)],
                        [(0, 1920, 1984)],
                    ],
                    2,
                    0,
                ),
            )
            # A one-chunk slice between two boundaries is one piece.
            self.assertEqual(
                pieces([200], [True], 3),
                ([[(0, 0, 64)], [(0, 64, 128)], [(0, 128, 192)]], 2, 0),
            )
            # The smallest non-empty whole tensor of each range.
            self.assertEqual(
                pieces([0, 100, 3000, 100], [False, True, False, True], 3),
                ([[(1, 0, -1)], [(3, 0, -1)]], 0, 2),
            )
            with mock.patch.object(clipping, "_ADAMW_CHECK_WHOLE_MAX", 99):
                self.assertEqual(
                    pieces([0, 100, 3000, 100], [False, True, False, True], 3),
                    ([[], []], 0, 0),
                )

    def test_cuttable_needs_ilp_multiple_and_contiguity(self) -> None:
        with mock.patch.object(clipping, "_ADAMW_SPLIT_ALIGN", 64):
            t = torch.zeros(1000)
            self.assertTrue(clipping._adamw_cuttable([t, t, t], 1000))
            self.assertFalse(clipping._adamw_cuttable([torch.zeros(1001)] * 3, 1001))
            self.assertFalse(clipping._adamw_cuttable([torch.zeros(64)] * 3, 64))
            strided = torch.zeros(2000)[::2]
            self.assertFalse(clipping._adamw_cuttable([t, strided, t], 1000))


class TestAdamWStreams(_Base):
    def test_split_matches_unsplit_and_orders_streams(self) -> None:
        def odd_step(step: int, model: _Model) -> None:
            if step == 2:  # per-tensor step counters must stay per tensor
                model.opt.state[model.big[4]]["step"].fill_(7.0)

        env = dict(_env(adamw=True), ONEFLOW_OPT_ADAMW_STREAMS_K="3")
        with self.assertLogs(level="INFO") as logs:
            self.run_pair([env] * 4, before=odd_step)
        text = "\n".join(logs.output)
        self.assertIn(
            "ONEFLOW_OPT_ADAMW_STREAMS: split torch._fused_adamw_ (7 torch.float32 "
            "tensors, 11830 elements) with K=3 into ranges of [3924, 3938, 3968] "
            "elements (2 tensors cut at 64-element boundaries) on cpu: range 0 on the "
            "current stream main, the others on side streams side1, side2",
            text,
        )
        self.assertIn("split torch._fused_adamw_ (2 torch.float16 tensors", text)
        self.assertIn("keep the unsplit fused AdamW", text)
        # Only the unfused optimizer of the nested CombinedOptimizer declines
        # (other loggers, e.g. Triton's import banner, are not ours).
        warnings = [line for line in logs.output if line.startswith("WARNING:root:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("AdamW is not a fused torch Adam(W)", warnings[0])
        sides = [self.side("adamw", j) for j in range(2)]
        for side in sides:
            self.assertEqual(len(self.ops_on(side, "_fused_adamw_")), 2 * 4)
            self.assert_joined(side, 2 * 4)
        # The step increments of _fused_adam come before the side streams start.
        first_wait = self.index(("wait", sides[0], "main"))
        adds = [
            i
            for i, e in enumerate(self.ops.log)
            if e[0] == "op" and e[1] == "_foreach_add_" and i < first_wait
        ]
        self.assertTrue(adds)
        # fp32 and fp16 calls both split; heads stay unsplit on main.
        self.assertEqual(len(self.ops_on("main", "_fused_adamw_")), 4 * 4)

    def test_default_k_is_2(self) -> None:
        with self.assertLogs(level="INFO") as logs:
            self.run_pair([_env(adamw=True)] * 2)
        self.assertIn(
            "with K=2 into ranges of [5908, 5922] elements (1 tensors cut at "
            "64-element boundaries) on cpu: range 0 on the current stream main, the "
            "others on side streams side1",
            "\n".join(logs.output),
        )
        self.assertEqual(list(keyed._OPT_SIDE_STREAMS), [("cpu", "adamw", 0)])

    def test_first_step_keeps_unsplit(self) -> None:
        ref, alt = _Model(), _Model()
        ref.step(0, _OFF)
        with self.assertLogs(level="INFO") as logs:
            with _OpStreams(self.ops):
                alt.step(0, _env(adamw=True, check=True))
        self.assertIn(
            "INFO:root:ONEFLOW_OPT_ADAMW_STREAMS: the first step of an optimizer "
            "(its lazy state init, e.g. RecOptimizer.init_state) keeps the unsplit "
            "fused AdamW; the split starts with its next step",
            logs.output,
        )
        warnings = [line for line in logs.output if "WARNING" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("AdamW is not a fused torch Adam(W)", warnings[0])
        self.assertEqual(keyed._OPT_SIDE_STREAMS, {})
        self.assertEqual(clipping._ADAMW_STREAMS["checks"], 3)
        self.assertEqual(clipping._ADAMW_STREAMS["calls"], 0)
        self.assertEqual([e for e in self.ops.log if e[0] == "wait"], [])
        self.assert_same(ref, alt)
        ref.step(1, _OFF)
        alt.step(1, _env(adamw=True))
        self.assert_same(ref, alt)
        self.assertIn(("cpu", "adamw", 0), keyed._OPT_SIDE_STREAMS)

    def test_k_values(self) -> None:
        for k in (2, 3, 4):
            with self.subTest(k=k):
                keyed._OPT_SIDE_STREAMS.clear()
                env = dict(_env(adamw=True), ONEFLOW_OPT_ADAMW_STREAMS_K=str(k))
                self.run_pair([env] * 2)
                self.assertEqual(
                    sorted(r for (_, r, _) in keyed._OPT_SIDE_STREAMS),
                    ["adamw"] * (k - 1),
                )

    def test_tensor_lr_without_warmup(self) -> None:
        def build() -> Tuple[GradientClippingOptimizer, List[torch.nn.Parameter]]:
            params = _params(_BIG_SHAPES, 4)
            return _clipped("d", params, lr=torch.tensor(2e-3), warmup=False), params

        (ref, rp), (alt, ap) = build(), build()
        for step in range(4):
            _set_grads(rp, step)
            _set_grads(ap, step)
            ref.step()
            with mock.patch.dict(os.environ, _env(adamw=step > 0)):
                alt.step()
            for x, y in zip(rp, ap):
                self.assertTrue(torch.equal(_bits(x), _bits(y)))
        self.assertIn(("cpu", "adamw", 0), keyed._OPT_SIDE_STREAMS)

    def test_toggles_per_step(self) -> None:
        envs = [_env(adamw=s % 2 == 0, small=s % 3 == 0) for s in range(6)]
        self.run_pair(envs)

    def test_declines_keep_stock_kernels(self) -> None:
        cases = [
            ({"ONEFLOW_OPT_ADAMW_STREAMS_K": "1"}, {}, "is not an integer in 2..8"),
            ({}, {"fused": False, "foreach": True}, "is not a fused torch Adam(W)"),
            ({}, {"cls": torch.optim.Adam}, "Adam is not torch.optim.AdamW"),
            ({}, {"amsgrad": True}, "amsgrad"),
        ]
        for extra_env, kwargs, message in cases:
            with self.subTest(message=message):
                keyed._OPT_STREAMS_LOGGED.clear()
                # pyrefly: ignore [bad-argument-type]
                ref = _clipped("d", _params(_BIG_SHAPES, 4), **kwargs)
                # pyrefly: ignore [bad-argument-type]
                alt = _clipped("d", _params(_BIG_SHAPES, 4), **kwargs)
                env = dict(_env(adamw=True), **extra_env)
                with self.assertLogs(level="WARNING") as logs:
                    for step in range(2):
                        for opt in (ref, alt):
                            # pyrefly: ignore [bad-argument-type]
                            _set_grads(list(opt.params.values())[:-1], step)
                        ref.step()
                        with mock.patch.dict(os.environ, env):
                            alt.step()
                self.assertIn(message, "\n".join(logs.output))
                for x, y in zip(
                    list(ref.params.values())[:-1], list(alt.params.values())[:-1]
                ):
                    self.assertTrue(torch.equal(_bits(x), _bits(y)))
                self.assertEqual(keyed._OPT_SIDE_STREAMS, {})

    def test_unsupported_device_warns_once(self) -> None:
        self.ops.supported = False
        with self.assertLogs(level="WARNING") as logs:
            self.run_pair([_env(adamw=True)] * 3)
        text = "\n".join(logs.output)
        self.assertEqual(text.count("no torch._fused_adamw_ call was split"), 1)
        self.assertIn("cpu is not a supported device", text)
        self.assertEqual(keyed._OPT_SIDE_STREAMS, {})

    def test_restores_fused_adamw(self) -> None:
        original = torch._fused_adamw_
        model = _Model()
        model.step(0, _OFF)
        seen: List[Any] = []
        real_step = KeyedOptimizerWrapper.step

        def spy(opt: KeyedOptimizerWrapper, closure: Any = None) -> None:
            seen.append(torch._fused_adamw_)
            real_step(opt, closure)

        with mock.patch.object(KeyedOptimizerWrapper, "step", spy):
            model.step(1, _env(adamw=True))
        self.assertIsInstance(seen[-1], clipping._AdamWStreams)
        self.assertTrue(all(s is original for s in seen[:-1]))
        self.assertIs(torch._fused_adamw_, original)
        with mock.patch.object(
            KeyedOptimizerWrapper, "step", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                model.step(2, _env(adamw=True))
        self.assertIs(torch._fused_adamw_, original)
        self.assertEqual(getattr(keyed._OPT_STREAMS_TLS, "clipping", 0), 0)

    def test_default_path_untouched(self) -> None:
        original = torch._fused_adamw_
        model = _Model()
        seen: List[Any] = []
        real_step = KeyedOptimizerWrapper.step

        def spy(opt: KeyedOptimizerWrapper, closure: Any = None) -> None:
            seen.append(torch._fused_adamw_)
            real_step(opt, closure)

        env = {k: v for k, v in os.environ.items() if not k.startswith("ONEFLOW_")}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(KeyedOptimizerWrapper, "step", spy):
                with self.assertNoLogs(level="INFO"):
                    for step in range(3):
                        _set_grads(model.all_params(), step)
                        model.opt.step()
        self.assertTrue(seen and all(s is original for s in seen))
        self.assertEqual(self.ops.log, [])
        self.assertEqual(keyed._OPT_SIDE_STREAMS, {})

    def test_other_threads_pass_through(self) -> None:
        model = _Model()
        model.step(0, _OFF)
        calls: List[Tuple[int, int]] = []
        original = torch._fused_adamw_

        def spy(*args: Any, **kwargs: Any) -> None:
            first = args[0][0]
            calls.append((first.untyped_storage().data_ptr(), first.numel()))
            original(*args, **kwargs)

        p = torch.nn.Parameter(torch.zeros(4100))
        state = (
            [torch.ones(4100)]
            + [torch.zeros(4100) for _ in range(2)]
            + [torch.ones(())]
        )
        kwargs: Dict[str, Any] = {
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.99,
            "weight_decay": 0.0,
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "grad_scale": None,
            "found_inf": None,
        }

        def other() -> None:
            torch._fused_adamw_(
                [p.data], [state[0]], [state[1]], [state[2]], [], [state[3]], **kwargs
            )

        real_step = KeyedOptimizerWrapper.step

        def run_other(opt: KeyedOptimizerWrapper, closure: Any = None) -> None:
            if isinstance(torch._fused_adamw_, clipping._AdamWStreams):
                thread = threading.Thread(target=other)
                thread.start()
                thread.join()
            real_step(opt, closure)

        with mock.patch.object(torch, "_fused_adamw_", spy):
            with mock.patch.object(KeyedOptimizerWrapper, "step", run_other):
                model.step(1, _env(adamw=True))
        # The other thread's call reached the stock op whole.
        mine = [n for ptr, n in calls if ptr == p.untyped_storage().data_ptr()]
        self.assertEqual(mine, [4100])
        self.assertNotEqual(float(p.detach().abs().sum()), 0.0)

    def test_check_passes_then_splits(self) -> None:
        with self.assertLogs(level="INFO") as logs:
            self.run_pair([_env(adamw=True, check=True)] * 5)
        matched = [line for line in logs.output if "matched" in line]
        self.assertEqual(len(matched), 1)
        for part in (
            "INFO:root:ONEFLOW_OPT_STREAMS_CHECK: ONEFLOW_OPT_ADAMW_STREAMS matched "
            "the unsplit torch._fused_adamw_ bitwise (params, exp_avg, exp_avg_sq "
            "of the 64-element chunks on both sides of every cut boundary and of "
            "the smallest whole tensor of each range; ",
            "torch.float32 calls: 4 pieces: both sides of all 1 cut boundaries in 1 "
            "cut tensors, and 2 whole tensors",
            "torch.float16 calls: 2 pieces: both sides of all 0 cut boundaries in 0 "
            "cut tensors, and 2 whole tensors",
            ") in 3 calls",
        ):
            self.assertIn(part, matched[0])
        self.assertEqual(clipping._ADAMW_STREAMS["checks"], 0)
        self.assertFalse(clipping._ADAMW_STREAMS["off"])
        # Checked calls run pieces on the side stream too; no deferred verdicts.
        self.assertTrue(self.ops_on(self.side("adamw"), "_fused_adamw_"))
        self.assertEqual([e for e in self.ops.log if e[0] == "defer"], [])

    def test_check_mismatch_is_fatal_and_keeps_stock_result(self) -> None:
        ref, alt = _Model(), _Model()
        with mock.patch.object(torch, "_fused_adamw_", _wrong_on_side(self.ops)):
            for model in (ref, alt):
                model.step(0, _OFF)
            ref.step(1, _OFF)  # the perturbed op on main only
            with self.assertLogs(level="ERROR") as logs:
                with self.assertRaises(RuntimeError) as raised:
                    alt.step(1, _env(adamw=True, check=True))
        # The first split call is the fp16 (2 pieces) or the fp32 one (4).
        message = str(raised.exception)
        self.assertRegex(
            message,
            "^ONEFLOW_OPT_STREAMS_CHECK: rank 0: the split fused AdamW differs from "
            r"the unsplit call in [24] checked pieces \(kept the unsplit result\); "
            "ONEFLOW_OPT_ADAMW_STREAMS is not bitwise safe on this rank, stopping$",
        )
        self.assertEqual(logs.output, [f"ERROR:root:{message}"])
        self.assertTrue(clipping._ADAMW_STREAMS["off"])
        # The raising call left the stock result in its live tensors.
        group = slice(7, None) if "in 2 checked" in message else slice(7)
        for p, q in zip(ref.big[group], alt.big[group]):
            for x, y in [(p, q)] + [
                (ref.opt.state[p][k], alt.opt.state[q][k])
                for k in ("exp_avg", "exp_avg_sq")
            ]:
                self.assertTrue(torch.equal(_bits(x), _bits(y)))
        with self.assertLogs(level="WARNING") as logs:
            alt.step(2, _env(adamw=True))
        self.assertIn(
            "latched off after a ONEFLOW_OPT_STREAMS_CHECK mismatch",
            "\n".join(logs.output),
        )

    def test_check_catches_a_bug_at_a_slice_start(self) -> None:
        """A side-stream bug at the first element of a cut slice is caught
        at any tensor size: the chunk after the cut boundary is a piece."""
        original = torch._fused_adamw_

        def wrong_slice_start(*args: Any, **kwargs: Any) -> None:
            original(*args, **kwargs)
            if self.ops.cur.name != "main":
                for p in args[0]:
                    p.view(-1)[0] += 1.0

        params = _params([(5000,)], 7)
        opt = _clipped("d", params, warmup=False)
        _set_grads(params, 0)
        with mock.patch.dict(os.environ, _OFF):
            opt.step()
        _set_grads(params, 1)
        env = _env(adamw=True, check=True)
        with mock.patch.object(clipping, "_ADAMW_CHECK_WHOLE_MAX", 0):
            with mock.patch.object(torch, "_fused_adamw_", wrong_slice_start):
                with mock.patch.dict(os.environ, env):
                    with self.assertRaises(RuntimeError) as raised:
                        opt.step()
        self.assertIn("in 2 checked pieces", str(raised.exception))

    def test_deferred_mismatch_is_fatal_at_the_next_poll(self) -> None:
        clipping._ADAMW_STREAMS["checks"] = 0
        env = dict(_env(adamw=True, check=True), ONEFLOW_OPT_STREAMS_CHECK_EVERY="1")
        self.ops.ready = False
        with mock.patch.object(torch, "_fused_adamw_", _wrong_on_side(self.ops)):
            # Every call is a deferred check, so the live result stays stock.
            ref, alt = self.run_pair([env] * 2)
            self.assertEqual(len(keyed._OPT_STREAMS_PENDING), 4)
            self.ops.ready = True
            with self.assertLogs(level="ERROR") as logs:
                with self.assertRaises(RuntimeError) as raised:
                    alt.step(3, env)
        self.assertRegex(
            str(raised.exception),
            "^ONEFLOW_OPT_STREAMS_CHECK: rank 0: the split fused AdamW differs from "
            r"the unsplit call in [24] checked pieces \(kept the unsplit result\) "
            "in a deferred check; ONEFLOW_OPT_ADAMW_STREAMS is not bitwise safe on "
            "this rank, stopping$",
        )
        self.assertEqual(len(logs.output), 1)
        self.assertTrue(clipping._ADAMW_STREAMS["off"])

    def test_invalid_check_every_warns(self) -> None:
        env = dict(_env(adamw=True, check=True), ONEFLOW_OPT_STREAMS_CHECK_EVERY="x")
        with self.assertLogs(level="WARNING") as logs:
            self.run_pair([env] * 3)
        self.assertIn(
            "ONEFLOW_OPT_STREAMS_CHECK_EVERY='x' is not a non-negative integer; no "
            "periodic checks",
            "\n".join(logs.output),
        )
        self.assertEqual(clipping._ADAMW_STREAMS["checks"], 0)


def _half_env(half: bool, check: bool = False, flat: bool = True) -> Dict[str, str]:
    return dict(
        _env(check=check, flat=flat),
        ONEFLOW_CLIP_FLAT_HALF_COEF="1" if half else "0",
    )


def _half_grads(seed: int) -> List[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(n, generator=gen) * 3).to(dtype)
        for n, dtype in (
            (300, torch.float16),
            (17, torch.float16),
            (40, torch.float32),
            (65, torch.bfloat16),
        )
    ]


def _gpu_slow_path(g: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """What the CUDA per-grad mul_ computes: the fp32 coefficient loaded as
    the grad's dtype, the product in float."""
    if g.dtype == coef.dtype:
        return g * coef
    return (g.float() * coef.to(g.dtype).float()).to(g.dtype)


class TestHalfCoef(_Base):
    def setUp(self) -> None:
        super().setUp()
        patch = mock.patch.object(
            clipping,
            "_HALF_COEF",
            {"checks": 3, "off": False, "seen": False},
            create=True,
        )
        patch.start()
        self.addCleanup(patch.stop)

    def mul(self, half: bool, coef: float, check: bool = False) -> List[torch.Tensor]:
        grads = _half_grads(0)
        with mock.patch.dict(os.environ, _half_env(half, check)):
            clipping._clip_flat_mul_(grads, torch.tensor(coef))
        return grads

    def test_cpu_declines_and_keeps_results(self) -> None:
        ref = self.mul(False, 0.3)
        with self.assertLogs(level="WARNING") as logs:
            alt = self.mul(True, 0.3)
        self.assertIn(
            "ONEFLOW_CLIP_FLAT_HALF_COEF=1 declined for torch.float16 grads on "
            "cpu: only the CUDA slow path rounds the coefficient",
            "\n".join(logs.output),
        )
        for x, y in zip(ref, alt):
            self.assertTrue(torch.equal(_bits(x), _bits(y)))

    def test_engaged_uses_fast_path_with_rounded_coefficient(self) -> None:
        calls: List[Tuple[torch.dtype, torch.dtype]] = []
        original = torch._foreach_mul_

        def spy(tensors: List[torch.Tensor], scalar: Any) -> None:
            calls.append((tensors[0].dtype, scalar.dtype))
            original(tensors, scalar)

        expected = [_gpu_slow_path(g, torch.tensor(0.3)) for g in _half_grads(0)]
        with mock.patch.object(clipping, "_half_coef_decline", return_value=""):
            with mock.patch.object(torch, "_foreach_mul_", spy):
                with self.assertLogs(level="INFO") as logs:
                    got = self.mul(True, 0.3)
        self.assertEqual(
            sorted(calls, key=str),
            sorted(
                [
                    (torch.float16, torch.float16),
                    (torch.float32, torch.float32),
                    (torch.bfloat16, torch.bfloat16),
                ],
                key=str,
            ),
        )
        self.assertIn(
            "ONEFLOW_CLIP_FLAT_HALF_COEF: the clip multiply of 2 torch.float16 "
            "grads on cpu takes the foreach fast path with the coefficient cast "
            "to torch.float16",
            "\n".join(logs.output),
        )
        for x, y in zip(got, expected):
            self.assertTrue(torch.equal(_bits(x), _bits(y)))
        # fp32 grads keep the fp32 coefficient.
        self.assertTrue(torch.equal(_bits(got[2]), _bits(self.mul(False, 0.3)[2])))

    def test_check_passes_with_exact_coefficient(self) -> None:
        with mock.patch.object(clipping, "_half_coef_decline", return_value=""):
            with self.assertLogs(level="INFO") as logs:
                for _ in range(3):
                    got = self.mul(True, 0.5, check=True)
        self.assertIn(
            "ONEFLOW_OPT_STREAMS_CHECK: ONEFLOW_CLIP_FLAT_HALF_COEF matched the "
            "per-grad clip multiply bitwise in 3 calls",
            "\n".join(logs.output),
        )
        for x, y in zip(got, self.mul(False, 0.5)):
            self.assertTrue(torch.equal(_bits(x), _bits(y)))

    def test_check_mismatch_keeps_per_grad_result(self) -> None:
        # On CPU the per-grad path does not round the coefficient, so the
        # check sees the difference.
        with mock.patch.object(clipping, "_half_coef_decline", return_value=""):
            with self.assertLogs(level="ERROR") as logs:
                got = self.mul(True, 0.3, check=True)
        self.assertIn("the fast clip multiply differs", logs.output[0])
        self.assertTrue(clipping._HALF_COEF["off"])
        for x, y in zip(got, self.mul(False, 0.3)):
            self.assertTrue(torch.equal(_bits(x), _bits(y)))

    def test_no_op_reports(self) -> None:
        model = _Model(buckets=True)
        model.step(0, _OFF)
        with self.assertLogs(level="INFO") as logs:
            model.step(1, _half_env(True, flat=False))
        self.assertIn(
            "ONEFLOW_CLIP_FLAT_HALF_COEF=1 has no effect: it applies to the "
            "multiply of ONEFLOW_CLIP_FLAT, which is off",
            "\n".join(logs.output),
        )
        with mock.patch.object(clipping, "_CLIP_FLAT_MIN_SEGMENT", 64):
            with self.assertLogs(level="INFO") as logs:
                model.step(2, dict(_half_env(True), ONEFLOW_CLIP_FLAT_SEGMENTS="1"))
        text = "\n".join(logs.output)
        self.assertIn("an optimizer had no non-float32 grads", text)
        self.assertIn("declined for torch.float16 grads on cpu", text)

    def test_engaged_group_is_not_a_no_op(self) -> None:
        params = _params([(400,)], 0, torch.float16) + _params([(100,), (60,)], 1)
        opt = CombinedOptimizer([("default", _clipped("default", params))])
        for step in range(2):
            _set_bucket_grads(params, step)
            env = dict(_half_env(step > 0), ONEFLOW_CLIP_FLAT_SEGMENTS="1")
            with mock.patch.object(clipping, "_CLIP_FLAT_MIN_SEGMENT", 64):
                with mock.patch.dict(os.environ, env):
                    with mock.patch.object(
                        clipping, "_half_coef_decline", return_value=""
                    ):
                        with self.assertLogs(level="INFO") as logs:
                            opt.step()
        text = "\n".join(logs.output)
        self.assertIn("takes the foreach fast path", text)
        self.assertNotIn("nothing to cast", text)

    def test_flat_clipping_unchanged_on_cpu(self) -> None:
        with mock.patch.object(clipping, "_CLIP_FLAT_MIN_SEGMENT", 64):
            ref_env = _env(flat=True, segments=True)
            env = dict(ref_env, ONEFLOW_CLIP_FLAT_HALF_COEF="1")
            self.run_pair([env] * 3, ref_env=ref_env, buckets=True)


@unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
class TestHalfCoefOnGpu(unittest.TestCase):
    def test_bitwise_on_gpu(self) -> None:
        with mock.patch.object(
            clipping,
            "_HALF_COEF",
            {"checks": 3, "off": False, "seen": False},
            create=True,
        ):
            for seed in range(4):
                coef = torch.rand((), generator=torch.Generator().manual_seed(seed))
                ref = [g.cuda() for g in _half_grads(seed)]
                alt = [g.clone() for g in ref]
                with mock.patch.dict(os.environ, _half_env(False)):
                    clipping._clip_flat_mul_(ref, coef.cuda())
                with mock.patch.dict(os.environ, _half_env(True, check=seed < 3)):
                    clipping._clip_flat_mul_(alt, coef.cuda())
                for x, y, g in zip(ref, alt, _half_grads(seed)):
                    self.assertTrue(torch.equal(_bits(x), _bits(y)))
                    expected = _gpu_slow_path(g.cuda(), coef.cuda())
                    self.assertTrue(torch.equal(_bits(x), _bits(expected)))
            self.assertFalse(clipping._HALF_COEF["off"])

    def test_scalar_half_gradient_keeps_float_coefficient(self) -> None:
        coef = torch.tensor(0.3333, dtype=torch.float32, device="cuda")
        ref = [
            torch.tensor(0.3, dtype=torch.float16, device="cuda"),
            torch.full((2, 3), 0.3, dtype=torch.float16, device="cuda"),
        ]
        alt = [g.clone() for g in ref]
        torch._foreach_mul_(ref, coef)
        with mock.patch.dict(
            os.environ, _half_env(True, check=False)
        ), mock.patch.object(
            clipping, "_HALF_COEF", {"checks": 3, "off": False, "seen": False}
        ):
            clipping._clip_flat_foreach_mul_(alt, coef, coef.device, half_coef=True)
        for expected, actual in zip(ref, alt):
            self.assertTrue(torch.equal(_bits(expected), _bits(actual)))


@unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
class TestOnGpu(unittest.TestCase):
    def setUp(self) -> None:
        for patch in _fresh_state():
            patch.start()
            self.addCleanup(patch.stop)

    def test_real_streams_bitwise(self) -> None:
        def model() -> Tuple[CombinedOptimizer, List[torch.nn.Parameter]]:
            torch.manual_seed(0)
            heads = [
                torch.nn.Parameter(torch.randn(1000, device="cuda")) for _ in range(3)
            ]
            big = [
                torch.nn.Parameter(torch.randn(n, device="cuda"))
                for n in (300000, 7, 123456, 65536 * 5, 999)
            ]
            opt = CombinedOptimizer(
                [
                    ("h", _clipped("h", heads)),
                    ("d", _clipped("d", big)),
                ]
            )
            return opt, heads + big

        with mock.patch.object(
            keyed, "_OPT_STREAM_BIG_NUMEL", 1 << 18
        ), mock.patch.object(
            clipping, "_OPT_STREAM_BIG_NUMEL", 1 << 18
        ), mock.patch.object(
            clipping, "_ADAMW_SPLIT_MIN_NUMEL", 1 << 16
        ):
            (ref, rp), (alt, ap) = model(), model()
            for step in range(9):
                for p, q in zip(rp, ap):
                    gen = torch.Generator(device="cuda").manual_seed(step)
                    p.grad = torch.randn(p.shape, device="cuda", generator=gen)
                    q.grad = p.grad.clone()
                with mock.patch.dict(os.environ, _OFF):
                    ref.step()
                env = dict(
                    _env(adamw=True, small=False, check=True),
                    ONEFLOW_OPT_ADAMW_STREAMS_K=str(2 + step % 2),
                    ONEFLOW_OPT_STREAMS_CHECK_EVERY="2",
                )
                with mock.patch.dict(os.environ, env):
                    alt.step()
            torch.cuda.synchronize()
            keyed._opt_streams_poll()
            self.assertEqual(keyed._OPT_STREAMS_PENDING, [])
            self.assertEqual(clipping._ADAMW_STREAMS["checks"], 0)
            for p, q in zip(rp, ap):
                self.assertTrue(torch.equal(_bits(p), _bits(q)))


if __name__ == "__main__":
    unittest.main()
