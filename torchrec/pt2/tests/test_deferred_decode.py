#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from typing import Any, TYPE_CHECKING

import torch
from fbgemm_gpu.quantize_utils import fp32_to_mx4, mx4_to_float
from pyre_extensions import none_throws
from torch._dynamo.testing import CompileCounterWithBackend

# The decode ops are Triton kernels: CUDA only, and Triton is not built for
# every Python version.
_TESTS_SUPPORTED: bool = torch.cuda.is_available() and sys.version_info < (3, 15)

if TYPE_CHECKING or _TESTS_SUPPORTED:
    from torchrec.pt2 import deferred_decode as dd

GROUP = 32
DIM = 16


@torch.library.custom_op("deferred_decode_test::square", mutates_args=())
def _square(x: torch.Tensor) -> torch.Tensor:
    return x * x


@_square.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _square_setup(
    ctx: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> None:
    ctx.save_for_backward(inputs[0])


def _square_backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
    (x,) = ctx.saved_tensors
    return 2 * x * grad


# An opaque op that saves its input for backward, like the fused GIS + LN.
_square.register_autograd(_square_backward, setup_context=_square_setup)


class _A2A(torch.autograd.Function):
    """The all-to-all wait: placeholder out, gradient straight back."""

    @staticmethod
    def forward(ctx: Any, src: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=src.dtype, device=src.device).expand(src.shape)

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return grad


def _model(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    f0, f1 = dd.materialize_all([f0, f1])
    return _square(f0).sum() + _square(f1).sum()


def _dense_model(f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
    return _square(f0).sum() + _square(f1).sum()


@unittest.skipIf(not _TESTS_SUPPORTED, "needs CUDA and Triton")
class DeferredDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _decode(
        self,
        encoded: torch.Tensor,
        group: int,
        rows: list[int],
        outputs: list[list[tuple[int, int]]],
    ) -> list[torch.Tensor]:
        flat = torch.zeros((), device="cuda").expand(encoded.numel())
        return dd.materialize_all(dd.placeholders(encoded, flat, rows, outputs, group))

    def test_decode_bit_identical_to_fbgemm(self) -> None:
        values = torch.randn(32 * 1000, device="cuda") * 10
        values[:64] = 0  # an all-zero group: exponent 2^-127 and +0 codes
        values[100:200] *= 1e-38
        encoded = fp32_to_mx4(values, GROUP)
        ref = mx4_to_float(encoded, GROUP)
        # Output 0: one piece; output 1: two column pieces joined; output 2: empty.
        rows = [64, 37, 0]
        outputs = [[(0, 16)], [(31, 8), (32 * 7 + 3, 24)], [(20000, 16)]]
        out = self._decode(encoded, GROUP, rows, outputs)
        expected = [
            ref[:1024].view(64, 16),
            torch.cat(
                [
                    ref[31 : 31 + 37 * 8].view(37, 8),
                    ref[227 : 227 + 37 * 24].view(37, 24),
                ],
                dim=1,
            ),
            ref[:0].view(0, 16),
        ]
        for got, want in zip(out, expected):
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)))

    def test_decode_group_aligned(self) -> None:
        # Starts, widths and dims are multiples of the group size: the kernel
        # stores each group as one vector.
        group = 8
        values = torch.randn(group * 4000, device="cuda")
        values[:16] = 0
        encoded = fp32_to_mx4(values, group)
        ref = mx4_to_float(encoded, group)
        rows = [100, 75]
        outputs = [[(0, 16)], [(3200, 32), (1600, 16)]]
        out = self._decode(encoded, group, rows, outputs)
        expected = [
            ref[:1600].view(100, 16),
            torch.cat(
                [
                    ref[3200 : 3200 + 75 * 32].view(75, 32),
                    ref[1600 : 1600 + 1200].view(75, 16),
                ],
                dim=1,
            ),
        ]
        for got, want in zip(out, expected):
            self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)))

    def _layout(self, b0: int, b1: int) -> tuple[int, int, int, int]:
        # f0, then f1's two column shards, with gaps; not group-aligned.
        a = b0 * DIM + 3
        b = a + b1 * DIM + 5
        n = b + b1 * DIM
        return a, b, n, (n + GROUP - 1) // GROUP * GROUP

    def _dense(
        self, decoded: torch.Tensor, b0: int, b1: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, _, _ = self._layout(b0, b1)
        src = decoded.clone().requires_grad_()
        f0 = src[: b0 * DIM].view(b0, DIM)
        f1 = torch.cat(
            [src[a : a + b1 * DIM].view(b1, DIM), src[b : b + b1 * DIM].view(b1, DIM)],
            dim=1,
        )
        return src, f0, f1

    def _deferred(
        self, encoded: torch.Tensor, n_pad: int, b0: int, b1: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, _, _ = self._layout(b0, b1)
        src = torch.zeros(n_pad, device="cuda", requires_grad=True)
        flat = _A2A.apply(src)
        torch._dynamo.mark_dynamic(flat, 0)
        f0, f1 = dd.placeholders(
            encoded,
            flat,
            [b0, b1],
            [[(0, DIM)], [(a, DIM), (b, DIM)]],
            GROUP,
        )
        return src, f0, f1

    def _check_decode_matches_fbgemm(
        self,
        values: torch.Tensor,
        group: int,
        rows: list[int],
        outputs: list[list[tuple[int, int]]],
        dtype: torch.dtype,
    ) -> None:
        encoded = fp32_to_mx4(values, group)
        ref = mx4_to_float(encoded, group).to(dtype)
        flat = torch.zeros((), dtype=dtype, device="cuda").expand(ref.numel())
        out = dd.materialize_all(dd.placeholders(encoded, flat, rows, outputs, group))
        for r, ps, got in zip(rows, outputs, out):
            want = torch.cat(
                [ref[start : start + r * dim].view(r, dim) for start, dim in ps], dim=1
            )
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.dtype, dtype)
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            nan = want.isnan()
            # NaN payloads may differ; every NaN must land where fbgemm has one.
            self.assertTrue(torch.equal(got.isnan(), nan))
            self.assertTrue(
                torch.equal(got[~nan].view(bits), want[~nan].view(bits)),
                f"group {group} dtype {dtype} rows {r} pieces {ps}",
            )

    def _random_layout(
        self, gen: torch.Generator, n: int, num_outputs: int
    ) -> tuple[list[int], list[list[tuple[int, int]]]]:
        rows, outputs = [], []
        for _ in range(num_outputs):
            r = int(torch.randint(0, 40, (), generator=gen))
            pieces = []
            for _ in range(int(torch.randint(1, 4, (), generator=gen))):
                dim = int(torch.randint(1, 70, (), generator=gen))
                start = int(torch.randint(0, n - r * dim + 1, (), generator=gen))
                pieces.append((start, dim))
            rows.append(r)
            outputs.append(pieces)
        return rows, outputs

    def test_decode_matches_fbgemm_sweep(self) -> None:
        # Every group size, both output dtypes, value ranges from subnormal to
        # near the fp32 maximum, special values, and random piece layouts.
        gen = torch.Generator().manual_seed(0)
        f32 = torch.finfo(torch.float32)
        specials = torch.tensor(
            [
                0.0,
                -0.0,
                f32.max,
                -f32.max,
                f32.tiny,
                -f32.tiny,
                1e-45,
                -1e-45,
                float("inf"),
                float("-inf"),
                float("nan"),
                1.0,
                -1.0,
                6.0,
                0.5,
            ],
            device="cuda",
        )
        for group in (8, 16, 32):
            n = group * 3000
            wide = torch.randn(n, device="cuda") * torch.exp2(
                torch.randint(-140, 120, (n,), device="cuda").float()
            )
            mixed = torch.randn(n, device="cuda")
            idx = torch.randint(0, n, (n // 50,), device="cuda")
            mixed[idx] = specials[
                torch.randint(0, len(specials), idx.shape, device="cuda")
            ]
            cases = {
                "normal": torch.randn(n, device="cuda") * 10,
                "wide_range": wide,
                "specials": mixed,
                "constant_groups": torch.full((n,), 3.0, device="cuda"),
            }
            for name, values in cases.items():
                for dtype in (torch.float32, torch.bfloat16):
                    rows, outputs = self._random_layout(gen, n, 6)
                    with self.subTest(group=group, case=name, dtype=dtype):
                        self._check_decode_matches_fbgemm(
                            values, group, rows, outputs, dtype
                        )

    @unittest.skipIf(
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory < 40 * 2**30,
        "needs 40 GiB of GPU memory",
    )
    def test_decode_offsets_past_int32(self) -> None:
        # A decoded buffer of more than 2^31 values: pieces start past the
        # int32 range, as in the largest all-to-all payloads.
        group = 32
        n = (2**31 // group + 4096) * group
        values = torch.randn(n, device="cuda")
        start = n - 64 * 48
        self._check_decode_matches_fbgemm(
            values,
            group,
            [64, 3],
            [[(start, 48)], [(2**31 + 5, 7), (17, 9)]],
            torch.float32,
        )

    def test_compiled_grads_match_and_decode_not_saved(self) -> None:
        counter = CompileCounterWithBackend("inductor")
        model = torch.compile(_model, backend=counter)
        ref_model = torch.compile(_dense_model, backend="inductor")
        for b0, b1 in [(7, 5), (9, 6), (13, 11)]:
            _, _, _, n_pad = self._layout(b0, b1)
            encoded = fp32_to_mx4(torch.randn(n_pad, device="cuda"), GROUP)
            decoded = mx4_to_float(encoded, GROUP)

            src, f0, f1 = self._dense(decoded, b0, b1)
            torch._dynamo.mark_dynamic(f0, 0)
            torch._dynamo.mark_dynamic(f1, 0)
            ref_loss = ref_model(f0, f1)
            ref_loss.backward()
            ref_grad = src.grad

            src, f0, f1 = self._deferred(encoded, n_pad, b0, b1)
            torch._dynamo.mark_dynamic(f0, 0)
            torch._dynamo.mark_dynamic(f1, 0)
            torch._dynamo.mark_dynamic(encoded, 0)
            saved: list[tuple[tuple[int, ...], torch.dtype, int]] = []

            def pack(
                t: torch.Tensor,
                saved: list[tuple[tuple[int, ...], torch.dtype, int]] = saved,
            ) -> torch.Tensor:
                saved.append((tuple(t.shape), t.dtype, t.untyped_storage().nbytes()))
                return t

            with torch.profiler.profile() as prof:
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    loss = model(f0, f1)
                loss.backward()
            # One decode of the shared payload in forward, one in backward, and
            # one gradient scatter.
            compiled = [
                e.name
                for e in prof.events()
                if e.name.startswith("torchrec::mx4_decode")
                and e.cpu_parent is not None
                and e.cpu_parent.name.startswith("## Call CompiledFxGraph")
            ]
            self.assertEqual(
                sorted(compiled),
                ["torchrec::mx4_decode"] * 2 + ["torchrec::mx4_decode_backward"],
            )

            self.assertTrue(
                torch.equal(loss.view(torch.int32), ref_loss.view(torch.int32))
            )
            self.assertTrue(
                torch.equal(
                    none_throws(src.grad).view(torch.int32),
                    none_throws(ref_grad).view(torch.int32),
                )
            )
            # Only the payload, the piece table and zero-memory placeholders
            # are kept.
            floats = [s for s in saved if s[1].is_floating_point and s[2] > 4]
            self.assertEqual(floats, [], saved)
        self.assertEqual(counter.frame_count, 1)

    def test_unused_placeholder_gets_zero_gradient(self) -> None:
        b0, b1 = 7, 5
        _, _, _, n_pad = self._layout(b0, b1)
        encoded = fp32_to_mx4(torch.randn(n_pad, device="cuda"), GROUP)
        decoded = mx4_to_float(encoded, GROUP)
        src, f0, f1 = self._deferred(encoded, n_pad, b0, b1)
        (f1,) = dd.materialize_all([f1])
        (f1 * f1).sum().backward()
        ref_src, _, ref_f1 = self._dense(decoded, b0, b1)
        (ref_f1 * ref_f1).sum().backward()
        self.assertTrue(torch.equal(none_throws(src.grad), none_throws(ref_src.grad)))

    def _deferred_layout(
        self,
        encoded: torch.Tensor,
        group: int,
        calls: list[tuple[list[int], list[list[tuple[int, int]]]]],
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        n = mx4_to_float(encoded, group).numel()
        src = torch.zeros(n, device="cuda", requires_grad=True)
        flat = _A2A.apply(src)
        return src, [
            dd.placeholders(encoded, flat, rows, outputs, group)
            for rows, outputs in calls
        ]

    def _dense_layout(
        self,
        decoded: torch.Tensor,
        calls: list[tuple[list[int], list[list[tuple[int, int]]]]],
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        src = decoded.clone().requires_grad_()
        return src, [
            [
                torch.cat(
                    [src[s : s + r * d].view(r, d) for s, d in ps]
                    or [src.new_empty(r, 0)],
                    dim=1,
                )
                for r, ps in zip(rows, outputs)
            ]
            for rows, outputs in calls
        ]

    def _check_layout(
        self,
        calls: list[tuple[list[int], list[list[tuple[int, int]]]]],
        order: list[tuple[int, int]],
        group: int = GROUP,
    ) -> None:
        """Decodes placeholders `order` (call, output) through `materialize_all`
        and checks the values and the flat gradient against slicing the decoded
        buffer, bit for bit."""
        encoded = fp32_to_mx4(torch.randn(group * 200, device="cuda") * 4, group)
        decoded = mx4_to_float(encoded, group)
        src, groups = self._deferred_layout(encoded, group, calls)
        ref_src, ref_groups = self._dense_layout(decoded, calls)
        got = dd.materialize_all([groups[c][o] for c, o in order])
        want = [ref_groups[c][o] for c, o in order]
        weights = [torch.randn(w.shape, device="cuda") for w in want]
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertTrue(torch.equal(g.view(torch.int32), w.view(torch.int32)))
        torch.stack([(g * k).sum() for g, k in zip(got, weights)]).sum().backward()
        torch.stack([(w * k).sum() for w, k in zip(want, weights)]).sum().backward()
        self.assertTrue(
            torch.equal(
                none_throws(src.grad).view(torch.int32),
                none_throws(ref_src.grad).view(torch.int32),
            )
        )

    def test_repeated_placeholder(self) -> None:
        # The same placeholder twice: decoded once, gradients summed.
        self._check_layout(
            [([10, 7], [[(0, 16)], [(200, 9)]])], [(0, 0), (0, 1), (0, 0)]
        )

    def test_two_layouts_on_one_payload(self) -> None:
        self._check_layout([([4], [[(0, 16)]]), ([3], [[(1000, 5)]])], [(0, 0), (1, 0)])

    def test_output_without_pieces(self) -> None:
        self._check_layout([([3, 2], [[(0, 4)], []])], [(0, 0), (0, 1)])

    def test_overlapping_pieces(self) -> None:
        # Overlapping reads get summed gradients (at most two per value, so the
        # sum does not depend on order); values no piece reads get zero.
        self._check_layout(
            [([1, 1, 2], [[(0, 1600)], [(0, 1600)], [(3000, 8), (3004, 8)]])],
            [(0, 0), (0, 1), (0, 2)],
        )

    def test_signed_byte_payload(self) -> None:
        group = 32
        encoded = fp32_to_mx4(torch.randn(group * 64, device="cuda") * 1e3, group)
        flat = torch.zeros((), device="cuda").expand(group * 64)
        (want,) = dd.materialize_all(
            dd.placeholders(encoded, flat, [64], [[(0, group)]], group)
        )
        (got,) = dd.materialize_all(
            dd.placeholders(encoded.view(torch.int8), flat, [64], [[(0, group)]], group)
        )
        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)))

    @unittest.skipIf(
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory < 70 * 2**30,
        "needs 70 GiB of GPU memory",
    )
    def test_output_rows_times_width_past_int32(self) -> None:
        # One output of two 512-wide pieces with rows * width past 2^31: row
        # addresses need 64-bit arithmetic in both kernels.
        group = 32
        rows = 2**31 // 1024 + 4096
        second = rows * 512 + 4096 * group
        n = second + rows * 512
        encoded = fp32_to_mx4(torch.randn(n, device="cuda"), group)
        flat = torch.zeros((), device="cuda").expand(n)
        (out,) = dd.materialize_all(
            dd.placeholders(encoded, flat, [rows], [[(0, 512), (second, 512)]], group)
        )
        ref = mx4_to_float(encoded, group)
        del encoded
        for c, s in ((0, 0), (512, second)):
            self.assertTrue(
                torch.equal(
                    out[:, c : c + 512].reshape(-1).view(torch.int32),
                    ref[s : s + rows * 512].view(torch.int32),
                )
            )
        del ref
        table = torch.tensor([[0, 0, 512], [0, second, 512]])
        grad = dd.mx4_decode_backward([out], table, [0], n, out.dtype, out.device)
        for c, s in ((0, 0), (512, second)):
            self.assertTrue(
                torch.equal(
                    grad[s : s + rows * 512].view(torch.int32),
                    out[:, c : c + 512].reshape(-1).view(torch.int32),
                )
            )
        self.assertEqual(int(torch.count_nonzero(grad[rows * 512 : second])), 0)
