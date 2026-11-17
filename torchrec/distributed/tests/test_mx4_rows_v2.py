#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""CPU tests of TORCHREC_SEQ_A2A_MX4_V2: the row-tiled MX4 kernels of
mx4_rows_v2 against fbgemm's and torchrec's row-map kernels under the Triton
interpreter, and their routing in comm_ops (scopes, declines, self-check and
fallbacks) with stand-in kernels."""

import contextlib
import dataclasses
import importlib.util
import itertools
import logging
import multiprocessing
import os
import unittest
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import fbgemm_gpu
import torch
import torchrec.distributed.comm_ops as comm_ops
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)
from torchrec.distributed.types import QuantizedCommCodecs

# Exit code of the Triton interpreter checks when the interpreter is missing.
_TRITON_INTERPRETER_UNAVAILABLE = 77
_HAS_TRITON: bool = importlib.util.find_spec("triton") is not None
# The encode interpreter checks compare bit for bit with fbgemm's MX4 kernels;
# released fbgemm_gpu builds can lag the in-repo kernels.
_IN_REPO_FBGEMM: bool = not getattr(fbgemm_gpu, "open_source", False)
_PREFIX = "TORCHREC_SEQ_A2A_MX4_V2"
_SEQ_A2A_DEFAULT_ON_FLAGS: Tuple[str, ...] = (
    _PREFIX,
    "TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD",
    "TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD",
    "TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD",
    "TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE",
    "TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP",
    "TORCHREC_SEQ_UNBUCKETIZE_COMPOSE",
    "TORCHREC_UNBUCKETIZE_GATHER_BWD",
)
D = 512
# Rows of the forward inputs; not _MX4V2_EDGE_ROWS, which marks edge inputs
# for the stand-in kernels.
_FWD_ROWS = 23


def _codecs() -> QuantizedCommCodecs:
    # The job's qcomms: MX4 in groups of 8 forward and of 256 backward.
    return get_qcomm_codecs(
        QCommsConfig(
            forward_precision=CommType.MX4,
            backward_precision=CommType.MX4,
            mx4_quantize_dim=8,
            mx4_quantize_dim_bwd=256,
        )
    )


def _reset() -> None:
    for scope in comm_ops._MX4V2_SCOPES:
        comm_ops._mx4v2_state[scope] = comm_ops._Mx4V2State()
    comm_ops._mx4v2_warned.clear()
    comm_ops._mx4v2_edge_cache.clear()


@contextlib.contextmanager
def _env(flags: Dict[str, str]) -> Iterator[None]:
    """Sets TORCHREC_SEQ_A2A_MX4_V2 ("" key) and its _CHECK / _DIV / _CFG
    flags (by suffix), with every other one unset. The default-on sequence
    all2all paths, MX4_V2 included unless given, are pinned off."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(_PREFIX)}
    env.update(dict.fromkeys(_SEQ_A2A_DEFAULT_ON_FLAGS, "0"))
    env.update({_PREFIX + (f"_{k}" if k else ""): v for k, v in flags.items()})
    with mock.patch.dict(os.environ, env, clear=True):
        yield


@contextlib.contextmanager
def _gpu_branch(hip: Optional[str] = "7.0") -> Iterator[None]:
    """Every tensor reports is_cuda and torch reports a ROCm build (`hip`),
    so the routing takes its GPU branches on CPU tensors. fbgemm imports its
    Triton MX4 entry points only when a GPU is present; under the interpreter
    they run on CPU, so they are provided here."""
    from fbgemm_gpu import quantize_utils
    from fbgemm_gpu.triton import quantize_mx4
    from fbgemm_gpu.triton.quantize import triton_dequantize_mx4

    with mock.patch.object(
        quantize_utils, "quantize_mx4", quantize_mx4, create=True
    ), mock.patch.object(
        quantize_utils, "triton_dequantize_mx4", triton_dequantize_mx4, create=True
    ), mock.patch.object(
        torch.Tensor, "is_cuda", new_callable=mock.PropertyMock
    ) as is_cuda, mock.patch.object(
        torch.version, "hip", hip
    ), mock.patch.object(
        torch.cuda, "is_current_stream_capturing", lambda: False
    ):
        is_cuda.return_value = True
        yield


def _tl_bool_op(op: str, a: object, b: object) -> object:
    """`a and b` / `a or b` as compiled Triton evaluates them: elementwise on
    tensors, Python's on constexprs."""
    import triton.language as tl

    if isinstance(a, tl.constexpr):
        a = a.value
    if isinstance(b, tl.constexpr):
        b = b.value
    stop = (lambda v: not v) if op == "and" else bool
    for v in (a, b):
        if not isinstance(v, tl.tensor) and stop(v):
            return v
    tensors = [v for v in (a, b) if isinstance(v, tl.tensor)]
    if len(tensors) < 2:
        return tensors[0] if tensors else b
    if op == "and":
        return (tensors[0] != 0) & (tensors[1] != 0)
    return (tensors[0] != 0) | (tensors[1] != 0)


def _tl_not(a: object) -> object:
    import triton.language as tl

    if isinstance(a, tl.constexpr):
        a = a.value
    return a == 0 if isinstance(a, tl.tensor) else not a


@contextlib.contextmanager
def _gfx950_semantics(shifts: bool = True, bool_ops: bool = True) -> Iterator[None]:
    """Makes the Triton interpreter run kernels the way they run compiled for
    gfx950, where the two differ for fbgemm's MX4 quantize kernel:

    - shifts: numpy gives 0 (or -1) for shift amounts of the width or more;
      the gfx950 shift instructions use the amount's low log2(width) bits;
    - bool_ops: `and` / `or` / `not` on tensors are Python's in the
      interpreter (a block tensor is always true), and elementwise when
      compiled.

    The row-tiled kernels need neither: they mask their shift amount and use
    & and ~."""
    import ast

    from triton.runtime import interpreter

    # Interpreter internals; Triton releases without them cannot run these checks.
    if not (
        all(
            hasattr(getattr(interpreter, "InterpreterBuilder", None), op)
            for op in ("create_shl", "create_lshr", "create_ashr")
        )
        and hasattr(getattr(interpreter, "FunctionRewriter", None), "ast_transformer")
        and hasattr(getattr(interpreter, "InterpretedFunction", None), "rewritten_fn")
    ):
        raise SystemExit(_TRITON_INTERPRETER_UNAVAILABLE)
    builder = interpreter.InterpreterBuilder
    saved = (builder.create_shl, builder.create_lshr, builder.create_ashr)
    saved_transformer = interpreter.FunctionRewriter.ast_transformer

    def masked(op):
        def shift(self, lhs, rhs):
            width = lhs.data.dtype.itemsize * 8
            data = (rhs.data & (width - 1)).astype(rhs.data.dtype)
            return op(self, lhs, interpreter.TensorHandle(data, rhs.dtype))

        return shift

    class BoolOps(type(saved_transformer)):
        def visit_Assign(self, node):
            node.value = self.visit(node.value)
            return super().visit_Assign(node)

        def visit_BoolOp(self, node):
            self.generic_visit(node)
            op = ast.Constant("and" if isinstance(node.op, ast.And) else "or")
            expr = node.values[0]
            for value in node.values[1:]:
                expr = ast.Call(
                    ast.Name("_tl_bool_op", ast.Load()), [op, expr, value], []
                )
            return expr

        def visit_UnaryOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.Not):
                return ast.Call(ast.Name("_tl_not", ast.Load()), [node.operand], [])
            return node

    if shifts:
        builder.create_shl, builder.create_lshr, builder.create_ashr = (
            masked(op) for op in saved
        )
    if bool_ops:
        # pyrefly: ignore [missing-attribute]
        interpreter._tl_bool_op = _tl_bool_op
        # pyrefly: ignore [missing-attribute]
        interpreter._tl_not = _tl_not
        interpreter.FunctionRewriter.ast_transformer = BoolOps()
    interpreter.InterpretedFunction.rewritten_fn.clear()
    try:
        yield
    finally:
        builder.create_shl, builder.create_lshr, builder.create_ashr = saved
        interpreter.FunctionRewriter.ast_transformer = saved_transformer
        interpreter.InterpretedFunction.rewritten_fn.clear()


@contextlib.contextmanager
def _no_mx4v2_logs(test: unittest.TestCase) -> Iterator[None]:
    """Asserts that nothing logs about TORCHREC_SEQ_A2A_MX4_V2."""
    with test.assertLogs(comm_ops.logger, "DEBUG") as logs:
        yield
        comm_ops.logger.info("end")
    assert not [o for o in logs.output if _PREFIX in o], logs.output


def _a2a_world1(x: torch.Tensor, codecs: QuantizedCommCodecs) -> torch.Tensor:
    """All2All_Seq_Req and its wait on the default (world size 1) group, for
    T=2 features of B=2 rows (alltoall_sequence skips a group of one)."""
    import torch.distributed as dist

    pg = dist.GroupMember.WORLD
    a2ai = comm_ops.All2AllSequenceInfo(
        embedding_dim=x.shape[1],
        lengths_after_sparse_data_all2all=torch.IntTensor([[2, 3], [1, 2]]),
        forward_recat_tensor=torch.IntTensor([0, 1]),
        backward_recat_tensor=torch.IntTensor([0, 1]),
        input_splits=[x.shape[0]],
        output_splits=[x.shape[0]],
        codecs=codecs,
    )
    # pyrefly: ignore [bad-argument-type]
    myreq = comm_ops.Request(pg, device=x.device)
    comm_ops.All2All_Seq_Req.apply(pg, myreq, a2ai, x)
    return myreq.wait()


@contextlib.contextmanager
def _world1() -> Iterator[None]:
    import torch.distributed as dist
    from torchrec.test_utils import get_free_port

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{get_free_port()}",
        rank=0,
        world_size=1,
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("TORCHREC_")}
    try:
        with mock.patch.dict(os.environ, env, clear=True):
            yield
    finally:
        dist.destroy_process_group()


def _interpreter_or_exit() -> None:
    try:
        import numpy
        from triton.runtime.interpreter import InterpretedFunction  # noqa: F401

        # The interpreter takes runtime loop bounds with int() of a
        # one-element array, which newer numpy rejects.
        int(numpy.ones(1))
    except (ImportError, TypeError, DeprecationWarning):
        raise SystemExit(_TRITON_INTERPRETER_UNAVAILABLE)


def _interpret_like_compiled(target: Callable[[], None]) -> None:
    """Runs target with the interpreter's integer-overflow sanitizer off:
    compiled kernels only sanitize in debug mode, and fbgemm's MX4 rounding
    adds to the int32 bits of NaN payloads, which wraps when compiled."""
    _interpreter_or_exit()
    from triton.runtime import interpreter

    builder = interpreter.interpreter_builder
    if hasattr(builder.options, "sanitize_overflow"):
        builder.options = dataclasses.replace(builder.options, sanitize_overflow=False)
    target()


def _run_interpreted(test: unittest.TestCase, target: object) -> None:
    # TRITON_INTERPRET must be set before the kernels are defined, so the
    # check runs in a fresh process.
    ctx = multiprocessing.get_context("spawn")
    with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
        # pyrefly: ignore [bad-argument-type]
        p = ctx.Process(target=_interpret_like_compiled, args=(target,))
        p.start()
    p.join()
    if p.exitcode == _TRITON_INTERPRETER_UNAVAILABLE:
        test.skipTest("Needs the Triton interpreter (triton and numpy)")
    test.assertEqual(0, p.exitcode)


def _ref_encode(
    a: torch.Tensor, row_idx: Optional[torch.Tensor], group_size: int
) -> torch.Tensor:
    """A CPU stand-in for an old encode kernel: fbgemm's reference MX4 encode
    of the rows."""
    from fbgemm_gpu.triton.quantize_ref import py_quantize_mx4

    rows = a.view(-1, a.shape[-1])
    rows = rows if row_idx is None else rows[row_idx]
    return py_quantize_mx4(rows.float().contiguous(), group_size, rounding_mode=2).view(
        -1
    )


def _rowmap_info(
    codecs: QuantizedCommCodecs, dim: int = D
) -> Tuple[comm_ops.All2AllSequenceInfo, torch.Tensor, int]:
    """An All2AllSequenceInfo for _seq_a2a_fwd_quantize_rowmap of _FWD_ROWS
    rows, the input's lengths in elements and the world size."""
    T, W, B = 2, 2, 3
    gen = torch.Generator().manual_seed(0)
    lengths = torch.randint(0, 4, (T, W * B), generator=gen).int()
    lengths[0, 0] += _FWD_ROWS - int(lengths.sum())
    recat = torch.IntTensor([j + i * W for j in range(W) for i in range(T)])
    a2ai = comm_ops.All2AllSequenceInfo(
        embedding_dim=dim,
        lengths_after_sparse_data_all2all=lengths,
        forward_recat_tensor=recat,
        # pyrefly: ignore [bad-argument-type]
        backward_recat_tensor=None,
        input_splits=[],
        output_splits=[],
        codecs=codecs,
    )
    return a2ai, lengths * dim, W


def _snan_row(dim: int, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    """A row with signaling NaNs of both signs among ordinary values, for the
    interpreter checks only (on GPU their group max follows the compiler's
    max order, see mx4_rows_v2)."""
    row = torch.randn(1, dim, generator=gen).to(dtype)
    bits = [0x7F81, -127] if dtype == torch.bfloat16 else [0x7F800001, -6291456]
    itype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    row[0, ::4] = torch.tensor(bits, dtype=itype).view(dtype).repeat(dim // 8)
    return row


def _rows_info(codecs: QuantizedCommCodecs) -> comm_ops.All2AllSequenceInfo:
    return comm_ops.All2AllSequenceInfo(
        embedding_dim=D,
        lengths_after_sparse_data_all2all=torch.zeros(1, 1, dtype=torch.int),
        forward_recat_tensor=None,
        # pyrefly: ignore [bad-argument-type]
        backward_recat_tensor=None,
        input_splits=[],
        output_splits=[],
        codecs=codecs,
    )


class _Kernels:
    """Stand-ins for the old kernels and the row-tiled ones, patched where
    comm_ops looks them up. The row-tiled stand-in returns the old kernel's
    bytes unless `bad` says otherwise for its mul_rcp argument."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[object, ...]]] = []
        self.bad: Dict[bool, bool] = {True: False, False: False}
        self.bad_edge = False
        # Whether `bad` also corrupts the edge runs.
        self.bad_on_edge = True
        self.fail = False
        self.fail_edge = False

    def names(self) -> List[str]:
        return [c[0] for c in self.calls]

    def _corrupt(self, out: torch.Tensor, mul_rcp: bool, edge: bool) -> torch.Tensor:
        if (self.bad[mul_rcp] and (self.bad_on_edge or not edge)) or (
            edge and self.bad_edge
        ):
            out = out.clone()
            out.view(torch.uint8).view(-1)[0] ^= 1
        return out

    def rowmap(self, a, row_idx, D, group_size, rounding):
        self.calls.append(("rowmap", (D, group_size, rounding)))
        return _ref_encode(a, row_idx, group_size)

    def chunked(self, a, group_size, rounding_mode, rows_per_chunk, gpp):
        self.calls.append(("chunked", (group_size, rounding_mode, rows_per_chunk, gpp)))
        return _ref_encode(a, None, group_size)

    def encode_rows(
        self, a, row_idx, D, group_size, rounding, block_r, num_warps, mul_rcp
    ):
        self.calls.append(
            (
                "v2_encode",
                (
                    row_idx is not None,
                    D,
                    group_size,
                    rounding,
                    block_r,
                    num_warps,
                    mul_rcp,
                ),
            )
        )
        edge = a.shape[0] == comm_ops._MX4V2_EDGE_ROWS
        if self.fail or (edge and self.fail_edge):
            raise RuntimeError("compile failure")
        return self._corrupt(_ref_encode(a, row_idx, group_size), mul_rcp, edge)

    @contextlib.contextmanager
    def patched(self) -> Iterator["_Kernels"]:
        from torchrec.distributed import mx4_rowmap_quantize as mx4q, mx4_rows_v2

        with mock.patch.object(
            mx4q, "quantize_mx4_rowmap", self.rowmap
        ), mock.patch.object(
            mx4q, "quantize_mx4_rows_chunked", self.chunked
        ), mock.patch.object(
            mx4_rows_v2, "mx4_encode_rows", self.encode_rows
        ):
            yield self


class MX4RowsV2RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def test_requested_scopes(self) -> None:
        req = comm_ops._mx4v2_requested
        with _env({}):
            assert not any(req(s) for s in comm_ops._MX4V2_SCOPES)
        for flag, want in (
            ("0", ()),
            ("", ()),
            ("1", comm_ops._MX4V2_SCOPES),
            ("fwd_enc", ("fwd_enc",)),
            (" bwd_enc , fwd_enc", ("fwd_enc", "bwd_enc")),
        ):
            with _env({"": flag}):
                got = tuple(s for s in comm_ops._MX4V2_SCOPES if req(s))
            assert got == tuple(s for s in comm_ops._MX4V2_SCOPES if s in want), flag
        # Unknown scopes warn once and are ignored.
        with _env({"": "bwd_enc,bogus"}), self.assertLogs(
            comm_ops.logger, "WARNING"
        ) as logs:
            assert req("bwd_enc") and not req("fwd_enc") and req("bwd_enc")
        assert len(logs.records) == 1 and "bogus" in logs.output[0]

    def test_cfg(self) -> None:
        cfg = comm_ops._mx4v2_cfg
        for spec, scope, want in (
            ("", "fwd_enc", (4, 4)),
            ("8:4", "bwd_enc", (8, 4)),
            ("fwd_enc=2:2,bwd_enc=16:8", "bwd_enc", (16, 8)),
            ("fwd_enc=2:2,bwd_enc=16:8", "fwd_enc", (2, 2)),
            ("fwd_enc=2:2", "bwd_enc", (4, 4)),
        ):
            with _env({"CFG": spec}):
                assert cfg(scope) == want, (spec, scope)
        for spec in ("3:4", "4:16", "4", "a:b", "bwd_enc=4:3"):
            with _env({"CFG": spec}), self.assertLogs(comm_ops.logger, "WARNING"):
                assert cfg("bwd_enc") == (4, 4), spec

    def _fwd(
        self, x: torch.Tensor, codecs: QuantizedCommCodecs, hip: Optional[str] = "7.0"
    ) -> torch.Tensor:
        a2ai, lengths, W = _rowmap_info(codecs, x.shape[1])
        with _gpu_branch(hip):
            ret = comm_ops._seq_a2a_fwd_quantize_rowmap(a2ai, x, lengths, W)
        assert ret is not None
        return ret[1]

    def _bwd(self, grad: torch.Tensor, codecs: QuantizedCommCodecs) -> torch.Tensor:
        ctx = codecs.backward.create_context()
        with _gpu_branch():
            out = comm_ops._seq_a2a_bwd_chunked_encode(_rows_info(codecs), grad, ctx)
        assert out is not None
        return out

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_fwd_enc_routing(self) -> None:
        codecs = _codecs()
        x = torch.randn(_FWD_ROWS, D, generator=torch.Generator().manual_seed(1))
        k = _Kernels()
        with k.patched():
            # Off, or another scope only: the row-map kernel, and no logs.
            for flags in ({}, {"": "0"}, {"": "bwd_enc", "CHECK": "1"}):
                with _env(flags), _no_mx4v2_logs(self):
                    ref = self._fwd(x, codecs)
                assert k.names() == ["rowmap"], flags
                assert k.calls[0][1] == (D, 8, 2)
                k.calls.clear()
            # On: the row-tiled kernel alone, with the job's arguments, and one
            # info line per config.
            for flags in ({"": "1"}, {"": "fwd_enc"}):
                _reset()
                with _env(flags), self.assertLogs(comm_ops.logger, "INFO") as logs:
                    assert torch.equal(self._fwd(x, codecs), ref)
                    assert torch.equal(self._fwd(x, codecs), ref)
                assert k.names() == ["v2_encode"] * 2
                assert k.calls[0][1] == (True, D, 8, 2, 4, 4, True)
                assert len(logs.records) == 1, logs.output
                assert "TORCHREC_SEQ_A2A_MX4_V2 fwd_enc on" in logs.output[0]
                k.calls.clear()
            # The flags are read per call.
            for flags, want in (
                ({}, "rowmap"),
                ({"": "fwd_enc", "CFG": "fwd_enc=8:2", "DIV": "1"}, "v2_encode"),
                ({"": "0"}, "rowmap"),
                ({"": "fwd_enc"}, "v2_encode"),
            ):
                with _env(flags):
                    assert torch.equal(self._fwd(x, codecs), ref)
                assert k.names() == [want], flags
                k.calls.clear()
            with _env({"": "fwd_enc", "CFG": "8:2", "DIV": "1"}):
                self._fwd(x, codecs)
            assert k.calls[0][1] == (True, D, 8, 2, 8, 2, False)

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_bwd_enc_routing(self) -> None:
        codecs = _codecs()
        grad = torch.randn(37, D, generator=torch.Generator().manual_seed(2))
        k = _Kernels()
        with k.patched():
            with _env({}), _no_mx4v2_logs(self):
                ref = self._bwd(grad, codecs)
            assert k.names() == ["chunked"]
            k.calls.clear()
            with _env({"": "bwd_enc"}):
                assert torch.equal(self._bwd(grad, codecs), ref)
                # A flat gradient goes through the same [rows, D] encode.
                assert torch.equal(self._bwd(grad.view(-1), codecs), ref)
            # One launch over all the rows.
            assert k.names() == ["v2_encode"] * 2
            assert k.calls[0][1] == (False, D, 256, 2, 4, 4, True)
            k.calls.clear()
            # RoundingMode.even for a context without a rounding mode, as in
            # fp32_to_mx4.
            ctx = codecs.backward.create_context()
            # pyrefly: ignore [missing-attribute]
            ctx.rounding_mode = None
            with _env({"": "1"}), _gpu_branch():
                out = comm_ops._seq_a2a_bwd_chunked_encode(
                    _rows_info(codecs), grad, ctx
                )
            assert out is not None and torch.equal(out, ref)
            assert k.calls[0][1] == (False, D, 256, 2, 4, 4, True)

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_declines(self) -> None:
        codecs = _codecs()
        x = torch.randn(_FWD_ROWS, D, generator=torch.Generator().manual_seed(3))
        k = _Kernels()
        with k.patched(), _env({"": "1"}):
            # Each decline takes the old kernel and warns once per reason.
            for hip, mode, reason in (
                (None, contextlib.nullcontext(), "not a ROCm build"),
                ("7.0", torch.inference_mode(), "inference mode"),
            ):
                with self.assertLogs(comm_ops.logger, "WARNING") as logs, mode:
                    for _ in range(2):
                        self._fwd(x, codecs, hip=hip)
                assert k.names() == ["rowmap"] * 2, reason
                assert len(logs.records) == 1 and reason in logs.output[0]
                k.calls.clear()
            for patch in (
                mock.patch.object(comm_ops, "is_torchdynamo_compiling", lambda: True),
                mock.patch.object(torch.jit, "is_scripting", lambda: True),
            ):
                comm_ops._mx4v2_warned.clear()
                with patch, self.assertLogs(comm_ops.logger, "WARNING") as logs:
                    assert not comm_ops._mx4v2_active("bwd_enc", D, 256)
                assert "compiling or scripting" in logs.output[0]
            # Shapes the kernels do not take: 768 is not a power of two.
            with self.assertLogs(comm_ops.logger, "WARNING") as logs:
                for _ in range(2):
                    self._fwd(torch.randn(_FWD_ROWS, 768), codecs)
            assert k.names() == ["rowmap"] * 2
            assert len(logs.records) == 1
            assert "D=768 mx_group_size=8 is not a supported shape" in logs.output[0]
            with mock.patch.object(torch.version, "hip", "7.0"):
                assert comm_ops._mx4v2_active("bwd_enc", D, 256)
                assert not comm_ops._mx4v2_active("bwd_enc", 4096, 256)
                assert not comm_ops._mx4v2_active("bwd_enc", D, 4)

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_self_check(self) -> None:
        codecs = _codecs()
        x = torch.randn(_FWD_ROWS, D, generator=torch.Generator().manual_seed(4))
        state = comm_ops._mx4v2_state["fwd_enc"]
        k = _Kernels()
        with k.patched(), _env({"": "fwd_enc", "CHECK": "1"}):
            with self.assertLogs(comm_ops.logger, "WARNING") as logs:
                ref = self._fwd(x, codecs)
                for _ in range(4):
                    assert torch.equal(self._fwd(x, codecs), ref)
            # Three checked calls, each also running the old kernel, and one
            # edge run of both kernels.
            assert state.checks == 3 and state.mismatches == 0 and not state.off
            assert k.names().count("rowmap") == 3 + 1
            assert k.names().count("v2_encode") == 5 + 1
            checks = [o for o in logs.output if "self-check" in o]
            assert len(checks) == 3 and all("ok=True edge_ok=True" in o for o in checks)
            assert "self-check 1/3" in checks[0] and "self-check 3/3" in checks[2]
        # A mismatch returns the old kernel's bytes; an encode then divides by
        # the scale and checks from scratch, and a second mismatch turns the
        # scope off.
        for bad_rcp, bad_div, bad_edge, on_edge, want_off in (
            (True, False, False, True, False),
            (True, False, False, False, False),
            (True, True, False, True, True),
            (False, False, True, True, True),
        ):
            _reset()
            state = comm_ops._mx4v2_state["fwd_enc"]
            k = _Kernels()
            k.bad = {True: bad_rcp, False: bad_div}
            k.bad_edge = bad_edge
            k.bad_on_edge = on_edge
            with k.patched(), _env({"": "fwd_enc", "CHECK": "1"}):
                with self.assertLogs(comm_ops.logger, "WARNING") as logs:
                    for _ in range(5):
                        assert torch.equal(self._fwd(x, codecs), ref)
            case = (bad_rcp, bad_div, bad_edge, on_edge)
            assert any("self-check FAILED" in o for o in logs.output), case
            assert state.off == want_off, case
            assert state.mismatches == (2 if want_off else 1), case
            checks = [o for o in logs.output if "self-check " in o and "/3" in o]
            first_ok = "ok=False" if bad_rcp else "ok=True"
            assert first_ok in checks[0], (case, checks[0])
            if not want_off:
                # The division form is checked three times more.
                assert len(checks) == 4, (case, checks)
                assert "self-check 3/3" in checks[-1] and "mul_rcp=False" in checks[-1]
                assert state.checks == 3, case
                assert k.calls[-1] == ("v2_encode", (True, D, 8, 2, 4, 4, False))
            else:
                assert k.names()[-1] == "rowmap"
        # An edge run that raises counts as a mismatch.
        _reset()
        state = comm_ops._mx4v2_state["fwd_enc"]
        k = _Kernels()
        k.fail_edge = True
        with k.patched(), _env({"": "fwd_enc", "CHECK": "1"}):
            with self.assertLogs(comm_ops.logger, "ERROR"):
                assert torch.equal(self._fwd(x, codecs), ref)
        assert state.mismatches == 1 and not state.off
        # Without _CHECK nothing compares: the row-tiled bytes go out as they
        # are, which is what the self-check arbitrates.
        _reset()
        k = _Kernels()
        k.bad = {True: True, False: True}
        with k.patched(), _env({"": "fwd_enc"}):
            assert not torch.equal(self._fwd(x, codecs), ref)

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_launch_failure(self) -> None:
        codecs = _codecs()
        grad = torch.randn(20, D, generator=torch.Generator().manual_seed(5))
        k = _Kernels()
        with k.patched(), _env({"": "bwd_enc"}):
            k.fail = True
            with self.assertLogs(comm_ops.logger, "ERROR"):
                out = self._bwd(grad, codecs)
            assert k.names() == ["v2_encode", "chunked"]
            assert comm_ops._mx4v2_state["bwd_enc"].off
            # The failure stays inside the scope: the chunked path stays on.
            assert not comm_ops._seq_a2a_bwd_chunked_disabled
            k.fail = False
            k.calls.clear()
            with self.assertLogs(comm_ops.logger, "WARNING") as logs:
                assert torch.equal(self._bwd(grad, codecs), out)
                assert torch.equal(self._bwd(grad, codecs), out)
            assert k.names() == ["chunked", "chunked"]
            assert len(logs.records) == 1 and "turned off" in logs.output[0]

    def test_codec_path_warns(self) -> None:
        codecs = _codecs()
        x = torch.randn(4, D)
        ctx = codecs.forward.create_context()
        ref = codecs.forward.encode(x, ctx)
        encode = comm_ops._seq_a2a_codec_encode
        with _env({}), _no_mx4v2_logs(self):
            assert torch.equal(encode(codecs.forward, x, ctx, "fwd_enc", "here"), ref)
        with _env({"": "fwd_enc"}), self.assertLogs(comm_ops.logger, "WARNING") as logs:
            for _ in range(2):
                assert torch.equal(
                    encode(codecs.forward, x, ctx, "fwd_enc", "here"), ref
                )
                encode(codecs.forward, x, ctx, "bwd_enc", "there")
        assert len(logs.records) == 1 and "here runs the codec" in logs.output[0]

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_codec_path_call_sites(self) -> None:
        """With their fused paths off, All2All_Seq_Req.forward and
        All2All_Seq_Req_Wait.backward encode through _seq_a2a_codec_encode,
        and TORCHREC_SEQ_A2A_MX4_V2 leaves the result as it is."""
        codecs = _codecs()
        sites: List[str] = []
        encode = comm_ops._seq_a2a_codec_encode

        def spy(codec, x, ctx, scope, what):
            sites.append(scope)
            return encode(codec, x, ctx, scope, what)

        def run(flags: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(8, D, generator=torch.Generator().manual_seed(6))
            x.requires_grad_(True)
            with _env(flags):
                out = _a2a_world1(x, codecs)
                out.backward(torch.ones_like(out))
            grad = x.grad
            assert grad is not None
            return out.detach(), grad

        with _world1(), mock.patch.object(comm_ops, "_seq_a2a_codec_encode", spy):
            ref = run({})
            assert sites == ["fwd_enc", "bwd_enc"], sites
            with self.assertLogs(comm_ops.logger, "WARNING") as logs:
                out = run({"": "1"})
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        for scope in ("fwd_enc", "bwd_enc"):
            warned = [o for o in logs.output if f"{scope} declined" in o]
            assert len(warned) == 1 and "runs the codec" in warned[0], logs.output

    # Interpreter checks (TRITON_INTERPRET=1 in a fresh process).

    @classmethod
    def _check_encode_kernels(cls) -> None:
        _interpreter_or_exit()
        from fbgemm_gpu.triton.quantize import triton_quantize_mx4
        from torchrec.distributed import mx4_rows_v2 as v2
        from torchrec.distributed.mx4_rowmap_quantize import (
            mx4_rows_chunk_plan,
            quantize_mx4_rowmap,
            quantize_mx4_rows_chunked,
        )
        from triton.runtime.interpreter import InterpretedFunction

        assert isinstance(v2._mx4v2_encode_kernel, InterpretedFunction)
        gen = torch.Generator().manual_seed(0)
        cpu = torch.device("cpu")
        for (dim, g, rowmap), dtype in itertools.product(
            ((512, 8, True), (512, 256, False), (256, 32, True), (128, 128, False)),
            (torch.bfloat16, torch.float32),
        ):
            x = torch.cat(
                [
                    comm_ops._mx4v2_edge_input(dim, dtype, cpu),
                    torch.randn(5, dim, generator=gen).to(dtype),
                    _snan_row(dim, dtype, gen),
                ]
            )
            rows = x.shape[0]
            idx = torch.randperm(rows, generator=gen) if rowmap else None
            src = x if idx is None else x[idx].contiguous()
            for mode in (2, 0, 1, 4):
                with _gfx950_semantics():
                    if rowmap and 64 * g <= dim:
                        # pyrefly: ignore [bad-argument-type]
                        ref = quantize_mx4_rowmap(x, idx, dim, g, mode)
                    else:
                        ref = triton_quantize_mx4(src, g, rounding_mode=mode).view(-1)
                    if not rowmap:
                        plan = mx4_rows_chunk_plan(rows, dim, x.element_size(), g)
                        chunked = quantize_mx4_rows_chunked(
                            # pyrefly: ignore
                            x,
                            g,
                            mode,
                            # pyrefly: ignore [unsupported-operation]
                            plan[0],
                            # pyrefly: ignore [unsupported-operation]
                            plan[2],
                        )
                        assert torch.equal(chunked, ref)
                    variants = ((True, 4), (False, 4), (True, 1), (True, 8))
                    for mul_rcp, block_r in variants if mode == 2 else variants[:2]:
                        out = v2.mx4_encode_rows(
                            x, idx, dim, g, mode, block_r=block_r, mul_rcp=mul_rcp
                        )
                        assert torch.equal(out, ref), (
                            dim,
                            g,
                            dtype,
                            mode,
                            mul_rcp,
                            block_r,
                        )
                assert torch.equal(v2.mx4_encode_rows(x, idx, dim, g, mode), ref)
        # Empty input.
        assert v2.mx4_encode_rows(torch.empty(0, 512), None, 512, 8, 2).numel() == 0

    @unittest.skipUnless(_IN_REPO_FBGEMM, "needs the in-repo fbgemm MX4 kernels")
    def test_encode_kernels_interpret_cpu(self) -> None:
        _run_interpreted(self, self._check_encode_kernels)

    @classmethod
    def _check_encode_routing(cls) -> None:
        """_seq_a2a_fwd_quantize_rowmap and _seq_a2a_bwd_chunked_encode with
        the real kernels: TORCHREC_SEQ_A2A_MX4_V2 and its self-check leave the
        payload as it is."""
        _interpreter_or_exit()
        codecs = _codecs()
        gen = torch.Generator().manual_seed(7)
        logging.basicConfig(level=logging.INFO)
        with _gfx950_semantics():
            for dtype in (torch.bfloat16, torch.float32):
                _reset()
                a2ai, lengths, W = _rowmap_info(codecs)
                x = (torch.randn(_FWD_ROWS, D, generator=gen) * 1e-3).to(dtype)
                grad = torch.randn(40, D, generator=gen).to(dtype)
                ctx = codecs.backward.create_context()
                with _gpu_branch():
                    with _env({}):
                        ref_f = comm_ops._seq_a2a_fwd_quantize_rowmap(
                            a2ai, x, lengths, W
                        )
                        ref_b = comm_ops._seq_a2a_bwd_chunked_encode(
                            _rows_info(codecs), grad, ctx
                        )
                    for flags in (
                        {"": "1", "CHECK": "1"},
                        {"": "1"},
                        {"": "1", "DIV": "1", "CHECK": "1"},
                    ):
                        for _ in range(2):
                            with _env(flags):
                                out_f = comm_ops._seq_a2a_fwd_quantize_rowmap(
                                    a2ai, x, lengths, W
                                )
                                out_b = comm_ops._seq_a2a_bwd_chunked_encode(
                                    _rows_info(codecs), grad, ctx
                                )
                            # pyrefly: ignore
                            assert torch.equal(out_f[1], ref_f[1]) and torch.equal(
                                # pyrefly: ignore
                                out_f[0],
                                # pyrefly: ignore [unsupported-operation]
                                ref_f[0],
                            )
                            # pyrefly: ignore [bad-argument-type]
                            assert torch.equal(out_b, ref_b)
                for scope in ("fwd_enc", "bwd_enc"):
                    state = comm_ops._mx4v2_state[scope]
                    assert state.checks == 3 and state.mismatches == 0 and not state.off
                    assert len(state.edge_checked) == 2, state.edge_checked

    @unittest.skipUnless(_IN_REPO_FBGEMM, "needs the in-repo fbgemm MX4 kernels")
    def test_encode_routing_interpret_cpu(self) -> None:
        _run_interpreted(self, self._check_encode_routing)


# Decode tests.


def _ref_decode(
    payload: torch.Tensor,
    row_idx: Optional[torch.Tensor],
    n_rows: int,
    dim: int,
    group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """A CPU stand-in for an old decode kernel: fbgemm's reference MX4 decode
    of the packed rows."""
    from fbgemm_gpu.triton.quantize_ref import py_dequantize_mx4

    rows = payload.view(-1, dim // 2 + dim // group_size)
    rows = rows[:n_rows] if row_idx is None else rows[row_idx]
    return py_dequantize_mx4(rows.contiguous().view(-1), group_size).to(out_dtype)


class _DecodeKernels(_Kernels):
    def rowmap_dequant(self, payload, row_idx, dim, group_size, out_dtype):
        self.calls.append(("rowmap_dequant", (dim, group_size, out_dtype)))
        return _ref_decode(
            payload, row_idx, row_idx.numel(), dim, group_size, out_dtype
        )

    def decode_rows(
        self, payload, row_idx, n_rows, dim, group_size, out_dtype, block_r, num_warps
    ):
        self.calls.append(
            (
                "v2_decode",
                (
                    row_idx is not None,
                    n_rows,
                    dim,
                    group_size,
                    out_dtype,
                    block_r,
                    num_warps,
                ),
            )
        )
        if self.fail:
            raise RuntimeError("compile failure")
        edge = n_rows == comm_ops._MX4V2_EDGE_ROWS or n_rows == 256
        out = _ref_decode(payload, row_idx, n_rows, dim, group_size, out_dtype)
        return self._corrupt(out, False, edge)

    @contextlib.contextmanager
    def patched(self) -> Iterator["_Kernels"]:
        from torchrec.distributed import mx4_rowmap_dequant as mx4dq, mx4_rows_v2

        with super().patched(), mock.patch.object(
            mx4dq, "dequantize_mx4_rowmap", self.rowmap_dequant
        ), mock.patch.object(mx4_rows_v2, "mx4_decode_rows", self.decode_rows):
            yield self


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _codecs_out(output_dtype: Optional[SparseType]) -> QuantizedCommCodecs:
    return get_qcomm_codecs(
        QCommsConfig(
            forward_precision=CommType.MX4,
            backward_precision=CommType.MX4,
            mx4_quantize_dim=8,
            mx4_quantize_dim_bwd=256,
            output_dtype=output_dtype,
        )
    )


def _bwd_rowmap_info(
    codecs: QuantizedCommCodecs,
) -> Tuple[comm_ops.All2AllSequenceInfo, int]:
    """An All2AllSequenceInfo for _seq_a2a_bwd_dequant_rowmap, and its rows."""
    T, W, B = 2, 2, 3
    gen = torch.Generator().manual_seed(8)
    lengths = torch.randint(0, 4, (T * W, B), generator=gen).int()
    lengths[0, 0] += _FWD_ROWS - int(lengths.sum())
    recat = torch.IntTensor([i + j * T for i in range(T) for j in range(W)])
    a2ai = comm_ops.All2AllSequenceInfo(
        embedding_dim=D,
        lengths_after_sparse_data_all2all=lengths,
        forward_recat_tensor=None,
        backward_recat_tensor=recat,
        input_splits=[],
        output_splits=[],
        codecs=codecs,
        permuted_lengths_after_sparse_data_all2all=lengths * D,
    )
    return a2ai, _FWD_ROWS


class MX4RowsV2DecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        _reset()

    def tearDown(self) -> None:
        _reset()

    def _payload(self, codec, rows: int, seed: int) -> Tuple[torch.Tensor, object]:
        ctx = codec.create_context()
        gen = torch.Generator().manual_seed(seed)
        return codec.encode(torch.randn(rows, D, generator=gen).view(-1), ctx), ctx

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_fwd_dec_routing(self) -> None:
        for output_dtype, out_dtype in (
            (None, torch.float32),
            (SparseType.BF16, torch.bfloat16),
        ):
            _reset()
            codec = _codecs_out(output_dtype).forward
            payload, ctx = self._payload(codec, _FWD_ROWS, 9)
            k = _DecodeKernels()
            row_bytes = D // 2 + D // 8

            def cpu_decode(
                p: torch.Tensor,
                c: object,
                row_bytes: int = row_bytes,
                out_dtype: torch.dtype = out_dtype,
            ) -> torch.Tensor:
                return _ref_decode(p, None, p.numel() // row_bytes, D, 8, out_dtype)

            with k.patched(), mock.patch.object(
                codec, "decode", side_effect=cpu_decode
            ) as dec:
                with _env({}), _gpu_branch(), _no_mx4v2_logs(self):
                    ref = comm_ops._seq_a2a_fwd_decode(codec, D, payload, ctx)
                assert dec.call_count == 1 and k.calls == []
                with _env({"": "fwd_dec"}), _gpu_branch(), self.assertLogs(
                    comm_ops.logger, "INFO"
                ) as logs:
                    out = comm_ops._seq_a2a_fwd_decode(codec, D, payload, ctx)
                assert dec.call_count == 1
                assert k.calls == [
                    ("v2_decode", (False, _FWD_ROWS, D, 8, out_dtype, 4, 4))
                ]
                assert torch.equal(_bits(out), _bits(ref)) and out.dtype == out_dtype
                assert len(logs.records) == 1 and "fwd_dec on" in logs.output[0]
                # The unbucketize-order decode (TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP)
                # decodes its gathered rows the same way.
                row_map = torch.randperm(
                    _FWD_ROWS, generator=torch.Generator().manual_seed(1)
                )
                a2ai = _rows_info(_codecs_out(output_dtype))
                # pyrefly: ignore [missing-attribute]
                a2ai.codecs.forward = codec
                with _env({"": "fwd_dec"}), _gpu_branch():
                    got = comm_ops._seq_a2a_fwd_decode_unbucketize(
                        a2ai, payload, ctx, row_map
                    )
                assert got is not None
                assert torch.equal(_bits(got), _bits(ref.view(-1, D)[row_map]))
                assert k.names()[-1] == "v2_decode"

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_bwd_dec_routing(self) -> None:
        for output_dtype, out_dtype in (
            (None, torch.float32),
            (SparseType.BF16, torch.bfloat16),
        ):
            _reset()
            codecs = _codecs_out(output_dtype)
            a2ai, rows = _bwd_rowmap_info(codecs)
            payload, ctx = self._payload(codecs.backward, rows, 10)
            k = _DecodeKernels()
            with k.patched():
                with _env({}), _gpu_branch():
                    ref = comm_ops._seq_a2a_bwd_dequant_rowmap(a2ai, payload, ctx)
                assert k.names() == ["rowmap_dequant"]
                k.calls.clear()
                with _env({"": "bwd_dec"}), _gpu_branch():
                    out = comm_ops._seq_a2a_bwd_dequant_rowmap(a2ai, payload, ctx)
            assert out is not None and ref is not None
            assert k.calls == [("v2_decode", (True, rows, D, 256, out_dtype, 4, 4))]
            assert torch.equal(_bits(out), _bits(ref))

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_fwd_dec_declines(self) -> None:
        codecs = _codecs()
        codec = codecs.forward
        payload, ctx = self._payload(codec, _FWD_ROWS, 11)
        fp16 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP16, backward_precision=CommType.FP16
            )
        ).forward
        k = _DecodeKernels()
        with k.patched(), _env({"": "1"}), _gpu_branch(), mock.patch.object(
            codec, "decode", lambda p, c: p
        ):
            for args, reason in (
                (
                    (fp16, D, fp16.encode(torch.randn(4 * D)), fp16.create_context()),
                    "not an fbgemm MX4",
                ),
                ((codec, D, payload[:-1], ctx), "not whole packed rows"),
                ((codec, D, payload.view(torch.int8), ctx), "not uint8"),
                ((codec, D, payload.view(-1, 2), ctx), "not contiguous and 1-D"),
            ):
                with self.assertLogs(comm_ops.logger, "WARNING") as logs:
                    comm_ops._seq_a2a_fwd_decode(*args)
                assert reason in logs.output[0], logs.output
            for attr, value, reason in (
                ("_loss_scale", 2.0, "without a loss scale"),
                ("_output_dtype", SparseType.FP16, "output dtype"),
            ):
                comm_ops._mx4v2_warned.clear()
                with mock.patch.object(codec, attr, value), mock.patch.object(
                    codec, "decode", lambda p, c: p
                ), self.assertLogs(comm_ops.logger, "WARNING") as logs:
                    out = comm_ops._seq_a2a_fwd_decode(codec, D, payload, ctx)
                assert out is payload and reason in logs.output[0], logs.output
        assert k.calls == []

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_decode_self_check(self) -> None:
        codecs = _codecs()
        a2ai, rows = _bwd_rowmap_info(codecs)
        payload, ctx = self._payload(codecs.backward, rows, 12)
        state = comm_ops._mx4v2_state["bwd_dec"]
        k = _DecodeKernels()
        with k.patched(), _env({"": "bwd_dec", "CHECK": "1"}), _gpu_branch():
            ref = comm_ops._seq_a2a_bwd_dequant_rowmap(a2ai, payload, ctx)
            for _ in range(3):
                assert torch.equal(
                    # pyrefly: ignore [bad-argument-type]
                    comm_ops._seq_a2a_bwd_dequant_rowmap(a2ai, payload, ctx),
                    # pyrefly: ignore [bad-argument-type]
                    ref,
                )
        assert state.checks == 3 and not state.off
        assert k.names().count("rowmap_dequant") == 3 + 1
        # A decode has no second form: its first mismatch turns it off.
        for bad, bad_edge in ((True, False), (False, True)):
            _reset()
            state = comm_ops._mx4v2_state["bwd_dec"]
            k = _DecodeKernels()
            k.bad = {True: False, False: bad}
            k.bad_edge = bad_edge
            with k.patched(), _env({"": "bwd_dec", "CHECK": "1"}), _gpu_branch():
                with self.assertLogs(comm_ops.logger, "ERROR"):
                    for _ in range(3):
                        out = comm_ops._seq_a2a_bwd_dequant_rowmap(a2ai, payload, ctx)
                        # pyrefly: ignore [bad-argument-type]
                        assert torch.equal(out, ref)
            assert state.off and state.mismatches == 1
            assert k.names()[-1] == "rowmap_dequant"

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_codec_decode_call_sites(self) -> None:
        """All2All_Seq_Req_Wait.forward decodes through _seq_a2a_fwd_decode and
        All2All_Seq_Req.backward, without its row-map decode, through
        _seq_a2a_codec_decode."""
        codecs = _codecs()
        sites: List[str] = []
        fwd_decode = comm_ops._seq_a2a_fwd_decode
        codec_decode = comm_ops._seq_a2a_codec_decode

        def fwd_spy(codec, dim, payload, ctx):
            sites.append("fwd_dec")
            return fwd_decode(codec, dim, payload, ctx)

        def codec_spy(codec, payload, ctx, scope, what):
            sites.append(scope)
            return codec_decode(codec, payload, ctx, scope, what)

        with _world1(), mock.patch.object(
            comm_ops, "_seq_a2a_fwd_decode", fwd_spy
        ), mock.patch.object(comm_ops, "_seq_a2a_codec_decode", codec_spy):
            x = torch.randn(8, D, requires_grad=True)
            with _env({"": "1"}), self.assertLogs(comm_ops.logger, "WARNING") as logs:
                out = _a2a_world1(x, codecs)
                out.backward(torch.ones_like(out))
        assert sites == ["fwd_dec", "bwd_dec"], sites
        # On CPU torch fwd_dec declines (no ROCm); bwd_dec runs the codec.
        assert any("fwd_dec declined" in o and "ROCm" in o for o in logs.output)
        assert any(
            "bwd_dec declined" in o and "runs the codec" in o for o in logs.output
        )

    @classmethod
    def _check_decode_kernels(cls) -> None:
        _interpreter_or_exit()
        from fbgemm_gpu.triton.quantize import triton_dequantize_mx4
        from torchrec.distributed import mx4_rows_v2 as v2
        from torchrec.distributed.mx4_rowmap_dequant import dequantize_mx4_rowmap
        from triton.runtime.interpreter import InterpretedFunction

        assert isinstance(v2._mx4v2_decode_kernel, InterpretedFunction)
        gen = torch.Generator().manual_seed(1)
        cpu = torch.device("cpu")
        for (dim, g, rowmap), out_dtype in itertools.product(
            (
                (512, 8, False),
                (512, 256, True),
                (256, 32, True),
                (256, 32, False),
                (128, 128, False),
            ),
            (torch.bfloat16, torch.float32),
        ):
            edge, n = comm_ops._mx4v2_edge_payload(dim, g, cpu)
            row_bytes = dim // 2 + dim // g
            rand = torch.randint(0, 256, (7 * row_bytes,), generator=gen).to(
                torch.uint8
            )
            payload = torch.cat([edge, rand])
            rows = n + 7
            idx = torch.randperm(rows, generator=gen) if rowmap else None
            if rowmap:
                # pyrefly: ignore [bad-argument-type]
                ref = dequantize_mx4_rowmap(payload, idx, dim, g, out_dtype)
                full = triton_dequantize_mx4(payload, g, output_dtype=out_dtype)
                assert torch.equal(
                    _bits(ref), _bits(full.view(rows, dim)[idx].view(-1))
                )
            else:
                ref = triton_dequantize_mx4(payload, g, output_dtype=out_dtype)
            for block_r in (4, 1, 8):
                out = v2.mx4_decode_rows(
                    payload, idx, rows, dim, g, out_dtype, block_r=block_r
                )
                assert out.dtype == out_dtype
                assert torch.equal(_bits(out), _bits(ref)), (dim, g, out_dtype, block_r)
            # Every table value at every exponent byte, signed zeros included.
            assert torch.unique(_bits(ref)).numel() > 1000
        assert (
            v2.mx4_decode_rows(
                torch.empty(0, dtype=torch.uint8), None, 0, 512, 8, torch.float32
            ).numel()
            == 0
        )

    def test_decode_kernels_interpret_cpu(self) -> None:
        _run_interpreted(self, self._check_decode_kernels)

    @classmethod
    def _check_decode_routing(cls) -> None:
        """_seq_a2a_fwd_decode and _seq_a2a_bwd_dequant_rowmap with the real
        kernels: TORCHREC_SEQ_A2A_MX4_V2 and its self-check leave the decoded
        bits as they are."""
        _interpreter_or_exit()
        logging.basicConfig(level=logging.INFO)
        for output_dtype in (None, SparseType.BF16):
            _reset()
            codecs = _codecs_out(output_dtype)
            a2ai, rows = _bwd_rowmap_info(codecs)
            gen = torch.Generator().manual_seed(13)
            fctx, bctx = (
                codecs.forward.create_context(),
                codecs.backward.create_context(),
            )
            # The inputs come from fbgemm's CPU encode; only the decode routing
            # runs as if on the GPU.
            fwd = codecs.forward.encode(torch.randn(rows * D, generator=gen), fctx)
            bwd = codecs.backward.encode(torch.randn(rows * D, generator=gen), bctx)
            with _gpu_branch():
                with _env({}):
                    ref_f = comm_ops._seq_a2a_fwd_decode(codecs.forward, D, fwd, fctx)
                    ref_b = comm_ops._seq_a2a_bwd_dequant_rowmap(a2ai, bwd, bctx)
                for flags in (
                    {"": "1", "CHECK": "1"},
                    {"": "1"},
                    {"": "fwd_dec,bwd_dec", "CHECK": "1"},
                ):
                    for _ in range(2):
                        with _env(flags):
                            out_f = comm_ops._seq_a2a_fwd_decode(
                                codecs.forward, D, fwd, fctx
                            )
                            out_b = comm_ops._seq_a2a_bwd_dequant_rowmap(
                                a2ai, bwd, bctx
                            )
                        assert torch.equal(_bits(out_f), _bits(ref_f))
                        # pyrefly: ignore [bad-argument-type]
                        assert torch.equal(_bits(out_b), _bits(ref_b))
            for scope in ("fwd_dec", "bwd_dec"):
                state = comm_ops._mx4v2_state[scope]
                assert state.checks == 3 and state.mismatches == 0 and not state.off
                assert len(state.edge_checked) == 1, state.edge_checked

    def test_decode_routing_interpret_cpu(self) -> None:
        _run_interpreted(self, self._check_decode_routing)
