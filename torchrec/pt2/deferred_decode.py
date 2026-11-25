#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decode MX4 embedding all-to-all payloads where the embeddings are used.

With deferred decoding, an output dist returns placeholders instead of decoded
embeddings: zero-memory NaN tensors of the embedding's shape that point at the
payload they decode from. `materialize_all(xs)` turns placeholders into the
embeddings. Called in a compiled region next to the ops that use the embeddings,
it puts the decode in that graph, so the compiler keeps the small payload for
backward and recomputes the decode there instead of keeping the decoded
embeddings.

All of a payload's embeddings are decoded by one op with one kernel launch, each
into its own `[rows, dim]` tensor, joining column-wise shards in place. The op's
backward writes their gradients into the all-to-all's gradient with one kernel
launch.

`materialize_all` is plain torch code plus custom ops (no torchrec.distributed
imports) so dynamo traces it.
"""

import functools
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


class DeferredMX4(NamedTuple):
    """An MX4 all-to-all payload whose embeddings are decoded where they are used.

    `table` is a CPU int64 `[pieces, 3]` tensor of `(output, start, dim)`: each
    piece is a column block of output `output`, `rows * dim` values stored
    row-major from element `start` of the decoded buffer. Pieces are grouped by
    output and in column order within an output. `flat` is the all-to-all's
    output, a placeholder of the decoded buffer's shape: the embeddings'
    gradient goes to it.
    """

    payload: torch.Tensor
    flat: torch.Tensor
    table: torch.Tensor
    group_size: int


# Name of the attribute on a placeholder that holds `(DeferredMX4, output)`.
DEFERRED_ATTR = "_deferred_mx4"

# Each row of a kernel's piece table is int64: pointer of the output (decode) or
# gradient (backward), the piece's first column in it, start in the decoded
# buffer, number of values, piece width, row stride, first block, log2 of the
# piece width (-1 if not a power of two).
_BLOCK_GROUPS = 256
_GROUPS_PER_PROGRAM = 16 * _BLOCK_GROUPS
_BLOCK = 1024
_ELEMS_PER_PROGRAM = 16 * _BLOCK


@triton.jit
def _decode_pieces_kernel(
    payload_ptr,
    table_ptr,
    num_pieces,
    out_dtype_ptr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    LOOP: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    ALIGNED: tl.constexpr,
    WIDE: tl.constexpr,
) -> None:
    """Each program decodes `LOOP * BLOCK_GROUPS` MX4 groups of one piece. The
    piece is the last table row whose first block is at or before the program
    id. `out_dtype_ptr` is any output; only its pointer type is used.
    `ALIGNED`: every piece starts on a group boundary and the piece widths,
    output widths, columns and output addresses are multiples of `GROUP_SIZE`
    elements, so each group lands in one output row and is stored as one
    contiguous vector. `WIDE`: offsets or row addresses reach 2^31 elements, so
    index arithmetic is done in int64.

    Same arithmetic as fbgemm's `_kernel_dequantize_mx4`, so bit-identical to it.
    Each group of `GROUP_SIZE` values is stored as `GROUP_SIZE // 2` bytes of
    4-bit codes (low nibble first) followed by one biased fp32 exponent byte.
    """
    pid = tl.program_id(0).to(tl.int64)
    lo = pid * 0
    hi = lo + num_pieces
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        right = tl.load(table_ptr + mid * 8 + 6) <= pid
        lo = tl.where(right, mid, lo)
        hi = tl.where(right, hi, mid)
    row = table_ptr + lo * 8
    out_ptr = tl.load(row).to(out_dtype_ptr.dtype)
    start = tl.load(row + 2)
    log2_dim = tl.load(row + 7).to(tl.int32)
    if WIDE:
        col = tl.load(row + 1)
        numel = tl.load(row + 3)
        dim = tl.load(row + 4)
        stride = tl.load(row + 5)
        block = pid - tl.load(row + 6)
    else:
        col = tl.load(row + 1).to(tl.int32)
        numel = tl.load(row + 3).to(tl.int32)
        dim = tl.load(row + 4).to(tl.int32)
        stride = tl.load(row + 5).to(tl.int32)
        block = (pid - tl.load(row + 6)).to(tl.int32)

    PACKED: tl.constexpr = GROUP_SIZE // 2 + 1
    HALF: tl.constexpr = GROUP_SIZE // 2
    first_group = start // GROUP_SIZE
    shift = (start - first_group * GROUP_SIZE).to(block.dtype)
    payload_ptr += first_group * PACKED
    j = tl.arange(0, GROUP_SIZE)
    for it in range(LOOP):
        # Groups relative to the piece's first group, and the position of their
        # first value relative to the piece's start.
        g = (block * LOOP + it) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        g_offs = g * GROUP_SIZE - shift
        g_mask = (g_offs < numel) & (g_offs + GROUP_SIZE > 0)

        exp = tl.load(payload_ptr + g * PACKED + HALF, mask=g_mask, other=0)
        exp = exp.to(tl.int32) - 127
        # 2^exp from its fp32 bits; 2^-127 is the subnormal 0x00400000.
        scale = tl.where(exp >= -126, (exp + 127) << 23, 1 << 22).to(
            tl.float32, bitcast=True
        )
        byte = tl.load(
            payload_ptr + g[:, None] * PACKED + tl.arange(0, HALF)[None, :],
            mask=g_mask[:, None],
            other=0,
        ).to(tl.int32)
        code = tl.reshape(
            tl.join(byte & 0xF, (byte >> 4) & 0xF), [BLOCK_GROUPS, GROUP_SIZE]
        )
        # E2M1 code s|e1 e0|m is (-1)^s * 2^(e-1) * (1 + m/2) for e > 0 and
        # (-1)^s * m/2 for e = 0, built from fp32 bits. Code 8 decodes to +0, as
        # in fbgemm's lookup table.
        e = (code >> 1) & 0x3
        m = code & 0x1
        bits = tl.where(e > 0, ((e + 126) << 23) | (m << 22), m * 0x3F000000)
        bits = bits | (((code >= 8) & ((code & 0x7) > 0)).to(tl.int32) << 31)
        value = bits.to(tl.float32, bitcast=True) * scale[:, None]

        if ALIGNED:
            # A shift instead of an integer division for power-of-two widths.
            if log2_dim >= 0:
                r = g_offs >> log2_dim
            else:
                r = g_offs // dim
            base = tl.multiple_of(r * stride + col + (g_offs - r * dim), GROUP_SIZE)
            tl.store(
                tl.multiple_of(out_ptr, 16) + base[:, None] + j[None, :],
                value,
                mask=g_mask[:, None],
            )
        else:
            offs = g_offs[:, None] + j[None, :]
            r = offs // dim
            tl.store(
                out_ptr + r * stride + col + (offs - r * dim),
                value,
                mask=(offs >= 0) & (offs < numel),
            )


@triton.jit
def _scatter_pieces_kernel(
    table_ptr,
    num_pieces,
    out_ptr,
    src_dtype_ptr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WIDE: tl.constexpr,
) -> None:
    """The decode's layout in reverse: each program copies `LOOP * BLOCK` values
    of one piece from its `[rows, stride]` source into the flat output.
    `src_dtype_ptr` is any source; only its pointer type is used. `WIDE` as in
    `_decode_pieces_kernel`."""
    pid = tl.program_id(0).to(tl.int64)
    lo = pid * 0
    hi = lo + num_pieces
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        right = tl.load(table_ptr + mid * 8 + 6) <= pid
        lo = tl.where(right, mid, lo)
        hi = tl.where(right, hi, mid)
    row = table_ptr + lo * 8
    src_ptr = tl.load(row).to(src_dtype_ptr.dtype)
    start = tl.load(row + 2)
    log2_dim = tl.load(row + 7).to(tl.int32)
    if WIDE:
        col = tl.load(row + 1)
        numel = tl.load(row + 3)
        dim = tl.load(row + 4)
        stride = tl.load(row + 5)
        block = pid - tl.load(row + 6)
    else:
        col = tl.load(row + 1).to(tl.int32)
        numel = tl.load(row + 3).to(tl.int32)
        dim = tl.load(row + 4).to(tl.int32)
        stride = tl.load(row + 5).to(tl.int32)
        block = (pid - tl.load(row + 6)).to(tl.int32)
    for it in range(LOOP):
        offs = (block * LOOP + it) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        if log2_dim >= 0:
            r = offs >> log2_dim
        else:
            r = offs // dim
        value = tl.load(src_ptr + r * stride + col + (offs - r * dim), mask=mask)
        tl.store(out_ptr + start + offs, value, mask=mask)


def _piece_columns(
    table: torch.Tensor,
    outputs: Sequence[int],
    tensors: Sequence[Optional[torch.Tensor]],
) -> np.ndarray:
    """`[pieces, 7]` int64 rows of pointer, column, start, numel, dim, row stride
    and log2 of dim for the pieces of `outputs`, `tensors[i]` being output
    `outputs[i]`. Pieces of a missing tensor are left out.

    Built with numpy: a few hundred pieces is where per-op overhead, not
    arithmetic, sets the cost.
    """
    spec = table.numpy()
    if len(spec) == 0:
        return np.zeros((0, 7), dtype=np.int64)
    pos = np.full(max(int(spec[:, 0].max()), max(outputs, default=0)) + 1, -1)
    pos[np.asarray(outputs, dtype=np.int64)] = np.arange(len(outputs))
    p = pos[spec[:, 0]]
    present = np.asarray([t is not None for t in tensors], dtype=bool)
    keep = p >= 0
    keep[keep] = present[p[keep]]
    first = np.ones(len(spec), dtype=bool)
    first[1:] = spec[1:, 0] != spec[:-1, 0]
    dim = spec[:, 2]
    before = np.cumsum(dim) - dim
    col = before - np.maximum.accumulate(np.where(first, before, 0))
    spec, p, col, dim = spec[keep], p[keep], col[keep], dim[keep]
    ptr = np.asarray([t.data_ptr() if t is not None else 0 for t in tensors])
    rows = np.asarray([t.shape[0] if t is not None else 0 for t in tensors])
    stride = np.asarray([t.stride(0) if t is not None else 0 for t in tensors])
    pow2 = (dim & (dim - 1)) == 0
    log2_dim = np.where(pow2, np.log2(np.maximum(dim, 1)).astype(np.int64), -1)
    return np.stack(
        [ptr[p], col, spec[:, 1], rows[p] * dim, dim, stride[p], log2_dim],
        axis=1,
    ).astype(np.int64)


def _needs_wide(cols: np.ndarray, slack: int) -> bool:
    """Whether a kernel's int32 index arithmetic could reach 2^31 for these
    pieces: offsets within a piece run up to its numel plus one program's span
    (`slack`), and row addresses up to rows times the row stride."""
    if len(cols) == 0:
        return False
    numel, dim, stride = cols[:, 3], cols[:, 4], cols[:, 5]
    rows = numel // np.maximum(dim, 1)
    bound = np.maximum(numel + slack, (rows + 1) * stride + slack)
    return bool(bound.max() >= 2**31)


def _tiles_exactly(cols: np.ndarray, numel: int) -> bool:
    """Whether the pieces cover `[0, numel)` exactly once."""
    order = np.argsort(cols[:, 2], kind="stable")
    start, size = cols[order, 2], cols[order, 3]
    keep = size > 0
    start, size = start[keep], size[keep]
    end = start + size
    return (
        len(start) > 0
        and start[0] == 0
        and end[-1] == numel
        and bool(np.all(start[1:] == end[:-1]))
    )


def _kernel_table(
    cols: np.ndarray, blocks: np.ndarray, device: torch.device
) -> torch.Tensor:
    first_block = np.cumsum(blocks) - blocks
    table = np.concatenate([cols[:, :6], first_block[:, None], cols[:, 6:]], axis=1)
    return torch.from_numpy(table).pin_memory().to(device, non_blocking=True)


@torch.library.custom_op("torchrec::mx4_decode", mutates_args=())
def mx4_decode(
    payload: torch.Tensor,
    flat: torch.Tensor,
    table: torch.Tensor,
    outputs: list[int],
    rows: list[int],
    widths: list[int],
    group_size: int,
) -> list[torch.Tensor]:
    """Decodes outputs `outputs` (see `DeferredMX4`) of an MX4 payload with one
    kernel launch, output `outputs[i]` as a `[rows[i], widths[i]]` tensor.
    `flat` only carries the gradient: the backward writes the outputs'
    gradients into a tensor of its shape.
    """
    assert len(set(outputs)) == len(outputs), "outputs must be distinct"
    assert payload.element_size() == 1, "an MX4 payload is a byte tensor"
    # The kernel reads raw bytes: unsigned, densely packed.
    payload = payload.contiguous().view(torch.uint8)
    outs = [
        torch.empty((r, w), dtype=flat.dtype, device=payload.device)
        for r, w in zip(rows, widths)
    ]
    cols = _piece_columns(table, outputs, outs)
    ptr, col, start, numel, dim, stride = cols[:, :6].T
    groups = np.where(
        numel > 0, (start + numel - 1) // group_size - start // group_size + 1, 0
    )
    blocks = (groups + _GROUPS_PER_PROGRAM - 1) // _GROUPS_PER_PROGRAM
    num_blocks = int(blocks.sum())
    if num_blocks == 0:
        return outs
    elem = outs[0].element_size()
    aligned = not np.any(
        (start | col | dim | stride | (ptr // elem)) % group_size
    ) and not np.any(ptr % 16)
    _decode_pieces_kernel[(num_blocks,)](
        payload,
        _kernel_table(cols, blocks, payload.device),
        len(cols),
        outs[0],
        GROUP_SIZE=group_size,
        BLOCK_GROUPS=_BLOCK_GROUPS,
        LOOP=_GROUPS_PER_PROGRAM // _BLOCK_GROUPS,
        SEARCH_STEPS=max(1, (len(cols) - 1).bit_length()),
        ALIGNED=aligned,
        WIDE=_needs_wide(cols, (_GROUPS_PER_PROGRAM + 1) * group_size),
    )
    return outs


@mx4_decode.register_fake
def _(
    payload: torch.Tensor,
    flat: torch.Tensor,
    table: torch.Tensor,
    outputs: list[int],
    rows: list[int],
    widths: list[int],
    group_size: int,
) -> list[torch.Tensor]:
    return [payload.new_empty((r, w), dtype=flat.dtype) for r, w in zip(rows, widths)]


@torch.library.custom_op("torchrec::mx4_decode_backward", mutates_args=())
def mx4_decode_backward(
    grads: list[Optional[torch.Tensor]],
    table: torch.Tensor,
    outputs: list[int],
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """The gradient of `mx4_decode`'s `flat` (`numel` values) for its outputs'
    gradients `grads`, with one kernel launch. Values no gradient covers are
    zero, and where pieces overlap their gradients are summed, as autograd sums
    them for overlapping views."""
    assert len(set(outputs)) == len(outputs), "outputs must be distinct"
    out = torch.empty(numel, dtype=dtype, device=device)
    grads = [g if g is None or g.stride(-1) == 1 else g.contiguous() for g in grads]
    cols = _piece_columns(table, outputs, grads)
    if not _tiles_exactly(cols, numel):
        out.zero_()
        if not _disjoint(cols):
            # Overlapping pieces, which TorchRec layouts never produce: a plain
            # accumulation is enough.
            _accumulate_pieces(out, table, outputs, grads)
            return out
    piece_numel = cols[:, 3]
    blocks = (piece_numel + _ELEMS_PER_PROGRAM - 1) // _ELEMS_PER_PROGRAM
    num_blocks = int(blocks.sum())
    if num_blocks == 0:
        return out
    _scatter_pieces_kernel[(num_blocks,)](
        _kernel_table(cols, blocks, out.device),
        len(cols),
        out,
        next(g for g in grads if g is not None),
        BLOCK=_BLOCK,
        LOOP=_ELEMS_PER_PROGRAM // _BLOCK,
        SEARCH_STEPS=max(1, (len(cols) - 1).bit_length()),
        WIDE=_needs_wide(cols, _ELEMS_PER_PROGRAM),
    )
    return out


def _disjoint(cols: np.ndarray) -> bool:
    order = np.argsort(cols[:, 2], kind="stable")
    start, size = cols[order, 2], cols[order, 3]
    keep = size > 0
    start, size = start[keep], size[keep]
    return bool(np.all(start[1:] >= start[:-1] + size[:-1]))


def _accumulate_pieces(
    out: torch.Tensor,
    table: torch.Tensor,
    outputs: Sequence[int],
    grads: Sequence[Optional[torch.Tensor]],
) -> None:
    index = {o: i for i, o in enumerate(outputs)}
    col: dict[int, int] = {}
    for o, start, dim in table.tolist():
        c = col.get(o, 0)
        col[o] = c + dim
        g = grads[index[o]] if o in index else None
        if g is None or g.shape[0] * dim == 0:
            continue
        out[start : start + g.shape[0] * dim].view(-1, dim).add_(g[:, c : c + dim])


@mx4_decode_backward.register_fake
def _(
    grads: list[Optional[torch.Tensor]],
    table: torch.Tensor,
    outputs: list[int],
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device=device)


def _decode_setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: list[torch.Tensor]
) -> None:
    _, flat, table, outputs, _, _, _ = inputs
    ctx.save_for_backward(table)
    ctx.outputs = outputs
    ctx.flat = (flat.shape[0], flat.dtype, flat.device)


def _decode_backward(ctx: Any, grads: list[Optional[torch.Tensor]]) -> tuple[Any, ...]:
    (table,) = ctx.saved_tensors
    grad = mx4_decode_backward(list(grads), table, ctx.outputs, *ctx.flat)
    return None, grad, None, None, None, None, None


mx4_decode.register_autograd(_decode_backward, setup_context=_decode_setup_context)


def placeholders(
    payload: torch.Tensor,
    flat: torch.Tensor,
    rows: Sequence[int],
    outputs: Sequence[Sequence[tuple[int, int]]],
    group_size: int,
) -> list[torch.Tensor]:
    """Placeholders for the embeddings in an MX4 payload: embedding `i` is
    `[rows[i], width_i]`, made of the `(start, dim)` pieces `outputs[i]` in
    column order, `width_i` the sum of their dims (see `DeferredMX4`)."""
    table = torch.tensor(
        [(i, start, dim) for i, ps in enumerate(outputs) for start, dim in ps],
        dtype=torch.int64,
    ).view(-1, 3)
    group = DeferredMX4(payload, flat, table, group_size)
    nan = torch.full((), float("nan"), dtype=flat.dtype, device=flat.device)
    result = []
    for i, (r, ps) in enumerate(zip(rows, outputs)):
        x = nan.expand(r, sum(dim for _, dim in ps))
        setattr(x, DEFERRED_ATTR, (group, i))
        result.append(x)
    return result


def deferred(x: torch.Tensor) -> Optional[tuple[DeferredMX4, int]]:
    """The payload and output a placeholder decodes from, or None."""
    # Not a tensor under torch.fx symbolic tracing; nothing to decode there.
    if not isinstance(x, torch.Tensor):
        return None
    return getattr(x, DEFERRED_ATTR, None)


def _recompute_all(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    return CheckpointPolicy.MUST_RECOMPUTE


def _decode_groups(
    payloads: list[torch.Tensor],
    flats: list[torch.Tensor],
    tables: list[torch.Tensor],
    specs: list[tuple[list[int], list[int], list[int], int]],
) -> list[list[torch.Tensor]]:
    return [
        mx4_decode(payload, flat, table, outputs, rows, widths, group_size)
        for payload, flat, table, (outputs, rows, widths, group_size) in zip(
            payloads, flats, tables, specs
        )
    ]


def materialize_all(xs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Returns the embeddings the placeholders in `xs` stand for; other tensors
    pass through. Each payload is decoded with one kernel launch.

    The decode is marked must-recompute: a compiled graph that saves the results
    for backward saves the payload instead and decodes again in backward.
    """
    out = list(xs)
    # Per placeholder group: the group, its distinct outputs with the shape of
    # each, and for every position in `xs` the index of its output.
    groups: list[
        tuple[DeferredMX4, dict[int, int], list[tuple[int, int]], list[tuple[int, int]]]
    ] = []
    for i, x in enumerate(xs):
        found = deferred(x)
        if found is None:
            continue
        group, output = found
        entry = next((e for e in groups if e[0] is group), None)
        if entry is None:
            entry = (group, {}, [], [])
            groups.append(entry)
        _, index, shapes, positions = entry
        if output not in index:
            index[output] = len(shapes)
            shapes.append((x.shape[0], x.shape[1]))
        positions.append((i, index[output]))
    if not groups:
        return out
    decoded = checkpoint(
        _decode_groups,
        [g.payload for g, _, _, _ in groups],
        [g.flat for g, _, _, _ in groups],
        [g.table for g, _, _, _ in groups],
        [
            (
                list(index),
                [r for r, _ in shapes],
                [w for _, w in shapes],
                g.group_size,
            )
            for g, index, shapes, _ in groups
        ],
        use_reentrant=False,
        context_fn=functools.partial(
            create_selective_checkpoint_contexts, _recompute_all
        ),
    )
    for (_, _, _, positions), values in zip(groups, decoded):
        for i, k in positions:
            out[i] = values[k]
    return out
