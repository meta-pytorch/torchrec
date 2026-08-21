#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused pooled (gather + sum) TPU embedding lookup on the SparseCore.

`embedding_pooled_lookup_fn(indices, offsets, dev_weights, emb_dim)` gathers each
bag's rows and sum-pools them, returning the pooled `[B, emb_dim]` output. The pooling
reduction runs entirely inside a fused SparseCore Pallas kernel (adapted from the torus
`bench_tpu_gather_pool_pallas_sc` fused gather-sum kernel): it gathers `RBLK*K` rows
into VMEM via the `plsc.Indices` indirect gather, reduces over the pool dim in VMEM in
16-lane column slices, and writes only the pooled `[nbag, dim]` output -- the
`[nbag*pool, dim]` gather intermediate never touches HBM.

The fused kernel pools a FIXED number `K` of ids per bag. Jagged (offset-driven) bags
are supported by a host-side repack in `embedding_pooled_lookup_fn`: each bag is padded
to `K = round_up(max_bag_len, 8)` ids, and the pad slots point at an appended zero row
so they contribute nothing to the sum. This keeps the reduction on the SparseCore (which
cannot cheaply read per-bag offset scalars on the vector subcore) at the cost of padding
every bag to the batch's max length.

Forward only (no autograd backward yet).
"""

import functools

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
from torch_tpu._internal import pallas

# SparseCore vector lane width (NL) and the number of partial accumulators (NACC)
# used to break the pool-dim reduction into independent chains.
NL, NACC = 16, 16


def _rblk(K, D):
    """Rows-per-block: largest power-of-two (<=4) bag block whose double-buffered
    gather buffer stays under the ~500 KiB VMEM budget."""
    r = 1
    while 2 * (r * 2) * K * D * 4 <= 500 * 1024 and r * 2 <= 4:
        r *= 2
    return r


# ─── Fused SparseCore gather-sum kernel (from torus bench_tpu_gather_pool_pallas_sc) ──
def _kernel(idx_hbm, table_hbm, out_hbm, idx_v, buf, ov, sem, *, RBLK, K):
    sc = pltpu.get_tpu_info().sparse_core
    num_sub = sc.num_cores * sc.num_subcores
    B, D, TILE = out_hbm.shape[0], buf.shape[-1], RBLK * K
    nb = pl.cdiv(B, RBLK * num_sub)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(nb + 1, num_sub),
        core_axis_name=("c", "s"),
        dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
    )
    def _inner():
        blk, core = pl.program_id(0), pl.program_id(1)

        @pl.when(blk < nb)  # gather block blk
        def _():
            p = jax.lax.rem(blk, 2)
            o0 = (blk * num_sub + core) * RBLK
            pltpu.sync_copy(idx_hbm.at[pl.ds(o0 * K, TILE)], idx_v.at[p])
            pltpu.make_async_copy(
                table_hbm.at[plsc.Indices(idx_v.at[p])], buf.at[p], sem.at[p]
            ).start()

        @pl.when(blk > 0)  # reduce block blk-1 (overlaps gather)
        def _():
            q = jax.lax.rem(blk - 1, 2)
            o0 = ((blk - 1) * num_sub + core) * RBLK
            pltpu.make_async_copy(
                table_hbm.at[plsc.Indices(idx_v.at[q])], buf.at[q], sem.at[q]
            ).wait()
            for r in range(RBLK):
                base = r * K

                @plsc.parallel_loop(0, D, step=NL)
                def _cols(c0, base=base, r=r):  # bind loop vars (flake8 B023)
                    cs = pl.ds(c0, NL)
                    accs = [buf[q, base + a, cs] for a in range(NACC)]
                    for k in range(NACC, K):
                        accs[k % NACC] = accs[k % NACC] + buf[q, base + k, cs]
                    acc = accs[0]
                    for a in range(1, NACC):
                        acc = acc + accs[a]
                    ov[r, cs] = acc

            pltpu.sync_copy(ov, out_hbm.at[pl.ds(o0, RBLK)])

    _inner()


def build(B, K, D):
    """f(table[V, D] f32, idx[B*K] i32) -> out[B, D] f32 (fused gather + sum-pool).

    Returns the plain JAX function (not jitted) so ``jax_op`` can trace/export it.
    """
    RBLK = _rblk(K, D)
    sc = pltpu.get_tpu_info().sparse_core
    num_sub = sc.num_cores * sc.num_subcores
    Bpad = pl.cdiv(B, RBLK * num_sub) * RBLK * num_sub
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
    )
    ker = pl.kernel(
        functools.partial(_kernel, RBLK=RBLK, K=K),
        out_type=jax.ShapeDtypeStruct((Bpad, D), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,
            needs_layout_passes=False,
        ),
        scratch_types=[
            pltpu.VMEM((2, RBLK * K), jnp.int32),
            pltpu.VMEM((2, RBLK * K, D), jnp.float32),
            pltpu.VMEM((RBLK, D), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        mesh=mesh,
    )

    def run(table, idx):
        idxp = idx if Bpad == B else jnp.pad(idx, (0, (Bpad - B) * K))
        return ker(idxp, table)[:B]

    return run


def _pooled_gather_jax(idx: jax.Array, table: jax.Array, pool: int) -> jax.Array:
    """Fused uniform gather + sum-pool: idx[B*pool] i32, table[V, D] f32 -> [B, D]."""
    B = idx.shape[0] // pool
    D = table.shape[1]
    return build(B, pool, D)(table, idx)


_pooled_gather_fn = pallas.jax_op("pallas::embedding_pooled_gather", _pooled_gather_jax)


# ─── Backward: gradient w.r.t. the embedding table ───────────────────────────
# The forward is out[b] = sum_{k} table[idx[b*pool + k]], so the gradient scatter-adds
# each bag's grad back onto every row it gathered: grad_table[idx[j]] += grad_out[j//pool].
# Runs on the TensorCore (dense scatter-add via .at[].add), mirroring single_lookup.
def _pooled_gather_bwd_jax(
    grad_out: jax.Array, idx: jax.Array, pool: int, num_rows: int, emb_dim: int
) -> jax.Array:
    grad_expanded = jnp.repeat(
        grad_out, pool, axis=0
    )  # [B*pool, D]; row j -> grad_out[j//pool]
    return (
        jnp.zeros((num_rows, emb_dim), dtype=grad_out.dtype).at[idx].add(grad_expanded)
    )


_pooled_gather_bwd_fn = pallas.jax_op(
    "pallas::embedding_pooled_gather_bwd", _pooled_gather_bwd_jax
)


def _pooled_gather_backward(ctx, grad_out):
    (idx,) = ctx.saved_tensors
    grad_table = _pooled_gather_bwd_fn(
        grad_out=grad_out.contiguous(),
        idx=idx,
        pool=ctx.pool,
        num_rows=ctx.num_rows,
        emb_dim=ctx.emb_dim,
    )
    # One grad per forward input: (idx, table, pool).
    return None, grad_table, None


def _pooled_gather_setup_context(ctx, inputs, output):
    idx, table, pool = inputs
    ctx.save_for_backward(idx)
    ctx.pool = pool
    ctx.num_rows = table.shape[0]
    ctx.emb_dim = table.shape[1]


torch.library.register_autograd(
    "pallas::embedding_pooled_gather",
    _pooled_gather_backward,
    setup_context=_pooled_gather_setup_context,
)


def embedding_pooled_lookup_fn(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    dev_weights: torch.Tensor,
    emb_dim: int,
    pooling_mode: str = "sum",
) -> torch.Tensor:
    """Jagged sum/mean-pool via the fused SparseCore kernel (pad-to-max + zero-row sentinel).

    Differentiable w.r.t. ``dev_weights`` (the grad flows through the pad/cat back to the
    table). Any ``emb_dim`` is accepted -- it is padded up to a multiple of the SparseCore
    lane width internally.

    Args:
        indices: 1-D int32 flat ids for all bags, concatenated.
        offsets: 1-D int32 [B + 1] bag boundaries into ``indices``.
        dev_weights: [V, emb_dim] embedding table (on the TPU).
        emb_dim: embedding width.
        pooling_mode: "sum" or "mean".

    Returns:
        [B, emb_dim] pooled output.
    """
    device = dev_weights.device
    V, D = dev_weights.shape
    # The kernel reduces the dim in NL-wide lanes, so pad emb_dim up to a multiple of NL.
    D_pad = ((D + NL - 1) // NL) * NL

    # Repack on-device (no host round trip): only the pool size K needs a host value.
    off = offsets.to(device=device, dtype=torch.int64)
    lengths = off[1:] - off[:-1]
    B = int(lengths.numel())
    total = int(off[-1].item())  # one scalar sync
    max_len = int(lengths.max().item()) if B > 0 else 0  # one scalar sync
    # K rounded up to the SparseCore id tile (8) so RBLK*K stays tile-aligned, and
    # clamped to >= NACC: the kernel's reduction seeds NACC accumulators unconditionally,
    # so a smaller pool would read past the bag into the next bag's rows.
    K = max(((max_len + 7) // 8) * 8, NACC)

    # [B, K] index matrix, pad slots -> sentinel row V (an appended zero row -> adds 0).
    idx_mat = torch.full((B, K), V, dtype=torch.int32, device=device)
    if total > 0:
        sample_idx = torch.repeat_interleave(torch.arange(B, device=device), lengths)
        within = torch.arange(total, device=device) - off[:-1][sample_idx]
        idx_mat[sample_idx, within] = indices.to(device=device, dtype=torch.int32)
    idx_flat = idx_mat.reshape(-1)

    # Pad columns to D_pad and append a zero sentinel row; both extensions are zero so
    # they don't affect the pooled sum, and autograd carries the grad back to dev_weights.
    table = dev_weights
    if D_pad != D:
        table = torch.nn.functional.pad(table, (0, D_pad - D))
    zero_row = torch.zeros(1, D_pad, dtype=table.dtype, device=device)
    table_p = torch.cat([table, zero_row], dim=0)  # sentinel row index == V

    out = _pooled_gather_fn(idx=idx_flat, table=table_p, pool=K)[:, :D]  # [B, D]
    if pooling_mode == "mean":
        denom = lengths.clamp(min=1).unsqueeze(1).to(out.dtype)
        out = out / denom
    return out


if __name__ == "__main__":
    # Correctness check vs torch CPU embedding_bag. Run with:
    #   ./run_pod.sh run single_pooled_lookup.py
    import torch.distributed as dist
    import torch.nn.functional as F

    # pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
    import torch_tpu  # noqa: F401  (registers the "tpu" device + "tpu_dist" backend)

    # run_dist_file.sh launches one process per TPU. Set up the process group so all
    # ranks stay in lockstep; each rank runs the same check independently on its own
    # device (the kernel does no collectives), then barrier + destroy for a clean exit
    # (early-exiting non-zero ranks wedges the distributed runtime teardown -> hang).
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()

    torch.manual_seed(0)

    NUM_ROWS = 10000
    EMB_DIM = 32
    BATCH_SIZES = [1, 8, 64, 512]
    POOLING_AVG = 5  # average ids per bag

    weight_cpu = torch.randn(NUM_ROWS, EMB_DIM, dtype=torch.float32)
    weight_tpu = weight_cpu.to("tpu")

    for mode in ("sum", "mean"):
        for batch_size in BATCH_SIZES:
            lengths = torch.randint(1, 2 * POOLING_AVG, (batch_size,))
            total_ids = int(lengths.sum().item())
            indices = torch.randint(0, NUM_ROWS, (total_ids,), dtype=torch.int32)
            offsets = torch.cat(
                [torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)]
            )

            # --- forward ---
            ref_out = F.embedding_bag(
                input=indices.to(torch.int64),
                weight=weight_cpu,
                offsets=offsets[:-1].to(torch.int64),
                mode=mode,
            )
            tpu_out = embedding_pooled_lookup_fn(
                indices=indices.to("tpu"),
                offsets=offsets.to("tpu"),
                dev_weights=weight_tpu,
                emb_dim=EMB_DIM,
                pooling_mode=mode,
            ).to("cpu")
            fwd_err = (ref_out - tpu_out).abs().max().item()
            assert torch.allclose(
                ref_out, tpu_out, atol=1e-4
            ), f"{mode} fwd B={batch_size}: max abs diff = {fwd_err}"

            # --- backward (grad w.r.t. the table) ---
            grad_seed = torch.randn(batch_size, EMB_DIM, dtype=torch.float32)
            w_ref = weight_cpu.clone().requires_grad_(True)
            F.embedding_bag(
                input=indices.to(torch.int64),
                weight=w_ref,
                offsets=offsets[:-1].to(torch.int64),
                mode=mode,
            ).backward(grad_seed)
            w_dev = weight_cpu.clone().to("tpu")
            w_dev.requires_grad_(True)
            embedding_pooled_lookup_fn(
                indices=indices.to("tpu"),
                offsets=offsets.to("tpu"),
                dev_weights=w_dev,
                emb_dim=EMB_DIM,
                pooling_mode=mode,
            ).backward(grad_seed.to("tpu"))
            assert w_ref.grad is not None and w_dev.grad is not None
            bwd_err = (w_ref.grad - w_dev.grad.to("cpu")).abs().max().item()
            assert torch.allclose(
                w_ref.grad, w_dev.grad.to("cpu"), atol=1e-3
            ), f"{mode} bwd B={batch_size}: max abs diff = {bwd_err}"

            if rank == 0:
                print(
                    f"OK  {mode:>4}  B={batch_size:>4}  "
                    f"fwd_err={fwd_err:.2e}  bwd_err={bwd_err:.2e}",
                    flush=True,
                )

    if rank == 0:
        print("All pooled-lookup fwd+bwd correctness checks passed.", flush=True)

    dist.barrier()
    dist.destroy_process_group()
