# Chunked Fully Sharded Triton TBE

## Summary

`ChunkedShardedTritonBatchedFusedEmbeddingBag` combines the allocator-friendly
storage of `ChunkedTritonTableBatchedEmbeddingBags` with the Fully Sharded 2D
weight lifecycle. The logical flattened table layout is unchanged, while the
full weight is held in a fixed number of persistent, disjoint CUDA tensors.

The class runs one reduce-scatter and all-gather per chunk at the same points as
`ShardedBatchedFusedEmbeddingBag`. It never constructs a contiguous flattened
weight tensor. Row-aligned chunk boundaries provide near-equal allocations and
allow table state to be represented as zero-copy fragments.

## Weight lifecycle

1. Initialization allocates the full weight as balanced row-aligned chunks and
   initializes each table through views into those chunks.
2. Forward passes the full chunks directly to the chunk-aware Triton kernels.
3. Immediately after forward, a dedicated stream launches an asynchronous
   `AVG` reduce-scatter for every chunk. A composite awaitable releases the
   full-chunk storages only after all collectives complete.
4. The full-backward pre-hook resizes each original chunk storage and
   synchronously all-gathers its shard. Fused Triton backward then updates the
   reconstructed chunks in place.

The full chunk Tensor objects are never replaced. Triton autograd saves those
exact objects during forward, so all-gather must restore their storages rather
than install newly allocated tensors. Collective output is written through a
fresh storage alias to avoid changing the saved Tensor version counter.

## TorchRec integration

Each logical table is exposed as a `LocalShardsWrapper` containing row fragments
from one or more chunks. These wrappers support initialization, state-dict load
and save, named parameters, and `TritonEmbeddingFusedOptimizer` without copying
the table into contiguous storage. DTensor state construction flattens nested
fragments, and load copies overlapping logical regions. This structural support
does not establish production checkpoint-resume safety for every
row-fragmented physical layout. D117131202 later observed discontinuous NE with
a fixed 16-chunk layout and mitigated it with table-aware boundaries. Until that
mitigation is ported, production checkpoint use must remain limited to
explicitly validated layouts. Row-wise Adagrad state remains contiguous; only
table weights are chunked.

`FUSED_TRITON` selects the chunked sharded class only when the environment uses
`ShardingStrategy.FULLY_SHARDED`. DMP caches the persistent chunks and optimizer
state so explicit and periodic `sync()` calls preserve the contiguous path's
semantics. Fully Sharded reduce-scatter still averages weights after each
training lookup.

`VariableLengthEmbeddingArchSharder` advertises `FUSED_TRITON` support, so VLE
lookups use the same kernel selection path as EBC and
FeatureProcessedEmbeddingBagCollection. Multi-kernel variable-batch VLE inputs
use the concat-based merge because Triton TBE returns its own output instead of
writing into the Split/SSD preallocated VBE buffer. VLE also waits for Triton
forward completion before starting output-distribution collectives.

This enablement does not include Triton FP8 row-scale, QAT, or quantized
optimizer plumbing. Existing logical checkpoint integration is retained with
the row-fragment safety limitation described above. VLE uses the precision and
optimizer configured through the existing non-FP8 TBE path.

## Configuration gating

There is no JustKnobs gate for the chunked implementation. A table uses
`ChunkedShardedTritonBatchedFusedEmbeddingBag` only when all of these conditions
hold:

1. Its planned compute kernel is `FUSED_TRITON`.
2. The lookup receives a `ShardingEnv2D` whose strategy is `FULLY_SHARDED`.
3. The table is in HBM, is not data parallel, and its module sharder supports
   `FUSED_TRITON`.

The wrapper reads `num_weight_chunks` from the grouped embedding fused
parameters and defaults to four. APF plumbing for
`training.tbe.num_weight_chunks` is intentionally isolated in the follow-up
configuration diff. The value controls chunk count only after the chunked class
has been selected; it does not select Triton or change the sharding strategy.

The optimized Triton TBE paths from the current stack remain enabled for tables
that use `FUSED_TRITON`; this graft does not restore the historical opt-in
optimization gate. The
`training.sharding.two_dim_sparse_parallelism.rs_awaitable_hook_module` setting
controls when completed reduce-scatter storage is released and is not a module
allowlist.

To request `FUSED_TRITON` for every eligible table, use the global handle:

```yaml
training:
  tbe:
    compute_kernel: FUSED_TRITON
    num_weight_chunks: 4
  sharding:
    two_dim_sparse_parallelism:
      sharding_strategy: FULLY_SHARDED
```

Table or unified-group `compute_kernel` overrides take precedence over
`training.tbe.compute_kernel`, so conflicting overrides must be removed or
changed to `FUSED_TRITON`. The planner currently receives both `FUSED_TRITON`
and `FUSED` as candidates and may retain `FUSED` when Triton is unsupported;
the generated sharding plan must be checked to confirm coverage.

With VLE Triton support, this configuration covers eligible EBC,
FeatureProcessedEmbeddingBagCollection, and VariableLengthEmbeddingArch tables.
The verified global APS configuration selected chunked lookups for
`sparse_arch.ebc`, `sparse_arch.position_ebc`,
`sparse_arch.score_bucketize_ebc`, and all nine event-model
`vl_embedding_arch` modules present in the model. HBM and non-data-parallel
eligibility restrictions still apply, so “all tables” means all tables
supported by the Triton planner rather than UVM or data-parallel tables.

## Trace-driven synchronization reconciliation

The first current-stack trace, pipeline `1087356707079053`, found one
`cudaStreamSynchronize` on ranks 0 through 3 and two on ranks 4 through 7 under
`TritonTBEBackward`. Dimension-bucket dispatch constructed a CUDA tensor from a
host list during every backward pass, producing an implicit host-to-device copy
and synchronization.

The reconciled implementation caches invariant bucket row capacities directly
on the target device during module initialization. The existing Triton run
classification kernel computes the capped prefix bases, so dynamic batch sizes
remain correct without an additional launch, allocation, or host-device copy.
The post-fix eight-B200 trace has zero `cudaStreamSynchronize` and zero
`cudaDeviceSynchronize` calls nested under the chunked lookup, `TritonTBE`, or
`TritonTBEBackward` on every rank.

## Initial constraints and risks

- NVIDIA CUDA only; AMD remains unsupported.
- `bag_size_hints` histogram kernels remain disabled because they access the
  contiguous weight pointer directly.
- Only one training forward may be outstanding for a module. A second forward
  before backward is rejected instead of retaining another complete chunk set.
- Training-mode `no_grad` forwards still follow the Fully Sharded lifecycle;
  callers should use evaluation mode or explicitly materialize weights before
  another forward.
- Chunk planning and collective order must be identical on every replica rank.
- Repeated shards of one table within a grouped kernel must use the same local
  column width, matching the existing fused TBE optimizer-parameter layout.
- Padding is computed independently per chunk; row-aligned dimensions make it
  negligible for the initial two-replica APS configuration.
- Initialization preserves each table's requested distribution but not the
  bitwise random-number sequence of contiguous initialization.
- Logical checkpoint keys and fragment-copy support do not make every fixed
  row-fragmented layout production-resume safe. The table-aware boundary
  mitigation from D117131202 is not part of this graft.

## Verification

Two-rank tests compare forward output, fused updates, optimizer state, and the
next lookup against `ShardedBatchedFusedEmbeddingBag`. Direct B200 coverage also
includes collective padding, Fully Sharded row-wise and column-wise layouts,
dynamic dimension buckets, weighted VBE gradients, and pointer refresh.

The consolidated CPU/config regression run passed 137 tests with three known
skips and no failures:

- [Wrapper, checkpoint, lookup, DMP, APF forwarding, planner, and grouping tests](https://www.internalfb.com/intern/testinfra/testrun/3377700110322308)

The final local APS run used eight B200 GPUs, global `FUSED_TRITON`, 2D
`FULLY_SHARDED`, and `num_weight_chunks=4`. It completed 50 batches and 3,200
samples, and every worker shut down cleanly:

- [APS pipeline](https://www.internalfb.com/mlhub/pipeline/4520746088166486)
- Every rank contains 12 chunked lookups, 48 reduce-scatter annotations, 48
  all-gather annotations, 12 `TritonTBE` ranges, and 12 `TritonTBEBackward`
  ranges.
- Every lookup contains exactly four reduce-scatters. All 48 all-gathers are
  under the backward hook, and all 12 `TritonTBE` ranges are directly under a
  chunked lookup.
- Every rank contains 48 c10d and 48 NCCL instances of each collective.
- Chunked forward kernels executed on all ranks, with counts
  `26, 26, 30, 30, 26, 26, 34, 34` for ranks 0 through 7.
- Every rank has zero `cudaStreamSynchronize` and zero `cudaDeviceSynchronize`
  calls nested under chunked lookup, `TritonTBE`, or `TritonTBEBackward`.

Final APS GPU traces:

- [Rank 0](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-0.Sep_03_18_22_04.2196713.pt.trace.json.gz&bucket=aps_traces)
- [Rank 1](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-1.Sep_03_18_22_04.2196818.pt.trace.json.gz&bucket=aps_traces)
- [Rank 2](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-2.Sep_03_18_22_04.2196811.pt.trace.json.gz&bucket=aps_traces)
- [Rank 3](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-3.Sep_03_18_22_04.2196788.pt.trace.json.gz&bucket=aps_traces)
- [Rank 4](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-4.Sep_03_18_22_04.2196748.pt.trace.json.gz&bucket=aps_traces)
- [Rank 5](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-5.Sep_03_18_22_04.2196801.pt.trace.json.gz&bucket=aps_traces)
- [Rank 6](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-6.Sep_03_18_22_04.2196725.pt.trace.json.gz&bucket=aps_traces)
- [Rank 7](https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/dynocli/devgpu004.kcm1.facebook.com/rank-7.Sep_03_18_22_04.2196762.pt.trace.json.gz&bucket=aps_traces)
