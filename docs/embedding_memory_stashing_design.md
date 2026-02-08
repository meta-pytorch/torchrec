# Design: Embedding Memory Stashing (EMS)

## 1. Overview

The massive embedding tables used in modern RecSys models often strain GPU High Bandwidth Memory (HBM). While sharding is now common for embedding tables, models still frequently exceed single-device HBM capacity, with the activation and dense parts increasingly contributing to memory usage, limiting batch size or complexity.

To overcome this, the HBM-to-CPU Embedding Table Weights Stashing design proposes asynchronously offloading embedding weights from HBM to high-speed pinned CPU memory **after** the forward pass lookup, and restoring them before the backward pass (fused embedding grad compute and optimize). This temporarily frees substantial HBM for other operations (like intermediate activations), maximizing GPU utilization and enabling larger effective model sizes.

The goal is:
1. **After lookup**: Async copy weights from HBM (GPU) → CPU (pinned memory)
2. **Before lookup backward**: Restore weights from CPU → HBM
3. **Impact**: Reduce peak HBM memory without QPS regression

---

## 2. Background

### 2.1 Distributed Embedding in TorchRec

TorchRec provides distributed embedding infrastructure for RecSys. At its core, embedding operations in TorchRec are split into three phases:
1. **Input Distribution (input_dist)**: Sparse features (user IDs, item IDs, etc.) are redistributed across GPUs via all-to-all communication so each GPU receives the features it needs for its local embedding tables.
2. **Embedding Lookup**: Each GPU performs lookups on its local embedding tables. The embedding weights (`weights_dev`) are stored in GPU HBM and accessed by FBGEMM's Table Batched Embedding (TBE) kernels.
3. **Output Distribution (output_dist)**: The resulting embeddings are redistributed back via all-to-all so each GPU has the embeddings needed for its batch.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  input_dist │ ──► │   lookup    │ ──► │ output_dist │
│  (all2all)  │     │ (TBE kernel)│     │  (all2all)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    weights_dev (HBM)
```

### 2.2 Memory Characteristics

The key insight for EMS is that **embedding weights are only accessed at specific points** in the training loop:
- **Forward**: During the lookup phase (beginning of forward pass)
- **Backward**: During gradient computation and optimizer update (end of backward pass)

Between these points—during dense model forward/backward computation—the embedding weights sit idle in HBM while activations compete for memory. The HBM peak typically occurs at the forward→backward transition when activations are at maximum.

This access pattern creates an opportunity: **stash embedding weights to CPU during the idle window** to free HBM for activations, then restore before backward.

---

## 3. Why Now?

This Embedding Memory Stashing (EMS) approach only becomes practically feasible and highly effective with the advent of modern hardware architectures, especially NVIDIA's GB200/GB300 series, which provide significantly increased host-to-device (CPU-to-HBM) bandwidth, exceeding 400 GB/s. The high bandwidth is crucial because the embedding table weights must be stashed to CPU memory after the forward pass and restored before the backward pass.

Given that the embedding table is utilized only at the beginning of the model's forward computation and updated at the very end of the backward pass, there is a considerable time window (headroom) during the intermediate computations where this data transfer (stashing and restoring) can occur, leveraging the high-speed interconnect.

Furthermore, this method is especially beneficial because the HBM memory peak, driven by intermediate activations, typically occurs right at the transition from forward to backward; by offloading the weights during this critical period, EMS effectively reduces the memory requirements in the late forward and early backward phases, maximizing the available HBM for activations and enabling larger model complexity or batch sizes.

---

## 4. Preliminary Estimation

### 4.1 GB200 Key Memory and Interconnect Specifications
1. CPU-to-GPU Interconnect (NVLink-C2C): 900 GB/s (bidirectional)
2. CPU Memory Bandwidth (LPDDR5X): Up to 512 GB/s
3. GPU Memory Bandwidth (HBM3e): Up to 16 TB/s per Blackwell GPU
4. GPU-to-GPU Interconnect (NVLink 5): 1.8 TB/s per GPU

 {F1985330541}

### 4.2 Transfer Time Estimation
- Typical train cycle: 800 ms ~ 1,600 ms
- Assuming 60 ms for stashing/restoring with 450 GB/s NVLink-C2C
- **~27 GB HBM saving per rank** (theoretical)

---

## 5. Architecture

### 5.1 Training Loop with EMS

The diagram below shows how EMS integrates into the training loop. The key insight is that stash/restore operations run on a separate `memcpy_stream`, allowing them to overlap with compute on the default stream.

```
                              Forward Pass                         │             Backward Pass
                                                                   │
 ┌─────────┐   ┌────────┐   ┌───────────┐   ┌─────────────────┐    │    ┌─────────────────┐   ┌───────────┐   ┌───────────┐
 │ input   │──►│ lookup │──►│  output   │──►│  dense forward  │    │    │  dense backward │──►│  EBC BWD  │──►│ optimizer │
 │  dist   │   │  (TBE) │   │   dist    │   │  (MLP, etc.)    │────┼───►│                 │   │ + update  │   │   step    │
 └─────────┘   └────────┘   └───────────┘   └─────────────────┘    │    └─────────────────┘   └───────────┘   └───────────┘
                   │             │                                 │              │                 ▲
                   │             │                                 │              │                 │
                   ▼             │                                 │              ▼                 │
              weights_dev        │                                 │         ┌─────────┐            │
              accessed           │                                 │         │ restore │────────────┘
                                 │                                 │         │ (async) │
                                 ▼                                 │         └─────────┘
                           ┌──────────┐                            │              ▲
                           │  stash   │                            │              │
                           │ (async)  │                            │         CPU buffer
                           └──────────┘                            │         (pinned)
                                 │                                 │              │
                                 ▼                                                │
                            CPU buffer ───────────────────────────────────────────┘
                            (pinned)
```

### 5.2 Stash Phase (HBM → CPU)

Triggered immediately after embedding lookup completes:

1. **Allocate pinned CPU buffer**: Create a CPU tensor with `pin_memory=True` matching the shape of `weights_dev`. Pinned memory enables DMA transfers without CPU involvement.
2. **Async copy on memcpy stream**:
    1. `stream.wait_stream(current_stream)` ensures lookup is complete
    2. `cpu_buffer.copy_(weights_dev, non_blocking=True)` initiates async DMA transfer
    3. CPU thread returns immediately while GPU handles the copy
3. **Free HBM storage**: After copy completes, `weights_dev.untyped_storage().resize_(0)` releases GPU memory immediately, making it available for activations.

```python
# Simplified stash flow
cpu_buffer = torch.empty(weights_dev.shape, pin_memory=True)

with torch.cuda.stream(memcpy_stream):
    memcpy_stream.wait_stream(main_stream)  # Wait for lookup to complete
    cpu_buffer.copy_(weights_dev, non_blocking=True)
    stash_event.record()

# non-ideal implementation, see discussion below
stash_event.synchronize()  # Wait for copy to complete
weights_dev.untyped_storage().resize_(0)  # Free HBM
```

### 5.3 Restore Phase (CPU → HBM)

Triggered by autograd hook before embedding grad-dist (all2all_bwd):

1. **Re-allocate HBM storage**: `weights_dev.untyped_storage().resize_(storage_size)` allocates fresh GPU memory.
2. **Async copy back**:
    1. Create a temporary tensor viewing the same storage to avoid autograd version counter issues
    2. `tmp.copy_(cpu_buffer, non_blocking=True)` initiates async DMA transfer
3. **Stream synchronization**: `current_stream.wait_event(restore_event)` ensures backward doesn't start until restore completes.

```python
# Simplified restore flow
weights_dev.untyped_storage().resize_(storage_size)

with torch.cuda.stream(memcpy_stream):
    tmp = torch.tensor([], dtype=weights_dev.dtype, device=weights_dev.device)
    tmp.set_(weights_dev.untyped_storage(), 0, weights_dev.shape, weights_dev.stride())
    tmp.copy_(cpu_buffer, non_blocking=True)
    restore_event.record()

# non-ideal implementation, see discussion below
torch.cuda.current_stream().wait_event(restore_event)  # Wait for copy to complete before TBE bwd
```

### 5.4 Autograd Hook Integration

The restore function is registered as a backward hook on the output distribution tensor. This ensures restore is triggered at the right point in the backward pass:

```python
# In compute_and_output_dist:
restore_callback = stash_embedding_weights(lookup, memcpy_stream)
dist_awaitable._tensor_awaitable.dummy_tensor.register_hook(restore_callback)
```

When autograd traverses the computation graph during backward, it calls the hook when starting the backward all-to-all communication of the embedding gradient (grad_dist), triggering the restore so weights are ready for EBC backward.

### 5.5 Stream Management

CUDA operations on the same stream execute sequentially, while operations on different streams can execute concurrently. By using a dedicated `memcpy_stream` for stash/restore operations, we can overlap data transfers with compute operations on the default stream.
1. The data transfer happens "in the background" without blocking the GPU for dense computation, effectively hiding the transfer latency.
2. It occupies the same host-to-device IO resource which copy-batch-to-gpu also uses, so it's better to use the same stream, and the operations can run sequentially to avoid the overhead of context switching.
3. However, depending on how NVLink-C2C operates, it may be beneficial to use a separate stream for host-to-device transfers and device-to-host transfers if they are not interfering with each other.

Re-use/Freeing HBM memory is a tricky topic. CUDA caching allocator (CCA) runs on the host side, so it's not a GPU stream operation. The current implementation uses stash_event.synchronize() to block the CPU thread until the copy completes, which is a major performance gap in the prototype.
1. The CCA runs in ahead of the GPU stream, so it doesn't know if the stashing is done or not when the next operation comes to allocate memory.
2. If the CCA decides to reuse this particular HBM memory, the GPU stream will need to wait for the stashing stream.



**Timeline Diagram**

```
Time ────────────────────────────────────────────────────────────────────────────────────────────────►

                    Forward                                              Backward
    ┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐

Compute Stream:
    ┌────────┐                ┌─────────────────┐  ┌─────────────────┐                 ┌─────────┐
    │ lookup │                │  dense forward  │  │  dense backward │                 │ TBE BWD │
    └────────┘                └─────────────────┘  └─────────────────┘                 └─────────┘
             │                                                   |                        ▲
             │                                                   |                        │
D2D Comm     │                                                   |                        │
Stream:      │  ┌────────────┐                                   |   ┌────────────┐       │
             └─►│ output_dist│                                   └─► │  grad_dist │───────┤
                └────────────┘                                       └────────────┘       │
                                                                                          │
            |                                                    |                        │
H2D Memcpy  |                                                    |                        │
Stream:     |  ┌─────────────────┐                               |   ┌─────────────────┐  │
            └─►│  stash (D2H)    │                               └─► │  restore (H2D)  │──┘
               └─────────────────┘                                   └─────────────────┘
```

- **Stash** starts right after lookup ends
- **Restore** starts when grad_dist begins (triggered by autograd hook)
- **TBE BWD** waits for both grad_dist and restore to complete (whichever finishes later)

**Synchronization Points**

| Sync Point | Mechanism | Purpose |
|------------|-----------|---------|
| Lookup → Stash | `memcpy_stream.wait_stream(compute_stream)` | Ensure weights are fully written before copying to CPU |
| Stash → Free HBM | `stash_event.synchronize()` (CPU-blocking) | Ensure data is safely on CPU before freeing GPU memory |
| Restore → TBE BWD | `compute_stream.wait_event(restore_event)` | Ensure weights are restored before backward uses them |

---

## 6. Prototype & Benchmark Results

### 6.1 Implementation Summary

**Core API**: `stash_embedding_weights()` in `embeddingbag.py`

The function takes a lookup module and an optional CUDA stream, then returns a restore callback. Key implementation choices:

- **Pinned memory for CPU buffers**: Allocates CPU tensors with `pin_memory=True`, enabling async DMA transfers where the GPU can access CPU memory directly without intermediate staging buffers.
- **Async copy with dedicated stream**: Uses a separate `memcpy_stream` to perform HBM↔CPU transfers, allowing overlap with compute operations on the default stream.
- **`resize_(0)` for immediate HBM release**: After stashing completes, calls `weights_dev.untyped_storage().resize_(0)` to immediately free GPU memory while keeping the tensor metadata intact for later restoration.
- **Temporary tensor trick for restore**: During restore, creates a temporary tensor viewing the same storage to perform the copy. This avoids incrementing the original tensor's autograd version counter, which would otherwise cause "modified by in-place operation" errors during backward.

**Integration**: Modified `TrainPipelineSparseDist`

- Added `memcpy_stream` parameter for stash/restore operations
- **Autograd hook for restore trigger**: The restore callback is registered as a backward hook on the output distribution tensor (`dist_awaitable._tensor_awaitable.dummy_tensor.register_hook(restore)`). This ensures restore is triggered at exactly the right point—when grad_dist starts—giving enough time for weights to be ready before TBE backward.

**Diffs**:
- D92585742: Core stashing implementation [#3744]
- D92586272: Pipeline integration and benchmarks [#3745]

### 6.2 Benchmark Setup

The benchmark uses a basic sparse data distribution pipeline with a sparse NN model (embedding tables + MLP). This setup isolates the embedding memory behavior without interference from complex model architectures.

**Sharding Plan**
P2174913338

**Traces**

 {F1985330672}
 {F1985330674}

**Memory Snapshot**

 {F1985330678}
 {F1985330681}

**Run Script:**

```bash
buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_train_pipeline -- \
    --yaml_config=fbcode/torchrec/distributed/benchmark/yaml/sparse_data_dist_base.yml \
    --memory_snapshot=True \
    --name=sdd_embedding_stash
```

**Run Script (OSS):**
```bash
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
    --yaml_config=torchrec/distributed/benchmark/yaml/sparse_data_dist_base.yml \
    --memory_snapshot=True \
    --name=sdd_embedding_stash
  ```

### 6.3 Benchmark Results

| Metric | Baseline | With EMS | Delta |
|--------|----------|----------|-------|
| GPU Peak Mem Alloc | [43.28 GB](https://www.internalfb.com/pytorch_memory_visualizer/torchrec_benchmark_traces/tree/permanent_traces/DIFF/D92585742/memory-sparse_data_dist_base-rank0.pickle) | [31.97 GB](https://www.internalfb.com/pytorch_memory_visualizer/torchrec_benchmark_traces/tree/permanent_traces/DIFF/D92585742/memory-sdd_embedding_stash-rank0.pickle) | **-11.31 GB** |
| GPU Peak Mem Reserved | 66.51 GB | 53.89 GB | **-12.62 GB** |
| GPU Runtime | [14.5s](https://www.internalfb.com/intern/kernelhub/perfetto?trace_path=tree/permanent_traces/DIFF/D92585742/trace-sparse_data_dist_base-rank0.json.gz&bucket=torchrec_benchmark_traces) | [59.6s](https://www.internalfb.com/intern/kernelhub/perfetto?trace_path=tree/permanent_traces/DIFF/D92585742/trace-sdd_embedding_stash-rank0.json.gz&bucket=torchrec_benchmark_traces) | +45s (see gaps) |
| CPU Peak RSS | 30.66 GB | 45.51 GB | +14.85 GB (expected) |

**Key Observation**: Memory savings validated (~12GB), but significant runtime overhead in prototype due to slow host-to-device copy (~45s).

---

## 7. Gaps & Solutions

### 7.1 Timing of Freeing HBM

As discussed in Section 5.5, the current implementation uses `stash_event.synchronize()` to block the CPU thread until the D2H copy completes before freeing HBM, resulting in ~4x slowdown.

**Proposed Solutions**

1. **Deferred `resize_(0)` with stream synchronization**: Move the `resize_(0)` call to a later point in the forward pass (e.g., via a hook or FSDP integration). Before any operation that might reuse this HBM, insert `main_stream.wait(stash_event)` to ensure the D2H transfer has completed. This approach is simpler but requires careful placement of the sync point.
 {F1985328852}
2. **AsyncIO thread for completion monitoring**: Spawn a separate asyncIO thread that polls for stash completion. Once the D2H transfer finishes, the thread triggers `resize_(0)` to free HBM. This allows the main CPU thread to continue issuing GPU operations without blocking, maximizing overlap between compute and memory transfer.
 {F1985328853}

### 7.2 When to Trigger Restore

The current implementation triggers the restore callback at the start of `all2all_bwd` (grad_dist). However, on GB200/GB300 hardware, the all-to-all communication is significantly faster than the host-to-device copy—even accounting for potential straggler effects.

This timing mismatch means the restore may not complete before TBE backward needs the weights. To address this, we may need to trigger the restore earlier in the backward pass. Finding the optimal injection point in the autograd engine will require experimentation and tuning based on actual transfer and computation timings.

### 7.3 Functionality Gaps

| Gap | Current State | Solution |
|-----|---------------|----------|
| EC support | Only EBC supported | Extend `stash_embedding_weights` to support EmbeddingCollection |
| Sharding type coverage | Untested for all types | Verify and test all sharding types (CW, RW, TWRW, GRID, 2D, etc.) |
| TBE kernel variants | Unknown compatibility with EMO, SSD-TBE, etc. | Investigate interaction with different TBE kernel types; may require kernel-specific handling |
| Checkpointing compatibility | Unclear behavior during save/load | Verify stashed weights are properly restored before checkpointing; handle in-flight transfers |
| Multi-GPU coordination | Unknown behavior with FSDP/DDP | Test and document interaction with FSDP/DDP wrappers |

---

## 8. Other HBM Reduction Techniques

### 8.1 Embedding Memory Offloading (EMO)

EMO offloads embedding weights to CPU memory and fetches only the required rows during lookup. Unlike EMS, which temporarily stashes entire tables, EMO keeps weights on CPU permanently and only brings accessed rows to GPU.

- **Pros**: Significant HBM reduction; no need for frequent full-table transfers
- **Cons**: Not applicable to all tables (e.g., high-frequency tables); still consumes HBM for cached rows
- **Compatibility**: Theoretically compatible with EMS; requires verification

### 8.2 Activation Checkpointing and Offloading (AC/AO)

AC/AO reduces HBM usage by checkpointing or offloading activations during forward pass, then recomputing or reloading them during backward. Activations are often the dominant contributor to HBM peak.

- **Pros**: Targets activations, which are a major component of HBM peak
- **Cons**: Requires recomputation (AC) or additional data transfer (AO); more tuning needed; shorter idle window for offloading
- **Compatibility**: Theoretically compatible with EMS; AO may compete for host-device bandwidth during stash/restore

### 8.3 Fully Sharded Data Parallel (FSDP)

FSDP shards model parameters across GPUs and gathers them on-demand during forward/backward passes. It primarily targets dense layer weights rather than embeddings.

- **Pros**: Reduces HBM usage for dense layers by sharding weights across devices
- **Cons**: Requires careful tuning of prefetch timing to avoid stalls
- **Compatibility**: Theoretically compatible with EMS; both can operate on different parts of the model (FSDP on dense, EMS on embeddings)

---

## 9. Discussion

### 9.1 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Host-device bandwidth dependency | Slower transfers on older hardware | Reduce stash size; selectively stash large tables/modules |
| Pinned memory exhaustion | OOM on CPU side | Reduce stash size; selectively stash large tables/modules |
| Tuning complexity | Hard to onboard; slows production adoption | Provide tooling; integrate with TorchRec planner |

### 9.2 Open Questions

- What is the optimal sync strategy to minimize overhead while ensuring correctness?
- Where and how should we inject/hook the callbacks within a training step?
- How can the TorchRec planner help determine stashing configuration automatically?
- How do we ensure reliability and handle edge cases (e.g., failed transfers, timeouts)?
- What metrics and logging are needed for production debugging and monitoring?

---

## 10. Appendix

**Prototype Diffs**
- [D92585742](https://www.internalfb.com/diff/D92585742): Core `stash_embedding_weights` implementation
- [D92586272](https://www.internalfb.com/diff/D92586272): Pipeline integration and benchmarks

**References**
- [GB200 Key Memory and Interconnect Specifications](link)
- [TorchRec Train Pipeline Documentation](link)
- [FBGEMM TBE Documentation](link)

**Benchmark Artifacts**
- [Manifold folder](https://www.internalfb.com/manifold/explorer/torchrec_benchmark_traces/tree/permanent_traces/DIFF/D92585742)
