---
name: sharding-stats
description: Investigate and explain TorchRec planner sharding statistics output, especially how HBM storage is computed per table and per rank. Use when the user asks about sharding stats, storage breakdown, or memory estimation.
allowed-tools: Read, Grep, Bash, Task
argument-hint: [table name, sharding type, or question about stats]
---

# Sharding Stats Investigation Guide

Investigate and explain TorchRec planner sharding statistics, especially HBM storage computation, for: **$ARGUMENTS**

## Instructions

You are analyzing the output of `EmbeddingStats` (in `torchrec/distributed/planner/stats.py`). This skill covers how to read the stats table and how each number is computed from source code.

## Key Source Files

1. **`torchrec/distributed/planner/stats.py`** — Generates the bordered stats table output (`EmbeddingStats` class)
2. **`torchrec/distributed/planner/shard_estimators.py`** — Core estimation logic:
   - `EmbeddingStorageEstimator` — orchestrates storage estimation
   - `calculate_shard_storages()` — assembles final `Storage` per shard
   - `_calculate_shard_io_sizes()` — dispatches to sharding-type-specific I/O calculations
   - `_calculate_rw_shard_io_sizes()`, `_calculate_tw_shard_io_sizes()`, etc.
   - `_calculate_storage_specific_sizes()` — tensor + optimizer + cache aux
   - `_calculate_tensor_sizes()` — proportional tensor size per shard
   - `_calculate_optimizer_sizes()` — optimizer state multiplier
   - `calculate_pipeline_io_cost()` — pipeline type I/O multipliers
3. **`torchrec/distributed/planner/storage_reservations.py`** — Dense storage and KJT storage reservation
4. **`torchrec/distributed/planner/types.py`** — `Perf`, `Storage`, `ShardingOption` dataclasses
5. **`torchrec/distributed/planner/utils.py`** — `bytes_to_gb` (1 GB = 2^30 bytes), `bytes_to_mb`
6. **`torchrec/distributed/embedding_types.py`** — Sharder `storage_usage()` implementations

## Stats Table Structure

The stats output has these sections:

1. **Per-Rank Summary**: Rank, HBM (GB), DDR (GB), Perf (ms), Input (MB), Output (MB), Shards
2. **Parameter Info Table**: Per-table details (FQN, Sharding, Compute Kernel, Perf, Storage, etc.)
3. **Batch Size & Compute Kernels**: Global batch size, kernel counts and storage
4. **Imbalance Statistics**: Total Variation, Total Distance, Chi Divergence, KL Divergence
5. **Peak Memory Estimation**: Top-tier HBM pressure per rank
6. **Storage Reservation**: Reserved, Planning, Dense, KJT storage

## HBM Storage Computation — Step by Step

### Per-Rank HBM Formula

```
used_hbm[rank] = sparse_hbm[rank] + dense_storage.hbm + kjt_storage.hbm
```

- `sparse_hbm[rank]` = sum of `shard.storage.hbm` for every embedding shard placed on that rank
- `dense_storage` = non-embedding model parameters (from `HeuristicalStorageReservation`)
- `kjt_storage` = KeyedJaggedTensor input buffers

### Per-Shard HBM Formula (the core calculation)

Each shard's HBM is computed in `calculate_shard_storages()`:

```
shard.storage.hbm = hbm_specific_size + pipeline_io_cost
```

Where:

```
hbm_specific_size = tensor_size + optimizer_size + cache_aux_size
```

### Step 1: Base Tensor Storage (`sharder.storage_usage()`)

The sharder determines the raw tensor bytes:

- **EmbeddingBagCollectionSharder**: `hbm_storage = num_embeddings × emb_dim × element_size`
- **EmbeddingCollectionSharder**: `hbm_storage = num_embeddings × emb_dim × element_size + num_embeddings × 4`
  - The extra `shape[0] × 4` bytes is metadata overhead for sequence embeddings
- For UVM caching kernels: tensor goes to DDR, HBM gets `ddr_storage × caching_ratio`

### Step 2: Per-Shard Tensor Size (`_calculate_tensor_sizes()`)

```python
tensor_size = ceil(hbm_storage × prod(shard_size) / prod(full_shape))
```

For RW sharding with `world_size` shards: `shard_size = [num_embeddings / world_size, emb_dim]`

### Step 3: Optimizer Size (`_calculate_optimizer_sizes()`)

```python
optimizer_size = ceil(tensor_size × optimizer_multiplier)
```

| Optimizer | Multiplier |
|---|---|
| SGD | 0 |
| Adam | 2 |
| RowWiseAdagrad | 1 / emb_dim |
| Default/unknown | 1 |
| None (inference) | 0 |

### Step 4: Cache Auxiliary State (`_calculate_cache_aux_state_sizes()`)

Only applies to UVM caching (`fused_uvm_caching` kernel). For `fused` kernel: **0**.

### Step 5: I/O Sizes (sharding-type-specific)

Computed by `_calculate_shard_io_sizes()` → dispatches to type-specific functions.

**Constants:**
- `input_data_type_size = 8` (BIGINT_DTYPE, int64 indices)
- `output_data_type_size = tensor.element_size()` (or `output_dtype` if specified)

**For RW sharding** (`_calculate_rw_shard_io_sizes()`):

```python
batch_inputs = sum(input_length_i × num_poolings_i × batch_size_i) / world_size
batch_outputs = batch_inputs  # if non-pooled (sequence)
              = sum(num_poolings_i × batch_size_i)  # if pooled

input_size  = ceil(batch_inputs × world_size × input_data_type_size)    # per shard
output_size = ceil(batch_outputs × world_size × shard_dim × output_data_type_size)  # per shard
```

**For TW sharding** (`_calculate_tw_shard_io_sizes()`):

```python
batch_inputs = sum(input_length_i × num_poolings_i × batch_size_i)  # no division
input_size  = ceil(batch_inputs × world_size × input_data_type_size)
output_size = ceil(batch_outputs × world_size × emb_dim × output_data_type_size)
```

**For CW sharding** (`_calculate_cw_shard_io_sizes()`):

```python
# Same as TW but output uses shard_sizes[i][1] (shard column dim) instead of full emb_dim
output_size = ceil(batch_outputs × world_size × shard_sizes[i][1] × output_data_type_size)
```

**Critical insight for sequence (non-pooled) embeddings:**
When `is_pooled=False`, `batch_outputs = batch_inputs`, which means output includes one full embedding vector per input index. With large input_lengths, the output buffer can be **enormous** — often 90%+ of total storage.

### Step 6: Pipeline I/O Cost (`calculate_pipeline_io_cost()`)

```python
output_contribution = output_size if count_ephemeral_storage_cost else 0
```

| Pipeline Type | Formula |
|---|---|
| `NONE` (catch-all) | `input_size + output_size` |
| `TRAIN_SPARSE_DIST` | `2 × input_size + output_contribution` |
| `TRAIN_PREFETCH_SPARSE_DIST` | `3 × input_size + (1 + 6/max_pass) × prefetch_size + output_contribution` |
| Inference | `0` |

- `prefetch_size = input_size` if table is cached, else `0`
- `count_ephemeral_storage_cost` defaults to `False`

### Step 7: Final Per-Shard Storage

```python
shard.storage.hbm = hbm_specific_size + pipeline_io_cost
```

### Step 8: Total Storage (shown in Parameter Info table)

The "Storage (HBM, DDR)" column in the parameter info table shows:

```python
total_storage = sum(shard.storage for shard in sharding_option.shards)
```

This is the sum across ALL shards (all ranks), not per-rank.

## Dense Storage Reservation

From `HeuristicalStorageReservation`:

```
dense_storage.hbm = (total_model_params - embedding_params) × multiplier + buffers
```

- Training multiplier = 6.0 (1× params + 2× optimizer state + 3× DDP gradient buffers)
- Inference multiplier = 1.0

## KJT Storage Reservation

```
kjt_storage.hbm = total_kjt_size × kjt_multiplier
```

- Training multiplier = 20 (pipelined batches)
- Inference multiplier = 1

## Investigation Checklist

When analyzing a table's storage, determine these parameters:

1. **Tensor dtype**: float32 (4 bytes) or float16/bfloat16 (2 bytes)? Cross-check with per-rank Output (MB) column.
2. **Sharder type**: EmbeddingBagCollectionSharder vs EmbeddingCollectionSharder (EC adds `shape[0] × 4`)
3. **is_pooled**: pooled (EmbeddingBag) vs sequence (Embedding) — check "Output" column
4. **Optimizer**: check `_optimizer_classes` attribute on tensor; RowWiseAdagrad is common for RecSys
5. **Pipeline type**: NONE, TRAIN_SPARSE_DIST, or TRAIN_PREFETCH_SPARSE_DIST
6. **count_ephemeral_storage_cost**: defaults to False
7. **Compute kernel**: fused, fused_uvm_caching, quant, etc.
8. **Caching ratio**: only matters for UVM caching kernels

## Worked Example: RW Sequence Embedding

Given: hash_size=80M, emb_dim=128, dtype=fp16, 4 features, sum(input_lengths)=6066, batch_size=2560, world_size=96, RW sharding, fused kernel, RowWiseAdagrad, PipelineType.NONE

1. **Tensor storage**: `80M × 128 × 2 = 20,480,000,000 bytes` (19.07 GB)
2. **Per-shard tensor**: `ceil(20,480,000,000 × 833,333 / 80,000,000) = 213,333,248 bytes`
3. **Optimizer**: `ceil(213,333,248 / 128) = 1,666,666 bytes` (~1.6 MB)
4. **hbm_specific**: `213,333,248 + 1,666,666 = 214,999,914 bytes` (~205 MB)
5. **I/O (RW, non-pooled)**:
   - `batch_inputs = 2560 × 6066 / 96 = 161,760`
   - `input_size = ceil(161,760 × 96 × 8) = 124,231,680` (~118 MB)
   - `output_size = ceil(161,760 × 96 × 128 × 2) = 3,975,413,760` (~3.70 GB)
6. **Pipeline (NONE)**: `124,231,680 + 3,975,413,760 = 4,099,645,440` (~3.82 GB)
7. **Per-shard total**: `214,999,914 + 4,099,645,440 = 4,314,645,354` (~4.02 GB)
8. **Total (96 shards)**: `~385.8 GB`

**Key insight**: Output buffer is ~92% of total — sequence embeddings with large input_lengths dominate storage.

| Component | Per Shard | Total (96 shards) | % |
|---|---|---|---|
| Embedding weights | 203 MB | 19.07 GB | 4.9% |
| Optimizer (RowWiseAdagrad) | 1.6 MB | 0.15 GB | 0.04% |
| Input buffer (int64) | 118 MB | 11.1 GB | 2.9% |
| Output buffer (fp16) | 3.70 GB | 355.5 GB | 92.1% |
| **Total** | **~4.02 GB** | **~385.8 GB** | |

## Example Stats Output

See [`sharding_stats_example.txt`](https://github.com/meta-pytorch/torchrec/blob/main/docs/wiki/sharding_stats_example.txt) for a full example.
