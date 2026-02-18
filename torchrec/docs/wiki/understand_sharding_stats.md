# 1. Overview

This document explains how to read and interpret the output of `EmbeddingStats` (defined in `torchrec/distributed/planner/stats.py`), which is logged after the TorchRec sharding planner produces a sharding plan.

The stats output is a bordered text table logged to the Python logger. It contains the following sections:

1. A **header** with planner run metadata
2. A **per-rank summary table** showing memory, perf, I/O, and shard counts per GPU
3. A **per-table parameter info table** showing detailed per-embedding-table statistics
4. **Compute kernel summaries**
5. **Imbalance statistics** measuring how evenly work/memory is distributed
6. **Max perf / max HBM analysis** identifying bottleneck ranks
7. **Critical path estimates**
8. **Peak memory pressure tiers**
9. **Storage reservation breakdown**
10. **Top tables** causing the most perf or HBM pressure

## 1.1 Example Output

For a full example of the stats output, see [`sharding_stats_example.txt`](https://github.com/meta-pytorch/torchrec/blob/main/docs/wiki/sharding_stats_example.txt).

> **Tip:** Select "Unwrap" in the Paste tool to view the output without line wrapping, which makes the wide tables much easier to read.

---

# 2. Header

```
--- Planner Statistics ---
--- Evaluated 0 proposal(s), found 0 possible plan(s), ran for 48.19s ---
```

| Field | Meaning |
|-------|---------|
| **Evaluated N proposal(s)** | Number of sharding proposals the planner enumerated and evaluated. |
| **found N possible plan(s)** | Number of those proposals that successfully passed partitioning (fit within device memory constraints). |
| **ran for Xs** | Wall-clock time the planner took to find the plan. |

> Note: "Evaluated 0 proposal(s), found 0 possible plan(s)" can appear when the plan was provided via constraints rather than searched.

---

# 3. Per-Rank Summary Table

```
Rank   HBM (GB)    DDR (GB)              Perf (ms)    Input (MB)    Output (MB)        Shards
   0  99.737 (72%)   0.0 (0%)  1026.65 (66,408,132,420,0)    1683.06      10438.85    CW: 3 RW: 2
```

## 3.1 Columns

| Column | Unit | How It Is Calculated |
|--------|------|----------------------|
| **Rank** | — | GPU rank index (0 to world_size - 1). |
| **HBM (GB)** | GB (%) | Total estimated HBM usage for this rank. Calculated as: `sparse_hbm[rank] + dense_storage.hbm + kjt_storage.hbm`. The percentage is relative to the *planning memory* (total HBM minus reserved HBM): `used_hbm / ((1 - reserved_hbm_percent) * device_total_hbm)`. |
| **DDR (GB)** | GB (%) | Total estimated DDR (host memory) usage. Calculated as: `used_ddr[rank] + dense_storage.ddr + kjt_storage.ddr`. Percentage is relative to total DDR capacity. |
| **Perf (ms)** | ms | Estimated total perf cost for this rank per iteration, followed by a breakdown: `Total (Fwd Compute, Fwd Comms, Bwd Compute, Bwd Comms, Prefetch Compute)`. See Section 3.3. |
| **Input (MB)** | MB/iteration | Total input data size flowing into all shards on this rank per iteration. |
| **Output (MB)** | MB/iteration | Total output data size produced by all shards on this rank per iteration. |
| **Shards** | count | Number of embedding table shards on this rank, grouped by sharding type abbreviation. |

## 3.2 Sharding Type Abbreviations

| Abbreviation | Full Name | Description |
|---|---|---|
| **DP** | `DATA_PARALLEL` | Entire table replicated on every rank |
| **TW** | `TABLE_WISE` | Entire table placed on one rank |
| **RW** | `ROW_WISE` | Table rows split across all ranks |
| **CW** | `COLUMN_WISE` | Table columns (embedding dim) split across ranks |
| **TWRW** | `TABLE_ROW_WISE` | Row-wise within a node, table-wise across nodes |
| **TWCW** | `TABLE_COLUMN_WISE` | Column-wise within a node, table-wise across nodes |
| **GS** | `GRID_SHARD` | 2D grid sharding (rows and columns) |

## 3.3 Perf Breakdown

The perf value is displayed as: `Total (FwdComp, FwdComms, BwdComp, BwdComms, PrefetchComp)`

Each component is the sum across all shards placed on that rank:

```
Total = Fwd Compute + Fwd Comms + Bwd Compute + Bwd Comms + Prefetch Compute
```

1. **Fwd Compute**: Estimated forward-pass compute time (embedding lookups, pooling).
2. **Fwd Comms**: Estimated forward-pass communication time (all-to-all, all-reduce, etc.).
3. **Bwd Compute**: Estimated backward-pass compute time (gradient computation).
4. **Bwd Comms**: Estimated backward-pass communication time.
5. **Prefetch Compute**: Estimated prefetch compute time (used with UVM caching / embedding offload). Often 0 when not using UVM.

Values less than 1 ms are shown with one significant figure (e.g., `0.3`); values >= 1 ms are shown rounded to the nearest integer.

> **Note:** `input_dist_comms` is also tracked in the `Perf` dataclass but is excluded from the `total` property and not shown in the stats output.

## 3.4 HBM Calculation Detail

For each rank, the HBM usage is the sum of three components:

```python
used_hbm[rank] = sparse_hbm[rank] + dense_storage.hbm + kjt_storage.hbm
```

**Sparse HBM** (`sparse_hbm[rank]`): Sum of `shard.storage.hbm` for every embedding shard placed on that rank. This varies per rank depending on which tables/shards are assigned to it. Each shard's storage is a composite of the following components:

1. **Embedding table weights**: Proportional shard of `hash_size * emb_dim * dtype_size`.
2. **Optimizer state** (training only): A multiple of the weight size depending on optimizer type (e.g., Adam adds 2x for first and second moment, RowWiseAdagrad adds 1/emb_dim per row, SGD adds 0).
3. **Cache auxiliary state** (UVM caching only): Index maps, LRU timestamps, and cache metadata.
4. **Input (KJT) buffer**: Size of the sparse index input for this shard. This is data-dependent, computed as `sum(input_length * num_poolings * batch_size) * input_data_type_size` (varies by sharding type).
5. **Output (embedding) buffer**: Size of the embedding output for this shard. Also data-dependent, computed as `sum(num_poolings * batch_size) * emb_dim * output_data_type_size` (varies by sharding type).
6. **Pipeline multiplier**: The I/O buffers above are further multiplied depending on the pipeline type — e.g., `TrainPipelineSparseDist` uses 2x input (double-buffered), `TrainPrefetchSparseDist` uses 3x input plus prefetch buffers.

**Dense Storage HBM** (`dense_storage.hbm`): Estimated memory for the non-embedding (dense) parts of the model (e.g., MLP layers, dense linear layers, batch norms). This is **uniform across all ranks**. It is computed by `HeuristicalStorageReservation` as:

```
dense_storage = (total_model_params - embedding_params) * multiplier + buffers
```

1. **`multiplier`** defaults to **6.0** for training, accounting for: the parameter itself (1x) + optimizer state (~2x for Adam momentum and variance) + DDP gradient copies (~3x) = 6x the raw parameter size.
2. **`multiplier`** is **1.0** for inference (`InferenceStorageReservation`), since there is no optimizer state or gradient replication.
3. **Buffers** (e.g., running stats in batch norm) are added at 1x — they are not multiplied since they have no optimizer state.
4. The `dense_tensor_estimate` constructor parameter can override this calculation with a user-provided value.
5. When using `FixedPercentageStorageReservation` (common with FSDP), dense storage is **not computed separately** — instead, a flat percentage of total HBM is reserved.

**KJT Storage HBM** (`kjt_storage.hbm`): Estimated memory for KeyedJaggedTensor (sparse feature) input tensors on device. Also uniform across ranks. For training, a multiplier of 20 is applied to account for pipelined batches; for inference the multiplier is 1.

## 3.5 Input/Output Size Calculation

Input and output sizes are computed per shard by `_calculate_shard_io_sizes()`, which dispatches to sharding-type-specific formulas. The general pattern:

1. **Input size** = `sum(input_length * num_poolings * batch_size) * input_data_type_size` (in bytes, then converted to MB)
   1. `input_data_type_size` is `BIGINT_DTYPE` (8 bytes for int64 indices)
2. **Output size** = `sum(num_poolings * batch_size) * emb_dim * output_data_type_size` (for pooled embeddings)
   1. `output_data_type_size` is the element size of the embedding tensor (typically 4 bytes for float32)

The exact formula varies by sharding type (e.g., CW shards use `shard_sizes` for the column dimension, RW considers the full embedding dim, etc.).

---

# 4. Parameter Info Table

```
FQN  Sharding  Compute Kernel  Perf (ms)  Storage (HBM, DDR)  Cache Load Factor  Sum Pooling Factor  Sum Num Poolings  Num Indices  Output  Weighted  Sharder  Features  Emb Dim (CW Dim)  Hash Size  Ranks  Batch Sizes
```

This table provides per-embedding-table details for every table in the sharding plan.

| Column | Description |
|--------|-------------|
| **FQN** | Fully qualified name of the embedding table in the model (e.g., `inner_model.sparse_arch.ebc.table_XXX`). |
| **Sharding** | Sharding type abbreviation (CW, RW, TW, etc.). |
| **Compute Kernel** | The embedding compute kernel used (e.g., `fused`, `fused_uvm_caching`, `quant`). |
| **Perf (ms)** | Estimated perf cost across all shards of this table: `Total (FwdComp, FwdComms, BwdComp, BwdComms, PrefetchComp)`. Summed over all shards of the table. |
| **Storage (HBM, DDR)** | Total storage across all shards: `(HBM in GB, DDR in GB)`. |
| **Cache Load Factor** | Only shown for `fused_uvm_caching` kernel. The fraction of the embedding table cached in HBM. `None` for non-UVM kernels. |
| **Sum Pooling Factor** | `sum(input_lengths)` — total expected pooling factor across all features in this table. Represents the average number of lookups per sample per feature, summed over features. |
| **Sum Num Poolings** | `sum(num_poolings)` — total number of pooling operations. For most tables this equals the number of features. |
| **Num Indices** | `sum(input_length_i * num_poolings_i)` — total number of embedding index lookups per sample across all features. |
| **Output** | `pooled` or `sequence`. Pooled means the output is reduced (sum/mean) over the variable-length input; sequence means the full sequence of embeddings is returned. |
| **Weighted** | `weighted` or `unweighted`. Whether the embedding lookup uses per-index weights. |
| **Sharder** | The sharder class name (e.g., `EmbeddingBagCollectionSharder`, `EmbeddingCollectionSharder`). |
| **Features** | Number of features served by this table. |
| **Emb Dim (CW Dim)** | Full embedding dimension. For CW/TWCW/GS sharding, also shows the per-shard column dimension in parentheses, e.g., `1024 (128)` means the full dim is 1024 and each CW shard has 128 columns. |
| **Hash Size** | Number of rows in the embedding table (`tensor.shape[0]`). |
| **Ranks** | Which GPU ranks hold shards of this table. Consecutive ranges are collapsed (e.g., `0-95`). |
| **Batch Sizes** | Per-feature batch sizes. Shown as `value*count` for repeated values (e.g., `2560*4` means batch size 2560 for 4 features). Only shown if any table has custom batch sizes. |

---

# 5. Batch Size and Compute Kernels

## 5.1 Batch Size

```
Batch Size: 2560
```

The global training batch size used by the planner for cost estimation.

## 5.2 Compute Kernels Count and Storage

```
Compute Kernels Count:
   fused: 1935

Compute Kernels Storage:
   fused: HBM: 1748.026 GB, DDR: 0.0 GB
```

1. **Count**: Number of embedding tables using each compute kernel type.
2. **Storage**: Total `Storage(hbm, ddr)` across all tables using each kernel, summing `sharding_option.total_storage` for each table.

---

# 6. Imbalance Statistics

```
Total Perf Imbalance Statistics
Total Variation: 0.003
Total Distance: 0.056
Chi Divergence: 0.000
KL Divergence: 0.001

HBM Imbalance Statistics
Total Variation: 0.000
...
```

## 6.1 Metrics

These metrics measure how evenly perf or memory is distributed across ranks. All values range from **0 (perfectly balanced) to 1 (maximally imbalanced)**.

The distribution `p` is formed by normalizing the per-rank values (perf or HBM) so they sum to 1. A perfectly balanced distribution would have `p_i = 1/k` for all `k` ranks. The metrics then measure deviation from this uniform distribution:

| Metric | Formula | Intuition |
|--------|---------|-----------|
| **Total Variation** | `max(\|p_i - 1/k\|)` | Maximum deviation of any single rank from the ideal uniform share. Sensitive to the single worst-case rank. |
| **Total Distance** | `sum(\|p_i - 1/k\|)` | Sum of all deviations. Captures overall spread of imbalance across all ranks. |
| **Chi Divergence** | `sum((p_i - 1/k)^2 * k) / max_chi_sq` | Normalized chi-squared divergence. `max_chi_sq = ((N-1)/N)^2 * N + (N-1) * (1/N)` where N = number of ranks. Emphasizes large deviations (squared). |
| **KL Divergence** | `sum(p_i * log(k * p_i)) / log(k)` | Normalized Kullback-Leibler divergence (information-theoretic). Divided by `log(k)` (the maximum possible KL divergence) to normalize to [0, 1]. |

## 6.2 Computed Distributions

These statistics are computed separately for:

1. **Total Perf** — based on `perf.total` per rank
2. **HBM** — based on `used_hbm` per rank
3. **DDR** — based on `used_ddr` per rank (only if any DDR is used)

---

# 7. Max Perf and Max HBM Analysis

```
Maximum of Total Perf: 1300.325 ms on rank 28
Mean Total Perf: 1003.567 ms
Max Total Perf is 29.6% greater than the mean
Maximum of Forward Compute: 87.646 ms on rank 70
Maximum of Forward Comms: 596.845 ms on rank 28
Maximum of Backward Compute: 175.292 ms on rank 70
Maximum of Backward Comms: 609.511 ms on rank 28
Maximum of Prefetch Compute: 0.0 ms on ranks 0-95
Sum of Maxima: 1469.293 ms
```

| Metric | Calculation | Meaning |
|--------|-------------|---------|
| **Maximum of Total Perf** | `max(perf[rank].total for all ranks)` | The bottleneck rank's total perf. In synchronous training, all ranks wait for the slowest rank, so this determines iteration time. |
| **Mean Total Perf** | `mean(perf[rank].total for all ranks)` | Average perf across ranks. |
| **Max is X% greater than the mean** | `(max - mean) / mean * 100` | Quantifies the perf imbalance. Lower is better. |
| **Maximum of Fwd/Bwd Compute/Comms** | `max(perf[rank].<component>)` and which rank | Identifies which perf component and which rank is the bottleneck for each phase. |
| **Sum of Maxima** | `max(fwd_compute) + max(fwd_comms) + max(bwd_compute) + max(bwd_comms) + max(prefetch_compute)` | An upper bound on iteration time assuming zero overlap between phases. The actual "Maximum of Total Perf" can be lower if the max of each component falls on different ranks. |

---

# 8. Estimated Sharding Distribution

```
Estimated Sharding Distribution
Sparse only Max HBM: 19.174 GB on ranks [28, 43]
Sparse only Min HBM: 16.008 GB on rank [30]
Max HBM: 100.041 GB on ranks [28, 43]
Min HBM: 96.875 GB on rank [30]
Mean HBM: 99.076 GB on rank []
Low Median HBM: 99.376 GB on rank [55]
High Median HBM: 99.385 GB on rank [88]
```

| Metric | Meaning |
|--------|---------|
| **Sparse only Max/Min HBM** | Max/min of `sparse_hbm[rank]` — the embedding shard storage *without* dense and KJT overhead. Shows the raw embedding memory distribution. |
| **Max/Min HBM** | Max/min of `used_hbm[rank]` — including dense + KJT storage on top of sparse. This is the total estimated memory. |
| **Mean HBM** | Average `used_hbm` across ranks. The `rank []` notation means the mean doesn't correspond to an exact rank. |
| **Low/High Median HBM** | `statistics.median_low` / `statistics.median_high` of per-rank HBM. Shows the middle-of-the-distribution memory usage. |

The difference between "Sparse only" and total HBM tells you how much memory is consumed by the dense model and KJT features (uniform across ranks):

```
dense_storage.hbm + kjt_storage.hbm = Max HBM - Sparse only Max HBM
```

In the example: `100.041 - 19.174 = 80.867 GB` per rank from dense + KJT.

---

# 9. Critical Path Estimates

```
Critical Path (comms): 1376.864
Critical Path (compute): 262.937
Critical Path (comms + compute): 1639.801
```

## 9.1 Assumptions

The critical path estimate models synchronization points during distributed training. It makes the following assumptions:

1. There is a synchronization barrier after each of the 4 phases (fwd/bwd × comms/compute).
2. Within communication: operations for shards of the same **module + sharding type** group are executed sequentially, and ranks synchronize between groups.
3. Within computation: operations for shards of the same **module** are executed sequentially per rank, with rank-level synchronization.

## 9.2 Calculation

1. For each `(module, sharding_type, direction)` group, the critical path takes `max(sum of comms perf across shards on each rank)` — i.e., the rank that has the most communication work for that group.
2. **Critical Path (comms)** = sum of these per-group maxima across all groups.
3. **Critical Path (compute)** = `max(sum of fwd_compute per rank) + max(sum of bwd_compute per rank)`.
4. **Critical Path (comms + compute)** = comms + compute.

This estimate is typically higher than "Maximum of Total Perf" because it accounts for cross-rank synchronization within each communication group.

---

# 10. HBM Delta and Peak Memory Pressure

```
Max HBM is 0.974% greater than the mean

Top HBM Memory Usage Estimation: 100.041 GB
Top Tier #5 Estimated Peak HBM Pressure: 99.983 GB on ranks 18-19
Top Tier #4 Estimated Peak HBM Pressure: 99.988 GB on rank 76
Top Tier #3 Estimated Peak HBM Pressure: 99.996 GB on rank 42
Top Tier #2 Estimated Peak HBM Pressure: 100.031 GB on rank 63
Top Tier #1 Estimated Peak HBM Pressure: 100.041 GB on ranks 28,43
```

1. **Max HBM is X% greater than the mean**: `(max_used_hbm - mean_used_hbm) / mean_used_hbm * 100`. Shows how much the worst-case rank exceeds the average.
2. **Top HBM Memory Usage Estimation**: The peak HBM value (`max(used_hbm)`).
3. **Top Tier #1-5**: The top 5 distinct HBM usage tiers, each showing the HBM value and which ranks fall into that tier. Ranks within 1 MB of each other (using `math.isclose` with `abs_tol=1.0` in MB) are grouped into the same tier. Tiers are displayed from #5 (least pressured of the top 5) to #1 (most pressured).

This helps identify which ranks are closest to running out of memory and may OOM.

---

# 11. Storage Reservation Breakdown

```
Reserved Memory:
   HBM: 46.0 GB
   Percent of Total HBM: 25%
Planning Memory:
   HBM: 138.0 GB, DDR: 128.0 GB
   Percent of Total HBM: 75%

Dense Storage (per rank):
   HBM: 62.667 GB, DDR: 0.0 GB

KJT Storage (per rank):
   HBM: 18.2 GB, DDR: 0.0 GB
```

| Field | Meaning |
|-------|---------|
| **Reserved Memory** | Memory set aside as a safety buffer (not available for embedding shards). Determined by `StorageReservation` policy (e.g., `HeuristicalStorageReservation` or `FixedPercentageStorageReservation`). The percentage is the reservation fraction (e.g., 25% of total device HBM). |
| **Planning Memory** | Memory available for the planner to allocate embedding shards: `(1 - reserved_percent) * total_device_hbm`. The planner must fit all shards within this budget. |
| **Dense Storage (per rank)** | Estimated memory for non-embedding model parameters (e.g., dense layers, MLP weights). This is uniform across all ranks and is estimated by the storage reservation heuristic. See Section 3.4 for calculation details. |
| **KJT Storage (per rank)** | Estimated memory for KeyedJaggedTensor (sparse feature) input tensors on device. Also uniform across ranks. See Section 3.4 for calculation details. |

The relationship:

```
Total Device HBM = Reserved HBM + Planning HBM
used_hbm[rank]   = sparse_hbm[rank] + Dense HBM + KJT HBM
```

The HBM percentage shown in the per-rank table is `used_hbm[rank] / Planning HBM`.

---

# 12. Top 5 Tables Causing Max Perf / Max HBM

```
Top 5 Tables Causing Max Perf:
   F3_ONLINE_OFFLINE_MERGE_ADS_UDS_FB_FEED_VPV_P90_VPVD_EBF_FEATURE_FIRST_OBJECT_FBID_LIST
   ...

Top 5 Tables Causing Max HBM:
   F3_ADFINDER_USER_ADS_UDS_FIR_AD_IMPRESSION_UNSAMPLED_AD_ENTITY_EQUIVALENCE_KEY_LIST: 5.452 GB on rank [43]
   ...
```

1. **Top 5 Tables Causing Max Perf**: Tables contributing the most to the rank with the highest total perf. These are identified by `_find_imbalance_tables(best_plan)` which finds the tables responsible for the perf bottleneck.
2. **Top 5 Tables Causing Max HBM**: Tables with the largest per-shard HBM usage, along with their storage size and rank placement. Identified by `_find_imbalance_tables(best_plan, target_imbalance="hbm")`.

These help pinpoint which tables to investigate when trying to reduce perf or memory imbalance (e.g., by changing sharding strategy, reducing hash size, or adjusting embedding dimension).

---

# 13. Quick Reference: Key Formulas

| What | Formula |
|------|---------|
| **Total Perf (per rank)** | `fwd_compute + fwd_comms + bwd_compute + bwd_comms + prefetch_compute` |
| **Used HBM (per rank)** | `sum(shard.storage.hbm for shards on rank) + dense_storage.hbm + kjt_storage.hbm` |
| **HBM %** | `used_hbm / ((1 - reserved_pct) * device_hbm)` |
| **Input size (bytes)** | `sum(input_length * num_poolings * batch_size) * 8` (per shard, varies by sharding type) |
| **Output size (bytes)** | `sum(num_poolings * batch_size) * emb_dim * element_size` (per shard, varies by sharding type) |
| **bytes_to_gb** | `num_bytes / (1024^3)` |
| **bytes_to_mb** | `num_bytes / (1024^2)` |
| **Total Variation** | `max(\|p_i - 1/k\|)` where `p` is normalized distribution over `k` ranks |
| **Total Distance** | `sum(\|p_i - 1/k\|)` |
| **Chi Divergence** | `sum((p_i - 1/k)^2 * k) / max_chi_sq(k)` |
| **KL Divergence** | `sum(p_i * log(k * p_i)) / log(k)` |
| **Critical Path (comms)** | `sum over (module, sharding_type, direction) groups of max(per-rank comms within group)` |
| **Critical Path (compute)** | `max(per-rank fwd_compute) + max(per-rank bwd_compute)` |

---

# 14. Common Use Cases

## 14.1 Diagnosing OOM

1. Check the **Top Tier #1 Estimated Peak HBM Pressure** — if close to **Planning Memory HBM**, you're at risk of OOM.
2. Look at **Top 5 Tables Causing Max HBM** to identify which tables to re-shard or resize.
3. Compare **Sparse only Max HBM** vs **Max HBM** to understand how much comes from embeddings vs dense/KJT overhead.

## 14.2 Diagnosing Slow Training

1. Check **Maximum of Total Perf** and compare to **Mean Total Perf**. A large delta (e.g., >10%) indicates imbalance.
2. Look at the per-component maxima (Fwd Compute, Fwd Comms, Bwd Compute, Bwd Comms) to identify whether the bottleneck is compute-bound or communication-bound.
3. Check **Top 5 Tables Causing Max Perf** to identify which tables to optimize.
4. Review **Imbalance Statistics** — high values (close to 1) indicate severe imbalance.

## 14.3 Understanding Sharding Decisions

1. The **Parameter Info table** shows exactly how each table is sharded, on which ranks, and with what perf/storage cost.
2. Compare **Emb Dim (CW Dim)** — for CW-sharded tables, the CW dim tells you how many column slices exist.
3. Check the **Ranks** column to see the placement of each table.
