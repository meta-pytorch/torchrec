# Running torchrec using NVTabular DataLoader

This example demonstrates how to use NVIDIA's NVTabular for high-performance data preprocessing
with TorchRec distributed training. NVTabular provides GPU-accelerated data loading that can
significantly improve training throughput for recommendation models.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        NVTabular + TorchRec Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────┐  │
│   │   Raw Criteo TSV    │     │  NVTabular Preproc  │     │ Parquet Files   │  │
│   │    (1TB+ data)      │ ──► │   (GPU-accelerated) │ ──► │ (Optimized I/O) │  │
│   └─────────────────────┘     └─────────────────────┘     └─────────────────┘  │
│                                                                                 │
│                                       │                                         │
│                                       ▼                                         │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                      NVTabular DataLoader                               │  │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│   │  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │  │  ...    │       │  │
│   │  │ Worker  │  │ Worker  │  │ Worker  │  │ Worker  │  │         │       │  │
│   │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │  │
│   │       │            │            │            │            │            │  │
│   │       ▼            ▼            ▼            ▼            ▼            │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐   │  │
│   │  │          Direct GPU Memory Transfer (No CPU Bottleneck)         │   │  │
│   │  └─────────────────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                         │
│                                       ▼                                         │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                     TorchRec Distributed Training                       │  │
│   │                                                                         │  │
│   │   GPU 0          GPU 1          GPU 2          GPU 3                    │  │
│   │  ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐                  │  │
│   │  │Shard │       │Shard │       │Shard │       │Shard │   Embedding     │  │
│   │  │  0   │       │  1   │       │  2   │       │  3   │   Shards        │  │
│   │  └──────┘       └──────┘       └──────┘       └──────┘                  │  │
│   │     │              │              │              │                      │  │
│   │     └──────────────┴──────────────┴──────────────┘                      │  │
│   │                         │                                               │  │
│   │                         ▼                                               │  │
│   │              ┌─────────────────────┐                                    │  │
│   │              │    DLRM Forward     │                                    │  │
│   │              │  (All-to-All Comm)  │                                    │  │
│   │              └─────────────────────┘                                    │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Preprocessing Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        NVTabular Preprocessing Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Step 1: Raw Data (Criteo 1TB Click Logs)                                       │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│    day_0.tsv, day_1.tsv, ..., day_23.tsv                                        │
│    ┌────────────────────────────────────────────────────────────────────────┐   │
│    │ label │ dense_0 │ ... │ dense_12 │ sparse_0 │ sparse_1 │ ... │ sparse_25│  │
│    │   1   │  0.25   │ ... │   0.0    │  abc123  │  xyz789  │ ... │  def456  │  │
│    │   0   │  0.50   │ ... │   0.3    │  ghi012  │  jkl345  │ ... │  mno678  │  │
│    └────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Step 2: NVTabular Operations (GPU-Accelerated)                                 │
│  ══════════════════════════════════════════════                                 │
│                                                                                 │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│    │   Categorify    │ ─► │   FillMissing   │ ─► │    Normalize    │            │
│    │  (Hash sparse   │    │  (Handle nulls  │    │  (Scale dense   │            │
│    │   features)     │    │   in data)      │    │   features)     │            │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                 │
│  Step 3: Output (Parquet with Optimized Schema)                                 │
│  ═════════════════════════════════════════════                                  │
│                                                                                 │
│    criteo_binary/split/                                                         │
│    ├── train/                                                                   │
│    │   ├── part.0.parquet    ◄─── Columnar format, GPU-readable                 │
│    │   ├── part.1.parquet                                                       │
│    │   └── ...                                                                  │
│    ├── valid/                                                                   │
│    │   └── ...                                                                  │
│    └── _metadata              ◄─── Schema for fast loading                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Why NVTabular + TorchRec?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Traditional vs NVTabular Data Loading                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Traditional (CPU-based DataLoader):                                            │
│  ════════════════════════════════════                                           │
│                                                                                 │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│    │   Disk   │ ──► │   CPU    │ ──► │  Memory  │ ──► │   GPU    │              │
│    │   I/O    │     │  Decode  │     │  Copy    │     │Transfer  │              │
│    │ (slow)   │     │ (slow)   │     │          │     │(PCIe)    │              │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘              │
│                                                                                 │
│    ⚠️  CPU becomes bottleneck with large batch sizes                            │
│    ⚠️  PCIe transfer overhead for every batch                                   │
│    ⚠️  GPU idle time waiting for data                                           │
│                                                                                 │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  NVTabular (GPU-accelerated DataLoader):                                        │
│  ════════════════════════════════════════                                       │
│                                                                                 │
│    ┌──────────┐     ┌──────────────────────────────────────────────┐            │
│    │   Disk   │ ──► │   GPU Direct Storage (GDS) / NVMe           │            │
│    │   I/O    │     │                                              │            │
│    │(Parquet) │     │  ┌────────────────────────────────────────┐  │            │
│    └──────────┘     │  │  GPU Memory (Data already on device!)  │  │            │
│                     │  │  - Decode Parquet                       │  │            │
│                     │  │  - Batch assembly                       │  │            │
│                     │  │  - Feature transformation               │  │            │
│                     │  └────────────────────────────────────────┘  │            │
│                     └──────────────────────────────────────────────┘            │
│                                                                                 │
│    ✅ Eliminates CPU bottleneck                                                 │
│    ✅ Minimal PCIe transfers                                                    │
│    ✅ GPU fully utilized for both data prep and training                        │
│                                                                                 │
│  Performance Comparison (8x A100 GPUs):                                         │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│    CPU DataLoader:    ~500K samples/sec                                         │
│    NVTabular:         ~2M+ samples/sec   ◄─── 4x+ speedup!                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Setup Instructions

First run nvtabular preprocessing to first convert the criteo TSV files to parquet, and perform offline preprocessing.

Please follow the installation instructions in the [README](https://github.com/pytorch/torchrec/tree/main/torchrec/datasets/scripts/nvt) of torchrec/torchrec/datasets/scripts/nvt.

## Training Command

Afterward, to run the model across 8 GPUs, use the below command

```
torchx run -s local_cwd dist.ddp -j 1x8 --script train_torchrec.py -- --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 --over_arch_layer_sizes 1024,1024,512,256,1 --dense_arch_layer_sizes 512,256,128 --embedding_dim 128 --binary_path <path_to_nvt_output>/criteo_binary/split/ --learning_rate 1.0 --validation_freq_within_epoch 1000000 --throughput_check_freq_within_epoch 1000000 --batch_size 256
```

To run with adagrad as an optimizer, use the below flag

```
---adagrad
```

# Test on A100s

## Preliminary Training Results

**Setup:**
* Dataset: Criteo 1TB Click Logs dataset
* CUDA 11.1, NCCL 2.10.3.
* AWS p4d24xlarge instances, each with 8 40GB NVIDIA A100s.

**Results**

Reproducing MLPerfV1 settings
1. Embedding per features + model architecture
2. Learning Rate fixed at 1.0 with SGD
3. Dataset setup:
    - No frequency thresholding
4. Report > .8025 on validation set (0.8027645945549011 from above script)
5. Global batch size 2048
