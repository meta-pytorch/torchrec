# TorchRec DLRM Golden Training Example

This example demonstrates **distributed training of a Deep Learning Recommendation Model (DLRM)** using TorchRec's model-parallel capabilities. It showcases production-ready patterns for training large-scale recommendation models with sharded embeddings across multiple GPUs.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DLRM Architecture                                    │
│                                                                             │
│  Dense Features                          Sparse Features                    │
│  (Numerical)                             (Categorical IDs)                  │
│       │                                        │                            │
│       ▼                                        ▼                            │
│  ┌─────────────┐                    ┌───────────────────┐                  │
│  │  Dense Arch │                    │ EmbeddingBagCollection               │
│  │    (MLP)    │                    │ (Sharded across GPUs)                │
│  │             │                    │                   │                  │
│  │ Input: 13   │                    │  ┌─────┐ ┌─────┐ │                  │
│  │ → 64 → 128  │                    │  │ T1  │ │ T2  │ │  ... (26 tables) │
│  └──────┬──────┘                    │  └──┬──┘ └──┬──┘ │                  │
│         │                           └─────┼──────┼─────┘                  │
│         │                                 │      │                         │
│         ▼                                 ▼      ▼                         │
│    Dense Output                     Pooled Embeddings                      │
│    (128-dim)                        (26 × 128-dim)                         │
│         │                                 │                                │
│         └──────────────┬──────────────────┘                                │
│                        │                                                   │
│                        ▼                                                   │
│               ┌─────────────────┐                                          │
│               │ Feature Cross   │                                          │
│               │ (Dot Products)  │                                          │
│               └────────┬────────┘                                          │
│                        │                                                   │
│                        ▼                                                   │
│               ┌─────────────────┐                                          │
│               │   Over Arch     │                                          │
│               │     (MLP)       │                                          │
│               │  → 64 → 1       │                                          │
│               └────────┬────────┘                                          │
│                        │                                                   │
│                        ▼                                                   │
│                    Logits                                                  │
│               (Click Prediction)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          DLRM TRAINING ITERATION BREAKDOWN                               │
│                                                                                          │
│  STEP 1: Batch Preparation                                                               │
│  ═══════════════════════                                                                 │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │   Input Batch (Criteo-style data)                                             │    │
│     │                                                                               │    │
│     │   Dense Features (13 numerical)           Sparse Features (26 categorical)   │    │
│     │   ┌─────────────────────────────┐         ┌─────────────────────────────┐    │    │
│     │   │  [batch, 13]                │         │  KeyedJaggedTensor          │    │    │
│     │   │                             │         │                             │    │    │
│     │   │  age, time_on_site,         │         │  keys: ["cat_0", "cat_1",   │    │    │
│     │   │  num_clicks, ...            │         │         ... "cat_25"]       │    │    │
│     │   │                             │         │  values: [id1, id2, ...]    │    │    │
│     │   │  (normalized floats)        │         │  lengths: [1,1,1,...]       │    │    │
│     │   └─────────────────────────────┘         └─────────────────────────────┘    │    │
│     │                                                                               │    │
│     │   Labels: [0, 1, 0, 1, ...]  (click/no-click)                                │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                        │                                     │
│           ▼                                        ▼                                     │
│  STEP 2: Forward Pass                                                                    │
│  ═══════════════════                                                                     │
│                                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                                    │  │
│  │   DENSE ARCH (Bottom MLP)              SPARSE ARCH (Embedding Lookups)            │  │
│  │   ──────────────────────              ────────────────────────────────            │  │
│  │                                                                                    │  │
│  │   dense_features                       sparse_features (KJT)                       │  │
│  │   [batch, 13]                          26 categorical features                     │  │
│  │        │                                      │                                    │  │
│  │        ▼                                      ▼                                    │  │
│  │   ┌─────────────┐                    ┌──────────────────────────────┐             │  │
│  │   │ Linear(13→64)│                    │   EmbeddingBagCollection     │             │  │
│  │   │ ReLU         │                    │   (26 tables × 1M × 128)     │             │  │
│  │   │ Linear(64→128)│                   │                              │             │  │
│  │   │ ReLU         │                    │  ┌────┐ ┌────┐     ┌────┐   │             │  │
│  │   └──────┬───────┘                    │  │ T0 │ │ T1 │ ... │T25 │   │             │  │
│  │          │                            │  └──┬─┘ └──┬─┘     └──┬─┘   │             │  │
│  │          ▼                            │     │      │          │     │             │  │
│  │   dense_output                        │     └──────┴────┬─────┘     │             │  │
│  │   [batch, 128]                        └─────────────────┼───────────┘             │  │
│  │          │                                              │                          │  │
│  │          │                                              ▼                          │  │
│  │          │                                   pooled_embeddings                     │  │
│  │          │                                   [batch, 26 × 128]                     │  │
│  │          │                                              │                          │  │
│  │          └──────────────────────┬───────────────────────┘                          │  │
│  │                                 │                                                  │  │
│  │                                 ▼                                                  │  │
│  │                    ┌────────────────────────────┐                                  │  │
│  │                    │     FEATURE INTERACTION    │                                  │  │
│  │                    │     (Dot Products)         │                                  │  │
│  │                    │                            │                                  │  │
│  │                    │  Concatenate:              │                                  │  │
│  │                    │  [dense, embed_0, embed_1, │                                  │  │
│  │                    │   ... embed_25]            │                                  │  │
│  │                    │                            │                                  │  │
│  │                    │  Pairwise dot products:    │                                  │  │
│  │                    │  all pairs of vectors      │                                  │  │
│  │                    └────────────┬───────────────┘                                  │  │
│  │                                 │                                                  │  │
│  │                                 ▼                                                  │  │
│  │                    ┌────────────────────────────┐                                  │  │
│  │                    │      OVER ARCH (Top MLP)   │                                  │  │
│  │                    │                            │                                  │  │
│  │                    │  Linear(features → 64)     │                                  │  │
│  │                    │  ReLU                      │                                  │  │
│  │                    │  Linear(64 → 1)            │                                  │  │
│  │                    │  (no activation - logits)  │                                  │  │
│  │                    └────────────┬───────────────┘                                  │  │
│  │                                 │                                                  │  │
│  │                                 ▼                                                  │  │
│  │                           logits [batch, 1]                                        │  │
│  │                         (click probability)                                        │  │
│  │                                                                                    │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│           │                                                                              │
│           ▼                                                                              │
│  STEP 3: Loss Computation                                                                │
│  ═══════════════════════                                                                 │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │   Binary Cross-Entropy with Logits:                                           │    │
│     │                                                                               │    │
│     │   loss = BCEWithLogitsLoss(logits, labels)                                    │    │
│     │                                                                               │    │
│     │        = -1/N Σ [y·log(σ(x)) + (1-y)·log(1-σ(x))]                             │    │
│     │                                                                               │    │
│     │   where: σ(x) = 1/(1 + e^(-x))  (sigmoid)                                     │    │
│     │          y = label (0 = no click, 1 = click)                                  │    │
│     │          x = logit (model output)                                             │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  STEP 4: Backward Pass with In-Backward Optimizer                                        │
│  ═══════════════════════════════════════════════                                         │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │   STANDARD APPROACH (less efficient):                                         │    │
│     │   ──────────────────────────────────                                          │    │
│     │                                                                               │    │
│     │   loss.backward()           →  Compute all gradients                          │    │
│     │   dense_optimizer.step()    →  Update MLP weights                             │    │
│     │   sparse_optimizer.step()   →  Update embeddings (separate pass!)             │    │
│     │                                                                               │    │
│     │                                                                               │    │
│     │   TORCHREC FUSED APPROACH (used in this example):                             │    │
│     │   ─────────────────────────────────────────────                               │    │
│     │                                                                               │    │
│     │   ┌─────────────────────────────────────────────────────────────────────┐    │    │
│     │   │  apply_optimizer_in_backward(RowWiseAdagrad, sparse_params, lr=0.1) │    │    │
│     │   └─────────────────────────────────────────────────────────────────────┘    │    │
│     │                                                                               │    │
│     │   loss.backward()           →  For dense params: compute gradients            │    │
│     │                             →  For sparse params: compute grad + UPDATE       │    │
│     │                                                  (fused in single kernel!)    │    │
│     │   dense_optimizer.step()    →  Update MLP weights only                        │    │
│     │                                                                               │    │
│     │   Result: ~15% faster training, better memory efficiency                      │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## GPU Memory Layout Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    GPU MEMORY LAYOUT WITH MODEL PARALLELISM                              │
│                                                                                          │
│   Without TorchRec (Data Parallel Only):                                                 │
│   ──────────────────────────────────────                                                 │
│                                                                                          │
│   Each GPU must hold ALL embedding tables → Limits model size!                           │
│                                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐                 │
│   │  GPU 0                          │  GPU 1                          │                 │
│   │  ┌─────────────────────────┐    │  ┌─────────────────────────┐    │                 │
│   │  │  Full Copy of ALL       │    │  │  Full Copy of ALL       │    │                 │
│   │  │  26 Embedding Tables    │    │  │  26 Embedding Tables    │    │                 │
│   │  │  (26 × 1M × 128 × 4B    │    │  │  (26 × 1M × 128 × 4B    │    │                 │
│   │  │   = 13.3 GB each!)      │    │  │   = 13.3 GB each!)      │    │                 │
│   │  └─────────────────────────┘    │  └─────────────────────────┘    │                 │
│   │  ┌───────────┐                  │  ┌───────────┐                  │                 │
│   │  │ Dense MLP │                  │  │ Dense MLP │                  │                 │
│   │  └───────────┘                  │  └───────────┘                  │                 │
│   └───────────────────────────────────────────────────────────────────┘                 │
│                                                                                          │
│   With TorchRec DistributedModelParallel:                                                │
│   ─────────────────────────────────────                                                  │
│                                                                                          │
│   Embedding tables SHARDED across GPUs → Scale to massive models!                        │
│                                                                                          │
│   ┌───────────────────────────────────────────────────────────────────┐                 │
│   │  GPU 0                          │  GPU 1                          │                 │
│   │  ┌─────────────────────────┐    │  ┌─────────────────────────┐    │                 │
│   │  │  Embedding Tables       │    │  │  Embedding Tables       │    │                 │
│   │  │  T0, T1, T2, ... T12    │    │  │  T13, T14, ... T25      │    │                 │
│   │  │  (13 tables)            │    │  │  (13 tables)            │    │                 │
│   │  │  ~6.7 GB                │    │  │  ~6.7 GB                │    │                 │
│   │  └─────────────────────────┘    │  └─────────────────────────┘    │                 │
│   │  ┌───────────┐                  │  ┌───────────┐                  │                 │
│   │  │ Dense MLP │ (replicated)     │  │ Dense MLP │ (replicated)     │                 │
│   │  └───────────┘                  │  └───────────┘                  │                 │
│   └───────────────────────────────────────────────────────────────────┘                 │
│                          │                     │                                         │
│                          └─────────┬───────────┘                                         │
│                                    │                                                     │
│                                    ▼                                                     │
│                    ┌───────────────────────────────┐                                    │
│                    │  NCCL All-to-All Communication │                                    │
│                    │                               │                                    │
│                    │  Each GPU sends requests for  │                                    │
│                    │  embeddings it needs to other │                                    │
│                    │  GPUs and receives back       │                                    │
│                    │  the looked-up values         │                                    │
│                    └───────────────────────────────┘                                    │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  SHARDING STRATEGIES (automatically selected by TorchRec planner):               │   │
│   │                                                                                  │   │
│   │  • Table-wise:    Each table on one GPU (shown above)                            │   │
│   │  • Row-wise:      Table rows split across GPUs (for huge single tables)          │   │
│   │  • Column-wise:   Embedding dimensions split across GPUs                         │   │
│   │  • Table-row-wise: Combination for very large tables                             │   │
│   │  • Data-parallel: Small tables replicated on all GPUs                            │   │
│   │                                                                                  │   │
│   │  TorchRec's planner automatically chooses optimal sharding per table!            │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Pipelined Training Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    TrainPipelineSparseDist: OVERLAPPED EXECUTION                         │
│                                                                                          │
│   WITHOUT PIPELINING (Sequential):                                                       │
│   ─────────────────────────────────                                                      │
│                                                                                          │
│   Time ──────────────────────────────────────────────────────────────────────────────►  │
│                                                                                          │
│   Batch 0: [Load] [All2All] [Forward] [Backward]                                        │
│   Batch 1:                                        [Load] [All2All] [Forward] [Backward] │
│   Batch 2:                                                                     ...      │
│                                                                                          │
│   Problem: GPU idle during data loading and communication!                              │
│                                                                                          │
│                                                                                          │
│   WITH TrainPipelineSparseDist (Overlapped):                                             │
│   ──────────────────────────────────────────                                             │
│                                                                                          │
│   Time ──────────────────────────────────────────────────────────────────────────────►  │
│                                                                                          │
│   Stage 0   Stage 1      Stage 2         Stage 3                                        │
│   (CPU)     (GPU Async)  (GPU Compute)   (GPU Compute)                                  │
│                                                                                          │
│   ┌──────┐  ┌──────────┐ ┌─────────────┐ ┌──────────────┐                               │
│   │ Load │  │ All2All  │ │   Forward   │ │   Backward   │                               │
│   │ Data │  │ Comm     │ │   Pass      │ │   Pass       │                               │
│   └──┬───┘  └────┬─────┘ └──────┬──────┘ └──────┬───────┘                               │
│      │           │              │               │                                        │
│      ▼           ▼              ▼               ▼                                        │
│                                                                                          │
│   Batch 0: [LOAD]─────►                                                                 │
│   Batch 1:        [LOAD]─────►                                                          │
│   Batch 2:               [LOAD]─────►                                                   │
│   Batch 3:                      [LOAD]─────►                                            │
│                                                                                          │
│   Batch 0:        [A2A]──────────►                                                      │
│   Batch 1:               [A2A]──────────►                                               │
│   Batch 2:                      [A2A]──────────►                                        │
│                                                                                          │
│   Batch 0:               [FORWARD]─────────►                                            │
│   Batch 1:                       [FORWARD]─────────►                                    │
│   Batch 2:                               [FORWARD]─────────►                            │
│                                                                                          │
│   Batch 0:                       [BACKWARD]───────►                                     │
│   Batch 1:                               [BACKWARD]───────►                             │
│   Batch 2:                                       [BACKWARD]───────►                     │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  RESULT: Near 100% GPU utilization - communication hidden behind computation!    │   │
│   │                                                                                  │   │
│   │  • Data loading happens while GPU computes                                       │   │
│   │  • All-to-All communication overlaps with backward pass of previous batch        │   │
│   │  • 3-stage pipeline ensures smooth execution                                     │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quantized Communication Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      QUANTIZED COMMUNICATION (QCommsConfig)                              │
│                                                                                          │
│   Multi-node training bottleneck: Network bandwidth for embedding gradients             │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Node 0 (GPUs 0-3)                         Node 1 (GPUs 4-7)                      │  │
│   │  ┌──────────────────┐                      ┌──────────────────┐                   │  │
│   │  │ Embedding        │    ◄─── Network ───► │ Embedding        │                   │  │
│   │  │ Gradients        │        Bandwidth     │ Gradients        │                   │  │
│   │  │ (Large!)         │        Bottleneck    │ (Large!)         │                   │  │
│   │  └──────────────────┘                      └──────────────────┘                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   Solution: Quantize embeddings/gradients during communication                           │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                   │  │
│   │   FP32 (Standard)              FP16 (Forward)             BF16 (Backward)        │  │
│   │   ────────────────             ─────────────              ──────────────         │  │
│   │                                                                                   │  │
│   │   32 bits per value            16 bits per value          16 bits per value      │  │
│   │   Full precision               Good for inference         Good for gradients     │  │
│   │                                                                                   │  │
│   │   ┌────────────────┐          ┌────────────────┐         ┌────────────────┐      │  │
│   │   │████████████████│          │████████        │         │████████        │      │  │
│   │   │████████████████│          │                │         │                │      │  │
│   │   │████████████████│          │  2x smaller    │         │  2x smaller    │      │  │
│   │   │████████████████│          │  = 2x faster   │         │  better range  │      │  │
│   │   └────────────────┘          └────────────────┘         └────────────────┘      │  │
│   │                                                                                   │  │
│   │   Code:                                                                           │  │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│   │   │  qcomms_config = QCommsConfig(                                          │    │  │
│   │   │      forward_precision=CommType.FP16,   # Forward: FP16 (inference ok)  │    │  │
│   │   │      backward_precision=CommType.BF16,  # Backward: BF16 (stable grads) │    │  │
│   │   │  )                                                                      │    │  │
│   │   │                                                                         │    │  │
│   │   │  sharder = EmbeddingBagCollectionSharder(                               │    │  │
│   │   │      qcomm_codecs_registry=get_qcomm_codecs_registry(qcomms_config)     │    │  │
│   │   │  )                                                                      │    │  │
│   │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   Impact on Multi-Node Training:                                                         │
│   ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│   │  Without QComms     │  FP32 → FP32                  │  100% bandwidth usage       │ │
│   │  With FP16/BF16     │  FP32 → FP16 → ... → FP32     │  50% bandwidth usage (2x!)  │ │
│   │  With INT8          │  FP32 → INT8 → ... → FP32     │  25% bandwidth usage (4x!)  │ │
│   └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## What This Example Demonstrates

### TorchRec Features
- **`EmbeddingBagCollection`**: Efficient embedding lookups for 26 categorical features
- **`DistributedModelParallel`**: Automatic sharding of embedding tables across GPUs
- **`TrainPipelineSparseDist`**: Overlapped communication, computation, and data transfer
- **`RowWiseAdagrad`**: Row-wise optimizer for sparse embeddings (fused with backward pass)
- **`QCommsConfig`**: Quantized communication (FP16/BF16) for efficient multi-node training
- **In-backward Optimizer**: Fuses gradient computation with optimizer step for embeddings

### Production Training Patterns
- **SPMD Training**: Each process runs the same script with different data shards
- **Model Parallelism**: Large embedding tables sharded across GPUs
- **Pipeline Parallelism**: Overlapped data loading, communication, and computation
- **Mixed Precision**: FP16 forward, BF16 backward for communication efficiency

## Directory Structure

```
golden_training/
├── README.md                      # This file
├── BUCK                           # Build configuration
├── __init__.py
├── train_dlrm.py                  # Main distributed training script
├── train_dlrm_data_parallel.py    # Data-parallel variant (simpler)
└── tests/
    ├── __init__.py
    └── test_train_dlrm.py         # Unit tests
```

## Scripts

### `train_dlrm.py` - Model-Parallel Training (Recommended)

Full-featured distributed training with model-parallel embedding sharding:

**Key features:**
- Embedding tables sharded across GPUs using `DistributedModelParallel`
- Communication/computation overlap via `TrainPipelineSparseDist`
- Row-wise Adagrad with in-backward fusion
- Quantized communication for multi-node training

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_embeddings` | 1M | Size of each embedding table |
| `embedding_dim` | 128 | Embedding dimension |
| `dense_arch_layer_sizes` | [64, 128] | Dense MLP architecture |
| `over_arch_layer_sizes` | [64, 1] | Over (output) MLP architecture |
| `learning_rate` | 0.1 | Learning rate for RowWiseAdagrad |
| `num_iterations` | 1000 | Training iterations |
| `qcomm_forward_precision` | FP16 | Forward pass communication precision |
| `qcomm_backward_precision` | BF16 | Backward pass communication precision |

### `train_dlrm_data_parallel.py` - Data-Parallel Training (Simpler)

Simpler variant using standard data parallelism (DDP):

**When to use:**
- Smaller models that fit on a single GPU
- Quick prototyping and debugging
- When model parallelism isn't needed

## Running the Example

### Prerequisites

```bash
# Install TorchRec
pip install torchrec

# For distributed training
pip install torchx
```

### Local Training (Single Node, Multi-GPU)

Using [TorchX](https://pytorch.org/torchx/main/quickstart.html) (recommended):

```bash
# 2 GPUs on local machine
torchx run -s local_cwd dist.ddp -j 1x2 --script train_dlrm.py

# 4 GPUs on local machine
torchx run -s local_cwd dist.ddp -j 1x4 --script train_dlrm.py

# With custom parameters
torchx run -s local_cwd dist.ddp -j 1x2 --script train_dlrm.py -- \
    --num_embeddings 100000 \
    --embedding_dim 64 \
    --num_iterations 500
```

Using `torchrun` (alternative):

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train_dlrm.py

# 4 GPUs with custom params
torchrun --nproc_per_node=4 train_dlrm.py \
    --num_embeddings 100000 \
    --num_iterations 500
```

### Cluster Training (Multi-Node)

```bash
# Slurm cluster - 8 GPUs across 2 nodes
torchx run -s slurm dist.ddp \
    -j 2x4 \
    --gpu 4 \
    --script train_dlrm.py

# AWS/Kubernetes - see TorchX docs for scheduler-specific options
```

## Key Code Walkthrough

### 1. Embedding Configuration

```python
# Create embedding configs for 26 categorical features (Criteo-style)
eb_configs = [
    EmbeddingBagConfig(
        name=f"t_{feature_name}",
        embedding_dim=embedding_dim,      # 128
        num_embeddings=num_embeddings,    # 1M
        feature_names=[feature_name],
    )
    for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
]
```

### 2. Model Creation with Meta Device

```python
# Create model on "meta" device (deferred materialization)
dlrm_model = DLRM(
    embedding_bag_collection=EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("meta"),  # Key: no memory allocated yet
    ),
    dense_in_features=len(DEFAULT_INT_NAMES),
    dense_arch_layer_sizes=dense_arch_layer_sizes,
    over_arch_layer_sizes=over_arch_layer_sizes,
    dense_device=device,
)
```

### 3. In-Backward Optimizer Fusion

```python
# Fuse optimizer with backward pass for embeddings
apply_optimizer_in_backward(
    RowWiseAdagrad,
    train_model.model.sparse_arch.parameters(),
    {"lr": learning_rate},
)
```

### 4. Distributed Model Wrapping

```python
# Wrap with DistributedModelParallel for automatic sharding
model = DistributedModelParallel(
    module=train_model,
    device=device,
    sharders=[sharder],  # EmbeddingBagCollectionSharder with QComms
)
```

### 5. Training Pipeline

```python
# Use pipelined training for overlap
train_pipeline = TrainPipelineSparseDist(
    model,
    non_fused_optimizer,  # For dense parameters only
    device,
)

# Training loop
for _ in tqdm(range(num_iterations)):
    train_pipeline.progress(train_iterator)
```

## Quantized Communication

For multi-node training, quantized communication significantly reduces bandwidth:

```python
qcomm_codecs_registry = get_qcomm_codecs_registry(
    qcomms_config=QCommsConfig(
        forward_precision=CommType.FP16,   # 16-bit forward
        backward_precision=CommType.BF16,  # BF16 for training stability
    )
)
sharder = EmbeddingBagCollectionSharder(qcomm_codecs_registry=qcomm_codecs_registry)
```

**Precision recommendations:**
- **Forward**: FP16 (good accuracy, 2x compression)
- **Backward**: BF16 (maintains training stability, 2x compression)
- **Experimental**: INT8/FP8 for more aggressive compression

## Integration with Other Examples

This training example produces models that can be used with:

| Example | Purpose |
|---------|---------|
| [`prediction/`](../prediction/) | Single-GPU inference with trained DLRM |
| [`inference_legacy/`](../inference_legacy/) | Production inference with quantization |
| [`retrieval/`](../retrieval/) | Use DLRM as ranking stage after retrieval |

## Performance Tips

### Training Speed
1. **Use `TrainPipelineSparseDist`** for communication/computation overlap
2. **Enable quantized comms** (`QCommsConfig`) for multi-node
3. **Tune batch size** for GPU memory utilization
4. **Use in-backward optimizer** for embeddings

### Memory Efficiency
1. **Use `meta` device** for deferred initialization
2. **Model parallelism** for large embedding tables
3. **Gradient checkpointing** for very large models

### Debugging
1. Start with **data-parallel** (`train_dlrm_data_parallel.py`) for correctness
2. Use **small configs** first (`num_embeddings=1000`)
3. Check **`buck log what-failed`** for build issues

## Common Issues & Troubleshooting

### 1. CUDA Out of Memory
```bash
# Reduce embedding size
torchx run ... --script train_dlrm.py -- --num_embeddings 100000 --embedding_dim 64
```

### 2. NCCL Timeout
```bash
# Increase timeout for large models
export NCCL_TIMEOUT=1800  # 30 minutes
```

### 3. Slow First Iteration
- Normal: First iteration includes JIT compilation and sharding plan computation
- Subsequent iterations will be much faster

### 4. Process Group Issues
```bash
# Ensure clean state
export CUDA_VISIBLE_DEVICES=0,1  # Explicitly set GPUs
```

## References

- [DLRM Paper](https://arxiv.org/abs/1906.00091) - Original DLRM architecture
- [TorchRec Documentation](https://pytorch.org/torchrec/)
- [TorchX Documentation](https://pytorch.org/torchx/)
- [Criteo Dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) - Standard benchmark data
