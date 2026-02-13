# TorchRec Examples Integration Guide

This guide demonstrates how to combine TorchRec's examples to build a **complete, production-ready recommendation system**. Each example covers a specific stage of the recommendation pipeline, and this guide shows how they work together.

## Complete Recommendation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            PRODUCTION RECOMMENDATION SYSTEM                                      │
│                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              TRAINING PIPELINE                                           │   │
│  │                                                                                          │   │
│  │   ┌──────────────────────┐         ┌──────────────────────┐                             │   │
│  │   │   STAGE 1: RETRIEVAL │         │   STAGE 2: RANKING   │                             │   │
│  │   │   (Two-Tower Model)  │         │   (DLRM Model)       │                             │   │
│  │   │                      │         │                      │                             │   │
│  │   │   Example:           │         │   Example:           │                             │   │
│  │   │   retrieval/         │         │   golden_training/   │                             │   │
│  │   │   two_tower_train.py │         │   train_dlrm.py      │                             │   │
│  │   └──────────┬───────────┘         └──────────┬───────────┘                             │   │
│  │              │                                │                                          │   │
│  │              ▼                                ▼                                          │   │
│  │   ┌──────────────────────┐         ┌──────────────────────┐                             │   │
│  │   │  Trained Retrieval   │         │   Trained Ranking    │                             │   │
│  │   │  Model + FAISS Index │         │   Model Checkpoint   │                             │   │
│  │   └──────────┬───────────┘         └──────────┬───────────┘                             │   │
│  │              │                                │                                          │   │
│  └──────────────┼────────────────────────────────┼──────────────────────────────────────────┘   │
│                 │                                │                                              │
│  ┌──────────────┼────────────────────────────────┼──────────────────────────────────────────┐   │
│  │              │      INFERENCE PIPELINE        │                                          │   │
│  │              ▼                                ▼                                          │   │
│  │   ┌──────────────────────┐         ┌──────────────────────┐                             │   │
│  │   │   STAGE 1: RETRIEVE  │         │   STAGE 2: RANK      │                             │   │
│  │   │   (Quantized Model)  │         │   (Quantized Model)  │                             │   │
│  │   │                      │         │                      │                             │   │
│  │   │   Example:           │         │   Example:           │                             │   │
│  │   │   retrieval/         │────────►│   prediction/        │                             │   │
│  │   │   two_tower_         │   100   │   predict_using_     │                             │   │
│  │   │   retrieval.py       │ items   │   torchrec.py        │                             │   │
│  │   └──────────────────────┘         └──────────┬───────────┘                             │   │
│  │                                               │                                          │   │
│  │                                               ▼                                          │   │
│  │                                    ┌──────────────────────┐                             │   │
│  │                                    │   Top 10 Ranked      │                             │   │
│  │                                    │   Recommendations    │                             │   │
│  │                                    └──────────────────────┘                             │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Example Directory Map

| Directory | Purpose | Stage | Key Files |
|-----------|---------|-------|-----------|
| [`retrieval/`](./retrieval/) | Two-tower model training & inference | Candidate Retrieval | `two_tower_train.py`, `two_tower_retrieval.py` |
| [`golden_training/`](./golden_training/) | DLRM distributed training | Ranking Training | `train_dlrm.py` |
| [`prediction/`](./prediction/) | DLRM inference | Ranking Inference | `predict_using_torchrec.py` |
| [`inference_legacy/`](./inference_legacy/) | Production inference patterns | Production Serving | `dlrm_predict.py`, `dlrm_packager.py` |
| [`bert4rec/`](./bert4rec/) | Sequential recommendation | Alternative Model | `bert4rec_main.py` |
| [`zch/`](./zch/) | Zero-collision hashing | Feature Engineering | `main.py`, `sparse_arch.py` |
| [`sharding/`](./sharding/) | Sharding strategies tutorial | Learning | `sharding.ipynb`, `uvm.ipynb` |

## How Training Works: Visual Guide

Understanding how TorchRec training works is essential for building recommendation systems. This section provides visual explanations of the key concepts.

### The Training Loop: Bird's Eye View

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE TRAINING WORKFLOW                                         │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 1: SETUP (Happens Once)                                                   │   │
│   │                                                                                  │   │
│   │  1. Initialize Process Group (NCCL backend for GPU communication)               │   │
│   │  2. Create Model (EmbeddingBagCollection + MLP layers)                          │   │
│   │  3. Wrap with DistributedModelParallel (DMP)                                    │   │
│   │     - TorchRec planner analyzes GPU memory & table sizes                        │   │
│   │     - Automatically shards embedding tables across GPUs                         │   │
│   │  4. Create Optimizers (sparse: RowWiseAdagrad, dense: SGD/Adam)                 │   │
│   │  5. Initialize DataLoader                                                        │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                              │
│                                           ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 2: TRAINING LOOP (Repeats for N iterations)                               │   │
│   │                                                                                  │   │
│   │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│   │  │  for batch in dataloader:                                                │    │   │
│   │  │                                                                          │    │   │
│   │  │    Step 1: LOAD DATA                                                     │    │   │
│   │  │    ├── Read batch from disk/memory                                       │    │   │
│   │  │    ├── Convert to KeyedJaggedTensor (KJT)                                │    │   │
│   │  │    └── Move to GPU                                                       │    │   │
│   │  │                              │                                           │    │   │
│   │  │                              ▼                                           │    │   │
│   │  │    Step 2: FORWARD PASS                                                  │    │   │
│   │  │    ├── Embedding lookups (sharded across GPUs)                           │    │   │
│   │  │    ├── All-to-All communication (gather embeddings)                      │    │   │
│   │  │    ├── Feature interactions (dense × sparse)                             │    │   │
│   │  │    └── MLP forward → logits                                              │    │   │
│   │  │                              │                                           │    │   │
│   │  │                              ▼                                           │    │   │
│   │  │    Step 3: COMPUTE LOSS                                                  │    │   │
│   │  │    └── BCEWithLogitsLoss(logits, labels)                                 │    │   │
│   │  │                              │                                           │    │   │
│   │  │                              ▼                                           │    │   │
│   │  │    Step 4: BACKWARD PASS                                                 │    │   │
│   │  │    ├── loss.backward()                                                   │    │   │
│   │  │    ├── Gradient computation for all params                               │    │   │
│   │  │    └── [FUSED] Sparse optimizer update during backward                   │    │   │
│   │  │                              │                                           │    │   │
│   │  │                              ▼                                           │    │   │
│   │  │    Step 5: OPTIMIZER STEP                                                │    │   │
│   │  │    ├── Dense optimizer updates MLP weights                               │    │   │
│   │  │    └── (Sparse already updated in backward)                              │    │   │
│   │  │                              │                                           │    │   │
│   │  │                              ▼                                           │    │   │
│   │  │    Step 6: LOG & CHECKPOINT                                              │    │   │
│   │  │    └── Every N steps: save model, log metrics                            │    │   │
│   │  │                                                                          │    │   │
│   │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                              │
│                                           ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  PHASE 3: EXPORT (Happens Once After Training)                                   │   │
│   │                                                                                  │   │
│   │  1. Save model checkpoint (torch.save)                                           │   │
│   │  2. For retrieval: Build FAISS index from item embeddings                       │   │
│   │  3. Optional: Quantize model for inference (INT8)                               │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Embedding Table Sharding: How Large Models Fit on GPUs

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    WHY SHARDING MATTERS FOR RECOMMENDATION MODELS                        │
│                                                                                          │
│   Problem: Recommendation models have HUGE embedding tables                              │
│                                                                                          │
│   Example: E-commerce recommendation system                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  • User embeddings:    100M users × 128 dims × 4 bytes  = 51.2 GB              │   │
│   │  • Product embeddings: 50M items × 128 dims × 4 bytes   = 25.6 GB              │   │
│   │  • Category embeddings: 10K cats × 64 dims × 4 bytes    = 2.6 MB               │   │
│   │  • Brand embeddings:   100K brands × 64 dims × 4 bytes  = 25.6 MB              │   │
│   │  ─────────────────────────────────────────────────────────────────────          │   │
│   │  Total: ~77 GB  (Won't fit on a single 80GB GPU with activations!)              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Solution: TorchRec's DistributedModelParallel shards tables across GPUs               │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │  8-GPU Setup (80GB each = 640GB total):                                          │   │
│   │                                                                                  │   │
│   │  GPU 0        GPU 1        GPU 2        GPU 3                                    │   │
│   │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐                               │   │
│   │  │User    │   │User    │   │User    │   │User    │                               │   │
│   │  │0-12.5M │   │12.5-25M│   │25-37.5M│   │37.5-50M│                               │   │
│   │  │(6.4GB) │   │(6.4GB) │   │(6.4GB) │   │(6.4GB) │                               │   │
│   │  └────────┘   └────────┘   └────────┘   └────────┘                               │   │
│   │                                                                                  │   │
│   │  GPU 4        GPU 5        GPU 6        GPU 7                                    │   │
│   │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐                               │   │
│   │  │User    │   │User    │   │Product │   │Product │                               │   │
│   │  │50-62.5M│   │62.5-75M│   │0-25M   │   │25-50M  │                               │   │
│   │  │(6.4GB) │   │(6.4GB) │   │(12.8GB)│   │(12.8GB)│                               │   │
│   │  │        │   │        │   │        │   │        │                               │   │
│   │  │User    │   │User    │   │Cat+    │   │Cat+    │                               │   │
│   │  │75-87.5M│   │87.5-100│   │Brand   │   │Brand   │                               │   │
│   │  │(6.4GB) │   │(6.4GB) │   │(~14MB) │   │(~14MB) │                               │   │
│   │  └────────┘   └────────┘   └────────┘   └────────┘                               │   │
│   │                                                                                  │   │
│   │  Each GPU: ~10-15GB embeddings + room for activations, gradients, optimizer      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   All-to-All Communication Pattern:                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │  Batch of 1024 samples needs user_id embeddings:                                 │   │
│   │                                                                                  │   │
│   │  • 128 user_ids map to GPU 0's shard                                             │   │
│   │  • 120 user_ids map to GPU 1's shard                                             │   │
│   │  • 135 user_ids map to GPU 2's shard                                             │   │
│   │  • ... (distributed based on hash(user_id) % num_gpus)                           │   │
│   │                                                                                  │   │
│   │  Each GPU:                                                                       │   │
│   │  1. Looks up embeddings for IDs in its shard                                     │   │
│   │  2. Sends embeddings to requesting GPUs (All-to-All)                             │   │
│   │  3. Receives embeddings from other GPUs                                          │   │
│   │  4. Assembles complete embedding tensor for its portion of batch                 │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Understanding KeyedJaggedTensor (KJT)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    KEYED JAGGED TENSOR: THE CORE DATA STRUCTURE                          │
│                                                                                          │
│   Why KJT? Recommendation features are JAGGED (variable-length per sample)               │
│                                                                                          │
│   Example: User browsing history (some users browse 3 items, some browse 100)            │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Raw Data (4 users):                                                             │   │
│   │                                                                                  │   │
│   │  User 0: browsed [item_101, item_205]              → 2 items                     │   │
│   │  User 1: browsed [item_42, item_88, item_15, item_301]  → 4 items               │   │
│   │  User 2: browsed [item_77]                         → 1 item                      │   │
│   │  User 3: browsed [item_12, item_99, item_56]       → 3 items                     │   │
│   │                                                                                  │   │
│   │  Cannot use regular Tensor (different lengths per row!)                          │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  KeyedJaggedTensor Representation:                                               │   │
│   │                                                                                  │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────┐  │   │
│   │  │  keys:    ["browsed_items"]                                               │  │   │
│   │  │                                                                           │  │   │
│   │  │  values:  [101, 205, 42, 88, 15, 301, 77, 12, 99, 56]                     │  │   │
│   │  │            ├─────┤  ├───────────────┤  ├──┤ ├──────────┤                  │  │   │
│   │  │            User 0   User 1           User 2 User 3                        │  │   │
│   │  │                                                                           │  │   │
│   │  │  lengths: [2, 4, 1, 3]  ← Number of items per user                        │  │   │
│   │  │                                                                           │  │   │
│   │  └───────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                  │   │
│   │  Memory layout: Contiguous 1D tensor + lengths = efficient GPU operations!       │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  After EmbeddingBagCollection (with pooling="sum"):                              │   │
│   │                                                                                  │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────┐  │   │
│   │  │                                                                           │  │   │
│   │  │  User 0: emb[101] + emb[205]               → 1 × embedding_dim            │  │   │
│   │  │  User 1: emb[42] + emb[88] + emb[15] + emb[301] → 1 × embedding_dim       │  │   │
│   │  │  User 2: emb[77]                           → 1 × embedding_dim            │  │   │
│   │  │  User 3: emb[12] + emb[99] + emb[56]       → 1 × embedding_dim            │  │   │
│   │  │                                                                           │  │   │
│   │  │  Output: Tensor[4, embedding_dim]  ← Fixed-size, ready for MLP!           │  │   │
│   │  │                                                                           │  │   │
│   │  └───────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Multiple Feature Types in One KJT:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │  keys:    ["user_id", "browsed_items", "purchased_items"]                        │   │
│   │  values:  [u0, u1, u2, u3, | b0, b1, b2, ... | p0, p1, ...]                      │   │
│   │  lengths: [[1,1,1,1], [2,4,1,3], [1,2,0,1]]  ← jagged per feature!              │   │
│   │                                                                                  │   │
│   │  Efficient: One CUDA kernel processes all features together                      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Training Performance: Pipelining & Optimization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING OPTIMIZATIONS IN TORCHREC                                    │
│                                                                                          │
│   Optimization 1: TrainPipelineSparseDist                                                │
│   ─────────────────────────────────────                                                  │
│                                                                                          │
│   Without Pipeline (Sequential):                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Time: ═══════════════════════════════════════════════════════════════════►     │   │
│   │                                                                                  │   │
│   │  GPU:  [Idle][Comm][Compute][Idle][Comm][Compute][Idle]...                       │   │
│   │         ▲          ▲              ▲                                              │   │
│   │         │          │              │                                              │   │
│   │    Wait for    Forward/      Wait for                                            │   │
│   │    data load   Backward      next batch                                          │   │
│   │                                                                                  │   │
│   │  GPU Utilization: ~40%  ❌                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   With TrainPipelineSparseDist (3-Stage Pipeline):                                       │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Time: ═══════════════════════════════════════════════════════════════════►     │   │
│   │                                                                                  │   │
│   │  Stage 0 (CPU):  [Load B0][Load B1][Load B2][Load B3]...                         │   │
│   │  Stage 1 (GPU):        [Comm B0][Comm B1][Comm B2][Comm B3]...                   │   │
│   │  Stage 2 (GPU):              [Fwd B0][Fwd B1][Fwd B2][Fwd B3]...                 │   │
│   │  Stage 3 (GPU):                    [Bwd B0][Bwd B1][Bwd B2][Bwd B3]...           │   │
│   │                                                                                  │   │
│   │  GPU Utilization: ~95%  ✅                                                       │   │
│   │                                                                                  │   │
│   │  How: While computing batch N, load batch N+1 and communicate batch N+2          │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Optimization 2: In-Backward Optimizer Fusion                                           │
│   ─────────────────────────────────────────────                                          │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │  Standard Training (3 separate steps):                                           │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │  1. forward()   → Compute activations                                      │ │   │
│   │  │  2. backward()  → Compute ALL gradients                                    │ │   │
│   │  │  3. optimizer() → Update ALL parameters                                    │ │   │
│   │  │                                                                            │ │   │
│   │  │  Problem: Embedding gradients are sparse (only accessed rows have grads)   │ │   │
│   │  │           Standard optimizer loops over ALL parameters = wasteful          │ │   │
│   │  └────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                                                                  │   │
│   │  TorchRec Fused (2 steps with fused sparse update):                              │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │  1. forward()                → Compute activations                         │ │   │
│   │  │  2. backward() + fused_opt() → Compute gradients AND update sparse params  │ │   │
│   │  │     dense_optimizer()        → Update only dense params                    │ │   │
│   │  │                                                                            │ │   │
│   │  │  apply_optimizer_in_backward(RowWiseAdagrad, model.sparse_params, lr=0.1)  │ │   │
│   │  │                                                                            │ │   │
│   │  │  Result: ~15% faster training + lower memory peak                          │ │   │
│   │  └────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Optimization 3: Quantized Communication (QComms)                                       │
│   ─────────────────────────────────────────────────                                      │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │  Multi-node training: Network bandwidth = bottleneck                             │   │
│   │                                                                                  │   │
│   │  Without QComms:   FP32 embeddings sent across network (32 bits/value)          │   │
│   │  With QComms FP16: FP16 embeddings sent across network (16 bits/value)          │   │
│   │                    → 2x bandwidth reduction, minimal accuracy loss              │   │
│   │                                                                                  │   │
│   │  Code:                                                                           │   │
│   │  ┌──────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │  qcomms_config = QCommsConfig(                                           │   │   │
│   │  │      forward_precision=CommType.FP16,                                    │   │   │
│   │  │      backward_precision=CommType.BF16,                                   │   │   │
│   │  │  )                                                                       │   │   │
│   │  └──────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Integration

### Phase 1: Training the Retrieval Model

The retrieval model generates candidate items efficiently using approximate nearest neighbor (ANN) search.

```bash
# Navigate to retrieval example
cd torchrec/github/examples/retrieval

# Train two-tower model (distributed)
torchrun --nproc_per_node=2 two_tower_train.py --save_dir ./checkpoints

# This produces:
# - ./checkpoints/model.pt       (trained model weights)
# - ./checkpoints/faiss.index    (FAISS index with item embeddings)
```

**What happens:**
1. Query tower learns to embed users into a vector space
2. Candidate tower learns to embed items into the same space
3. Model is trained to maximize similarity for positive user-item pairs
4. Item embeddings are indexed in FAISS for fast retrieval

### Phase 2: Training the Ranking Model

The ranking model scores candidates with fine-grained features.

```bash
# Navigate to golden_training example
cd torchrec/github/examples/golden_training

# Train DLRM model (distributed)
torchrun --nproc_per_node=4 train_dlrm.py

# For production training with quantized comms:
torchx run -s local_cwd dist.ddp -j 1x4 --script train_dlrm.py
```

**What happens:**
1. Embedding tables are sharded across GPUs using model parallelism
2. Dense and sparse features are combined with feature interactions
3. Model learns to predict click/conversion probability
4. Checkpoints are saved for inference

### Phase 3: Inference Pipeline

#### Stage 1: Candidate Retrieval

```bash
cd torchrec/github/examples/retrieval

# Load trained model and run retrieval
python two_tower_retrieval.py --load_dir ./checkpoints
```

**Code integration point:**
```python
# In your serving code, use TwoTowerRetrieval
from modules.two_tower import TwoTowerRetrieval

# Initialize retrieval model with trained weights
retrieval_model = TwoTowerRetrieval(
    faiss_index=faiss_index,
    query_ebc=query_ebc,
    candidate_ebc=candidate_ebc,
    layer_sizes=[128, 64],
    k=100,  # Retrieve top 100 candidates
)

# Get candidates for a user
candidates = retrieval_model(user_query_kjt)
```

#### Stage 2: Candidate Ranking

```bash
cd torchrec/github/examples/prediction

# Run ranking on retrieved candidates
python predict_using_torchrec.py
```

**Code integration point:**
```python
# In your serving code, use the DLRM model
from predict_using_torchrec import TorchRecDLRM, DLRMRatingWrapper

# Load trained ranking model
ranking_model = DLRMRatingWrapper(dlrm_model)
ranking_model.load_state_dict(torch.load("ranking_model.pt"))

# Score the candidates from retrieval
scores = ranking_model(dense_features, candidate_sparse_features)

# Return top-K ranked items
top_k_indices = torch.argsort(scores, descending=True)[:10]
```

## Complete Integration Example

Here's a complete example combining retrieval and ranking:

```python
#!/usr/bin/env python3
"""
Complete recommendation pipeline combining retrieval and ranking.
"""

import torch
import faiss
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Import from examples
from retrieval.modules.two_tower import TwoTowerRetrieval
from prediction.predict_using_torchrec import TorchRecDLRM, DLRMRatingWrapper


class RecommendationPipeline:
    """
    End-to-end recommendation pipeline.

    Combines:
    - Two-tower retrieval for candidate generation
    - DLRM ranking for final scoring
    """

    def __init__(
        self,
        retrieval_model: TwoTowerRetrieval,
        ranking_model: DLRMRatingWrapper,
        num_candidates: int = 100,
        num_results: int = 10,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.retrieval_model = retrieval_model
        self.ranking_model = ranking_model
        self.num_candidates = num_candidates
        self.num_results = num_results
        self.device = device

        # Set models to eval mode
        self.retrieval_model.eval()
        self.ranking_model.eval()

    @torch.no_grad()
    def recommend(
        self,
        user_features: KeyedJaggedTensor,
        user_dense_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate recommendations for a user.

        Args:
            user_features: Sparse user features as KJT
            user_dense_features: Dense user features

        Returns:
            Tensor of recommended item IDs
        """
        # Stage 1: Retrieval - get candidate items
        # TwoTowerRetrieval returns logits for top-k candidates
        retrieval_logits = self.retrieval_model(user_features)

        # Get candidate IDs from FAISS index (stored in model)
        # In practice, you'd extract these from the retrieval model
        candidate_ids = self._get_candidate_ids(retrieval_logits)

        # Stage 2: Ranking - score candidates
        # Build features for candidate items
        candidate_sparse_features = self._build_candidate_features(candidate_ids)

        # Expand user features for each candidate
        batch_size = len(candidate_ids)
        expanded_dense = user_dense_features.expand(batch_size, -1)

        # Score candidates with ranking model
        ranking_scores = self.ranking_model(
            expanded_dense,
            candidate_sparse_features,
        )

        # Return top-K ranked items
        top_k_indices = torch.argsort(ranking_scores, descending=True)[:self.num_results]
        return candidate_ids[top_k_indices]

    def _get_candidate_ids(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract candidate IDs from retrieval output."""
        # Implementation depends on how TwoTowerRetrieval stores candidates
        # This is a placeholder
        return torch.arange(self.num_candidates, device=self.device)

    def _build_candidate_features(self, candidate_ids: torch.Tensor) -> KeyedJaggedTensor:
        """Build KJT for candidate items."""
        # In practice, look up item features from a feature store
        batch_size = len(candidate_ids)
        return KeyedJaggedTensor(
            keys=["item_id", "category"],
            values=torch.cat([candidate_ids, torch.zeros_like(candidate_ids)]),
            lengths=torch.ones(batch_size * 2, dtype=torch.int32, device=self.device),
        )


def create_pipeline(
    retrieval_checkpoint_dir: str,
    ranking_checkpoint_path: str,
    device: torch.device = torch.device("cuda:0"),
) -> RecommendationPipeline:
    """
    Create a recommendation pipeline from checkpoints.

    Args:
        retrieval_checkpoint_dir: Directory with retrieval model and FAISS index
        ranking_checkpoint_path: Path to ranking model checkpoint
        device: Device to run inference on

    Returns:
        Configured RecommendationPipeline
    """
    # Load FAISS index
    faiss_index = faiss.read_index(f"{retrieval_checkpoint_dir}/faiss.index")
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    # Create retrieval model (simplified - see actual example for full code)
    # ... model creation code ...

    # Create ranking model (simplified - see actual example for full code)
    # ... model creation code ...

    # Return pipeline
    # return RecommendationPipeline(retrieval_model, ranking_model)
    pass


# Usage example
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_pipeline(
        retrieval_checkpoint_dir="./retrieval/checkpoints",
        ranking_checkpoint_path="./golden_training/checkpoints/model.pt",
    )

    # Generate recommendations for a user
    user_features = KeyedJaggedTensor(
        keys=["user_id"],
        values=torch.tensor([12345]),
        lengths=torch.tensor([1]),
    )
    user_dense = torch.randn(1, 13)  # 13 dense features

    recommendations = pipeline.recommend(user_features, user_dense)
    print(f"Recommended items: {recommendations}")
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                        │
│                                                                              │
│   User Request                                                               │
│   {user_id: 12345, features: {...}}                                          │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     FEATURE PROCESSING                               │   │
│   │   retrieval/modules/feature_processor.py                            │   │
│   │                                                                      │   │
│   │   Raw Features → Validated → KeyedJaggedTensor                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     RETRIEVAL STAGE                                  │   │
│   │   retrieval/two_tower_retrieval.py                                  │   │
│   │                                                                      │   │
│   │   User KJT → Query Tower → Embedding → FAISS → 100 Candidates       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         │ candidate_ids: [item_1, item_2, ..., item_100]                    │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     FEATURE ENRICHMENT                               │   │
│   │   (Your feature store / database)                                   │   │
│   │                                                                      │   │
│   │   candidate_ids → Item Features → User×Item Features                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     RANKING STAGE                                    │   │
│   │   prediction/predict_using_torchrec.py                              │   │
│   │                                                                      │   │
│   │   Dense + Sparse Features → DLRM → Scores → Sort → Top 10          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│   Response: [item_42, item_17, item_89, ...]                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## TorchRec Components Used Across Examples

| Component | Retrieval | Golden Training | Prediction | Purpose |
|-----------|:---------:|:---------------:|:----------:|---------|
| `EmbeddingBagCollection` | ✅ | ✅ | ✅ | Embedding lookups |
| `KeyedJaggedTensor` | ✅ | ✅ | ✅ | Sparse feature format |
| `DistributedModelParallel` | ✅ | ✅ | ✅ | Model sharding |
| `TrainPipelineSparseDist` | ✅ | ✅ | ❌ | Pipelined training |
| `RowWiseAdagrad` | ✅ | ✅ | ❌ | Sparse optimizer |
| `QCommsConfig` | ❌ | ✅ | ❌ | Quantized communication |
| `quantize_embeddings` | ✅ | ❌ | ❌ | INT8 quantization |

## Performance Optimization Guide

### Training Optimizations

| Optimization | Where Applied | Impact |
|-------------|---------------|--------|
| Model Parallelism | Both | Enables large embedding tables |
| In-backward Optimizer | Golden Training | ~15% faster training |
| Quantized Comms | Golden Training | 2x bandwidth reduction |
| Pipeline Parallelism | Both | Overlap compute/comm |

### Inference Optimizations

| Optimization | Where Applied | Impact |
|-------------|---------------|--------|
| INT8 Quantization | Retrieval | 4x smaller embeddings |
| FAISS IVFPQ | Retrieval | ~100x faster than brute force |
| Batched Inference | Prediction | Better GPU utilization |
| TorchScript | Inference Legacy | JIT compilation |

## Migration Path

If you're building a new recommendation system:

1. **Start Simple** → Use `prediction/` example for a single-stage ranker
2. **Add Retrieval** → Integrate `retrieval/` for candidate generation
3. **Scale Training** → Use `golden_training/` patterns for distributed training
4. **Productionize** → Apply `inference_legacy/` patterns for serving

## Common Integration Patterns

### Pattern 1: Batch Inference Service

```python
# Process multiple users in parallel
user_batch_kjt = KeyedJaggedTensor.from_offsets_sync(...)
candidates_batch = retrieval_model(user_batch_kjt)
scores_batch = ranking_model(dense_batch, sparse_batch)
```

### Pattern 2: Real-time Serving with Caching

```python
# Cache retrieval results, re-rank on the fly
cached_candidates = redis.get(f"candidates:{user_id}")
if cached_candidates is None:
    cached_candidates = retrieval_model(user_kjt)
    redis.set(f"candidates:{user_id}", cached_candidates, ttl=3600)
scores = ranking_model(dense, sparse)
```

### Pattern 3: A/B Testing with Multiple Models

```python
# Route to different ranking models
if user_id % 100 < 10:  # 10% traffic
    scores = ranking_model_v2(dense, sparse)
else:
    scores = ranking_model_v1(dense, sparse)
```

## Troubleshooting Integration Issues

### Issue: Embedding Dimension Mismatch

**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Solution:** Ensure consistent `embedding_dim` across retrieval and ranking:
```python
# Both models should use the same dimension
EMBEDDING_DIM = 64  # or 128

# Retrieval
retrieval_config = EmbeddingBagConfig(embedding_dim=EMBEDDING_DIM, ...)

# Ranking
ranking_config = EmbeddingBagConfig(embedding_dim=EMBEDDING_DIM, ...)
```

### Issue: Feature Key Mismatch

**Symptom:** `KeyError: 'user_id'` or missing features

**Solution:** Verify KJT keys match model expectations:
```python
# Check what keys the model expects
print(model.embedding_bag_collection.embedding_bag_configs())

# Ensure your KJT has matching keys
kjt = KeyedJaggedTensor(
    keys=["user_id", "item_id"],  # Must match config
    values=...,
    lengths=...,
)
```

### Issue: Device Mismatch

**Symptom:** `RuntimeError: Expected all tensors to be on the same device`

**Solution:** Ensure consistent device placement:
```python
device = torch.device("cuda:0")
model = model.to(device)
kjt = kjt.to(device)
dense = dense.to(device)
```

## Next Steps

1. **Explore Individual Examples**: Start with the example most relevant to your use case
2. **Run the Tutorials**: See `sharding/` for interactive Jupyter notebooks
3. **Customize for Your Data**: Replace random data with your actual features
4. **Scale Up**: Use TorchX for multi-node training
5. **Production Deploy**: Apply quantization and optimization patterns

## References

- [TorchRec Documentation](https://pytorch.org/torchrec/)
- [TorchRec GitHub](https://github.com/pytorch/torchrec)
- [DLRM Paper](https://arxiv.org/abs/1906.00091)
- [Two-Tower Paper](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
