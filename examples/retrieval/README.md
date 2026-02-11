# TorchRec Two-Tower Retrieval Example

This example demonstrates building a production-ready **two-tower recommendation system** using TorchRec, complete with distributed training, FAISS indexing for fast candidate retrieval, and quantized inference.

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  TRAINING PHASE                          │
                    │  ┌─────────────────┐       ┌─────────────────┐          │
                    │  │   Query Tower    │       │ Candidate Tower │          │
                    │  │   (User Side)    │       │   (Item Side)   │          │
                    │  │                  │       │                 │          │
                    │  │ ┌──────────────┐ │       │ ┌─────────────┐ │          │
                    │  │ │EmbeddingBag  │ │       │ │EmbeddingBag │ │          │
                    │  │ │Collection    │ │       │ │Collection   │ │          │
                    │  │ └──────┬───────┘ │       │ └──────┬──────┘ │          │
                    │  │        │         │       │        │        │          │
                    │  │ ┌──────▼───────┐ │       │ ┌──────▼──────┐ │          │
                    │  │ │  MLP Layers  │ │       │ │ MLP Layers  │ │          │
                    │  │ └──────┬───────┘ │       │ └──────┬──────┘ │          │
                    │  │        │         │       │        │        │          │
                    │  └────────┼─────────┘       └────────┼────────┘          │
                    │           │                          │                   │
                    │           └──────────┬───────────────┘                   │
                    │                      │                                   │
                    │               Dot Product Loss                           │
                    │              (BCEWithLogitsLoss)                         │
                    └─────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────┐
                    │                  INFERENCE PHASE                         │
                    │                                                          │
                    │  User Query ──► Query Tower ──► Query Embedding          │
                    │                                        │                 │
                    │                                        ▼                 │
                    │                              ┌─────────────────┐         │
                    │                              │   FAISS Index   │         │
                    │                              │    (IVFPQ)      │         │
                    │                              │                 │         │
                    │                              │ Contains all    │         │
                    │                              │ item embeddings │         │
                    │                              └────────┬────────┘         │
                    │                                       │                  │
                    │                                       ▼                  │
                    │                              Top-K Candidates            │
                    │                              (for ranking stage)         │
                    └─────────────────────────────────────────────────────────┘
```

## Training Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        TWO-TOWER TRAINING ITERATION                                  │
│                                                                                      │
│  STEP 1: Data Loading                                                                │
│  ════════════════════                                                                │
│                                                                                      │
│     Raw Data (Parquet/CSV)                                                           │
│           │                                                                          │
│           ▼                                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐             │
│     │  DataLoader with RandomRecDataset                                │             │
│     │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │             │
│     │  │  Batch 0    │  │  Batch 1    │  │  Batch 2    │  ...         │             │
│     │  │ user_ids    │  │ user_ids    │  │ user_ids    │              │             │
│     │  │ item_ids    │  │ item_ids    │  │ item_ids    │              │             │
│     │  │ labels      │  │ labels      │  │ labels      │              │             │
│     │  └─────────────┘  └─────────────┘  └─────────────┘              │             │
│     └─────────────────────────────────────────────────────────────────┘             │
│           │                                                                          │
│           ▼                                                                          │
│  STEP 2: Feature Transformation to KJT                                               │
│  ════════════════════════════════════                                                │
│                                                                                      │
│     ┌──────────────────────────────────────────────────────────────────┐            │
│     │           KeyedJaggedTensor (KJT) Format                          │            │
│     │                                                                   │            │
│     │   keys:    ["user_id", "item_id"]                                │            │
│     │   values:  [u1, u2, u3, u4, i1, i2, i3, i4]  ← flattened IDs     │            │
│     │   lengths: [1, 1, 1, 1, 1, 1, 1, 1]          ← features per user │            │
│     │                                                                   │            │
│     │   Enables efficient batched embedding lookups                     │            │
│     └──────────────────────────────────────────────────────────────────┘            │
│           │                                                                          │
│           ▼                                                                          │
│  STEP 3: Forward Pass (Parallel on Both Towers)                                      │
│  ═════════════════════════════════════════════                                       │
│                                                                                      │
│     ┌────────────────────────────────────────────────────────────────────────┐      │
│     │                                                                         │      │
│     │   Query Tower (User)              Candidate Tower (Item)               │      │
│     │   ────────────────────            ──────────────────────               │      │
│     │                                                                         │      │
│     │   user_kjt                        item_kjt                              │      │
│     │       │                               │                                 │      │
│     │       ▼                               ▼                                 │      │
│     │   ┌─────────────────┐            ┌─────────────────┐                   │      │
│     │   │ EmbeddingBag    │            │ EmbeddingBag    │                   │      │
│     │   │ Lookup          │            │ Lookup          │                   │      │
│     │   │ (Sharded        │            │ (Sharded        │                   │      │
│     │   │  across GPUs)   │            │  across GPUs)   │                   │      │
│     │   └────────┬────────┘            └────────┬────────┘                   │      │
│     │            │                              │                             │      │
│     │            ▼                              ▼                             │      │
│     │   user_embeddings                 item_embeddings                       │      │
│     │   [batch, embed_dim]              [batch, embed_dim]                    │      │
│     │            │                              │                             │      │
│     │            ▼                              ▼                             │      │
│     │   ┌─────────────────┐            ┌─────────────────┐                   │      │
│     │   │ MLP Projection  │            │ MLP Projection  │                   │      │
│     │   │ 64 → 128 → 64   │            │ 64 → 128 → 64   │                   │      │
│     │   └────────┬────────┘            └────────┬────────┘                   │      │
│     │            │                              │                             │      │
│     │            ▼                              ▼                             │      │
│     │   query_embedding                 candidate_embedding                   │      │
│     │   [batch, 64]                     [batch, 64]                           │      │
│     │                                                                         │      │
│     └────────────────────────────────────────────────────────────────────────┘      │
│           │                               │                                          │
│           └───────────────┬───────────────┘                                          │
│                           │                                                          │
│                           ▼                                                          │
│  STEP 4: Loss Computation                                                            │
│  ════════════════════════                                                            │
│                                                                                      │
│     ┌──────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                           │    │
│     │   logits = dot_product(query_embedding, candidate_embedding)              │    │
│     │          = Σ (query[i] * candidate[i])  for each dimension                │    │
│     │                                                                           │    │
│     │   loss = BCEWithLogitsLoss(logits, labels)                                │    │
│     │        = -[y * log(σ(x)) + (1-y) * log(1 - σ(x))]                         │    │
│     │                                                                           │    │
│     │   where: σ = sigmoid, y = label (0 or 1), x = logit                       │    │
│     │                                                                           │    │
│     └──────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                          │
│           ▼                                                                          │
│  STEP 5: Backward Pass & Optimizer Update                                            │
│  ═══════════════════════════════════════                                             │
│                                                                                      │
│     ┌──────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                           │    │
│     │   loss.backward()  ──►  Compute gradients for:                            │    │
│     │                         • MLP weights (dense)                             │    │
│     │                         • Embedding tables (sparse)                       │    │
│     │                                                                           │    │
│     │   Dense Optimizer:     SGD/Adam on MLP parameters                         │    │
│     │   Sparse Optimizer:    RowWiseAdagrad (fused with backward)               │    │
│     │                                                                           │    │
│     │   ┌─────────────────────────────────────────────────────────────────┐    │    │
│     │   │  In-Backward Optimizer Fusion (for embeddings):                  │    │    │
│     │   │                                                                  │    │    │
│     │   │  Standard:  forward → backward → optimizer.step()                │    │    │
│     │   │  Fused:     forward → backward + optimizer.step() (combined)     │    │    │
│     │   │                                                                  │    │    │
│     │   │  Result: ~15% faster training for sparse parameters              │    │    │
│     │   └─────────────────────────────────────────────────────────────────┘    │    │
│     │                                                                           │    │
│     └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Distributed Training Visualization

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-GPU TRAINING WITH MODEL PARALLELISM                      │
│                                                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │                         DistributedModelParallel (DMP)                           │ │
│   │                                                                                  │ │
│   │   Sharding: Table-wise partitioning of embedding tables across GPUs             │ │
│   │                                                                                  │ │
│   │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │ │
│   │   │    GPU 0      │  │    GPU 1      │  │    GPU 2      │  │    GPU 3      │   │ │
│   │   │               │  │               │  │               │  │               │   │ │
│   │   │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │   │ │
│   │   │ │ User Emb  │ │  │ │ User Emb  │ │  │ │ Item Emb  │ │  │ │ Item Emb  │ │   │ │
│   │   │ │ (Shard 0) │ │  │ │ (Shard 1) │ │  │ │ (Shard 0) │ │  │ │ (Shard 1) │ │   │ │
│   │   │ │           │ │  │ │           │ │  │ │           │ │  │ │           │ │   │ │
│   │   │ │ IDs 0-250K│ │  │ │IDs 250K-  │ │  │ │ IDs 0-250K│ │  │ │IDs 250K-  │ │   │ │
│   │   │ │           │ │  │ │    500K   │ │  │ │           │ │  │ │    500K   │ │   │ │
│   │   │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │   │ │
│   │   │               │  │               │  │               │  │               │   │ │
│   │   │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │   │ │
│   │   │ │ MLP Copy  │ │  │ │ MLP Copy  │ │  │ │ MLP Copy  │ │  │ │ MLP Copy  │ │   │ │
│   │   │ │ (Dense)   │ │  │ │ (Dense)   │ │  │ │ (Dense)   │ │  │ │ (Dense)   │ │   │ │
│   │   │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │   │ │
│   │   └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │ │
│   │          │                  │                  │                  │            │ │
│   │          └──────────────────┴──────────────────┴──────────────────┘            │ │
│   │                                      │                                          │ │
│   │                                      ▼                                          │ │
│   │                         ┌─────────────────────────┐                             │ │
│   │                         │   NCCL All-to-All       │                             │ │
│   │                         │   Communication         │                             │ │
│   │                         │                         │                             │ │
│   │                         │ Gather embeddings from  │                             │ │
│   │                         │ all shards to each GPU  │                             │ │
│   │                         └─────────────────────────┘                             │ │
│   │                                                                                  │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │                    TrainPipelineSparseDist (Communication Overlap)               │ │
│   │                                                                                  │ │
│   │   Timeline (3 consecutive batches):                                             │ │
│   │                                                                                  │ │
│   │   Time ──────────────────────────────────────────────────────────────────────►  │ │
│   │                                                                                  │ │
│   │   Batch N:   [Data Load]  [All2All Comm]  [Forward]  [Backward]                 │ │
│   │   Batch N+1:              [Data Load]     [All2All]  [Forward]   [Backward]     │ │
│   │   Batch N+2:                              [Data Load][All2All]   [Forward] ...  │ │
│   │                                                                                  │ │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐   │ │
│   │   │                        OVERLAP ZONES                                     │   │ │
│   │   │                                                                          │   │ │
│   │   │  • Data loading overlaps with computation                                │   │ │
│   │   │  • All-to-All communication overlaps with computation                    │   │ │
│   │   │  • Achieves near 100% GPU utilization                                    │   │ │
│   │   └─────────────────────────────────────────────────────────────────────────┘   │ │
│   │                                                                                  │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Embedding Lookup Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           KEYED JAGGED TENSOR → EMBEDDING LOOKUP                         │
│                                                                                          │
│   Input: KeyedJaggedTensor (variable-length features per sample)                         │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Example: User features for batch of 4 users                                      │  │
│   │                                                                                   │  │
│   │  User 0: visited pages [101, 203, 57]     (3 pages)                               │  │
│   │  User 1: visited pages [42]               (1 page)                                │  │
│   │  User 2: visited pages [88, 15]           (2 pages)                               │  │
│   │  User 3: visited pages [301, 22, 19, 7]   (4 pages)                               │  │
│   │                                                                                   │  │
│   │  KJT Representation:                                                              │  │
│   │  ┌────────────────────────────────────────────────────────────────────┐          │  │
│   │  │  keys:    ["page_id"]                                              │          │  │
│   │  │  values:  [101, 203, 57, 42, 88, 15, 301, 22, 19, 7]  (flattened)  │          │  │
│   │  │  lengths: [3, 1, 2, 4]                                  (per user)  │          │  │
│   │  └────────────────────────────────────────────────────────────────────┘          │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                           EmbeddingBagCollection                                  │  │
│   │                                                                                   │  │
│   │   Embedding Table "page_id" (1M pages × 64 dimensions)                            │  │
│   │   ┌──────────────────────────────────────────────────────────────┐               │  │
│   │   │  ID  │  Dim 0  │  Dim 1  │  Dim 2  │  ...  │  Dim 63 │       │               │  │
│   │   │──────│─────────│─────────│─────────│───────│─────────│       │               │  │
│   │   │   0  │  0.12   │ -0.34   │  0.56   │  ...  │  0.78   │       │               │  │
│   │   │   1  │  0.23   │  0.45   │ -0.67   │  ...  │  0.89   │       │               │  │
│   │   │  ... │   ...   │   ...   │   ...   │  ...  │   ...   │       │               │  │
│   │   │ 101  │  0.11   │  0.22   │  0.33   │  ...  │  0.44   │ ◄──── Lookup          │  │
│   │   │  ... │   ...   │   ...   │   ...   │  ...  │   ...   │       │               │  │
│   │   └──────────────────────────────────────────────────────────────┘               │  │
│   │                                                                                   │  │
│   │   Pooling: "sum" (combine multiple embeddings per user)                          │  │
│   │                                                                                   │  │
│   │   User 0: emb[101] + emb[203] + emb[57]  →  pooled_emb_0 [64-dim]                │  │
│   │   User 1: emb[42]                         →  pooled_emb_1 [64-dim]                │  │
│   │   User 2: emb[88] + emb[15]              →  pooled_emb_2 [64-dim]                │  │
│   │   User 3: emb[301]+emb[22]+emb[19]+emb[7]→  pooled_emb_3 [64-dim]                │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Output: KeyedTensor                                                              │  │
│   │                                                                                   │  │
│   │  ┌────────────────────────────────────────────────────────────────────┐          │  │
│   │  │  keys:   ["page_id"]                                               │          │  │
│   │  │  values: Tensor[4, 64]  ← (batch_size, embedding_dim)              │          │  │
│   │  └────────────────────────────────────────────────────────────────────┘          │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## What This Example Demonstrates

### TorchRec Features
- **`EmbeddingBagCollection`**: Efficient embedding lookups for categorical features
- **`DistributedModelParallel`**: Model-parallel sharding of embedding tables across GPUs
- **`TrainPipelineSparseDist`**: Overlapped communication and computation for training
- **`KeyedJaggedTensor`**: Efficient sparse tensor format for variable-length features
- **`RowWiseAdagrad`**: Row-wise optimizer for efficient embedding training
- **Quantization**: INT8 quantization for efficient inference using `torchrec.quant`

### Production Patterns
- **Two-tower architecture**: Separate encoders for queries (users) and candidates (items)
- **FAISS integration**: IVFPQ index for approximate nearest neighbor search
- **Model serialization**: Saving and loading trained models
- **Distributed inference**: Quantized, sharded model for production serving

## Directory Structure

```
retrieval/
├── README.md                    # This file
├── BUCK                         # Build configuration
├── __init__.py
├── data/
│   ├── __init__.py
│   └── dataloader.py           # MovieLens-style data loading
├── modules/
│   ├── __init__.py
│   └── two_tower.py            # TwoTower, TwoTowerTrainTask, TwoTowerRetrieval
├── knn_index.py                # FAISS index utilities
├── two_tower_train.py          # Training script
├── two_tower_retrieval.py      # Inference script
└── tests/
    ├── __init__.py
    ├── test_two_tower_train.py
    └── test_two_tower_retrieval.py
```

## Scripts

### `two_tower_train.py` - Training

Trains a two-tower model with the following features:
- **Distributed training** using `DistributedModelParallel` for model-parallel embedding sharding
- **Pipelined training** with overlapped data loading, communication, and computation
- **FAISS index building** from trained item embeddings
- **Model checkpointing** for later inference

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_embeddings` | 1M | Size of embedding tables |
| `embedding_dim` | 64 | Embedding dimension |
| `layer_sizes` | [128, 64] | MLP hidden layer sizes |
| `learning_rate` | 0.01 | Learning rate for RowWiseAdagrad |
| `batch_size` | 32 | Training batch size |
| `num_iterations` | 100 | Number of training iterations |
| `num_centroids` | 100 | FAISS IVF centroids |
| `num_subquantizers` | 8 | FAISS PQ subquantizers |

### `two_tower_retrieval.py` - Inference

Loads trained models and performs retrieval:
- **Model loading** from checkpoints
- **INT8 quantization** for efficient inference
- **FAISS index loading** for fast ANN search
- **End-to-end retrieval** wrapping query tower and FAISS in a single module

## Running the Example

### Prerequisites

```bash
# Install dependencies
conda install -c conda-forge pytorch faiss-gpu
pip install torchx torchrec
```

### Training

We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) for distributed execution:

```bash
# Local training (2 GPUs)
torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script two_tower_train.py -- --save_dir ./checkpoints

# Cluster training (8 GPUs on Slurm)
torchx run -s slurm dist.ddp -j 1x8 --gpu 8 --script two_tower_train.py -- --save_dir ./checkpoints

# Single-node multi-GPU (alternative)
torchrun --nproc_per_node=2 two_tower_train.py --save_dir ./checkpoints
```

### Inference

```bash
# Load trained model and run retrieval
CUDA_VISIBLE_DEVICES=0,1 python two_tower_retrieval.py --load_dir ./checkpoints
```

## Model Components

### TwoTower (`modules/two_tower.py`)

The main training model that embeds queries and candidates into the same vector space:

```python
class TwoTower(nn.Module):
    """
    Embeds two different entities (query/candidate) into the same space.

    Args:
        embedding_bag_collection: EBC with exactly 2 embedding tables
        layer_sizes: MLP layer sizes for projection
        device: Target device
    """
```

### TwoTowerTrainTask

Wraps `TwoTower` with loss computation for training:

```python
class TwoTowerTrainTask(nn.Module):
    """
    Training wrapper with BCEWithLogitsLoss.

    Forward returns:
        Tuple[loss, (loss_detached, logits, labels)]
    """
```

### TwoTowerRetrieval

Inference module that wraps query tower + FAISS index:

```python
class TwoTowerRetrieval(nn.Module):
    """
    Retrieval model for inference.

    Combines:
        - Query tower for user embedding
        - FAISS index for KNN search
        - Candidate tower for final scoring
    """
```

## FAISS Index Configuration

The example uses **IVFPQ** (Inverted File with Product Quantization) for memory-efficient ANN search:

| Parameter | Meaning | Tuning Guidance |
|-----------|---------|-----------------|
| `num_centroids` (nlist) | Voronoi cells for IVF | More = better recall, slower indexing |
| `num_probe` (nprobe) | Cells to search at query time | More = better recall, slower search |
| `num_subquantizers` | PQ compression subvectors | More = better accuracy, more memory |
| `bits_per_code` | Bits per subvector | 8 is typical; lower = more compression |

**Tuning tips:**
- Sweep `nprobe` in powers of 2 (1, 2, 4, 8, ...) and measure recall
- `nprobe = nlist` is equivalent to brute-force search
- Use recall@K metrics to find the right tradeoff

## Integration with Ranking

This retrieval example generates **candidate embeddings** that can be used in a two-stage recommendation system:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Retrieval  │────►│   Ranking    │────►│    Final     │
│  (This Ex.)  │     │  (DLRM-style)│     │   Results    │
│              │     │              │     │              │
│  ~100-1000   │     │  Score each  │     │   Top 10     │
│  candidates  │     │  candidate   │     │   items      │
└──────────────┘     └──────────────┘     └──────────────┘
```

For the ranking stage, see:
- [`golden_training/`](../golden_training/) - DLRM training example
- [`prediction/`](../prediction/) - DLRM inference example

## Performance Considerations

### Training
- Use `TrainPipelineSparseDist` for overlapped communication
- Enable quantized communication with `QCommsConfig` for multi-node training
- Consider `RowWiseAdagrad` for sparse embeddings (better than dense Adam)

### Inference
- Quantize embeddings to INT8 with `torchrec.quant`
- Use FAISS GPU index for faster retrieval
- Tune FAISS parameters based on recall requirements

## Common Issues & Troubleshooting

### 1. Out of Memory
- Reduce `num_embeddings` or `embedding_dim`
- Use fewer GPUs with larger per-GPU memory
- Enable gradient checkpointing

### 2. Slow Training
- Ensure NCCL backend is used for GPU training
- Use `TrainPipelineSparseDist` for communication overlap
- Check data loading isn't the bottleneck

### 3. Poor Recall
- Increase `nprobe` for FAISS search
- Train longer or with more data
- Verify embedding normalization

## References

- [TorchRec Documentation](https://pytorch.org/torchrec/)
- [Two-Tower Paper](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [DLRM Paper](https://arxiv.org/abs/1906.00091)
