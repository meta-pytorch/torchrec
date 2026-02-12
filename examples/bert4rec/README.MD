# TorchRec Bert4Rec Example

`bert4rec_main.py` trains, validates, and tests a [BERT4REC](https://arxiv.org/abs/1904.06690) with TorchRec. The Bert4Rec is a Sequential Recommendation with Bidirectional Encoder Representations from Transformer and the model contains both data parallel components (e.g. transformation blocks) and model parallel components (e.g. item embeddings). The Bert4Rec model can be run with either a random dataloader or the movielens dataset (https://grouplens.org/datasets/movielens/). Notice that the final output layer of the model is different from the original paper which used the embedding layer weight to get the final output by matmul, from our experimentation, we used a linear layer instead which brings us higher performance

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              BERT4REC MODEL ARCHITECTURE                                 │
│                                                                                          │
│   Input: User's historical item sequence (e.g., last 16 items viewed)                   │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  User Sequence: [item_23, item_107, item_42, [MASK], item_88, item_15, ...]      │  │
│   │                                               ▲                                   │  │
│   │                                               │                                   │  │
│   │                              Masked item to predict (training)                    │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                         ITEM EMBEDDING LAYER                                      │  │
│   │                     (Model Parallel with TorchRec)                                │  │
│   │                                                                                   │  │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│   │   │   EmbeddingBagCollection                                                │    │  │
│   │   │   • 500K+ items × 128 dimensions                                        │    │  │
│   │   │   • Sharded across GPUs using DistributedModelParallel                  │    │  │
│   │   │   • Each item ID → dense embedding vector                               │    │  │
│   │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│   │                                                                                   │  │
│   │   Output: [batch, seq_len, embed_dim]                                            │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                      POSITIONAL ENCODING                                          │  │
│   │                                                                                   │  │
│   │   Add learnable position embeddings to capture sequence order                     │  │
│   │   Position embedding: [max_len, embed_dim]                                        │  │
│   │                                                                                   │  │
│   │   embedded_items = item_embeddings + position_embeddings                          │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                     TRANSFORMER ENCODER BLOCKS                                    │  │
│   │                      (Data Parallel - replicated)                                 │  │
│   │                                                                                   │  │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│   │   │  Block 1                                                                │    │  │
│   │   │  ┌────────────────────┐    ┌────────────────────┐                       │    │  │
│   │   │  │ Multi-Head Self    │    │ Feed-Forward       │                       │    │  │
│   │   │  │ Attention          │ ──►│ Network            │                       │    │  │
│   │   │  │ (8 heads)          │    │ (128→512→128)      │                       │    │  │
│   │   │  └────────────────────┘    └────────────────────┘                       │    │  │
│   │   │         + LayerNorm + Residual connections                              │    │  │
│   │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│   │                                    │                                              │  │
│   │                                    ▼                                              │  │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│   │   │  Block 2  (same structure)                                              │    │  │
│   │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│   │                                    │                                              │  │
│   │                                    ▼                                              │  │
│   │                                   ...                                             │  │
│   │                                    │                                              │  │
│   │                                    ▼                                              │  │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│   │   │  Block N                                                                │    │  │
│   │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                         OUTPUT LAYER                                              │  │
│   │                                                                                   │  │
│   │   Linear layer: [embed_dim → num_items]                                          │  │
│   │   Predicts probability distribution over all items                                │  │
│   │                                                                                   │  │
│   │   Output: [batch, seq_len, num_items] → softmax → item probabilities             │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Training Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           BERT4REC TRAINING ITERATION                                    │
│                                                                                          │
│  STEP 1: Data Preparation with Masking                                                   │
│  ══════════════════════════════════════                                                  │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │  Original Sequence:  [item_23, item_107, item_42, item_88, item_15, item_301] │    │
│     │                                                                               │    │
│     │  Masking (mask_prob = 0.2):                                                   │    │
│     │  ┌───────────────────────────────────────────────────────────────────────┐   │    │
│     │  │  80% → Replace with [MASK] token                                      │   │    │
│     │  │  10% → Replace with random item                                       │   │    │
│     │  │  10% → Keep original item                                             │   │    │
│     │  └───────────────────────────────────────────────────────────────────────┘   │    │
│     │                                                                               │    │
│     │  Masked Sequence: [item_23, [MASK], item_42, item_88, [MASK], item_301]      │    │
│     │  Labels:          [  -100,  107,     -100,    -100,    15,     -100  ]        │    │
│     │                           ▲                          ▲                        │    │
│     │                     Predict this            Predict this                      │    │
│     │                     (-100 = ignore)         (-100 = ignore)                   │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  STEP 2: Embedding Lookup (Model Parallel)                                               │
│  ═════════════════════════════════════════                                               │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │  DMP Mode (DistributedModelParallel):                                         │    │
│     │  ┌─────────────────────────────────────────────────────────────────────────┐ │    │
│     │  │                                                                          │ │    │
│     │  │  GPU 0                  GPU 1                  GPU 2                     │ │    │
│     │  │  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │ │    │
│     │  │  │ Items 0-166K │      │Items 166-333K│      │Items 333-500K│           │ │    │
│     │  │  │   (Shard 0)  │      │   (Shard 1)  │      │   (Shard 2)  │           │ │    │
│     │  │  └──────────────┘      └──────────────┘      └──────────────┘           │ │    │
│     │  │         │                     │                     │                    │ │    │
│     │  │         └─────────────────────┴─────────────────────┘                    │ │    │
│     │  │                               │                                          │ │    │
│     │  │                          All-to-All                                      │ │    │
│     │  │                        Communication                                     │ │    │
│     │  │                               │                                          │ │    │
│     │  │                               ▼                                          │ │    │
│     │  │              Each GPU gets embeddings for its batch                      │ │    │
│     │  └─────────────────────────────────────────────────────────────────────────┘ │    │
│     │                                                                               │    │
│     │  DDP Mode (Data Parallel):                                                    │    │
│     │  ┌─────────────────────────────────────────────────────────────────────────┐ │    │
│     │  │  Each GPU has FULL copy of embedding table                              │ │    │
│     │  │  • Simpler but uses more memory                                         │ │    │
│     │  │  • Gradient sync via AllReduce                                          │ │    │
│     │  └─────────────────────────────────────────────────────────────────────────┘ │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  STEP 3: Transformer Forward Pass                                                        │
│  ═══════════════════════════════                                                         │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │  Self-Attention: Each position attends to ALL positions (bidirectional)      │    │
│     │                                                                               │    │
│     │  Attention Matrix (seq_len × seq_len):                                        │    │
│     │  ┌─────────────────────────────────────────┐                                 │    │
│     │  │      pos0  pos1  pos2  pos3  pos4  pos5 │                                 │    │
│     │  │ pos0  ■     ■     ■     ■     ■     ■   │  ■ = attention weight           │    │
│     │  │ pos1  ■     ■     ■     ■     ■     ■   │                                 │    │
│     │  │ pos2  ■     ■     ■     ■     ■     ■   │  Unlike GPT (causal),           │    │
│     │  │ pos3  ■     ■     ■     ■     ■     ■   │  BERT sees FULL context         │    │
│     │  │ pos4  ■     ■     ■     ■     ■     ■   │  (past + future)                │    │
│     │  │ pos5  ■     ■     ■     ■     ■     ■   │                                 │    │
│     │  └─────────────────────────────────────────┘                                 │    │
│     │                                                                               │    │
│     │  Multi-Head Attention (8 heads):                                              │    │
│     │  • Each head learns different attention patterns                              │    │
│     │  • Concatenate heads → Linear projection                                      │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  STEP 4: Loss Computation (Masked Language Model)                                        │
│  ═══════════════════════════════════════════════                                         │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │  Output logits: [batch, seq_len, num_items]                                   │    │
│     │                                                                               │    │
│     │  Only compute loss on MASKED positions:                                       │    │
│     │                                                                               │    │
│     │  ┌────────────────────────────────────────────────────────────────────────┐  │    │
│     │  │  Position 1 (was masked):                                              │  │    │
│     │  │    logits[1] = [0.1, 0.2, ..., 2.5, ...]  (500K values)               │  │    │
│     │  │    label = 107                                                         │  │    │
│     │  │    loss = CrossEntropyLoss(logits[1], 107)                            │  │    │
│     │  │                                                                        │  │    │
│     │  │  Position 4 (was masked):                                              │  │    │
│     │  │    logits[4] = [0.3, 0.1, ..., 1.8, ...]                              │  │    │
│     │  │    label = 15                                                          │  │    │
│     │  │    loss = CrossEntropyLoss(logits[4], 15)                             │  │    │
│     │  │                                                                        │  │    │
│     │  │  Total Loss = mean(losses for all masked positions)                    │  │    │
│     │  └────────────────────────────────────────────────────────────────────────┘  │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  STEP 5: Backward Pass & Gradient Sync                                                   │
│  ══════════════════════════════════════                                                  │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │  DMP Mode:                                                                    │    │
│     │  ┌─────────────────────────────────────────────────────────────────────────┐ │    │
│     │  │  • Embedding gradients: All-to-All (send grads to owning GPU)          │ │    │
│     │  │  • Transformer gradients: AllReduce (sync across all GPUs)             │ │    │
│     │  │  • Optimizer: RowWiseAdagrad for embeddings, Adam for transformer      │ │    │
│     │  └─────────────────────────────────────────────────────────────────────────┘ │    │
│     │                                                                               │    │
│     │  DDP Mode:                                                                    │    │
│     │  ┌─────────────────────────────────────────────────────────────────────────┐ │    │
│     │  │  • All gradients: AllReduce (sync full gradients across all GPUs)      │ │    │
│     │  │  • Standard PyTorch DDP behavior                                        │ │    │
│     │  └─────────────────────────────────────────────────────────────────────────┘ │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## DMP vs DDP Mode Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        DMP (Model Parallel) vs DDP (Data Parallel)                       │
│                                                                                          │
│   ┌──────────────────────────────────┐    ┌──────────────────────────────────┐          │
│   │  DDP Mode (--mode ddp)           │    │  DMP Mode (--mode dmp)           │          │
│   │  ════════════════════            │    │  ════════════════════            │          │
│   │                                  │    │                                  │          │
│   │  GPU 0        GPU 1              │    │  GPU 0        GPU 1              │          │
│   │  ┌────────┐   ┌────────┐         │    │  ┌────────┐   ┌────────┐         │          │
│   │  │Full Emb│   │Full Emb│         │    │  │Emb     │   │Emb     │         │          │
│   │  │(500K)  │   │(500K)  │         │    │  │Shard 0 │   │Shard 1 │         │          │
│   │  │        │   │        │         │    │  │(250K)  │   │(250K)  │         │          │
│   │  ├────────┤   ├────────┤         │    │  ├────────┤   ├────────┤         │          │
│   │  │Transf. │   │Transf. │         │    │  │Transf. │   │Transf. │         │          │
│   │  │(copy)  │   │(copy)  │         │    │  │(copy)  │   │(copy)  │         │          │
│   │  └────────┘   └────────┘         │    │  └────────┘   └────────┘         │          │
│   │                                  │    │                                  │          │
│   │  Memory: 2× embedding size       │    │  Memory: 1× embedding size       │          │
│   │  Communication: AllReduce only   │    │  Communication: All-to-All       │          │
│   │                                  │    │                                  │          │
│   │  ✓ Simpler to understand         │    │  ✓ Scales to larger models       │          │
│   │  ✗ Limited by GPU memory         │    │  ✓ 33-40% faster (see results)   │          │
│   │  ✗ Slower gradient sync          │    │  ✓ Memory efficient              │          │
│   │                                  │    │                                  │          │
│   └──────────────────────────────────┘    └──────────────────────────────────┘          │
│                                                                                          │
│   Performance Comparison (8 GPUs, 500K items):                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   DDP: ████████████████████████████████░░░░░░░░░░░░░░░░  1592 rec/s             │   │
│   │   DMP: ████████████████████████████████████████████████  2225 rec/s  (+39.8%)   │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           BERT4REC INFERENCE (Next Item Prediction)                      │
│                                                                                          │
│   User's Recent History:                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  [item_23, item_107, item_42, item_88, item_15, ???]                              │  │
│   │                                                    ▲                              │  │
│   │                                           What should we recommend?               │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Append [MASK] token at the end:                                                  │  │
│   │  [item_23, item_107, item_42, item_88, item_15, [MASK]]                           │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Model Forward Pass:                                                              │  │
│   │                                                                                   │  │
│   │  Embeddings → Transformer Blocks → Output Layer                                   │  │
│   │                                                                                   │  │
│   │  Output: logits[MASK_position] = [score_0, score_1, ..., score_499999]            │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Top-K Selection:                                                                 │  │
│   │                                                                                   │  │
│   │  softmax(logits) → probabilities                                                  │  │
│   │  top_k(probabilities, k=10) → [item_301, item_42, item_88, ...]                   │  │
│   │                                                                                   │  │
│   │  Recommendation: item_301 (highest probability)                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   Evaluation Metrics:                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  • HR@K (Hit Rate):  Was the true next item in top-K predictions?               │   │
│   │  • NDCG@K:           Normalized Discounted Cumulative Gain (rank-aware metric)  │   │
│   │                                                                                  │   │
│   │  Higher is better for both metrics (range: 0 to 1)                               │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
bert4rec/
├── README.MD                              # This file
├── BUCK                                   # Build configuration
├── bert4rec_main.py                       # Main training script
├── bert4rec_metrics.py                    # Evaluation metrics (HR@K, NDCG@K)
├── data/
│   ├── BUCK
│   └── bert4rec_movielens_datasets.py    # MovieLens data loading
├── dataloader/
│   ├── BUCK
│   └── bert4rec_movielens_dataloader.py  # Batch preparation with masking
├── models/
│   ├── BUCK
│   └── bert4rec.py                       # BERT4REC model definition
└── tests/
    └── test_bert4rec_main.py             # Unit tests
```

## Tested Cloud Instances
| Cloud | Instance Type | GPUs     | vCPUs | Memory (GB) |
|-------|---------------|----------|-------|-------------|
| AWS   | p3.16xlarge   | 8 x V100 | 64    | 488         |

# Running

## Torchx
We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) to run. Here we use the [DDP builtin](https://pytorch.org/torchx/main/components/distributed.html)

1. pip install torchx
2. (optional) setup a slurm or kubernetes cluster
3.
    a. locally for DMP: `torchx run -s local_cwd dist.ddp -j 1x2 --script bert4rec_main.py -- --dataset_name random --mode dmp`
    b. locally for DDP: `torchx run -s local_cwd dist.ddp -j 1x2 --script bert4rec_main.py -- --dataset_name random --mode ddp`
    c. remotely for DMP: `torchx run -s slurm dist.ddp -j 1x8 --script bert4rec_main.py -- --dataset_name random --mode dmp`
    d. remotely for DDP: `torchx run -s slurm dist.ddp -j 1x8 --script bert4rec_main.py -- --dataset_name random --mode ddp`


## Preliminary Training Results

**Setup:**
* Dataset: Random Dataset
* CUDA 11.3, NCCL 2.10.3.
* AWS p3.16xlarge instances, each with 8 Tesla V100s.
* Shared flags for all runs include: `--dataset_name random --random_user_count 3000 --random_item_count 500000 --random_size 8000000 --lr 0.001 --mask_prob 0.2 --train_batch_size 8 --val_batch_size 8 --max_len 16 --emb_dim 128 --num_epochs 1 --mode dmp`


**Results**

###### Recommendation Metrics Reproduce

| Dataset | Metrics | Paper  | Repro  |
|---------|---------|--------|--------|
| ML-1M   | HR@1    | 0.2863 | 0.2636 |
|         | HR@5    | 0.5876 | 0.5895 |
|         | HR@10   | 0.6970 | 0.7078 |
|         | NDCG@5  | 0.4454 | 0.4356 |
|         | NDCG@10 | 0.4818 | 0.4740 |
| ML-20M  | HR@1    | 0.3440 | 0.3273 |
|         | HR@5    | 0.6323 | 0.6399 |
|         | HR@10   | 0.7473 | 0.7601 |
|         | NDCG@5  | 0.4967 | 0.4929 |
|         | NDCG@10 | 0.5340 | 0.5319 |

###### Infra Metrics Improvement

|Number of GPUs|Local Batch Size|Global Batch Size|DDP Train Records/Second|DMP Train Records/Second|Improvements |
| --- | --- | --- | --- | --- | --- |
|1|8|8|242.56 rec/s| 324.24 rec/s | 33.6%
|4|8|32|757.12 rec/s| 1058.56 rec/s | 39.8%
|8|8|64|1592.32 rec/s | 2225.28 rec/s | 39.8%

**Reproduce**

Run the following command to reproduce the results for a single node (8 GPUs) on AWS.

```
torchx run -s local_cwd dist.ddp -j 1x8 --script bert4rec_folder/bert4rec_main.py -- --dataset_name random --random_user_count 3000 --random_item_count 500000 --random_size 8000000 --lr 0.001 --mask_prob 0.2 --train_batch_size 8 --val_batch_size 8 --max_len 16 --emb_dim 128 --num_epochs 1 --mode dmp
```
Upon scheduling the job, there should be an output that looks like this:

```
torchx 2022-03-23 05:38:40 INFO     Log files located in: /tmp/torchx_fby9w0w9/torchx/bert4rec_main-txq11m6736w20c/bert4rec_main/0
local_cwd://torchx/bert4rec_main-txq11m6736w20c
torchx 2022-03-23 05:38:40 INFO     Waiting for the app to finish...
bert4rec_main/0 WARNING:__main__:
bert4rec_main/0 *****************************************
bert4rec_main/0 Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
bert4rec_main/0 *****************************************
bert4rec_main/0 [2]:Filtering triplets
bert4rec_main/0 [7]:Filtering triplets
bert4rec_main/0 [4]:Filtering triplets
bert4rec_main/0 [6]:Filtering triplets
bert4rec_main/0 [3]:Filtering triplets
bert4rec_main/0 [5]:Filtering triplets
bert4rec_main/0 [2]:Densifying index
bert4rec_main/0 [7]:Densifying index
bert4rec_main/0 [4]:Densifying index
bert4rec_main/0 [6]:Densifying index
bert4rec_main/0 [3]:Densifying index
bert4rec_main/0 [5]:Densifying index
bert4rec_main/0 [1]:Filtering triplets
bert4rec_main/0 [0]:Filtering triplets
bert4rec_main/0 [1]:Densifying index
bert4rec_main/0 [0]:Densifying index
bert4rec_main/0 [2]:Splitting
bert4rec_main/0 [2]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f52d986dcd0>
bert4rec_main/0 [6]:Splitting
bert4rec_main/0 [6]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f22b398bbe0>
bert4rec_main/0 [7]:Splitting
bert4rec_main/0 [7]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f90fe1b3be0>
bert4rec_main/0 [4]:Splitting
bert4rec_main/0 [4]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f6f53e3ccd0>
bert4rec_main/0 [3]:Splitting
bert4rec_main/0 [3]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f3babf7ecd0>
bert4rec_main/0 [5]:Splitting
bert4rec_main/0 [5]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f675bedbcd0>
bert4rec_main/0 [1]:Splitting
bert4rec_main/0 [1]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f84351f7cd0>
```

**TODO/Work In Progress**
* Run QPS test on A100s machines
* Use Ads Synthetic data to get the QPS number
