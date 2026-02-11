# TorchRec Transfer Learning Example

This example showcases **training a distributed model using TorchRec with pretrained embeddings**. The embeddings are initialized with pretrained values (assumed to be loaded from storage, such as parquet). For large pretrained embeddings, we use the `share_memory_` API to efficiently load tensors from shared memory across multiple processes.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       TRANSFER LEARNING FOR RECOMMENDATION MODELS                        │
│                                                                                          │
│   Traditional Training (from scratch):                                                   │
│   ────────────────────────────────────                                                   │
│                                                                                          │
│   ┌──────────────────┐                     ┌──────────────────┐                         │
│   │  Random Init     │    Many Epochs      │  Trained Model   │                         │
│   │  Embeddings      │ ──────────────────► │  Embeddings      │                         │
│   │  (noise)         │    Slow convergence │  (meaningful)    │                         │
│   └──────────────────┘                     └──────────────────┘                         │
│                                                                                          │
│                                                                                          │
│   Transfer Learning (this example):                                                      │
│   ────────────────────────────────                                                       │
│                                                                                          │
│   ┌──────────────────┐                     ┌──────────────────┐                         │
│   │  Pretrained      │    Fewer Epochs     │  Fine-tuned      │                         │
│   │  Embeddings      │ ──────────────────► │  Embeddings      │                         │
│   │  (meaningful)    │    Fast convergence │  (task-specific) │                         │
│   └──────────────────┘                     └──────────────────┘                         │
│          ▲                                                                               │
│          │                                                                               │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Pretrained Sources:                                                              │  │
│   │  • Word2Vec / GloVe embeddings for text                                           │  │
│   │  • Item2Vec from collaborative filtering                                          │  │
│   │  • Embeddings from a related task (e.g., search → recommendations)                │  │
│   │  • Pre-computed embeddings from a larger model                                    │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Training Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING TRAINING WORKFLOW                                   │
│                                                                                          │
│  PHASE 1: Load Pretrained Embeddings                                                     │
│  ═══════════════════════════════════                                                     │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │   Storage (Parquet/HDF5/Pickle)                                               │    │
│     │   ┌─────────────────────────────────────────────────────────────────────┐    │    │
│     │   │  pretrained_embeddings.parquet                                      │    │    │
│     │   │  ┌─────────────────────────────────────────────────────────────┐   │    │    │
│     │   │  │  item_id │ embedding_dim_0 │ embedding_dim_1 │ ... │ dim_127│   │    │    │
│     │   │  │  ────────│─────────────────│─────────────────│─────│────────│   │    │    │
│     │   │  │     0    │     0.123       │     -0.456      │ ... │  0.789 │   │    │    │
│     │   │  │     1    │     0.234       │      0.567      │ ... │ -0.123 │   │    │    │
│     │   │  │    ...   │      ...        │       ...       │ ... │   ...  │   │    │    │
│     │   │  │  1000000 │     0.345       │     -0.678      │ ... │  0.456 │   │    │    │
│     │   │  └─────────────────────────────────────────────────────────────┘   │    │    │
│     │   └─────────────────────────────────────────────────────────────────────┘    │    │
│     │                                    │                                          │    │
│     │                                    ▼                                          │    │
│     │   Load into Shared Memory (for multi-process training)                        │    │
│     │   ┌─────────────────────────────────────────────────────────────────────┐    │    │
│     │   │  pretrained_tensor = torch.from_numpy(df.values)                    │    │    │
│     │   │  pretrained_tensor.share_memory_()  # Move to shared memory         │    │    │
│     │   │                                                                     │    │    │
│     │   │  ┌─────────────────────────────────────────────────────────────┐   │    │    │
│     │   │  │  Shared Memory Region (accessible by all processes)         │   │    │    │
│     │   │  │  ┌───────────────────────────────────────────────────────┐ │   │    │    │
│     │   │  │  │  Tensor[1000000, 128] = 512 MB                        │ │   │    │    │
│     │   │  │  │  (1M items × 128 dims × 4 bytes)                      │ │   │    │    │
│     │   │  │  └───────────────────────────────────────────────────────┘ │   │    │    │
│     │   │  └─────────────────────────────────────────────────────────────┘   │    │    │
│     │   └─────────────────────────────────────────────────────────────────────┘    │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  PHASE 2: Initialize Model with Pretrained Weights                                       │
│  ═════════════════════════════════════════════════                                       │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │   Create EmbeddingBagCollection with custom initialization                    │    │
│     │                                                                               │    │
│     │   ┌─────────────────────────────────────────────────────────────────────┐    │    │
│     │   │  # Standard: random initialization                                   │    │    │
│     │   │  ebc = EmbeddingBagCollection(...)                                  │    │    │
│     │   │                                                                     │    │    │
│     │   │  # Transfer learning: copy pretrained weights                       │    │    │
│     │   │  for table in ebc.embedding_bags:                                   │    │    │
│     │   │      table.weight.data.copy_(pretrained_tensor)                     │    │    │
│     │   │      # table.weight now contains meaningful embeddings!             │    │    │
│     │   └─────────────────────────────────────────────────────────────────────┘    │    │
│     │                                                                               │    │
│     │   Memory Layout After Initialization:                                         │    │
│     │   ┌─────────────────────────────────────────────────────────────────────┐    │    │
│     │   │                                                                     │    │    │
│     │   │  GPU 0              GPU 1              GPU 2              GPU 3     │    │    │
│     │   │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐│    │    │
│     │   │  │Pretrained  │    │Pretrained  │    │Pretrained  │    │Pretrained││    │    │
│     │   │  │Items 0-250K│    │Items 250K- │    │Items 500K- │    │Items 750K││    │    │
│     │   │  │            │    │    500K    │    │    750K    │    │   -1M    ││    │    │
│     │   │  │ (Shard 0)  │    │ (Shard 1)  │    │ (Shard 2)  │    │(Shard 3) ││    │    │
│     │   │  └────────────┘    └────────────┘    └────────────┘    └──────────┘│    │    │
│     │   │                                                                     │    │    │
│     │   │  Each shard initialized with corresponding slice of pretrained data │    │    │
│     │   │                                                                     │    │    │
│     │   └─────────────────────────────────────────────────────────────────────┘    │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                              │
│           ▼                                                                              │
│  PHASE 3: Fine-tuning Training Loop                                                      │
│  ══════════════════════════════════                                                      │
│                                                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────┐    │
│     │                                                                               │    │
│     │   Standard TorchRec training with pretrained initialization:                  │    │
│     │                                                                               │    │
│     │   for batch in dataloader:                                                    │    │
│     │       ┌──────────────────────────────────────────────────────────────────┐   │    │
│     │       │  1. Forward Pass                                                  │   │    │
│     │       │     • Embedding lookup (using pretrained weights)                 │   │    │
│     │       │     • MLP forward                                                 │   │    │
│     │       │     • Compute logits                                              │   │    │
│     │       │                                                                   │   │    │
│     │       │  2. Loss Computation                                              │   │    │
│     │       │     • BCEWithLogitsLoss(logits, labels)                           │   │    │
│     │       │                                                                   │   │    │
│     │       │  3. Backward Pass                                                 │   │    │
│     │       │     • Compute gradients for all parameters                        │   │    │
│     │       │     • (Optional) Freeze some layers if needed                     │   │    │
│     │       │                                                                   │   │    │
│     │       │  4. Optimizer Step                                                │   │    │
│     │       │     • Update pretrained embeddings with new gradients             │   │    │
│     │       │     • Fine-tune to task-specific patterns                         │   │    │
│     │       └──────────────────────────────────────────────────────────────────┘   │    │
│     │                                                                               │    │
│     │   Benefits of starting from pretrained:                                       │    │
│     │   ┌─────────────────────────────────────────────────────────────────────┐    │    │
│     │   │  • Faster convergence (embeddings already meaningful)               │    │    │
│     │   │  • Better generalization (learned from more data)                   │    │    │
│     │   │  • Works with limited training data                                 │    │    │
│     │   │  • Cold-start problem mitigation (new items have related vectors)   │    │    │
│     │   └─────────────────────────────────────────────────────────────────────┘    │    │
│     │                                                                               │    │
│     └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Shared Memory Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    WHY SHARED MEMORY? MULTI-PROCESS DATA LOADING                         │
│                                                                                          │
│   Problem: Loading 512MB embeddings per process × 8 processes = 4GB memory waste!       │
│                                                                                          │
│   WITHOUT share_memory_():                                                               │
│   ────────────────────────                                                               │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   Process 0 Memory     Process 1 Memory     Process 2 Memory                    │   │
│   │   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                  │   │
│   │   │ Copy of        │   │ Copy of        │   │ Copy of        │   ...            │   │
│   │   │ Pretrained     │   │ Pretrained     │   │ Pretrained     │                  │   │
│   │   │ (512 MB)       │   │ (512 MB)       │   │ (512 MB)       │                  │   │
│   │   └────────────────┘   └────────────────┘   └────────────────┘                  │   │
│   │                                                                                  │   │
│   │   Total: 8 × 512 MB = 4 GB (wasteful!)                                          │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│                                                                                          │
│   WITH share_memory_() (this example):                                                   │
│   ─────────────────────────────────────                                                  │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │                    Shared Memory Region (OS-level)                               │   │
│   │                    ┌──────────────────────────────────┐                          │   │
│   │                    │   Pretrained Embeddings          │                          │   │
│   │                    │   (512 MB - single copy)         │                          │   │
│   │                    └──────────────────────────────────┘                          │   │
│   │                        ▲       ▲       ▲       ▲                                 │   │
│   │                        │       │       │       │                                 │   │
│   │   ┌────────────────────┼───────┼───────┼───────┼────────────────────┐            │   │
│   │   │                    │       │       │       │                    │            │   │
│   │   │  Process 0 ────────┘       │       │       │                    │            │   │
│   │   │  Process 1 ────────────────┘       │       │                    │            │   │
│   │   │  Process 2 ────────────────────────┘       │                    │            │   │
│   │   │  Process 3 ────────────────────────────────┘                    │            │   │
│   │   │  ...                                                            │            │   │
│   │   └─────────────────────────────────────────────────────────────────┘            │   │
│   │                                                                                  │   │
│   │   Total: 512 MB (efficient!)                                                     │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Code Pattern:                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   # In main process (before spawning workers):                                   │   │
│   │   pretrained = torch.load("embeddings.pt")                                       │   │
│   │   pretrained.share_memory_()  # Move to shared memory                           │   │
│   │                                                                                  │   │
│   │   # Spawn worker processes                                                       │   │
│   │   mp.spawn(train_worker, args=(pretrained, ...), nprocs=world_size)             │   │
│   │                                                                                  │   │
│   │   # In each worker:                                                              │   │
│   │   def train_worker(rank, pretrained, ...):                                       │   │
│   │       # pretrained is accessible without copying!                                │   │
│   │       model.embedding.weight.data.copy_(pretrained)                              │   │
│   │       ...                                                                        │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Fine-tuning Strategies Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         FINE-TUNING STRATEGIES FOR TRANSFER LEARNING                     │
│                                                                                          │
│   Strategy 1: Full Fine-tuning (default)                                                 │
│   ──────────────────────────────────────                                                 │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                   │  │
│   │   ┌─────────────────────┐         ┌─────────────────────┐                        │  │
│   │   │ Pretrained          │         │ Fine-tuned          │                        │  │
│   │   │ Embeddings          │  ───►   │ Embeddings          │                        │  │
│   │   │ (all trainable)     │ Update  │ (task-specific)     │                        │  │
│   │   └─────────────────────┘  ALL    └─────────────────────┘                        │  │
│   │                                                                                   │  │
│   │   • All embedding weights updated during training                                 │  │
│   │   • Best when you have enough task-specific data                                  │  │
│   │   • May overfit with limited data                                                 │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   Strategy 2: Frozen Embeddings + Trainable MLP                                          │
│   ─────────────────────────────────────────────                                          │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                   │  │
│   │   ┌─────────────────────┐                      ┌─────────────────────┐           │  │
│   │   │ Pretrained          │  ───────────────►    │ Same Pretrained     │           │  │
│   │   │ Embeddings          │  FROZEN (no update)  │ Embeddings          │           │  │
│   │   │ requires_grad=False │                      │                     │           │  │
│   │   └─────────────────────┘                      └─────────────────────┘           │  │
│   │            │                                                                      │  │
│   │            ▼                                                                      │  │
│   │   ┌─────────────────────┐                      ┌─────────────────────┐           │  │
│   │   │ Random MLP          │  ───────────────►    │ Trained MLP         │           │  │
│   │   │ requires_grad=True  │  UPDATE              │ (task-specific)     │           │  │
│   │   └─────────────────────┘                      └─────────────────────┘           │  │
│   │                                                                                   │  │
│   │   Code:                                                                           │  │
│   │   for param in model.embedding.parameters():                                      │  │
│   │       param.requires_grad = False                                                 │  │
│   │                                                                                   │  │
│   │   • Preserves pretrained knowledge                                                │  │
│   │   • Faster training (fewer gradients to compute)                                  │  │
│   │   • Best when pretrained embeddings are high quality                              │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   Strategy 3: Gradual Unfreezing                                                         │
│   ──────────────────────────────                                                         │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                   │  │
│   │   Epoch 1-5:    Train MLP only (embeddings frozen)                                │  │
│   │   Epoch 6-10:   Unfreeze top 25% of embeddings, train both                        │  │
│   │   Epoch 11-15:  Unfreeze top 50% of embeddings                                    │  │
│   │   Epoch 16+:    Unfreeze all, fine-tune everything                                │  │
│   │                                                                                   │  │
│   │   Timeline:                                                                       │  │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐    │  │
│   │   │ Epoch:  1──────5──────10──────15──────20                                │    │  │
│   │   │                                                                         │    │  │
│   │   │ MLP:    [████TRAINING████████████████████████████████████]              │    │  │
│   │   │ Emb 75%:[░░░░░░░░░░░░TRAINING████████████████████████████]              │    │  │
│   │   │ Emb 50%:[░░░░░░░░░░░░░░░░░░░░░TRAINING███████████████████]              │    │  │
│   │   │ Emb 25%:[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░TRAINING██████████]              │    │  │
│   │   │                                                                         │    │  │
│   │   │ ░ = Frozen, █ = Training                                                │    │  │
│   │   └─────────────────────────────────────────────────────────────────────────┘    │  │
│   │                                                                                   │  │
│   │   • Prevents catastrophic forgetting of pretrained knowledge                      │  │
│   │   • Best for limited data or sensitive domains                                    │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Common Use Cases

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING USE CASES IN RECOMMENDATIONS                        │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Use Case 1: Cross-Domain Transfer                                                │  │
│   │  ─────────────────────────────────                                                │  │
│   │                                                                                   │  │
│   │  Source: Movie recommendations     Target: Book recommendations                   │  │
│   │  ┌─────────────────────────┐      ┌─────────────────────────┐                    │  │
│   │  │ Movie Embeddings        │ ───► │ Book Model              │                    │  │
│   │  │ (genre, actor, style)   │      │ (leverage genre/style)  │                    │  │
│   │  └─────────────────────────┘      └─────────────────────────┘                    │  │
│   │                                                                                   │  │
│   │  • Users who like action movies may like thriller books                          │  │
│   │  • Genre/theme similarities transfer across domains                              │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Use Case 2: Cold-Start Mitigation                                                │  │
│   │  ─────────────────────────────────                                                │  │
│   │                                                                                   │  │
│   │  Problem: New items have no interaction data                                      │  │
│   │                                                                                   │  │
│   │  ┌─────────────────────────┐      ┌─────────────────────────┐                    │  │
│   │  │ Item Description        │ ───► │ Pretrained Language     │ ───► Item Embedding│  │
│   │  │ (text metadata)         │      │ Model (BERT/Word2Vec)   │                    │  │
│   │  └─────────────────────────┘      └─────────────────────────┘                    │  │
│   │                                                                                   │  │
│   │  New item "Sci-Fi Space Opera" gets embedding similar to other sci-fi items     │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Use Case 3: Multilingual Transfer                                                │  │
│   │  ─────────────────────────────────                                                │  │
│   │                                                                                   │  │
│   │  Source: English product recs     Target: German product recs                    │  │
│   │  ┌─────────────────────────┐      ┌─────────────────────────┐                    │  │
│   │  │ English Model           │ ───► │ German Model            │                    │  │
│   │  │ (100M interactions)     │      │ (1M interactions)       │                    │  │
│   │  └─────────────────────────┘      └─────────────────────────┘                    │  │
│   │                                                                                   │  │
│   │  • Product relationships (iPhone → case) are language-agnostic                   │  │
│   │  • Transfer learned patterns to low-resource languages                           │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Use Case 4: Model Compression                                                    │  │
│   │  ─────────────────────────────                                                    │  │
│   │                                                                                   │  │
│   │  Source: Large Teacher Model      Target: Small Student Model                    │  │
│   │  ┌─────────────────────────┐      ┌─────────────────────────┐                    │  │
│   │  │ 1B param model          │ ───► │ 100M param model        │                    │  │
│   │  │ (512-dim embeddings)    │      │ (64-dim embeddings)     │                    │  │
│   │  └─────────────────────────┘      └─────────────────────────┘                    │  │
│   │                                                                                   │  │
│   │  • Distill knowledge from large model to smaller, faster model                   │  │
│   │  • Initialize student with projected teacher embeddings                          │  │
│   │                                                                                   │  │
│   └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
transfer_learning/
├── README.md                            # This file
├── BUCK                                 # Build configuration
├── __init__.py
└── train_from_pretrained_embedding.py   # Main training script
```

## Running

```bash
# Basic execution
python train_from_pretrained_embedding.py

# With torchx for distributed training
torchx run -s local_cwd dist.ddp -j 1x2 --script train_from_pretrained_embedding.py
```

## Key Code Patterns

### Loading Pretrained Embeddings into Shared Memory

```python
import torch
import torch.multiprocessing as mp

# Load pretrained embeddings (e.g., from parquet)
pretrained_embeddings = torch.load("pretrained_embeddings.pt")

# Move to shared memory for efficient multi-process access
pretrained_embeddings.share_memory_()

# Spawn training processes
mp.spawn(
    train_worker,
    args=(pretrained_embeddings, ...),
    nprocs=world_size,
)
```

### Initializing TorchRec Model with Pretrained Weights

```python
from torchrec import EmbeddingBagCollection

# Create embedding collection
ebc = EmbeddingBagCollection(
    tables=[
        EmbeddingBagConfig(
            name="item_embedding",
            embedding_dim=128,
            num_embeddings=1_000_000,
        ),
    ],
)

# Initialize with pretrained weights
for name, table in ebc.embedding_bags.items():
    table.weight.data.copy_(pretrained_embeddings)
```

## References

- [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html) - PyTorch multiprocessing utilities
- [Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html) - Shared memory and data loading
- [TorchRec Documentation](https://pytorch.org/torchrec/) - Distributed embeddings
- [Transfer Learning Survey](https://arxiv.org/abs/1911.02685) - Comprehensive overview of transfer learning techniques
