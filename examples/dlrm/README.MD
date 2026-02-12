# TorchRec DLRM Example

This directory provides a reference to the official **Deep Learning Recommendation Model (DLRM)** implementation using TorchRec.

## Official Implementation

For the complete DLRM implementation with TorchRec, see:
**[Facebookresearch/dlrm torchrec implementation](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/)**

## DLRM Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           DLRM (Deep Learning Recommendation Model)                      │
│                                                                                          │
│   Input Features:                                                                        │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐             │
│   │     Dense Features (13)         │   │    Sparse Features (26)          │             │
│   │     (numerical: age, time, etc) │   │    (categorical: user, item, etc)│             │
│   └───────────────┬─────────────────┘   └───────────────┬─────────────────┘             │
│                   │                                     │                                │
│                   ▼                                     ▼                                │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐             │
│   │        BOTTOM MLP               │   │    EMBEDDING LOOKUPS             │             │
│   │   (Dense Arch)                  │   │    (Sparse Arch)                 │             │
│   │                                 │   │                                  │             │
│   │   ┌───────────────────────┐    │   │   ┌──────┐ ┌──────┐    ┌──────┐ │             │
│   │   │ Linear(13 → 512)      │    │   │   │ E1   │ │ E2   │... │ E26  │ │             │
│   │   │ ReLU                  │    │   │   │128dim│ │128dim│    │128dim│ │             │
│   │   │ Linear(512 → 256)     │    │   │   └──┬───┘ └──┬───┘    └──┬───┘ │             │
│   │   │ ReLU                  │    │   │      │        │           │      │             │
│   │   │ Linear(256 → 128)     │    │   │      └────────┴─────┬─────┘      │             │
│   │   └───────────┬───────────┘    │   │                     │            │             │
│   │               │                 │   │                     │            │             │
│   └───────────────┼─────────────────┘   └─────────────────────┼────────────┘             │
│                   │ [batch, 128]                              │ [batch, 26, 128]         │
│                   │                                           │                          │
│                   └─────────────────────┬─────────────────────┘                          │
│                                         │                                                │
│                                         ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         FEATURE INTERACTION                                      │   │
│   │                                                                                  │   │
│   │   Concatenate: [dense_output, emb_1, emb_2, ..., emb_26] → 27 vectors           │   │
│   │                                                                                  │   │
│   │   Pairwise Dot Products:                                                         │   │
│   │   ┌────────────────────────────────────────────────────────────────────────┐    │   │
│   │   │  interactions[i,j] = dot(vector_i, vector_j)  for all i < j            │    │   │
│   │   │                                                                         │    │   │
│   │   │  Number of interactions = C(27,2) = 27×26/2 = 351                       │    │   │
│   │   └────────────────────────────────────────────────────────────────────────┘    │   │
│   │                                                                                  │   │
│   │   Concatenate: [dense_output, interactions] → [batch, 128 + 351]                │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                                │
│                                         ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              TOP MLP (Over Arch)                                 │   │
│   │                                                                                  │   │
│   │   ┌───────────────────────────────────────────────────────────────────────┐     │   │
│   │   │ Linear(479 → 1024)                                                    │     │   │
│   │   │ ReLU                                                                  │     │   │
│   │   │ Linear(1024 → 512)                                                    │     │   │
│   │   │ ReLU                                                                  │     │   │
│   │   │ Linear(512 → 256)                                                     │     │   │
│   │   │ ReLU                                                                  │     │   │
│   │   │ Linear(256 → 1)                                                       │     │   │
│   │   └───────────────────────────────────────────────────────────────────────┘     │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                                │
│                                         ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              OUTPUT                                              │   │
│   │                                                                                  │   │
│   │   logits [batch, 1] → sigmoid → click probability (0 to 1)                      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DLRM TRAINING DATA FLOW                                     │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                            CRITEO DATASET FORMAT                                 │   │
│   │                                                                                  │   │
│   │   Each row represents a user-ad interaction:                                     │   │
│   │   ┌──────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │  Label │ Dense_1 │ Dense_2 │...│ Dense_13 │ Cat_1 │ Cat_2 │...│ Cat_26  │  │   │
│   │   │────────│─────────│─────────│───│──────────│───────│───────│───│─────────│  │   │
│   │   │   1    │  0.23   │  0.45   │...│   0.12   │  abc  │  def  │...│   xyz   │  │   │
│   │   │   0    │  0.67   │  0.89   │...│   0.34   │  ghi  │  jkl  │...│   uvw   │  │   │
│   │   └──────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                  │   │
│   │   • Label: 1 = clicked, 0 = not clicked                                          │   │
│   │   • Dense: 13 numerical features (log-transformed counts, etc.)                  │   │
│   │   • Categorical: 26 hashed categorical features (user ID, ad ID, etc.)           │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         TORCHREC DATA TRANSFORMATION                             │   │
│   │                                                                                  │   │
│   │   Dense Features:                                                                │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  torch.Tensor [batch_size, 13]                                          │   │   │
│   │   │  (standard dense tensor)                                                │   │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                  │   │
│   │   Sparse Features → KeyedJaggedTensor:                                           │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  keys:    ["cat_0", "cat_1", ..., "cat_25"]                             │   │   │
│   │   │  values:  [id1, id2, id3, ...]  (all IDs flattened)                     │   │   │
│   │   │  lengths: [1, 1, 1, ...]        (1 feature per sample per category)     │   │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                  │   │
│   │   Labels:                                                                        │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  torch.Tensor [batch_size]                                              │   │   │
│   │   │  (0.0 or 1.0)                                                           │   │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Distributed Training with TorchRec

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      WHY TORCHREC FOR DLRM?                                              │
│                                                                                          │
│   Challenge: DLRM embedding tables are MASSIVE                                           │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Criteo 1TB Dataset:                                                             │   │
│   │  • 26 categorical features                                                       │   │
│   │  • Up to 40M unique values per feature                                           │   │
│   │  • 128-dimensional embeddings                                                    │   │
│   │  • Total: 26 × 40M × 128 × 4 bytes ≈ 530 GB of embeddings!                      │   │
│   │                                                                                  │   │
│   │  This won't fit on a single GPU (or even a single machine)!                      │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Solution: TorchRec Model Parallelism                                                   │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   8-GPU Node (A100 80GB each):                                                   │   │
│   │                                                                                  │   │
│   │   GPU 0        GPU 1        GPU 2        GPU 3                                   │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                            │   │
│   │   │ cat_0-2 │  │ cat_3-5 │  │ cat_6-9 │  │cat_10-13│                            │   │
│   │   │ ~70GB   │  │ ~70GB   │  │ ~70GB   │  │ ~70GB   │                            │   │
│   │   └─────────┘  └─────────┘  └─────────┘  └─────────┘                            │   │
│   │                                                                                  │   │
│   │   GPU 4        GPU 5        GPU 6        GPU 7                                   │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                            │   │
│   │   │cat_14-17│  │cat_18-20│  │cat_21-23│  │cat_24-25│                            │   │
│   │   │ ~70GB   │  │ ~50GB   │  │ ~50GB   │  │ ~30GB   │                            │   │
│   │   │         │  │ + MLP   │  │ + MLP   │  │ + MLP   │                            │   │
│   │   └─────────┘  └─────────┘  └─────────┘  └─────────┘                            │   │
│   │                                                                                  │   │
│   │   TorchRec automatically:                                                        │   │
│   │   • Shards embedding tables across GPUs                                          │   │
│   │   • Handles All-to-All communication                                             │   │
│   │   • Overlaps communication with computation                                      │   │
│   │   • Optimizes memory layout                                                      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Related Examples in This Repository

| Example | Description |
|---------|-------------|
| [`golden_training/`](../golden_training/) | Complete DLRM training with TorchRec |
| [`prediction/`](../prediction/) | DLRM inference and prediction |
| [`nvt_dataloader/`](../nvt_dataloader/) | DLRM with NVTabular data preprocessing |

## References

- [DLRM Paper](https://arxiv.org/abs/1906.00091) - Original DLRM architecture
- [TorchRec Documentation](https://pytorch.org/torchrec/)
- [Criteo Dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)
