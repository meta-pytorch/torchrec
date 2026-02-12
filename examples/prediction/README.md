# DLRM Prediction Example

This example demonstrates how to use a Deep Learning Recommendation Model (DLRM) for making predictions using TorchRec capabilities. The code includes:

1. A DLRM implementation using TorchRec's EmbeddingBagCollection and KeyedJaggedTensor
2. Training with random data
3. Evaluation
4. Making sample predictions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           DLRM PREDICTION PIPELINE                                       │
│                                                                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────┐    │
│   │                              INPUT FEATURES                                     │    │
│   │                                                                                 │    │
│   │   Dense Features (numerical)         Sparse Features (categorical)             │    │
│   │   ┌───────────────────────────┐      ┌───────────────────────────┐             │    │
│   │   │ user_age: 25              │      │ user_id: 12345            │             │    │
│   │   │ time_on_page: 45.2        │      │ item_id: 67890            │             │    │
│   │   │ num_clicks: 3             │      │ category: "electronics"   │             │    │
│   │   │ ...                       │      │ ...                       │             │    │
│   │   └───────────────────────────┘      └───────────────────────────┘             │    │
│   │                                                                                 │    │
│   └────────────────────────────────────────────────────────────────────────────────┘    │
│                    │                                    │                                │
│                    ▼                                    ▼                                │
│   ┌────────────────────────────────┐    ┌────────────────────────────────┐              │
│   │       DENSE PROCESSING         │    │     SPARSE PROCESSING          │              │
│   │                                │    │                                │              │
│   │   torch.Tensor [batch, 13]     │    │   KeyedJaggedTensor            │              │
│   │           │                    │    │   ┌──────────────────────────┐ │              │
│   │           ▼                    │    │   │ keys: ["user_id",       │ │              │
│   │   ┌────────────────────┐       │    │   │       "item_id", ...]   │ │              │
│   │   │   Bottom MLP       │       │    │   │ values: [123, 456, ...] │ │              │
│   │   │   13 → 512 → 256   │       │    │   │ lengths: [1, 1, 1, ...] │ │              │
│   │   │   → 128            │       │    │   └──────────────────────────┘ │              │
│   │   └─────────┬──────────┘       │    │           │                    │              │
│   │             │                  │    │           ▼                    │              │
│   │             │                  │    │   ┌────────────────────┐       │              │
│   │             │                  │    │   │EmbeddingBagCollection│     │              │
│   │             │                  │    │   │ (multiple tables)   │     │              │
│   │             │                  │    │   └─────────┬──────────┘       │              │
│   └─────────────┼──────────────────┘    └─────────────┼──────────────────┘              │
│                 │ [batch, 128]                        │ [batch, num_sparse, 128]        │
│                 │                                     │                                  │
│                 └─────────────────┬───────────────────┘                                  │
│                                   │                                                      │
│                                   ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         FEATURE INTERACTION                                      │   │
│   │                                                                                  │   │
│   │   1. Concatenate all embeddings: [dense_emb, sparse_emb_1, ..., sparse_emb_n]   │   │
│   │   2. Compute pairwise dot products between all embedding pairs                  │   │
│   │   3. Concatenate: [dense_emb, dot_products]                                     │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                   │                                                      │
│                                   ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              TOP MLP                                             │   │
│   │                                                                                  │   │
│   │   ┌───────────────────────────────────────────────────────────────────────┐     │   │
│   │   │ Linear → ReLU → Linear → ReLU → Linear → Sigmoid                      │     │   │
│   │   └───────────────────────────────────────────────────────────────────────┘     │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                   │                                                      │
│                                   ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              OUTPUT                                              │   │
│   │                                                                                  │   │
│   │   Prediction: 0.87 (rating on 0-5 scale)                                        │   │
│   │                                                                                  │   │
│   │   Interpretation: User likely to rate this item ~4.4 stars                      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Training Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         DLRM TRAINING ITERATION                                          │
│                                                                                          │
│   STEP 1: Data Loading                                                                   │
│   ════════════════════                                                                   │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Random Dataset (simulated user-item interactions)                               │   │
│   │                                                                                  │   │
│   │  ┌──────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │  Batch:                                                                   │   │   │
│   │  │  • dense_features: Tensor[256, 13]  (256 samples × 13 dense features)    │   │   │
│   │  │  • sparse_features: KJT with keys ["user", "item", "category", ...]      │   │   │
│   │  │  • labels: Tensor[256]  (ground truth ratings)                           │   │   │
│   │  └──────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   STEP 2: Forward Pass                                                                   │
│   ════════════════════                                                                   │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   predictions = model(dense_features, sparse_features)                           │   │
│   │                                                                                  │   │
│   │   Internally:                                                                    │   │
│   │   1. dense_out = bottom_mlp(dense_features)      # [256, 128]                   │   │
│   │   2. sparse_out = embedding_bags(sparse_kjt)     # [256, num_sparse, 128]       │   │
│   │   3. interactions = dot_products(dense_out, sparse_out)                         │   │
│   │   4. predictions = top_mlp(concat(dense_out, interactions))  # [256, 1]         │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   STEP 3: Loss Computation                                                               │
│   ═══════════════════════                                                                │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   loss = MSELoss(predictions, labels)                                            │   │
│   │                                                                                  │   │
│   │   MSE = (1/N) × Σ (prediction_i - label_i)²                                     │   │
│   │                                                                                  │   │
│   │   For rating prediction: measures how far off our predicted ratings are         │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   STEP 4: Backward Pass & Optimization                                                   │
│   ═══════════════════════════════════                                                    │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                  │   │
│   │   loss.backward()          # Compute gradients                                   │   │
│   │   optimizer.step()         # Update parameters                                   │   │
│   │   optimizer.zero_grad()    # Reset gradients                                     │   │
│   │                                                                                  │   │
│   │   Learning Rate Schedule: StepLR (decay every 5 epochs)                         │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                          TRAINING PROGRESS                                       │   │
│   │                                                                                  │   │
│   │   Epoch 1:  Loss: 2.5432  ████░░░░░░░░░░░░░░░░░░░░░░░░░░                        │   │
│   │   Epoch 2:  Loss: 1.8721  ████████░░░░░░░░░░░░░░░░░░░░░░                        │   │
│   │   Epoch 3:  Loss: 1.2345  ████████████░░░░░░░░░░░░░░░░░░                        │   │
│   │   Epoch 4:  Loss: 0.8234  ████████████████░░░░░░░░░░░░░░                        │   │
│   │   Epoch 5:  Loss: 0.5123  ████████████████████░░░░░░░░░░                        │   │
│   │   ...                                                                            │   │
│   │   Epoch 10: Loss: 0.1234  ████████████████████████████████  ✓                   │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         MAKING PREDICTIONS                                               │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                          SINGLE USER PREDICTION                                  │   │
│   │                                                                                  │   │
│   │   User 42 wants recommendations for items [101, 102, 103, 104, 105]             │   │
│   │                                                                                  │   │
│   │   ┌──────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │  Input:                                                                   │  │   │
│   │   │  • user_id: 42                                                           │  │   │
│   │   │  • item_ids: [101, 102, 103, 104, 105]                                   │  │   │
│   │   │  • user_features: age=25, tenure=365, ...                                │  │   │
│   │   └──────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                    │                                             │   │
│   │                                    ▼                                             │   │
│   │   ┌──────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │  Model Inference (no gradients):                                         │  │   │
│   │   │                                                                          │  │   │
│   │   │  with torch.no_grad():                                                   │  │   │
│   │   │      predictions = model(dense_features, sparse_kjt)                     │  │   │
│   │   │                                                                          │  │   │
│   │   └──────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                    │                                             │   │
│   │                                    ▼                                             │   │
│   │   ┌──────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │  Output Predictions:                                                     │  │   │
│   │   │                                                                          │  │   │
│   │   │  Item 101: 4.23 ★★★★☆  (Predicted rating)                               │  │   │
│   │   │  Item 102: 3.87 ★★★★☆                                                   │  │   │
│   │   │  Item 103: 2.15 ★★☆☆☆                                                   │  │   │
│   │   │  Item 104: 4.89 ★★★★★  ← Highest! Recommend this                        │  │   │
│   │   │  Item 105: 3.42 ★★★☆☆                                                   │  │   │
│   │   │                                                                          │  │   │
│   │   └──────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## TorchRec Integration

This implementation has been updated to use TorchRec's capabilities:
- Uses `KeyedJaggedTensor` for sparse features
- Uses `EmbeddingBagCollection` for embedding tables
- Follows the DLRM architecture as described in the paper: https://arxiv.org/abs/1906.00091

The example demonstrates how to leverage TorchRec's efficient sparse feature handling for recommendation models.

## Dependencies

Install the required dependencies:

```bash
# Install PyTorch
pip install torch torchvision

# Install NumPy
pip install numpy

# Install TorchRec
pip install torchrec
```

**Important**: This implementation now requires torchrec to run, as it uses TorchRec's specialized modules for recommendation systems.

## Running the Example Locally

1. Download the `predict_using_torchrec.py` file to your local machine.

2. Run the example:

```bash
python3 predict_using_torchrec.py
```

3. If you're using a different Python environment:

```bash
# For conda environments
conda activate your_environment_name
python predict_using_torchrec.py

# For virtual environments
source your_venv/bin/activate
python predict_using_torchrec.py
```

## What to Expect

When you run the example, you'll see:

1. Training progress for 10 epochs with loss and learning rate information
2. Evaluation results showing MSE and RMSE metrics
3. Sample predictions for a specific user on multiple items

## Implementation Details

This example uses TorchRec's capabilities to implement a DLRM model that:

- Takes dense features and sparse features (as KeyedJaggedTensor) as input
- Processes dense features through a bottom MLP
- Processes sparse features through EmbeddingBagCollection
- Computes feature interactions using dot products
- Processes the interactions through a top MLP
- Outputs rating predictions on a 0-5 scale

The implementation demonstrates how to use TorchRec's specialized modules for recommendation systems, making it more efficient and scalable than a custom implementation.

## Key TorchRec Components Used

1. **KeyedJaggedTensor**: Efficiently represents sparse features with variable lengths
2. **EmbeddingBagConfig**: Configures embedding tables with parameters like dimensions and feature names
3. **EmbeddingBagCollection**: Manages multiple embedding tables for different categorical features

## Troubleshooting

If you encounter any issues:

1. **Python version**: This code has been tested with Python 3.8+. Make sure you're using a compatible version.

2. **PyTorch and TorchRec installation**: If you have issues with PyTorch or TorchRec, try installing specific versions:
   ```bash
   pip install torch==2.0.0 torchvision==0.15.0
   pip install torchrec==0.5.0
   ```

3. **Memory issues**: If you run out of memory, try reducing the batch size by modifying this line in the code:
   ```python
   batch_size = 256  # Try a smaller value like 64 or 32
   ```

4. **CPU vs GPU**: The code automatically uses CUDA if available. To force CPU usage, modify:
   ```python
   device = torch.device("cpu")
   ```

5. **TorchRec compatibility**: If you encounter compatibility issues with TorchRec, make sure you're using compatible versions of PyTorch and TorchRec.
