#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

# TorchRec imports
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# Mock dataset for demonstration.
class RecommendationDataset(Dataset):
    """
    A PyTorch Dataset class for generating random user-item interaction data
    for recommendation systems.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.
        user_ids (torch.Tensor): Tensor of user IDs for each sample.
        item_ids (torch.Tensor): Tensor of item IDs for each sample.
        ratings (torch.Tensor): Tensor of ratings for each sample.
        user_categories (torch.Tensor): Tensor of user categories for each sample.
        item_categories (torch.Tensor): Tensor of item categories for each sample.
        dense_features (torch.Tensor): Tensor of dense features for each sample.
    """

    def __init__(
        self, num_users: int = 1000, num_items: int = 500, num_samples: int = 10000
    ) -> None:
        """
        Initializes the RecommendationDataset with random data.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items.
            num_samples (int): Number of samples to generate.
        """
        self.num_samples: int = num_samples
        self.num_users: int = num_users
        self.num_items: int = num_items

        # Generate random user-item interactions
        self.user_ids: torch.Tensor = torch.randint(0, num_users, (num_samples,))
        self.item_ids: torch.Tensor = torch.randint(0, num_items, (num_samples,))

        # Generate random ratings (0-5)
        self.ratings: torch.Tensor = torch.randint(0, 6, (num_samples,)).float()

        # Generate some categorical features
        self.user_categories: torch.Tensor = torch.randint(
            0, 10, (num_samples,)
        )  # 10 user categories
        self.item_categories: torch.Tensor = torch.randint(
            0, 20, (num_samples,)
        )  # 20 item categories

        # Generate dense features (normalized to [0, 1])
        self.dense_features: torch.Tensor = torch.rand(
            num_samples, 4
        )  # 4 dense features

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    # pyrefly: ignore[bad-param-name-override]
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing user_id, item_id, user_category,
                  item_category, dense_features, and rating for the sample.
        """
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "user_category": self.user_categories[idx],
            "item_category": self.item_categories[idx],
            "dense_features": self.dense_features[idx],
            "rating": self.ratings[idx],
        }


# TorchRec DLRM model using KeyedJaggedTensor and EmbeddingBagCollection
class TorchRecDLRM(nn.Module):
    """
    A DLRM model implementation using TorchRec's EmbeddingBagCollection and KeyedJaggedTensor.
    This model follows the architecture described in the DLRM paper:
    https://arxiv.org/abs/1906.00091

    Args:
        embedding_bag_collection: EmbeddingBagCollection for sparse features
        dense_in_features: Number of dense features
        dense_arch_layer_sizes: Layer sizes for the dense (bottom) MLP
        over_arch_layer_sizes: Layer sizes for the over (top) MLP

    Example:
        ```
        # Create embedding bag configs
        eb_configs = [
            EmbeddingBagConfig(
                name="user_id",
                embedding_dim=64,
                num_embeddings=1000,
                feature_names=["user_id"],
            ),
            EmbeddingBagConfig(
                name="item_id",
                embedding_dim=64,
                num_embeddings=500,
                feature_names=["item_id"],
            ),
        ]

        # Create EmbeddingBagCollection
        ebc = EmbeddingBagCollection(
            tables=eb_configs,
            device=torch.device("cpu"),
        )

        # Create TorchRecDLRM model
        model = TorchRecDLRM(
            embedding_bag_collection=ebc,
            dense_in_features=4,
            dense_arch_layer_sizes=[32, 64],
            over_arch_layer_sizes=[128, 64, 1],
        )

        # Forward pass
        batch_size = 2
        dense_features = torch.rand(batch_size, 4)

        # Create KeyedJaggedTensor for sparse features
        values = torch.tensor([0, 1, 2, 3])
        lengths = torch.ones(4, dtype=torch.int32)
        sparse_features = KeyedJaggedTensor(
            keys=["user_id", "item_id"],
            values=values,
            lengths=lengths,
        )

        # Get predictions
        logits = model(dense_features, sparse_features)
        ```
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int = 4,
        dense_arch_layer_sizes: Optional[List[int]] = None,
        over_arch_layer_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if dense_arch_layer_sizes is None:
            dense_arch_layer_sizes = [32, 64]

        if over_arch_layer_sizes is None:
            over_arch_layer_sizes = [128, 64, 1]

        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection

        # Get embedding dimension from the first embedding table
        embedding_dim = self.embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim

        # Dense arch (bottom MLP)
        layers: List[nn.Module] = []
        input_dim = dense_in_features
        for output_dim in dense_arch_layer_sizes:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.dense_arch: nn.Sequential = nn.Sequential(*layers)

        # Feature interaction: dot product of all pairs
        num_sparse_features = len(self.embedding_bag_collection.embedding_bag_configs())
        num_interactions = num_sparse_features + 1  # +1 for dense features
        num_pairs = (num_interactions * (num_interactions - 1)) // 2

        # Over arch (top MLP)
        over_input_dim = embedding_dim + num_pairs
        over_layers: List[nn.Module] = []
        input_dim = over_input_dim
        for i, output_dim in enumerate(over_arch_layer_sizes):
            over_layers.append(nn.Linear(input_dim, output_dim))
            if i < len(over_arch_layer_sizes) - 1:
                over_layers.append(nn.ReLU())
            input_dim = output_dim
        self.over_arch: nn.Sequential = nn.Sequential(*over_layers)

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedJaggedTensor
    ) -> torch.Tensor:
        """
        Forward pass of the DLRM model.

        Args:
            dense_features: Dense input features
            sparse_features: Sparse input features as KeyedJaggedTensor

        Returns:
            torch.Tensor: Model output logits
        """
        # Process dense features
        dense_output = self.dense_arch(dense_features)

        # Process sparse features
        sparse_output = self.embedding_bag_collection(sparse_features)

        # Get embeddings as a list
        embeddings = [sparse_output[f] for f in sparse_output.keys()]

        # Feature interaction
        all_features = [dense_output] + embeddings
        interactions = []

        # Add original dense output
        interactions.append(dense_output)

        # Compute pairwise dot products
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                dot_product = torch.sum(
                    all_features[i] * all_features[j], dim=1, keepdim=True
                )
                interactions.append(dot_product)

        # Concatenate all interactions
        interaction_output = torch.cat(interactions, dim=1)

        # Over arch
        logits = self.over_arch(interaction_output)

        return logits


# DLRM wrapper for rating prediction
class DLRMRatingWrapper(nn.Module):
    """
    Wrapper for DLRM model to scale the output to [0, 5] for rating prediction.

    Args:
        dlrm_model: The DLRM model to wrap

    Example:
        ```
        # Create embedding bag configs
        eb_configs = [
            EmbeddingBagConfig(
                name="user_id",
                embedding_dim=64,
                num_embeddings=1000,
                feature_names=["user_id"],
            ),
            EmbeddingBagConfig(
                name="item_id",
                embedding_dim=64,
                num_embeddings=500,
                feature_names=["item_id"],
            ),
        ]

        # Create EmbeddingBagCollection
        ebc = EmbeddingBagCollection(
            tables=eb_configs,
            device=torch.device("cpu"),
        )

        # Create base model
        base_model = TorchRecDLRM(
            embedding_bag_collection=ebc,
            dense_in_features=4,
        )

        # Create wrapper
        model_wrapper = DLRMRatingWrapper(base_model)

        # Forward pass
        batch_size = 2
        dense_features = torch.rand(batch_size, 4)

        # Create KeyedJaggedTensor for sparse features
        values = torch.tensor([0, 1, 2, 3])
        lengths = torch.ones(4, dtype=torch.int32)
        sparse_features = KeyedJaggedTensor(
            keys=["user_id", "item_id"],
            values=values,
            lengths=lengths,
        )

        # Get predictions (scaled to 0-5 range)
        predictions = model_wrapper(dense_features, sparse_features)
        ```
    """

    def __init__(self, dlrm_model: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = dlrm_model
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedJaggedTensor
    ) -> torch.Tensor:
        """
        Forward pass of the DLRM wrapper.

        Args:
            dense_features: Dense input features
            sparse_features: Sparse input features as KeyedJaggedTensor

        Returns:
            torch.Tensor: Rating prediction scaled to [0, 5]
        """
        logits = self.model(dense_features, sparse_features)
        # Scale output to [0, 5] for rating prediction
        return self.sigmoid(logits.squeeze()) * 5.0


def create_kjt_from_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> KeyedJaggedTensor:
    """
    Create a KeyedJaggedTensor from a batch of data.

    Args:
        batch: Batch of data containing categorical features
        device: Device to place the KeyedJaggedTensor on

    Returns:
        KeyedJaggedTensor: Sparse features in KeyedJaggedTensor format
    """
    # For this example, each categorical feature has exactly one value per sample
    # So lengths are all 1s
    batch_size = batch["user_id"].size(0)
    lengths = torch.ones(batch_size * 4, dtype=torch.int32, device=device)

    # Concatenate all values
    values = torch.cat(
        [
            batch["user_id"],
            batch["item_id"],
            batch["user_category"],
            batch["item_category"],
        ]
    ).to(device)

    # Create KeyedJaggedTensor
    return KeyedJaggedTensor(
        keys=["user_id", "item_id", "user_category", "item_category"],
        values=values,
        lengths=lengths,
    )


def train_dlrm_model() -> Tuple[DLRMRatingWrapper, str]:
    """
    Trains the Deep Learning Recommendation Model (DLRM) using a specified dataset and hyperparameters.

    Returns:
        tuple: A tuple containing the trained DLRM model and the model filename.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DLRM on device: {device}")

    # Hyperparameters
    num_users = 1000
    num_items = 500
    num_user_categories = 10
    num_item_categories = 20
    embedding_dim = 64
    batch_size = 256
    learning_rate = 0.001
    num_epochs = 10

    # Create dataset and dataloader
    dataset = RecommendationDataset(num_users, num_items, 10000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create embedding bag configs
    eb_configs = [
        EmbeddingBagConfig(
            name="user_id",
            embedding_dim=embedding_dim,
            num_embeddings=num_users,
            feature_names=["user_id"],
        ),
        EmbeddingBagConfig(
            name="item_id",
            embedding_dim=embedding_dim,
            num_embeddings=num_items,
            feature_names=["item_id"],
        ),
        EmbeddingBagConfig(
            name="user_category",
            embedding_dim=embedding_dim,
            num_embeddings=num_user_categories,
            feature_names=["user_category"],
        ),
        EmbeddingBagConfig(
            name="item_category",
            embedding_dim=embedding_dim,
            num_embeddings=num_item_categories,
            feature_names=["item_category"],
        ),
    ]

    # Create EmbeddingBagCollection
    ebc = EmbeddingBagCollection(
        tables=eb_configs,
        device=device,
    )

    # Create TorchRecDLRM model
    model = TorchRecDLRM(
        embedding_bag_collection=ebc,
        dense_in_features=4,
        dense_arch_layer_sizes=[32, embedding_dim],
        over_arch_layer_sizes=[128, 64, 1],
    ).to(device)

    # Create a wrapper to scale the output to [0, 5] for rating prediction
    model_wrapper = DLRMRatingWrapper(model).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model_wrapper.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(
        f"DLRM Model parameters: {sum(p.numel() for p in model_wrapper.parameters()):,}"
    )

    # Training loop
    model_wrapper.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            dense_features = batch["dense_features"].to(device)
            ratings = batch["rating"].to(device)

            # Create KeyedJaggedTensor for sparse features
            sparse_features = create_kjt_from_batch(batch, device)

            # Forward pass
            predictions = model_wrapper(dense_features, sparse_features)
            loss = criterion(predictions, ratings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Update learning rate
        scheduler.step()

        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )

    print("DLRM Training completed!")

    # Save the model
    model_filename = "dlrm_model.pth"
    torch.save(
        {
            "model_state_dict": model_wrapper.state_dict(),
            "embedding_dim": embedding_dim,
            "num_users": num_users,
            "num_items": num_items,
            "num_user_categories": num_user_categories,
            "num_item_categories": num_item_categories,
            "dense_in_features": 4,
            "dense_arch_layer_sizes": [32, embedding_dim],
            "over_arch_layer_sizes": [128, 64, 1],
        },
        model_filename,
    )
    print(f"DLRM Model saved as {model_filename}")
    return model_wrapper, model_filename


def evaluate_dlrm_model(
    model: nn.Module,
    dataloader: DataLoader[RecommendationDataset],
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the DLRM model on a dataset.

    Args:
        model (nn.Module): The DLRM model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        device (torch.device): The device to perform evaluation on (CPU or GPU).

    Returns:
        tuple: A tuple containing the average loss and root mean square error (RMSE).
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            dense_features = batch["dense_features"].to(device)
            ratings = batch["rating"].to(device)

            # Create KeyedJaggedTensor for sparse features
            sparse_features = create_kjt_from_batch(batch, device)

            # Forward pass
            predictions = model(dense_features, sparse_features)
            loss = criterion(predictions, ratings)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    rmse = np.sqrt(avg_loss)
    print(f"DLRM Evaluation - MSE: {avg_loss:.4f}, RMSE: {rmse:.4f}")
    return avg_loss, rmse


def make_dlrm_predictions(
    model: DLRMRatingWrapper,
    user_id: int,
    item_ids: List[int],
    user_category: int,
    item_categories: List[int],
    device: torch.device,
) -> np.ndarray:
    """Make predictions using DLRM for a user on multiple items"""
    model.eval()
    with torch.no_grad():
        batch_size = len(item_ids)

        # Prepare inputs
        user_ids_tensor = torch.tensor([user_id] * batch_size).to(device)
        item_ids_tensor = torch.tensor(item_ids).to(device)
        user_cats = torch.tensor([user_category] * batch_size).to(device)
        item_cats = torch.tensor(item_categories).to(device)

        # Generate random dense features for demonstration
        dense_features = torch.rand(batch_size, 4).to(device)

        # Create KeyedJaggedTensor for sparse features
        # For this example, each categorical feature has exactly one value per sample
        lengths = torch.ones(batch_size * 4, dtype=torch.int32, device=device)

        # Concatenate all values
        values = torch.cat(
            [
                user_ids_tensor,
                item_ids_tensor,
                user_cats,
                item_cats,
            ]
        )

        # Create KeyedJaggedTensor
        sparse_features = KeyedJaggedTensor(
            keys=["user_id", "item_id", "user_category", "item_category"],
            values=values,
            lengths=lengths,
        )

        # Make predictions
        predictions = model(dense_features, sparse_features)

        # Convert to numpy array and ensure it's a 1D array
        numpy_predictions = predictions.cpu().numpy()
        # Flatten in case it's not already 1D
        return numpy_predictions.flatten()


def remove_model_file(model_filename: str) -> None:
    """
    Removes the model file if it exists.

    Args:
        model_filename (str): The filename of the model to be removed.
    """
    if os.path.exists(model_filename):
        try:
            os.remove(model_filename)
            print(f"Successfully removed the file: {model_filename}")
        except PermissionError:
            print(f"Permission denied: {model_filename}")
        except Exception as e:
            print(f"An error occurred while trying to remove the file: {e}")
    else:
        print(f"File does not exist: {model_filename}")


def main() -> None:
    """
    Main function to orchestrate the training, evaluation, and prediction
    processes of the Deep Learning Recommendation Model (DLRM).

    This function performs the following steps:
    1. Trains the DLRM model using a specified dataset and hyperparameters.
    2. Evaluates the trained model on a separate evaluation dataset.
    3. Makes sample predictions for a specific user on multiple items.
    4. Cleans up the model after training to free up resources.
    """
    print("Starting DLRM (Deep Learning Recommendation Model) Training...")

    try:
        # Train the DLRM model
        print("Starting training...")
        trained_model, model_filename = train_dlrm_model()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create evaluation dataset
    eval_dataset = RecommendationDataset(1000, 500, 2000)
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    # Evaluate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nEvaluating DLRM model...")
    evaluate_dlrm_model(trained_model, eval_dataloader, device)

    # Example prediction
    print("\nMaking sample DLRM predictions...")
    sample_user_id = 42
    sample_items = [10, 25, 50, 100, 200]
    sample_user_cat = 3
    sample_item_cats = [5, 12, 8, 15, 2]

    predictions = make_dlrm_predictions(
        trained_model,
        sample_user_id,
        sample_items,
        sample_user_cat,
        sample_item_cats,
        device,
    )

    print(f"DLRM Predictions for user {sample_user_id}:")
    for item_id, pred in zip(sample_items, predictions):
        print(f"  Item {item_id}: {pred:.2f}")

    # clean the model after training
    print(f"Cleaning the model {model_filename}")
    # comment this line if you want to keep the model file
    remove_model_file(model_filename)


if __name__ == "__main__":
    main()
