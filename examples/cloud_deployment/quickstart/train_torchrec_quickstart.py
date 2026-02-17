#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TorchRec Cloud Quickstart - Self-contained DLRM Training Script

This script provides a ready-to-run example of distributed TorchRec training
using synthetic data. No dataset download required!

Usage:
    # Single GPU (local testing)
    python train_torchrec_quickstart.py --batch_size 1024

    # Multi-GPU (local)
    torchrun --nproc_per_node=4 train_torchrec_quickstart.py --batch_size 4096

    # Multi-node (cloud deployment via Kubernetes)
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:29500 train_torchrec_quickstart.py

For more options, run:
    python train_torchrec_quickstart.py --help
"""

import argparse
import os
import time
from typing import cast, Iterator, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

# TorchRec imports
try:
    from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
    from torchrec.distributed.planner.shard_estimators import (
        EmbeddingPerfEstimator,
        EmbeddingStorageEstimator,
    )
    from torchrec.distributed.types import ModuleSharder
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.optim.keyed import KeyedOptimizerWrapper
except ImportError:
    print("=" * 60)
    print("TorchRec not installed! Please run:")
    print("  pip install torchrec fbgemm-gpu")
    print("=" * 60)
    raise


# =============================================================================
# Model Definition - DLRM (Deep Learning Recommendation Model)
# =============================================================================


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron.

    Args:
        in_size (int): Input feature dimension.
        layer_sizes (List[int]): Output sizes for each linear layer.
        bias (bool): Whether to include bias in linear layers.
        activation (str): Activation function ("relu" or "sigmoid").

    Example::

        >>> mlp = MLP(in_size=10, layer_sizes=[64, 32])
        >>> x = torch.randn(4, 10)
        >>> output = mlp(x)
        >>> assert output.shape == (4, 32)
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        bias: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for out_size in layer_sizes:
            layers.append(nn.Linear(in_size, out_size, bias=bias))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, in_size)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, layer_sizes[-1])``.
        """
        return self.mlp(x)


class DLRM(nn.Module):
    """Deep Learning Recommendation Model (DLRM).

    Architecture:
        Dense features → Bottom MLP → ┐
                                      ├→ Feature Interaction → Top MLP → Output
        Sparse features → Embeddings → ┘

    Args:
        embedding_bag_collection (EmbeddingBagCollection): TorchRec embedding bag
            collection for sparse feature lookups.
        dense_in_features (int): Number of dense (continuous) input features.
        dense_arch_layer_sizes (List[int]): Layer sizes for the bottom MLP
            that processes dense features.
        over_arch_layer_sizes (List[int]): Layer sizes for the top MLP
            that processes feature interactions.

    Example::

        >>> from torchrec import EmbeddingBagCollection
        >>> from torchrec.modules.embedding_configs import EmbeddingBagConfig
        >>> ebc = EmbeddingBagCollection(
        ...     tables=[
        ...         EmbeddingBagConfig(
        ...             name="t1", embedding_dim=64,
        ...             num_embeddings=100, feature_names=["f1"],
        ...         ),
        ...     ],
        ...     device=torch.device("meta"),
        ... )
        >>> model = DLRM(
        ...     embedding_bag_collection=ebc,
        ...     dense_in_features=13,
        ...     dense_arch_layer_sizes=[64],
        ...     over_arch_layer_sizes=[128, 64, 1],
        ... )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ) -> None:
        super().__init__()

        if len(over_arch_layer_sizes) < 2:
            raise ValueError(
                f"over_arch_layer_sizes must have at least 2 elements, "
                f"got {len(over_arch_layer_sizes)}"
            )

        if not embedding_bag_collection.embedding_bag_configs():
            raise ValueError(
                "embedding_bag_collection must contain at least one config"
            )

        if not dense_arch_layer_sizes:
            raise ValueError("dense_arch_layer_sizes must not be empty")

        embedding_dim = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim

        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"dense_arch_layer_sizes[-1] ({dense_arch_layer_sizes[-1]}) must equal "
                f"embedding_dim ({embedding_dim}) for feature interaction to work"
            )

        self.embedding_bag_collection = embedding_bag_collection
        self.dense_arch = MLP(
            dense_in_features,
            dense_arch_layer_sizes,
        )

        # Calculate interaction output size
        num_sparse_features = len(embedding_bag_collection.embedding_bag_configs())
        num_features = num_sparse_features + 1  # +1 for dense features

        # Feature interaction: dot product of all pairs
        # Output size = (num_features * (num_features - 1)) / 2 + embedding_dim
        interaction_size = (num_features * (num_features - 1)) // 2 + embedding_dim

        self.over_arch = MLP(
            interaction_size,
            over_arch_layer_sizes[:-1],
        )
        self.final_linear = nn.Linear(
            over_arch_layer_sizes[-2], over_arch_layer_sizes[-1]
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """Forward pass through the DLRM model.

        Args:
            dense_features (torch.Tensor): Dense features of shape
                ``(batch_size, dense_in_features)``.
            sparse_features (KeyedJaggedTensor): Sparse features as a
                KeyedJaggedTensor.

        Returns:
            torch.Tensor: Prediction probabilities of shape ``(batch_size,)``.
        """
        # Process dense features through bottom MLP
        dense_output = self.dense_arch(dense_features)

        # Get sparse embeddings
        embedding_output = self.embedding_bag_collection(sparse_features)

        # Stack all embeddings
        sparse_tensors = []
        for key in embedding_output.keys():
            sparse_tensors.append(embedding_output[key])
        sparse_output = torch.stack(sparse_tensors, dim=1)

        # Add dense output as another "embedding"
        dense_output = dense_output.unsqueeze(1)

        # Concatenate dense and sparse
        combined = torch.cat([dense_output, sparse_output], dim=1)

        # Feature interaction (dot product of all pairs)
        interaction = torch.bmm(combined, combined.transpose(1, 2))

        # Get upper triangular part (excluding diagonal)
        triu_indices = torch.triu_indices(
            combined.shape[1], combined.shape[1], offset=1, device=combined.device
        )
        flat_interaction = interaction[:, triu_indices[0], triu_indices[1]]

        # Concatenate with dense output
        concat = torch.cat([dense_output.squeeze(1), flat_interaction], dim=1)

        # Top MLP
        output = self.over_arch(concat)
        output = self.final_linear(output)

        return torch.sigmoid(output.squeeze(-1))


# =============================================================================
# Synthetic Data Generator
# =============================================================================


class SyntheticDataset(IterableDataset):
    """
    Generates synthetic recommendation data for training.
    No data download required!
    """

    def __init__(
        self,
        num_embeddings: int,
        num_dense_features: int,
        num_sparse_features: int,
        batch_size: int,
        num_batches: int,
        sparse_feature_names: List[str],
    ) -> None:
        self.num_embeddings = num_embeddings
        self.num_dense_features = num_dense_features
        self.num_sparse_features = num_sparse_features
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.sparse_feature_names = sparse_feature_names

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, KeyedJaggedTensor, torch.Tensor]]:
        for _ in range(self.num_batches):
            # Generate random dense features
            dense_features = torch.rand(self.batch_size, self.num_dense_features)

            # Generate random sparse features (KeyedJaggedTensor)
            values_list = []
            lengths_list = []

            for _ in range(self.num_sparse_features):
                # Each sample has 1-3 values per sparse feature
                lengths = torch.randint(1, 4, (self.batch_size,))
                values = torch.randint(
                    0, self.num_embeddings, (int(lengths.sum().item()),)
                )
                values_list.append(values)
                lengths_list.append(lengths)

            # Concatenate all values and lengths
            all_values = torch.cat(values_list)
            all_lengths = torch.cat(lengths_list)

            sparse_features = KeyedJaggedTensor(
                keys=self.sparse_feature_names,
                values=all_values,
                lengths=all_lengths,
            )

            # Generate random labels (binary classification)
            labels = torch.randint(0, 2, (self.batch_size,)).float()

            yield dense_features, sparse_features, labels


# =============================================================================
# Training Functions
# =============================================================================


def setup_distributed() -> Tuple[int, int, torch.device]:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        # Single process mode
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_model(
    num_embeddings: int,
    embedding_dim: int,
    num_sparse_features: int,
    num_dense_features: int,
    dense_arch_layer_sizes: List[int],
    over_arch_layer_sizes: List[int],
    device: torch.device,
    world_size: int,
) -> nn.Module:
    """Create DLRM model with TorchRec embeddings."""

    # Create embedding configs for each sparse feature
    sparse_feature_names = [f"sparse_{i}" for i in range(num_sparse_features)]

    embedding_configs = [
        EmbeddingBagConfig(
            name=f"embedding_{i}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[sparse_feature_names[i]],
        )
        for i in range(num_sparse_features)
    ]

    # Create EmbeddingBagCollection
    embedding_bag_collection = EmbeddingBagCollection(
        tables=embedding_configs,
        device=torch.device("meta"),  # Use meta device for sharding
    )

    # Create DLRM model
    model = DLRM(
        embedding_bag_collection=embedding_bag_collection,
        dense_in_features=num_dense_features,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
    )

    if world_size > 1:
        # Use TorchRec's DistributedModelParallel for automatic sharding
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,
                compute_device="cuda",
            ),
            enumerator=EmbeddingEnumerator(
                topology=Topology(
                    world_size=world_size,
                    compute_device="cuda",
                ),
                estimator=[
                    EmbeddingPerfEstimator(
                        topology=Topology(world_size=world_size, compute_device="cuda")
                    ),
                    EmbeddingStorageEstimator(
                        topology=Topology(world_size=world_size, compute_device="cuda")
                    ),
                ],
            ),
        )

        sharders = cast(
            List[ModuleSharder[nn.Module]],
            [EmbeddingBagCollectionSharder()],
        )
        model = DistributedModelParallel(
            module=model,
            device=device,
            sharders=sharders,
            plan=planner.collective_plan(
                model,
                sharders,
                dist.GroupMember.WORLD,
            ),
        )
    else:
        # Single GPU mode
        model = model.to(device)

    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (dense_features, sparse_features, labels) in enumerate(dataloader):
        # Move data to device
        dense_features = dense_features.to(device)
        sparse_features = sparse_features.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(dense_features, sparse_features)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress
        if batch_idx % 10 == 0 and rank == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * dense_features.shape[0] / elapsed
            print(
                f"Epoch {epoch} | Batch {batch_idx} | "
                f"Loss: {loss.item():.4f} | "
                f"Throughput: {samples_per_sec:.0f} samples/sec"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TorchRec Cloud Quickstart - DLRM Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python train_torchrec_quickstart.py --batch_size 1024

  # Multi-GPU
  torchrun --nproc_per_node=4 train_torchrec_quickstart.py --batch_size 4096

  # Custom model size
  python train_torchrec_quickstart.py --num_embeddings 1000000 --embedding_dim 128
        """,
    )

    # Model configuration
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100000,
        help="Number of embeddings per table (default: 100000)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--num_sparse_features",
        type=int,
        default=26,
        help="Number of sparse (categorical) features (default: 26)",
    )
    parser.add_argument(
        "--num_dense_features",
        type=int,
        default=13,
        help="Number of dense (continuous) features (default: 13)",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Bottom MLP layer sizes (default: 512,256,64)",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,256,128,1",
        help="Top MLP layer sizes (default: 512,256,128,1)",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch size (default: 8192)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--num_batches_per_epoch",
        type=int,
        default=100,
        help="Number of batches per epoch (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=0,
        help="Save checkpoint every N iterations. 0 disables checkpointing (default: 0)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to save/load checkpoints (default: '')",
    )
    parser.add_argument(
        "--enable_spot_recovery",
        action="store_true",
        help="Enable spot instance recovery mode for AWS/Azure",
    )
    parser.add_argument(
        "--enable_preemption_recovery",
        action="store_true",
        help="Enable preemption recovery mode for GCP",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("TorchRec Cloud Quickstart - DLRM Training")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Batch size: {args.batch_size}")
        print(f"Num embeddings: {args.num_embeddings}")
        print(f"Embedding dim: {args.embedding_dim}")
        print("=" * 60)

    # Parse layer sizes
    dense_arch_layer_sizes = [int(x) for x in args.dense_arch_layer_sizes.split(",")]
    over_arch_layer_sizes = [int(x) for x in args.over_arch_layer_sizes.split(",")]

    # Create sparse feature names
    sparse_feature_names = [f"sparse_{i}" for i in range(args.num_sparse_features)]

    # Create model
    model = create_model(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        num_sparse_features=args.num_sparse_features,
        num_dense_features=args.num_dense_features,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        device=device,
        world_size=world_size,
    )

    # Create optimizer
    if world_size > 1:
        # Use KeyedOptimizerWrapper for distributed training
        optimizer = KeyedOptimizerWrapper(
            dict(model.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9
        )

    # Create loss function
    criterion = nn.BCELoss()

    # Create synthetic dataset
    dataset = SyntheticDataset(
        num_embeddings=args.num_embeddings,
        num_dense_features=args.num_dense_features,
        num_sparse_features=args.num_sparse_features,
        batch_size=args.batch_size,
        num_batches=args.num_batches_per_epoch,
        sparse_feature_names=sparse_feature_names,
    )

    # Note: IterableDataset doesn't need a sampler for distributed training
    # Each process will generate its own random data
    dataloader = DataLoader(dataset, batch_size=None)

    # Training loop
    if rank == 0:
        print("\nStarting training...")
        print("-" * 60)

    total_start_time = time.time()
    avg_loss = 0.0

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            rank=rank,
            epoch=epoch + 1,
        )
        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            samples_per_sec = args.num_batches_per_epoch * args.batch_size / epoch_time
            print(
                f"\nEpoch {epoch + 1}/{args.num_epochs} completed | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.2f}s | "
                f"Throughput: {samples_per_sec:.0f} samples/sec\n"
            )
            print("-" * 60)

    total_time = time.time() - total_start_time

    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final loss: {avg_loss:.4f}")
        print("=" * 60)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
