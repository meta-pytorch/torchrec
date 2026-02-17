#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
TorchX component for running TorchRec DLRM training on AWS EKS.

.. deprecated::
    TorchX is no longer actively maintained. For new deployments, use
    ``torchrun`` with Kubernetes Jobs or Kubeflow Training Operator instead.
    See aws/README.md for recommended deployment options.

This module provides a TorchX AppDef for distributed DLRM training
on AWS using EKS (Elastic Kubernetes Service) with GPU instances.

Example usage:
    torchx run -s kubernetes -cfg namespace=default \\
        cloud_deployment.aws.cloud_component:run_dlrm_aws \\
        --num_trainers 16 \\
        --dataset_path /fsx/criteo/ \\
        --batch_size 2048

For more information, see:
    https://pytorch.org/torchx/
"""

import os
from typing import List, Optional

import torchx.specs as specs
from torchx.components.dist import ddp


def run_dlrm_aws(
    num_trainers: int = 8,
    dataset_path: str = "/fsx/criteo/",
    batch_size: int = 2048,
    learning_rate: float = 1.0,
    embedding_dim: int = 128,
    num_epochs: int = 1,
    checkpoint_path: Optional[str] = None,
    *script_args: str,
) -> specs.AppDef:
    """
    TorchX AppDef for running DLRM training on AWS EKS.

    This component configures distributed training optimized for AWS p4d.24xlarge
    instances with 8x A100 GPUs each. It sets up appropriate NCCL environment
    variables for EFA (Elastic Fabric Adapter) networking.

    Args:
        num_trainers: Total number of GPU trainers. Must be multiple of 8 for
            multi-node training. Default: 8 (single node).
        dataset_path: Path to preprocessed Criteo dataset on FSx for Lustre.
            Default: /fsx/criteo/
        batch_size: Global batch size for training. Default: 2048.
        learning_rate: Learning rate for optimizer. Default: 1.0
        embedding_dim: Embedding dimension for sparse features. Default: 128.
        num_epochs: Number of training epochs. Default: 1.
        checkpoint_path: Optional S3 path for saving checkpoints.
        script_args: Additional arguments to pass to train_dlrm.py.

    Returns:
        specs.AppDef: TorchX application definition for the training job.

    Raises:
        ValueError: If num_trainers > 8 and not a multiple of 8.

    Example:
        >>> app = run_dlrm_aws(num_trainers=16, batch_size=4096)
        >>> # Submit to AWS EKS
        >>> torchx run -s kubernetes app
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    entrypoint = os.path.join(cwd, "..", "..", "golden_training", "train_dlrm.py")

    # AWS Deep Learning Container image with PyTorch and TorchRec
    image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-ec2"

    # Validate multi-node configuration
    if num_trainers > 8 and num_trainers % 8 != 0:
        raise ValueError(
            f"Multi-node training requires trainers in multiples of 8, got {num_trainers}. "
            "Each p4d.24xlarge node has 8 GPUs."
        )

    nproc_per_node = 8 if num_trainers >= 8 else num_trainers
    num_replicas = max(num_trainers // 8, 1)

    # Criteo 1TB embedding table configuration
    num_embeddings_per_feature = (
        "40000000,39060,17295,7424,20265,3,7122,1543,63,"
        "40000000,3067956,405282,10,2209,11938,155,4,976,14,"
        "40000000,40000000,40000000,590152,12973,108,36"
    )

    # Build training arguments
    train_args: List[str] = [
        f"--num_embeddings_per_feature={num_embeddings_per_feature}",
        f"--embedding_dim={embedding_dim}",
        "--dense_arch_layer_sizes=512,256,128",
        "--over_arch_layer_sizes=1024,1024,512,256,1",
        f"--batch_size={batch_size}",
        f"--learning_rate={learning_rate}",
        f"--num_epochs={num_epochs}",
        f"--dataset_path={dataset_path}",
    ]

    if checkpoint_path:
        train_args.append(f"--checkpoint_path={checkpoint_path}")

    train_args.extend(script_args)

    # Environment variables optimized for AWS EFA
    env = {
        # NCCL settings for EFA
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0",
        "NCCL_NET_GDR_LEVEL": "2",
        "NCCL_PROTO": "simple",
        # EFA settings
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "FI_PROVIDER": "efa",
        # Improve stability
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_TIMEOUT": "22",
    }

    return ddp(
        *train_args,
        name="torchrec-dlrm-aws",
        image=image,
        # AWS p4d.24xlarge instance specs
        cpu=96,
        gpu=8,
        memMB=1100000,  # ~1.1 TB
        script=entrypoint,
        j=f"{num_replicas}x{nproc_per_node}",
        env=env,
    )


def run_dlrm_aws_spot(
    num_trainers: int = 8,
    dataset_path: str = "/fsx/criteo/",
    checkpoint_interval: int = 1000,
    checkpoint_path: str = "s3://your-bucket/checkpoints/",
) -> specs.AppDef:
    """
    TorchX AppDef for DLRM training on AWS Spot instances.

    This variant is optimized for cost savings using Spot instances with
    frequent checkpointing for fault tolerance.

    Args:
        num_trainers: Total number of GPU trainers.
        dataset_path: Path to dataset on FSx for Lustre.
        checkpoint_interval: Save checkpoint every N iterations.
        checkpoint_path: S3 path for checkpoint storage.

    Returns:
        specs.AppDef: TorchX application definition.
    """
    return run_dlrm_aws(
        num_trainers,
        dataset_path,
        2048,  # batch_size
        1.0,  # learning_rate
        128,  # embedding_dim
        1,  # num_epochs
        checkpoint_path,
        f"--checkpoint_interval={checkpoint_interval}",
        "--enable_spot_recovery",
    )
