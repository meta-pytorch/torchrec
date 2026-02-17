#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Tests for cloud deployment components (AWS, Azure, GCP).

Validates argument construction, environment variable configuration,
input validation, and multi-node setup for each cloud provider.
"""

import unittest

from torchrec.github.examples.cloud_deployment.aws.cloud_component import (
    run_dlrm_aws,
    run_dlrm_aws_spot,
)
from torchrec.github.examples.cloud_deployment.azure.cloud_component import (
    run_dlrm_azure,
    run_dlrm_azure_spot,
)
from torchrec.github.examples.cloud_deployment.gcp.cloud_component import (
    run_dlrm_gcp,
    run_dlrm_gcp_preemptible,
)


class AWSComponentTest(unittest.TestCase):
    """Tests for the AWS TorchX component."""

    def test_returns_valid_app_def(self) -> None:
        """Test that run_dlrm_aws returns a valid AppDef."""
        app = run_dlrm_aws(num_trainers=8)
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "torchrec-dlrm-aws")

    def test_single_node_configuration(self) -> None:
        """Test single node (8 GPUs) configuration."""
        app = run_dlrm_aws(num_trainers=8)
        self.assertEqual(len(app.roles), 1)
        role = app.roles[0]
        self.assertEqual(role.num_replicas, 1)
        self.assertEqual(role.resource.gpu, 8)
        self.assertEqual(role.resource.cpu, 96)

    def test_multi_node_configuration(self) -> None:
        """Test multi-node configuration with 16 trainers (2 nodes)."""
        app = run_dlrm_aws(num_trainers=16)
        role = app.roles[0]
        self.assertEqual(role.num_replicas, 2)

    def test_invalid_multi_node_raises_error(self) -> None:
        """Test that non-multiple-of-8 trainers raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            run_dlrm_aws(num_trainers=10)
        self.assertIn("multiples of 8", str(ctx.exception))

    def test_less_than_8_trainers_valid(self) -> None:
        """Test that fewer than 8 trainers does not raise."""
        app = run_dlrm_aws(num_trainers=4)
        self.assertIsNotNone(app)

    def test_efa_environment_variables_set(self) -> None:
        """Test that AWS EFA networking env vars are configured."""
        app = run_dlrm_aws(num_trainers=8)
        role = app.roles[0]
        env = role.env
        self.assertEqual(env["FI_EFA_USE_DEVICE_RDMA"], "1")
        self.assertEqual(env["FI_PROVIDER"], "efa")
        self.assertEqual(env["NCCL_IB_DISABLE"], "0")

    def test_custom_batch_size_passed(self) -> None:
        """Test that custom batch_size is included in the training args."""
        app = run_dlrm_aws(num_trainers=8, batch_size=4096)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--batch_size=4096", args_str)

    def test_custom_embedding_dim_passed(self) -> None:
        """Test that custom embedding_dim is included in the training args."""
        app = run_dlrm_aws(num_trainers=8, embedding_dim=256)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--embedding_dim=256", args_str)

    def test_checkpoint_path_included(self) -> None:
        """Test that checkpoint path is included when provided."""
        app = run_dlrm_aws(
            num_trainers=8, checkpoint_path="s3://my-bucket/checkpoints/"
        )
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--checkpoint_path=s3://my-bucket/checkpoints/", args_str)

    def test_no_checkpoint_path_when_none(self) -> None:
        """Test that checkpoint_path is not included when None."""
        app = run_dlrm_aws(num_trainers=8, checkpoint_path=None)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertNotIn("--checkpoint_path", args_str)

    def test_spot_variant_includes_spot_recovery(self) -> None:
        """Test that the Spot variant passes --enable_spot_recovery."""
        app = run_dlrm_aws_spot(num_trainers=8)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--enable_spot_recovery", args_str)

    def test_spot_variant_includes_checkpoint_interval(self) -> None:
        """Test that the Spot variant passes --checkpoint_interval."""
        app = run_dlrm_aws_spot(num_trainers=8, checkpoint_interval=500)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--checkpoint_interval=500", args_str)

    def test_memory_matches_p4d_spec(self) -> None:
        """Test that memory is set for p4d.24xlarge (~1.1TB)."""
        app = run_dlrm_aws(num_trainers=8)
        role = app.roles[0]
        self.assertEqual(role.resource.memMB, 1100000)


class AzureComponentTest(unittest.TestCase):
    """Tests for the Azure TorchX component."""

    def test_returns_valid_app_def(self) -> None:
        """Test that run_dlrm_azure returns a valid AppDef."""
        app = run_dlrm_azure(num_trainers=8)
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "torchrec-dlrm-azure")

    def test_single_node_configuration(self) -> None:
        """Test single node configuration."""
        app = run_dlrm_azure(num_trainers=8)
        role = app.roles[0]
        self.assertEqual(role.num_replicas, 1)
        self.assertEqual(role.resource.gpu, 8)
        self.assertEqual(role.resource.cpu, 96)

    def test_multi_node_configuration(self) -> None:
        """Test multi-node configuration with 24 trainers (3 nodes)."""
        app = run_dlrm_azure(num_trainers=24)
        role = app.roles[0]
        self.assertEqual(role.num_replicas, 3)

    def test_invalid_multi_node_raises_error(self) -> None:
        """Test that non-multiple-of-8 trainers raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            run_dlrm_azure(num_trainers=12)
        self.assertIn("multiples of 8", str(ctx.exception))

    def test_infiniband_environment_variables_set(self) -> None:
        """Test that Azure InfiniBand env vars are configured."""
        app = run_dlrm_azure(num_trainers=8)
        role = app.roles[0]
        env = role.env
        self.assertEqual(env["NCCL_IB_DISABLE"], "0")
        self.assertEqual(env["NCCL_IB_HCA"], "mlx5")
        self.assertIn("UCX_TLS", env)

    def test_custom_dataset_path_passed(self) -> None:
        """Test that custom dataset_path is included in the training args."""
        app = run_dlrm_azure(num_trainers=8, dataset_path="/mnt/custom/data/")
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--dataset_path=/mnt/custom/data/", args_str)

    def test_spot_variant_includes_spot_recovery(self) -> None:
        """Test that the Azure Spot variant passes --enable_spot_recovery."""
        app = run_dlrm_azure_spot(num_trainers=8)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--enable_spot_recovery", args_str)

    def test_memory_matches_nd96_spec(self) -> None:
        """Test that memory is set for ND96asr_v4 (~900GB)."""
        app = run_dlrm_azure(num_trainers=8)
        role = app.roles[0]
        self.assertEqual(role.resource.memMB, 900000)


class GCPComponentTest(unittest.TestCase):
    """Tests for the GCP TorchX component."""

    def test_returns_valid_app_def(self) -> None:
        """Test that run_dlrm_gcp returns a valid AppDef."""
        app = run_dlrm_gcp(num_trainers=8)
        self.assertIsNotNone(app)
        self.assertEqual(app.name, "torchrec-dlrm-gcp")

    def test_single_node_configuration(self) -> None:
        """Test single node configuration."""
        app = run_dlrm_gcp(num_trainers=8)
        role = app.roles[0]
        self.assertEqual(role.num_replicas, 1)
        self.assertEqual(role.resource.gpu, 8)
        self.assertEqual(role.resource.cpu, 96)

    def test_multi_node_configuration(self) -> None:
        """Test multi-node configuration with 32 trainers (4 nodes)."""
        app = run_dlrm_gcp(num_trainers=32)
        role = app.roles[0]
        self.assertEqual(role.num_replicas, 4)

    def test_invalid_multi_node_raises_error(self) -> None:
        """Test that non-multiple-of-8 trainers raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            run_dlrm_gcp(num_trainers=14)
        self.assertIn("multiples of 8", str(ctx.exception))

    def test_gcp_disables_infiniband(self) -> None:
        """Test that GCP disables InfiniBand (not available on GCP)."""
        app = run_dlrm_gcp(num_trainers=8)
        role = app.roles[0]
        env = role.env
        self.assertEqual(env["NCCL_IB_DISABLE"], "1")

    def test_gcp_enables_p2p(self) -> None:
        """Test that GCP enables P2P communication."""
        app = run_dlrm_gcp(num_trainers=8)
        role = app.roles[0]
        env = role.env
        self.assertEqual(env["NCCL_P2P_DISABLE"], "0")

    def test_custom_learning_rate_passed(self) -> None:
        """Test that custom learning_rate is included in the training args."""
        app = run_dlrm_gcp(num_trainers=8, learning_rate=0.5)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--learning_rate=0.5", args_str)

    def test_custom_num_epochs_passed(self) -> None:
        """Test that custom num_epochs is included in the training args."""
        app = run_dlrm_gcp(num_trainers=8, num_epochs=5)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--num_epochs=5", args_str)

    def test_preemptible_variant_includes_recovery(self) -> None:
        """Test that GCP preemptible variant includes recovery flag."""
        app = run_dlrm_gcp_preemptible(num_trainers=8)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--enable_preemption_recovery", args_str)

    def test_preemptible_variant_checkpoint_interval(self) -> None:
        """Test that preemptible variant passes custom checkpoint_interval."""
        app = run_dlrm_gcp_preemptible(num_trainers=8, checkpoint_interval=2000)
        role = app.roles[0]
        args_str = " ".join(role.args)
        self.assertIn("--checkpoint_interval=2000", args_str)

    def test_memory_matches_a2_highgpu_spec(self) -> None:
        """Test that memory is set for a2-highgpu-8g (~1.3TB)."""
        app = run_dlrm_gcp(num_trainers=8)
        role = app.roles[0]
        self.assertEqual(role.resource.memMB, 1300000)


class CrossCloudConsistencyTest(unittest.TestCase):
    """Tests for consistency across all cloud provider components."""

    def test_all_providers_include_nccl_debug(self) -> None:
        """Test that all providers set NCCL_DEBUG=INFO."""
        for fn in [run_dlrm_aws, run_dlrm_azure, run_dlrm_gcp]:
            app = fn(num_trainers=8)
            role = app.roles[0]
            self.assertEqual(
                role.env.get("NCCL_DEBUG"),
                "INFO",
                f"NCCL_DEBUG not set for {fn.__name__}",
            )

    def test_all_providers_include_embedding_config(self) -> None:
        """Test that all providers pass --num_embeddings_per_feature."""
        for fn in [run_dlrm_aws, run_dlrm_azure, run_dlrm_gcp]:
            app = fn(num_trainers=8)
            role = app.roles[0]
            args_str = " ".join(role.args)
            self.assertIn(
                "--num_embeddings_per_feature=",
                args_str,
                f"Missing num_embeddings_per_feature for {fn.__name__}",
            )

    def test_all_providers_reject_invalid_trainers(self) -> None:
        """Test that all providers raise ValueError for invalid trainer counts."""
        for fn in [run_dlrm_aws, run_dlrm_azure, run_dlrm_gcp]:
            with self.assertRaises(ValueError, msg=f"{fn.__name__} should reject 10"):
                fn(num_trainers=10)

    def test_all_providers_accept_fewer_than_8(self) -> None:
        """Test that all providers accept fewer than 8 trainers."""
        for fn in [run_dlrm_aws, run_dlrm_azure, run_dlrm_gcp]:
            app = fn(num_trainers=4)
            self.assertIsNotNone(app, f"{fn.__name__} should accept 4 trainers")

    def test_all_providers_use_8_gpus_per_node(self) -> None:
        """Test that all providers configure 8 GPUs per node."""
        for fn in [run_dlrm_aws, run_dlrm_azure, run_dlrm_gcp]:
            app = fn(num_trainers=8)
            role = app.roles[0]
            self.assertEqual(
                role.resource.gpu,
                8,
                f"GPU count wrong for {fn.__name__}",
            )

    def test_all_providers_scale_replicas_correctly(self) -> None:
        """Test that all providers compute correct replica count for multi-node."""
        for fn in [run_dlrm_aws, run_dlrm_azure, run_dlrm_gcp]:
            app = fn(num_trainers=24)
            role = app.roles[0]
            self.assertEqual(
                role.num_replicas,
                3,
                f"Replica count wrong for {fn.__name__} with 24 trainers",
            )


if __name__ == "__main__":
    unittest.main()
