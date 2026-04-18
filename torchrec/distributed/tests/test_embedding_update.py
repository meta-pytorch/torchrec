#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.dist_data import _get_recat
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
    EmbeddingCollectionSharder,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig, NoEvictionPolicy
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestECModel(nn.Module):
    def __init__(self, tables: List[EmbeddingConfig], device: torch.device) -> None:
        super().__init__()
        self.ec = EmbeddingCollection(tables=tables, device=device)

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        return self.ec(features)


class ExpectedMsgNotFoundException(Exception):
    pass


class TestEmbeddingUpdate(MultiProcessTestBase):
    # Note all tests are disabled on OSS due to incompatibility

    def _gpu_check(self, world_size: int) -> None:
        if torch.cuda.device_count() < world_size:
            self.skipTest(
                f"Not enough GPUs, this test requires at least {world_size} GPUs"
            )

    def _get_example_configs_and_input(
        self,
    ) -> tuple[list[EmbeddingConfig], list[KeyedJaggedTensor]]:
        return (
            [
                EmbeddingConfig(
                    num_embeddings=8000,
                    embedding_dim=64,
                    name="table_0",
                    feature_names=["feature_0", "feature_1"],
                    total_num_buckets=20,
                    use_virtual_table=True,
                    enable_embedding_update=True,
                    virtual_table_eviction_policy=NoEvictionPolicy(),
                ),
                EmbeddingConfig(
                    num_embeddings=8000,
                    embedding_dim=64,
                    name="table_1",
                    feature_names=["feature_2"],
                    total_num_buckets=40,
                    use_virtual_table=True,
                    enable_embedding_update=True,
                    virtual_table_eviction_policy=NoEvictionPolicy(),
                ),
                EmbeddingConfig(
                    num_embeddings=8000,
                    embedding_dim=64,
                    name="table_2",
                    feature_names=["feature_3"],
                ),
            ],
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1", "feature_2", "feature_3"],
                    values=torch.randint(0, 8000, (13,)),
                    lengths=torch.LongTensor([2, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 0]),
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1", "feature_2", "feature_3"],
                    values=torch.randint(0, 8000, (12,)),
                    lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 0, 0, 3, 1, 0, 2]),
                ),
            ],
        )

    def test_sharded_embedding_update_disabled_in_oss_compatibility(
        self,
        # sharding_type: str,
        # kernel_type: str,
    ) -> None:
        WORLD_SIZE = 2
        self._gpu_check(WORLD_SIZE)
        tables, inputs_per_rank = self._get_example_configs_and_input()
        backend = "nccl"
        embeddings_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.cat(
                    (
                        input["feature_0"].values(),
                        input["feature_1"].values(),
                        input["feature_2"].values(),
                    )
                ),
                lengths=input.lengths()[: -input["feature_3"].lengths().size(0)],
                weights=torch.rand(
                    int(
                        torch.sum(
                            input.lengths()[: -input["feature_3"].lengths().size(0)]
                        ).item()
                    ),
                    64,
                    dtype=torch.float32,
                ),
            )
            for input in inputs_per_rank
        ]
        self._run_multi_process_test(
            callable=sharded_embedding_update,
            world_size=WORLD_SIZE,
            tables=tables,
            backend=backend,
            inputs_per_rank=inputs_per_rank,
            embeddings_per_rank=embeddings_per_rank,
        )

    def test_embedding_update_through_dmp_success_disabled_in_oss_compatibility(
        self,
    ) -> None:
        WORLD_SIZE = 2
        self._gpu_check(WORLD_SIZE)
        tables, inputs_per_rank = self._get_example_configs_and_input()
        embeddings_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.cat(
                    (
                        input["feature_0"].values(),
                        input["feature_1"].values(),
                        input["feature_2"].values(),
                    )
                ),
                lengths=input.lengths()[: -input["feature_3"].lengths().size(0)],
                weights=torch.rand(
                    int(
                        torch.sum(
                            input.lengths()[: -input["feature_3"].lengths().size(0)]
                        ).item()
                    ),
                    64,
                    dtype=torch.float32,
                ),
            )
            for input in inputs_per_rank
        ]
        self._run_multi_process_test(
            callable=sharded_embedding_update,
            world_size=WORLD_SIZE,
            tables=tables,
            backend="nccl",
            inputs_per_rank=inputs_per_rank,
            embeddings_per_rank=embeddings_per_rank,
            update_through_dmp=True,
        )

    def test_embedding_update_through_dmp_fail_disabled_in_oss_compatibility(
        self,
    ) -> None:
        WORLD_SIZE = 2
        self._gpu_check(WORLD_SIZE)
        tables, inputs_per_rank = self._get_example_configs_and_input()
        # feature 2 doesn't exist, should error out
        embeddings_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.cat(
                    (
                        input["feature_0"].values(),
                        input["feature_1"].values(),
                    )
                ),
                lengths=input.lengths()[
                    : -(
                        input["feature_3"].lengths().size(0)
                        + input["feature_2"].lengths().size(0)
                    )
                ],
                weights=torch.rand(
                    int(
                        torch.sum(
                            input.lengths()[
                                : -(
                                    input["feature_2"].lengths().size(0)
                                    + input["feature_3"].lengths().size(0)
                                )
                            ]
                        ).item()
                    ),
                    64,
                    dtype=torch.float32,
                ),
            )
            for input in inputs_per_rank
        ]
        self._run_multi_process_test(
            callable=sharded_embedding_update,
            world_size=WORLD_SIZE,
            tables=tables,
            backend="nccl",
            inputs_per_rank=inputs_per_rank,
            embeddings_per_rank=embeddings_per_rank,
            update_through_dmp=True,
            expected_failure_msg="write_dist feature names",
        )

    def test_embedding_update_variable_stride_kjt_disabled_in_oss_compatibility(
        self,
    ) -> None:
        WORLD_SIZE = 2
        self._gpu_check(WORLD_SIZE)
        tables, inputs_per_rank = self._get_example_configs_and_input()
        embedding_dim = 64
        # Build write embeddings derived from the forward inputs so shapes
        # match for verification, but provide stride_per_key_per_rank to
        # exercise the variable stride code path through write_dist / dist_init.
        embeddings_per_rank = []
        for input in inputs_per_rank:
            feat_0_vals = input["feature_0"].values()
            feat_1_vals = input["feature_1"].values()
            feat_2_vals = input["feature_2"].values()
            feat_0_lengths = input["feature_0"].lengths()
            feat_1_lengths = input["feature_1"].lengths()
            feat_2_lengths = input["feature_2"].lengths()

            all_values = torch.cat((feat_0_vals, feat_1_vals, feat_2_vals))
            all_lengths = torch.cat((feat_0_lengths, feat_1_lengths, feat_2_lengths))
            num_values = int(all_lengths.sum().item())

            stride_per_key_per_rank = [
                [int(feat_0_lengths.size(0))],
                [int(feat_1_lengths.size(0))],
                [int(feat_2_lengths.size(0))],
            ]

            kjt = KeyedJaggedTensor(
                keys=["feature_0", "feature_1", "feature_2"],
                values=all_values,
                lengths=all_lengths,
                weights=torch.rand(num_values, embedding_dim, dtype=torch.float32),
                stride_per_key_per_rank=stride_per_key_per_rank,
            )
            embeddings_per_rank.append(kjt)

        self._run_multi_process_test(
            callable=sharded_embedding_update,
            world_size=WORLD_SIZE,
            tables=tables,
            backend="nccl",
            inputs_per_rank=inputs_per_rank,
            embeddings_per_rank=embeddings_per_rank,
        )

    def test_embedding_update_config_not_enabled_disabled_in_oss_compatibility(
        self,
    ) -> None:
        WORLD_SIZE = 2
        self._gpu_check(WORLD_SIZE)
        tables, inputs_per_rank = self._get_example_configs_and_input()
        for table in tables:
            table.enable_embedding_update = False
        embeddings_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.cat(
                    (
                        input["feature_0"].values(),
                        input["feature_1"].values(),
                        input["feature_2"].values(),
                    )
                ),
                lengths=input.lengths()[: -input["feature_3"].lengths().size(0)],
                weights=torch.rand(
                    int(
                        torch.sum(
                            input.lengths()[: -input["feature_3"].lengths().size(0)]
                        ).item()
                    ),
                    64,
                    dtype=torch.float32,
                ),
            )
            for input in inputs_per_rank
        ]
        self._run_multi_process_test(
            callable=sharded_embedding_update,
            world_size=WORLD_SIZE,
            tables=tables,
            backend="nccl",
            inputs_per_rank=inputs_per_rank,
            embeddings_per_rank=embeddings_per_rank,
            update_through_dmp=True,
            expected_failure_msg="No writable sharded modules found",
        )


class TestDistInit2DWeightsVariableBatch(unittest.TestCase):
    """
    Unit tests for KeyedJaggedTensor.dist_init with 2D weights and variable
    batch sizes per rank.

    When stride_per_rank has different values across ranks, dist_init's
    non-variable_stride_per_key branch calls permute_1D_sparse_data (the vec
    kernel) instead of permute_2D_sparse_data. This exercises the fix in
    permute_1D_data_kernel_vec that adds correct 2D weights support.
    """

    def _reference_permute_1d_2d_weights(
        self,
        recat: torch.Tensor,
        lengths: torch.Tensor,
        values: torch.Tensor,
        weights_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Python reference for permute_1D_sparse_data with 2D weights.

        recat[i] = input segment index for output segment i.
        lengths[j] = number of values in input segment j.
        weights_2d has shape [total_indices, embedding_dim].

        The CPU kernel returns 1D weights (flattening 2D input), so this
        Python loop is used as ground truth instead.
        """
        input_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), lengths.long().cumsum(0)]
        )
        perm_lengths = lengths[recat.long()]
        out_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), perm_lengths.long().cumsum(0)]
        )
        total_out = int(perm_lengths.sum().item())
        embedding_dim = weights_2d.size(1)

        perm_values = torch.empty(total_out, dtype=values.dtype)
        perm_weights = torch.empty(total_out, embedding_dim, dtype=weights_2d.dtype)
        for i, r in enumerate(recat.tolist()):
            src = int(input_offsets[r].item())
            length = int(lengths[r].item())
            dst = int(out_offsets[i].item())
            perm_values[dst : dst + length] = values[src : src + length]
            perm_weights[dst : dst + length] = weights_2d[src : src + length]
        return perm_lengths.int(), perm_values, perm_weights

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dist_init_2d_weights_variable_batch_per_rank(self) -> None:
        """
        Verify that dist_init correctly permutes 2D weights [N, embedding_dim]
        when stride_per_rank differs across ranks (variable batch per rank).

        Setup:
          - 2 keys, world_size=2, stride_per_rank=[3, 5] (rank0≠rank1)
          - single_batch_per_rank=False  →  permute_1D_sparse_data is called
          - weights shape: [total_indices, embedding_dim] — the 2D case

        Compared against a Python reference loop (CPU kernel flattens 2D
        weights to 1D and cannot serve as ground truth).
        """
        device = torch.device("cuda:0")
        keys = ["feature_0", "feature_1"]
        num_keys = len(keys)
        world_size = 2
        # Different strides trigger the variable-batch branch in dist_init,
        # which calls permute_1D_sparse_data with 2D weights.
        stride_per_rank = [3, 5]
        embedding_dim = 64

        # After AllToAll, lengths has num_keys * sum(stride_per_rank) segments:
        #   [key0_r0(3), key0_r1(5), key1_r0(3), key1_r1(5)]
        torch.manual_seed(42)
        num_segments = num_keys * sum(stride_per_rank)
        lengths = torch.randint(1, 4, (num_segments,), dtype=torch.int32)
        total_indices = int(lengths.sum().item())
        values = torch.randint(0, 1000, (total_indices,), dtype=torch.int32)
        # 2D weights: each index has an embedding_dim-dimensional weight row
        weights_2d = torch.rand(total_indices, embedding_dim, dtype=torch.float32)

        # _get_recat with different batch_size_per_rank calls
        # expand_into_jagged_permute to produce an element-level permutation of
        # the lengths array. dist_init then passes this as the segment permute
        # to permute_1D_sparse_data.
        recat_cpu = _get_recat(
            local_split=num_keys,
            num_splits=world_size,
            device=torch.device("cpu"),
            batch_size_per_rank=stride_per_rank,
        )
        recat_cuda = _get_recat(
            local_split=num_keys,
            num_splits=world_size,
            device=device,
            batch_size_per_rank=stride_per_rank,
        )
        assert recat_cpu is not None

        # Python reference (CPU kernel cannot serve as ground truth since it
        # flattens 2D weights to 1D)
        ref_lengths, ref_values, ref_weights = self._reference_permute_1d_2d_weights(
            recat_cpu, lengths, values, weights_2d
        )

        # CUDA: goes through permute_1D_data_kernel_vec with 2D weights fix
        cuda_kjt = KeyedJaggedTensor.dist_init(
            keys=keys,
            tensors=[
                lengths.to(device),
                values.to(device),
                weights_2d.to(device),
            ],
            variable_stride_per_key=False,
            num_workers=world_size,
            recat=recat_cuda,
            stride_per_rank=stride_per_rank,
        )

        torch.testing.assert_close(cuda_kjt.lengths().cpu(), ref_lengths)
        torch.testing.assert_close(cuda_kjt.values().cpu(), ref_values)
        assert cuda_kjt.weights() is not None
        torch.testing.assert_close(cuda_kjt.weights().cpu(), ref_weights)
        # Confirm weights remain 2D after permutation
        self.assertEqual(cuda_kjt.weights().dim(), 2)
        self.assertEqual(cuda_kjt.weights().size(1), embedding_dim)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dist_init_2d_weights_various_embedding_dims(self) -> None:
        """
        Test dist_init with 2D weights for embedding_dim values that exercise
        both vec4 (dim divisible by 4) and scalar fallback (dim not divisible
        by 4) code paths in permute_1D_data_kernel_vec.
        """
        device = torch.device("cuda:0")
        keys = ["feature_0", "feature_1", "feature_2"]
        num_keys = len(keys)
        world_size = 2
        stride_per_rank = [4, 7]  # different → variable-batch branch

        for embedding_dim in [4, 7, 8, 16, 64]:
            with self.subTest(embedding_dim=embedding_dim):
                torch.manual_seed(embedding_dim)
                num_segments = num_keys * sum(stride_per_rank)
                lengths = torch.randint(1, 5, (num_segments,), dtype=torch.int32)
                total_indices = int(lengths.sum().item())
                values = torch.randint(0, 1000, (total_indices,), dtype=torch.int32)
                weights_2d = torch.rand(
                    total_indices, embedding_dim, dtype=torch.float32
                )

                recat_cpu = _get_recat(
                    local_split=num_keys,
                    num_splits=world_size,
                    device=torch.device("cpu"),
                    batch_size_per_rank=stride_per_rank,
                )
                recat_cuda = _get_recat(
                    local_split=num_keys,
                    num_splits=world_size,
                    device=device,
                    batch_size_per_rank=stride_per_rank,
                )
                assert recat_cpu is not None

                ref_lengths, ref_values, ref_weights = (
                    self._reference_permute_1d_2d_weights(
                        recat_cpu, lengths, values, weights_2d
                    )
                )

                cuda_kjt = KeyedJaggedTensor.dist_init(
                    keys=keys,
                    tensors=[
                        lengths.to(device),
                        values.to(device),
                        weights_2d.to(device),
                    ],
                    variable_stride_per_key=False,
                    num_workers=world_size,
                    recat=recat_cuda,
                    stride_per_rank=stride_per_rank,
                )

                assert cuda_kjt.weights() is not None
                torch.testing.assert_close(cuda_kjt.lengths().cpu(), ref_lengths)
                torch.testing.assert_close(cuda_kjt.values().cpu(), ref_values)
                torch.testing.assert_close(cuda_kjt.weights().cpu(), ref_weights)
                self.assertEqual(cuda_kjt.weights().size(1), embedding_dim)


def sharded_embedding_update(
    rank: int,
    world_size: int,
    tables: List[EmbeddingConfig],
    backend: str,
    embeddings_per_rank: List[KeyedJaggedTensor],
    inputs_per_rank: List[KeyedJaggedTensor],
    local_size: Optional[int] = None,
    update_through_dmp: bool = False,  # update through DMP or ShardedEmbeddingCollection
    expected_failure_msg: (
        str | None
    ) = None,  # a string that will appear in the error message if the test fails
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        assert ctx.pg is not None
        model = TestECModel(
            tables=tables,
            device=ctx.device,
        )

        sharder = EmbeddingCollectionSharder()
        per_param_sharding = {
            "table_0": row_wise(
                compute_kernel=EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value
            ),
            "table_1": row_wise(
                compute_kernel=EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value
            ),
            "table_2": data_parallel(),
        }
        sharding_plan = construct_module_sharding_plan(
            model.ec,
            per_param_sharding=per_param_sharding,
            local_size=local_size,
            world_size=world_size,
            device_type=ctx.device.type,
            # pyrefly: ignore[bad-argument-type]
            sharder=sharder,
        )

        set_propogate_device(True)
        sharded_model = DistributedModelParallel(
            model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=ShardingPlan({"ec": sharding_plan}),
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )

        kjts = inputs_per_rank[rank]
        sharded_model(kjts.to(ctx.device))
        torch.cuda.synchronize()
        failure_raised = False
        try:
            if update_through_dmp:
                sharded_model.write(embeddings_per_rank[rank].to(ctx.device))
            else:
                # pyrefly: ignore[missing-attribute]
                sharded_model._dmp_wrapped_module.ec.write(
                    embeddings_per_rank[rank].to(ctx.device)
                )
        except Exception as e:
            failure_raised = True
            if expected_failure_msg is None:
                raise ExpectedMsgNotFoundException(
                    f"Expected failure message is None but an exception was raised: {e}"
                )
            if expected_failure_msg not in str(e):
                raise ExpectedMsgNotFoundException(
                    f"Expected failure message {expected_failure_msg} not found in the actual exception: {e}"
                )
            return

        assert (
            expected_failure_msg is None and not failure_raised
        ), "Expected the run to fail but it succeeded"

        torch.cuda.synchronize()
        expected_embeddings = {
            key: embeddings_per_rank[rank][key].weights()
            for key in embeddings_per_rank[rank].keys()
        }
        embeddings = None
        embeddings = sharded_model(kjts.to(ctx.device))
        for key, values in expected_embeddings.items():
            torch.testing.assert_close(
                torch.cat(embeddings[key].to_dense()),
                values.to_dense().to(ctx.device),
                rtol=1e-3,
                atol=1e-3,
            )
