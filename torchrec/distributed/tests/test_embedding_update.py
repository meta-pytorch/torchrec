#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.distributed import DistributedModelParallel
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
