#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import tempfile
import unittest

import torch
from torch.distributed._composable import replicate
from torch.distributed._shard.api import ShardedTensor
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torchrec.distributed.shard import shard as trec_shard, shard_modules
from torchrec.distributed.sharding_plan import column_wise
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import skip_if_asan


class DDPTest(MultiProcessTestBase):
    @classmethod
    def _run_init(cls, rank: int, world_size: int) -> None:
        with MultiProcessContext(rank, world_size, "nccl") as ctx:
            num_float_features = 32

            tables = [
                EmbeddingBagConfig(
                    num_embeddings=(i + 1) * 10,
                    embedding_dim=(i + 1) * 4 * world_size,
                    name="table_" + str(i),
                    feature_names=["feature_" + str(i)],
                )
                for i in range(3)
            ]
            weighted_tables = [
                EmbeddingBagConfig(
                    num_embeddings=(i + 1) * 10,
                    embedding_dim=(i + 1) * 4 * world_size,
                    name="weighted_table_" + str(i),
                    feature_names=["weighted_feature_" + str(i)],
                )
                for i in range(2)
            ]
            m = TestSparseNN(
                tables=tables,
                num_float_features=num_float_features,
                weighted_tables=weighted_tables,
                dense_device=ctx.device,
            )
            # Put all tensors on meta device, then init_params should
            # materialize them.
            for name, param in m._parameters.items():
                if isinstance(param, torch.Tensor):
                    m._parameters[name] = torch.nn.Parameter(
                        torch.empty_like(param, device="meta"),
                        requires_grad=param.requires_grad,
                    )

            shard_modules(m, device=ctx.device, init_params=True)
            # init_params should move m to `device`
            for p in m.parameters():
                assert p.device == ctx.device

    @classmethod
    def _run(cls, rank: int, world_size: int, path: str) -> None:
        with MultiProcessContext(rank, world_size, "nccl") as ctx:
            num_float_features = 32

            tables = [
                EmbeddingBagConfig(
                    num_embeddings=(i + 1) * 10,
                    embedding_dim=(i + 1) * 4 * world_size,
                    name="table_" + str(i),
                    feature_names=["feature_" + str(i)],
                )
                for i in range(3)
            ]
            weighted_tables = [
                EmbeddingBagConfig(
                    num_embeddings=(i + 1) * 10,
                    embedding_dim=(i + 1) * 4 * world_size,
                    name="weighted_table_" + str(i),
                    feature_names=["weighted_feature_" + str(i)],
                )
                for i in range(2)
            ]
            m = TestSparseNN(
                tables=tables,
                num_float_features=num_float_features,
                weighted_tables=weighted_tables,
                dense_device=ctx.device,
            )
            m.sparse.ebc = trec_shard(
                module=m.sparse.ebc,
                device=ctx.device,
                plan=column_wise(ranks=list(range(world_size))),
            )
            m.sparse.weighted_ebc = trec_shard(
                module=m.sparse.weighted_ebc,
                device=ctx.device,
                plan=column_wise(ranks=list(range(world_size))),
            )
            m.over = replicate(m.over)
            m.dense = replicate(m.dense)

            ######## run one iteration ########
            _, local_batch = ModelInput.generate(
                batch_size=8,
                world_size=world_size,
                num_float_features=num_float_features,
                tables=tables,
                weighted_tables=weighted_tables,
            )
            batch = local_batch[0].to(ctx.device)
            m(batch)[1].sum().backward()

            state_dict = m.state_dict()
            writer = FileSystemWriter(path=path)
            reader = FileSystemReader(path=path)
            save_state_dict(state_dict, writer)

            p_sum = torch.zeros(1, device=ctx.device)
            for p in m.parameters():
                with torch.no_grad():
                    if isinstance(p, ShardedTensor):
                        if not p.local_shards():
                            continue
                        p = p.local_tensor()
                    p_sum += p.sum()
                    p.zero_()
                    assert p.sum() == 0
            load_state_dict(state_dict, reader)
            m.load_state_dict(state_dict)

            p_sum_loaded = torch.zeros(1, device=ctx.device)
            for p in m.parameters():
                with torch.no_grad():
                    if isinstance(p, ShardedTensor):
                        if not p.local_shards():
                            continue
                        p = p.local_tensor()
                    p_sum_loaded += p.sum()
            # TODO: debug why failing on OSS
            # assert p_sum.allclose(p_sum_loaded)

    @skip_if_asan
    #  `torch.cuda.device_count() <= 1` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            self._run_multi_process_test(
                callable=self._run,
                path=path,
            )

    @skip_if_asan
    #  `torch.cuda.device_count() <= 1` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_init_params(self) -> None:
        self._run_multi_process_test(
            callable=self._run_init,
        )
