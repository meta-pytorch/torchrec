#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from typing import cast, List

import hypothesis.strategies as st
import torch
import torch.nn as nn
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import given, settings, Verbosity
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.test_utils.emb_sharder import TestEBCSharder
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port, init_distributed_single_host

SHARDING_TYPES: List[str] = [
    ShardingType.TABLE_WISE.value,
    ShardingType.COLUMN_WISE.value,
    ShardingType.ROW_WISE.value,
]


class ShardedEmbeddingModuleCachePrefetchTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.backend = "nccl"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.pg = init_distributed_single_host(
            backend=self.backend, rank=0, world_size=1
        )

        self.embedding_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=4,
                num_embeddings=40,
            ),
        ]
        self.model = EmbeddingBagCollection(
            tables=self.embedding_config,
            device=self.device,
        )

    def get_cache_unique_misses(
        self, emb_module: SplitTableBatchedEmbeddingBagsCodegen
    ) -> int:
        (_, _, _, num_unique_misses, _, _) = (
            emb_module.get_uvm_cache_stats(use_local_cache=True).detach().cpu().tolist()
        )
        return num_unique_misses

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        sharding_type=st.sampled_from(SHARDING_TYPES),
        cache_load_factor=st.sampled_from([0.5, 0.8]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_sharded_ebc_cache_prefetch(
        self,
        sharding_type: str,
        cache_load_factor: float,
    ) -> None:
        batch_0_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([1, 2, 3]),
            lengths=torch.LongTensor([2, 0, 1]),
        ).to(self.device)

        batch_1_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([10, 11, 12]),
            lengths=torch.LongTensor([2, 0, 1]),
        ).to(self.device)

        batch_2_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([30, 31, 33]),
            lengths=torch.LongTensor([1, 1, 1]),
        ).to(self.device)

        fused_params = {
            "prefetch_pipeline": True,
            "gather_uvm_cache_stats": True,
            "cache_load_factor": cache_load_factor,
        }

        sharder = TestEBCSharder(
            sharding_type=sharding_type,
            kernel_type=EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            fused_params=fused_params,
        )
        sharded_model = DistributedModelParallel(
            module=self.model,
            env=ShardingEnv.from_process_group(self.pg),
            init_data_parallel=False,
            device=self.device,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    sharder,
                )
            ],
        )
        sharded_ebc = sharded_model.module
        self.assertIsInstance(sharded_ebc, ShardedEmbeddingBagCollection)
        lookups = sharded_ebc._lookups
        # pyrefly: ignore[bad-index, missing-attribute]
        emb_module = lookups[0]._emb_modules[0]._emb_module
        self.assertIsInstance(emb_module, SplitTableBatchedEmbeddingBagsCodegen)

        # Embedding lookup without prior prefetch
        sharded_ebc(batch_0_kjt)

        # We should have 3 unique misses since nothing was stored in cache yet
        self.assertEqual(self.get_cache_unique_misses(emb_module), 3)

        kjt_list = KJTList([batch_1_kjt])
        sharded_ebc.prefetch(kjt_list)

        # Reset cache stats so that local uvm cache stats are reset
        # otherwise, the number of cache misses will be non-zero after the forward pass
        emb_module.reset_uvm_cache_stats()

        # Embedding lookup will not prefetch here because prefetch() was called previously
        sharded_ebc(batch_1_kjt)

        # No unique misses expected since we prefetched all indices for batch 1
        self.assertEqual(self.get_cache_unique_misses(emb_module), 0)

        # Do forward pass w/ indices that were prefetched previously
        sharded_ebc(batch_1_kjt)

        # No unique misses expected since indices have been prefetched
        self.assertEqual(self.get_cache_unique_misses(emb_module), 0)

        # Try fetching indices different than the ones prefetched previously
        sharded_ebc(batch_2_kjt)

        # Unique cache misses should be 3 since indices requested by batch 2 are not presented in the cache
        self.assertEqual(self.get_cache_unique_misses(emb_module), 3)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        sharding_type=st.sampled_from(SHARDING_TYPES),
        cache_load_factor=st.sampled_from([0.5, 0.8]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_sharded_ebc_cache_purge(
        self,
        sharding_type: str,
        cache_load_factor: float,
    ) -> None:
        batch_1_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([5, 6, 7]),
            lengths=torch.LongTensor([2, 0, 1]),
        ).to(self.device)

        fused_params = {
            "prefetch_pipeline": True,
            "gather_uvm_cache_stats": True,
            "cache_load_factor": cache_load_factor,
        }

        sharder = TestEBCSharder(
            sharding_type=sharding_type,
            kernel_type=EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            fused_params=fused_params,
        )
        sharded_model = DistributedModelParallel(
            module=self.model,
            env=ShardingEnv.from_process_group(self.pg),
            init_data_parallel=False,
            device=self.device,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    sharder,
                )
            ],
        )
        sharded_ebc = sharded_model.module
        lookups = sharded_ebc._lookups
        # pyrefly: ignore[bad-index, missing-attribute]
        emb_module = lookups[0]._emb_modules[0]._emb_module

        kjt_list = KJTList([batch_1_kjt])
        # pyrefly: ignore[not-callable]
        sharded_ebc.prefetch(kjt_list)

        # Reset cache stats so that local uvm cache stats are reset
        # otherwise, the number of cache misses will be non-zero after the forward pass
        emb_module.reset_uvm_cache_stats()

        # No prefetch called here
        sharded_ebc(batch_1_kjt)
        self.assertEqual(self.get_cache_unique_misses(emb_module), 0)

        # Implicitly call cache purge by invoking pre-load_state_dict hook
        sharded_ebc.load_state_dict(sharded_ebc.state_dict())

        sharded_ebc(batch_1_kjt)

        # We should have 3 unique misses since we purged the cache after the first lookup
        self.assertEqual(self.get_cache_unique_misses(emb_module), 3)
