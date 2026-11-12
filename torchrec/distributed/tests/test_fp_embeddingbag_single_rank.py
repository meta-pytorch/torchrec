#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st, Verbosity
from torch.distributed._shard.sharded_tensor import Shard, ShardMetadata
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSingleRankBase,
)
from torchrec.distributed.tests.test_fp_embeddingbag_utils import (
    create_module_and_freeze,
    get_configs,
    get_kjt_inputs,
    TestFPEBCSharder,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Not enough GPUs, this test requires at least one GPU",
)
class CheckpointLoadConcatDeviceTest(unittest.TestCase):
    def _load_checkpoint(
        self,
        checkpoint_weight: torch.Tensor,
        shard_offsets: list[list[int]],
        concat_on_cpu: bool,
        use_shards_wrapper: bool = False,
    ) -> tuple[list[torch.device], torch.Tensor]:
        """Load a full checkpoint tensor into a table with 2x2 local shards.

        Returns the devices of the concatenated tensor seen before the copy
        into the destination table, and the resulting table weight.
        """
        device = torch.device("cuda")
        module = ShardedEmbeddingBagCollection.__new__(ShardedEmbeddingBagCollection)
        nn.Module.__init__(module)
        table = nn.EmbeddingBag(num_embeddings=4, embedding_dim=2, device=device)
        module.embedding_bags = nn.ModuleDict({"table": table})
        module._checkpoint_load_concat_on_cpu = concat_on_cpu
        module._table_name_to_config = {
            "table": EmbeddingBagConfig(name="table", embedding_dim=2, num_embeddings=4)
        }
        # Citrine C3: create GPU checkpoint fixtures directly on the target device.
        local_tensors = [torch.empty(2, 2, device=device) for _ in shard_offsets]
        if use_shards_wrapper:
            module._model_parallel_name_to_local_shards = OrderedDict({"table": []})
            module._model_parallel_name_to_shards_wrapper = OrderedDict(
                {
                    "table": {
                        "local_tensors": local_tensors,
                        "local_offsets": shard_offsets,
                    }
                }
            )
        else:
            module._model_parallel_name_to_local_shards = OrderedDict(
                {
                    "table": [
                        Shard(
                            tensor=tensor,
                            metadata=ShardMetadata(
                                shard_sizes=[2, 2],
                                shard_offsets=offset,
                                placement="rank:0/cuda:0",
                            ),
                        )
                        for tensor, offset in zip(local_tensors, shard_offsets)
                    ]
                }
            )
            module._model_parallel_name_to_shards_wrapper = OrderedDict({"table": {}})
        module._lookups = []
        module._register_load_state_dict_pre_hook(
            module._pre_load_state_dict_hook, with_module=True
        )

        concatenated_devices: list[torch.device] = []

        def capture_checkpoint_tensor_device(
            sharded_ebc: ShardedEmbeddingBagCollection,
            state_dict: dict[str, object],
            prefix: str,
            *args: object,
        ) -> None:
            value = state_dict[f"{prefix}embedding_bags.table.weight"]
            if isinstance(value, torch.Tensor):
                concatenated_devices.append(value.device)

        module._register_load_state_dict_pre_hook(
            capture_checkpoint_tensor_device, with_module=True
        )
        module.load_state_dict(
            OrderedDict({"embedding_bags.table.weight": checkpoint_weight})
        )
        return concatenated_devices, table.weight

    def test_column_wise_shards_concatenate_on_cpu(self) -> None:
        checkpoint_weight = torch.arange(
            8, dtype=torch.float32, device=torch.device("cuda")
        ).view(2, 4)

        devices, weight = self._load_checkpoint(
            checkpoint_weight, shard_offsets=[[0, 0], [0, 2]], concat_on_cpu=True
        )

        self.assertEqual(devices, [torch.device("cpu")])
        torch.testing.assert_close(
            weight,
            torch.tensor(
                [[0, 1], [4, 5], [2, 3], [6, 7]],
                dtype=torch.float32,
                device=torch.device("cuda"),
            ),
        )

    def test_row_wise_shards_concatenate_on_cpu(self) -> None:
        checkpoint_weight = torch.arange(
            8, dtype=torch.float32, device=torch.device("cuda")
        ).view(4, 2)

        for use_shards_wrapper in (False, True):
            with self.subTest(use_shards_wrapper=use_shards_wrapper):
                devices, weight = self._load_checkpoint(
                    checkpoint_weight,
                    shard_offsets=[[0, 0], [2, 0]],
                    concat_on_cpu=True,
                    use_shards_wrapper=use_shards_wrapper,
                )

                self.assertEqual(devices, [torch.device("cpu")])
                torch.testing.assert_close(weight, checkpoint_weight)

    def test_row_wise_shards_concatenate_on_gpu_by_default(self) -> None:
        checkpoint_weight = torch.arange(
            8, dtype=torch.float32, device=torch.device("cuda")
        ).view(4, 2)

        devices, weight = self._load_checkpoint(
            checkpoint_weight, shard_offsets=[[0, 0], [2, 0]], concat_on_cpu=False
        )

        self.assertEqual([d.type for d in devices], ["cuda"])
        torch.testing.assert_close(weight, checkpoint_weight)


class FPModelParallelStateDictTest(ModelParallelSingleRankBase):
    def setUp(self, backend: str = "nccl") -> None:
        super().setUp(backend=backend)

        self.use_fp_collection = True

    def _create_tables(self) -> None:
        self.tables += get_configs()

    def _generate_batch(self) -> ModelInput:
        kjt_input_per_rank = get_kjt_inputs()
        batch = kjt_input_per_rank[0].to(self.device)
        # pyrefly: ignore[bad-return]
        return batch

    def _create_model(self) -> nn.Module:
        return create_module_and_freeze(
            tables=self.tables,
            use_fp_collection=self.use_fp_collection,
            device=torch.device("meta"),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ]
        ),
        is_training=st.booleans(),
        use_fp_collection=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_load_state_dict(
        self,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
        use_fp_collection: bool,
    ) -> None:
        self.use_fp_collection = use_fp_collection
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestFPEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                ),
            ),
        ]
        models, batch = self._generate_dmps_and_batch(
            sharders=sharders,
        )
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # validate the models are equivalent
        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)
        self._compare_models(m1, m2)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_numerical_equivalence_between_kernel_types(
        self,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
    ) -> None:
        self._set_table_weights_precision(dtype)
        fused_params = {
            "stochastic_rounding": stochastic_rounding,
            "cache_precision": dtype,
        }

        fused_sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestFPEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                    fused_params=fused_params,
                ),
            ),
        ]
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestFPEBCSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    fused_params=fused_params,
                ),
            ),
        ]

        (fused_model, _), _ = self._generate_dmps_and_batch(fused_sharders)
        (model, _), batch = self._generate_dmps_and_batch(sharders)

        # load the baseline model's state_dict onto the new model
        model.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", fused_model.state_dict())
        )

        if is_training:
            for _ in range(4):
                self._train_models(fused_model, model, batch)
        # skip eval here because it will cause numerical difference
        # TODO figure out why
        if not is_training or not stochastic_rounding:
            self._eval_models(
                fused_model, model, batch, is_deterministic=not stochastic_rounding
            )
        self._compare_models(
            fused_model, model, is_deterministic=not stochastic_rounding
        )
