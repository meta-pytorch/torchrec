#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest
from typing import Dict, List, Tuple

import torch
from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.quant_state import sharded_tbes_weights_spec, WeightSpec
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    assert_weight_spec,
    create_test_model,
    create_test_model_ebc_only_no_quantize,
    model_input_to_forward_args,
    prep_inputs,
    quantize,
    shard_qebc,
)
from torchrec.fx import symbolic_trace
from torchrec.inference.modules import set_pruning_data
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

SPARSE_NN_EBC_MODULE = "_module.sparse.ebc"
SEQUENTIAL_NN_EBC_MODULE = "0"


def prune_and_quantize_model(
    model: torch.nn.Module,
    pruning_ebc_dict: Dict[str, int],
) -> torch.nn.Module:
    set_pruning_data(model, pruning_ebc_dict)

    quant_state_dict_split_scale_bias = True
    quant_model = quantize(
        module=model,
        inplace=False,
        quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
    )

    return quant_model


def create_quant_and_sharded_ebc_models(
    num_embedding: int,
    emb_dim: int,
    world_size: int,
    batch_size: int,
    sharding_type: ShardingType,
    device: torch.device,
    feature_processor: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, int]]:
    mi = create_test_model_ebc_only_no_quantize(
        num_embedding,
        emb_dim,
        world_size,
        batch_size,
        num_features=1,
        num_weighted_features=0,
        dense_device=device,
        sparse_device=device,
        feature_processor=feature_processor,
    )
    mi.model.to(device)
    num_rows_post_pruned = num_embedding // 2

    pruning_ebc_dict = {"table_0": num_rows_post_pruned}
    quant_model = prune_and_quantize_model(mi.model, pruning_ebc_dict)

    quant_model = quant_model[0]
    mi.quant_model = quant_model

    sharded_model = shard_qebc(
        mi,
        sharding_type=sharding_type,
        device=device,
        expected_shards=None,
        feature_processor=feature_processor,
    )

    sharded_model.load_state_dict(quant_model.state_dict())

    return quant_model, sharded_model, pruning_ebc_dict


class QuantPruneTest(unittest.TestCase):
    def check_tbe_pruned(
        self, sharded_model: torch.nn.Module, pruned_dict: Dict[str, int]
    ) -> None:
        for module in sharded_model.modules():
            if module.__class__.__name__ == "IntNBitTableBatchedEmbeddingBagsCodegen":
                #  `Union[Tensor, Module]`.
                for i, spec in enumerate(module.embedding_specs):
                    if spec[0] in pruned_dict:
                        self.assertEqual(
                            module.split_embedding_weights()[i][0].size(0),
                            pruned_dict[spec[0]],
                        )
                        self.assertEqual(
                            spec[1],
                            pruned_dict[spec[0]],
                        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_tw(self) -> None:
        batch_size: int = 4
        world_size = 2
        local_device = torch.device("cuda:0")

        num_embedding = 100
        emb_dim = 64
        pruned_entry = 40

        # hash, dim, pruned_hash
        table_specs: List[Tuple[int, int, int]] = [
            (num_embedding, emb_dim, num_embedding),
            (num_embedding, emb_dim, num_embedding - pruned_entry),
        ]
        pruning_ebc_dict: Dict[str, int] = {}
        pruning_ebc_dict["table_1"] = num_embedding - pruned_entry

        mi = create_test_model(
            num_embedding,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=len(table_specs),
            pruning_dict=pruning_ebc_dict,
        )

        expected_shards = [
            [
                (
                    (0, 0, table_specs[0][2], table_specs[0][1]),
                    "rank:0/cuda:0",
                ),
            ],
            [
                (
                    (0, 0, table_specs[1][2], table_specs[1][1]),
                    "rank:1/cuda:1",
                ),
            ],
        ]

        quant_model = mi.quant_model
        quant_state_dict = quant_model.state_dict()

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.TABLE_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(quant_state_dict)
        quant_output = quant_model(*inputs[0])
        sharded_output = sharded_model(*inputs[0])
        assert_close(quant_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        buffer = io.BytesIO()
        torch.jit.save(gm_script, buffer)
        buffer.seek(0)
        loaded_gm_script = torch.jit.load(buffer)
        gm_script_output = loaded_gm_script(*inputs[0])
        assert_close(quant_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0", "table_1"],
            ShardingType.TABLE_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_tw_one_ebc(self) -> None:
        batch_size: int = 1
        world_size: int = 2
        local_device = torch.device("cuda:0")
        num_embedding = 200
        emb_dim = 10
        sharding_type = ShardingType.TABLE_WISE

        quant_model, sharded_model, pruned_dict = create_quant_and_sharded_ebc_models(
            num_embedding=num_embedding,
            emb_dim=emb_dim,
            world_size=world_size,
            batch_size=batch_size,
            sharding_type=sharding_type,
            device=local_device,
        )

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.tensor([0, 1, 2], dtype=torch.int32).cuda(),
            lengths=torch.tensor([1, 1, 1], dtype=torch.int32).cuda(),
            weights=None,
        )

        q_output = quant_model(kjt)
        s_output = sharded_model(kjt)

        assert_close(q_output["feature_0"], s_output["feature_0"])

        self.check_tbe_pruned(sharded_model, pruned_dict)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_cw(self) -> None:
        batch_size: int = 4
        world_size = 2
        local_device = torch.device("cuda:0")

        num_embedding = 200
        emb_dim = 512
        pruned_entry = 100

        # hash, dim, pruned_hash
        table_specs: List[Tuple[int, int, int]] = [
            (num_embedding, emb_dim, num_embedding - pruned_entry),
        ]
        pruning_ebc_dict: Dict[str, int] = {"table_0": num_embedding - pruned_entry}

        mi = create_test_model(
            num_embedding,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=len(table_specs),
            pruning_dict=pruning_ebc_dict,
        )

        expected_shards = [
            [
                (
                    (0, 0, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:0/cuda:0",
                ),
                (
                    (0, 128, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:1/cuda:1",
                ),
                (
                    (0, 256, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:0/cuda:0",
                ),
                (
                    (0, 384, table_specs[0][2], table_specs[0][1] // 4),
                    "rank:1/cuda:1",
                ),
            ],
        ]

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(mi.quant_model.state_dict())
        quant_output = mi.quant_model(*inputs[0])
        sharded_output = sharded_model(*inputs[0])
        assert_close(quant_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        buffer = io.BytesIO()
        torch.jit.save(gm_script, buffer)
        buffer.seek(0)
        loaded_gm_script = torch.jit.load(buffer)
        gm_script_output = loaded_gm_script(*inputs[0])
        assert_close(quant_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0"],
            ShardingType.COLUMN_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_qebc_pruned_cw_one_ebc(self) -> None:
        batch_size: int = 1
        world_size: int = 2
        local_device = torch.device("cuda:0")
        num_embedding = 200
        emb_dim = 512
        sharding_type = ShardingType.COLUMN_WISE

        quant_model, sharded_model, pruned_dict = create_quant_and_sharded_ebc_models(
            num_embedding=num_embedding,
            emb_dim=emb_dim,
            world_size=world_size,
            batch_size=batch_size,
            sharding_type=sharding_type,
            device=local_device,
        )

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.tensor([0, 1, 2, 59, 60, 99], dtype=torch.int32).cuda(),
            lengths=torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int32).cuda(),
            weights=None,
        )

        q_output = quant_model(kjt)
        s_output = sharded_model(kjt)

        assert_close(q_output["feature_0"], s_output["feature_0"])

        self.check_tbe_pruned(sharded_model, pruned_dict)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_fpqebc_pruned_tw_one_fpebc(self) -> None:
        batch_size: int = 1
        world_size: int = 2
        local_device = torch.device("cuda:0")
        num_embedding = 200
        emb_dim = 512
        sharding_type = ShardingType.COLUMN_WISE

        quant_model, sharded_model, pruned_dict = create_quant_and_sharded_ebc_models(
            num_embedding=num_embedding,
            emb_dim=emb_dim,
            world_size=world_size,
            batch_size=batch_size,
            sharding_type=sharding_type,
            device=local_device,
            feature_processor=True,
        )

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.tensor([0, 1, 2, 59, 60, 99], dtype=torch.int32).cuda(),
            lengths=torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int32).cuda(),
            weights=None,
        )

        q_output = quant_model(kjt)
        s_output = sharded_model(kjt)

        assert_close(q_output["feature_0"], s_output["feature_0"])

        self.check_tbe_pruned(sharded_model, pruned_dict)
