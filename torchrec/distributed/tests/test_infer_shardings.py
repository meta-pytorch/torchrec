#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import math
import unittest
from typing import Dict, List, Tuple

import hypothesis.strategies as st

import torch
from hypothesis import given, settings
from torchrec import (
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.infer_utils import (
    get_path_device_tuples,
    get_tbes_from_sharded_module,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollectionSharder,
    QuantFeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.quant_state import sharded_tbes_weights_spec, WeightSpec
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    placement,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    assert_weight_spec,
    create_cw_min_partition_constraints,
    create_test_model,
    KJTInputWrapper,
    model_input_to_forward_args,
    model_input_to_forward_args_kjt,
    prep_inputs,
    quantize,
    quantize_fpebc,
    shard_qebc,
    shard_qec,
    TestModelInfo,
    TestQuantEBCSharder,
    TestQuantECSharder,
)
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import (
    dtype_to_data_type,
    TimestampBasedEvictionPolicy,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    FeatureProcessorsCollection,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)

torch.fx.wrap("len")


class TimeGapPoolingCollectionModule(FeatureProcessorsCollection):
    def __init__(
        self,
        feature_pow: float,
        feature_min: float,
        feature_max: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.feature_pow = feature_pow
        self.device = device

        param = torch.empty(
            [math.ceil(math.pow(feature_max, feature_pow)) + 2],
            device=device,
        )
        self.register_buffer("w", param)

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        scores_list = []
        for feature_name in features.keys():
            jt = features[feature_name]
            scores = jt.weights()
            scores = torch.clamp(
                scores,
                min=self.feature_min,
                max=self.feature_max,
            )
            indices = torch.floor(torch.pow(scores, self.feature_pow))
            indices = indices.to(torch.int32)
            scores = torch.index_select(self.w, 0, indices)
            scores_list.append(scores)

        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            weights=(
                torch.cat(scores_list) if scores_list else features.weights_or_none()
            ),
            lengths=features.lengths(),
            stride=features.stride(),
        )


def placement_helper(device_type: str, index: int = 0) -> str:
    if device_type == "cpu":
        return f"rank:0/{device_type}"  # cpu only use rank 0

    return f"rank:{index}/{device_type}:{index}"


class InferShardingsTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        set_propogate_device(True)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_tw(self, weight_dtype: torch.dtype, device_type: str) -> None:
        num_embeddings = 256
        emb_dim = 16
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim),
                    placement(device_type, 0, world_size),
                ),
            ]
        ]
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

        sharded_model.load_state_dict(non_sharded_model.state_dict())

        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0"],
            ShardingType.TABLE_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8]),
        device_type=st.sampled_from(["cuda"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_tw_ebc_full_rank_weighted_ebc_with_empty_rank(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 256
        emb_dim = 16
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            num_features=6,  # 6 sparse features on ebc
            num_weighted_features=1,  # only 1 weighted sparse feature on weighted_ebc
        )

        non_sharded_model = mi.quant_model

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.TABLE_WISE,
            device=local_device,
            expected_shards=None,
            shard_score_ebc=True,
        )

        self.assertEqual(
            len(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `sparse`.
                sharded_model._module.sparse.ebc._lookups[0]._embedding_lookups_per_rank
            ),
            2,
        )
        self.assertEqual(
            len(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `sparse`.
                sharded_model._module.sparse.weighted_ebc._lookups[
                    0
                ]._embedding_lookups_per_rank
            ),
            1,
        )

        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())

        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw(self, weight_dtype: torch.dtype, device_type: str) -> None:
        num_embeddings = 256
        emb_dim = 16
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            [
                ((0, 0, num_emb_half, emb_dim), placement(device_type, 0, world_size)),
                (
                    (num_emb_half, 0, num_emb_half, emb_dim),
                    placement(device_type, 1, world_size),
                ),
            ]
        ]
        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())

        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0"],
            ShardingType.ROW_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_with_virtual_table_eviction(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 256
        emb_dim = 16
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        eviction_policy = TimestampBasedEvictionPolicy()
        eviction_policy.init_metaheader_config(
            dtype_to_data_type(torch.float16), emb_dim
        )
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            virtual_table_eviction_policy=eviction_policy,
        )
        for t in mi.tables:
            self.assertIsNotNone(t.virtual_table_eviction_policy)
            self.assertEqual(
                # pyre-ignore [16]
                t.virtual_table_eviction_policy.get_embedding_dim(),
                emb_dim,
            )

        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            [
                ((0, 0, num_emb_half, emb_dim), placement(device_type, 0, world_size)),
                (
                    (num_emb_half, 0, num_emb_half, emb_dim),
                    placement(device_type, 1, world_size),
                ),
            ]
        ]
        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())

        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0"],
            ShardingType.ROW_WISE.value,
        )

        self.assertIsNotNone(
            weights_spec[
                "_module.sparse.ebc.tbes.0.0.table_0.weight"
            ].virtual_table_dim_offsets
        )
        self.assertEqual(
            # pyre-ignore [16]
            weights_spec[
                "_module.sparse.ebc.tbes.0.0.table_0.weight"
            ].virtual_table_dim_offsets[0],
            8,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        test_permute=st.booleans(),
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw(
        self, test_permute: bool, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 64
        emb_dim = 512
        emb_dim_4 = emb_dim // 4
        local_size = 2
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_ranks: List[int] = [0, 1, 0, 1] if not test_permute else [1, 0, 1, 0]
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_4),
                    placement(device_type, expected_ranks[0], world_size),
                ),
                (
                    (0, 1 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, expected_ranks[1], world_size),
                ),
                (
                    (0, 2 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, expected_ranks[2], world_size),
                ),
                (
                    (0, 3 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, expected_ranks[3], world_size),
                ),
            ]
        ]

        plan = None
        if test_permute:
            sharder = TestQuantEBCSharder(
                sharding_type=ShardingType.COLUMN_WISE.value,
                kernel_type=EmbeddingComputeKernel.QUANT.value,
                shardable_params=[table.name for table in mi.tables],
            )

            module_plan = construct_module_sharding_plan(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `sparse`.
                non_sharded_model._module.sparse.ebc,
                per_param_sharding={
                    "table_0": column_wise(ranks=[1, 0, 1, 0]),
                },
                # pyre-ignore
                sharder=sharder,
                local_size=local_size,
                world_size=world_size,
                device_type=device_type,
            )

            plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())
        # torchrec.distributed.test_utils.test_sharding.copy_state_dict(sharded_model.state_dict(), non_sharded_model.state_dict()) does not work for CW due to non-trivial qscaleshift copy which is handled in shardedQEBC load_state_dict

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

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
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        weight_dtype=st.sampled_from([torch.qint8]),
        device_type=st.sampled_from(["cuda"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_uneven_cw(self, weight_dtype: torch.dtype, device_type: str) -> None:
        num_embeddings = 64
        emb_dim = 512
        dim_1 = 0
        dim_2 = 128
        dim_3 = 128
        dim_4 = 256
        local_size = 4
        world_size = 4
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        expected_ranks: List[int] = [0, 1, 2, 3]
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, dim_1),
                    placement(device_type, expected_ranks[0], world_size),
                ),
                (
                    (0, dim_1, num_embeddings, dim_2),
                    placement(device_type, expected_ranks[1], world_size),
                ),
                (
                    (0, dim_1 + dim_2, num_embeddings, dim_3),
                    placement(device_type, expected_ranks[2], world_size),
                ),
                (
                    (0, dim_1 + dim_2 + dim_3, num_embeddings, dim_4),
                    placement(device_type, expected_ranks[3], world_size),
                ),
            ]
        ]
        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.COLUMN_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )
        module_plan = construct_module_sharding_plan(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
            #  attribute `sparse`.
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": column_wise(size_per_rank=[dim_1, dim_2, dim_3, dim_4]),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
            device_type=device_type,
        )
        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )

        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        torch.testing.assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        torch.testing.assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)

        for table_name, expected_shard in zip(["table_0"], expected_shards):
            unsharded_weight_fqn = (
                f"_module.sparse.ebc.embedding_bags.{table_name}.weight"
            )
            for (offset_r, offset_c, size_r, size_c), rank in expected_shard:
                tbe_idx = int(rank.split(":")[-1])
                sharded_weight_fqn: str = (
                    f"_module.sparse.ebc.tbes.{tbe_idx}.0.{table_name}.weight"
                )

                assert sharded_weight_fqn in weights_spec
                wspec = weights_spec[sharded_weight_fqn]
                assert wspec.fqn == unsharded_weight_fqn
                assert wspec.shard_sizes == [size_r, size_c]
                assert wspec.shard_offsets == [offset_r, offset_c]
                assert wspec.sharding_type == ShardingType.COLUMN_WISE.value

                for qcomp in ["qscale", "qbias"]:
                    sharded_weight_qcomp_fqn: str = f"{sharded_weight_fqn}_{qcomp}"
                    assert sharded_weight_qcomp_fqn in weights_spec
                    wqcomp_spec = weights_spec[sharded_weight_qcomp_fqn]
                    assert wqcomp_spec.fqn == f"{unsharded_weight_fqn}_{qcomp}"
                    assert wqcomp_spec.shard_sizes == [size_r, 2]
                    assert wqcomp_spec.shard_offsets == [0, 0]
                    assert wqcomp_spec.sharding_type == ShardingType.COLUMN_WISE.value

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        emb_dim=st.sampled_from([192, 128]),
        test_permute=st.booleans(),
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_with_smaller_emb_dim(
        self,
        emb_dim: int,
        test_permute: bool,
        weight_dtype: torch.dtype,
        device_type: str,
    ) -> None:
        num_embeddings = 64
        emb_dim_4 = emb_dim // 4
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        constraints = create_cw_min_partition_constraints([("table_0", emb_dim_4)])
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            constraints=constraints,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_4),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 1 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, 1, world_size),
                ),
                (
                    (0, 2 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 3 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, 1, world_size),
                ),
            ]
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
        sharded_model.load_state_dict(non_sharded_model.state_dict())
        # torchrec.distributed.test_utils.test_sharding.copy_state_dict(sharded_model.state_dict(), non_sharded_model.state_dict()) does not work for CW due to non-trivial qscaleshift copy which is handled in shardedQEBC load_state_dict

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

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
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_multiple_tables_with_permute(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 64
        emb_dim = 512
        emb_dim_2 = 512 // 2
        local_size = 2
        world_size = 2
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=2,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_2),
                    placement(device_type, 1, world_size),
                ),
                (
                    (0, 1 * emb_dim_2, num_embeddings, emb_dim_2),
                    placement(device_type, 0, world_size),
                ),
            ],
            [
                (
                    (0, 0, num_embeddings, emb_dim_2),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 1 * emb_dim_2, num_embeddings, emb_dim_2),
                    placement(device_type, 1, world_size),
                ),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.COLUMN_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": column_wise(ranks=[1, 0]),
                "table_1": column_wise(ranks=[0, 1]),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
            device_type=device_type,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())
        # torchrec.distributed.test_utils.test_sharding.copy_state_dict(sharded_model.state_dict(), non_sharded_model.state_dict()) does not work for CW due to non-trivial qscaleshift copy which is handled in shardedQEBC load_state_dict

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module.sparse.ebc",
            "embedding_bags",
            ["table_0", "table_1"],
            ShardingType.COLUMN_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_irregular_shard_placement(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 64
        emb_dim = 384
        emb_dim_2 = emb_dim // 2
        emb_dim_3 = emb_dim // 3
        local_size = 4
        world_size = 4
        batch_size = 4
        local_device = torch.device(f"{device_type}:0")
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            num_features=3,
            weight_dtype=weight_dtype,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_2),
                    placement(device_type, 2, world_size),
                ),
                (
                    (0, 1 * emb_dim_2, num_embeddings, emb_dim_2),
                    placement(device_type, 1, world_size),
                ),
            ],
            [
                (
                    (0, 0, num_embeddings, emb_dim_2),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 1 * emb_dim_2, num_embeddings, emb_dim_2),
                    placement(device_type, 3, world_size),
                ),
            ],
            [
                (
                    (0, 0, num_embeddings, emb_dim_3),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 1 * emb_dim_3, num_embeddings, emb_dim_3),
                    placement(device_type, 2, world_size),
                ),
                (
                    (0, 2 * emb_dim_3, num_embeddings, emb_dim_3),
                    placement(device_type, 3, world_size),
                ),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.COLUMN_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": column_wise(ranks=[2, 1]),
                "table_1": column_wise(ranks=[0, 3]),
                "table_2": column_wise(ranks=[0, 2, 3]),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
            device_type=device_type,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())
        # torchrec.distributed.test_utils.test_sharding.copy_state_dict(sharded_model.state_dict(), non_sharded_model.state_dict()) does not work for CW due to non-trivial qscaleshift copy which is handled in shardedQEBC load_state_dict

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        device_type_weight_dtype=st.sampled_from(
            [
                ("cuda", torch.qint8),
                ("cuda", torch.quint4x2),
                # ("cpu", torch.qint8), column sharding is currently not supported in CPU inference sharding
                # ("cpu", torch.quint4x2),
            ]
        ),
    )
    @settings(max_examples=4, deadline=None)
    def test_cw_sequence(
        self, device_type_weight_dtype: Tuple[str, torch.dtype]
    ) -> None:
        device_type, weight_dtype = device_type_weight_dtype
        num_embeddings = 4
        emb_dim = 512
        emb_dim_4 = emb_dim // 4
        world_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim_4),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 1 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, 1, world_size),
                ),
                (
                    (0, 2 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, 0, world_size),
                ),
                (
                    (0, 3 * emb_dim_4, num_embeddings, emb_dim_4),
                    placement(device_type, 1, world_size),
                ),
            ],
        ] * 2
        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.COLUMN_WISE,  # column wise sharding the model
            device=local_device,
            expected_shards=expected_shards,
        )
        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())

        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(sharded_output, non_sharded_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module_kjt_input.0",
            "embeddings",
            ["table_0", "table_1"],
            ShardingType.COLUMN_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_tw_sequence(self, weight_dtype: torch.dtype, device_type: str) -> None:
        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, num_embeddings, emb_dim),
                    placement(device_type, 0, world_size),
                ),
            ],
            [
                (
                    (0, 0, num_embeddings, emb_dim),
                    placement(device_type, 1, world_size),
                ),
            ],
        ]
        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.TABLE_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module_kjt_input.0",
            "embeddings",
            ["table_0", "table_1"],
            ShardingType.TABLE_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        device_type_weight_dtype=st.sampled_from(
            [
                ("cuda", torch.qint8),
                ("cuda", torch.quint4x2),
                ("cpu", torch.qint8),
                ("cpu", torch.quint4x2),
            ]
        ),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_sequence(
        self, device_type_weight_dtype: Tuple[str, torch.dtype]
    ) -> None:
        device_type, weight_dtype = device_type_weight_dtype
        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        num_emb_half = num_embeddings // 2
        expected_shards = [
            [
                ((0, 0, num_emb_half, emb_dim), placement(device_type, 0, world_size)),
                (
                    (num_emb_half, 0, num_emb_half, emb_dim),
                    placement(device_type, 1, world_size),
                ),
            ],
        ] * 2
        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        weights_spec: Dict[str, WeightSpec] = sharded_tbes_weights_spec(sharded_model)
        assert_weight_spec(
            weights_spec,
            expected_shards,
            "_module_kjt_input.0",
            "embeddings",
            ["table_0", "table_1"],
            ShardingType.ROW_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_sequence_uneven(self, weight_dtype: torch.dtype, device: str) -> None:
        num_embeddings = 512
        emb_dim = 64
        world_size = 4
        local_size = 4
        batch_size = 4
        local_device = torch.device("cuda:0" if device == "cuda" else device)

        topology: Topology = Topology(world_size=world_size, compute_device=device)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=4,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, 256, 64),
                    placement_helper(device, 0),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (384, 0, 64, 64),
                    placement_helper(device, 2),
                ),
                (
                    (448, 0, 64, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 128, 64),
                    placement_helper(device, 0),
                ),
                (
                    (128, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (384, 0, 128, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 256, 64),
                    placement_helper(device, 0),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (384, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (512, 0, 0, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 0, 64),
                    placement_helper(device, 0),
                ),
                (
                    (0, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (128, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (256, 0, 256, 64),
                    placement_helper(device, 3),
                ),
            ],
        ]
        sharder = TestQuantECSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            non_sharded_model._module_kjt_input[0],
            per_param_sharding={
                "table_0": row_wise(
                    ([256, 128, 64, 64], device),
                ),
                "table_1": row_wise(([128, 128, 128, 128], device)),
                "table_2": row_wise(([256, 128, 128, 0], device)),
                "table_3": row_wise(([0, 128, 128, 256], device)),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module_kjt_input.0": module_plan})

        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])

        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        tbes = get_tbes_from_sharded_module(sharded_model._module_kjt_input[0])
        for tbe in tbes:
            self.assertTrue(tbe.weight_initialized)

        path_device_lists = get_path_device_tuples(sharded_model)

        for path_device in path_device_lists:
            assert device in path_device[1]

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_mix_tw_rw_sequence(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        local_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=3,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model

        sharder = QuantEmbeddingCollectionSharder()

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            non_sharded_model._module_kjt_input[0],
            per_param_sharding={
                "table_0": row_wise(),
                "table_1": table_wise(rank=0),
                "table_2": table_wise(rank=1),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module_kjt_input.0": module_plan})

        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            plan=plan,
            expected_shards=None,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_mix_tw_rw_sequence_missing_feature_on_rank(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        local_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )

        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                EmbeddingCollection(
                    tables=mi.tables,
                    device=mi.sparse_device,
                )
            )
        )

        mi.model.training = False
        mi.quant_model = quantize(
            mi.model,
            inplace=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        non_sharded_model = mi.quant_model

        sharder = QuantEmbeddingCollectionSharder()

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedS...
            non_sharded_model._module_kjt_input[0],
            per_param_sharding={
                "table_0": row_wise(),
                "table_1": table_wise(rank=1),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module_kjt_input.0": module_plan})

        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            plan=plan,
            expected_shards=None,
        )

        inputs = [
            model_input_to_forward_args_kjt(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]

        sharded_model.load_state_dict(non_sharded_model.state_dict())
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 2,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        uneven_shard_pattern=st.sampled_from(
            [
                (512, 256, 128, 128),
                (500, 256, 128, 128),
            ]
        ),
        device=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_uneven_sharding(
        self,
        weight_dtype: torch.dtype,
        uneven_shard_pattern: Tuple[int, int, int, int],
        device: str,
    ) -> None:
        num_embeddings, size0, size1, size2 = uneven_shard_pattern
        size2 = min(size2, num_embeddings - size0 - size1)
        emb_dim = 64
        local_size = 3
        world_size = 3
        batch_size = 4
        local_device = torch.device("cuda:0" if device == "cuda" else device)
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            num_features=1,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, size0, 64),
                    placement_helper(device, 0),
                ),
                (
                    (size0, 0, size1, 64),
                    placement_helper(device, 1),
                ),
                (
                    (size0 + size1, 0, size2, 64),
                    placement_helper(device, 2),
                ),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": row_wise(([size0, size1, size2], device)),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_rw_uneven_sharding_mutiple_table(
        self,
        weight_dtype: torch.dtype,
        device: str,
    ) -> None:
        num_embeddings = 512
        emb_dim = 64
        local_size = 4
        world_size = 4
        batch_size = 1
        local_device = torch.device("cuda:0" if device == "cuda" else device)
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            num_features=4,
        )

        non_sharded_model = mi.quant_model
        expected_shards = [
            [
                (
                    (0, 0, 256, 64),
                    placement_helper(device, 0),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (384, 0, 64, 64),
                    placement_helper(device, 2),
                ),
                (
                    (448, 0, 64, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 128, 64),
                    placement_helper(device, 0),
                ),
                (
                    (128, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (384, 0, 128, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 256, 64),
                    placement_helper(device, 0),
                ),
                (
                    (256, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (384, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (512, 0, 0, 64),
                    placement_helper(device, 3),
                ),
            ],
            [
                (
                    (0, 0, 0, 64),
                    placement_helper(device, 0),
                ),
                (
                    (0, 0, 128, 64),
                    placement_helper(device, 1),
                ),
                (
                    (128, 0, 128, 64),
                    placement_helper(device, 2),
                ),
                (
                    (256, 0, 256, 64),
                    placement_helper(device, 3),
                ),
            ],
        ]

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
        )

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": row_wise(
                    ([256, 128, 64, 64], device),
                ),
                "table_1": row_wise(([128, 128, 128, 128], device)),
                "table_2": row_wise(([256, 128, 128, 0], device)),
                "table_3": row_wise(([0, 128, 128, 256], device)),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs available",
    )
    # pyre-fixme[56]Pyre was not able to infer the type of argument `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        weight_dtype=st.sampled_from([torch.qint8, torch.quint4x2]),
        device=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=4, deadline=None)
    def test_mix_sharding_mutiple_table(
        self,
        weight_dtype: torch.dtype,
        device: str,
    ) -> None:
        num_embeddings = 512
        emb_dim = 64
        local_size = 4
        world_size = 4
        batch_size = 1
        local_device = torch.device("cuda:0" if device == "cuda" else device)
        mi = create_test_model(
            num_embeddings,
            emb_dim,
            world_size,
            batch_size,
            dense_device=local_device,
            sparse_device=local_device,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
            num_features=4,
        )

        non_sharded_model = mi.quant_model

        sharder = QuantEmbeddingBagCollectionSharder()

        module_plan = construct_module_sharding_plan(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `sparse`.
            non_sharded_model._module.sparse.ebc,
            per_param_sharding={
                "table_0": row_wise(
                    ([256, 128, 64, 64], device),
                ),
                "table_1": row_wise(([128, 128, 128, 128], device)),
                "table_2": column_wise(ranks=[0, 1]),
                "table_3": table_wise(rank=0),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=local_size,
            world_size=world_size,
        )

        plan = ShardingPlan(plan={"_module.sparse.ebc": module_plan})

        sharded_model = shard_qebc(
            mi=mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=None,  # expected_shards,
            plan=plan,
        )
        inputs = [
            model_input_to_forward_args(inp.to(local_device))
            for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
        ]
        sharded_model.load_state_dict(non_sharded_model.state_dict())

        # We need this first inference to make all lazy init in forward
        sharded_output = sharded_model(*inputs[0])
        non_sharded_output = non_sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(sharded_model)
        gm_script = torch.jit.script(gm)
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8]),
        device_type=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=1, deadline=None)
    def test_sharded_quant_fp_ebc_tw(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")
        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )
        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                FeatureProcessedEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        tables=mi.tables,
                        is_weighted=True,
                        device=mi.sparse_device,
                    ),
                    TimeGapPoolingCollectionModule(
                        feature_pow=1.0,
                        feature_min=-1.0,
                        feature_max=1.0,
                        device=mi.sparse_device,
                    ),
                )
            )
        )
        model_inputs: List[ModelInput] = prep_inputs(
            mi, world_size, batch_size, long_indices=False
        )
        inputs = []
        for model_input in model_inputs:
            kjt = model_input.idlist_features
            assert isinstance(kjt, KeyedJaggedTensor)
            kjt = kjt.to(local_device)
            weights = torch.rand(
                kjt._values.size(0), dtype=torch.float, device=local_device
            )
            inputs.append(
                (
                    kjt._keys,
                    kjt._values,
                    weights,
                    kjt._lengths,
                    kjt._offsets,
                )
            )

        mi.model(*inputs[0])
        print(f"model:\n{mi.model}")

        mi.quant_model = quantize_fpebc(
            module=mi.model,
            inplace=False,
            register_tbes=True,
            quant_state_dict_split_scale_bias=False,
            weight_dtype=weight_dtype,
        )
        quant_model = mi.quant_model
        print(f"quant_model:\n{quant_model}")
        non_sharded_output = mi.quant_model(*inputs[0])

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )
        sharder = QuantFeatureProcessedEmbeddingBagCollectionSharder()
        # pyre-ignore
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

        sharded_model = _shard_modules(
            module=quant_model,
            # pyre-fixme[6]: For 2nd argument expected
            #  `Optional[List[ModuleSharder[Module]]]` but got
            #  `List[QuantFeatureProcessedEmbeddingBagCollectionSharder]`.
            sharders=[sharder],
            device=local_device,
            plan=plan,
            # pyre-ignore
            env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
        )
        print(f"sharded_model:\n{sharded_model}")
        for n, m in sharded_model.named_modules():
            print(f"sharded_model.MODULE[{n}]:{type(m)}")

        # Check that FP is registered as module
        count_registered_fp: int = 0
        for _, m in sharded_model.named_modules():
            if isinstance(m, TimeGapPoolingCollectionModule):
                count_registered_fp += 1

        assert count_registered_fp == world_size

        sharded_output = sharded_model(*inputs[0])
        assert_close(non_sharded_output, sharded_output)

        gm: torch.fx.GraphModule = symbolic_trace(
            sharded_model,
            leaf_modules=[
                "TimeGapPoolingCollectionModule",
                "IntNBitTableBatchedEmbeddingBagsCodegen",
            ],
        )

        # Check that FP was traced as a call_module
        fp_call_module: int = 0
        for node in gm.graph.nodes:
            if node.op == "call_module":
                m = gm
                for attr in node.target.split("."):
                    m = getattr(m, attr)
                if isinstance(m, TimeGapPoolingCollectionModule):
                    fp_call_module += 1

        assert fp_call_module == world_size
        print(f"fx.graph:\n{gm.graph}")

        gm_script = torch.jit.script(gm)
        print(f"gm_script:\n{gm_script}")
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output, gm_script_output)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        weight_dtype=st.sampled_from([torch.qint8]),
        device_type=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(max_examples=2, deadline=None)
    def test_sharded_quant_mc_ec_rw(
        self, weight_dtype: torch.dtype, device_type: str
    ) -> None:
        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        batch_size = 2
        local_device = torch.device(f"{device_type}:0")

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=1,
            num_float_features=10,
            num_weighted_features=0,
            topology=topology,
        )
        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        mi.tables = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                ManagedCollisionEmbeddingCollection(
                    EmbeddingCollection(
                        tables=mi.tables,
                        device=mi.sparse_device,
                    ),
                    ManagedCollisionCollection(
                        managed_collision_modules={
                            "table_0": MCHManagedCollisionModule(
                                zch_size=num_embeddings,
                                input_hash_size=4000,
                                device=mi.sparse_device,
                                eviction_interval=2,
                                eviction_policy=DistanceLFU_EvictionPolicy(),
                            )
                        },
                        embedding_configs=mi.tables,
                    ),
                )
            )
        )
        model_inputs: List[ModelInput] = prep_inputs(
            mi, world_size, batch_size, long_indices=True
        )
        inputs = []
        for model_input in model_inputs:
            kjt = model_input.idlist_features
            assert isinstance(kjt, KeyedJaggedTensor)
            kjt = kjt.to(local_device)
            weights = None
            inputs.append(
                (
                    kjt._keys,
                    kjt._values,
                    weights,
                    kjt._lengths,
                    kjt._offsets,
                )
            )

        mi.model(*inputs[0])
        print(f"model:\n{mi.model}")
        assert mi.model.training is True
        mi.quant_model = quantize(
            module=mi.model,
            inplace=False,
            register_tbes=False,
            quant_state_dict_split_scale_bias=True,
            weight_dtype=weight_dtype,
        )
        quant_model = mi.quant_model
        assert quant_model.training is False
        non_sharded_output = mi.quant_model(*inputs[0])

        topology: Topology = Topology(world_size=world_size, compute_device=device_type)
        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )
        sharder = QuantEmbeddingCollectionSharder()
        # pyre-ignore
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

        sharded_model = shard_qec(
            mi,
            sharding_type=ShardingType.ROW_WISE,
            device=local_device,
            expected_shards=None,
            plan=plan,
        )

        print(f"sharded_model:\n{sharded_model}")
        for n, m in sharded_model.named_modules():
            print(f"sharded_model.MODULE[{n}]:{type(m)}")

        sharded_model.load_state_dict(quant_model.state_dict())
        sharded_output = sharded_model(*inputs[0])

        assert_close(non_sharded_output[0], sharded_output[0])
        gm: torch.fx.GraphModule = symbolic_trace(
            sharded_model,
            leaf_modules=[
                "IntNBitTableBatchedEmbeddingBagsCodegen",
                "ComputeJTDictToKJT",
            ],
        )

        print(f"fx.graph:\n{gm.graph}")
        gm_script = torch.jit.script(gm)
        print(f"gm_script:\n{gm_script}")
        gm_script_output = gm_script(*inputs[0])
        assert_close(sharded_output[0], gm_script_output[0])

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    # pyre-ignore
    @given(
        compute_device=st.sampled_from(["cuda", "cpu"]),
    )
    @settings(max_examples=2, deadline=None)
    def test_sharded_quant_fp_ebc_tw_meta(self, compute_device: str) -> None:
        # Simulate inference, take unsharded cpu model and shard on meta
        # Use PositionWeightedModuleCollection, FP used in production

        num_embeddings = 10
        emb_dim = 16
        world_size = 2
        batch_size = 2
        local_device = torch.device(compute_device)

        mi = TestModelInfo(
            dense_device=local_device,
            sparse_device=local_device,
            num_features=2,
            num_float_features=10,
            num_weighted_features=0,
        )

        mi.tables = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=emb_dim,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(mi.num_features)
        ]

        max_feature_lengths = {"feature_0": 20}

        mi.model = KJTInputWrapper(
            module_kjt_input=torch.nn.Sequential(
                FeatureProcessedEmbeddingBagCollection(
                    EmbeddingBagCollection(
                        tables=mi.tables,
                        is_weighted=True,
                        device=mi.sparse_device,
                    ),
                    PositionWeightedModuleCollection(
                        max_feature_lengths=max_feature_lengths,
                        device=mi.sparse_device,
                    ),
                )
            )
        )
        model_inputs: List[ModelInput] = prep_inputs(
            mi, world_size, batch_size, long_indices=False, count=1
        )
        inputs = []
        kjt = model_inputs[0].idlist_features
        assert isinstance(kjt, KeyedJaggedTensor)
        kjt = kjt.to(local_device)
        weights = torch.rand(
            kjt._values.size(0), dtype=torch.float, device=local_device
        )

        inputs = [
            kjt._keys,
            kjt._values,
            weights,
            kjt._lengths,
            kjt._offsets,
        ]

        mi.model(*inputs)
        print(f"model:\n{mi.model}")

        mi.quant_model = quantize_fpebc(
            module=mi.model,
            inplace=False,
            register_tbes=True,
            quant_state_dict_split_scale_bias=False,
            weight_dtype=torch.int8,
        )
        quant_model = mi.quant_model
        print(f"quant_model:\n{quant_model}")

        topology: Topology = Topology(
            world_size=world_size, compute_device=compute_device
        )
        mi.planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology, is_inference=True),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )
        sharder = QuantFeatureProcessedEmbeddingBagCollectionSharder()
        # pyre-ignore
        plan = mi.planner.plan(
            mi.quant_model,
            [sharder],
        )

        sharded_model = _shard_modules(
            module=quant_model,
            # pyre-fixme[6]: For 2nd argument expected
            #  `Optional[List[ModuleSharder[Module]]]` but got
            #  `List[QuantFeatureProcessedEmbeddingBagCollectionSharder]`.
            sharders=[sharder],
            # shard on meta to simulate device movement from cpu -> meta QFPEBC
            device=torch.device("meta"),
            plan=plan,
            env=ShardingEnv.from_local(world_size=topology.world_size, rank=0),
        )
        print(f"sharded_model:\n{sharded_model}")
        for n, m in sharded_model.named_modules():
            print(f"sharded_model.MODULE[{n}]:{type(m)}")

        # Check that FP is registered as module
        count_registered_fp: int = 0
        for _, m in sharded_model.named_modules():
            if isinstance(m, PositionWeightedModuleCollection):
                count_registered_fp += 1

        assert count_registered_fp == world_size

        # Move inputs to meta now that we shard on meta
        for i, input in enumerate(inputs):
            if isinstance(input, torch.Tensor):
                inputs[i] = input.to(torch.device("meta"))

        # move dense params also to meta
        sharded_model.to("meta")
        sharded_model(*inputs)
        # Don't care about the output since we are sharding on meta

        gm: torch.fx.GraphModule = symbolic_trace(
            sharded_model,
            leaf_modules=[
                "PositionWeightedModuleCollection",
                "IntNBitTableBatchedEmbeddingBagsCodegen",
            ],
        )

        # Check that FP was traced as a call_module
        fp_call_module: int = 0
        for node in gm.graph.nodes:
            if node.op == "call_module":
                m = gm
                for attr in node.target.split("."):
                    m = getattr(m, attr)
                if isinstance(m, PositionWeightedModuleCollection):
                    fp_call_module += 1

        assert fp_call_module == world_size
        print(f"fx.graph:\n{gm.graph}")

        gm_script = torch.jit.script(gm)
        print(f"gm_script:\n{gm_script}")
        gm_script(*inputs)
