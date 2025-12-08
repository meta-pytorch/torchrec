#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, OrderedDict, Tuple
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.tbe.ssd.utils.partially_materialized_tensor import (
    PartiallyMaterializedTensor,
)
from hypothesis import assume, given, settings, strategies as st, Verbosity
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torchrec import distributed as trec_dist
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    EmbeddingTableConfig,
)
from torchrec.distributed.embeddingbag import (
    logger as embeddingbag_logger,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.fused_embeddingbag import ShardedFusedEmbeddingBagCollection
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.test_utils.test_model import (
    _get_default_rtol_and_atol,
    ModelInput,
    TestSparseNN,
    TestSparseNNBase,
)
from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    create_test_sharder,
    gen_model_and_input,
    ModelInputCallable,
    SharderType,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    DataType,
    EmbeddingBagConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port, seed_and_log


class InferenceModelParallelTestBase(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def tearDown(self) -> None:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
        super().tearDown()

    def _test_sharded_forward(
        self,
        world_size: int,
        model_class: TestSparseNNBase,
        embedding_groups: Dict[str, List[str]],
        tables: List[EmbeddingTableConfig],
        sharders: List[ModuleSharder[nn.Module]],
        quantize_callable: Callable[[nn.Module], nn.Module],
        quantize_callable_kwargs: Dict[str, Any],
        dedup_features_names: Optional[List[str]] = None,
        dedup_tables: Optional[List[EmbeddingTableConfig]] = None,
        weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        # pyre-ignore [9]
        generate: ModelInputCallable = ModelInput.generate,
    ) -> None:
        default_rank = 0
        cuda_device = torch.device(f"cuda:{default_rank}")
        torch.cuda.set_device(cuda_device)

        # Generate model & inputs.
        (global_model, _inputs) = gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            dedup_tables=dedup_tables,
            dedup_feature_names=dedup_features_names,
            embedding_groups=embedding_groups,
            world_size=1,  # generate only one copy of feature for inference
            num_float_features=16,
            dense_device=cuda_device,
            sparse_device=cuda_device,
            generate=generate,
            indices_dtype=torch.int32,
            lengths_dtype=torch.int32,
        )
        global_model = quantize_callable(global_model, **quantize_callable_kwargs)
        local_input = _inputs[0][1][default_rank].to(cuda_device)

        # Shard model.
        if dedup_features_names:
            local_model = model_class(
                tables=cast(
                    List[BaseEmbeddingConfig],
                    tables + dedup_tables if dedup_tables else tables,
                ),
                weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
                dedup_feature_names=dedup_features_names,
                embedding_groups=embedding_groups,
                dense_device=cuda_device,
                sparse_device=torch.device("meta"),
                num_float_features=16,
            )
        else:
            local_model = model_class(
                tables=cast(
                    List[BaseEmbeddingConfig],
                    tables,
                ),
                weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
                embedding_groups=embedding_groups,
                dense_device=cuda_device,
                sparse_device=torch.device("meta"),
                num_float_features=16,
            )
        local_model = quantize_callable(local_model, **quantize_callable_kwargs)

        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size, "cuda"),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.plan(local_model, sharders)

        # Generate a sharded model on a default rank.
        local_model = DistributedModelParallel(
            module=local_model,
            env=ShardingEnv.from_local(world_size, default_rank),
            device=cuda_device,
            plan=plan,
            sharders=sharders,
            init_data_parallel=False,
        )

        # materialize inference sharded model on one device for dense part
        local_model = local_model.copy(cuda_device)

        copy_state_dict(local_model.state_dict(), global_model.state_dict())

        # Run a single training step of the sharded model.
        with torch.inference_mode():
            shard_pred = local_model(local_input)

        # Run second training step of the unsharded model.
        with torch.inference_mode():
            global_pred = global_model(local_input)

        # Compare predictions of sharded vs unsharded models.
        rtol, atol = _get_default_rtol_and_atol(global_pred, shard_pred)
        torch.testing.assert_close(global_pred, shard_pred, rtol=rtol, atol=atol)


class ModelParallelSparseOnlyBase(unittest.TestCase):
    def setUp(self, backend: str = "nccl") -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        self.backend = backend
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if self.backend == "nccl" and self.device == torch.device("cpu"):
            self.skipTest("NCCL not supported on CPUs.")

        dist.init_process_group(backend=self.backend)

    def tearDown(self) -> None:
        dist.destroy_process_group()

    def test_sharding_ebc_as_top_level(self) -> None:
        model = self._create_sharded_model()

        self.assertTrue(isinstance(model.module, ShardedEmbeddingBagCollection))

    def test_sharding_fused_ebc_as_top_level(self) -> None:
        embedding_dim = 128
        num_embeddings = 256
        ebc = FusedEmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="large_table",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["my_feature"],
                    pooling=PoolingType.SUM,
                ),
            ],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        model = DistributedModelParallel(ebc, device=self.device)

        self.assertTrue(isinstance(model.module, ShardedFusedEmbeddingBagCollection))

    @patch("torch._utils_internal.justknobs_check")
    def test_sharding_ebc_input_validation_enabled(self, mock_jk: Mock) -> None:
        model = self._create_sharded_model()
        kjt = KeyedJaggedTensor(
            keys=["my_feature", "my_feature"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
        )
        mock_jk.return_value = True

        with self.assertRaisesRegex(ValueError, "keys must be unique"):
            model(kjt)

        # Count only calls with the input "pytorch/torchrec:enable_kjt_validation"
        # This ignores any other calls to justknobs_check() with other inputs
        # and protects the test from breaking when new JK checks are added.
        validation_calls = [
            call
            for call in mock_jk.call_args_list
            if len(call[0]) > 0
            and call[0][0] == "pytorch/torchrec:enable_kjt_validation"
        ]
        self.assertEqual(
            1,
            len(validation_calls),
            "There should be exactly one call to JK with pytorch/torchrec:enable_kjt_validation",
        )

    @patch("torch._utils_internal.justknobs_check")
    def test_sharding_ebc_validate_input_only_once(self, mock_jk: Mock) -> None:
        model = self._create_sharded_model()
        kjt = KeyedJaggedTensor(
            keys=["my_feature"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
        ).to(self.device)
        mock_jk.return_value = True

        with self.assertLogs(embeddingbag_logger, level="INFO") as logs:
            model(kjt)
            model(kjt)
            model(kjt)

        # Count only calls with the input "pytorch/torchrec:enable_kjt_validation"
        # This ignores any other calls to justknobs_check() with other inputs
        # and protects the test from breaking when new JK checks are added.
        validation_calls = [
            call
            for call in mock_jk.call_args_list
            if len(call[0]) > 0
            and call[0][0] == "pytorch/torchrec:enable_kjt_validation"
        ]
        self.assertEqual(
            1,
            len(validation_calls),
            "There should be exactly one call to JK with pytorch/torchrec:enable_kjt_validation",
        )
        matched_logs = list(
            filter(lambda s: "Validating input features..." in s, logs.output)
        )
        self.assertEqual(1, len(matched_logs))

    @patch("torch._utils_internal.justknobs_check")
    def test_sharding_ebc_input_validation_disabled(self, mock_jk: Mock) -> None:
        model = self._create_sharded_model()
        kjt = KeyedJaggedTensor(
            keys=["my_feature", "my_feature"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
        ).to(self.device)
        mock_jk.return_value = False

        # Without KJT validation, input_dist will not raise exceptions
        try:
            model(kjt)
        except ValueError:
            self.fail("Input validation should not be enabled.")

        # Count only calls with the input "pytorch/torchrec:enable_kjt_validation"
        # This ignores any other calls to justknobs_check() with other inputs
        # and protects the test from breaking when new JK checks are added.
        validation_calls = [
            call
            for call in mock_jk.call_args_list
            if len(call[0]) > 0
            and call[0][0] == "pytorch/torchrec:enable_kjt_validation"
        ]
        self.assertEqual(
            1,
            len(validation_calls),
            "There should be exactly one call to JK with pytorch/torchrec:enable_kjt_validation",
        )

    def _create_sharded_model(
        self, embedding_dim: int = 128, num_embeddings: int = 256
    ) -> DistributedModelParallel:
        ebc = EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="large_table",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["my_feature"],
                    pooling=PoolingType.SUM,
                ),
            ],
        )
        return DistributedModelParallel(ebc, device=self.device)


class ModelParallelSingleRankBase(unittest.TestCase):
    def setUp(self, backend: str = "nccl") -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        self.backend = backend
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if self.backend == "nccl" and self.device == torch.device("cpu"):
            self.skipTest("NCCL not supported on CPUs.")

        dist.init_process_group(backend=backend)

        self.batch_size = 20
        self.num_float_features = 10
        self.tables = []
        self.weighted_tables = []

        self._create_tables()

    def tearDown(self) -> None:
        dist.destroy_process_group()
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _create_tables(self) -> None:
        pass

    def _set_table_weights_precision(self, dtype: DataType) -> None:
        for table in self.tables:
            table.data_type = dtype

        for weighted_table in self.weighted_tables:
            weighted_table.data_type = dtype

    def _create_model(self) -> nn.Module:
        return TestSparseNN(
            tables=self.tables,
            num_float_features=self.num_float_features,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
        )

    def _generate_batch(self) -> ModelInput:
        _, local_batch = ModelInput.generate(
            batch_size=self.batch_size,
            world_size=1,
            num_float_features=self.num_float_features,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )
        batch = local_batch[0].to(self.device)
        return batch

    def _generate_dmps_and_batch(
        self,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        constraints: Optional[Dict[str, trec_dist.planner.ParameterConstraints]] = None,
    ) -> Tuple[List[DistributedModelParallel], ModelInput]:

        if constraints is None:
            constraints = {}
        if sharders is None:
            sharders = get_default_sharders()

        batch = self._generate_batch()

        dmps = []
        pg = dist.GroupMember.WORLD
        assert pg is not None, "Process group is not initialized"
        env = ShardingEnv.from_process_group(pg)

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=trec_dist.comm.get_local_size(env.world_size),
                world_size=env.world_size,
                compute_device=self.device.type,
            ),
            constraints=constraints,
        )

        for _ in range(2):
            # Create two identical models, wrap both in DMP
            m = self._create_model()
            if pg is not None:
                plan = planner.collective_plan(m, sharders, pg)
            else:
                plan = planner.plan(m, sharders)

            dmp = DistributedModelParallel(
                module=m,
                init_data_parallel=False,
                device=self.device,
                sharders=sharders,
                plan=plan,
            )

            with torch.no_grad():
                dmp(batch)
                dmp.init_data_parallel()
            dmps.append(dmp)
        return (dmps, batch)

    def _train_models(
        self,
        m1: DistributedModelParallel,
        m2: DistributedModelParallel,
        batch: ModelInput,
    ) -> None:
        loss1, pred1 = m1(batch)
        loss2, pred2 = m2(batch)
        loss1.backward()
        loss2.backward()

    def _eval_models(
        self,
        m1: DistributedModelParallel,
        m2: DistributedModelParallel,
        batch: ModelInput,
        is_deterministic: bool = True,
        tolerance: Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            loss1, pred1 = m1(batch)
            loss2, pred2 = m2(batch)
        if is_deterministic:
            self.assertTrue(torch.equal(loss1, loss2))
            self.assertTrue(torch.equal(pred1, pred2))
        else:
            if tolerance:
                torch.testing.assert_close(loss1, loss2, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(pred1, pred2, rtol=tolerance, atol=tolerance)
            else:
                rtol, atol = _get_default_rtol_and_atol(loss1, loss2)
                torch.testing.assert_close(loss1, loss2, rtol=rtol, atol=atol)
                rtol, atol = _get_default_rtol_and_atol(pred1, pred2)
                torch.testing.assert_close(pred1, pred2, rtol=rtol, atol=atol)

    def _compare_models(
        self,
        m1: DistributedModelParallel,
        m2: DistributedModelParallel,
        is_deterministic: bool = True,
        use_virtual_table: bool = False,
        tolerance: Optional[float] = None,
    ) -> None:
        sd1 = m1.state_dict()
        sd2 = m2.state_dict()
        for key, value in sd2.items():
            v2 = sd1[key]
            if isinstance(value, ShardedTensor):
                assert isinstance(v2, ShardedTensor)
                self.assertEqual(len(value.local_shards()), len(v2.local_shards()))
                for local_shard_id, (dst, src) in enumerate(
                    zip(value.local_shards(), v2.local_shards())
                ):
                    src_tensor = None
                    dst_tensor = None
                    if isinstance(dst.tensor, PartiallyMaterializedTensor):
                        assert isinstance(src.tensor, PartiallyMaterializedTensor)
                        if use_virtual_table:
                            # kvz zch emb table comparison, id is non-continuous
                            wid_key = key[: key.rfind(".")] + ".weight_id"
                            src_wid = sd1[wid_key].local_shards()[local_shard_id].tensor
                            dst_wid = sd2[wid_key].local_shards()[local_shard_id].tensor

                            sorted_src_wid = torch.sort(src_wid.view(-1))[0]
                            sorted_dst_wid = torch.sort(dst_wid.view(-1))[0]
                            assert torch.equal(sorted_src_wid, sorted_dst_wid)
                            # kvz zch emb table comparison, id is non-continuous
                            src_tensor = src.tensor.get_weights_by_ids(sorted_src_wid)
                            dst_tensor = dst.tensor.get_weights_by_ids(sorted_dst_wid)
                        else:
                            # normal ssd offloading emb table comparison
                            src_tensor = src.tensor.full_tensor()
                            dst_tensor = dst.tensor.full_tensor()
                    else:
                        src_tensor = torch.sort(src.tensor.flatten()).values
                        dst_tensor = torch.sort(dst.tensor.flatten()).values
                    if is_deterministic:
                        self.assertTrue(torch.allclose(src_tensor, dst_tensor))
                    else:
                        if tolerance:
                            rtol, atol = tolerance, tolerance
                        else:
                            rtol, atol = _get_default_rtol_and_atol(
                                src_tensor, dst_tensor
                            )
                        torch.testing.assert_close(
                            src_tensor, dst_tensor, rtol=rtol, atol=atol
                        )
            elif isinstance(value, DTensor):
                assert isinstance(v2, DTensor)
                self.assertEqual(
                    len(value._local_tensor.local_shards()),  # pyre-ignore[16]
                    len(v2._local_tensor.local_shards()),
                )
                for dst, src in zip(
                    value._local_tensor.local_shards(), v2._local_tensor.local_shards()
                ):
                    if is_deterministic:
                        self.assertTrue(torch.equal(src, dst))
                    else:
                        if tolerance:
                            rtol, atol = tolerance, tolerance
                        else:
                            rtol, atol = _get_default_rtol_and_atol(src, dst)
                        torch.testing.assert_close(
                            src._local_tensor, dst._local_tensor, rtol=rtol, atol=atol
                        )
            else:
                dst = value
                src = v2
                if is_deterministic:
                    self.assertTrue(torch.equal(src, dst))
                else:
                    if tolerance:
                        rtol, atol = tolerance, tolerance
                    else:
                        rtol, atol = _get_default_rtol_and_atol(src, dst)
                    torch.testing.assert_close(src, dst, rtol=rtol, atol=atol)


class ModelParallelStateDictBase(ModelParallelSingleRankBase):
    def _create_tables(self) -> None:
        num_features = 4
        num_weighted_features = 2

        self.tables += [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables += [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

    def test_parameter_init(self) -> None:
        class MyModel(nn.Module):
            def __init__(self, device: str, val: float) -> None:
                super().__init__()
                self.p = nn.Parameter(
                    torch.empty(3, dtype=torch.float32, device=device)
                )
                self.val = val
                self.reset_parameters()

            def reset_parameters(self) -> None:
                nn.init.constant_(self.p, self.val)

        # Check that already allocated parameters are left 'as is'.
        unsharded_model = MyModel(device=self.device, val=3.2)
        sharded_model = DistributedModelParallel(
            unsharded_model,
            device=self.device,
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([3.2, 3.2, 3.2], dtype=np.float32),
            sharded_param.detach().cpu().numpy(),
        )

        # Check that parameters over 'meta' device are allocated and initialized.
        meta_model = MyModel(device="meta", val=7.5)
        sharded_model = DistributedModelParallel(
            meta_model,
            device=self.device,
        )
        sharded_param = next(sharded_model.parameters())
        np.testing.assert_array_equal(
            np.array([7.5, 7.5, 7.5], dtype=np.float32),
            sharded_param.detach().cpu().numpy(),
        )

    def test_meta_device_dmp_state_dict(self) -> None:
        # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
        #  `Optional[ProcessGroup]`.
        env = ShardingEnv.from_process_group(dist.GroupMember.WORLD)

        m1 = self._create_model()
        # dmp with real device
        dmp1 = DistributedModelParallel(
            module=m1,
            init_data_parallel=False,
            init_parameters=False,
            sharders=get_default_sharders(),
            device=self.device,
            env=env,
            plan=EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=env.world_size, compute_device=self.device.type
                )
            ).plan(m1, get_default_sharders()),
        )

        m2 = self._create_model()
        # dmp with meta device
        dmp2 = DistributedModelParallel(
            module=m2,
            init_data_parallel=False,
            init_parameters=False,
            sharders=get_default_sharders(),
            device=torch.device("meta"),
            env=env,
            plan=EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=env.world_size, compute_device=self.device.type
                )
            ).plan(m2, get_default_sharders()),
        )

        sd1 = dmp1.state_dict()
        for key, v2 in dmp2.state_dict().items():
            v1 = sd1[key]
            if isinstance(v2, nn.parameter.UninitializedParameter) and isinstance(
                v1, nn.parameter.UninitializedParameter
            ):
                continue
            if isinstance(v2, ShardedTensor):
                self.assertTrue(isinstance(v1, ShardedTensor))
                assert len(v2.local_shards()) == 1
                dst = v2.local_shards()[0].tensor
            elif isinstance(v2, DTensor):
                self.assertTrue(isinstance(v1, DTensor))
                assert len(v2._local_tensor.local_shards()) == 1  # pyre-ignore[16]
                dst = v2._local_tensor.local_shards()[0]
            else:
                dst = v2
            if isinstance(v1, ShardedTensor):
                assert len(v1.local_shards()) == 1
                src = v1.local_shards()[0].tensor
            elif isinstance(v1, DTensor):
                assert len(v1._local_tensor.local_shards()) == 1
                src = v1._local_tensor.local_shards()[0]
            else:
                src = v1
            self.assertEqual(src.size(), dst.size())

    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_load_state_dict(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
    ) -> None:
        assume(
            self.device != torch.device("cpu")
            or kernel_type == EmbeddingComputeKernel.FUSED.value
        )

        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                ),
            ),
        ]
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # validate the models are equivalent
        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)
        self._compare_models(m1, m2)

    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_optimizer_load_state_dict(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        assume(
            self.device != torch.device("cpu")
            or kernel_type == EmbeddingComputeKernel.FUSED.value
        )

        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    fused_params={
                        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                    },
                ),
            ),
        ]
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # train m1 a bit, to make sure the optimizer state is not zero
        self._train_models(m1, m1, batch)
        # sync the state dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))
        # train both models, so they should diverage
        self._train_models(m1, m2, batch)
        # expect eval models to fail, since one model starts with non-zero optimizer state
        with self.assertRaises(AssertionError):
            self._eval_models(m1, m2, batch)

        # sync state dict again
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))
        # load state dict for optimizer as well
        opt1 = m1.fused_optimizer
        opt2 = m2.fused_optimizer
        opt1.load_state_dict(opt2.state_dict())

        self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)
        self._compare_models(m1, m2)

    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
            ]
        ),
        is_training=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_load_state_dict_dp(
        self, sharder_type: str, sharding_type: str, kernel_type: str, is_training: bool
    ) -> None:
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                ),
            ),
        ]
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # validate the models are equivalent
        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)
        self._compare_models(m1, m2)

    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_load_state_dict_prefix(
        self, sharder_type: str, sharding_type: str, kernel_type: str, is_training: bool
    ) -> None:
        assume(
            self.device != torch.device("cpu")
            or kernel_type == EmbeddingComputeKernel.FUSED.value
        )

        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ),
        ]
        (m1, m2), batch = self._generate_dmps_and_batch(sharders)

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", m1.state_dict(prefix="alpha")),
            prefix="alpha",
        )

        # validate the models are equivalent
        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)
        self._compare_models(m1, m2)

    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_params_and_buffers(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        assume(
            self.device != torch.device("cpu")
            or kernel_type == EmbeddingComputeKernel.FUSED.value
        )

        sharders = [
            create_test_sharder(sharder_type, sharding_type, kernel_type),
        ]
        # pyre-ignore[6]
        (m, _), batch = self._generate_dmps_and_batch(sharders=sharders)
        print(f"Sharding Plan: {m._plan}")
        state_dict_keys = set(m.state_dict().keys())
        param_keys = set(dict(m.named_parameters()).keys())
        buffer_keys = set(dict(m.named_buffers()).keys())
        self.assertEqual(state_dict_keys, {*param_keys, *buffer_keys})

    # pyre-ignore
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.COLUMN_WISE.value,
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_load_state_dict_cw_multiple_shards(
        self, sharder_type: str, sharding_type: str, kernel_type: str, is_training: bool
    ) -> None:
        assume(
            self.device != torch.device("cpu")
            or kernel_type == EmbeddingComputeKernel.FUSED.value
        )

        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    fused_params={
                        "learning_rate": 0.2,
                        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                    },
                ),
            ),
        ]

        constraints = defaultdict(lambda: trec_dist.planner.ParameterConstraints())
        num_cw_shards_per_table = {}
        for table in self.tables + self.weighted_tables:
            constraints[table.name].min_partition = 4
            num_cw_shards_per_table[table.name] = table.embedding_dim // 4

        (m1, m2), batch = self._generate_dmps_and_batch(
            sharders, constraints=constraints
        )

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # load optimizer state dict

        # Check to see that we can load optimizer state
        src_optimizer = m1.fused_optimizer
        dst_optimizer = m2.fused_optimizer

        src_optimizer_state_dict = src_optimizer.state_dict()
        dst_optimizer_state_dict = dst_optimizer.state_dict()
        m2.fused_optimizer.load_state_dict(src_optimizer_state_dict)

        # validate the models are equivalent
        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)

        sd1 = m1.state_dict()
        for key, value in m2.state_dict().items():
            if "." in key:
                table_name = key.split(".")[-2]
                v2 = sd1[key]
                if isinstance(value, ShardedTensor):
                    self.assertEqual(
                        len(value.local_shards()), num_cw_shards_per_table[table_name]
                    )
                    dst = value.local_shards()[0].tensor
                elif isinstance(value, DTensor):
                    self.assertEqual(
                        len(value._local_tensor.local_shards()),  # pyre-ignore[16]
                        num_cw_shards_per_table[table_name],
                    )
                    dst = value._local_tensor.local_shards()[0]
                else:
                    dst = value

                if isinstance(v2, ShardedTensor):
                    self.assertEqual(
                        len(value.local_shards()), num_cw_shards_per_table[table_name]
                    )

                    for src_local_shard, dst_local_shard in zip(
                        value.local_shards(), v2.local_shards()
                    ):
                        self.assertTrue(
                            torch.equal(src_local_shard.tensor, dst_local_shard.tensor)
                        )
                elif isinstance(v2, DTensor):
                    self.assertEqual(
                        len(value._local_tensor.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    for src_local_shard, dst_local_shard in zip(
                        value._local_tensor.local_shards(),
                        v2._local_tensor.local_shards(),
                    ):
                        self.assertTrue(torch.equal(src_local_shard, dst_local_shard))
                else:
                    src = v2
                    self.assertTrue(torch.equal(src, dst))

        for param_name, dst_param_group in dst_optimizer_state_dict.items():
            src_param_group = src_optimizer_state_dict[param_name]

            for state_key, dst_opt_state in dst_param_group.items():
                table_name = state_key.split(".")[-2]
                src_opt_state = src_param_group[state_key]
                if isinstance(dst_opt_state, ShardedTensor):
                    self.assertIsInstance(src_param_group[state_key], ShardedTensor)

                    self.assertEqual(
                        len(dst_opt_state.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    self.assertEqual(
                        len(src_opt_state.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    for src_local_shard, dst_local_shard in zip(
                        src_opt_state.local_shards(), dst_opt_state.local_shards()
                    ):
                        self.assertTrue(
                            torch.equal(src_local_shard.tensor, dst_local_shard.tensor)
                        )
                elif isinstance(dst_opt_state, DTensor):
                    self.assertIsInstance(src_opt_state, DTensor)

                    self.assertEqual(
                        len(dst_opt_state._local_tensor.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    self.assertEqual(
                        len(src_opt_state._local_tensor.local_shards()),
                        num_cw_shards_per_table[table_name],
                    )

                    for src_local_shard, dst_local_shard in zip(
                        src_opt_state._local_tensor.local_shards(),
                        dst_opt_state._local_tensor.local_shards(),
                    ):
                        self.assertTrue(torch.equal(src_local_shard, dst_local_shard))
                elif isinstance(dst_opt_state, torch.Tensor):
                    self.assertIsInstance(src_opt_state, torch.Tensor)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
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
        sharder_type: str,
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
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    EmbeddingComputeKernel.FUSED.value,
                    fused_params=fused_params,
                ),
            ),
        ]
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
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
        self._eval_models(
            fused_model, model, batch, is_deterministic=not stochastic_rounding
        )
        self._compare_models(
            fused_model, model, is_deterministic=not stochastic_rounding
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_rowwise_adagrad_numerical_equivalence(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
    ) -> None:
        learning_rate = 0.1
        fused_params = {
            "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            "learning_rate": learning_rate,
        }

        fused_sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    EmbeddingComputeKernel.FUSED.value,
                    fused_params=fused_params,
                ),
            ),
        ]
        dense_sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    EmbeddingComputeKernel.DENSE.value,
                    fused_params=fused_params,
                ),
            ),
        ]
        (fused_model, _), _ = self._generate_dmps_and_batch(fused_sharders)
        (dense_model, _), batch = self._generate_dmps_and_batch(dense_sharders)

        dense_opt = RowWiseAdagrad(
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `parameters`.
            dense_model.module.sparse.parameters(),
            lr=learning_rate,
            eps=1e-8,  # TBE has default eps 1e-8
        )

        # load the baseline model's state_dict onto the new model
        dense_model.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", fused_model.state_dict())
        )

        for _ in range(4):
            dense_opt.zero_grad()
            loss1, pred1 = fused_model(batch)
            loss2, pred2 = dense_model(batch)
            loss1.backward()
            loss2.backward()
            dense_opt.step()

        self._eval_models(fused_model, dense_model, batch, is_deterministic=False)
        self._compare_models(fused_model, dense_model, is_deterministic=False)
