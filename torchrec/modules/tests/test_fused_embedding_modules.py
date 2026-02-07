#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type

import hypothesis.strategies as st
import torch
import torch.fx
import torchrec
from fbgemm_gpu.split_table_batched_embeddings_ops_training import EmbeddingLocation
from hypothesis import given, settings
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.fused_embedding_modules import (
    fuse_embedding_optimizer,
    FusedEmbeddingBagCollection,
    FusedEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

devices: List[torch.device] = [torch.device("cpu")]
if torch.cuda.device_count() > 1:
    devices.append(torch.device("cuda"))


class TestModel(torch.nn.Module):
    def __init__(self, ebc: EmbeddingBagCollection) -> None:
        super().__init__()
        self.ebc = ebc
        self.over_arch = torch.nn.Linear(
            4,
            1,
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ebc_output = self.ebc.forward(kjt).to_dict()
        sparse_features = []
        for key in kjt.keys():
            sparse_features.append(ebc_output[key])
        sparse_features = torch.cat(sparse_features, dim=0)
        return self.over_arch(sparse_features)


class TestSequentialModel(torch.nn.Module):
    def __init__(self, ec: EmbeddingCollection) -> None:
        super().__init__()
        self.ec = ec
        self.over_arch = torch.nn.Linear(
            ec.embedding_dim(),
            1,
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ec_output = self.ec.forward(kjt)
        sparse_features = []
        for key in kjt.keys():
            sparse_features.extend(ec_output[key].to_dense())
        sparse_features = torch.Tensor(sparse_features)
        sparse_features = torch.sum(sparse_features)
        return self.over_arch(sparse_features)


class FusedEmbeddingBagCollectionTest(unittest.TestCase):
    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_unweighted(
        self,
        device: torch.device,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        ).to(device)

        pooled_embeddings = ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2", "f3"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f3"].shape, (features.stride(), 4))

        torch.testing.assert_close(
            pooled_embeddings["f1"][1], torch.zeros(4).to(device)
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"][0], torch.zeros(4).to(device)
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"][2], torch.zeros(4).to(device)
        )

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_unweighted_batchsize_1_shared(
        self,
        device: torch.device,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1", "f2"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0
        # f1   [0,1]
        # f2   [3]
        # f3   [4,5]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 3, 4, 5]),
            lengths=torch.tensor([2, 1, 2]),
        ).to(device)

        pooled_embeddings = ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2", "f3"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f3"].shape, (features.stride(), 4))

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_unweighted_batchsize_1_one_table(
        self,
        device: torch.device,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1", "f2"]
        )
        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0
        # f1   [0,1]
        # f2   [3]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 3]),
            lengths=torch.tensor([2, 1]),
        ).to(device)

        pooled_embeddings = ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_shared_tables(
        self,
        device: torch.device,
    ) -> None:
        ebc_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1", "f2"]
        )
        ebc = FusedEmbeddingBagCollection(
            tables=[ebc_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),
        ).to(device)

        pooled_embeddings = ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))

        torch.testing.assert_close(
            pooled_embeddings["f1"][1], torch.zeros(4).to(device)
        )

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_state_dict(
        self,
        device: torch.device,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=2, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        ebc.load_state_dict(ebc.state_dict())

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_state_dict_manual(self, device: torch.device) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=2, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        ebc.load_state_dict(
            OrderedDict(
                [
                    (
                        "embedding_bags.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).to(device),
                    ),
                    (
                        "embedding_bags.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]).to(device),
                    ),
                    (
                        "embedding_bags.t3.weight",
                        torch.Tensor([[16, 16, 16, 16], [32, 32, 32, 32]]).to(device),
                    ),
                ]
            )
        )

        state_dict = ebc.state_dict()
        torch.testing.assert_close(
            state_dict["embedding_bags.t1.weight"],
            torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).to(device),
        ),
        torch.testing.assert_close(
            state_dict["embedding_bags.t2.weight"],
            torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]).to(device),
        ),
        torch.testing.assert_close(
            state_dict["embedding_bags.t3.weight"],
            torch.Tensor([[16, 16, 16, 16], [32, 32, 32, 32]]).to(device),
        )

        #     0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f2   [0]    [1]    [0,1]
        # f3   []    []    [0]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 0, 0, 1, 0, 1, 0]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2, 0, 0, 1]),
        ).to(device)

        pooled_embeddings = ebc(features)

        torch.testing.assert_close(
            pooled_embeddings["f1"],
            torch.tensor(
                [[3, 3, 3, 3], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32
            ).to(device),
        )
        torch.testing.assert_close(
            pooled_embeddings["f2"],
            torch.tensor(
                [[4, 4, 4, 4], [8, 8, 8, 8], [12, 12, 12, 12]], dtype=torch.float32
            ).to(device),
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"],
            torch.tensor(
                [[0, 0, 0, 0], [0, 0, 0, 0], [16, 16, 16, 16]], dtype=torch.float32
            ).to(device),
        )

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_shared_tables_shared_features(self, device: torch.device) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=4,
            num_embeddings=2,
            feature_names=["f1", "shared_f1"],
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=2,
            feature_names=["f2", "shared_f1"],
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        ebc.load_state_dict(
            OrderedDict(
                [
                    (
                        "embedding_bags.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).to(device),
                    ),
                    (
                        "embedding_bags.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]).to(device),
                    ),
                ]
            )
        )

        #     0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f2   [0]    [1]    [0,1]
        # shared_f1   []    [0]    [0,1]

        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "shared_f1"],
            values=torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2, 0, 1, 2]),
        ).to(device)

        pooled_embeddings = ebc(features)

        torch.testing.assert_close(
            pooled_embeddings["f1"],
            torch.tensor(
                [[3, 3, 3, 3], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32
            ).to(device),
        )

        torch.testing.assert_close(
            pooled_embeddings["shared_f1@t1"],
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [3, 3, 3, 3],
                ],
                dtype=torch.float32,
            ).to(device),
        )

        torch.testing.assert_close(
            pooled_embeddings["f2"],
            torch.tensor(
                [[4, 4, 4, 4], [8, 8, 8, 8], [12, 12, 12, 12]], dtype=torch.float32
            ).to(device),
        )

        torch.testing.assert_close(
            pooled_embeddings["shared_f1@t2"],
            torch.tensor(
                [[0, 0, 0, 0], [4, 4, 4, 4], [12, 12, 12, 12]], dtype=torch.float32
            ).to(device),
        )

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_weighted(self, device: torch.device) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            is_weighted=True,
            device=device,
        )

        ebc.load_state_dict(
            OrderedDict(
                [
                    (
                        "embedding_bags.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
                    ),
                    (
                        "embedding_bags.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]),
                    ),
                ]
            )
        )

        #     0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f2   [0]    [1]    [0,1]
        # feature

        #     0       1        2  <-- batch
        # f1  [1.0,2.0] [] [3.0]
        # f2   [5.0]    [7.0]    [11.0, 13.0]
        # weight
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 0, 0, 1, 0, 1]),
            weights=torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2]),
        ).to(device)

        pooled_embeddings = ebc(features)

        torch.testing.assert_close(
            pooled_embeddings["f1"],
            torch.tensor(
                [[5.0] * 4, [0.0] * 4, [3.0] * 4], dtype=torch.float32, device=device
            ),
        )

        torch.testing.assert_close(
            pooled_embeddings["f2"],
            torch.tensor(
                [[20.0] * 4, [56.0] * 4, [4 * 11 + 8 * 13] * 4],
                dtype=torch.float32,
                device=device,
            ),
        )

    @settings(deadline=None)
    @given(
        optimizer_type_and_kwargs=st.sampled_from(
            [
                (torch.optim.SGD, {"lr": 0.1}),
                (torch.optim.Adagrad, {"lr": 0.1}),
                (torchrec.optim.RowWiseAdagrad, {"lr": 0.1}),
            ]
        ),
        device=st.sampled_from(devices),
    )
    def test_optimizer_fusion(
        self,
        optimizer_type_and_kwargs: Tuple[Type[torch.optim.Optimizer], Dict[str, Any]],
        device: torch.device,
    ) -> None:
        optimizer_type, optimizer_kwargs = optimizer_type_and_kwargs
        tables = [
            EmbeddingBagConfig(
                num_embeddings=2,
                embedding_dim=4,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingBagConfig(
                num_embeddings=2,
                embedding_dim=4,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]

        fused_ebc = FusedEmbeddingBagCollection(
            tables=tables,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )

        ebc = EmbeddingBagCollection(tables=tables, device=device)

        state_dict = OrderedDict(
            [
                (
                    "embedding_bags.table_0.weight",
                    torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).to(device),
                ),
                (
                    "embedding_bags.table_1.weight",
                    torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]).to(device),
                ),
            ]
        )
        fused_ebc.load_state_dict(state_dict)
        ebc.load_state_dict(state_dict)

        #        0       1        2  <-- batch
        # "f1"   [] [0]    [0,1]
        # "f2"   [1]    [0,1]    []
        #  ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 0, 1, 1, 0, 1]),
            lengths=torch.tensor([0, 1, 2, 1, 2, 0]),
        ).to(device)

        opt = optimizer_type(ebc.parameters(), **optimizer_kwargs)

        def run_one_training_step() -> None:
            fused_pooled_embeddings = fused_ebc(features)
            fused_vals = []
            for _name, param in fused_pooled_embeddings.to_dict().items():
                fused_vals.append(param)
            torch.cat(fused_vals, dim=1).sum().backward()

            opt.zero_grad()
            pooled_embeddings = ebc(features)

            vals = []
            for _name, param in pooled_embeddings.to_dict().items():
                vals.append(param)
            torch.cat(vals, dim=1).sum().backward()
            opt.step()

        run_one_training_step()
        torch.testing.assert_close(
            ebc.state_dict()["embedding_bags.table_0.weight"],
            fused_ebc.state_dict()["embedding_bags.table_0.weight"],
        )

        run_one_training_step()
        torch.testing.assert_close(
            ebc.state_dict()["embedding_bags.table_0.weight"],
            fused_ebc.state_dict()["embedding_bags.table_0.weight"],
        )

        fused_optimizer = fused_ebc.fused_optimizer()
        fused_optimizer.load_state_dict(fused_optimizer.state_dict())

    def unweighted_replacement(
        self, device: torch.device, location: Optional[EmbeddingLocation] = None
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = EmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            device=device,
        )

        fused_ebc = fuse_embedding_optimizer(
            ebc,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        ).to(device)

        pooled_embeddings = fused_ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2", "f3"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f3"].shape, (features.stride(), 4))

        torch.testing.assert_close(
            pooled_embeddings["f1"][1], torch.zeros(4).to(device)
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"][0], torch.zeros(4).to(device)
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"][2], torch.zeros(4).to(device)
        )

    def test_fuse_embedding_optimizer_replacement_cpu(self) -> None:
        self.unweighted_replacement(torch.device("cpu"))

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    def test_fuse_embedding_optimizer_replacement_cuda(self) -> None:
        self.unweighted_replacement(torch.device("cuda"))

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    def test_fuse_embedding_optimizer_replacement_cuda_uvm_caching(self) -> None:
        self.unweighted_replacement(
            torch.device("cuda"), location=EmbeddingLocation.MANAGED_CACHING
        )

    @settings(deadline=None)
    @given(device=st.sampled_from(devices))
    def test_ebc_model_replacement(self, device: torch.device) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = EmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
        )
        test_model = TestModel(ebc).to(device)
        test_model = fuse_embedding_optimizer(
            test_model,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        self.assertIsInstance(test_model.ebc, FusedEmbeddingBagCollection)

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature

        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        ).to(device)
        test_model(features)

    def test_composability(self) -> None:
        device = torch.device("cpu")
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = EmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
        )
        test_model = TestModel(ebc).to(device)
        original_named_buffers = dict(test_model.named_buffers())
        original_named_modules = dict(test_model.named_modules())
        original_named_parameters = dict(test_model.named_parameters())

        fuse_embedding_optimizer(
            test_model,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        fused_named_buffers = dict(test_model.named_buffers())
        fused_named_modules = dict(test_model.named_modules())
        fused_named_parameters = dict(test_model.named_parameters())

        self.assertEqual(original_named_buffers.keys(), fused_named_buffers.keys())
        for buffer_key in original_named_buffers.keys():
            original_buffer, fused_buffer = (
                original_named_buffers[buffer_key],
                fused_named_buffers[buffer_key],
            )
            self.assertEqual(original_buffer.shape, fused_buffer.shape)

        self.assertEqual(original_named_modules.keys(), fused_named_modules.keys())

        self.assertEqual(
            original_named_parameters.keys(), fused_named_parameters.keys()
        )
        for param_key in original_named_parameters.keys():
            original_param, fused_param = (
                original_named_parameters[param_key],
                fused_named_parameters[param_key],
            )
            self.assertEqual(original_param.shape, fused_param.shape)


class FusedEmbeddingCollectionTest(unittest.TestCase):
    def _assert_dense_list_equality(
        self, dense_list_1: List[torch.Tensor], dense_list_2: List[torch.Tensor]
    ) -> None:
        self.assertEqual(len(dense_list_1), len(dense_list_2))
        for tensor_1, tensor_2 in zip(dense_list_1, dense_list_2):
            torch.testing.assert_close(tensor_1, tensor_2, rtol=0, atol=0)

    @unittest.skipIf(
        # TODO remove restriction once FBGEMM supports CPU
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    @settings(deadline=None)
    @given(device=st.sampled_from([torch.device("cuda")]))
    def test_forward_with_state_dict(
        self,
        device: torch.device,
    ) -> None:
        # this tests the common components of an EmbeddingCollection, namely
        # lookup with multiple features, shared features,
        # calling state_dict and calling load_state_dict
        e1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=4,
            num_embeddings=2,
            feature_names=["f1", "f1_1", "f_shared"],
        )
        e2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=2,
            feature_names=["f2", "f_shared"],
        )
        e3_config = EmbeddingConfig(
            name="t3", embedding_dim=4, num_embeddings=2, feature_names=["f3"]
        )

        ec = FusedEmbeddingCollection(
            tables=[e1_config, e2_config, e3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        non_fused_ec = EmbeddingCollection(
            tables=[e1_config, e2_config, e3_config],
            device=device,
        )

        ec.load_state_dict(
            OrderedDict(
                [
                    (
                        "embeddings.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).to(device),
                    ),
                    (
                        "embeddings.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]).to(device),
                    ),
                    (
                        "embeddings.t3.weight",
                        torch.Tensor([[16, 16, 16, 16], [32, 32, 32, 32]]).to(device),
                    ),
                ]
            )
        )

        non_fused_ec.load_state_dict(ec.state_dict())

        #    0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f1_1   [0] [1]    [1,0]
        # f2   [0]    [1]    [0,1]
        # f3   []    []    [0]
        # f_shared   [0]    [1]    [0,1]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f1_1", "f2", "f3", "f_shared"],
            values=torch.tensor([0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2, 1, 1, 2, 0, 0, 1, 1, 1, 2]),
        ).to(device)

        sequential_embeddings = ec(features)
        self.assertEqual(
            set(sequential_embeddings.keys()),
            {"f1", "f1_1", "f_shared@t1", "f2", "f_shared@t2", "f3"},
        )

        self._assert_dense_list_equality(
            sequential_embeddings["f1"].to_dense(),
            [
                torch.Tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]).to(device),
                torch.empty(0, 4).to(device),
                torch.Tensor([[1.0, 1.0, 1.0, 1.0]]).to(device),
            ],
        )

        self._assert_dense_list_equality(
            sequential_embeddings["f1_1"].to_dense(),
            [
                torch.Tensor([[1.0, 1.0, 1.0, 1.0]]).to(device),
                torch.Tensor([[2.0, 2.0, 2.0, 2.0]]).to(device),
                torch.Tensor([[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]).to(device),
            ],
        )

        non_fused_sequential_embeddings = non_fused_ec(features)

        for key in sequential_embeddings:
            self._assert_dense_list_equality(
                sequential_embeddings[key].to_dense(),
                non_fused_sequential_embeddings[key].to_dense(),
            )

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    @settings(deadline=None)
    @given(
        optimizer_type_and_kwargs=st.sampled_from(
            [
                (torch.optim.SGD, {"lr": 0.1}),
                (torch.optim.Adagrad, {"lr": 0.1}),
                (torchrec.optim.RowWiseAdagrad, {"lr": 0.1}),
            ]
        ),
        device=st.sampled_from([torch.device("cuda")]),
    )
    def test_optimizer_fusion(
        self,
        optimizer_type_and_kwargs: Tuple[Type[torch.optim.Optimizer], Dict[str, Any]],
        device: torch.device,
    ) -> None:
        optimizer_type, optimizer_kwargs = optimizer_type_and_kwargs
        embedding_configs = [
            EmbeddingConfig(
                num_embeddings=2,
                embedding_dim=4,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingConfig(
                num_embeddings=2,
                embedding_dim=4,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]

        fused_ec = FusedEmbeddingCollection(
            tables=embedding_configs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )

        ec = EmbeddingCollection(tables=embedding_configs, device=device)

        state_dict = OrderedDict(
            [
                (
                    "embeddings.table_0.weight",
                    torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]).to(device),
                ),
                (
                    "embeddings.table_1.weight",
                    torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]).to(device),
                ),
            ]
        )
        fused_ec.load_state_dict(state_dict)
        ec.load_state_dict(state_dict)

        #        0       1        2  <-- batch
        # "f1"   [] [0]    [0,1]
        # "f2"   [1]    [0,1]    []
        #  ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 0, 1, 1, 0, 1]),
            lengths=torch.tensor([0, 1, 2, 1, 2, 0]),
        ).to(device)

        opt = optimizer_type(ec.parameters(), **optimizer_kwargs)

        def run_one_training_step() -> None:
            fused_embeddings = fused_ec(features)
            fused_vals = []
            for _name, jt in fused_embeddings.items():
                fused_vals.extend(jt.to_dense())
            torch.cat(fused_vals).sum().backward()

            opt.zero_grad()
            sequence_embeddings = ec(features)
            vals = []
            for _name, jt in sequence_embeddings.items():
                vals.extend(jt.to_dense())
            torch.cat(vals).sum().backward()
            opt.step()

        run_one_training_step()
        torch.testing.assert_close(
            ec.state_dict()["embeddings.table_0.weight"],
            fused_ec.state_dict()["embeddings.table_0.weight"],
        )

        run_one_training_step()
        torch.testing.assert_close(
            ec.state_dict()["embeddings.table_0.weight"],
            fused_ec.state_dict()["embeddings.table_0.weight"],
        )

        fused_optimizer = fused_ec.fused_optimizer()
        fused_optimizer.load_state_dict(fused_optimizer.state_dict())

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    @settings(deadline=None)
    @given(
        device=st.sampled_from([torch.device("cuda")]),
    )
    def test_replacement_composability(self, device: torch.device) -> None:
        # tests replacement and composability

        ec_1 = EmbeddingConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        ec_2 = EmbeddingConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ec_3 = EmbeddingConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        embedding_collection = EmbeddingCollection(
            tables=[ec_1, ec_2, ec_3],
        )

        test_model = TestSequentialModel(embedding_collection).to(device)
        original_named_buffers = dict(test_model.named_buffers())
        original_named_modules = dict(test_model.named_modules())
        original_named_parameters = dict(test_model.named_parameters())

        fuse_embedding_optimizer(
            test_model,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        self.assertIsInstance(test_model.ec, FusedEmbeddingCollection)

        fused_named_buffers = dict(test_model.named_buffers())
        fused_named_modules = dict(test_model.named_modules())
        fused_named_parameters = dict(test_model.named_parameters())

        self.assertEqual(original_named_buffers.keys(), fused_named_buffers.keys())
        for buffer_key in original_named_buffers.keys():
            original_buffer, fused_buffer = (
                original_named_buffers[buffer_key],
                fused_named_buffers[buffer_key],
            )
            self.assertEqual(original_buffer.shape, fused_buffer.shape)

        self.assertEqual(original_named_modules.keys(), fused_named_modules.keys())

        self.assertEqual(
            original_named_parameters.keys(), fused_named_parameters.keys()
        )
        for param_key in original_named_parameters.keys():
            original_param, fused_param = (
                original_named_parameters[param_key],
                fused_named_parameters[param_key],
            )
            self.assertEqual(original_param.shape, fused_param.shape)
