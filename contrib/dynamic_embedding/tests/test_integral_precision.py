#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec_dynamic_embedding.id_transformer_group import IDTransformerGroup
from utils import init_dist, register_memory_io

register_memory_io()


class Model(nn.Module):
    def __init__(self, num_embeddings, init_max, init_min, batch_size):
        super().__init__()
        self.embedding_dim = 16
        self.batch_size = batch_size
        self.config = EmbeddingConfig(
            name="id",
            embedding_dim=self.embedding_dim,
            num_embeddings=num_embeddings,
            weight_init_max=init_max,
            weight_init_min=init_min,
        )
        self.emb = EmbeddingCollection(
            tables=[self.config], device=torch.device("meta")
        )
        self.dense = nn.Linear(16, 1)

    def forward(self, x):
        embeddings = (
            self.emb(x)["id"]
            .values()
            .reshape((self.batch_size, -1, self.embedding_dim))
        )
        fused = embeddings.sum(dim=1)
        output = self.dense(fused)
        pred = torch.sigmoid(output)
        return pred


class TestPSPrecision(unittest.TestCase):
    def testExtractTensor(self):
        init_dist()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        batch_size = 4
        model1 = Model(
            num_embeddings=1000, init_max=1, init_min=1, batch_size=batch_size
        )

        model2 = Model(
            num_embeddings=100, init_max=1, init_min=1, batch_size=batch_size
        )
        model2_config = model2.config

        model2.dense.weight.data.copy_(model1.dense.weight.data)
        model2.dense.bias.data.copy_(model1.dense.bias.data)

        def get_dmp(model):
            topology = Topology(
                world_size=dist.get_world_size(),
                local_world_size=dist.get_world_size(),
                compute_device="cuda",
            )

            fused_params = {
                "learning_rate": 1e-1,
                "optimizer": EmbOptimType.ADAM,
                "cache_load_factor": 0.1,
            }
            sharders = [
                EmbeddingBagCollectionSharder(fused_params=fused_params),
                FusedEmbeddingBagCollectionSharder(fused_params=fused_params),
                EmbeddingCollectionSharder(fused_params=fused_params),
            ]
            plan = EmbeddingShardingPlanner(topology=topology, constraints=None).plan(
                model, sharders
            )
            model = DMP(module=model, device=device, plan=plan, sharders=sharders)

            dense_optimizer = KeyedOptimizerWrapper(
                dict(in_backward_optimizer_filter(model.named_parameters())),
                lambda params: torch.optim.Adam(params, lr=1e-1),
            )
            optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

            return model, optimizer

        model1, optimizer1 = get_dmp(model1)
        model2, optimizer2 = get_dmp(model2)

        transformer = IDTransformerGroup(
            "memory://",
            model2,
            {"emb": [model2_config]},
            transform_config={"type": "naive"},
            ps_config={"chunk_size": 1024},
        )

        def sigmoid_crossentropy(y_true, y_pred):
            ce = nn.BCELoss()(y_pred, y_true)
            return torch.mean(torch.sum(ce, dim=-1))

        for i in range(100):
            kjt = KeyedJaggedTensor(
                keys=["id"],
                values=torch.randint(0, 1000, (40,), dtype=torch.long),
                lengths=torch.tensor([10, 10, 10, 10], dtype=torch.long),
            )
            kjts, fetch_handles = transformer.transform({"emb": kjt})
            for handle in fetch_handles:
                handle.wait()
            mapped_kjt = kjts["emb"]
            label = torch.randint(0, 2, (4, 1), device=device).float()
            kjt = kjt.to(device)
            mapped_kjt = mapped_kjt.to(device)

            output1 = model1(kjt)
            task_loss1 = sigmoid_crossentropy(label, output1)

            task_loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            output2 = model2(mapped_kjt)
            task_loss2 = sigmoid_crossentropy(label, output2)

            task_loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

            self.assertTrue(abs((task_loss1 - task_loss2).item()) < 1e-7)


if __name__ == "__main__":
    unittest.main()
