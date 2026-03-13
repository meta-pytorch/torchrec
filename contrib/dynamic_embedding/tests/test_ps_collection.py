# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec_dynamic_embedding.id_transformer_collection import IDTransformerCollection
from torchrec_dynamic_embedding.ps import PSCollection
from torchrec_dynamic_embedding.utils import _get_sharded_modules_recursive
from utils import init_dist, register_memory_io


register_memory_io()


class TestPSCollection(unittest.TestCase):
    def testExtractTensor(self):
        init_dist()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        class Model(nn.Module):
            def __init__(self, configs):
                super().__init__()
                self.emb = EmbeddingCollection(
                    tables=configs, device=torch.device("meta")
                )
                # this is incorrect, but doesn't matter.
                self.dense = nn.Linear(10, 10)

            def forward(self, x):
                return x

        configs = [
            EmbeddingConfig(
                name="AB", num_embeddings=4, embedding_dim=32, feature_names=["A", "B"]
            ),
            EmbeddingConfig(name="C", num_embeddings=8, embedding_dim=32),
        ]
        model = Model(configs=configs)
        model = DMP(module=model, device=device)

        plan = model.plan
        sharded_modules = _get_sharded_modules_recursive(model.module, "", plan)
        sharded_module, params_plan = sharded_modules["emb"]
        ps_collection = PSCollection.fromModule(
            "emb",
            sharded_module,
            params_plan,
            "memory://",
        )

        table_names = ps_collection.table_names()
        self.assertEqual(len(table_names), 2)
        self.assertTrue("AB" in table_names)
        self.assertTrue("C" in table_names)

    def testModuleEviction(self):
        init_dist()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        embedding_dim = 8
        configs = [
            EmbeddingConfig(
                name="AB",
                num_embeddings=4,
                embedding_dim=embedding_dim,
                feature_names=["A", "B"],
                # to check reinit
                weight_init_min=2,
                weight_init_max=2,
            ),
        ]
        model = EmbeddingCollection(tables=configs, device=torch.device("meta"))
        model = DMP(module=model, device=device)

        plan = model.plan
        sharded_modules = _get_sharded_modules_recursive(model.module, "", plan)
        sharded_module, params_plan = sharded_modules[""]
        ps_collection = PSCollection.fromModule(
            "", sharded_module, params_plan, "memory://"
        )

        transformer_collection = IDTransformerCollection(
            configs, ps_collection=ps_collection
        )

        # extract weight manually
        weight = (
            model.module.state_dict()["embeddings.AB.weight"].local_shards()[0].tensor
        )
        self.assertEqual(weight.shape[0], 4)
        self.assertEqual(weight.shape[1], embedding_dim)

        # init to zero
        weight[:] = torch.zeros_like(weight)

        global_kjt_1 = KeyedJaggedTensor(
            keys=["A", "B"],
            values=torch.tensor([1, 2, 3, 4, 1, 2]),
            lengths=torch.tensor([4, 2]),
        )
        cache_kjt, fetch_handles = transformer_collection.transform(global_kjt_1)
        for handle in fetch_handles:
            handle.wait()
        embedding = model(cache_kjt.to(device))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 2, 3, 0, 1])).item()
        )
        self.assertTrue(
            torch.allclose(
                embedding["A"].values(), torch.zeros_like(embedding["A"].values())
            )
        )
        self.assertTrue(
            torch.allclose(
                embedding["B"].values(), torch.zeros_like(embedding["B"].values())
            )
        )

        global_kjt_2 = KeyedJaggedTensor(
            keys=["A", "B"],
            values=torch.tensor([1, 2, 1, 5, 5, 6]),
            lengths=torch.tensor([3, 3]),
        )

        # this will evict 3 and 4
        cache_kjt, fetch_handles = transformer_collection.transform(global_kjt_2)
        for handle in fetch_handles:
            handle.wait()
        embedding = model(cache_kjt.to(device))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 0, 2, 2, 3])).item()
        )
        self.assertTrue(
            torch.allclose(
                embedding["A"].values(), torch.zeros_like(embedding["A"].values())
            )
        )
        self.assertTrue(
            torch.allclose(
                embedding["B"].values(), 2 * torch.ones_like(embedding["B"].values())
            )
        )

        # update to one
        weight[:] = torch.ones_like(weight)
        global_kjt_3 = KeyedJaggedTensor(
            keys=["A", "B"],
            values=torch.tensor([1, 2, 1, 4, 3, 4]),
            lengths=torch.tensor([3, 3]),
        )
        cache_kjt, fetch_handles = transformer_collection.transform(global_kjt_3)
        for handle in fetch_handles:
            handle.wait()
        embedding = model(cache_kjt.to(device))
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 0, 2, 3, 2])).item()
        )
        self.assertTrue(
            torch.allclose(
                embedding["A"].values(), torch.ones_like(embedding["A"].values())
            )
        )
        self.assertTrue(
            torch.allclose(
                embedding["B"].values(), torch.zeros_like(embedding["B"].values())
            )
        )


if __name__ == "__main__":
    unittest.main()
