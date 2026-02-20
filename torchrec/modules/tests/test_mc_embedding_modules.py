#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from copy import deepcopy
from typing import cast, Dict

import torch
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
    return_remapped_lengths_as_mask,
)
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)
from torchrec.sparse.jagged_tensor import (
    ComputeJTDictToKJT,
    KeyedJaggedTensor,
    KeyedTensor,
)


class MCHManagedCollisionEmbeddingBagCollectionTest(unittest.TestCase):
    def test_zch_ebc_ec_train(self) -> None:
        device = torch.device("cpu")
        zch_size = 20
        update_interval = 2
        update_size = 10

        embedding_bag_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        ebc = EmbeddingBagCollection(
            tables=embedding_bag_configs,
            device=device,
        )

        ec = EmbeddingCollection(
            tables=embedding_configs,
            device=device,
        )

        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    device=device,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc_ebc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_bag_configs,
        )

        mcc_ec = ManagedCollisionCollection(
            managed_collision_modules=deepcopy(mc_modules),
            embedding_configs=embedding_configs,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mcc_ebc,
            return_remapped_features=True,
        )
        mc_ec = ManagedCollisionEmbeddingCollection(
            ec,
            mcc_ec,
            return_remapped_features=True,
        )

        mc_modules = [mc_ebc, mc_ec]

        update_one = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.concat(
                [
                    torch.arange(1000, 1000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                    torch.arange(
                        1000 + 2 * update_size,
                        1000 + 3 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((3 * update_size,), dtype=torch.int64),
            weights=None,
        )

        for mc_module in mc_modules:

            self.assertEqual(
                mc_module._managed_collision_collection.open_slots()["t1"].item(),
                zch_size - 1,
            )  # (ZCH-1 slots)

            out1, remapped_kjt1 = mc_module.forward(update_one)

            self.assertEqual(
                mc_module._managed_collision_collection.open_slots()["t1"].item(),
                zch_size - 1,
            )  # prior update, ZCH-1 slots

            out2, remapped_kjt2 = mc_module.forward(update_one)

            self.assertEqual(
                mc_module._managed_collision_collection.open_slots()["t1"].item(), 0
            )  # post update, 0 slots

            assert remapped_kjt1 is not None
            assert remapped_kjt1.keys() == ["f1", "f2"]
            assert torch.all(
                remapped_kjt1["f1"].values() == zch_size - 1
            ), "all remapped ids should be mapped to end of range"
            assert torch.all(
                remapped_kjt1["f2"].values() == zch_size - 1
            ), "all remapped ids should be mapped to end of range"

            assert remapped_kjt2 is not None
            assert remapped_kjt2.keys() == ["f1", "f2"]
            assert torch.all(
                remapped_kjt2["f1"].values() == torch.arange(0, 10, dtype=torch.int64)
            )
            assert torch.all(
                remapped_kjt2["f2"].values()
                == torch.cat(
                    [
                        torch.arange(10, 19, dtype=torch.int64),
                        torch.tensor([zch_size - 1], dtype=torch.int64),  # empty value
                    ]
                )
            )

            if isinstance(mc_module, ManagedCollisionEmbeddingCollection):
                self.assertTrue(isinstance(out1, Dict))
                self.assertTrue(isinstance(out2, Dict))
                self.assertEqual(out1["f1"].values().size(), (update_size, 8))
                self.assertEqual(out2["f2"].values().size(), (update_size, 8))
            else:
                self.assertTrue(isinstance(out1, KeyedTensor))
                self.assertTrue(isinstance(out2, KeyedTensor))
                self.assertEqual(out1["f1"].size(), (update_size, 8))
                self.assertEqual(out2["f2"].size(), (update_size, 8))

            update_two = KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.concat(
                    [
                        torch.arange(2000, 2000 + update_size, dtype=torch.int64),
                        torch.arange(
                            1000 + update_size,
                            1000 + 2 * update_size,
                            dtype=torch.int64,
                        ),
                    ]
                ),
                lengths=torch.ones((2 * update_size,), dtype=torch.int64),
                weights=None,
            )
            out3, remapped_kjt3 = mc_module.forward(update_two)
            out4, remapped_kjt4 = mc_module.forward(update_two)

            assert remapped_kjt3 is not None
            assert torch.all(
                remapped_kjt3["f1"].values() == zch_size - 1
            ), "all remapped ids should be mapped to end of range"

            assert torch.all(
                remapped_kjt3["f2"].values() == remapped_kjt2["f2"].values()
            )

            assert remapped_kjt4 is not None
            assert torch.all(
                remapped_kjt4["f1"].values()
                == torch.cat(
                    [
                        torch.arange(1, 10, dtype=torch.int64),
                        torch.tensor([zch_size - 1], dtype=torch.int64),  # empty value
                    ]
                )
            )
            assert torch.all(
                remapped_kjt4["f2"].values()
                == torch.cat(
                    [
                        torch.arange(10, 19, dtype=torch.int64),
                        torch.tensor(
                            [0], dtype=torch.int64
                        ),  # assigned first open slot
                    ]
                )
            )

            if isinstance(mc_module, ManagedCollisionEmbeddingCollection):
                self.assertTrue(isinstance(out3, Dict))
                self.assertTrue(isinstance(out4, Dict))
                self.assertEqual(out3["f1"].values().size(), (update_size, 8))
                self.assertEqual(out4["f2"].values().size(), (update_size, 8))
            else:
                self.assertTrue(isinstance(out3, KeyedTensor))
                self.assertTrue(isinstance(out4, KeyedTensor))
                self.assertEqual(out3["f1"].size(), (update_size, 8))
                self.assertEqual(out4["f2"].size(), (update_size, 8))

    def test_zch_ebc_ec_eval(self) -> None:
        device = torch.device("cpu")
        zch_size = 20
        update_interval = 2
        update_size = 10

        embedding_bag_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        ebc = EmbeddingBagCollection(
            tables=embedding_bag_configs,
            device=device,
        )
        ec = EmbeddingCollection(
            tables=embedding_configs,
            device=device,
        )
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    device=device,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc_ebc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_bag_configs,
        )

        mcc_ec = ManagedCollisionCollection(
            managed_collision_modules=deepcopy(mc_modules),
            embedding_configs=embedding_configs,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mcc_ebc,
            return_remapped_features=True,
        )
        mc_ec = ManagedCollisionEmbeddingCollection(
            ec,
            mcc_ec,
            return_remapped_features=True,
        )

        mc_modules = [mc_ebc, mc_ec]

        for mc_module in mc_modules:
            update_one = KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.concat(
                    [
                        torch.arange(1000, 1000 + update_size, dtype=torch.int64),
                        torch.arange(
                            1000 + update_size,
                            1000 + 2 * update_size,
                            dtype=torch.int64,
                        ),
                    ]
                ),
                lengths=torch.ones((2 * update_size,), dtype=torch.int64),
                weights=None,
            )
            _, remapped_kjt1 = mc_module.forward(update_one)
            _, remapped_kjt2 = mc_module.forward(update_one)

            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt1["f1"].values()
                == zch_size - 1
            ), "all remapped ids should be mapped to end of range"
            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt1["f2"].values()
                == zch_size - 1
            ), "all remapped ids should be mapped to end of range"

            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt2["f1"].values()
                == torch.arange(0, 10, dtype=torch.int64)
            )
            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt2["f2"].values()
                == torch.cat(
                    [
                        torch.arange(10, 19, dtype=torch.int64),
                        torch.tensor([zch_size - 1], dtype=torch.int64),  # empty value
                    ]
                )
            )

            # Trigger eval mode, zch should not update
            mc_module.eval()

            update_two = KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.concat(
                    [
                        torch.arange(2000, 2000 + update_size, dtype=torch.int64),
                        torch.arange(
                            1000 + update_size,
                            1000 + 2 * update_size,
                            dtype=torch.int64,
                        ),
                    ]
                ),
                lengths=torch.ones((2 * update_size,), dtype=torch.int64),
                weights=None,
            )
            _, remapped_kjt3 = mc_module.forward(update_two)
            _, remapped_kjt4 = mc_module.forward(update_two)

            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt3["f1"].values()
                == zch_size - 1
            ), "all remapped ids should be mapped to end of range"

            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt3["f2"].values()
                # pyrefly: ignore[unsupported-operation]
                == remapped_kjt2["f2"].values()
            )

            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt4["f1"].values()
                == zch_size - 1
            ), "all remapped ids should be mapped to end of range"

            # pyrefly: ignore[no-matching-overload]
            assert torch.all(
                # pyrefly: ignore[unsupported-operation]
                remapped_kjt4["f2"].values()
                # pyrefly: ignore[unsupported-operation]
                == remapped_kjt2["f2"].values()
            )

    def test_mc_collection_traceable(self) -> None:
        device = torch.device("cpu")
        zch_size = 20
        update_interval = 2

        embedding_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    device=device,
                    input_hash_size=2 * zch_size,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_configs,
        )
        mcc.train(False)
        symbolic_trace(mcc, leaf_modules=[ComputeJTDictToKJT.__name__])

    def test_return_remapped_lengths_as_mask(self) -> None:
        mask = return_remapped_lengths_as_mask(
            KeyedJaggedTensor(
                keys=["f0"],
                values=torch.rand(6),
                lengths=torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 1, 1], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor(
                    [True, False, True, False, True, False, False, True, True, True]
                ),
            )
        )

    def test_mc_ebc_ec_forward_mutate_miss_lengths_parameter(self) -> None:
        """
        Test that ManagedCollisionEmbeddingBagCollection and ManagedCollisionEmbeddingCollection
        correctly accept and pass the mutate_miss_lengths parameter to the underlying
        managed collision collection.
        """
        device = torch.device("cpu")
        zch_size = 100

        embedding_bag_configs = [
            EmbeddingBagConfig(
                name="t1",
                num_embeddings=zch_size,
                embedding_dim=8,
                feature_names=["f1"],
            ),
        ]
        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                num_embeddings=zch_size,
                embedding_dim=8,
                feature_names=["f1"],
            ),
        ]
        ebc = EmbeddingBagCollection(
            tables=embedding_bag_configs,
            device=device,
        )

        ec = EmbeddingCollection(
            tables=embedding_configs,
            device=device,
        )

        mc_modules_ebc = {
            "t1": cast(
                ManagedCollisionModule,
                HashZchManagedCollisionModule(
                    zch_size=zch_size,
                    device=torch.device("cpu"),
                    total_num_buckets=10,
                    disable_fallback=False,
                    is_inference=True,
                ),
            ),
        }
        mc_modules_ec = {
            "t1": cast(
                ManagedCollisionModule,
                HashZchManagedCollisionModule(
                    zch_size=zch_size,
                    device=torch.device("cpu"),
                    total_num_buckets=10,
                    disable_fallback=False,
                    is_inference=True,
                ),
            ),
        }
        mcc_ebc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules_ebc,
            embedding_configs=embedding_bag_configs,
        )

        mcc_ec = ManagedCollisionCollection(
            managed_collision_modules=mc_modules_ec,
            embedding_configs=embedding_configs,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mcc_ebc,
            return_remapped_features=True,
        )
        mc_ec = ManagedCollisionEmbeddingCollection(
            ec,
            mcc_ec,
            return_remapped_features=True,
        )

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],
            values=torch.arange(1000, 1010, dtype=torch.int64),
            lengths=torch.ones((10,), dtype=torch.int64),
            weights=None,
        )

        # Test ManagedCollisionEmbeddingBagCollection with mutate_miss_lengths=True (default)
        out_ebc_true, remapped_kjt_ebc_true = mc_ebc.forward(
            kjt, mutate_miss_lengths=True
        )
        self.assertIsNotNone(out_ebc_true)
        self.assertIsNotNone(remapped_kjt_ebc_true)

        # Test ManagedCollisionEmbeddingBagCollection with mutate_miss_lengths=False
        out_ebc_false, remapped_kjt_ebc_false = mc_ebc.forward(
            kjt, mutate_miss_lengths=False
        )
        self.assertIsNotNone(out_ebc_false)
        self.assertIsNotNone(remapped_kjt_ebc_false)

        # Test ManagedCollisionEmbeddingCollection with mutate_miss_lengths=True (default)
        out_ec_true, remapped_kjt_ec_true = mc_ec.forward(kjt, mutate_miss_lengths=True)
        self.assertIsNotNone(out_ec_true)
        self.assertIsNotNone(remapped_kjt_ec_true)

        # Test ManagedCollisionEmbeddingCollection with mutate_miss_lengths=False
        out_ec_false, remapped_kjt_ec_false = mc_ec.forward(
            kjt, mutate_miss_lengths=False
        )
        self.assertIsNotNone(out_ec_false)
        self.assertIsNotNone(remapped_kjt_ec_false)

        # Verify the forward method accepts the parameter without errors
        # and produces valid outputs in all cases
        self.assertEqual(remapped_kjt_ebc_true.keys(), ["f1"])
        self.assertEqual(remapped_kjt_ebc_false.keys(), ["f1"])
        self.assertEqual(remapped_kjt_ec_true.keys(), ["f1"])
        self.assertEqual(remapped_kjt_ec_false.keys(), ["f1"])
        self.assertTrue(
            torch.equal(remapped_kjt_ebc_true.values(), remapped_kjt_ebc_false.values())
        )
        self.assertTrue(
            torch.equal(remapped_kjt_ec_true.values(), remapped_kjt_ec_false.values())
        )
        self.assertTrue(
            torch.equal(
                remapped_kjt_ebc_true.lengths(), remapped_kjt_ebc_false.lengths()
            )
        )
        self.assertTrue(
            torch.equal(remapped_kjt_ec_true.lengths(), remapped_kjt_ec_false.lengths())
        )
