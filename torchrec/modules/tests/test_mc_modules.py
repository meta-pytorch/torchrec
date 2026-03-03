#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict

import torch
from torchrec.fb.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.mc_modules import (
    average_threshold_filter,
    DistanceLFU_EvictionPolicy,
    dynamic_threshold_filter,
    LFU_EvictionPolicy,
    LRU_EvictionPolicy,
    ManagedCollisionCollection,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
    probabilistic_threshold_filter,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class TestEvictionPolicy(unittest.TestCase):
    def test_lfu_eviction(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [0] * 5)

        # insert some values to zch
        # we have 10 counts of 4 and 1 count of 5
        # pyrefly: ignore[unsupported-operation]
        mc_module._mch_sorted_raw_ids[0:2] = torch.tensor([4, 5])
        # pyrefly: ignore[unsupported-operation]
        mc_module._mch_counts[0:2] = torch.tensor([10, 1])

        ids = [3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        # 5, empty, empty, empty will be evicted
        # 6, 7, 8 will be added
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  `Union[Tensor, Module]`.
            # pyrefly: ignore[no-matching-overload]
            list(_mch_sorted_raw_ids),
            [4, 6, 7, 8, torch.iinfo(torch.int64).max],
        )
        # 11 counts of 5, 3 counts of 6, 3 counts of 7, 3 counts of 8
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [11, 3, 3, 3, torch.iinfo(torch.int64).max])

    def test_lru_eviction(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LRU_EvictionPolicy(decay_exponent=1.0),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_last_access_iter = mc_module._mch_last_access_iter
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_last_access_iter), [0] * 5)

        ids = [5, 6, 7]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)
        self.assertEqual(mc_module.open_slots().item(), 1)
        ids = [3, 4, 5]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)
        self.assertEqual(mc_module.open_slots().item(), 0)
        ids = [7, 8]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)
        self.assertEqual(mc_module.open_slots().item(), 0)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  `Union[Tensor, Module]`.
            # pyrefly: ignore[no-matching-overload]
            list(_mch_sorted_raw_ids),
            [3, 4, 7, 8, torch.iinfo(torch.int64).max],
        )
        _mch_last_access_iter = mc_module._mch_last_access_iter
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_last_access_iter), [2, 2, 3, 3, 3])
        self.assertEqual(mc_module.open_slots().item(), 0)

    def test_distance_lfu_eviction(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=DistanceLFU_EvictionPolicy(decay_exponent=1.0),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [0] * 5)
        _mch_last_access_iter = mc_module._mch_last_access_iter
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_last_access_iter), [0] * 5)

        ids = [5, 5, 5, 5, 5, 6]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [3, 4]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [7, 8]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  `Union[Tensor, Module]`.
            # pyrefly: ignore[no-matching-overload]
            list(_mch_sorted_raw_ids),
            [3, 5, 7, 8, torch.iinfo(torch.int64).max],
        )
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [1, 5, 1, 1, torch.iinfo(torch.int64).max])
        _mch_last_access_iter = mc_module._mch_last_access_iter
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_last_access_iter), [2, 1, 3, 3, 3])

    def test_distance_lfu_eviction_fast_decay(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=DistanceLFU_EvictionPolicy(decay_exponent=10.0),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [0] * 5)
        _mch_last_access_iter = mc_module._mch_last_access_iter
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_last_access_iter), [0] * 5)

        ids = [5, 5, 5, 5, 5, 6]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [3, 4]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [7, 8]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  `Union[Tensor, Module]`.
            # pyrefly: ignore[no-matching-overload]
            list(_mch_sorted_raw_ids),
            [3, 4, 7, 8, torch.iinfo(torch.int64).max],
        )
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [1, 1, 1, 1, torch.iinfo(torch.int64).max])
        _mch_last_access_iter = mc_module._mch_last_access_iter
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_last_access_iter), [2, 2, 3, 3, 3])

    def test_dynamic_threshold_filter(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(
                threshold_filtering_func=lambda tensor: dynamic_threshold_filter(
                    tensor, threshold_skew_multiplier=0.75
                )
            ),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [0] * 5)

        ids = [5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1]
        # threshold is len(ids) / unique_count(ids) * threshold_skew_multiplier
        # = 15 / 5 * 0.5 = 2.25
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  `Union[Tensor, Module]`.
            # pyrefly: ignore[no-matching-overload]
            list(_mch_sorted_raw_ids),
            [3, 4, 5, torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).max],
        )
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [3, 4, 5, 0, torch.iinfo(torch.int64).max])

    def test_average_threshold_filter(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(
                threshold_filtering_func=average_threshold_filter
            ),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [0] * 5)

        # insert some values to zch
        # we have 10 counts of 4 and 1 count of 5
        # pyrefly: ignore[unsupported-operation]
        mc_module._mch_sorted_raw_ids[0:2] = torch.tensor([4, 5])
        # pyrefly: ignore[unsupported-operation]
        mc_module._mch_counts[0:2] = torch.tensor([10, 1])

        ids = [3, 4, 5, 6, 6, 6, 7, 8, 8, 9, 10]
        # threshold is 1.375
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        # empty, empty will be evicted
        # 6, 8 will be added
        # 7 is not added because it's below the average threshold
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  `Union[Tensor, Module]`.
            # pyrefly: ignore[no-matching-overload]
            list(_mch_sorted_raw_ids),
            [4, 5, 6, 8, torch.iinfo(torch.int64).max],
        )
        # count for 4 is not updated since it's below the average threshold
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [10, 1, 3, 2, torch.iinfo(torch.int64).max])

    def test_probabilistic_threshold_filter(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(
                threshold_filtering_func=lambda tensor: probabilistic_threshold_filter(
                    tensor,
                    per_id_probability=0.01,
                )
            ),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        #  `Union[Tensor, Module]`.
        # pyrefly: ignore[no-matching-overload]
        self.assertEqual(list(_mch_counts), [0] * 5)

        unique_ids = [5, 4, 3, 2, 1]
        id_counts = [100, 80, 60, 40, 10]
        ids = [id for id, count in zip(unique_ids, id_counts) for _ in range(count)]
        # chance of being added is [0.63, 0.55, 0.45, 0.33]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }

        torch.manual_seed(42)
        for _ in range(10):
            mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            #  Module]` is not a function.
            # pyrefly: ignore[not-callable]
            sorted(_mch_sorted_raw_ids.tolist()),
            [2, 3, 4, 5, torch.iinfo(torch.int64).max],
        )
        # _mch_counts is like
        # [80, 180, 160, 800, 9223372036854775807]

    def test_fx_jit_script_not_training(self) -> None:
        model = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=1,
            input_hash_size=100,
        )

        model.train(False)
        gm = torch.fx.symbolic_trace(model)
        torch.jit.script(gm)


class TestManagedCollisionCollection(unittest.TestCase):
    def test_forward_passes_mutate_miss_lengths_parameter(self) -> None:
        """
        Test that ManagedCollisionCollection.forward correctly passes
        the mutate_miss_lengths parameter to the underlying mc_modules.
        """
        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                num_embeddings=100,
                embedding_dim=8,
                feature_names=["f1"],
            ),
        ]

        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                HashZchManagedCollisionModule(
                    zch_size=100,
                    device=torch.device("cpu"),
                    total_num_buckets=10,
                    disable_fallback=False,
                    is_inference=True,
                ),
            ),
        }

        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_configs,
        )

        kjt = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            lengths=torch.tensor([2, 2], dtype=torch.int64),
        )

        # Test with mutate_miss_lengths=True (default)
        output_true = mcc(kjt, mutate_miss_lengths=True)
        self.assertIsNotNone(output_true)
        self.assertEqual(output_true.keys(), ["f1"])

        # Test with mutate_miss_lengths=False
        output_false = mcc(kjt, mutate_miss_lengths=False)
        self.assertIsNotNone(output_false)
        self.assertEqual(output_false.keys(), ["f1"])

        # Verify the forward method accepts the parameter without errors
        # and produces valid output in both cases
        self.assertTrue(torch.equal(output_true.values(), output_false.values()))
        self.assertTrue(torch.equal(output_true.lengths(), output_false.lengths()))

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_return_zch_runtime_meta(self) -> None:
        device = torch.device("cuda")
        zch_size = 8
        total_num_buckets = 2

        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
            EmbeddingConfig(
                name="t2",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f3"],
            ),
        ]
        hash_zch_mc1 = HashZchManagedCollisionModule(
            zch_size=zch_size,
            device=device,
            name="t1",
            total_num_buckets=total_num_buckets,
            track_id_freq=True,
        )
        self.assertIsNotNone(hash_zch_mc1._hash_zch_runtime_meta)
        self.assertIsNotNone(hash_zch_mc1._hash_zch_identities)
        hash_zch_mc1._hash_zch_runtime_meta = torch.nn.Parameter(
            torch.arange(10, 10 + zch_size, device=device).reshape(zch_size, -1),
            requires_grad=False,
        )
        self.assertTrue(hash_zch_mc1._hash_zch_runtime_meta.size(0), zch_size)
        hash_zch_mc1._hash_zch_identities = torch.nn.Parameter(
            torch.tensor([50, 100, 999, 300, 400, 500, 600, -1], device=device),
            requires_grad=False,
        )
        hash_zch_mc2 = HashZchManagedCollisionModule(
            zch_size=zch_size,
            device=device,
            name="t2",
            total_num_buckets=total_num_buckets,
        )
        self.assertIsNone(hash_zch_mc2._hash_zch_runtime_meta)
        mc_modules_dict = {
            "t1": cast(
                ManagedCollisionModule,
                hash_zch_mc1,
            ),
            "t2": cast(
                ManagedCollisionModule,
                hash_zch_mc2,
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules_dict,
            embedding_configs=embedding_configs,
        )
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor(
                [100, 200, 300, 400, 500], dtype=torch.int64, device=device
            ),
            lengths=torch.tensor([2, 2, 1], dtype=torch.int64, device=device),
            weights=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        )
        remapped_ids = KeyedJaggedTensor.from_lengths_sync(
            keys=kjt.keys(),
            values=torch.tensor([1, 2, 3, 4, 0], dtype=torch.int64, device=device),
            lengths=kjt.lengths(),
            weights=kjt.weights(),
        )

        runtime_meta = mcc.lookup_runtime_meta(kjt, remapped_ids)
        self.assertIsNotNone(runtime_meta)
        self.assertEqual(runtime_meta.keys(), kjt.keys())
        self.assertTrue(
            torch.equal(
                runtime_meta.values(),
                torch.tensor([11, 0, 13, 14, -1], dtype=torch.int64, device=device),
            )
        )
        self.assertTrue(torch.equal(runtime_meta.lengths(), kjt.lengths()))
        self.assertTrue(torch.equal(runtime_meta.weights(), kjt.weights()))
