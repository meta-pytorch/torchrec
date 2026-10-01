#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
from torchrec.modules.tensor_pool import TensorPool


class TensorPoolTest(unittest.TestCase):
    def test_update_lookup(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )
        pool_size = 10
        dim = 3
        batch_size = 5
        dense_pool = TensorPool(
            pool_size=pool_size,
            dim=dim,
            dtype=torch.float,
            device=device,
        )
        update_ids = [1, 9, 4, 2, 6]
        ids_to_row = {1: 0, 9: 1, 4: 2, 2: 3, 6: 4}
        ids = torch.tensor(update_ids, dtype=torch.int, device=device)
        reference_values = torch.rand(
            (batch_size, dim), dtype=torch.float, device=device
        )
        dense_pool.update(ids=ids, values=reference_values)

        lookup_ids = torch.randint(
            low=0, high=pool_size, size=(batch_size,), dtype=torch.int, device=device
        )
        lookup_values = dense_pool.lookup(ids=lookup_ids)
        for i in range(batch_size):
            if lookup_ids[i] in update_ids:
                # pyrefly: ignore[bad-assignment]
                lookup_id: int = lookup_ids[i].int().item()
                torch.testing.assert_close(
                    reference_values[ids_to_row[lookup_id]],
                    lookup_values[i],
                )
            else:
                torch.testing.assert_close(
                    lookup_values[i],
                    torch.zeros(3, dtype=torch.float, device=device),
                    msg=f"{dense_pool._pool[lookup_ids[i]]}",
                )

        torch.testing.assert_close(dense_pool.pool_size, pool_size)

    def test_update_unique(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )
        pool_size = 10
        dim = 3
        batch_size = 5
        dense_pool = TensorPool(
            pool_size=pool_size,
            dim=dim,
            dtype=torch.float,
            device=device,
        )
        update_ids = [1, 9, 4, 2, 6]
        ids = torch.tensor(update_ids, dtype=torch.int, device=device)
        reference_values = torch.rand(
            (batch_size, dim), dtype=torch.float, device=device
        )
        dense_pool.update_unique(ids=ids, values=reference_values)

        torch.testing.assert_close(dense_pool.lookup(ids=ids), reference_values)

        untouched_ids = torch.tensor(
            [i for i in range(pool_size) if i not in update_ids],
            dtype=torch.int,
            device=device,
        )
        torch.testing.assert_close(
            dense_pool.lookup(ids=untouched_ids),
            torch.zeros((untouched_ids.numel(), dim), dtype=torch.float, device=device),
        )

    def test_update_validation_parity(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )
        pool_size = 10
        dim = 3
        batch_size = 5
        dense_pool = TensorPool(
            pool_size=pool_size,
            dim=dim,
            dtype=torch.float,
            device=device,
        )
        ids = torch.tensor([1, 9, 4, 2, 6], dtype=torch.int, device=device)
        values = torch.rand((batch_size, dim), dtype=torch.float, device=device)

        invalid_args = [
            ("values_not_2d", ids, values.flatten()),
            ("values_wrong_dim", ids, values[:, :-1]),
            ("values_wrong_dtype", ids, values.double()),
            ("ids_wrong_dtype", ids.float(), values),
            ("ids_out_of_bounds", ids + pool_size, values),
        ]
        if torch.cuda.is_available():
            invalid_args += [
                ("values_wrong_device", ids, values.cpu()),
                ("ids_wrong_device", ids.cpu(), values),
            ]

        updates = {
            "update": dense_pool.update,
            "update_unique": dense_pool.update_unique,
        }
        for case, bad_ids, bad_values in invalid_args:
            for method, update in updates.items():
                with self.subTest(case=case, method=method):
                    with self.assertRaises(AssertionError):
                        update(ids=bad_ids, values=bad_values)

    def test_conflict(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )
        pool_size = 10
        dim = 3
        batch_size = 5
        dense_pool = TensorPool(
            pool_size=pool_size,
            dim=dim,
            dtype=torch.float,
            device=device,
        )
        update_ids = [1, 9, 4, 1, 6]
        ids_to_row = {9: 1, 4: 2, 1: 3, 6: 4}  # The first 1 is deduped and removed
        ids = torch.tensor(update_ids, dtype=torch.int, device=device)
        reference_values = torch.rand(
            (batch_size, dim), dtype=torch.float, device=device
        )
        dense_pool.update(ids=ids, values=reference_values)

        lookup_ids = torch.randint(
            low=0, high=pool_size, size=(batch_size,), dtype=torch.int, device=device
        )
        lookup_values = dense_pool.lookup(ids=lookup_ids)
        for i in range(batch_size):
            if lookup_ids[i] in update_ids:
                # pyrefly: ignore[bad-assignment]
                lookup_id: int = lookup_ids[i].int().item()
                torch.testing.assert_close(
                    reference_values[ids_to_row[lookup_id]],
                    lookup_values[i],
                )
            else:
                torch.testing.assert_close(
                    lookup_values[i],
                    torch.zeros(3, dtype=torch.float, device=device),
                    msg=f"{dense_pool._pool[lookup_ids[i]]}",
                )

        torch.testing.assert_close(dense_pool.pool_size, pool_size)
