#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from parameterized import parameterized
from torchrec.distributed.model_tracker.delta_store import (
    _compute_unique_rows,
    DeltaStoreTrec,
)
from torchrec.distributed.model_tracker.types import (
    IndexedLookup,
    UniqueRows,
    UpdateMode,
)


class DeltaStoreTrecTest(unittest.TestCase):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)

    @dataclass
    class AppendDeleteTestParams:
        # input parameters
        table_fqn_to_lookups: Dict[str, List[IndexedLookup]]
        up_to_idx: Optional[int]
        # expected output parameters
        deleted_table_fqn_to_lookups: Dict[str, List[IndexedLookup]]

    @parameterized.expand(
        [
            (
                "empty_lookups",
                AppendDeleteTestParams(
                    table_fqn_to_lookups={},
                    up_to_idx=None,
                    deleted_table_fqn_to_lookups={},
                ),
            ),
            (
                "delete_all_lookups",
                AppendDeleteTestParams(
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1]),
                                states=torch.tensor([1]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([2]),
                                states=torch.tensor([2]),
                            ),
                        ]
                    },
                    up_to_idx=None,
                    deleted_table_fqn_to_lookups={},
                ),
            ),
            (
                "single_table_with_idx_from_start",
                AppendDeleteTestParams(
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1]),
                                states=torch.tensor([1]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([2]),
                                states=torch.tensor([2]),
                            ),
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([3]),
                                states=torch.tensor([3]),
                            ),
                            IndexedLookup(
                                batch_idx=4,
                                ids=torch.tensor([4]),
                                states=torch.tensor([4]),
                            ),
                        ]
                    },
                    up_to_idx=3,
                    deleted_table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([3]),
                                states=torch.tensor([3]),
                            ),
                            IndexedLookup(
                                batch_idx=4,
                                ids=torch.tensor([4]),
                                states=torch.tensor([4]),
                            ),
                        ]
                    },
                ),
            ),
            (
                "single_table_with_idx_x",
                AppendDeleteTestParams(
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=8,
                                ids=torch.tensor([8]),
                                states=torch.tensor([8]),
                            ),
                            IndexedLookup(
                                batch_idx=10,
                                ids=torch.tensor([10]),
                                states=torch.tensor([10]),
                            ),
                            IndexedLookup(
                                batch_idx=13,
                                ids=torch.tensor([13]),
                                states=torch.tensor([13]),
                            ),
                        ]
                    },
                    up_to_idx=13,
                    deleted_table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=13,
                                ids=torch.tensor([13]),
                                states=torch.tensor([13]),
                            ),
                        ]
                    },
                ),
            ),
            (
                "multi_table_with_idx_x",
                AppendDeleteTestParams(
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=9,
                                ids=torch.tensor([9]),
                                states=torch.tensor([9]),
                            ),
                        ],
                        "table_fqn_2": [
                            IndexedLookup(
                                batch_idx=9,
                                ids=torch.tensor([9]),
                                states=torch.tensor([9]),
                            ),
                            IndexedLookup(
                                batch_idx=10,
                                ids=torch.tensor([10]),
                                states=torch.tensor([10]),
                            ),
                        ],
                    },
                    up_to_idx=10,
                    deleted_table_fqn_to_lookups={
                        "table_fqn_1": [],
                        "table_fqn_2": [
                            IndexedLookup(
                                batch_idx=10,
                                ids=torch.tensor([10]),
                                states=torch.tensor([10]),
                            ),
                        ],
                    },
                ),
            ),
        ]
    )
    def test_append_and_delete(
        self, _test_name: str, test_params: AppendDeleteTestParams
    ) -> None:
        delta_store = DeltaStoreTrec()
        for table_fqn, lookup_list in test_params.table_fqn_to_lookups.items():
            for lookup in lookup_list:
                delta_store.append(
                    batch_idx=lookup.batch_idx,
                    fqn=table_fqn,
                    ids=lookup.ids,
                    states=lookup.states,
                )
        # Before deletion, check that the lookups are as expected
        self.assertEqual(
            delta_store.per_fqn_lookups,
            test_params.table_fqn_to_lookups,
        )
        delta_store.delete(test_params.up_to_idx)
        # After deletion, check that the lookups are as expected
        self.assertEqual(
            delta_store.per_fqn_lookups,
            test_params.deleted_table_fqn_to_lookups,
        )

    @dataclass
    class ComputeTestParams:
        # input parameters
        ids: List[torch.Tensor]
        embeddings: Optional[List[torch.Tensor]]
        updateMode: UpdateMode
        # expected output parameters
        expected_output: UniqueRows
        expect_assert: bool

    @parameterized.expand(
        [
            # test cases for UpdateMode.NONE
            (
                "unique_ids",
                ComputeTestParams(
                    ids=[
                        torch.tensor([1, 2, 3, 4, 5]),
                        torch.tensor([6, 7, 8, 9, 10]),
                    ],
                    embeddings=None,
                    updateMode=UpdateMode.NONE,
                    expected_output=UniqueRows(
                        ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        states=None,
                    ),
                    expect_assert=False,
                ),
            ),
            (
                "duplicate_ids",
                ComputeTestParams(
                    ids=[
                        torch.tensor([4, 1, 3, 6, 5, 2]),
                        torch.tensor([2, 10, 8, 4, 9, 7]),
                    ],
                    embeddings=None,
                    updateMode=UpdateMode.NONE,
                    expected_output=UniqueRows(
                        ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        states=None,
                    ),
                    expect_assert=False,
                ),
            ),
            # test case for UpdateMode.NONE with embeddings (should assert)
            (
                "none_mode_with_embeddings",
                ComputeTestParams(
                    ids=[
                        torch.tensor([1, 2, 3]),
                        torch.tensor([4, 5, 6]),
                    ],
                    embeddings=[
                        torch.tensor([[1.0], [2.0], [3.0]]),
                        torch.tensor([[4.0], [5.0], [6.0]]),
                    ],
                    updateMode=UpdateMode.NONE,
                    expected_output=UniqueRows(
                        ids=torch.tensor([]),
                        states=None,
                    ),
                    expect_assert=True,
                ),
            ),
            # test cases for UpdateMode.FIRST
            (
                "first_mode_without_embeddings",
                ComputeTestParams(
                    ids=[
                        torch.tensor([1, 2, 3]),
                        torch.tensor([4, 5, 6]),
                    ],
                    embeddings=None,
                    updateMode=UpdateMode.FIRST,
                    expected_output=UniqueRows(
                        ids=torch.tensor([]),
                        states=None,
                    ),
                    expect_assert=True,
                ),
            ),
            (
                "first_mode_unique_ids",
                ComputeTestParams(
                    ids=[
                        torch.tensor([1, 2, 3]),
                        torch.tensor([4, 5, 6]),
                    ],
                    embeddings=[
                        torch.tensor([[1.0], [2.0], [3.0]]),
                        torch.tensor([[4.0], [5.0], [6.0]]),
                    ],
                    updateMode=UpdateMode.FIRST,
                    expected_output=UniqueRows(
                        ids=torch.tensor([1, 2, 3, 4, 5, 6]),
                        states=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
                    ),
                    expect_assert=False,
                ),
            ),
            (
                "first_mode_duplicate_ids",
                ComputeTestParams(
                    ids=[
                        torch.tensor([4, 1, 3, 6, 5, 2]),
                        torch.tensor([2, 10, 8, 4, 9, 7]),
                    ],
                    embeddings=[
                        torch.tensor([[40.0], [10.0], [30.0], [60.0], [50.0], [20.0]]),
                        torch.tensor([[25.0], [100.0], [80.0], [45.0], [90.0], [70.0]]),
                    ],
                    updateMode=UpdateMode.FIRST,
                    expected_output=UniqueRows(
                        ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        # First occurrence of each ID is kept
                        states=torch.tensor(
                            [
                                [10.0],
                                [20.0],
                                [30.0],
                                [40.0],
                                [50.0],
                                [60.0],
                                [70.0],
                                [80.0],
                                [90.0],
                                [100.0],
                            ]
                        ),
                    ),
                    expect_assert=False,
                ),
            ),
            # test cases for UpdateMode.LAST
            (
                "last_mode_without_embeddings",
                ComputeTestParams(
                    ids=[
                        torch.tensor([1, 2, 3]),
                        torch.tensor([4, 5, 6]),
                    ],
                    embeddings=None,
                    updateMode=UpdateMode.LAST,
                    expected_output=UniqueRows(
                        ids=torch.tensor([]),
                        states=None,
                    ),
                    expect_assert=True,
                ),
            ),
            (
                "last_mode_unique_ids",
                ComputeTestParams(
                    ids=[
                        torch.tensor([1, 2, 3]),
                        torch.tensor([4, 5, 6]),
                    ],
                    embeddings=[
                        torch.tensor([[1.0], [2.0], [3.0]]),
                        torch.tensor([[4.0], [5.0], [6.0]]),
                    ],
                    updateMode=UpdateMode.LAST,
                    expected_output=UniqueRows(
                        ids=torch.tensor([1, 2, 3, 4, 5, 6]),
                        states=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
                    ),
                    expect_assert=False,
                ),
            ),
            (
                "last_mode_duplicate_ids",
                ComputeTestParams(
                    ids=[
                        torch.tensor([4, 1, 3, 6, 5, 2]),
                        torch.tensor([2, 10, 8, 4, 9, 7]),
                    ],
                    embeddings=[
                        torch.tensor([[40.0], [10.0], [30.0], [60.0], [50.0], [20.0]]),
                        torch.tensor([[25.0], [100.0], [80.0], [45.0], [90.0], [70.0]]),
                    ],
                    updateMode=UpdateMode.LAST,
                    expected_output=UniqueRows(
                        ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        # Last occurrence of each ID is kept
                        states=torch.tensor(
                            [
                                [10.0],
                                [25.0],
                                [30.0],
                                [45.0],
                                [50.0],
                                [60.0],
                                [70.0],
                                [80.0],
                                [90.0],
                                [100.0],
                            ]
                        ),
                    ),
                    expect_assert=False,
                ),
            ),
        ]
    )
    def test_compute_unique_rows(
        self, _test_name: str, test_params: ComputeTestParams
    ) -> None:
        if test_params.expect_assert:
            # If we expect an assertion error, check that it's raised
            with self.assertRaises(AssertionError):
                _compute_unique_rows(
                    test_params.ids, test_params.embeddings, test_params.updateMode
                )
        else:
            # Otherwise, proceed with the normal test
            result = _compute_unique_rows(
                test_params.ids, test_params.embeddings, test_params.updateMode
            )

            self.assertTrue(torch.equal(result.ids, test_params.expected_output.ids))
            self.assertTrue(
                torch.equal(
                    (result.states if result.states is not None else torch.empty(0)),
                    (
                        test_params.expected_output.states
                        if test_params.expected_output.states is not None
                        else torch.empty(0)
                    ),
                )
            )

    @dataclass
    class CompactTestParams:
        # input parameters
        updateMode: UpdateMode
        table_fqn_to_lookups: Dict[str, List[IndexedLookup]]
        start_idx: int
        end_idx: int
        # expected output parameters
        expected_delta: Dict[str, UniqueRows]
        expect_assert: bool = False

    @parameterized.expand(
        [
            # Test case for compaction with UpdateMode.NONE
            (
                "empty_lookups",
                CompactTestParams(
                    updateMode=UpdateMode.NONE,
                    table_fqn_to_lookups={},
                    start_idx=1,
                    end_idx=5,
                    expected_delta={},
                ),
            ),
            (
                "single_lookup_no_compaction",
                CompactTestParams(
                    updateMode=UpdateMode.NONE,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([1, 2, 3]),
                                states=None,
                            ),
                        ]
                    },
                    start_idx=1,
                    end_idx=5,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3]),
                            states=None,
                        ),
                    },
                ),
            ),
            (
                "multi_lookup_all_unique",
                CompactTestParams(
                    updateMode=UpdateMode.NONE,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=None,
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([4, 5, 6]),
                                states=None,
                            ),
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([7, 8, 9]),
                                states=None,
                            ),
                        ]
                    },
                    start_idx=1,
                    end_idx=3,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            states=None,
                        ),
                    },
                ),
            ),
            (
                "multi_lookup_with_duplicates",
                CompactTestParams(
                    updateMode=UpdateMode.NONE,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=None,
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([3, 4, 5]),
                                states=None,
                            ),
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([5, 6, 7]),
                                states=None,
                            ),
                            IndexedLookup(
                                batch_idx=4,
                                ids=torch.tensor([7, 8, 9]),
                                states=None,
                            ),
                        ]
                    },
                    start_idx=1,
                    end_idx=4,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            states=None,
                        ),
                    },
                ),
            ),
            # Test case for compaction with UpdateMode.FIRST
            (
                "multi_lookup_with_duplicates_first_mode",
                CompactTestParams(
                    updateMode=UpdateMode.FIRST,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=torch.tensor([[10.0], [20.0], [30.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([3, 4, 5]),
                                states=torch.tensor([[35.0], [40.0], [50.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([5, 6, 7]),
                                states=torch.tensor([[55.0], [60.0], [70.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=4,
                                ids=torch.tensor([7, 8, 9]),
                                states=torch.tensor([[75.0], [80.0], [90.0]]),
                            ),
                        ]
                    },
                    start_idx=1,
                    end_idx=4,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            states=torch.tensor(
                                [
                                    [10.0],
                                    [20.0],
                                    [30.0],
                                    [40.0],
                                    [50.0],
                                    [60.0],
                                    [70.0],
                                    [80.0],
                                    [90.0],
                                ]
                            ),
                        ),
                    },
                ),
            ),
            (
                "multiple_tables_first_mode",
                CompactTestParams(
                    updateMode=UpdateMode.FIRST,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=torch.tensor([[10.0], [20.0], [30.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([3, 4, 5]),
                                states=torch.tensor([[35.0], [40.0], [50.0]]),
                            ),
                        ],
                        "table_fqn_2": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([10, 20, 30]),
                                states=torch.tensor([[100.0], [200.0], [300.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([30, 40, 50]),
                                states=torch.tensor([[350.0], [400.0], [500.0]]),
                            ),
                        ],
                    },
                    start_idx=1,
                    end_idx=3,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3, 4, 5]),
                            states=torch.tensor(
                                [[10.0], [20.0], [30.0], [40.0], [50.0]]
                            ),
                        ),
                        "table_fqn_2": UniqueRows(
                            ids=torch.tensor([10, 20, 30, 40, 50]),
                            states=torch.tensor(
                                [[100.0], [200.0], [300.0], [400.0], [500.0]]
                            ),
                        ),
                    },
                ),
            ),
            # Test case for compaction with UpdateMode.LAST
            (
                "multi_lookup_with_duplicates_last_mode",
                CompactTestParams(
                    updateMode=UpdateMode.LAST,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=torch.tensor([[10.0], [20.0], [30.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([3, 4, 5]),
                                states=torch.tensor([[35.0], [40.0], [50.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=3,
                                ids=torch.tensor([5, 6, 7]),
                                states=torch.tensor([[55.0], [60.0], [70.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=4,
                                ids=torch.tensor([7, 8, 9]),
                                states=torch.tensor([[75.0], [80.0], [90.0]]),
                            ),
                        ]
                    },
                    start_idx=1,
                    end_idx=4,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            states=torch.tensor(
                                [
                                    [10.0],
                                    [20.0],
                                    [35.0],
                                    [40.0],
                                    [55.0],
                                    [60.0],
                                    [75.0],
                                    [80.0],
                                    [90.0],
                                ]
                            ),
                        ),
                    },
                ),
            ),
            (
                "multiple_tables_last_mode",
                CompactTestParams(
                    updateMode=UpdateMode.LAST,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=torch.tensor([[10.0], [20.0], [30.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([3, 4, 5]),
                                states=torch.tensor([[35.0], [40.0], [50.0]]),
                            ),
                        ],
                        "table_fqn_2": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([10, 20, 30]),
                                states=torch.tensor([[100.0], [200.0], [300.0]]),
                            ),
                            IndexedLookup(
                                batch_idx=2,
                                ids=torch.tensor([30, 40, 50]),
                                states=torch.tensor([[350.0], [400.0], [500.0]]),
                            ),
                        ],
                    },
                    start_idx=1,
                    end_idx=3,
                    expected_delta={
                        "table_fqn_1": UniqueRows(
                            ids=torch.tensor([1, 2, 3, 4, 5]),
                            states=torch.tensor(
                                [[10.0], [20.0], [35.0], [40.0], [50.0]]
                            ),
                        ),
                        "table_fqn_2": UniqueRows(
                            ids=torch.tensor([10, 20, 30, 40, 50]),
                            states=torch.tensor(
                                [[100.0], [200.0], [350.0], [400.0], [500.0]]
                            ),
                        ),
                    },
                ),
            ),
            # Test case for invalid start_idx and end_idx
            (
                "invalid_indices",
                CompactTestParams(
                    updateMode=UpdateMode.NONE,
                    table_fqn_to_lookups={
                        "table_fqn_1": [
                            IndexedLookup(
                                batch_idx=1,
                                ids=torch.tensor([1, 2, 3]),
                                states=None,
                            ),
                        ]
                    },
                    start_idx=5,
                    end_idx=3,
                    expected_delta={},
                    expect_assert=True,
                ),
            ),
        ]
    )
    def test_compact(self, _test_name: str, test_params: CompactTestParams) -> None:
        """
        Test the compact method of DeltaStore.
        """
        # Create a DeltaStoreTrec with the specified updateMode
        delta_store = DeltaStoreTrec(updateMode=test_params.updateMode)

        # Populate the DeltaStore with the test lookups
        for table_fqn, lookup_list in test_params.table_fqn_to_lookups.items():
            for lookup in lookup_list:
                delta_store.append(
                    batch_idx=lookup.batch_idx,
                    fqn=table_fqn,
                    ids=lookup.ids,
                    states=lookup.states,
                )
        if test_params.expect_assert:
            # If we expect an assertion error, check that it's raised
            with self.assertRaises(AssertionError):
                delta_store.compact(
                    start_idx=test_params.start_idx, end_idx=test_params.end_idx
                )
        else:
            # Call the compact method
            delta_store.compact(
                start_idx=test_params.start_idx, end_idx=test_params.end_idx
            )
            # Verify the result using get_unique method
            delta_result = delta_store.get_unique()

            # compare all fqns in the result
            for table_fqn, delta_rows in test_params.expected_delta.items():
                # Comparing ids
                self.assertTrue(delta_result[table_fqn].ids.allclose(delta_rows.ids))
                # Comparing embeddings/states
                if (
                    delta_rows.states is not None
                    and delta_result[table_fqn].states is not None
                ):
                    self.assertTrue(
                        # pyre-ignore
                        delta_result[table_fqn].states.allclose(delta_rows.states)
                    )
