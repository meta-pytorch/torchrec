#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Optional, Tuple

from torchrec.distributed._collective_tag import (
    _collective_tag_from,
    _COLLECTIVE_TAG_MAX_BYTES,
)


class CollectiveTagFromTest(unittest.TestCase):
    # Single-process unit tests for the _collective_tag_from helper.
    _INT32_MAX = 0x7FFFFFFF

    def test_empty_parts_fits_signed_int32(self) -> None:
        # Empty parts must still produce a value that fits signed int32.
        # blake2b returns a 4-byte digest; we mask to int31.
        tag = _collective_tag_from()
        self.assertGreaterEqual(tag, 0)
        self.assertLessEqual(tag, self._INT32_MAX)

    def test_various_parts_all_fit_signed_int32(self) -> None:
        # Sanity check: the tag shapes both production call sites use
        # produce non-negative int32 values. Splits are passed raw with
        # length_limit=None so their full value is hashed.
        _CAP = _COLLECTIVE_TAG_MAX_BYTES
        # (parts, per_part_length_limits)
        cases: List[Tuple[Tuple[object, ...], Optional[List[Optional[int]]]]] = [
            # KJTAllToAllSplits: (name, keys, splits) with splits uncapped
            (
                ("KJTAllToAllSplits", ["f0", "f1"], (1, 1)),
                [_CAP, _CAP, None],
            ),
            # FusedKJTListSplits flat shape: (name, count, "has_splits",
            # splits, tensor count, keys, ..., "no_splits", ...).
            (
                (
                    "FusedKJTListSplits",
                    3,
                    "has_splits",
                    (1, 1),
                    2,
                    ("f0", "f1"),
                    "no_splits",
                    "has_splits",
                    (1,),
                    1,
                    ("f2",),
                ),
                [_CAP, _CAP, _CAP, None, _CAP, _CAP, _CAP, _CAP, None, _CAP, _CAP],
            ),
            # Boundary: empty keys, empty splits
            (("KJTAllToAllSplits", [], ()), [_CAP, _CAP, None]),
            # Boundary: fused with zero requests
            (("FusedKJTListSplits", 0), None),
            # Long single string, guards against a length-related bug in the hash
            (("x" * 1024,), None),
        ]
        for parts, budgets in cases:
            with self.subTest(parts=parts):
                tag = _collective_tag_from(*parts, per_part_length_limits=budgets)
                self.assertGreaterEqual(tag, 0)
                self.assertLessEqual(tag, self._INT32_MAX)

    def test_intra_list_element_boundary_is_unambiguous(self) -> None:
        # List elements are length-prefixed, so ["a,b", "c"] and
        # ["a", "b,c"] must produce different tags.
        self.assertNotEqual(
            _collective_tag_from("X", ["a,b", "c"]),
            _collective_tag_from("X", ["a", "b,c"]),
        )

    def test_doubly_nested_list_framing_disambiguates(self) -> None:
        # Length-prefixing works through nested containers, so
        # [[a, b], [c]] and [[a], [b, c]] must produce different tags.
        self.assertNotEqual(
            _collective_tag_from("X", [["a", "b"], ["c"]]),
            _collective_tag_from("X", [["a"], ["b", "c"]]),
        )

    def test_separator_is_unambiguous(self) -> None:
        # Parts are joined by NUL, which is safe because tag parts never
        # contain \x00. Regression guard: if we ever used a printable
        # separator, ("a", "b") and ("a,b") would produce the same bytes
        # and hash to the same tag.
        self.assertNotEqual(
            _collective_tag_from("a", "b"),
            _collective_tag_from("a,b"),
        )

    def test_length_divergence_detected(self) -> None:
        # Lists of different lengths must produce different tags.
        self.assertNotEqual(
            _collective_tag_from("X", ["a"] * 10),
            _collective_tag_from("X", ["a"] * 11),
        )

    def test_small_list_element_change_detected_via_iteration(self) -> None:
        # When the list is small enough to fit in the length limit,
        # any element change produces a different tag.
        self.assertNotEqual(
            _collective_tag_from("X", ["a"] * 10 + ["b"]),
            _collective_tag_from("X", ["a"] * 10 + ["c"]),
        )

    def test_large_input_stays_in_int32_and_stays_deterministic(self) -> None:
        # Even with a huge input, the tag stays in int32 range and
        # is stable across repeated calls.
        big = ["k"] * 100_000
        tag_1 = _collective_tag_from("X", big)
        tag_2 = _collective_tag_from("X", big)
        self.assertEqual(tag_1, tag_2)
        self.assertGreaterEqual(tag_1, 0)
        self.assertLessEqual(tag_1, self._INT32_MAX)

    def test_length_still_distinguished_beyond_cap(self) -> None:
        # Two lists of different lengths must produce different tags
        # even when both are past the length limit.
        self.assertNotEqual(
            _collective_tag_from("X", ["k"] * 100_000),
            _collective_tag_from("X", ["k"] * 200_000),
        )

    def test_past_cap_middle_divergence_documented_collision(self) -> None:
        # Documented blind spot: two lists that match up to the length
        # limit produce the same tag, even if they differ past it.
        a = ["padding"] * 10_000 + ["same_middle"] + ["padding"] * 9999 + ["tail"]
        b = ["padding"] * 10_000 + ["diff_middle"] + ["padding"] * 9999 + ["tail"]
        self.assertEqual(
            _collective_tag_from("X", a),
            _collective_tag_from("X", b),
        )

    def test_huge_scalar_string_and_bytes_bounded_allocation(self) -> None:
        # A huge str or bytes gets truncated to the length limit, so it
        # hashes to the same tag as its shorter prefix.
        huge_str = "x" * 1_000_000
        huge_bytes = b"x" * 1_000_000
        prefix_str = "x" * _COLLECTIVE_TAG_MAX_BYTES
        prefix_bytes = b"x" * _COLLECTIVE_TAG_MAX_BYTES
        self.assertEqual(
            _collective_tag_from("X", huge_str),
            _collective_tag_from("X", prefix_str),
        )
        self.assertEqual(
            _collective_tag_from("X", huge_bytes),
            _collective_tag_from("X", prefix_bytes),
        )

    def test_fused_flat_shape_high_n_middle_splits_divergence(self) -> None:
        # Each request's splits are passed with length_limit=None, so a
        # change in one request's splits values is caught even when the
        # surrounding keys are very wide.
        _CAP = _COLLECTIVE_TAG_MAX_BYTES
        huge_keys = tuple(f"feat_{i}" for i in range(5000))
        splits_a = (2,) * 64 + (1,) * 63 + (2,)
        splits_b = (2,) * 64 + (3,) * 63 + (2,)

        def _flat(
            divergent_at_7: Tuple[int, ...],
        ) -> Tuple[Tuple[object, ...], List[Optional[int]]]:
            parts: List[object] = ["FusedKJTListSplits", 15]
            budgets: List[Optional[int]] = [_CAP, _CAP]
            for j in range(15):
                splits = divergent_at_7 if j == 7 else splits_a
                parts.extend(("has_splits", splits, 1, huge_keys))
                budgets.extend([_CAP, None, _CAP, _CAP])
            return tuple(parts), budgets

        parts_a, budgets_a = _flat(splits_a)
        parts_b, budgets_b = _flat(splits_b)
        self.assertNotEqual(
            _collective_tag_from(*parts_a, per_part_length_limits=budgets_a),
            _collective_tag_from(*parts_b, per_part_length_limits=budgets_b),
        )

    def test_fused_flat_shape_reordered_same_type_requests(self) -> None:
        # Two requests with the same splits but different keys must
        # produce different tags depending on their order.
        _CAP = _COLLECTIVE_TAG_MAX_BYTES
        splits = (2, 2, 2, 2)
        req_a = ("has_splits", splits, 1, tuple(f"a{i}" for i in range(256)))
        req_b = ("has_splits", splits, 1, tuple(f"b{i}" for i in range(256)))
        budgets = [_CAP, _CAP] + [_CAP, None, _CAP, _CAP] * 2
        self.assertNotEqual(
            _collective_tag_from(
                "FusedKJTListSplits",
                2,
                *req_a,
                *req_b,
                per_part_length_limits=budgets,
            ),
            _collective_tag_from(
                "FusedKJTListSplits",
                2,
                *req_b,
                *req_a,
                per_part_length_limits=budgets,
            ),
        )

    def test_rw_bucketized_keys_do_not_hide_splits(self) -> None:
        # Realistic RW-bucketized shape: 78 features × 256 world_size
        # replicas. Because splits are passed with length_limit=None,
        # a change in a single split value is caught even with a very
        # wide keys list.
        _CAP = _COLLECTIVE_TAG_MAX_BYTES
        bucketized_keys = [f"feat_{i}" for i in range(78)] * 256
        splits_a = (78,) * 256
        splits_b = (78,) * 128 + (77,) + (78,) * 127
        self.assertNotEqual(
            _collective_tag_from(
                "KJTAllToAllSplits",
                bucketized_keys,
                splits_a,
                per_part_length_limits=[_CAP, _CAP, None],
            ),
            _collective_tag_from(
                "KJTAllToAllSplits",
                bucketized_keys,
                splits_b,
                per_part_length_limits=[_CAP, _CAP, None],
            ),
        )

    def test_tower_zero_heavy_split_destination_divergence(self) -> None:
        # Cross-PG routing shape: mostly-zero splits with a nonzero
        # entry that moves position. Different positions must produce
        # different tags.
        _CAP = _COLLECTIVE_TAG_MAX_BYTES
        keys = [f"k{i}" for i in range(256)]
        splits_a = (0, 0, 32, 0, 0, 0, 0, 0)
        splits_b = (0, 0, 0, 32, 0, 0, 0, 0)
        self.assertNotEqual(
            _collective_tag_from(
                "KJTAllToAllSplits",
                keys,
                splits_a,
                per_part_length_limits=[_CAP, _CAP, None],
            ),
            _collective_tag_from(
                "KJTAllToAllSplits",
                keys,
                splits_b,
                per_part_length_limits=[_CAP, _CAP, None],
            ),
        )

    def test_per_part_length_limits_none_slot_fully_hashes(self) -> None:
        # A slot with limit=None hashes the entire part. This is what
        # protects splits from truncation. Put the divergent value near
        # the end of a long tuple so the default 32K-byte cap can't
        # reach it: at 3 bytes/element under length-prefix framing, the
        # cap covers ~10,900 elements.
        n = 50000
        splits_a = (2,) * (n - 1) + (1,)
        splits_b = (2,) * (n - 1) + (3,)
        # With uncapped slot: caught.
        self.assertNotEqual(
            _collective_tag_from("X", splits_a, per_part_length_limits=[None, None]),
            _collective_tag_from("X", splits_b, per_part_length_limits=[None, None]),
        )
        # Without uncapped slot: NOT caught (default cap truncates
        # before reaching the divergent value at the end).
        self.assertEqual(
            _collective_tag_from("X", splits_a),
            _collective_tag_from("X", splits_b),
        )

    def test_per_part_length_limits_length_must_match_parts(self) -> None:
        # Passing a budgets list of the wrong length is a programming
        # error and should fail loudly.
        with self.assertRaises(AssertionError):
            _collective_tag_from(
                "X",
                "Y",
                per_part_length_limits=[_COLLECTIVE_TAG_MAX_BYTES],
            )
