#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for sharded_relay_utils.py.

These tests run on CPU with no real GPU, NCCL, or RCCLX stack.  All
distributed calls and the FusedShardedRelayMultiGroup are replaced with
unittest.mock objects so tests are fast and hermetic.

Test classes
============
FlatCacheTest
    Tests for the grow-only active flat buffer cache (_active_flat_cache) and
    the shared helper placeholder cache (_placeholder_cache) that replaced the
    old per-tensor scratch scheme.

FlatAllreduceTest
    Tests for allreduce_tensors_with_sharded_relay with the flat-concat
    approach: N tensors → pack into flat buf → ONE call → unpack.

FusedShardedRelayValidationTest
    Tests for FusedShardedRelayMultiGroup.allreduce_multi_group validation
    logic (tensor size mismatch, count=0 skipping).  These test the kernel
    API directly and are unchanged by the flat-concat rewrite.
"""

from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torchrec.distributed.sharded_relay_utils import (
    _get_active_flat_buf,
    _get_active_output_flat_buf,
    _get_placeholder_buf,
    all_gather_tensors_with_sharded_relay,
    all_to_all_tensors_with_sharded_relay,
    allreduce_tensors_with_sharded_relay,
    reduce_scatter_tensors_with_sharded_relay,
    ShardedRelayState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cpu")


def _make_state(
    rank: int = 0,
    sparse_group_size: int = 2,
    local_size: int = 8,
) -> ShardedRelayState:
    """
    Build a ShardedRelayState suitable for CPU-only unit tests.

    - intra_node_pytorch_pg=None triggers the fallback path in
      allreduce_tensors_with_sharded_relay (no dist.all_gather calls).
    - fused is a MagicMock; allreduce_multi_group records every call.
    """
    num_sparse_groups = local_size // sparse_group_size
    active_ranks = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
        for g in range(num_sparse_groups)
    ]
    mock_fused = MagicMock()
    mock_fused.allreduce_multi_group = MagicMock()
    return ShardedRelayState(
        fused=mock_fused,
        intra_node_pytorch_pg=None,
        local_rank=rank,
        sparse_group_size=sparse_group_size,
        my_sparse_group=rank // sparse_group_size,
        num_sparse_groups=num_sparse_groups,
        local_size=local_size,
        precomputed_active_ranks=active_ranks,
        _rcclx_comm=None,
    )


# ---------------------------------------------------------------------------
# Tests for the flat buffer caches
# ---------------------------------------------------------------------------


class FlatCacheTest(unittest.TestCase):
    """Tests for _get_active_flat_buf and _get_placeholder_buf."""

    # --- _get_active_flat_buf ---

    def test_active_flat_buf_exact_size_on_first_call(self) -> None:
        state = _make_state()
        buf = _get_active_flat_buf(state, 100, torch.float32, _DEVICE)
        self.assertEqual(buf.numel(), 100)
        self.assertEqual(buf.dtype, torch.float32)

    def test_active_flat_buf_reused_when_same_size(self) -> None:
        state = _make_state()
        buf1 = _get_active_flat_buf(state, 100, torch.float32, _DEVICE)
        buf2 = _get_active_flat_buf(state, 100, torch.float32, _DEVICE)
        self.assertEqual(buf1.data_ptr(), buf2.data_ptr())

    def test_active_flat_buf_narrowed_view_when_size_shrinks(self) -> None:
        state = _make_state()
        big = _get_active_flat_buf(state, 1000, torch.float32, _DEVICE)
        small = _get_active_flat_buf(state, 500, torch.float32, _DEVICE)
        self.assertEqual(small.numel(), 500)
        # narrow() shares storage with the cached buffer
        self.assertEqual(big.data_ptr(), small.data_ptr())

    def test_active_flat_buf_reallocates_when_size_grows(self) -> None:
        state = _make_state()
        _get_active_flat_buf(state, 100, torch.float32, _DEVICE)
        big = _get_active_flat_buf(state, 200, torch.float32, _DEVICE)
        self.assertEqual(big.numel(), 200)

    def test_active_flat_buf_separate_per_dtype(self) -> None:
        """bf16 (weights) and fp32 (optimizer states) must have separate buffers."""
        state = _make_state()
        bf16 = _get_active_flat_buf(state, 100, torch.bfloat16, _DEVICE)
        fp32 = _get_active_flat_buf(state, 100, torch.float32, _DEVICE)
        # Re-fetching bf16 must return the same buffer (not reallocated).
        bf16_again = _get_active_flat_buf(state, 100, torch.bfloat16, _DEVICE)
        self.assertEqual(bf16.data_ptr(), bf16_again.data_ptr())
        self.assertNotEqual(bf16.data_ptr(), fp32.data_ptr())

    # --- _get_placeholder_buf ---

    def test_placeholder_buf_is_single_element(self) -> None:
        state = _make_state()
        buf = _get_placeholder_buf(state, torch.float32, _DEVICE)
        self.assertEqual(buf.numel(), 1)
        self.assertEqual(buf.dtype, torch.float32)

    def test_placeholder_buf_shared_across_calls(self) -> None:
        """The helper placeholder is ignored by the kernel, so one buffer is
        reused across every helper group and both in/out slots."""
        state = _make_state()
        buf1 = _get_placeholder_buf(state, torch.float32, _DEVICE)
        buf2 = _get_placeholder_buf(state, torch.float32, _DEVICE)
        self.assertEqual(buf1.data_ptr(), buf2.data_ptr())

    def test_placeholder_buf_separate_per_dtype(self) -> None:
        """fp16 (weights) and fp32 (optimizer states) get distinct placeholders."""
        state = _make_state()
        fp16 = _get_placeholder_buf(state, torch.float16, _DEVICE)
        fp32 = _get_placeholder_buf(state, torch.float32, _DEVICE)
        fp16_again = _get_placeholder_buf(state, torch.float16, _DEVICE)
        self.assertEqual(fp16.data_ptr(), fp16_again.data_ptr())
        self.assertNotEqual(fp16.data_ptr(), fp32.data_ptr())


# ---------------------------------------------------------------------------
# Tests for allreduce_tensors_with_sharded_relay (flat-concat approach)
# ---------------------------------------------------------------------------


class FlatAllreduceTest(unittest.TestCase):
    def _call_count(self, state: ShardedRelayState) -> int:
        return state.fused.allreduce_multi_group.call_count

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.allreduce_multi_group.call_args_list

    # ------------------------------------------------------------------
    # Basic call-count correctness
    # ------------------------------------------------------------------

    def test_returns_immediately_when_no_tensors(self) -> None:
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(state, {}, "test")
        self.assertEqual(self._call_count(state), 0)

    def test_single_call_for_one_tensor(self) -> None:
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(100)]}, "test"
        )
        self.assertEqual(self._call_count(state), 1)

    def test_single_call_for_many_tables(self) -> None:
        """101 tensors → exactly ONE allreduce_multi_group call (not 101)."""
        state = _make_state(rank=0)
        tensors = [torch.zeros(100) for _ in range(101)]
        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "test")
        self.assertEqual(self._call_count(state), 1)

    def test_single_call_per_dtype_for_mixed_dicts(self) -> None:
        """Two dtypes in tensors_dict → two calls (one per dtype)."""
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state,
            {
                torch.float16: [torch.zeros(50, dtype=torch.float16)],
                torch.float32: [torch.zeros(50, dtype=torch.float32)],
            },
            "test",
        )
        self.assertEqual(self._call_count(state), 2)

    def test_bm_fm_counts_still_one_call(self) -> None:
        """BM-FM has [94,101,88,99] tensors per group; rank 0 (group 0) has 94."""
        state = _make_state(rank=0)
        tensors = [torch.zeros(100) for _ in range(94)]
        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "test")
        self.assertEqual(self._call_count(state), 1)

    # ------------------------------------------------------------------
    # Out-of-place vs in-place wiring
    # ------------------------------------------------------------------

    def test_out_of_place_active_output_wired_distinctly(self) -> None:
        """output_tensors_dict routes the active group's reduced result to a
        separate destination (out-of-place). The active input and output are
        distinct internal flat buffers; the caller's out buffer receives the
        result via the unpack step while the caller's input is preserved."""
        state = _make_state(rank=0)  # active for group 0
        inp = torch.ones(100)
        out = torch.zeros(100)

        sentinel = 42.0

        def _fill_active_output(*args, **kwargs) -> None:
            # Simulate the reduced result landing in the active output flat buf.
            outs = kwargs["output_tensors"]
            outs[state.my_sparse_group].fill_(sentinel)

        state.fused.allreduce_multi_group.side_effect = _fill_active_output

        allreduce_tensors_with_sharded_relay(
            state,
            {torch.float32: [inp]},
            "test",
            output_tensors_dict={torch.float32: [out]},
        )

        call_kwargs = self._all_calls(state)[0].kwargs
        self.assertIn("output_tensors", call_kwargs)
        output_tensors = call_kwargs["output_tensors"]
        self.assertIsNotNone(output_tensors)

        active = state.my_sparse_group
        active_input = call_kwargs["tensors"][active]
        active_output = output_tensors[active]
        # Active input and output are distinct flat buffers (out-of-place).
        self.assertNotEqual(active_output.data_ptr(), active_input.data_ptr())
        # The result is unpacked into the caller's out buffer...
        self.assertTrue(torch.all(out == sentinel))
        # ...while the caller's input buffer is preserved.
        self.assertTrue(torch.all(inp == 1.0))

    def test_in_place_passes_no_output_tensors(self) -> None:
        """Default (no output_tensors_dict) → in-place: output_tensors is None."""
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(100)]}, "test"
        )
        call_kwargs = self._all_calls(state)[0].kwargs
        self.assertIsNone(call_kwargs.get("output_tensors"))

    # ------------------------------------------------------------------
    # Flat buffer sizing passed to allreduce_multi_group
    # ------------------------------------------------------------------

    def test_active_group_total_numel_matches_sum_of_tensor_sizes(self) -> None:
        """The active group's per_group_sizes entry must equal sum(t.numel())."""
        state = _make_state(rank=0)  # active for group 0
        sizes = [100, 500, 750]
        tensors = [torch.zeros(s) for s in sizes]
        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "test")

        call_kwargs = self._all_calls(state)[0].kwargs
        active_size = call_kwargs["per_group_sizes"][state.my_sparse_group]
        self.assertEqual(active_size, sum(sizes))

    def test_helper_slots_use_shared_placeholder(self) -> None:
        """
        With internal helper scratch, helper groups pass a shared 1-element
        placeholder (the kernel ignores it), while per_group_sizes still carries
        each group's full total for the kernel geometry.
        """
        state = _make_state(rank=0)
        sizes = [100, 500, 750]
        tensors = [torch.zeros(s) for s in sizes]
        expected_total = sum(sizes)  # 1350

        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "test")

        call_kwargs = self._all_calls(state)[0].kwargs
        iter_tensors = call_kwargs["tensors"]
        iter_sizes = call_kwargs["per_group_sizes"]

        # Active group should have full total.
        self.assertEqual(
            iter_sizes[state.my_sparse_group],
            expected_total,
        )
        self.assertEqual(
            iter_tensors[state.my_sparse_group].numel(),
            expected_total,
        )

        # Helper groups: per_group_sizes still carries the full total_g, but the
        # tensor is a shared 1-element placeholder aliased across all helpers.
        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            self.assertEqual(
                iter_sizes[g],
                expected_total,
                f"group={g}: per_group_sizes should be full total",
            )
            self.assertEqual(
                iter_tensors[g].numel(),
                1,
                f"group={g}: helper tensor should be a 1-element placeholder",
            )
            helper_ptrs.add(iter_tensors[g].data_ptr())

        # All helper groups share ONE placeholder buffer (kernel ignores it).
        self.assertEqual(
            len(helper_ptrs),
            1,
            f"Expected 1 shared helper placeholder, got {len(helper_ptrs)}",
        )

    # ------------------------------------------------------------------
    # Values written back to original tensors
    # ------------------------------------------------------------------

    def test_values_written_back_to_original_tensors(self) -> None:
        """
        After allreduce, each original tensor must contain the values that
        the allreduce produced (written from the flat buffer back via unpack).
        We simulate this by having the mock fill active_flat with a sentinel.
        """
        state = _make_state(rank=0)
        my_tensor = torch.zeros(300)

        sentinel = 42.0

        def _fill_active_flat(*args, **kwargs) -> None:
            # Simulate the allreduce result: write sentinel into the active flat buf.
            tensors = kwargs.get("tensors", args[0] if args else [])
            my_g = state.my_sparse_group
            tensors[my_g].fill_(sentinel)

        state.fused.allreduce_multi_group.side_effect = _fill_active_flat

        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [my_tensor]}, "test"
        )

        # The unpack step must have copied sentinel back into my_tensor.
        self.assertTrue(
            torch.all(my_tensor == sentinel),
            f"Expected all values to be {sentinel}, got {my_tensor[:5]}",
        )

    def test_unpack_handles_multiple_tensors_correctly(self) -> None:
        """
        With multiple tensors of different sizes, the unpack step must write
        each tensor's slice of the flat buffer back to the correct original tensor.
        """
        state = _make_state(rank=0)
        t0 = torch.zeros(100)
        t1 = torch.zeros(200)
        t2 = torch.zeros(50)

        fill_values = [1.0, 2.0, 3.0]

        def _fill_by_slice(*args, **kwargs) -> None:
            tensors = kwargs.get("tensors", args[0] if args else [])
            my_g = state.my_sparse_group
            flat = tensors[my_g]
            flat[:100].fill_(fill_values[0])
            flat[100:300].fill_(fill_values[1])
            flat[300:350].fill_(fill_values[2])

        state.fused.allreduce_multi_group.side_effect = _fill_by_slice

        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [t0, t1, t2]}, "test"
        )

        self.assertTrue(torch.all(t0 == fill_values[0]))
        self.assertTrue(torch.all(t1 == fill_values[1]))
        self.assertTrue(torch.all(t2 == fill_values[2]))

    # ------------------------------------------------------------------
    # Metadata cache skips allgather on subsequent calls
    # ------------------------------------------------------------------

    @patch("torchrec.distributed.sharded_relay_utils.dist")
    def test_metadata_cache_skips_allgather_after_first_call(
        self, mock_dist: MagicMock
    ) -> None:
        """
        allgather must be called exactly once per (annotation, dtype) pair,
        regardless of how many training steps have passed.
        """
        state = _make_state(rank=0)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        # Set up the mock allgather to return a count of 200 for all ranks.
        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for t in tensor_list:
                t.fill_(200)

        mock_dist.all_gather.side_effect = _allgather_side_effect
        mock_dist.ReduceOp = dist.ReduceOp

        tensors = [torch.zeros(200, dtype=torch.float32)]

        # First call — should trigger allgather.
        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        # Second call — must use cached metadata, no new allgather.
        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        # Different annotation → new cache entry → one more allgather.
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: tensors}, "other_annotation"
        )
        self.assertEqual(mock_dist.all_gather.call_count, 2)

    # ------------------------------------------------------------------
    # Scratch buffers reused across training steps
    # ------------------------------------------------------------------

    def test_helper_flat_buf_reused_across_training_steps(self) -> None:
        """The helper flat buffer must not be reallocated on subsequent steps."""
        state = _make_state(rank=0)
        tensors = [torch.zeros(100)]

        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "step1")
        call1 = self._all_calls(state)[0].kwargs["tensors"]
        helper_g = 1
        ptr_step1 = call1[helper_g].data_ptr()

        allreduce_tensors_with_sharded_relay(state, {torch.float32: tensors}, "step1")
        call2 = self._all_calls(state)[1].kwargs["tensors"]
        ptr_step2 = call2[helper_g].data_ptr()

        self.assertEqual(
            ptr_step1,
            ptr_step2,
            "Helper flat buffer was reallocated between training steps (should be reused)",
        )

    def test_separate_flat_bufs_for_weights_and_optimizer(self) -> None:
        """
        Alternating between bf16 (weights sync) and fp32 (optimizer sync)
        must not trigger reallocation and must use separate buffers.
        The helper placeholder is keyed by (dtype, device), so each dtype has
        its own placeholder that is reused across calls.
        """
        state = _make_state(rank=0)
        helper_g = 1

        allreduce_tensors_with_sharded_relay(
            state, {torch.float16: [torch.zeros(100, dtype=torch.float16)]}, "weights"
        )
        ptr_fp16_1 = self._all_calls(state)[0].kwargs["tensors"][helper_g].data_ptr()

        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(100, dtype=torch.float32)]}, "opt"
        )
        ptr_fp32_1 = self._all_calls(state)[1].kwargs["tensors"][helper_g].data_ptr()

        # fp16 again — must reuse the fp16 buffer.
        allreduce_tensors_with_sharded_relay(
            state, {torch.float16: [torch.zeros(100, dtype=torch.float16)]}, "weights"
        )
        ptr_fp16_2 = self._all_calls(state)[2].kwargs["tensors"][helper_g].data_ptr()

        self.assertEqual(
            ptr_fp16_1, ptr_fp16_2, "fp16 buffer reallocated on second call"
        )
        self.assertNotEqual(
            ptr_fp16_1, ptr_fp32_1, "fp16 and fp32 share the same buffer"
        )

    # ------------------------------------------------------------------
    # Helper placeholder tests (shared 1-element buffer, kernel ignores it)
    # ------------------------------------------------------------------

    def test_one_fused_call_per_dtype(self) -> None:
        """Drive with one fp32 tensor; assert exactly 1 fused call with num_groups."""
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(100)]}, "test"
        )
        self.assertEqual(self._call_count(state), 1)
        call_kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(call_kwargs["num_groups"], state.num_sparse_groups)

    def test_helper_slots_share_one_placeholder(self) -> None:
        """All helper-group slots alias ONE placeholder; active does NOT alias it."""
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(200)]}, "test"
        )
        call_kwargs = self._all_calls(state)[0].kwargs
        iter_tensors = call_kwargs["tensors"]

        active_ptr = iter_tensors[state.my_sparse_group].data_ptr()
        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g != state.my_sparse_group:
                helper_ptrs.add(iter_tensors[g].data_ptr())

        self.assertEqual(
            len(helper_ptrs),
            1,
            "All helper groups should share one placeholder buffer",
        )
        self.assertNotIn(
            active_ptr, helper_ptrs, "Active buffer must not alias the placeholder"
        )

    def test_helper_slots_are_placeholders_with_full_geometry(self) -> None:
        """
        Drive an allreduce with heterogeneous per-group totals via mocked
        allgather; assert each helper slot is a 1-element placeholder while
        per_group_sizes carries that group's full total for the kernel geometry.
        """
        state = _make_state(rank=0)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        # Group totals: [100, 300, 200, 150].  Rank 0 is active for group 0.
        group_totals = [100, 300, 200, 150]

        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for r, t in enumerate(tensor_list):
                t.fill_(group_totals[r // state.sparse_group_size])

        with patch("torchrec.distributed.sharded_relay_utils.dist") as mock_dist:
            mock_dist.all_gather.side_effect = _allgather_side_effect
            mock_dist.ReduceOp = dist.ReduceOp
            allreduce_tensors_with_sharded_relay(
                state, {torch.float32: [torch.zeros(100)]}, "hetero"
            )

        call_kwargs = self._all_calls(state)[0].kwargs
        iter_tensors = call_kwargs["tensors"]
        iter_sizes = call_kwargs["per_group_sizes"]
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            self.assertEqual(
                iter_tensors[g].numel(),
                1,
                f"group={g}: helper should be a 1-element placeholder",
            )
            self.assertEqual(
                iter_sizes[g],
                group_totals[g],
                f"group={g}: per_group_sizes should carry the full total",
            )

    def test_active_buffer_unchanged(self) -> None:
        """Active flat buffer is still sized to my_total, reused, and distinct."""
        state = _make_state(rank=0)
        t1 = torch.zeros(300)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [t1]}, "active_test"
        )
        call_kwargs = self._all_calls(state)[0].kwargs
        active_tensor = call_kwargs["tensors"][state.my_sparse_group]
        self.assertEqual(active_tensor.numel(), 300)

    def test_bm_fm_real_totals_use_placeholder_helpers(self) -> None:
        """
        Using real BM-FM per-group totals, assert helper slots are shared
        1-element placeholders while per_group_sizes carries each group's full
        total for the kernel geometry.
        """
        state = _make_state(rank=0, sparse_group_size=2, local_size=8)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        bm_fm_fp16_totals = [
            12_002_982_488,
            12_245_126_152,
            12_014_370_640,
            12_057_805_952,
        ]

        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for r, t in enumerate(tensor_list):
                t.fill_(bm_fm_fp16_totals[r // state.sparse_group_size])

        with patch("torchrec.distributed.sharded_relay_utils.dist") as mock_dist:
            mock_dist.all_gather.side_effect = _allgather_side_effect
            mock_dist.ReduceOp = dist.ReduceOp
            tensors = [torch.zeros(1, dtype=torch.float16)]
            allreduce_tensors_with_sharded_relay(
                state, {torch.float16: tensors}, "bm_fm_2d_weight_sync"
            )

        call_kwargs = self._all_calls(state)[0].kwargs
        iter_tensors = call_kwargs["tensors"]
        iter_sizes = call_kwargs["per_group_sizes"]
        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            self.assertEqual(
                iter_tensors[g].numel(),
                1,
                f"group={g}: helper should be a 1-element placeholder",
            )
            self.assertEqual(
                iter_sizes[g],
                bm_fm_fp16_totals[g],
                f"group={g}: per_group_sizes should carry the full total",
            )
            helper_ptrs.add(iter_tensors[g].data_ptr())
        self.assertEqual(
            len(helper_ptrs), 1, "All helper groups should share one placeholder"
        )


# ---------------------------------------------------------------------------
# Tests for FusedShardedRelayMultiGroup.allreduce_multi_group validation
# (kernel API — unchanged by the flat-concat rewrite)
# ---------------------------------------------------------------------------


class FusedShardedRelayValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # rcclx_comm=None → _use_native=False; validation still runs.
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_raises_value_error_on_active_tensor_size_mismatch(self) -> None:
        """
        allreduce_multi_group must raise ValueError when an ACTIVE group's
        tensor.numel() does not match per_group_sizes[g].  Helper slots pass a
        placeholder buffer and are intentionally NOT size-validated (the kernel
        stages them into internal scratch).
        """
        fused = self._make_fused(rank=0)  # active for group 0
        tensors = [
            torch.zeros(640),  # group 0 (active) — mismatch (expected 500)
            torch.zeros(1),  # group 1 — helper placeholder (not validated)
            torch.zeros(1),  # group 2 — helper placeholder (not validated)
            torch.zeros(1),  # group 3 — helper placeholder (not validated)
        ]
        per_group_sizes = [500, 500, 750, 600]  # group 0: 640 vs 500

        with self.assertRaises(ValueError) as cm:
            fused.allreduce_multi_group(
                tensors=tensors,
                num_groups=4,
                per_group_sizes=per_group_sizes,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
                skip_validation=False,
            )

        err = str(cm.exception)
        self.assertIn("640", err)  # actual numel
        self.assertIn("500", err)  # expected size

    def test_helper_slot_size_mismatch_is_ignored(self) -> None:
        """
        A helper slot's tensor numel need not match per_group_sizes: the kernel
        stages helpers into internal scratch and never touches the placeholder.
        Validation must skip non-active slots (RuntimeError from the missing
        native API is expected, but never a ValueError).
        """
        fused = self._make_fused(rank=0)  # active for group 0 only
        tensors = [
            torch.zeros(500),  # group 0 (active) — matches
            torch.zeros(1),  # group 1 (helper) — placeholder, size ignored
            torch.zeros(1),  # group 2 (helper) — placeholder, size ignored
            torch.zeros(1),  # group 3 (helper) — placeholder, size ignored
        ]
        per_group_sizes = [500, 500, 750, 600]

        with self.assertRaises(RuntimeError) as cm:
            fused.allreduce_multi_group(
                tensors=tensors,
                num_groups=4,
                per_group_sizes=per_group_sizes,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
                skip_validation=False,
            )
        self.assertNotIsInstance(cm.exception, ValueError)

    def test_count_zero_group_skips_size_validation(self) -> None:
        """
        Regression for BM-FM failure (Python layer):
          ValueError: Tensor 3 has 1 elements, but per_group_sizes[3]=0

        The Python allreduce_multi_group must skip count=0 groups in its
        size validation — they carry a 1-element placeholder the kernel ignores.
        """
        fused = self._make_fused(rank=0)
        # Groups 0,1,2 have data; group 3 ran out — count=0, placeholder=1 elem.
        tensors = [torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(1)]
        per_group_sizes = [100, 100, 100, 0]

        # Must NOT raise ValueError.  RuntimeError (no native API) is expected.
        with self.assertRaises(RuntimeError) as cm:
            fused.allreduce_multi_group(
                tensors=tensors,
                num_groups=4,
                per_group_sizes=per_group_sizes,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
            )
        self.assertNotIsInstance(cm.exception, ValueError)

    def test_my_active_group_count_zero_skips_validation(self) -> None:
        """
        Same scenario from my own group's perspective: when this rank has run
        out of tensors (iter_idx >= my_tensor_count), my group slot gets
        count=0 and a 1-element placeholder.  Validation must skip it.
        """
        fused = self._make_fused(rank=0)
        # Group 0 (this rank's active group) has count=0 at this iteration.
        tensors = [torch.zeros(1), torch.zeros(200), torch.zeros(200), torch.zeros(200)]
        per_group_sizes = [0, 200, 200, 200]

        with self.assertRaises(RuntimeError) as cm:
            fused.allreduce_multi_group(
                tensors=tensors,
                num_groups=4,
                per_group_sizes=per_group_sizes,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
            )
        self.assertNotIsInstance(cm.exception, ValueError)


# ---------------------------------------------------------------------------
# Tests for reduce_scatter_tensors_with_sharded_relay (flat-concat approach)
# ---------------------------------------------------------------------------


class FlatReduceScatterTest(unittest.TestCase):
    """Tests for the reduce-scatter flat-concat helper.

    The active group's input is packed into ONE contiguous flat buffer holding
    nActiveRanks x recv_count elements (two blocks for sparse_group_size=2);
    the output holds recv_count elements.  fused.reduce_scatter_multi_group is
    a MagicMock that records every call (one tensor per group).
    """

    def _call_count(self, state: ShardedRelayState) -> int:
        return state.fused.reduce_scatter_multi_group.call_count

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.reduce_scatter_multi_group.call_args_list

    def test_returns_immediately_when_no_tensors(self) -> None:
        state = _make_state(rank=0)
        reduce_scatter_tensors_with_sharded_relay(state, {}, {}, "test")
        self.assertEqual(self._call_count(state), 0)

    def test_single_call_recv_count_is_half_input_total(self) -> None:
        state = _make_state(rank=0)
        # input total 200 -> recv_count 100 (sparse_group_size=2)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(100)]},
            "test",
        )
        self.assertEqual(self._call_count(state), 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["per_group_recv_counts"][state.my_sparse_group], 100)

    def test_single_call_per_dtype(self) -> None:
        state = _make_state(rank=0)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {
                torch.float16: [torch.zeros(40, dtype=torch.float16)],
                torch.float32: [torch.zeros(40, dtype=torch.float32)],
            },
            {
                torch.float16: [torch.zeros(20, dtype=torch.float16)],
                torch.float32: [torch.zeros(20, dtype=torch.float32)],
            },
            "test",
        )
        self.assertEqual(self._call_count(state), 2)

    def test_active_input_and_output_buffer_sizes(self) -> None:
        state = _make_state(rank=0)
        in_sizes = [100, 200, 300]  # total 600 -> recv 300
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(s) for s in in_sizes]},
            {torch.float32: [torch.zeros(300)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertEqual(kwargs["per_group_recv_counts"][my_g], 300)
        # Active group's inputs are packed into ONE contiguous flat buffer.
        self.assertEqual(kwargs["input_tensors"][my_g].numel(), 600)
        self.assertEqual(kwargs["output_tensors"][my_g].numel(), 300)

    def test_raises_when_input_total_not_divisible_by_group_size(self) -> None:
        state = _make_state(rank=0)  # sparse_group_size=2
        with self.assertRaises(ValueError):
            reduce_scatter_tensors_with_sharded_relay(
                state,
                {torch.float32: [torch.zeros(101)]},  # odd
                {torch.float32: [torch.zeros(50)]},
                "test",
            )

    def test_raises_when_output_total_mismatches_recv_count(self) -> None:
        state = _make_state(rank=0)
        with self.assertRaises(ValueError):
            reduce_scatter_tensors_with_sharded_relay(
                state,
                {torch.float32: [torch.zeros(200)]},  # recv 100
                {torch.float32: [torch.zeros(90)]},  # wrong
                "test",
            )

    def test_helper_slots_are_shared_placeholder(self) -> None:
        state = _make_state(rank=0)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(400)]},  # recv 200
            {torch.float32: [torch.zeros(200)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        in_tensors = kwargs["input_tensors"]
        out_tensors = kwargs["output_tensors"]
        recv = kwargs["per_group_recv_counts"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            # Helper slot is a 1-element placeholder; recv count keeps geometry.
            self.assertEqual(in_tensors[g].numel(), 1)
            self.assertEqual(recv[g], recv[state.my_sparse_group])
            # Helper in/out both alias the single shared placeholder.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), 1)

    def test_active_output_is_separate_flat_buffer(self) -> None:
        # Flat-concat scheme: the active group's output is a separate internal
        # flat buffer (one contiguous tensor per group), distinct from the
        # caller's output tensor; results are unpacked into the caller tensor
        # after the call.
        state = _make_state(rank=1)
        out_tensor = torch.zeros(100)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},  # recv 100
            {torch.float32: [out_tensor]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        out_seg = kwargs["output_tensors"][my_g]
        self.assertEqual(out_seg.numel(), 100)
        self.assertNotEqual(out_seg.data_ptr(), out_tensor.data_ptr())

    def test_out_of_place_output_distinct_from_input(self) -> None:
        state = _make_state(rank=0)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(100)]},
            "test",
            in_place=False,
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertNotEqual(
            kwargs["input_tensors"][my_g].data_ptr(),
            kwargs["output_tensors"][my_g].data_ptr(),
        )

    def test_in_place_output_aliases_input_owned_block(self) -> None:
        # In-place: the active output is an owned-block view into the active
        # input flat buffer at offset my_active_index * recv_count (the inverse
        # of the out-of-place distinct-pointer assertion above).
        state = _make_state(rank=1)  # my_active_index = 1 within group 0
        recv_count = 100
        inp = torch.zeros(2 * recv_count)  # nActiveRanks x recv_count
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [inp]},
            {torch.float32: [torch.zeros(recv_count)]},
            "test",
            in_place=True,
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        in_flat = kwargs["input_tensors"][my_g]
        out_seg = kwargs["output_tensors"][my_g]
        self.assertEqual(in_flat.numel(), 2 * recv_count)
        self.assertEqual(out_seg.numel(), recv_count)
        self.assertEqual(
            out_seg.data_ptr(),
            in_flat.data_ptr() + recv_count * in_flat.element_size(),
        )

    def test_values_written_back_to_output_tensors(self) -> None:
        state = _make_state(rank=0)
        out_tensor = torch.zeros(100)
        sentinel = 7.0

        def _fill(*args, **kwargs) -> None:
            outs = kwargs["output_tensors"]
            outs[state.my_sparse_group].fill_(sentinel)

        state.fused.reduce_scatter_multi_group.side_effect = _fill

        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [out_tensor]},
            "test",
        )
        self.assertTrue(torch.all(out_tensor == sentinel))

    def test_unpack_handles_multiple_output_tensors(self) -> None:
        state = _make_state(rank=0)
        o0 = torch.zeros(40)
        o1 = torch.zeros(60)  # total output 100 -> input 200
        fill_values = [1.0, 2.0]

        def _fill_by_slice(*args, **kwargs) -> None:
            # Flat-concat scheme: fill successive slices of the output flat buf.
            flat = kwargs["output_tensors"][state.my_sparse_group]
            flat[:40].fill_(fill_values[0])
            flat[40:100].fill_(fill_values[1])

        state.fused.reduce_scatter_multi_group.side_effect = _fill_by_slice

        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [o0, o1]},
            "test",
        )
        self.assertTrue(torch.all(o0 == fill_values[0]))
        self.assertTrue(torch.all(o1 == fill_values[1]))

    @patch("torchrec.distributed.sharded_relay_utils.dist")
    def test_metadata_cache_skips_allgather_after_first_call(
        self, mock_dist: MagicMock
    ) -> None:
        state = _make_state(rank=0)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for t in tensor_list:
                t.fill_(100)  # recv_count 100 for all ranks

        mock_dist.all_gather.side_effect = _allgather_side_effect
        mock_dist.ReduceOp = dist.ReduceOp

        ins = {torch.float32: [torch.zeros(200)]}
        outs = {torch.float32: [torch.zeros(100)]}

        reduce_scatter_tensors_with_sharded_relay(state, ins, outs, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        reduce_scatter_tensors_with_sharded_relay(state, ins, outs, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        reduce_scatter_tensors_with_sharded_relay(state, ins, outs, "other")
        self.assertEqual(mock_dist.all_gather.call_count, 2)

    def test_active_output_flat_buf_reused_across_steps(self) -> None:
        """Out-of-place output flat buffer must be reused (grow-only)."""
        state = _make_state(rank=0)
        b1 = _get_active_output_flat_buf(state, 100, torch.float32, _DEVICE)
        b2 = _get_active_output_flat_buf(state, 100, torch.float32, _DEVICE)
        self.assertEqual(b1.data_ptr(), b2.data_ptr())


# ---------------------------------------------------------------------------
# Tests for FusedShardedRelayMultiGroup.reduce_scatter_multi_group validation
# ---------------------------------------------------------------------------


class FusedReduceScatterValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # rcclx_comm=None -> _use_native=False; validation still runs.
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_raises_on_active_input_too_small(self) -> None:
        """Active input must hold nActiveRanks x recv_count elements."""
        fused = self._make_fused(rank=0)  # active for group 0
        # recv_count 100 -> input must be 200; pass 150.
        input_tensors = [
            torch.zeros(150),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(100),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_recv_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.reduce_scatter_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_recv_counts=per_group_recv_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
                skip_validation=False,
            )
        self.assertIn("200", str(cm.exception))

    def test_raises_on_active_output_too_small(self) -> None:
        fused = self._make_fused(rank=0)
        input_tensors = [
            torch.zeros(200),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(90),  # need 100
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_recv_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.reduce_scatter_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_recv_counts=per_group_recv_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
                skip_validation=False,
            )
        self.assertIn("100", str(cm.exception))

    def test_recv_count_zero_skips_validation(self) -> None:
        """recv_count=0 group carries a placeholder and must skip validation."""
        fused = self._make_fused(rank=0)
        # Group 0 (this rank's active group) ran out -> recv_count=0, placeholder.
        input_tensors = [
            torch.zeros(1),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(1),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_recv_counts = [0, 5, 5, 5]

        # Must NOT raise ValueError. RuntimeError (no native API) is expected.
        with self.assertRaises(RuntimeError) as cm:
            fused.reduce_scatter_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_recv_counts=per_group_recv_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                op=dist.ReduceOp.SUM,
            )
        self.assertNotIsInstance(cm.exception, ValueError)


# ---------------------------------------------------------------------------
# Tests for all_to_all_tensors_with_sharded_relay (flat-concat approach)
# ---------------------------------------------------------------------------


class FlatAllToAllTest(unittest.TestCase):
    """Tests for the all-to-all flat-concat helper (out-of-place only).

    The active group's input/output each hold nActiveRanks x segment_count
    elements (two segments for sparse_group_size=2).
    fused.all_to_all_multi_group is a MagicMock that records every call.
    """

    def _call_count(self, state: ShardedRelayState) -> int:
        return state.fused.all_to_all_multi_group.call_count

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.all_to_all_multi_group.call_args_list

    def test_returns_immediately_when_no_tensors(self) -> None:
        state = _make_state(rank=0)
        all_to_all_tensors_with_sharded_relay(state, {}, {}, "test")
        self.assertEqual(self._call_count(state), 0)

    def test_single_call_segment_count_is_half_input_total(self) -> None:
        state = _make_state(rank=0)
        # input total 200 -> segment_count 100; output total 200
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
        )
        self.assertEqual(self._call_count(state), 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["per_group_segment_counts"][state.my_sparse_group], 100)

    def test_single_call_per_dtype(self) -> None:
        state = _make_state(rank=0)
        all_to_all_tensors_with_sharded_relay(
            state,
            {
                torch.float16: [torch.zeros(40, dtype=torch.float16)],
                torch.float32: [torch.zeros(40, dtype=torch.float32)],
            },
            {
                torch.float16: [torch.zeros(40, dtype=torch.float16)],
                torch.float32: [torch.zeros(40, dtype=torch.float32)],
            },
            "test",
        )
        self.assertEqual(self._call_count(state), 2)

    def test_active_input_and_output_buffer_sizes(self) -> None:
        state = _make_state(rank=0)
        in_sizes = [100, 200, 300]  # total 600 -> segment 300
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(s) for s in in_sizes]},
            {torch.float32: [torch.zeros(600)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertEqual(kwargs["per_group_segment_counts"][my_g], 300)
        self.assertEqual(kwargs["input_tensors"][my_g].numel(), 600)
        self.assertEqual(kwargs["output_tensors"][my_g].numel(), 600)

    def test_raises_when_input_total_not_divisible_by_group_size(self) -> None:
        state = _make_state(rank=0)  # sparse_group_size=2
        with self.assertRaises(ValueError):
            all_to_all_tensors_with_sharded_relay(
                state,
                {torch.float32: [torch.zeros(101)]},  # odd
                {torch.float32: [torch.zeros(101)]},
                "test",
            )

    def test_raises_when_output_total_mismatches_input_total(self) -> None:
        state = _make_state(rank=0)
        with self.assertRaises(ValueError):
            all_to_all_tensors_with_sharded_relay(
                state,
                {torch.float32: [torch.zeros(200)]},
                {torch.float32: [torch.zeros(100)]},  # must equal input total
                "test",
            )

    def test_helper_slots_are_shared_placeholder(self) -> None:
        state = _make_state(rank=0)
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(400)]},  # segment 200
            {torch.float32: [torch.zeros(400)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        in_tensors = kwargs["input_tensors"]
        out_tensors = kwargs["output_tensors"]
        seg = kwargs["per_group_segment_counts"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            # Helper slot is a 1-element placeholder; segment count keeps geometry.
            self.assertEqual(in_tensors[g].numel(), 1)
            self.assertEqual(seg[g], seg[state.my_sparse_group])
            # Helper in/out both alias the single shared placeholder.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), 1)

    def test_output_flat_distinct_from_input_flat(self) -> None:
        """All-to-all is out-of-place: active output flat must differ from input."""
        state = _make_state(rank=0)
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertNotEqual(
            kwargs["input_tensors"][my_g].data_ptr(),
            kwargs["output_tensors"][my_g].data_ptr(),
        )

    def test_values_written_back_to_output_tensors(self) -> None:
        state = _make_state(rank=0)
        out_tensor = torch.zeros(200)
        sentinel = 9.0

        def _fill(*args, **kwargs) -> None:
            outs = kwargs["output_tensors"]
            outs[state.my_sparse_group].fill_(sentinel)

        state.fused.all_to_all_multi_group.side_effect = _fill

        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [out_tensor]},
            "test",
        )
        self.assertTrue(torch.all(out_tensor == sentinel))

    def test_unpack_handles_multiple_output_tensors(self) -> None:
        state = _make_state(rank=0)
        o0 = torch.zeros(80)
        o1 = torch.zeros(120)  # total 200 == input 200
        fill_values = [3.0, 4.0]

        def _fill_by_slice(*args, **kwargs) -> None:
            flat = kwargs["output_tensors"][state.my_sparse_group]
            flat[:80].fill_(fill_values[0])
            flat[80:200].fill_(fill_values[1])

        state.fused.all_to_all_multi_group.side_effect = _fill_by_slice

        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [o0, o1]},
            "test",
        )
        self.assertTrue(torch.all(o0 == fill_values[0]))
        self.assertTrue(torch.all(o1 == fill_values[1]))

    @patch("torchrec.distributed.sharded_relay_utils.dist")
    def test_metadata_cache_skips_allgather_after_first_call(
        self, mock_dist: MagicMock
    ) -> None:
        state = _make_state(rank=0)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for t in tensor_list:
                t.fill_(100)  # segment_count 100 for all ranks

        mock_dist.all_gather.side_effect = _allgather_side_effect
        mock_dist.ReduceOp = dist.ReduceOp

        ins = {torch.float32: [torch.zeros(200)]}
        outs = {torch.float32: [torch.zeros(200)]}

        all_to_all_tensors_with_sharded_relay(state, ins, outs, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        all_to_all_tensors_with_sharded_relay(state, ins, outs, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        all_to_all_tensors_with_sharded_relay(state, ins, outs, "other")
        self.assertEqual(mock_dist.all_gather.call_count, 2)


# ---------------------------------------------------------------------------
# Tests for FusedShardedRelayMultiGroup.all_to_all_multi_group validation
# ---------------------------------------------------------------------------


class FusedAllToAllValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_raises_on_active_input_too_small(self) -> None:
        fused = self._make_fused(rank=0)  # active for group 0
        # segment_count 100 -> input/output must be 200; pass input 150.
        input_tensors = [
            torch.zeros(150),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(200),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_segment_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.all_to_all_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                skip_validation=False,
            )
        self.assertIn("200", str(cm.exception))

    def test_raises_on_active_output_too_small(self) -> None:
        fused = self._make_fused(rank=0)
        input_tensors = [
            torch.zeros(200),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(150),  # need 200
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_segment_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.all_to_all_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                skip_validation=False,
            )
        self.assertIn("200", str(cm.exception))

    def test_raises_on_in_place(self) -> None:
        """In-place (input aliases output) for the active group must raise."""
        fused = self._make_fused(rank=0)
        shared = torch.zeros(200)
        input_tensors = [
            shared,
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            shared,  # aliases input for active group 0
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_segment_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.all_to_all_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                skip_validation=False,
            )
        self.assertIn("in-place", str(cm.exception))

    def test_segment_count_zero_skips_validation(self) -> None:
        """segment_count=0 group carries a placeholder and must skip validation."""
        fused = self._make_fused(rank=0)
        input_tensors = [
            torch.zeros(1),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(1),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_segment_counts = [0, 5, 5, 5]

        # Must NOT raise ValueError. RuntimeError (no native API) is expected.
        with self.assertRaises(RuntimeError) as cm:
            fused.all_to_all_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
            )
        self.assertNotIsInstance(cm.exception, ValueError)


# ---------------------------------------------------------------------------
# Tests for all_gather_tensors_with_sharded_relay (flat-concat approach)
# ---------------------------------------------------------------------------


class FlatAllGatherTest(unittest.TestCase):
    """Tests for the all-gather flat-concat helper (in-place + out-of-place).

    The active group's input holds send_count elements (this rank's
    contribution); its output holds nActiveRanks x send_count elements.
    fused.all_gather_multi_group is a MagicMock that records every call.
    """

    def _call_count(self, state: ShardedRelayState) -> int:
        return state.fused.all_gather_multi_group.call_count

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.all_gather_multi_group.call_args_list

    def test_returns_immediately_when_no_tensors(self) -> None:
        state = _make_state(rank=0)
        all_gather_tensors_with_sharded_relay(state, {}, {}, "test")
        self.assertEqual(self._call_count(state), 0)

    def test_single_call_output_is_group_size_times_input(self) -> None:
        state = _make_state(rank=0)
        # input total 100 (send_count); output total 200 (2 x send_count)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(100)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
        )
        self.assertEqual(self._call_count(state), 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["per_group_send_counts"][state.my_sparse_group], 100)

    def test_single_call_per_dtype(self) -> None:
        state = _make_state(rank=0)
        all_gather_tensors_with_sharded_relay(
            state,
            {
                torch.float16: [torch.zeros(20, dtype=torch.float16)],
                torch.float32: [torch.zeros(20, dtype=torch.float32)],
            },
            {
                torch.float16: [torch.zeros(40, dtype=torch.float16)],
                torch.float32: [torch.zeros(40, dtype=torch.float32)],
            },
            "test",
        )
        self.assertEqual(self._call_count(state), 2)

    def test_active_input_and_output_buffer_sizes(self) -> None:
        state = _make_state(rank=0)
        in_sizes = [100, 200]  # send_count 300
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(s) for s in in_sizes]},
            {torch.float32: [torch.zeros(600)]},  # 2 x 300
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertEqual(kwargs["per_group_send_counts"][my_g], 300)
        self.assertEqual(kwargs["input_tensors"][my_g].numel(), 300)
        self.assertEqual(kwargs["output_tensors"][my_g].numel(), 600)

    def test_raises_when_output_total_mismatches(self) -> None:
        state = _make_state(rank=0)
        with self.assertRaises(ValueError):
            all_gather_tensors_with_sharded_relay(
                state,
                {torch.float32: [torch.zeros(100)]},  # send 100 -> output 200
                {torch.float32: [torch.zeros(150)]},  # wrong
                "test",
            )

    def test_helper_slots_are_shared_placeholder(self) -> None:
        state = _make_state(rank=0)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},  # send 200
            {torch.float32: [torch.zeros(400)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        in_tensors = kwargs["input_tensors"]
        out_tensors = kwargs["output_tensors"]
        send = kwargs["per_group_send_counts"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            # Helper slot is a 1-element placeholder; send count keeps geometry.
            self.assertEqual(in_tensors[g].numel(), 1)
            self.assertEqual(send[g], send[state.my_sparse_group])
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), 1)

    def test_active_input_is_separate_flat_buffer(self) -> None:
        # Flat-concat scheme: the active group's input is a separate internal
        # flat send buffer (one contiguous tensor per group), distinct from the
        # caller's input tensor (packed into it before the call).
        state = _make_state(rank=1)
        in_tensor = torch.zeros(100)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [in_tensor]},  # send 100
            {torch.float32: [torch.zeros(200)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        in_seg = kwargs["input_tensors"][my_g]
        self.assertEqual(in_seg.numel(), 100)
        self.assertNotEqual(in_seg.data_ptr(), in_tensor.data_ptr())

    def test_out_of_place_input_distinct_from_output(self) -> None:
        state = _make_state(rank=0)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(100)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
            in_place=False,
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertNotEqual(
            kwargs["input_tensors"][my_g].data_ptr(),
            kwargs["output_tensors"][my_g].data_ptr(),
        )

    def test_in_place_input_aliases_output_own_slot(self) -> None:
        # In-place: the active input is an owned-slot view into the active
        # output flat buffer at offset my_active_index * send_count (the
        # inverse of the out-of-place distinct-pointer assertion above).
        state = _make_state(rank=1)  # my_active_index = 1 within group 0
        send_count = 100
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(send_count)]},
            {torch.float32: [torch.zeros(2 * send_count)]},
            "test",
            in_place=True,
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        in_seg = kwargs["input_tensors"][my_g]
        out_flat = kwargs["output_tensors"][my_g]
        self.assertEqual(out_flat.numel(), 2 * send_count)
        self.assertEqual(in_seg.numel(), send_count)
        self.assertEqual(
            in_seg.data_ptr(),
            out_flat.data_ptr() + send_count * out_flat.element_size(),
        )

    def test_values_written_back_to_output_tensors(self) -> None:
        state = _make_state(rank=0)
        out_tensor = torch.zeros(200)
        sentinel = 5.0

        def _fill(*args, **kwargs) -> None:
            outs = kwargs["output_tensors"]
            outs[state.my_sparse_group].fill_(sentinel)

        state.fused.all_gather_multi_group.side_effect = _fill

        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(100)]},
            {torch.float32: [out_tensor]},
            "test",
        )
        self.assertTrue(torch.all(out_tensor == sentinel))

    def test_unpack_handles_multiple_output_tensors(self) -> None:
        state = _make_state(rank=0)
        o0 = torch.zeros(80)
        o1 = torch.zeros(120)  # total 200 == 2 x send 100
        fill_values = [6.0, 7.0]

        def _fill_by_slice(*args, **kwargs) -> None:
            flat = kwargs["output_tensors"][state.my_sparse_group]
            flat[:80].fill_(fill_values[0])
            flat[80:200].fill_(fill_values[1])

        state.fused.all_gather_multi_group.side_effect = _fill_by_slice

        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(100)]},
            {torch.float32: [o0, o1]},
            "test",
        )
        self.assertTrue(torch.all(o0 == fill_values[0]))
        self.assertTrue(torch.all(o1 == fill_values[1]))

    @patch("torchrec.distributed.sharded_relay_utils.dist")
    def test_metadata_cache_skips_allgather_after_first_call(
        self, mock_dist: MagicMock
    ) -> None:
        state = _make_state(rank=0)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for t in tensor_list:
                t.fill_(100)  # send_count 100 for all ranks

        mock_dist.all_gather.side_effect = _allgather_side_effect
        mock_dist.ReduceOp = dist.ReduceOp

        ins = {torch.float32: [torch.zeros(100)]}
        outs = {torch.float32: [torch.zeros(200)]}

        all_gather_tensors_with_sharded_relay(state, ins, outs, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        all_gather_tensors_with_sharded_relay(state, ins, outs, "step")
        self.assertEqual(mock_dist.all_gather.call_count, 1)

        all_gather_tensors_with_sharded_relay(state, ins, outs, "other")
        self.assertEqual(mock_dist.all_gather.call_count, 2)


# ---------------------------------------------------------------------------
# Tests for FusedShardedRelayMultiGroup.all_gather_multi_group validation
# ---------------------------------------------------------------------------


class FusedAllGatherValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_raises_on_active_input_too_small(self) -> None:
        fused = self._make_fused(rank=0)  # active for group 0
        # send_count 100 -> input must be >= 100; pass 90.
        input_tensors = [
            torch.zeros(90),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(200),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_send_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.all_gather_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_send_counts=per_group_send_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                skip_validation=False,
            )
        self.assertIn("100", str(cm.exception))

    def test_raises_on_active_output_too_small(self) -> None:
        fused = self._make_fused(rank=0)
        # send_count 100 -> output must be >= 200; pass 150.
        input_tensors = [
            torch.zeros(100),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(150),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_send_counts = [100, 5, 5, 5]

        with self.assertRaises(ValueError) as cm:
            fused.all_gather_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_send_counts=per_group_send_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
                skip_validation=False,
            )
        self.assertIn("200", str(cm.exception))

    def test_send_count_zero_skips_validation(self) -> None:
        """send_count=0 group carries a placeholder and must skip validation."""
        fused = self._make_fused(rank=0)
        input_tensors = [
            torch.zeros(1),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        output_tensors = [
            torch.zeros(1),
            torch.zeros(10),
            torch.zeros(10),
            torch.zeros(10),
        ]
        per_group_send_counts = [0, 5, 5, 5]

        # Must NOT raise ValueError. RuntimeError (no native API) is expected.
        with self.assertRaises(RuntimeError) as cm:
            fused.all_gather_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=4,
                per_group_send_counts=per_group_send_counts,
                all_active_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
            )
        self.assertNotIsInstance(cm.exception, ValueError)


# ---------------------------------------------------------------------------
# Tests for the 4-active-rank allreduce path (2 groups x 4 active per group)
# ---------------------------------------------------------------------------


class FlatAllreduce4ActiveTest(unittest.TestCase):
    """allreduce_tensors_with_sharded_relay with sparse_group_size=4.

    8-rank node -> 2 groups of 4 active ranks; each rank is helper for the 1
    other group. Verifies the flat-concat path is generic over the active-rank
    count.
    """

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.allreduce_multi_group.call_args_list

    def test_single_call_two_groups(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(400)]}, "test"
        )
        self.assertEqual(state.fused.allreduce_multi_group.call_count, 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["num_groups"], 2)
        self.assertEqual(kwargs["per_group_sizes"][state.my_sparse_group], 400)

    def test_helper_slots_are_placeholder_4active(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(800)]}, "test"
        )
        kwargs = self._all_calls(state)[0].kwargs
        iter_tensors = kwargs["tensors"]
        iter_sizes = kwargs["per_group_sizes"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                self.assertEqual(iter_tensors[g].numel(), 800)
                continue
            # Helper slot is a 1-element placeholder; per_group_sizes keeps the
            # full geometry for the kernel.
            self.assertEqual(iter_tensors[g].numel(), 1)
            self.assertEqual(iter_sizes[g], 800)
            helper_ptrs.add(iter_tensors[g].data_ptr())

        # One shared placeholder across all helper groups.
        self.assertEqual(len(helper_ptrs), 1)

    def test_active_group_is_group_zero_for_rank0(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        self.assertEqual(state.my_sparse_group, 0)
        self.assertEqual(state.num_sparse_groups, 2)
        self.assertEqual(state.precomputed_active_ranks, [[0, 1, 2, 3], [4, 5, 6, 7]])


class FusedAllreduce4ActiveValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_4active_validation_accepts(self) -> None:
        """allreduce_multi_group must accept 4 active ranks (no ValueError);
        without a native comm it falls through to RuntimeError."""
        fused = self._make_fused(rank=0)
        tensors = [torch.zeros(400), torch.zeros(400)]
        per_group_sizes = [400, 400]

        with self.assertRaises(RuntimeError) as cm:
            fused.allreduce_multi_group(
                tensors=tensors,
                num_groups=2,
                per_group_sizes=per_group_sizes,
                all_active_ranks=[[0, 1, 2, 3], [4, 5, 6, 7]],
                op=dist.ReduceOp.SUM,
            )
        self.assertNotIsInstance(cm.exception, ValueError)


class FlatReduceScatter4ActiveTest(unittest.TestCase):
    """reduce_scatter_tensors_with_sharded_relay with sparse_group_size=4.

    8-rank node -> 2 groups of 4 active ranks; each rank is helper for the 1
    other group. The active group's input holds nActiveRanks x recv_count
    elements; the output holds recv_count. Verifies the flat-concat path is
    generic over the active-rank count.
    """

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.reduce_scatter_multi_group.call_args_list

    def test_single_call_two_groups(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        # input total 400 -> recv_count 100 (sparse_group_size=4)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(400)]},
            {torch.float32: [torch.zeros(100)]},
            "test",
        )
        self.assertEqual(state.fused.reduce_scatter_multi_group.call_count, 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["num_groups"], 2)
        self.assertEqual(kwargs["per_group_recv_counts"][state.my_sparse_group], 100)
        self.assertEqual(
            kwargs["input_tensors"][state.my_sparse_group].numel(),
            400,
        )
        self.assertEqual(
            kwargs["output_tensors"][state.my_sparse_group].numel(),
            100,
        )

    def test_helper_slots_are_placeholder_4active(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        # input total 800 -> recv_count 200
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(800)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        in_tensors = kwargs["input_tensors"]
        out_tensors = kwargs["output_tensors"]
        recv = kwargs["per_group_recv_counts"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                self.assertEqual(in_tensors[g].numel(), 800)
                self.assertEqual(out_tensors[g].numel(), 200)
                continue
            # Helper slot is a 1-element placeholder; recv count keeps geometry.
            self.assertEqual(in_tensors[g].numel(), 1)
            self.assertEqual(recv[g], recv[state.my_sparse_group])
            # Helper in/out both alias the single shared placeholder.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), 1)

    def test_in_place_output_aliases_input_owned_block(self) -> None:
        # 4-active in-place: the active output is an owned-block view into the
        # active input flat buffer at offset my_active_index * recv_count.
        state = _make_state(rank=2, sparse_group_size=4, local_size=8)
        recv_count = 100
        inp = torch.zeros(4 * recv_count)  # nActiveRanks x recv_count
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [inp]},
            {torch.float32: [torch.zeros(recv_count)]},
            "test",
            in_place=True,
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        in_flat = kwargs["input_tensors"][my_g]
        out_seg = kwargs["output_tensors"][my_g]
        self.assertEqual(out_seg.numel(), recv_count)
        # rank=2 -> my_active_index = 2 within its group.
        self.assertEqual(
            out_seg.data_ptr(),
            in_flat.data_ptr() + 2 * recv_count * in_flat.element_size(),
        )

    def test_active_group_is_group_zero_for_rank0(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        self.assertEqual(state.my_sparse_group, 0)
        self.assertEqual(state.num_sparse_groups, 2)
        self.assertEqual(state.precomputed_active_ranks, [[0, 1, 2, 3], [4, 5, 6, 7]])


class FusedReduceScatter4ActiveValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_4active_validation_accepts(self) -> None:
        """reduce_scatter_multi_group must accept 4 active ranks (no
        ValueError); without a native comm it falls through to RuntimeError.
        The active input holds nActiveRanks x recv_count = 4 x 100."""
        fused = self._make_fused(rank=0)
        input_tensors = [torch.zeros(400), torch.zeros(400)]
        output_tensors = [torch.zeros(100), torch.zeros(100)]
        per_group_recv_counts = [100, 100]

        with self.assertRaises(RuntimeError) as cm:
            fused.reduce_scatter_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=2,
                per_group_recv_counts=per_group_recv_counts,
                all_active_ranks=[[0, 1, 2, 3], [4, 5, 6, 7]],
                op=dist.ReduceOp.SUM,
            )
        self.assertNotIsInstance(cm.exception, ValueError)


class FlatAllToAll4ActiveTest(unittest.TestCase):
    """all_to_all_tensors_with_sharded_relay with sparse_group_size=4.

    8-rank node -> 2 groups of 4 active ranks. The active group's input/output
    each hold nActiveRanks x segment_count elements (out-of-place). Verifies the
    flat-concat path is generic over the active-rank count.
    """

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.all_to_all_multi_group.call_args_list

    def test_single_call_two_groups(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        # input total 400 -> segment_count 100 (sparse_group_size=4)
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(400)]},
            {torch.float32: [torch.zeros(400)]},
            "test",
        )
        self.assertEqual(state.fused.all_to_all_multi_group.call_count, 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["num_groups"], 2)
        self.assertEqual(kwargs["per_group_segment_counts"][state.my_sparse_group], 100)
        self.assertEqual(
            kwargs["input_tensors"][state.my_sparse_group].numel(),
            400,
        )
        self.assertEqual(
            kwargs["output_tensors"][state.my_sparse_group].numel(),
            400,
        )

    def test_output_flat_distinct_from_input_flat_4active(self) -> None:
        """All-to-all is out-of-place: active output flat must differ from input."""
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(400)]},
            {torch.float32: [torch.zeros(400)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        self.assertNotEqual(
            kwargs["input_tensors"][my_g].data_ptr(),
            kwargs["output_tensors"][my_g].data_ptr(),
        )

    def test_helper_slots_are_placeholder_4active(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        # input total 800 -> segment_count 200
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(800)]},
            {torch.float32: [torch.zeros(800)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        in_tensors = kwargs["input_tensors"]
        out_tensors = kwargs["output_tensors"]
        seg = kwargs["per_group_segment_counts"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                self.assertEqual(in_tensors[g].numel(), 800)
                self.assertEqual(out_tensors[g].numel(), 800)
                continue
            # Helper slot is a 1-element placeholder; segment count keeps geometry.
            self.assertEqual(in_tensors[g].numel(), 1)
            self.assertEqual(seg[g], seg[state.my_sparse_group])
            # Helper in/out both alias the single shared placeholder.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), 1)

    def test_active_group_is_group_zero_for_rank0(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        self.assertEqual(state.my_sparse_group, 0)
        self.assertEqual(state.num_sparse_groups, 2)
        self.assertEqual(state.precomputed_active_ranks, [[0, 1, 2, 3], [4, 5, 6, 7]])


class FusedAllToAll4ActiveValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_4active_validation_accepts(self) -> None:
        """all_to_all_multi_group must accept 4 active ranks (no ValueError);
        without a native comm it falls through to RuntimeError. The active
        input/output each hold nActiveRanks x segment_count = 4 x 100, distinct
        (out-of-place)."""
        fused = self._make_fused(rank=0)
        input_tensors = [torch.zeros(400), torch.zeros(400)]
        output_tensors = [torch.zeros(400), torch.zeros(400)]
        per_group_segment_counts = [100, 100]

        with self.assertRaises(RuntimeError) as cm:
            fused.all_to_all_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=2,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=[[0, 1, 2, 3], [4, 5, 6, 7]],
            )
        self.assertNotIsInstance(cm.exception, ValueError)

    def test_4active_in_place_rejected(self) -> None:
        """In-place (input aliases output) for the active group must raise at 4
        active ranks too."""
        fused = self._make_fused(rank=0)
        shared = torch.zeros(400)
        input_tensors = [shared, torch.zeros(400)]
        output_tensors = [shared, torch.zeros(400)]  # aliases input for group 0
        per_group_segment_counts = [100, 100]

        with self.assertRaises(ValueError) as cm:
            fused.all_to_all_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=2,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=[[0, 1, 2, 3], [4, 5, 6, 7]],
                skip_validation=False,
            )
        self.assertIn("in-place", str(cm.exception))


class FlatAllGather4ActiveTest(unittest.TestCase):
    """all_gather_tensors_with_sharded_relay with sparse_group_size=4.

    8-rank node -> 2 groups of 4 active ranks. The active group's input holds
    send_count elements and the output holds nActiveRanks x send_count. Verifies
    the flat-concat path is generic over the active-rank count, in both in-place
    and out-of-place modes.
    """

    def _all_calls(self, state: ShardedRelayState):
        return state.fused.all_gather_multi_group.call_args_list

    def test_single_call_two_groups(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        # input total 100 (send_count); output total 400 (4 x send_count)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(100)]},
            {torch.float32: [torch.zeros(400)]},
            "test",
        )
        self.assertEqual(state.fused.all_gather_multi_group.call_count, 1)
        kwargs = self._all_calls(state)[0].kwargs
        self.assertEqual(kwargs["num_groups"], 2)
        self.assertEqual(kwargs["per_group_send_counts"][state.my_sparse_group], 100)
        self.assertEqual(
            kwargs["input_tensors"][state.my_sparse_group].numel(),
            100,
        )
        self.assertEqual(
            kwargs["output_tensors"][state.my_sparse_group].numel(),
            400,
        )

    def test_helper_slots_are_placeholder_4active(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        # send_count 200 -> output 800
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(800)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        in_tensors = kwargs["input_tensors"]
        out_tensors = kwargs["output_tensors"]
        send = kwargs["per_group_send_counts"]

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                self.assertEqual(in_tensors[g].numel(), 200)
                self.assertEqual(out_tensors[g].numel(), 800)
                continue
            # Helper slot is a 1-element placeholder; send count keeps geometry.
            self.assertEqual(in_tensors[g].numel(), 1)
            self.assertEqual(send[g], send[state.my_sparse_group])
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), 1)

    def test_active_input_is_separate_flat_buffer_4active(self) -> None:
        # Flat-concat scheme: the active group's input is a separate internal
        # flat send buffer (one contiguous tensor per group), distinct from the
        # caller's input tensor (packed into it before the call).
        state = _make_state(rank=1, sparse_group_size=4, local_size=8)
        in_tensor = torch.zeros(100)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [in_tensor]},  # send 100
            {torch.float32: [torch.zeros(400)]},
            "test",
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        in_seg = kwargs["input_tensors"][my_g]
        self.assertEqual(in_seg.numel(), 100)
        self.assertNotEqual(in_seg.data_ptr(), in_tensor.data_ptr())

    def test_in_place_input_aliases_output_own_slot(self) -> None:
        # 4-active in-place: the active input is an owned-slot view into the
        # active output flat buffer at offset my_active_index * send_count.
        state = _make_state(rank=2, sparse_group_size=4, local_size=8)
        send_count = 100
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(send_count)]},
            {torch.float32: [torch.zeros(4 * send_count)]},
            "test",
            in_place=True,
        )
        kwargs = self._all_calls(state)[0].kwargs
        my_g = state.my_sparse_group
        in_seg = kwargs["input_tensors"][my_g]
        out_flat = kwargs["output_tensors"][my_g]
        self.assertEqual(in_seg.numel(), send_count)
        # rank=2 -> my_active_index = 2 within its group.
        self.assertEqual(
            in_seg.data_ptr(),
            out_flat.data_ptr() + 2 * send_count * out_flat.element_size(),
        )

    def test_active_group_is_group_zero_for_rank0(self) -> None:
        state = _make_state(rank=0, sparse_group_size=4, local_size=8)
        self.assertEqual(state.my_sparse_group, 0)
        self.assertEqual(state.num_sparse_groups, 2)
        self.assertEqual(state.precomputed_active_ranks, [[0, 1, 2, 3], [4, 5, 6, 7]])


class FusedAllGather4ActiveValidationTest(unittest.TestCase):
    def _make_fused(self, rank: int = 0):
        try:
            from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
                FusedShardedRelayMultiGroup,
            )
        except ImportError:
            self.skipTest("FusedShardedRelayMultiGroup not available")

        all_active_ranks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        return FusedShardedRelayMultiGroup(
            rcclx_comm=None,
            world_size=8,
            rank=rank,
            all_active_ranks=all_active_ranks,
        )

    def test_4active_validation_accepts(self) -> None:
        """all_gather_multi_group must accept 4 active ranks (no ValueError);
        without a native comm it falls through to RuntimeError. The active input
        holds send_count = 100 and the output holds nActiveRanks x send_count =
        400."""
        fused = self._make_fused(rank=0)
        input_tensors = [torch.zeros(100), torch.zeros(100)]
        output_tensors = [torch.zeros(400), torch.zeros(400)]
        per_group_send_counts = [100, 100]

        with self.assertRaises(RuntimeError) as cm:
            fused.all_gather_multi_group(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                num_groups=2,
                per_group_send_counts=per_group_send_counts,
                all_active_ranks=[[0, 1, 2, 3], [4, 5, 6, 7]],
            )
        self.assertNotIsInstance(cm.exception, ValueError)


# ---------------------------------------------------------------------------
# Tests for the low_precision kwarg
# ---------------------------------------------------------------------------


class LowPrecisionForwardingTest(unittest.TestCase):
    """
    That `low_precision` reaches the backend, and defaults to False.

    This suite is CPU-only and fully mocked, so low precision never actually
    executes here -- forwarding is the only thing it can meaningfully cover, and
    it is worth covering: the kwarg is threaded through four independent wrappers
    and a dropped one would be invisible, because the backend silently declines
    to full precision anyway and the numbers would still be correct.
    """

    def test_allreduce_defaults_to_full_precision(self) -> None:
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(100)]}, "test"
        )
        kwargs = state.fused.allreduce_multi_group.call_args_list[0].kwargs
        self.assertFalse(kwargs["low_precision"])

    def test_allreduce_forwards_low_precision(self) -> None:
        state = _make_state(rank=0)
        allreduce_tensors_with_sharded_relay(
            state, {torch.float32: [torch.zeros(100)]}, "test", low_precision=True
        )
        kwargs = state.fused.allreduce_multi_group.call_args_list[0].kwargs
        self.assertTrue(kwargs["low_precision"])

    def test_reduce_scatter_forwards_low_precision(self) -> None:
        state = _make_state(rank=0)
        reduce_scatter_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(100)]},
            "test",
            low_precision=True,
        )
        kwargs = state.fused.reduce_scatter_multi_group.call_args_list[0].kwargs
        self.assertTrue(kwargs["low_precision"])

    def test_all_to_all_forwards_low_precision(self) -> None:
        state = _make_state(rank=0)
        all_to_all_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(200)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
            low_precision=True,
        )
        kwargs = state.fused.all_to_all_multi_group.call_args_list[0].kwargs
        self.assertTrue(kwargs["low_precision"])

    def test_all_gather_forwards_low_precision(self) -> None:
        state = _make_state(rank=0)
        all_gather_tensors_with_sharded_relay(
            state,
            {torch.float32: [torch.zeros(100)]},
            {torch.float32: [torch.zeros(200)]},
            "test",
            low_precision=True,
        )
        kwargs = state.fused.all_gather_multi_group.call_args_list[0].kwargs
        self.assertTrue(kwargs["low_precision"])


if __name__ == "__main__":
    unittest.main()
