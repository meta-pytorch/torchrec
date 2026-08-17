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
    Tests for the grow-only flat buffer caches (_active_flat_cache and
    _helper_flat_cache) that replaced the old per-tensor scratch scheme.

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
    _get_helper_flat_buf,
    _passthrough_helper_size,
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
    """Tests for _get_active_flat_buf and _get_helper_flat_buf."""

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

    # --- _get_helper_flat_buf ---

    def test_helper_flat_buf_exact_size_on_first_call(self) -> None:
        state = _make_state()
        buf = _get_helper_flat_buf(state, 1, 100, torch.float32, _DEVICE)
        self.assertEqual(buf.numel(), 100)
        self.assertEqual(buf.dtype, torch.float32)

    def test_helper_flat_buf_reused_across_training_steps(self) -> None:
        state = _make_state()
        buf1 = _get_helper_flat_buf(state, 1, 200, torch.float32, _DEVICE)
        buf2 = _get_helper_flat_buf(state, 1, 200, torch.float32, _DEVICE)
        self.assertEqual(buf1.data_ptr(), buf2.data_ptr())

    def test_helper_flat_buf_separate_per_group(self) -> None:
        """With per-(group, dtype) keying, different group_idx values get separate buffers."""
        state = _make_state()
        buf0 = _get_helper_flat_buf(state, 1, 100, torch.float32, _DEVICE)
        buf1 = _get_helper_flat_buf(state, 2, 100, torch.float32, _DEVICE)
        self.assertNotEqual(buf0.data_ptr(), buf1.data_ptr())

    def test_helper_flat_buf_separate_per_dtype(self) -> None:
        """fp16 (weights) and fp32 (optimizer states) must not evict each other."""
        state = _make_state()
        fp16 = _get_helper_flat_buf(state, 1, 100, torch.float16, _DEVICE)
        fp32 = _get_helper_flat_buf(state, 1, 100, torch.float32, _DEVICE)
        fp16_again = _get_helper_flat_buf(state, 1, 100, torch.float16, _DEVICE)
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

    def test_helper_flat_buffer_total_numel_matches_passthrough_size(self) -> None:
        """
        With the passthrough kernel, helper buffers are sized to
        nActiveRanks × chunkSize (much smaller than the full per-group total).
        Each helper group has its own buffer (no aliasing).
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

        # Helper groups: per_group_sizes has the full total_g, but tensor
        # numel is the passthrough size (nActiveRanks × chunkSize).
        # All helper groups have the same total_g (fallback: all equal my_total).
        num_chunks = (state.local_size - state.sparse_group_size) + 1
        expected_helper_numel = _passthrough_helper_size(
            expected_total, state.sparse_group_size, num_chunks
        )
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
                expected_helper_numel,
                f"group={g}: helper tensor numel should be passthrough size",
            )
            helper_ptrs.add(iter_tensors[g].data_ptr())

        # Each helper group has its OWN buffer (no aliasing under phase-sync).
        self.assertEqual(
            len(helper_ptrs),
            state.num_sparse_groups - 1,
            f"Expected {state.num_sparse_groups - 1} distinct helper buffers, "
            f"got {len(helper_ptrs)}",
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
        The helper buffer is keyed by (group_idx, dtype), so each (group, dtype)
        pair has its own buffer.
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
    # Passthrough helper buffer tests (per-group keying, no aliasing)
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

    def test_helper_buffers_separate_per_group(self) -> None:
        """Each helper-group slot has its own data_ptr; active does NOT alias helpers."""
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
            state.num_sparse_groups - 1,
            "Each helper group should have its own buffer",
        )
        self.assertNotIn(
            active_ptr, helper_ptrs, "Active buffer must not alias helpers"
        )

    def test_helper_buffer_sized_to_passthrough_minimum(self) -> None:
        """
        Drive an allreduce with heterogeneous per-group totals via mocked
        allgather; assert each helper buffer is passthrough-sized.
        """
        state = _make_state(rank=0)
        state = dataclasses.replace(state, intra_node_pytorch_pg=MagicMock())

        # Group totals: [100, 300, 200, 150].  Rank 0 is active for group 0.
        group_totals = [100, 300, 200, 150]

        def _allgather_side_effect(tensor_list, _tensor, **_kwargs) -> None:
            for r, t in enumerate(tensor_list):
                t.fill_(group_totals[r // state.sparse_group_size])

        num_chunks = (state.local_size - state.sparse_group_size) + 1

        with patch("torchrec.distributed.sharded_relay_utils.dist") as mock_dist:
            mock_dist.all_gather.side_effect = _allgather_side_effect
            mock_dist.ReduceOp = dist.ReduceOp
            allreduce_tensors_with_sharded_relay(
                state, {torch.float32: [torch.zeros(100)]}, "hetero"
            )

        call_kwargs = self._all_calls(state)[0].kwargs
        iter_tensors = call_kwargs["tensors"]
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            expected_size = _passthrough_helper_size(
                group_totals[g], state.sparse_group_size, num_chunks
            )
            self.assertEqual(
                iter_tensors[g].numel(),
                expected_size,
                f"group={g}: helper should be passthrough-sized",
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

    def test_bm_fm_real_totals_per_group_helper_buffers(self) -> None:
        """
        Using real BM-FM per-group totals, assert that each helper group has
        its own passthrough-sized buffer.
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

        num_chunks = (state.local_size - state.sparse_group_size) + 1

        captured_helper_calls: list[tuple[int, int]] = []

        def _fake_helper(_state, group_idx, total, _dtype, _device):
            captured_helper_calls.append((group_idx, total))
            return torch.empty(total, dtype=_dtype, device="meta")

        def _fake_active(_state, total, _dtype, _device):
            return torch.zeros(1, dtype=_dtype)

        with patch(
            "torchrec.distributed.sharded_relay_utils._get_active_flat_buf",
            side_effect=_fake_active,
        ), patch(
            "torchrec.distributed.sharded_relay_utils._get_helper_flat_buf",
            side_effect=_fake_helper,
        ), patch(
            "torchrec.distributed.sharded_relay_utils.dist"
        ) as mock_dist:
            mock_dist.all_gather.side_effect = _allgather_side_effect
            mock_dist.ReduceOp = dist.ReduceOp
            tensors = [torch.zeros(1, dtype=torch.float16)]
            allreduce_tensors_with_sharded_relay(
                state, {torch.float16: tensors}, "bm_fm_2d_weight_sync"
            )

        # 3 calls to _get_helper_flat_buf (one per helper group, no aliasing)
        self.assertEqual(
            len(captured_helper_calls),
            3,
            f"Expected 3 helper buffer allocations, got {len(captured_helper_calls)}",
        )
        # Each helper buffer should be passthrough-sized
        for group_idx, total in captured_helper_calls:
            expected = _passthrough_helper_size(
                bm_fm_fp16_totals[group_idx], state.sparse_group_size, num_chunks
            )
            self.assertEqual(
                total,
                expected,
                f"group={group_idx}: helper buffer should be passthrough-sized",
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

    def test_raises_value_error_on_tensor_size_mismatch(self) -> None:
        """
        allreduce_multi_group must raise ValueError when tensor.numel() does
        not match per_group_sizes[g].  This is the validation that catches the
        bug where a 640M-element scratch buffer was passed with count=10M.
        """
        fused = self._make_fused(rank=0)
        tensors = [
            torch.zeros(1000),  # group 0 — matches
            torch.zeros(640),  # group 1 — will mismatch (expected 500)
            torch.zeros(750),  # group 2 — matches
            torch.zeros(600),  # group 3 — matches
        ]
        per_group_sizes = [1000, 500, 750, 600]  # group 1: 640 vs 500

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


class _PassthroughHelperSizePolicyTest(unittest.TestCase):
    """Tests for _passthrough_helper_size — Python ↔ C++ formula parity."""

    def test_passthrough_size_matches_2x_chunkSize_for_realistic_totals(self) -> None:
        """BM-FM fp16: ~12B elements per group, 8 ranks, 2 active → numChunks=7."""
        # Worked out by hand from the contract rather than re-derived with the
        # implementation's own expression -- recomputing
        # sparse_group_size * align_down(total_g // num_chunks, 128) here would
        # move both sides of the assertion together and could never fail:
        #
        #   12_002_982_488 // 7 = 1_714_711_784   (remainder 0)
        #   align_down(.., 128) = 1_714_711_680   (trims 104 elements)
        #   2 * 1_714_711_680   = 3_429_423_360   (< total_g, so no clamp)
        result = _passthrough_helper_size(12_002_982_488, 2, 7)
        self.assertEqual(result, 3_429_423_360)
        # Each of the 2 slots is a whole number of 128-element chunks, which is
        # the property the alignment exists for.
        self.assertEqual(result % (2 * 128), 0)

    def test_passthrough_size_falls_back_to_total_for_tiny_counts(self) -> None:
        """When total_g < num_chunks * CACHE_LINE_SIZE, chunkSize falls back to total_g."""
        total_g = 100
        sparse_group_size = 2
        num_chunks = 7
        result = _passthrough_helper_size(total_g, sparse_group_size, num_chunks)
        # chunk = 100 // 7 = 14, chunk_aligned = (14 // 128) * 128 = 0
        # fallback: chunk_aligned = total_g = 100
        # result = min(100, 2 * 100) = 100
        self.assertEqual(result, total_g)

    def test_passthrough_size_capped_at_total_g(self) -> None:
        """Result must never exceed total_g."""
        total_g = 128
        sparse_group_size = 2
        num_chunks = 7
        result = _passthrough_helper_size(total_g, sparse_group_size, num_chunks)
        self.assertLessEqual(result, total_g)

    def test_python_meets_cpp_min_required_at_alignment_boundary(self) -> None:
        """At exact alignment boundaries, Python and C++ formulas agree."""
        # total_g = 7 * 128 * 1000 = 896_000 divides evenly by num_chunks * 128,
        # so nothing is lost to alignment: 896_000 // 7 = 128_000, which is
        # already 128-aligned, and 2 * 128_000 = 256_000 stays under total_g.
        self.assertEqual(_passthrough_helper_size(7 * 128 * 1000, 2, 7), 256_000)

    def test_total_per_rank_helper_memory_is_6x_chunkSize(self) -> None:
        """
        For 4 groups / 2 active per group: each rank helps 3 groups.
        Total helper memory = 3 × 2 × chunkSize = 6 × chunkSize.
        """
        total_g = 12_002_982_488  # BM-FM fp16 group 0
        sparse_group_size = 2
        num_chunks = 7
        helper_per_group = _passthrough_helper_size(
            total_g, sparse_group_size, num_chunks
        )
        # Literals rather than a re-derivation of the implementation's own
        # expression, which would make these assertions unfalsifiable:
        # align_down(12_002_982_488 // 7, 128) = 1_714_711_680, and the helper
        # buffer holds 2 of those chunks.
        self.assertEqual(helper_per_group, 3_429_423_360)
        # 3 helper groups per rank, so 3 x 2 = 6 aligned chunks in total:
        # 6 * 1_714_711_680.
        self.assertEqual(3 * helper_per_group, 10_288_270_080)


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

    def test_helper_buffers_passthrough_sized_and_distinct(self) -> None:
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
        num_chunks = (state.local_size - state.sparse_group_size) + 1

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            expected = _passthrough_helper_size(
                recv[g], state.sparse_group_size, num_chunks
            )
            self.assertEqual(in_tensors[g].numel(), expected)
            # Helper uses one scratch buffer for both send and recv.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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

    def test_helper_buffers_passthrough_sized_and_distinct(self) -> None:
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
        num_chunks = (state.local_size - state.sparse_group_size) + 1

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            expected = _passthrough_helper_size(
                seg[g], state.sparse_group_size, num_chunks
            )
            self.assertEqual(in_tensors[g].numel(), expected)
            # Helper uses one scratch buffer for both send and recv.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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

    def test_helper_buffers_passthrough_sized_and_distinct(self) -> None:
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
        num_chunks = (state.local_size - state.sparse_group_size) + 1

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                continue
            expected = _passthrough_helper_size(
                send[g], state.sparse_group_size, num_chunks
            )
            self.assertEqual(in_tensors[g].numel(), expected)
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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

    def test_helper_buffers_passthrough_sized_4active(self) -> None:
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
            # Flat A>2 allreduce: the util sizes the helper 2*total_g
            # ((A+1)*oChunk <= 1.25*total_g), not the recursive _passthrough.
            expected = 2 * iter_sizes[g]
            self.assertEqual(iter_tensors[g].numel(), expected)
            helper_ptrs.add(iter_tensors[g].data_ptr())

        # Each rank is helper for exactly num_sparse_groups - 1 == 1 group.
        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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


class _PassthroughHelperSize4ActivePolicyTest(unittest.TestCase):
    """_passthrough_helper_size against hand-computed sizes, not the formula.

    Every expected value below is worked out by hand from the documented
    contract and written in as a literal. Re-deriving it with the same
    ``min(total_g, A * align_down(total_g // num_chunks, 128))`` expression the
    implementation uses would make the assertion unfalsifiable: any change to
    the formula would move both sides together and the test would still pass.
    The three branches of that contract are covered -- ordinary alignment loss,
    the ``min()`` clamp, and the ``chunk_aligned == 0`` fallback.
    """

    def test_passthrough_size_4active(self) -> None:
        # 4-active on an 8-rank node: num_chunks = (8 - 4) + 1 = 5.
        #
        # 12_002_982_488 // 5      = 2_400_596_497   (remainder 3)
        # align_down(.., 128)      = 2_400_596_480   (trims 17 elements)
        # 4 * 2_400_596_480        = 9_602_385_920   (< total_g, so no clamp)
        self.assertEqual(_passthrough_helper_size(12_002_982_488, 4, 5), 9_602_385_920)
        # A total that is already a multiple of num_chunks * 128 loses nothing to
        # alignment, and lands on the same size because the case above only had a
        # sub-128 remainder to trim.
        self.assertEqual(_passthrough_helper_size(12_002_982_400, 4, 5), 9_602_385_920)

    def test_passthrough_size_per_slot_is_chunk_aligned(self) -> None:
        # The buffer holds sparse_group_size slots, so whenever min() has not
        # clamped, each slot must be a whole number of 128-element chunks. This
        # is the property the 128-alignment exists for.
        for total_g, sparse_group_size, num_chunks in (
            (12_002_982_488, 4, 5),
            (12_002_982_488, 2, 7),
            (1_000_000, 4, 5),
        ):
            with self.subTest(total_g=total_g, A=sparse_group_size):
                result = _passthrough_helper_size(
                    total_g, sparse_group_size, num_chunks
                )
                self.assertLess(result, total_g, "expected no min() clamp here")
                self.assertEqual(result % (sparse_group_size * 128), 0)

    def test_passthrough_size_clamps_to_total(self) -> None:
        # sparse_group_size >= num_chunks makes A * chunk_aligned exceed total_g,
        # so min() clamps: 10_000 // 2 = 5_000 -> 4_992 -> 4 * 4_992 = 19_968,
        # which is larger than total_g.
        self.assertEqual(_passthrough_helper_size(10_000, 4, 2), 10_000)

    def test_passthrough_size_falls_back_below_one_chunk(self) -> None:
        # 500 // 5 = 100, and align_down(100, 128) == 0, so the fallback replaces
        # chunk_aligned with total_g; min() then clamps the product back to 500.
        self.assertEqual(_passthrough_helper_size(500, 4, 5), 500)
        # 2-active production shape, for contrast: num_chunks = (8 - 2) + 1 = 7,
        # 12_002_982_488 // 7 = 1_714_711_784 -> 1_714_711_680 -> x2.
        self.assertEqual(_passthrough_helper_size(12_002_982_488, 2, 7), 3_429_423_360)


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

    def test_helper_buffers_sized_to_2x_recv_count_4active(self) -> None:
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
            # A>2 reduce-scatter reduces at the helper and retains the
            # 2 * recvCount helper-buffer contract.
            expected = 2 * recv[g]
            self.assertEqual(in_tensors[g].numel(), expected)
            # Helper uses one scratch buffer for both send and recv.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        # Each rank is helper for exactly num_sparse_groups - 1 == 1 group.
        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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

    def test_helper_buffers_passthrough_sized_4active(self) -> None:
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

        helper_ptrs = set()
        for g in range(state.num_sparse_groups):
            if g == state.my_sparse_group:
                self.assertEqual(in_tensors[g].numel(), 800)
                self.assertEqual(out_tensors[g].numel(), 800)
                continue
            # Flat A>2 all-to-all is pure-direct: helpers do no work, so the
            # util passes a tiny placeholder (size 1), not a full helper buffer.
            self.assertEqual(in_tensors[g].numel(), 1)
            # Helper uses one scratch buffer for both send and recv.
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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

    def test_helper_buffers_passthrough_sized_4active(self) -> None:
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
            # Flat A>2 all-gather: the util sizes the helper A*send_count
            # (matches the kernel's A*cs broadcast scratch), not _passthrough.
            expected = state.sparse_group_size * send[g]
            self.assertEqual(in_tensors[g].numel(), expected)
            self.assertEqual(in_tensors[g].data_ptr(), out_tensors[g].data_ptr())
            helper_ptrs.add(in_tensors[g].data_ptr())

        self.assertEqual(len(helper_ptrs), state.num_sparse_groups - 1)

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


if __name__ == "__main__":
    unittest.main()
