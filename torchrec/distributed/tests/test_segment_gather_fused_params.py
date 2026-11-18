#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchrec.distributed.fused_params import (
    FUSED_PARAM_SEGMENT_GATHER_BACKWARD,
    split_fused_param_segment_gather_backward,
    tbe_fused_params,
)


class SegmentGatherBackwardFusedParamTest(unittest.TestCase):
    def test_absent_leaves_fused_params_unchanged(self) -> None:
        self.assertEqual(split_fused_param_segment_gather_backward(None), (False, None))
        fused_params = {"cache_load_factor": 0.2}
        enabled, rest = split_fused_param_segment_gather_backward(fused_params)
        self.assertFalse(enabled)
        self.assertIs(rest, fused_params)

    def test_present_is_read_and_stripped(self) -> None:
        for value in (True, False):
            fused_params = {
                "cache_load_factor": 0.2,
                FUSED_PARAM_SEGMENT_GATHER_BACKWARD: value,
            }
            enabled, rest = split_fused_param_segment_gather_backward(fused_params)
            self.assertEqual(enabled, value)
            self.assertEqual(rest, {"cache_load_factor": 0.2})
            # The caller's dict, which other sharders may share, is not modified.
            self.assertIn(FUSED_PARAM_SEGMENT_GATHER_BACKWARD, fused_params)

    def test_tbe_fused_params_drops_the_key(self) -> None:
        rest = tbe_fused_params(
            {"cache_load_factor": 0.2, FUSED_PARAM_SEGMENT_GATHER_BACKWARD: True}
        )
        self.assertEqual(rest, {"cache_load_factor": 0.2})
