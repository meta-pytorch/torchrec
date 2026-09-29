#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchrec.distributed.maglev.customized_pipeline import (
    ScheduleError,
    ScheduleValidator,
)

# What MaglevRail._dense_zero_bubble actually runs, for p=4 m=8. s0 is
# transcribed from the rank-0 Kineto trace of a MAST job; s1..s3 from the op
# order in that method's docstring. The generator exists to reproduce these, so
# they are the regression that keeps a given order and a derived one the same
# thing said two ways.
ZERO_BUBBLE_P4_M8 = [
    "F0 F1 F2 F3 I0 W0 F4 I1 W1 F5 I2 W2 F6 I3 W3 F7 I4 W4 I5 W5 I6 W6 I7 W7",
    "F0 F1 F2 I0 F3 I1 W0 F4 I2 W1 F5 I3 W2 F6 I4 W3 F7 I5 W4 I6 W5 I7 W6 W7",
    "F0 F1 I0 F2 I1 F3 I2 W0 F4 I3 W1 F5 I4 W2 F6 I5 W3 F7 I6 W4 I7 W5 W6 W7",
    "F0 I0 F1 I1 F2 I2 F3 I3 W0 F4 I4 W1 F5 I5 W2 F6 I6 W3 F7 I7 W4 W5 W6 W7",
]


class GenerateTest(unittest.TestCase):
    def test_reproduces_the_derived_order(self) -> None:
        self.assertEqual(ScheduleValidator.generate(4, 8), ZERO_BUBBLE_P4_M8)

    def test_deeper_stages_defer_weight_work_longer(self) -> None:
        # The direction is easy to get backwards, so pin it: stage 0 runs every
        # W straight after its own I, stage 3 holds the first one back.
        rows = ScheduleValidator.generate(4, 8)
        self.assertTrue(rows[0].startswith("F0 F1 F2 F3 I0 W0"))
        self.assertTrue(rows[3].startswith("F0 I0 F1 I1 F2 I2 F3 I3 W0"))

    def test_generated_order_validates(self) -> None:
        rows = ScheduleValidator.generate(4, 8)
        ScheduleValidator.validate(ScheduleValidator.parse(rows), 4, 8)

    def test_round_trip(self) -> None:
        self.assertEqual(
            ScheduleValidator.format_rows(ScheduleValidator.parse(ZERO_BUBBLE_P4_M8)),
            ZERO_BUBBLE_P4_M8,
        )


class ValidateTest(unittest.TestCase):
    def test_rejects_an_unparseable_token(self) -> None:
        with self.assertRaisesRegex(ScheduleError, "bad action token"):
            ScheduleValidator.parse(["F0 X1"])

    def test_rejects_a_monolithic_backward(self) -> None:
        # Rail's seams are popped by I and by nothing else, so a B would leave
        # them out of step and the sparse graph resuming from gradients nobody
        # filled. It is not part of the grammar.
        with self.assertRaisesRegex(ScheduleError, "bad action token"):
            ScheduleValidator.parse(["F0 B0"])

    def test_rejects_an_index_that_disagrees_with_the_queue_order(self) -> None:
        with self.assertRaisesRegex(ScheduleError, "out of order"):
            ScheduleValidator.validate(
                ScheduleValidator.parse(["F0 F2 F1 I0 W0 I1 W1 I2 W2"]), 1, 3
            )

    def test_rejects_an_input_grad_with_no_outstanding_forward(self) -> None:
        with self.assertRaisesRegex(ScheduleError, "no forward to consume"):
            ScheduleValidator.validate(ScheduleValidator.parse(["I0 F0 W0"]), 1, 1)

    def test_rejects_weight_work_before_its_input_grad(self) -> None:
        with self.assertRaisesRegex(ScheduleError, "no deferred"):
            ScheduleValidator.validate(ScheduleValidator.parse(["F0 W0 I0"]), 1, 1)

    def test_rejects_a_row_that_drops_a_microbatch(self) -> None:
        with self.assertRaisesRegex(ScheduleError, "expected 3"):
            ScheduleValidator.validate(
                ScheduleValidator.parse(["F0 F1 I0 W0 I1 W1"] * 2), 2, 3
            )

    def test_rejects_the_wrong_number_of_rows(self) -> None:
        with self.assertRaisesRegex(ScheduleError, "one per stage"):
            ScheduleValidator.validate(
                ScheduleValidator.parse(ScheduleValidator.generate(2, 4)), 4, 4
            )


class SimulateTest(unittest.TestCase):
    def test_detects_a_deadlock_no_single_row_reveals(self) -> None:
        # Each row on its own is legal -- right counts, forwards before their
        # backwards. Together stage 0 waits on a gradient stage 1 cannot send
        # until it receives an activation stage 0 is too blocked to send.
        rows = [
            "F0 I0 W0 F1 I1 W1",
            "F0 F1 I0 W0 I1 W1",
        ]
        for stage, row in enumerate(rows):
            ScheduleValidator.validate_row(
                ScheduleValidator.parse_row(row, stage), stage, 2
            )
        with self.assertRaisesRegex(ScheduleError, "deadlocks"):
            ScheduleValidator.validate(ScheduleValidator.parse(rows), 2, 2)

    def test_a_shallower_warmup_is_legal_and_costs_bubble(self) -> None:
        # The memory lever: one less microbatch in flight on stage 0, paid for
        # in idle slots rather than rejected.
        default = ScheduleValidator.generate(2, 4)
        shallow = ["F0 I0 W0 F1 I1 W1 F2 I2 W2 F3 I3 W3", default[1]]
        self.assertGreater(
            len(ScheduleValidator.validate(ScheduleValidator.parse(shallow), 2, 4)[0]),
            len(ScheduleValidator.validate(ScheduleValidator.parse(default), 2, 4)[0]),
        )

    def test_format_grid_marks_idle_slots(self) -> None:
        timeline = ScheduleValidator.validate(
            ScheduleValidator.parse(ScheduleValidator.generate(4, 8)), 4, 8
        )
        self.assertIn("s3 | .. .. ..", ScheduleValidator.format_grid(timeline))
