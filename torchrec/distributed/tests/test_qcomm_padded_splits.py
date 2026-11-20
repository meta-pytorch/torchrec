#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import unittest
from typing import Any, List, Optional
from unittest.mock import patch

import torchrec.distributed.comm_ops as comm_ops
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)

_MX4_GROUP_SIZE = 8
_PRECISIONS = (
    CommType.FP32,
    CommType.FP16,
    CommType.BF16,
    CommType.FP8,
    CommType.MX4,
)
_SPLITS = ([], [0, 256], [1, 257], [108, 180, 96])


def _codec(precision: CommType) -> Any:
    return get_qcomm_codecs(
        QCommsConfig(
            forward_precision=precision,
            backward_precision=precision,
            fp8_quantize_dim=256 if precision == CommType.FP8 else None,
            fp8_quantize_dim_bwd=256 if precision == CommType.FP8 else None,
            mx4_quantize_dim=_MX4_GROUP_SIZE if precision == CommType.MX4 else None,
            mx4_quantize_dim_bwd=(
                _MX4_GROUP_SIZE if precision == CommType.MX4 else None
            ),
        )
    ).backward


def _fp8_only_padded_splits(
    codec: Any, qcomm_ctx: Any, splits: List[int]
) -> Optional[List[int]]:
    """The helper as it was before MX4 support (`_fp8_rowwise_padded_splits`)."""
    if (
        not bool(int(os.environ.get("TORCHREC_ENABLE_FP8_ROWWISE_PADDING", 0)))
        or qcomm_ctx is None
        or getattr(qcomm_ctx, "row_dim", -1) <= 0
        or getattr(codec, "_comm_precision", None) != SparseType.FP8
    ):
        return None
    row_dim = qcomm_ctx.row_dim
    padded = [split + (row_dim - split % row_dim) % row_dim for split in splits]
    return padded if padded != splits else None


class _NoPrecisionAccess:
    @property
    def _comm_precision(self) -> Any:
        raise AssertionError("Disabled padding must not inspect codec precision")


class TestQCommPaddedSplits(unittest.TestCase):
    def _padded(
        self, precision: CommType, splits: List[int], mx4: str, fp8: str
    ) -> Optional[List[int]]:
        codec = _codec(precision)
        env = {
            "TORCHREC_ENABLE_MX4_PADDING": mx4,
            "TORCHREC_ENABLE_FP8_ROWWISE_PADDING": fp8,
        }
        with patch.dict(os.environ, env):
            return comm_ops._qcomm_padded_splits(codec, codec.create_context(), splits)

    def test_matches_fp8_only_helper_unless_mx4_padding_applies(self) -> None:
        for precision, fp8, mx4, use_context, splits in itertools.product(
            _PRECISIONS, ("0", "1"), ("0", "1"), (False, True), _SPLITS
        ):
            codec = _codec(precision)
            context = codec.create_context() if use_context else None
            env = {
                "TORCHREC_ENABLE_FP8_ROWWISE_PADDING": fp8,
                "TORCHREC_ENABLE_MX4_PADDING": mx4,
            }
            with patch.dict(os.environ, env), self.subTest(
                precision=precision,
                fp8=fp8,
                mx4=mx4,
                context=use_context,
                splits=splits,
            ):
                expected = _fp8_only_padded_splits(codec, context, splits)
                if mx4 == "1" and precision == CommType.MX4 and context is not None:
                    self.assertIsNone(expected)
                    rounded = [s + -s % _MX4_GROUP_SIZE for s in splits]
                    expected = rounded if rounded != splits else None
                self.assertEqual(
                    comm_ops._qcomm_padded_splits(codec, context, splits), expected
                )

    def test_mx4_flag_pads_misaligned_splits(self) -> None:
        self.assertEqual(
            self._padded(CommType.MX4, [108, 180, 0, 96], mx4="1", fp8="0"),
            [112, 184, 0, 96],
        )

    def test_mx4_aligned_splits_are_unchanged(self) -> None:
        self.assertIsNone(self._padded(CommType.MX4, [96, 160, 0], mx4="1", fp8="0"))

    def test_mx4_padding_is_off_by_default(self) -> None:
        codec = _codec(CommType.MX4)
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "TORCHREC_ENABLE_MX4_PADDING",
                "TORCHREC_ENABLE_FP8_ROWWISE_PADDING",
            )
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(
                comm_ops._qcomm_padded_splits(codec, codec.create_context(), [108])
            )

    def test_flags_only_affect_their_own_precision(self) -> None:
        self.assertIsNone(self._padded(CommType.MX4, [108], mx4="0", fp8="1"))
        self.assertIsNone(self._padded(CommType.FP8, [100], mx4="1", fp8="0"))
        self.assertEqual(self._padded(CommType.FP8, [100], mx4="1", fp8="1"), [256])

    def test_mx4_rejects_misaligned_split_sizes(self) -> None:
        codec = _codec(CommType.MX4)
        with self.assertRaisesRegex(AssertionError, "multiple of group_size"):
            codec.calc_quantized_size(108, codec.create_context())

    def test_disabled_padding_does_not_inspect_codec(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TORCHREC_ENABLE_MX4_PADDING": "0",
                "TORCHREC_ENABLE_FP8_ROWWISE_PADDING": "0",
            },
        ):
            self.assertIsNone(
                comm_ops._qcomm_padded_splits(
                    _NoPrecisionAccess(), _codec(CommType.MX4).create_context(), [1]
                )
            )
