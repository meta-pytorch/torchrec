#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import unittest
from typing import Any, Dict, List, Optional

from torchrec.distributed.batched_embedding_kernel import _populate_res_params
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.modules.embedding_configs import DataType, PoolingType


class PopulateResParamsTest(unittest.TestCase):
    """`_populate_res_params` reads RES knobs out of `fused_params`.

    Whatever it leaves behind is splatted into
    `SplitTableBatchedEmbeddingBagsCodegen.__init__`, which takes no
    `**kwargs`, so a key that is read but not removed is a TypeError on every
    rank at construction -- not a bad value, a crash.
    """

    def _config(
        self,
        table_names: List[str],
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> GroupedEmbeddingConfig:
        return GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.NONE,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=EmbeddingComputeKernel.FUSED,
            embedding_tables=[
                ShardedEmbeddingTable(
                    name=name,
                    data_type=DataType.FP32,
                    pooling=PoolingType.NONE,
                    has_feature_processor=False,
                    feature_names=[f"feature_{i}"],
                    compute_kernel=EmbeddingComputeKernel.FUSED,
                    embedding_dim=16,
                    num_embeddings=64,
                )
                for i, name in enumerate(table_names)
            ],
            fused_params=fused_params,
        )

    _HBM_KNOBS: Dict[str, Any] = {
        "enable_hbm_streaming": True,
        "res_hbm_drain_interval": 200,
        "res_use_copy_done_token": True,
    }

    def test_hbm_knobs_reach_res_params(self) -> None:
        fused_params: Dict[str, Any] = {
            "enable_raw_embedding_streaming": True,
            "res_enabled_tables": "t0",
            **self._HBM_KNOBS,
        }
        enable_res, res_params = _populate_res_params(
            self._config(["t0"], fused_params)
        )
        self.assertTrue(enable_res)
        self.assertTrue(res_params.enable_hbm_streaming)
        self.assertEqual(res_params.res_hbm_drain_interval, 200)
        self.assertTrue(res_params.res_use_copy_done_token)

    def test_knobs_are_removed_even_when_res_is_disabled(self) -> None:
        # The crash case, and the reason this test exists. Both early returns
        # in `_populate_res_params` are reachable on a live model: streaming
        # off entirely, and streaming on for a TBE group holding none of the
        # allowlisted tables. Tables are NOT grouped by RES enablement, so on a
        # model allowlisting one table every other group takes the second
        # return. A knob read after those returns stays in the dict and
        # reaches the TBE constructor.
        for name, fused_params in (
            (
                "streaming off",
                {"enable_raw_embedding_streaming": False, **self._HBM_KNOBS},
            ),
            (
                "no allowlisted table in this group",
                {
                    "enable_raw_embedding_streaming": True,
                    "res_enabled_tables": "elsewhere",
                    **self._HBM_KNOBS,
                },
            ),
        ):
            with self.subTest(name):
                enable_res, _ = _populate_res_params(self._config(["t0"], fused_params))
                self.assertFalse(enable_res)
                for key in self._HBM_KNOBS:
                    self.assertNotIn(key, fused_params)

    # A real name from the RES table set: feature-store derived, with a comma
    # inside the brace-delimited event spec.
    _COMMA_NAME = (
        "VIEWER_CLIPS_LAST_N_ENGAGED_MEDIA_NOT_INTERESTED VIEWER INF ONLINE "
        "{ EVENT_TYPE=ig_unified_not_interested, ACTION_SOURCE=three_dot_menu }_fs1003"
    )

    def _decode_enabled_tables(self, encoded: str) -> List[str]:
        """Decode via the real entry point, on a group holding every candidate
        table so the allowlist-intersection early return cannot mask the result.
        """
        fused_params: Dict[str, Any] = {
            "enable_raw_embedding_streaming": True,
            "res_enabled_tables": encoded,
        }
        _, res_params = _populate_res_params(
            self._config(["a", "b", self._COMMA_NAME], fused_params)
        )
        # Same crash condition as the test above: the key must not survive.
        self.assertNotIn("res_enabled_tables", fused_params)
        return res_params.res_enabled_tables

    def test_comma_joined_encoding_still_decodes(self) -> None:
        # The default encoding; must keep working unchanged.
        self.assertEqual(self._decode_enabled_tables("a,b"), ["a", "b"])

    def test_json_encoding_decodes(self) -> None:
        self.assertEqual(self._decode_enabled_tables('["a", "b"]'), ["a", "b"])

    def test_comma_bearing_name_survives_json_but_not_comma_joining(self) -> None:
        # The reason the JSON encoding exists.
        self.assertEqual(
            self._decode_enabled_tables(json.dumps([self._COMMA_NAME])),
            [self._COMMA_NAME],
        )
        # Comma-joined, the same name splits into two that match no table.
        self.assertNotIn(
            self._COMMA_NAME, self._decode_enabled_tables(self._COMMA_NAME)
        )

    def test_single_name_unaffected_by_either_encoding(self) -> None:
        self.assertEqual(self._decode_enabled_tables("a"), ["a"])
        self.assertEqual(self._decode_enabled_tables('["a"]'), ["a"])

    def test_the_two_empty_encodings_do_not_mean_the_same_thing(self) -> None:
        # The allowlist is only applied when it is non-empty, so an empty list
        # means "no filter" and every table in the group streams. Only the JSON
        # form can express that. The comma-joined form cannot: "".split(",") is
        # [""], a one-name allowlist that matches nothing and switches
        # streaming off -- the opposite of what an empty allowlist means.
        # Pinned rather than fixed, since changing it would silently turn
        # streaming on for any model that sends an empty list.
        for encoded, expected_enabled, expected_tables in (
            ("[]", True, []),
            ("", False, [""]),
        ):
            with self.subTest(encoded):
                fused_params: Dict[str, Any] = {
                    "enable_raw_embedding_streaming": True,
                    "res_enabled_tables": encoded,
                }
                enable_res, res_params = _populate_res_params(
                    self._config(["a", "b"], fused_params)
                )
                self.assertEqual(enable_res, expected_enabled)
                self.assertEqual(res_params.res_enabled_tables, expected_tables)
                self.assertNotIn("res_enabled_tables", fused_params)

    def test_malformed_json_list_raises(self) -> None:
        # Non-string elements would compare unequal to every table name and
        # disable streaming silently, so they have to fail loudly instead.
        for encoded in ("[123]", '["a", 2]'):
            with self.subTest(encoded):
                with self.assertRaisesRegex(ValueError, "JSON list of strings"):
                    self._decode_enabled_tables(encoded)


if __name__ == "__main__":
    unittest.main()
