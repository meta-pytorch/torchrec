#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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


if __name__ == "__main__":
    unittest.main()
