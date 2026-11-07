# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock, patch

import torch
from torchrec_dynamic_embedding.ps import PSCollection


class TestPSCollectionOptimizerState(unittest.TestCase):
    def testOptimizerStates(self):
        for layout in ("embeddings", "embeddingbags"):
            for state_names in (
                ("momentum1", "exp_avg_sq"),
                (),
                ("momentum1",),
                ("momentum1", "momentum2"),
                ("z_state", "a_state", "m_state"),
            ):
                with self.subTest(layout=layout, state_names=state_names):
                    state_dict = {}
                    optimizer_state = {}
                    params_plan = {}
                    expected = {}
                    for table_name in ("AB", "C"):
                        key = f"{layout}.{table_name}.weight"
                        tensors = [
                            torch.empty(2, 3) for _ in range(1 + len(state_names))
                        ]
                        state_dict[key] = tensors[0]
                        optimizer_state[key] = {
                            f"{table_name}.{name}": tensor
                            for name, tensor in zip(state_names, tensors[1:])
                        }
                        params_plan[table_name] = Mock()
                        expected[table_name] = tensors

                    sharded_module = Mock()
                    sharded_module.state_dict.return_value = state_dict
                    sharded_module.fused_optimizer.state_dict.return_value = {
                        "state": optimizer_state
                    }

                    with patch("torchrec_dynamic_embedding.ps.PS") as ps:
                        PSCollection.fromModule(
                            "emb", sharded_module, params_plan, "memory://"
                        )

                    self.assertEqual(ps.call_count, len(expected))
                    for call, (table_name, tensors) in zip(
                        ps.call_args_list, expected.items()
                    ):
                        self.assertEqual(call.args[0], f"emb.{table_name}")
                        actual = call.args[1]
                        self.assertEqual(len(actual), len(tensors))
                        for actual_tensor, expected_tensor in zip(actual, tensors):
                            self.assertIs(actual_tensor, expected_tensor)


if __name__ == "__main__":
    unittest.main()
