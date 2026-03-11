#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest
from collections import OrderedDict

import torch

# @manual=//torchrec/github/examples/retrieval/modules:two_tower
from ..two_tower import convert_TwoTower_to_TwoTowerRetrieval


class MainTest(unittest.TestCase):
    def test_convert(self) -> None:
        two_tower_sd = OrderedDict(
            [
                (
                    "two_tower.candidate_proj._mlp.0._linear.bias",
                    torch.tensor(
                        [
                            -0.2589,
                            -0.3664,
                            -0.3191,
                            0.3667,
                            0.2923,
                            -0.2439,
                            0.2610,
                            0.3119,
                        ]
                    ),
                ),
                (
                    "two_tower.candidate_proj._mlp.0._linear.weight",
                    torch.tensor(
                        [
                            [-0.4525, -0.0005, 0.1334, -0.0076],
                            [0.4443, 0.0297, 0.2783, 0.2075],
                            [-0.3367, 0.0073, 0.4144, 0.0370],
                            [0.4100, 0.3595, 0.0375, -0.4462],
                            [-0.4125, 0.3645, 0.4690, -0.0170],
                            [-0.4806, 0.2175, 0.2401, -0.3609],
                            [-0.3672, 0.2956, -0.3083, -0.1524],
                            [0.1550, 0.4074, 0.4917, -0.0786],
                        ]
                    ),
                ),
                (
                    "two_tower.ebc.embedding_bags.t_movieId.weight",
                    torch.tensor(
                        [
                            [-1.0224, 2.1255, 0.2938, -0.3660],
                            [-0.2080, -0.9102, -1.6196, 0.1855],
                            [-0.0800, 0.8877, 0.8245, 0.0624],
                            [0.0100, -0.2398, -0.2211, 1.4119],
                            [0.1357, -1.8482, 0.6797, 0.7645],
                            [-0.1150, -1.8141, 0.5762, 0.6446],
                            [0.1550, 2.4782, -0.4563, 1.1750],
                            [1.2856, 0.2956, 0.6441, -0.6674],
                            [0.9449, 1.1634, 0.7426, 0.1369],
                            [-0.8751, 0.2040, 0.3757, -0.5080],
                            [0.7173, -0.0691, 0.0540, -0.7716],
                            [0.2579, -0.3305, 0.1496, 0.5781],
                            [-0.3861, 1.9357, 1.1774, -0.9661],
                            [0.9736, -0.4227, -0.8357, -1.7554],
                            [0.2369, 1.5338, -0.3424, 0.2656],
                            [-0.3526, 0.0655, -0.4043, 0.3398],
                            [-1.1384, -0.1115, -3.6186, -1.4049],
                            [-0.7968, 1.2140, 0.2334, 1.9467],
                            [-0.3680, -0.4477, 0.2628, -0.4630],
                            [0.5460, -2.8865, 0.1272, -0.3576],
                        ]
                    ),
                ),
                (
                    "two_tower.ebc.embedding_bags.t_userId.weight",
                    torch.tensor(
                        [
                            [0.9171, 0.7738, -0.1238, -1.4314],
                            [-1.1191, -0.3796, 0.8652, -1.7757],
                            [0.5349, 0.7723, 1.1785, 1.0005],
                            [0.7798, -1.0045, 0.8984, 0.2987],
                            [0.6884, 0.6973, -2.1635, -0.4854],
                            [0.6553, 0.7058, -0.2754, -0.1921],
                            [-0.3268, -0.4062, 0.1498, 0.4016],
                            [-0.2232, 1.5895, -1.0041, 0.1968],
                            [0.2138, 2.3885, -0.5640, 0.3745],
                            [-1.0124, 0.2091, -1.6986, -0.7864],
                            [-0.8058, -0.4433, 0.5544, -0.8961],
                            [-1.0396, 1.6969, 0.3996, -1.4327],
                            [-0.8118, 1.8274, 0.8686, 1.2023],
                            [-0.2246, 0.7752, 1.5411, -0.2656],
                            [1.4841, 0.3564, -1.9135, -1.0813],
                            [0.0645, -1.0724, -1.3437, 0.4690],
                            [0.1855, 0.2288, -0.2699, -0.6590],
                            [1.7723, -1.7162, -0.7586, -1.5151],
                            [1.2450, 0.6082, 1.6509, 0.3979],
                            [0.2657, 1.7444, -0.4547, -0.2377],
                        ]
                    ),
                ),
                (
                    "two_tower.query_proj._mlp.0._linear.bias",
                    torch.tensor(
                        [
                            0.1262,
                            -0.2830,
                            0.4085,
                            -0.4051,
                            -0.1628,
                            -0.3620,
                            0.2199,
                            -0.4097,
                        ]
                    ),
                ),
                (
                    "two_tower.query_proj._mlp.0._linear.weight",
                    torch.tensor(
                        [
                            [-0.4192, 0.0745, 0.0750, -0.3884],
                            [0.1165, 0.0836, -0.1370, 0.3509],
                            [-0.0543, -0.4253, 0.4838, -0.0849],
                            [-0.3889, 0.0128, -0.1729, -0.1192],
                            [-0.1702, -0.2412, 0.0032, -0.4987],
                            [0.2538, 0.0597, -0.0854, 0.4666],
                            [0.0143, 0.4979, -0.2148, -0.3888],
                            [0.0969, -0.3520, 0.4976, 0.0753],
                        ]
                    ),
                ),
            ]
        )
        query_tables = ["t_userId"]
        candidate_tables = ["t_movieId"]
        retrieval_sd = convert_TwoTower_to_TwoTowerRetrieval(
            two_tower_sd.copy(), query_tables, candidate_tables
        )
        for k, v in retrieval_sd.items():
            k = k.replace("query_ebc", "two_tower.ebc")
            k = k.replace("candidate_ebc", "two_tower.ebc")
            torch.testing.assert_close(v, two_tower_sd[k], rtol=0, atol=0)
