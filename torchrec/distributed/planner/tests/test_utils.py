#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from typing import Callable, List, Optional
from unittest.mock import MagicMock

import torch
from torchrec.distributed.planner.types import Perf, Shard, ShardingOption, Storage
from torchrec.distributed.planner.utils import (
    _find_imbalance_tables,
    BinarySearchPredicate,
    LuusJaakolaSearch,
    reset_shard_rank,
)
from torchrec.distributed.types import ShardingType


class TestFindImbalanceTables(unittest.TestCase):
    def setUp(self) -> None:
        self.best_plan: List[ShardingOption] = []
        for i in range(10):
            shard_size = [100 * i, 8]
            shard_offsets = [[0, 0], [0, 8]]
            self.best_plan.append(
                ShardingOption(
                    name=f"table_{i}",
                    tensor=MagicMock(),
                    module=MagicMock(),
                    input_lengths=MagicMock(),
                    batch_size=MagicMock(),
                    sharding_type=ShardingType.COLUMN_WISE.value,
                    partition_by=MagicMock(),
                    compute_kernel=MagicMock(),
                    shards=[
                        Shard(size=shard_size, offset=offset)
                        for offset in shard_offsets
                    ],
                )
            )

    def test_find_perf_imbalance_tables(self) -> None:
        reset_shard_rank(self.best_plan)
        for i, sharding_option in enumerate(self.best_plan):
            for j, shard in enumerate(sharding_option.shards):
                shard.rank = 2 * i + j
                shard.perf = Perf(
                    fwd_compute=2 * i,
                    fwd_comms=2 * i,
                    bwd_compute=2 * i,
                    bwd_comms=2 * i,
                )

        expected_max_perf_table_names = ["table_9"]
        max_perf_table_names = [
            sharding_option.name
            for sharding_option in _find_imbalance_tables(self.best_plan)
        ]
        self.assertTrue(expected_max_perf_table_names, max_perf_table_names)

    def test_find_hbm_imbalance_tables(self) -> None:
        reset_shard_rank(self.best_plan)
        for i, sharding_option in enumerate(self.best_plan):
            for j, shard in enumerate(sharding_option.shards):
                shard.rank = 2 * i + j
                shard.storage = Storage(
                    hbm=2 * (10 - i),
                    ddr=0,
                )

        expected_max_hbm_table_names = ["table_0"]
        max_hbm_table_names = [
            sharding_option.name
            for sharding_option in _find_imbalance_tables(
                self.best_plan, target_imbalance="hbm"
            )
        ]
        self.assertTrue(expected_max_hbm_table_names, max_hbm_table_names)


class TestBinarySearchPredicate(unittest.TestCase):
    def test_binary_search_predicate(self) -> None:
        def F(x: int) -> bool:
            return x < 90

        def probes(
            search: BinarySearchPredicate, f: Callable[[int], bool]
        ) -> List[int]:
            r = []
            probe = search.next(True)
            while probe is not None:
                r.append(probe)
                probe = search.next(f(probe))
            return r

        got = probes(BinarySearchPredicate(0, 100, 0), F)
        self.assertEqual(got, [50, 75, 88, 94, 91, 89, 90])
        got = probes(BinarySearchPredicate(0, 100, 3), F)
        self.assertEqual(got, [50, 75, 88, 94, 91])
        got = probes(BinarySearchPredicate(0, 100, 20), F)
        self.assertEqual(got, [50, 75, 88])

        got = probes(BinarySearchPredicate(91, 100, 0), F)
        self.assertEqual(got, [95, 92, 91])
        got = probes(BinarySearchPredicate(1, 10, 0), F)
        self.assertEqual(got, [5, 8, 9, 10])

        got = probes(BinarySearchPredicate(1, 1, 0), F)
        self.assertEqual(got, [1])
        got = probes(BinarySearchPredicate(1, 0, 0), F)
        self.assertEqual(got, [])


class TestLuusJaakolaSearch(unittest.TestCase):

    # Find minimum of f between x0 and x1.
    # Evaluate multiple times with different random seeds to ensure we're not
    # just getting lucky.
    # Returns a Nx2 tensor of [xs, ys] of discovered minimums.
    @staticmethod
    def evaluate(
        x0: float,
        x1: float,
        f: Callable[[float], float],
        left_cost: Optional[float] = None,
    ) -> torch.Tensor:
        xs = []
        ys = []
        iterations = 16
        for i in range(5):
            search = LuusJaakolaSearch(x0, x1, iterations, seed=i, left_cost=left_cost)
            y = search.next(0.0)
            while y is not None:
                fy = f(y)
                y = search.next(fy)
            x, y = search.best()
            xs.append(x)
            ys.append(y)
        return torch.stack([torch.tensor(xs), torch.tensor(ys)], dim=1)

    def test_simple(self) -> None:
        # See N4816561 to view these results graphically
        def f1(x: float) -> float:
            return x

        def f2(x: float) -> float:
            return x * x - 10 * x + 10  # min at x = 5

        def f3(x: float) -> float:
            # bumpy function, overall min at x=30
            return (x - 30) ** 2 + 100 * math.sin(x)

        def f4(x: float) -> float:
            # spiky/non-smooth function, min at x = 30
            return (x - 30) ** 2 + (x % 10) * 100

        results = TestLuusJaakolaSearch.evaluate(0, 100, f1)
        want = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.int64)
        torch.testing.assert_close(results, want)

        results = TestLuusJaakolaSearch.evaluate(0, 100, f2)
        want = torch.tensor(
            [
                [3.51914, -12.80705],
                [4.22958, -14.40646],
                [5.41303, -14.82940],
                [2.35012, -7.97811],
                [4.18552, -14.33662],
            ]
        )
        torch.testing.assert_close(results, want)

        results = TestLuusJaakolaSearch.evaluate(0, 100, f3)
        want = torch.tensor(
            [
                [36.58517, -46.37988],
                [29.73184, -99.28705],
                [37.67208, 56.15779],
                [35.85468, -62.00219],
                [41.76223, 58.69744],
            ]
        )
        torch.testing.assert_close(results, want)

        results = TestLuusJaakolaSearch.evaluate(0, 100, f4)
        want = torch.tensor(
            [
                [23.68681, 408.53735],
                [31.62534, 165.17535],
                [32.81968, 289.91898],
                [42.81567, 445.80777],
                [22.53002, 308.80225],
            ]
        )
        torch.testing.assert_close(results, want)

    def test_iterations(self) -> None:
        search = LuusJaakolaSearch(0, 1, 3)
        y = search.next(0)
        probes = 0
        while y is not None:
            probes += 1
            fy = y
            y = search.next(fy)
        self.assertEqual(probes, 3)

    # https://github.com/pytorch/pytorch/issues/50334
    @staticmethod
    def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """One-dimensional linear interpolation for monotonically increasing sample
        points.

        Returns the one-dimensional piecewise linear interpolant to a function with
        given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

        Args:
          x: the :math:`x`-coordinates at which to evaluate the interpolated
              values.
          xp: the :math:`x`-coordinates of the data points, must be increasing.
          fp: the :math:`y`-coordinates of the data points, same length as `xp`.

        Returns:
          the interpolated values, same size as `x`.
        """
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]

    def test_real(self) -> None:
        # See N4816561 to view these results graphically

        # Real data collected from bin packing has non-smooth surface and many local minimums.
        # mem vs cost from cmf_icvr bin packing
        cmf_icvr = torch.tensor(
            [
                [4.6741845183e11, 2.3563506569e02],
                [4.6741845240e11, 2.3563506569e02],
                [4.7121749230e11, 2.3506600864e02],
                [4.7501653103e11, 2.3468280680e02],
                [4.7881557076e11, 2.3430065943e02],
                [4.8261460996e11, 2.3396533990e02],
                [4.8641364892e11, 2.3367888393e02],
                [4.9021268717e11, 2.3339395760e02],
                [4.9401172728e11, 2.3316084540e02],
                [4.9781076708e11, 2.3292654771e02],
                [5.0160980674e11, 2.3275780179e02],
                [5.0540884491e11, 2.3256067684e02],
                [5.0920788486e11, 2.3235742684e02],
                [5.1300692424e11, 2.3219262609e02],
                [5.1680596356e11, 2.3206849693e02],
                [5.2060500162e11, 2.3193348320e02],
                [5.2440404195e11, 2.3180536764e02],
                [5.2820308146e11, 2.3170546631e02],
                [5.3200212032e11, 2.3158138440e02],
                [5.3580115967e11, 2.3146545816e02],
                [5.3960019895e11, 2.3138856778e02],
                [5.4339923878e11, 2.3128211641e02],
                [5.4719827815e11, 2.3121699239e02],
                [5.5099731798e11, 2.3169756090e02],
                [5.5479635643e11, 2.3103278320e02],
                [5.5859539575e11, 2.3171106005e02],
                [5.6239443438e11, 2.3091072319e02],
                [5.6619349259e11, 2.3084920287e02],
                [5.6999251415e11, 2.3078335619e02],
                [5.7379155310e11, 2.3113596330e02],
                [5.7759059204e11, 2.3069988094e02],
                [5.8138963104e11, 2.3127273113e02],
                [5.8518866978e11, 2.3172034584e02],
                [5.8898770984e11, 2.3083009711e02],
                [5.9278674971e11, 2.3080842049e02],
                [5.9658578920e11, 2.3176370343e02],
                [6.0038482804e11, 2.3071235199e02],
                [6.0418386709e11, 2.3213900014e02],
                [6.0798290658e11, 2.3332448570e02],
                [6.1178194561e11, 2.3275468168e02],
                [6.1558098586e11, 2.3028775311e02],
                [6.1938002497e11, 2.3099002246e02],
                [6.2317906405e11, 2.3169044278e02],
                [6.2697810321e11, 2.3387964670e02],
                [6.3077714335e11, 2.3211138392e02],
                [6.3457618280e11, 2.3106450194e02],
                [6.3837522051e11, 2.3392878354e02],
                [6.4217426058e11, 2.3260742338e02],
                [6.4597330044e11, 2.3212726336e02],
                [6.4977233953e11, 2.3355375214e02],
                [6.5357137911e11, 2.3370492744e02],
                [6.5737041818e11, 2.3274859312e02],
                [6.6116945832e11, 2.3454963160e02],
                [6.6496849695e11, 2.3314306687e02],
                [6.6876753631e11, 2.3387508611e02],
                [6.7256657578e11, 2.3164114924e02],
                [6.7636561494e11, 2.3335876240e02],
                [6.8016465549e11, 2.3259160444e02],
                [6.8396369350e11, 2.3472844839e02],
                [6.8776273363e11, 2.3402051674e02],
                [6.9156177298e11, 2.3574191998e02],
                [6.9536081174e11, 2.3853930635e02],
                [6.9915984917e11, 2.3440978885e02],
                [7.0295889084e11, 2.3613333429e02],
                [7.0675792895e11, 2.3783556448e02],
                [7.1055696937e11, 2.3596357613e02],
                [7.1435600664e11, 2.4035834255e02],
                [7.1815504705e11, 2.3882352229e02],
                [7.2195408724e11, 2.4316494619e02],
                [7.2575312535e11, 2.4125740709e02],
                [7.2955216606e11, 2.3700425464e02],
                [7.3335120460e11, 2.4198517463e02],
                [7.3715024347e11, 2.4290543902e02],
                [7.4094928544e11, 2.3961167246e02],
                [7.4474832211e11, 2.4162098068e02],
                [7.4854736178e11, 2.4791162259e02],
                [7.5234640124e11, 2.4706576073e02],
                [7.5614544041e11, 2.4682659631e02],
                [7.5994447978e11, 2.4839164423e02],
                [7.6374351905e11, 2.5108968132e02],
                [7.6754255785e11, 2.5344371602e02],
                [7.7134159724e11, 2.6063943014e02],
                [7.7514063682e11, 2.4953670969e02],
                [7.7893967570e11, 2.5865807123e02],
                [7.8273871453e11, 2.6094569799e02],
                [7.8653775458e11, 2.6653191005e02],
                [7.9033679421e11, 2.6909497473e02],
                [7.9413583349e11, 2.7149400968e02],
                [7.9793487494e11, 2.7245403781e02],
                [8.0173391173e11, 2.8131908812e02],
                [8.0553295106e11, 2.9112192412e02],
                [8.0933199067e11, 2.9245070076e02],
                [8.1313102998e11, 2.8235347505e02],
                [8.1693006950e11, 2.9033406803e02],
                [8.2072910826e11, 3.0580905927e02],
                [8.2452814772e11, 3.1147292572e02],
                [8.2832723864e11, 3.0812470431e02],
                [8.3212622721e11, 3.4879506066e02],
                [8.3592526617e11, 3.2790815984e02],
                [8.3972430401e11, 3.6465536216e02],
                [8.4352334347e11, 3.9066552303e02],
            ],
            dtype=torch.float64,
        )

        mem: torch.Tensor = cmf_icvr[:, 0]
        cost: torch.Tensor = cmf_icvr[:, 1]

        def f(x: float) -> float:
            return TestLuusJaakolaSearch.interp(torch.tensor([x]), mem, cost).item()

        results = TestLuusJaakolaSearch.evaluate(mem.min().item(), mem.max().item(), f)
        want = torch.tensor(
            [
                [5.370294e11, 2.314406e02],
                [5.426136e11, 2.313041e02],
                [5.908549e11, 2.308194e02],
                [5.755533e11, 2.309337e02],
                [6.184178e11, 2.308121e02],
            ],
        )
        torch.testing.assert_close(results, want)

        results = TestLuusJaakolaSearch.evaluate(
            mem.min().item(), mem.max().item(), f, left_cost=cost[0].item()
        )
        want = torch.tensor(
            [
                [5.370294e11, 2.314406e02],
                # 2nd search finds better result given left_cost
                [5.918126e11, 2.308140e02],
                [5.908549e11, 2.308194e02],
                [5.755533e11, 2.309337e02],
                [6.184178e11, 2.308121e02],
            ],
        )
        torch.testing.assert_close(results, want)
