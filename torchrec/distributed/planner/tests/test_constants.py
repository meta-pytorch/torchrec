#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Optional

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import (
    DDR_MEM_BW,
    HBM_MEM_BW,
    HBM_TO_DDR_MEM_BW,
    kernel_bw_lookup,
    SSD_MEM_BW,
)


class TestKernelBWLookup(unittest.TestCase):
    def test_uvm_caching_bw(self) -> None:
        compute_device: str = "cuda"
        computer_kernel: str = EmbeddingComputeKernel.FUSED_UVM_CACHING.value

        caching_ratios: List[float] = [0, 0.25, 0.5, 0.75, 1]

        uvm_caching_bw: list[Optional[float]] = [
            kernel_bw_lookup(
                compute_device,
                computer_kernel,
                HBM_MEM_BW,
                DDR_MEM_BW,
                HBM_TO_DDR_MEM_BW,
                caching_ratio,
                ssd_mem_bw=SSD_MEM_BW,
            )
            for caching_ratio in caching_ratios
        ]
        expected_uvm_caching_bw: List[float] = [
            3435973.8368,
            26655640.7808,
            49875307.724800006,
            73094974.6688,
            96314641.6128,
        ]
        self.assertEqual(expected_uvm_caching_bw, uvm_caching_bw)

    def test_uvm_caching_bw_with_prefetch_pipeline(self) -> None:
        compute_device: str = "cuda"
        computer_kernel: str = EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        prefetch_pipeline: bool = True

        caching_ratios: List[float] = [0, 0.25, 0.5, 0.75, 1]

        uvm_caching_bw: list[Optional[float]] = [
            kernel_bw_lookup(
                compute_device,
                computer_kernel,
                HBM_MEM_BW,
                DDR_MEM_BW,
                HBM_TO_DDR_MEM_BW,
                caching_ratio,
                prefetch_pipeline,
                ssd_mem_bw=SSD_MEM_BW,
            )
            for caching_ratio in caching_ratios
        ]
        expected_uvm_caching_bw: List[float] = [
            963146416.128,
            963146416.128,
            963146416.128,
            963146416.128,
            963146416.128,
        ]

        self.assertEqual(expected_uvm_caching_bw, uvm_caching_bw)
