#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from datetime import timedelta
from typing import cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.rec_metric import RecMetricList
from torchrec.metrics.test_utils import gen_test_tasks
from torchrec.metrics.test_utils.mock_metrics import create_tensor_states, MockRecMetric
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class CPUOffloadedMetricModuleProcessGroupTest(MultiProcessTestBase):
    def test_injected_group_avoids_divergent_group_counter_rendezvous(
        self,
    ) -> None:
        self._run_multi_process_test(
            callable=_construct_with_divergent_process_group_counters,
            world_size=2,
        )


def _construct_with_divergent_process_group_counters(
    rank: int,
    world_size: int,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    module: CPUOffloadedRecMetricModule | None = None
    shared_cpu_pg: dist.ProcessGroup | None = None
    rank_local_process_groups: list[dist.ProcessGroup] = []
    try:
        shared_cpu_pg = cast(
            dist.ProcessGroup,
            dist.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
                timeout=timedelta(seconds=60),
            ),
        )
        if rank == 0:
            for _ in range(5):
                rank_local_process_groups.append(
                    cast(
                        dist.ProcessGroup,
                        dist.new_group(
                            ranks=[0],
                            backend="gloo",
                            use_local_synchronization=True,
                        ),
                    )
                )

        process_group_counts = [0] * world_size
        dist.all_gather_object(
            process_group_counts,
            dist.get_pg_count(),
            group=shared_cpu_pg,
        )
        assert process_group_counts[0] == process_group_counts[1] + 5

        tasks = gen_test_tasks(["task1"])
        metric = MockRecMetric(
            world_size=world_size,
            my_rank=rank,
            batch_size=1,
            tasks=tasks,
            initial_states=create_tensor_states(["cross_entropy_sum"]),
        )
        with patch(
            "torchrec.metrics.cpu_offloaded_metric_module.dist.new_group",
            side_effect=AssertionError("must not create a process group"),
        ):
            module = CPUOffloadedRecMetricModule(
                model_out_device=torch.device("cpu"),
                batch_size=1,
                world_size=world_size,
                rec_tasks=tasks,
                rec_metrics=RecMetricList([metric]),
                update_batch_size=1,
                cpu_process_group=shared_cpu_pg,
            )

        assert module.cpu_process_group is shared_cpu_pg
    finally:
        if module is not None:
            with patch.object(module, "_process_metric_compute_job", return_value={}):
                module.shutdown()
        if shared_cpu_pg is not None:
            dist.barrier(group=shared_cpu_pg)
        for process_group in reversed(rank_local_process_groups):
            dist.destroy_process_group(process_group)
        if shared_cpu_pg is not None:
            dist.barrier(group=shared_cpu_pg)
            dist.destroy_process_group(shared_cpu_pg)
        dist.destroy_process_group()
