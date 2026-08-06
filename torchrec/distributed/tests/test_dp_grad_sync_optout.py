#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# This test exercises DATA_PARALLEL replication and the DDP-wrap decision, which
# are device-independent; run it CPU/gloo-only so it needs no GPU. CUDA is hidden
# per-test (see the class below) to keep MultiProcessContext on its CPU path
# without mutating the process-wide env at import time.
import os
import unittest
from typing import cast, List
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding import (
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
    mark_data_parallel_skip_grad_sync,
    should_skip_data_parallel_grad_sync,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

_TABLE_NAME = "table_0"
_FEATURE_NAME = "feature_0"
_EMBEDDING_DIM = 8
_NUM_EMBEDDINGS = 16


def _build_ebc(device: torch.device) -> EmbeddingBagCollection:
    return EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name=_TABLE_NAME,
                embedding_dim=_EMBEDDING_DIM,
                num_embeddings=_NUM_EMBEDDINGS,
                feature_names=[_FEATURE_NAME],
            )
        ],
        device=device,
    )


def _build_ec(device: torch.device) -> EmbeddingCollection:
    return EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name=_TABLE_NAME,
                embedding_dim=_EMBEDDING_DIM,
                num_embeddings=_NUM_EMBEDDINGS,
                feature_names=[_FEATURE_NAME],
            )
        ],
        device=device,
    )


def _input_kjt(device: torch.device) -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=[_FEATURE_NAME],
        values=torch.tensor([1, 2, 3], dtype=torch.long, device=device),
        lengths=torch.tensor([2, 1], dtype=torch.long, device=device),
    )


def _dp_lookups(dmp: DistributedModelParallel) -> List[nn.Module]:
    """Lookups of the FIRST sharded embedding module found in the DMP. Each DMP in
    this test wraps exactly one EBC/EC, so returning the first match is sufficient."""
    for submodule in dmp.modules():
        if isinstance(
            submodule, (ShardedEmbeddingBagCollection, ShardedEmbeddingCollection)
        ):
            # pyre-ignore[7]: private but stable attribute used for the assertion
            return list(submodule._lookups)
    raise AssertionError("no sharded embedding module found in DMP")


def _resolve(output: object) -> object:
    return output.wait() if hasattr(output, "wait") else output


def _shard_dp(
    module: nn.Module,
    sharder: ModuleSharder[nn.Module],
    ctx: MultiProcessContext,
    device: torch.device,
) -> DistributedModelParallel:
    module_sharding_plan = construct_module_sharding_plan(
        module,
        per_param_sharding={_TABLE_NAME: data_parallel()},
        local_size=ctx.local_size,
        world_size=ctx.world_size,
        device_type=device.type,
    )
    return DistributedModelParallel(
        module=module,
        plan=ShardingPlan({"": module_sharding_plan}),
        # pyre-ignore[6]: ctx.pg is typed Optional[ProcessGroup]; MultiProcessContext
        # always initializes it before this call, so it is non-None here.
        env=ShardingEnv.from_process_group(ctx.pg),
        sharders=[sharder],
        device=device,
    )


def _run_dp_grad_sync_optout(
    rank: int,
    world_size: int,
    backend: str,
    is_ebc: bool,
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        # Force CPU/gloo: DATA_PARALLEL replication and the DDP-wrap decision are
        # device-independent, and this keeps the test runnable without GPUs.
        device = torch.device("cpu")

        default_module = _build_ebc(device) if is_ebc else _build_ec(device)

        frozen_module = _build_ebc(device) if is_ebc else _build_ec(device)
        # Opt the frozen copy out of the DATA_PARALLEL DDP gradient reducer.
        mark_data_parallel_skip_grad_sync(frozen_module)

        sharder = cast(
            ModuleSharder[nn.Module],
            EmbeddingBagCollectionSharder() if is_ebc else EmbeddingCollectionSharder(),
        )

        default_dmp = _shard_dp(default_module, sharder, ctx, device)
        frozen_dmp = _shard_dp(frozen_module, sharder, ctx, device)

        # Regression guard: the default DP module still gets a DDP reducer.
        default_lookups = _dp_lookups(default_dmp)
        assert any(
            isinstance(lookup, DistributedDataParallel) for lookup in default_lookups
        ), "default DATA_PARALLEL module must keep the DistributedDataParallel wrap"

        # The frozen/opt-out module has raw lookups (no DDP, no reducer/buckets).
        frozen_lookups = _dp_lookups(frozen_dmp)
        assert not any(
            isinstance(lookup, DistributedDataParallel) for lookup in frozen_lookups
        ), "opt-out DATA_PARALLEL module must skip the DistributedDataParallel wrap"

        # The unwrapped DP lookup still forwards correctly (DDP is only needed
        # for the backward all-reduce). Check the pooled (EBC) forward against a
        # local (unsharded) reference that mirrors the sharded module's actual
        # replicated weight -- the sharded module re-inits in reset_parameters,
        # so copy its state, not the source module's. The embedding kernels have
        # no deterministic CUDA variant, so relax the deterministic guard.
        if is_ebc:
            weight_key = f"embedding_bags.{_TABLE_NAME}.weight"
            local_module = _build_ebc(device)
            local_module.load_state_dict(
                {weight_key: frozen_dmp.state_dict()[weight_key].cpu()}
            )
            kjt = _input_kjt(device)
            # Relax the deterministic guard only around the forward (embedding kernels
            # have no deterministic CPU/CUDA variant), then restore it -- the flag is
            # process-global even inside this MultiProcessContext subprocess.
            prev_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            try:
                with torch.inference_mode():
                    frozen_out = _resolve(frozen_dmp(kjt))
                    local_out = local_module(kjt)
                torch.testing.assert_close(
                    # pyre-ignore[16]: _resolve() returns object; the EBC forward output
                    # (KeyedTensor) exposes .values().
                    frozen_out.values(),
                    # pyre-ignore[16]: local EBC forward returns a KeyedTensor (.values()).
                    local_out.values(),
                )
            finally:
                torch.use_deterministic_algorithms(prev_deterministic)


@skip_if_asan_class
class DataParallelGradSyncOptOutTest(MultiProcessTestBase):
    def _run(self, is_ebc: bool) -> None:
        # Hide CUDA only for this test (not at module import, which would leak into
        # other test modules). The spawned workers inherit the env before they import
        # torch, forcing MultiProcessContext onto its CPU/gloo path.
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self._run_multi_process_test(
                callable=_run_dp_grad_sync_optout,
                world_size=2,
                backend="gloo",
                is_ebc=is_ebc,
            )

    def test_ebc_dp_skip_grad_sync(self) -> None:
        self._run(is_ebc=True)

    def test_ec_dp_skip_grad_sync(self) -> None:
        self._run(is_ebc=False)


class MarkDataParallelSkipGradSyncTest(unittest.TestCase):
    """Unit tests for the marker helpers (no sharding / distributed setup needed)."""

    def test_unmarked_module_returns_false(self) -> None:
        # A module never passed to mark_data_parallel_skip_grad_sync opts in to the
        # default (DDP reducer kept), so should_skip_data_parallel_grad_sync is False.
        module = _build_ebc(torch.device("cpu"))
        self.assertFalse(should_skip_data_parallel_grad_sync(module))

    def test_marked_module_returns_true(self) -> None:
        module = _build_ebc(torch.device("cpu"))
        mark_data_parallel_skip_grad_sync(module)
        self.assertTrue(should_skip_data_parallel_grad_sync(module))

    def test_unsharded_module_does_not_warn(self) -> None:
        module = _build_ebc(torch.device("cpu"))
        with self.assertNoLogs("torchrec.modules.embedding_modules", level="WARNING"):
            mark_data_parallel_skip_grad_sync(module)

    def test_already_sharded_module_warns(self) -> None:
        # Marking after sharding is a no-op: the marker is read once at shard time.
        # `_lookups` (set by ShardedEmbeddingModule) stands in for a sharded module
        # so this stays a unit test with no distributed setup. Assigned through
        # object.__setattr__ because nn.Module.__setattr__ is typed to accept only
        # Module | Tensor, while the real attribute is a plain List[nn.Module].
        module = _build_ebc(torch.device("cpu"))
        object.__setattr__(module, "_lookups", [])
        with self.assertLogs(
            "torchrec.modules.embedding_modules", level="WARNING"
        ) as cm:
            mark_data_parallel_skip_grad_sync(module)
        self.assertIn("already-sharded", cm.output[0])
        # Still sets the attribute -- the warning is advisory, not a guard.
        self.assertTrue(should_skip_data_parallel_grad_sync(module))
