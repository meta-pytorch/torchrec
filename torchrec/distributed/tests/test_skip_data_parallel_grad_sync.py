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
from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
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
from torchrec.distributed.model_parallel import (
    DataParallelWrapper,
    DefaultDataParallelWrapper,
    DistributedModelParallel,
    find_ddp_modules,
)
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.distributed.test_utils.process_runner import SingleProcessContext
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
    mark_data_parallel_skip_grad_sync,
    should_skip_data_parallel_grad_sync,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
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
    ctx: SingleProcessContext,
    device: torch.device,
    skip_data_parallel_grad_sync: bool = False,
) -> DistributedModelParallel:
    module_sharding_plan = construct_module_sharding_plan(
        module,
        per_param_sharding={_TABLE_NAME: data_parallel()},
        local_size=ctx.world_size,
        world_size=ctx.world_size,
        device_type=device.type,
    )
    return DistributedModelParallel(
        module=module,
        plan=ShardingPlan({"": module_sharding_plan}),
        env=ShardingEnv.from_process_group(
            cast(torch.distributed.ProcessGroup, ctx.pg)
        ),
        sharders=[sharder],
        device=device,
        skip_data_parallel_grad_sync=skip_data_parallel_grad_sync,
    )


def _run_dp_grad_sync_optout(
    rank: int,
    world_size: int,
    backend: str,
    is_ebc: bool,
) -> None:
    with (
        patch.dict(
            os.environ,
            {
                "TORCHELASTIC_USE_AGENT_STORE": "False",
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(rank),
            },
        ),
        SingleProcessContext(backend=backend, pg_timeout=timedelta(seconds=60)) as ctx,
    ):
        # Force CPU/gloo: DATA_PARALLEL replication and the DDP-wrap decision are
        # device-independent, and this keeps the test runnable without GPUs.
        device = torch.device("cpu")

        default_module = _build_ebc(device) if is_ebc else _build_ec(device)

        frozen_module = _build_ebc(device) if is_ebc else _build_ec(device)

        sharder = cast(
            ModuleSharder[nn.Module],
            EmbeddingBagCollectionSharder() if is_ebc else EmbeddingCollectionSharder(),
        )

        default_dmp = _shard_dp(default_module, sharder, ctx, device)
        frozen_dmp = _shard_dp(
            frozen_module,
            sharder,
            ctx,
            device,
            skip_data_parallel_grad_sync=True,
        )

        # Regression guard: the default DP module still gets a DDP reducer.
        default_lookups = _dp_lookups(default_dmp)
        assert any(
            isinstance(lookup, DistributedDataParallel) for lookup in default_lookups
        ), "default DATA_PARALLEL module must keep the DistributedDataParallel wrap"

        # Its already-sharded module still holds DDP, so the opt-out must fail.
        try:
            DistributedModelParallel(
                module=default_dmp.module,
                env=default_dmp._env,
                device=device,
                plan=ShardingPlan({}),
                sharders=[sharder],
                skip_data_parallel_grad_sync=True,
            )
        except RuntimeError as e:
            assert "not supported by every sharder" in str(e), str(e)
        else:
            raise AssertionError(
                "skip_data_parallel_grad_sync=True must reject residual DDP"
            )

        # The frozen/opt-out module has raw lookups (no DDP, no reducer/buckets).
        frozen_lookups = _dp_lookups(frozen_dmp)
        assert not any(
            isinstance(lookup, DistributedDataParallel) for lookup in frozen_lookups
        ), "opt-out DATA_PARALLEL module must skip the DistributedDataParallel wrap"
        assert find_ddp_modules(frozen_dmp) == []

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
            # process-global even inside this subprocess.
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


def _run_startup_sync(rank: int, world_size: int, backend: str, mixed: bool) -> None:
    """Check raw EBC/EC lookup and dense startup state on two ranks."""
    with (
        patch.dict(
            os.environ,
            {
                "TORCHELASTIC_USE_AGENT_STORE": "False",
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(rank),
            },
        ),
        SingleProcessContext(backend=backend, pg_timeout=timedelta(seconds=60)) as ctx,
        torch.random.fork_rng(devices=[]),
        ExitStack() as cleanup,
    ):
        cleanup.callback(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(False)
        env = ShardingEnv.from_process_group(
            cast(torch.distributed.ProcessGroup, ctx.pg)
        )
        for is_ebc in (True, False):
            torch.manual_seed(42)
            embedding = (
                _build_ebc(torch.device("cpu"))
                if is_ebc
                else _build_ec(torch.device("cpu"))
            )
            if mixed:
                config_type = EmbeddingBagConfig if is_ebc else EmbeddingConfig
                tables = [
                    config_type(
                        name=name,
                        embedding_dim=8,
                        num_embeddings=16,
                        feature_names=[name],
                    )
                    for name in (_TABLE_NAME, "mp_table")
                ]
                embedding = (
                    EmbeddingBagCollection(
                        tables=cast(list[EmbeddingBagConfig], tables)
                    )
                    if is_ebc
                    else EmbeddingCollection(tables=cast(list[EmbeddingConfig], tables))
                )
            module = nn.Module()
            module.add_module("embedding", embedding)
            torch.manual_seed(42 + rank)
            module.add_module("dense", nn.Linear(8, 2))
            module.register_buffer("dense_buffer", torch.tensor(float(rank)))
            per_table = {_TABLE_NAME: data_parallel()}
            if mixed:
                per_table["mp_table"] = table_wise(rank=0, compute_kernel="dense")
            plan = ShardingPlan(
                {
                    "embedding": construct_module_sharding_plan(
                        embedding,
                        per_table,
                        local_size=world_size,
                        world_size=world_size,
                        device_type="cpu",
                    )
                }
            )
            sharders = [
                cast(
                    ModuleSharder[nn.Module],
                    (
                        EmbeddingBagCollectionSharder()
                        if is_ebc
                        else EmbeddingCollectionSharder()
                    ),
                )
            ]
            torch.manual_seed(42 + rank)
            control = DistributedModelParallel(
                deepcopy(module),
                env=env,
                device=torch.device("cpu"),
                plan=plan,
                sharders=sharders,
            )
            enabled_module = deepcopy(module)
            torch.manual_seed(42 + rank)
            enabled = DistributedModelParallel(
                enabled_module,
                env=env,
                device=torch.device("cpu"),
                plan=plan,
                sharders=sharders,
                skip_data_parallel_grad_sync=True,
            )
            unittest.TestCase().assertEqual(find_ddp_modules(enabled), [])
            torch.testing.assert_close(
                enabled.module.get_buffer("dense_buffer"), torch.tensor(float(rank))
            )
            checkpoint = OrderedDict({"dense.weight": torch.full((2, 8), 0.5)})
            if not mixed:
                control.load_state_dict(deepcopy(checkpoint), strict=False)
                enabled.load_state_dict(deepcopy(checkpoint), strict=False)
            mismatches = []
            for name, parameter in enabled.named_parameters():
                if not torch.equal(parameter, dict(control.named_parameters())[name]):
                    mismatches.append(name)
                if "mp_table" not in name:
                    rank_zero = parameter.detach().clone()
                    torch.distributed.broadcast(rank_zero, src=0, group=ctx.pg)
                    torch.testing.assert_close(parameter, rank_zero)
            all_mismatches: list[list[str] | None] = [None] * world_size
            torch.distributed.all_gather_object(
                all_mismatches, mismatches, group=ctx.pg
            )
            unittest.TestCase().assertEqual(all_mismatches, [[]] * world_size)
            if not mixed:
                features = KeyedJaggedTensor.from_lengths_sync(
                    keys=[_FEATURE_NAME],
                    values=torch.tensor([1, 2, 3]),
                    lengths=torch.tensor([1, 0, 2]),
                )
                predictions = []
                for model in (control, enabled):
                    output = model.module.get_submodule("embedding")(features).wait()
                    values = (
                        cast(KeyedTensor, output).values()
                        if is_ebc
                        else cast(dict[str, JaggedTensor], output)[
                            _FEATURE_NAME
                        ].values()
                    )
                    predictions.append(model.module.get_submodule("dense")(values))
                torch.testing.assert_close(predictions[0], predictions[1])
            dense = cast(nn.Linear, enabled.module.get_submodule("dense"))
            dense(torch.full((1, 8), float(rank + 1))).sum().backward()
            torch.testing.assert_close(
                dense.weight.grad, torch.full_like(dense.weight, float(rank + 1))
            )


@skip_if_asan_class
class DataParallelGradSyncOptOutTest(MultiProcessTestBase):
    def _run(self, is_ebc: bool) -> None:
        # Hide CUDA only for this test (not at module import, which would leak into
        # other test modules). The spawned workers inherit the env before they import
        # torch, forcing MultiProcessContext onto its CPU/gloo path.
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "", "TORCHELASTIC_USE_AGENT_STORE": "False"},
        ):
            self._run_multi_process_test(
                callable=_run_dp_grad_sync_optout,
                world_size=2,
                backend="gloo",
                is_ebc=is_ebc,
            )

    def test_startup_sync_and_partial_restore(self) -> None:
        """Validate replicated weights, predictions and local backward."""
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "", "TORCHELASTIC_USE_AGENT_STORE": "False"},
        ):
            self._run_multi_process_test(
                callable=_run_startup_sync, world_size=2, backend="gloo", mixed=False
            )

    def test_startup_sync_with_mixed_dp_mp_tables(self) -> None:
        """Keep model-parallel table shards out of the replicated-state broadcast."""
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "", "TORCHELASTIC_USE_AGENT_STORE": "False"},
        ):
            self._run_multi_process_test(
                callable=_run_startup_sync, world_size=2, backend="gloo", mixed=True
            )

    def test_ebc_dp_skip_grad_sync(self) -> None:
        self._run(is_ebc=True)

    def test_ec_dp_skip_grad_sync(self) -> None:
        self._run(is_ebc=False)


class _UnsupportedDataParallelWrapper(DataParallelWrapper):
    def wrap(
        self, dmp: DistributedModelParallel, env: ShardingEnv, device: torch.device
    ) -> None:
        raise AssertionError("Unsupported wrapper must not be invoked")


class DistributedModelParallelSkipDataParallelGradSyncTest(unittest.TestCase):
    def test_unsupported_custom_wrapper_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports forward-only initialization"):
            DistributedModelParallel(
                module=nn.Linear(4, 2),
                skip_data_parallel_grad_sync=True,
                data_parallel_wrapper=_UnsupportedDataParallelWrapper(),
            )

    def test_supported_custom_wrapper_initializes_without_reducers(self) -> None:
        class RecordingWrapper(DefaultDataParallelWrapper):
            def wrap_forward_only(
                self,
                dmp: DistributedModelParallel,
                env: ShardingEnv,
                device: torch.device,
            ) -> None:
                super().wrap_forward_only(dmp, env, device)
                dmp.module.register_buffer("initialized", torch.ones(1, device=device))

        wrapper = RecordingWrapper()
        dmp = DistributedModelParallel(
            module=nn.Linear(4, 2),
            plan=ShardingPlan({}),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            sharders=[],
            data_parallel_wrapper=wrapper,
            skip_data_parallel_grad_sync=True,
        )
        torch.testing.assert_close(dmp.module.get_buffer("initialized"), torch.ones(1))
        self.assertEqual(find_ddp_modules(dmp), [])

    def test_skip_grad_sync_preserves_dense_module_forward(self) -> None:
        module = nn.Linear(4, 2)
        inputs = torch.randn(3, 4)
        with torch.inference_mode():
            expected = module(inputs)

        # No process group, so construction would raise if DMP tried to build DDP.
        dmp = DistributedModelParallel(
            module=module,
            plan=ShardingPlan({}),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            sharders=[],
            skip_data_parallel_grad_sync=True,
        )
        with torch.inference_mode():
            torch.testing.assert_close(dmp(inputs), expected)
        with self.assertRaisesRegex(RuntimeError, "skip_data_parallel_grad_sync=True"):
            dmp.init_data_parallel()

    def test_skip_grad_sync_materializes_meta_params(self) -> None:
        dmp = DistributedModelParallel(
            module=nn.Linear(4, 2, device="meta"),
            plan=ShardingPlan({}),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            sharders=[],
            skip_data_parallel_grad_sync=True,
        )
        self.assertTrue(all(p.device.type == "cpu" for p in dmp.parameters()))

    def test_init_parameters_is_idempotent_after_materialization(self) -> None:
        dmp = DistributedModelParallel(
            module=nn.Linear(4, 2, device="meta"),
            plan=ShardingPlan({}),
            env=ShardingEnv.from_local(world_size=1, rank=0),
            sharders=[],
            skip_data_parallel_grad_sync=True,
        )

        materialized_state = {
            name: tensor.clone() for name, tensor in dmp.state_dict().items()
        }
        dmp._init_parameters(dmp.module)

        for name, tensor in dmp.state_dict().items():
            torch.testing.assert_close(tensor, materialized_state[name])

    def test_init_data_parallel_false_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires init_data_parallel=True"):
            DistributedModelParallel(
                module=nn.Linear(4, 2, device="meta"),
                plan=ShardingPlan({}),
                env=ShardingEnv.from_local(world_size=1, rank=0),
                sharders=[],
                device=torch.device("cpu"),
                init_data_parallel=False,
                skip_data_parallel_grad_sync=True,
            )


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
