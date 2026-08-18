#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.test_utils.process_runner import (
    check_required_env_vars,
    SingleProcessContext,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

TABLES = [
    EmbeddingConfig(
        name="user_table",
        embedding_dim=32,
        num_embeddings=100000,
        feature_names=["user"],
    ),
    EmbeddingConfig(
        name="item_table",
        embedding_dim=32,
        num_embeddings=50000,
        feature_names=["item"],
    ),
]


def tpu_is_available() -> bool:
    """Whether a TPU backend is importable and has a device attached."""
    try:
        import torch_tpu  # pyre-ignore[21]  # noqa: F401
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


def _register_tpu_kernels() -> None:
    """Bind the Pallas embedding kernels and the fbgemm TPU fallbacks."""
    from torchrec.experimental.torch_tpu.pallas import dispatcher, impl  # noqa: F401


class Model(nn.Module):
    """Wraps the EmbeddingCollection so it has a stable FQN ("ec") in the plan.

    Both the sharded model and the CPU reference use this wrapper so their state
    dict keys line up for ``copy_state_dict``.
    """

    def __init__(self, tables: list[EmbeddingConfig], device: torch.device) -> None:
        super().__init__()
        self.ec: EmbeddingCollection = EmbeddingCollection(tables=tables, device=device)

    def forward(self, kjt: KeyedJaggedTensor) -> dict[str, torch.Tensor]:
        """Look up embeddings and return per-feature value tensors.

        Args:
            kjt (KeyedJaggedTensor): sparse features to look up.

        Returns:
            dict[str, torch.Tensor]: feature name -> [sum_values, embedding_dim].
        """
        jt_dict = self.ec(kjt)
        return {key: jt.values() for key, jt in jt_dict.items()}


def _build_sharded_model(
    pg: dist.ProcessGroup, world_size: int, device: torch.device
) -> nn.Module:
    """RW + UNFUSED_TPU sharded EmbeddingCollection under DMP (explicit plan)."""
    model = Model(TABLES, torch.device("meta"))
    module_sharding_plan = construct_module_sharding_plan(
        model.ec,
        per_param_sharding={
            table.name: row_wise(
                compute_kernel=EmbeddingComputeKernel.UNFUSED_TPU.value
            )
            for table in TABLES
        },
        # pyrefly: ignore[bad-argument-type]
        sharder=EmbeddingCollectionSharder(),
        local_size=1,
        world_size=world_size,
        device_type=device.type,
    )
    return DistributedModelParallel(
        module=model,
        env=ShardingEnv.from_process_group(pg),
        device=device,
        plan=ShardingPlan({"ec": module_sharding_plan}),
        # pyrefly: ignore[bad-argument-type]
        sharders=[EmbeddingCollectionSharder()],
    )


def _gather_full_grads(
    sharded_model: nn.Module, world_size: int
) -> dict[str, torch.Tensor]:
    """All-gather each table's row-shard gradient into the full table, on CPU."""
    full_by_key: dict[str, torch.Tensor] = {}
    for key, sharded in sharded_model.state_dict().items():
        grad = sharded.local_shards()[0].tensor.grad
        rows = sharded.size()[0]
        assert grad.shape[0] * world_size == rows, (
            f"{key}: {rows} rows do not divide evenly across {world_size} ranks, so "
            "the row shards have different shapes and cannot be all_gathered"
        )
        gathered = [torch.empty_like(grad) for _ in range(world_size)]
        dist.all_gather(gathered, grad)
        full_by_key[key] = torch.cat(gathered, dim=0).cpu()
    return full_by_key


@unittest.skipIf(not tpu_is_available(), "requires an attached TPU")
@unittest.skipIf(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    "requires WORLD_SIZE > 1, launch with torch.distributed.run",
)
class RWShardingCorrectnessTest(unittest.TestCase):
    """RW + UNFUSED_TPU sharded EmbeddingCollection vs an unsharded CPU reference."""

    @classmethod
    def setUpClass(cls) -> None:
        check_required_env_vars()
        _register_tpu_kernels()

    def test_forward_and_backward_match_unsharded_cpu(self) -> None:
        with SingleProcessContext(backend="tpu_dist") as ctx:
            rank, world_size = ctx.rank, ctx.world_size
            device = torch.device("tpu")
            pg = ctx.pg
            assert pg is not None, "process group is not initialized"

            sharded_model = _build_sharded_model(pg, world_size, device)
            # Every rank must build an identical reference
            torch.manual_seed(0)
            ref_model = Model(TABLES, torch.device("cpu"))

            # Slice the reference's full tables into the row shards so both models
            # hold identical weights.
            with torch.no_grad():
                copy_state_dict(sharded_model.state_dict(), ref_model.state_dict())

            # One id per rank so the RW bucketize is an even all2all; identical on
            # every rank and reused as the reference input.
            keys = [table.feature_names[0] for table in TABLES]
            flat_ids = [
                k * ((table.num_embeddings + world_size - 1) // world_size)
                for table in TABLES
                for k in range(world_size)
            ]
            lengths = torch.tensor([1] * (len(TABLES) * world_size), dtype=torch.int32)
            kjt_tpu = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(flat_ids, dtype=torch.int32, device=device),
                lengths=lengths.to(device),
            )
            kjt_cpu = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(flat_ids, dtype=torch.int32),
                lengths=lengths,
            )

            output = sharded_model(kjt_tpu)
            ref_out = ref_model(kjt_cpu)

            for feature in keys:
                got = output[feature].float().cpu()
                expected = ref_out[feature].float()
                self.assertEqual(
                    got.shape, expected.shape, f"[rank {rank}] {feature} forward shape"
                )
                torch.testing.assert_close(
                    got,
                    expected,
                    rtol=1e-5,
                    atol=1e-3,
                    msg=lambda m, f=feature: f"[rank {rank}] {f} forward: {m}",
                )

            torch.cat(
                [output[feature].float() for feature in keys], dim=0
            ).sum().backward()

            torch.cat(
                [ref_out[feature].float() for feature in keys], dim=0
            ).sum().backward()

            ref_grads = dict(ref_model.named_parameters())
            for key, got in _gather_full_grads(sharded_model, world_size).items():
                torch.testing.assert_close(
                    got,
                    ref_grads[key].grad,
                    rtol=1e-5,
                    atol=1e-3,
                    msg=lambda m, k=key: f"[rank {rank}] {k} backward: {m}",
                )


if __name__ == "__main__":
    unittest.main()
