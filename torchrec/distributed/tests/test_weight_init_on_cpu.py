#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from typing import cast, List

import torch
import torch.nn as nn
from torch.distributed import _remote_device
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.batched_embedding_kernel import BatchedFusedEmbeddingBag
from torchrec.distributed.embedding_kernel import (
    _any_weights_off_plan_device,
    _weights_may_be_off_plan_device,
)
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
from torchrec.distributed.test_utils.emb_sharder import TestEBCSharder
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ModuleSharder,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardingType,
    ShardMetadata,
    TensorProperties,
)
from torchrec.distributed.utils import (
    _group_sharded_modules,
    align_shard_metadata_to_device,
    align_shards_metadata_to_device,
    EmbeddingQuantizationUtils,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.test_utils import get_free_port, init_distributed_single_host


def _placement(metadata: ShardMetadata) -> _remote_device:
    """Narrow `ShardMetadata.placement`, which the dataclass types as Optional."""
    placement = metadata.placement
    assert placement is not None
    return placement


def _as_tensor(value: object) -> torch.Tensor:
    """Narrow a registered-buffer lookup, typed `Tensor | Module | int`."""
    assert isinstance(value, torch.Tensor)
    return value


def _find_fused_kernel_module(model: nn.Module) -> BatchedFusedEmbeddingBag:
    """The torchrec kernel wrapper, whose `state_dict` is the one under test.

    `_group_sharded_modules` returns the raw FBGEMM TBE, which has no torchrec
    `state_dict` override. The `_lookups` hop mirrors that helper: lookups are a
    plain list, so they are not reachable through `named_modules()`.
    """
    found: List[BatchedFusedEmbeddingBag] = []

    def _walk(module: nn.Module) -> None:
        if isinstance(module, BatchedFusedEmbeddingBag):
            found.append(module)
        if hasattr(module, "_lookups"):
            # pyrefly: ignore[not-iterable]
            for lookup in module._lookups:
                _walk(lookup)
            return
        for _, child in module.named_children():
            _walk(child)

    _walk(model)
    assert found, "no BatchedFusedEmbeddingBag in the sharded model"
    return found[0]


def _shard_metadata(placement: str, row_offset: int = 0) -> ShardMetadata:
    return ShardMetadata(
        shard_offsets=[row_offset, 0],
        shard_sizes=[8, 4],
        placement=_remote_device(placement),
    )


class PlacementAlignmentTest(unittest.TestCase):
    """CPU-only coverage for the shard metadata realignment helpers."""

    def test_returns_input_when_placement_already_matches(self) -> None:
        metadata = _shard_metadata("rank:0/cpu")
        aligned = align_shard_metadata_to_device(metadata, torch.device("cpu"))
        self.assertIs(aligned, metadata)

    def test_rewrites_device_and_keeps_rank(self) -> None:
        metadata = _shard_metadata("rank:3/cuda:1")
        aligned = align_shard_metadata_to_device(metadata, torch.device("cpu"))
        self.assertEqual(_placement(aligned).device(), torch.device("cpu"))
        self.assertEqual(_placement(aligned).rank(), 3)
        # Offsets and sizes are untouched.
        self.assertEqual(aligned.shard_offsets, metadata.shard_offsets)
        self.assertEqual(aligned.shard_sizes, metadata.shard_sizes)

    def test_does_not_mutate_the_input(self) -> None:
        metadata = _shard_metadata("rank:0/cuda:0")
        align_shard_metadata_to_device(metadata, torch.device("cpu"))
        self.assertEqual(_placement(metadata).device(), torch.device("cuda:0"))

    def test_aligns_every_shard_in_global_metadata(self) -> None:
        global_metadata = ShardedTensorMetadata(
            shards_metadata=[
                _shard_metadata("rank:0/cuda:0"),
                _shard_metadata("rank:1/cuda:1"),
            ],
            size=torch.Size([16, 4]),
            tensor_properties=TensorProperties(dtype=torch.float32),
        )
        align_shards_metadata_to_device(global_metadata, torch.device("cpu"))
        self.assertEqual(
            [_placement(sm).rank() for sm in global_metadata.shards_metadata], [0, 1]
        )
        for shard_metadata in global_metadata.shards_metadata:
            self.assertEqual(_placement(shard_metadata).device(), torch.device("cpu"))

    def test_does_not_mutate_the_plan_behind_the_metadata(self) -> None:
        # `EnumerableShardingSpec.build_metadata` hands out `self.shards` by
        # reference, so the objects reachable from the built metadata are the
        # plan's own. Realigning must not edit them.
        spec = EnumerableShardingSpec(
            [
                _shard_metadata("rank:0/cuda:0", row_offset=0),
                _shard_metadata("rank:1/cuda:1", row_offset=8),
            ]
        )
        built = spec.build_metadata(
            tensor_sizes=torch.Size([16, 4]),
            tensor_properties=TensorProperties(dtype=torch.float32),
        )

        align_shards_metadata_to_device(built, torch.device("cpu"))

        self.assertEqual(
            [_placement(sm).device() for sm in built.shards_metadata],
            [torch.device("cpu"), torch.device("cpu")],
        )
        self.assertEqual(
            [_placement(sm).device() for sm in spec.shards],
            [torch.device("cuda:0"), torch.device("cuda:1")],
        )


def _assert_plan_placements_intact(rank: int, world_size: int) -> None:
    """Row-wise shard across ranks and check the global metadata stays truthful.

    Runs with `weight_init_on_cpu` off, which is the case that regressed:
    realigning the global metadata to the *local* shard's device rewrote every
    rank's entry to the local CUDA index, so each rank claimed every shard lived
    on its own GPU.
    """
    with MultiProcessContext(rank=rank, world_size=world_size, backend="nccl") as ctx:
        tables = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=64,
            )
        ]
        model = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        sharder = TestEBCSharder(
            sharding_type=ShardingType.ROW_WISE.value,
            kernel_type=EmbeddingComputeKernel.FUSED.value,
        )
        sharded_model = DistributedModelParallel(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[cast(ModuleSharder[nn.Module], sharder)],
            device=ctx.device,
            init_data_parallel=False,
        )

        weight = sharded_model.state_dict()["embedding_bags.table_0.weight"]
        devices = [_placement(sm).device() for sm in weight.metadata().shards_metadata]
        expected = [torch.device("cuda", r) for r in range(world_size)]
        assert (
            devices == expected
        ), f"global metadata misdescribes remote shards: {devices} != {expected}"


@unittest.skipIf(not torch.cuda.is_available(), "requires a GPU")
class WeightInitOnCpuTest(unittest.TestCase):
    """End-to-end coverage of `DistributedModelParallel(weight_init_on_cpu=...)`."""

    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.pg = init_distributed_single_host(backend="nccl", rank=0, world_size=1)

        self.embedding_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=40,
            ),
        ]

    def _shard(self, weight_init_on_cpu: bool) -> nn.Module:
        model = EmbeddingBagCollection(
            tables=self.embedding_config, device=torch.device("meta")
        )
        # The flag rides `fused_params` all the way into the FBGEMM TBE, which
        # torchrec splats them into -- there is no torchrec-side plumbing for it.
        sharder = TestEBCSharder(
            sharding_type=ShardingType.TABLE_WISE.value,
            kernel_type=EmbeddingComputeKernel.FUSED.value,
            fused_params={"weight_init_on_cpu": True} if weight_init_on_cpu else {},
        )
        return DistributedModelParallel(
            module=model,
            env=ShardingEnv.from_process_group(self.pg),
            sharders=[cast(ModuleSharder[nn.Module], sharder)],
            device=self.device,
            init_data_parallel=False,
        )

    def test_weights_land_on_init_device_and_metadata_stays_on_compute(self) -> None:
        sharded_model = self._shard(True)
        kernels = _group_sharded_modules(sharded_model)
        self.assertEqual(len(kernels), 1)
        kernel = kernels[0]
        self.assertEqual(kernel.weights_dev.device.type, "cpu")
        # Only the weight buffer relocates; the kernel still indexes on GPU.
        self.assertEqual(kernel.weights_offsets.device.type, "cuda")
        self.assertEqual(kernel.weights_placements.device.type, "cuda")
        # Optimizer state is unaffected.
        self.assertEqual(kernel.momentum1_dev.device.type, "cuda")

    def test_default_keeps_weights_on_compute_device(self) -> None:
        sharded_model = self._shard(False)
        kernel = _group_sharded_modules(sharded_model)[0]
        self.assertEqual(kernel.weights_dev.device.type, "cuda")

    def test_metadata_realignment_is_gated_on_the_flag(self) -> None:
        # Asserted on the torchrec wrapper, not the FBGEMM TBE: the wrapper is
        # where the flag is recorded, so a rename on the FBGEMM side cannot turn
        # the gate into a silent no-op.
        self.assertFalse(
            _weights_may_be_off_plan_device(
                _find_fused_kernel_module(self._shard(False))
            )
        )
        self.assertTrue(
            _weights_may_be_off_plan_device(
                _find_fused_kernel_module(self._shard(True))
            )
        )

    def test_module_level_gate_sees_the_kernels(self) -> None:
        # The sharded module decides whether to realign *global* metadata by
        # asking its own kernels. Lookups keep those in a plain list, so the walk
        # has to go through `_lookups`, not `named_modules()`.
        #
        # Asserted on the sharded module rather than the DMP wrapper, matching
        # what `_initialize_torch_state` passes: `_lookups` lives on the former.
        self.assertFalse(_any_weights_off_plan_device(self._shard(False).module))
        self.assertTrue(_any_weights_off_plan_device(self._shard(True).module))

    def test_unexpected_device_mismatch_still_raises(self) -> None:
        # The realignment must not become a blanket suppression: without
        # `weight_init_on_cpu`, a shard that disagrees with the plan is a bug and
        # ShardedTensor's assertion is what surfaces it. Asserted on the kernel's
        # own state_dict because the EBC builds its ShardedTensors once during
        # _initialize_torch_state, so a later mutation is never re-validated there.
        kernel_module = _find_fused_kernel_module(self._shard(False))
        emb_module = kernel_module.emb_module
        emb_module.weights_dev = _as_tensor(emb_module.weights_dev).cpu()

        with self.assertRaisesRegex(ValueError, "device"):
            kernel_module.state_dict()

    def test_state_dict_reflects_the_init_device(self) -> None:
        sharded_model = self._shard(True)
        state_dict = sharded_model.state_dict()
        weight = state_dict["embedding_bags.table_0.weight"]
        self.assertIsInstance(weight, ShardedTensor)
        local_shards = weight.local_shards()
        self.assertEqual(len(local_shards), 1)
        self.assertEqual(local_shards[0].tensor.device.type, "cpu")
        self.assertEqual(
            _placement(local_shards[0].metadata).device(), torch.device("cpu")
        )

    def test_recalculating_torch_state_reinitializes_weights(self) -> None:
        # Pins the contract that makes `recreate_embedding_modules` the only
        # legitimate caller: `_initialize_torch_state` ends in `reset_parameters`,
        # so rebuilding re-draws `init_fn` and discards whatever was in the
        # weights. Only safe where a checkpoint load follows immediately.
        sharded_model = self._shard(True)
        kernel = _group_sharded_modules(sharded_model)[0]
        _as_tensor(kernel.weights_dev).detach().fill_(0.5)

        EmbeddingQuantizationUtils()._recalculate_torch_state(sharded_model)

        kernel = _group_sharded_modules(sharded_model)[0]
        weights = _as_tensor(kernel.weights_dev).detach()
        self.assertFalse(
            torch.equal(weights, torch.full_like(weights, 0.5)),
            "expected reset_parameters to overwrite the weights",
        )


@unittest.skipIf(
    torch.cuda.device_count() < 2, "requires 2 GPUs to have distinct shard devices"
)
class GlobalMetadataMultiRankTest(MultiProcessTestBase):
    def test_sharding_does_not_misdescribe_remote_shards(self) -> None:
        self._run_multi_process_test(
            callable=_assert_plan_placements_intact, world_size=2
        )
