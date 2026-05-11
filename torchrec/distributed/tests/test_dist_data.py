#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import os
import random
import unittest
import unittest.mock
from typing import (
    cast,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import hypothesis.strategies as st
import torch
import torch.distributed as dist
from hypothesis import given, settings
from pyre_extensions import none_throws
from torchrec.distributed.dist_data import (
    _collective_tag_from,
    _get_recat,
    JaggedTensorAllToAll,
    KJTAllToAll,
    KJTAllToAllSplitsAwaitable,
    PooledEmbeddingsAllGather,
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsReduceScatter,
    SequenceEmbeddingsAllToAll,
    SplitsAllToAllAwaitable,
    VariableBatchPooledEmbeddingsAllToAll,
)
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import Awaitable, NullShardingContext
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


T = TypeVar("T", int, float, List[int])


# Lightly adapted from Stack Overflow #10823877
def _flatten(iterable: Iterable[T]) -> Generator[T, None, None]:
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        else:
            try:
                # pyrefly: ignore[no-matching-overload]
                new_iterator = iter(value)
            except TypeError:
                # pyrefly: ignore[invalid-yield]
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator


def _to_tensor(iterator: List[T], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(_flatten(iterator)), dtype=dtype)


def _generate_sparse_features_batch(
    keys: List[str],
    splits: List[int],
    batch_size_per_rank: List[int],
    is_weighted: bool = False,
) -> Tuple[List[KeyedJaggedTensor], List[KeyedJaggedTensor]]:
    world_size = len(splits)
    offsets = [0] + list(itertools.accumulate(splits))
    values = {}
    lengths = {}
    weights = {} if is_weighted else None

    for key in keys:
        lengths[key] = [
            [random.randint(0, 10) for _ in range(batch_size_per_rank[i])]
            for i in range(world_size)
        ]
        values[key] = [
            [random.randint(0, 1000) for _ in range(sum(lengths[key][i]))]
            for i in range(world_size)
        ]

        if weights:
            weights[key] = [
                [random.random() for _ in range(sum(lengths[key][i]))]
                for i in range(world_size)
            ]

    in_jagged: List[KeyedJaggedTensor] = []
    out_jagged: List[KeyedJaggedTensor] = []
    for i in range(world_size):
        in_jagged.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                lengths=_to_tensor([lengths[key][i] for key in keys], torch.int),
                values=_to_tensor([values[key][i] for key in keys], torch.int),
                weights=(
                    _to_tensor([weights[key][i] for key in keys], torch.float)
                    if weights
                    else None
                ),
            )
        )
        key_index = []
        out_keys = keys[offsets[i] : offsets[i + 1]]
        for key in out_keys:
            for j in range(world_size):
                key_index.append((key, j))
        out_jagged.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=out_keys,
                lengths=_to_tensor(
                    [lengths[key][j] for key, j in key_index],
                    torch.int,
                ),
                values=_to_tensor(
                    [values[key][j] for key, j in key_index],
                    torch.int,
                ),
                weights=(
                    _to_tensor(
                        [weights[key][j] for key, j in key_index],
                        torch.float,
                    )
                    if weights
                    else None
                ),
            )
        )
    return in_jagged, out_jagged


def _generate_variable_batch_sparse_features_batch(
    keys: List[str],
    splits: List[int],
    batch_size_per_rank_per_feature: List[List[List[int]]],
    is_weighted: bool = False,
) -> Tuple[List[KeyedJaggedTensor], List[KeyedJaggedTensor]]:
    world_size = len(splits)
    offsets = [0] + list(itertools.accumulate(splits))
    values = {}
    lengths = {}
    weights = {} if is_weighted else None

    for i, key in enumerate(keys):
        lengths[key] = [
            [
                random.randint(0, 10)
                for _ in range(sum(batch_size_per_rank_per_feature[rank][i]))
            ]
            for rank in range(world_size)
        ]
        values[key] = [
            [random.randint(0, 1000) for _ in range(sum(lengths[key][j]))]
            for j in range(world_size)
        ]

        if weights:
            weights[key] = [
                [random.random() for _ in range(sum(lengths[key][j]))]
                for j in range(world_size)
            ]

    in_jagged: List[KeyedJaggedTensor] = []
    out_jagged: List[KeyedJaggedTensor] = []
    for i in range(world_size):
        in_jagged.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                stride_per_key_per_rank=batch_size_per_rank_per_feature[i],
                lengths=_to_tensor([lengths[key][i] for key in keys], torch.int),
                values=_to_tensor([values[key][i] for key in keys], torch.int),
                weights=(
                    _to_tensor([weights[key][i] for key in keys], torch.float)
                    if weights
                    else None
                ),
            )
        )
        key_index = []
        out_keys = keys[offsets[i] : offsets[i + 1]]
        key_indices = [keys.index(k) for k in out_keys]
        batch_sizes_by_rank = list(zip(*batch_size_per_rank_per_feature))
        for key in out_keys:
            for j in range(world_size):
                key_index.append((key, j))

        out_jagged.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=out_keys,
                stride_per_key_per_rank=[
                    list(_flatten(batch_sizes_by_rank[key_idx]))
                    for key_idx in key_indices
                ],
                lengths=_to_tensor(
                    [lengths[key][j] for key, j in key_index],
                    torch.int,
                ),
                values=_to_tensor(
                    [values[key][j] for key, j in key_index],
                    torch.int,
                ),
                weights=(
                    _to_tensor(
                        [weights[key][j] for key, j in key_index],
                        torch.float,
                    )
                    if weights
                    else None
                ),
            )
        )
    return in_jagged, out_jagged


def _generate_pooled_embedding_batch(
    keys: List[str], dims: List[int], splits: List[int], batch_size_per_rank: List[int]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    world_size = len(splits)
    offsets = [0] + list(itertools.accumulate(splits))
    local_emb = {}
    B_global = sum(batch_size_per_rank)
    B_offsets = [0] + list(itertools.accumulate(batch_size_per_rank))

    for key, dim in zip(keys, dims):
        local_emb[key] = [
            [random.random() for _ in range(dim)] for _ in range(B_global)
        ]

    in_tensor: List[torch.Tensor] = []
    out_tensor: List[torch.Tensor] = []
    for i in range(world_size):
        in_keys = keys[offsets[i] : offsets[i + 1]]
        in_tensor.append(
            _to_tensor(
                [local_emb[key][b] for b in range(B_global) for key in in_keys],
                torch.float,
            ).view(B_global, -1)
            if in_keys
            else torch.empty(B_global, 0, dtype=torch.float)
        )
        out_tensor.append(
            _to_tensor(
                [
                    local_emb[key][b]
                    for b in range(B_offsets[i], B_offsets[i + 1])
                    for key in keys
                ],
                torch.float,
            ).view(batch_size_per_rank[i], -1)
        )

    return in_tensor, out_tensor


class KJTAllToAllTest(MultiProcessTestBase):
    @classmethod
    def _validate(
        cls,
        actual_output_awaitable: Union[KJTAllToAllSplitsAwaitable, KeyedJaggedTensor],
        expected_output_awaitable: Union[KJTAllToAllSplitsAwaitable, KeyedJaggedTensor],
    ) -> None:
        actual_output = (
            actual_output_awaitable
            if isinstance(actual_output_awaitable, KeyedJaggedTensor)
            else actual_output_awaitable.wait().wait()
        )
        expected_output = (
            expected_output_awaitable
            if isinstance(expected_output_awaitable, KeyedJaggedTensor)
            else expected_output_awaitable.wait().wait()
        )
        torch.testing.assert_close(
            actual_output.values().cpu(),
            expected_output.values().cpu(),
        )
        torch.testing.assert_close(
            (
                actual_output.weights().cpu()
                if actual_output.weights_or_none() is not None
                else []
            ),
            (
                expected_output.weights().cpu()
                if expected_output.weights_or_none() is not None
                else []
            ),
        )
        torch.testing.assert_close(
            actual_output.lengths().cpu(),
            expected_output.lengths().cpu(),
        )
        assert actual_output.keys() == expected_output.keys()

    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        _input: KeyedJaggedTensor,
        output: KeyedJaggedTensor,
        backend: str,
        splits: List[int],
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        device = torch.device(f"cuda:{rank}")
        if backend == "gloo":
            device = torch.device("cpu")
        _input = _input.to(device=device)
        output = output.to(device=device)
        pg = dist.group.WORLD
        lengths_a2a = KJTAllToAll(
            #  `Optional[_distributed_c10d.ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg=pg,
            splits=splits,
        )
        cls._validate(lengths_a2a(_input), output)
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        backend=st.sampled_from(["nccl"]),
        B=st.integers(min_value=1, max_value=2),
        features=st.integers(min_value=3, max_value=4),
        is_weighted=st.booleans(),
        variable_batch_size=st.booleans(),
    )
    @settings(max_examples=4, deadline=None)
    def test_features(
        self,
        backend: str,
        B: int,
        features: int,
        is_weighted: bool,
        variable_batch_size: bool,
    ) -> None:
        keys = [f"F{feature}" for feature in range(features)]
        rank0_split = random.randint(0, features)
        splits = [rank0_split, features - rank0_split]
        world_size = 2

        if variable_batch_size:
            batch_size_per_rank = [random.randint(B, B + 4), random.randint(B, B + 4)]
        else:
            batch_size_per_rank = [B, B]

        _input, output = _generate_sparse_features_batch(
            keys=keys,
            splits=splits,
            batch_size_per_rank=batch_size_per_rank,
            is_weighted=is_weighted,
        )

        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "_input": _input[rank],
                    "output": output[rank],
                    "backend": backend,
                    "splits": splits,
                }
            )

        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        backend=st.sampled_from(["nccl"]),
        B=st.integers(min_value=1, max_value=2),
        features=st.integers(min_value=3, max_value=4),
        is_weighted=st.booleans(),
        variable_batch_per_rank=st.booleans(),
    )
    @settings(max_examples=4, deadline=None)
    def test_variable_batch_features(
        self,
        backend: str,
        B: int,
        features: int,
        is_weighted: bool,
        variable_batch_per_rank: bool,
    ) -> None:
        keys = [f"F{feature}" for feature in range(features)]
        rank0_split = random.randint(0, features)
        splits = [rank0_split, features - rank0_split]
        world_size = 2

        if variable_batch_per_rank:
            batch_size_per_rank_per_feature = [
                [[random.randint(B, B + 4)] for _ in range(features)]
                for _ in range(world_size)
            ]
        else:
            batch_size_per_rank_per_feature = [
                [[random.randint(B, B + 4)] for _ in range(features)]
            ] * world_size

        _input, output = _generate_variable_batch_sparse_features_batch(
            keys=keys,
            splits=splits,
            batch_size_per_rank_per_feature=batch_size_per_rank_per_feature,
            is_weighted=is_weighted,
        )

        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "_input": _input[rank],
                    "output": output[rank],
                    "backend": backend,
                    "splits": splits,
                }
            )

        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


class PooledEmbeddingsAllToAllTest(MultiProcessTestBase):
    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        _input: torch.Tensor,
        output: torch.Tensor,
        backend: str,
        dim_sum_per_rank: List[int],
        batch_size_per_rank: List[int],
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.group.WORLD
        if backend == "gloo":
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{rank}")
        _input = _input.to(device=device)
        output = output.to(device=device)

        codecs = get_qcomm_codecs(qcomms_config)

        a2a = PooledEmbeddingsAllToAll(
            #  `Optional[_distributed_c10d.ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg=pg,
            dim_sum_per_rank=dim_sum_per_rank,
            device=device,
            codecs=codecs,
        )
        _input.requires_grad = True
        if len(set(batch_size_per_rank)) > 1:
            # variable batch size
            res = a2a(_input, batch_size_per_rank).wait()
        else:
            res = a2a(_input).wait()
        res.backward(res)

        atol, rtol = None, None
        if qcomms_config is not None:
            atol, rtol = 0.01, 0.01
            if (
                qcomms_config.forward_precision == CommType.FP8
                or qcomms_config.backward_precision == CommType.FP8
            ):
                atol, rtol = 0.05, 0.05

        torch.testing.assert_close(res, output, rtol=rtol, atol=atol)

        torch.testing.assert_close(
            _input.cpu().detach().div_(world_size),
            # pyrefly: ignore[missing-attribute]
            _input.grad.cpu().detach(),
            atol=atol,
            rtol=rtol,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        # backend=st.sampled_from(["gloo", "nccl"]),
        backend=st.sampled_from(["nccl"]),
        B=st.integers(min_value=2, max_value=3),
        features=st.integers(min_value=3, max_value=4),
        is_reversed=st.booleans(),
        variable_batch_size=st.booleans(),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                    backward_loss_scale=128.0,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP8,
                    backward_precision=CommType.FP8,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP8,
                    backward_precision=CommType.BF16,
                ),
            ]
        ),
    )
    @settings(max_examples=4, deadline=None)
    def test_pooled_embeddings(
        self,
        backend: str,
        B: int,
        features: int,
        is_reversed: bool,
        variable_batch_size: bool,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:
        world_size = 2
        keys = [f"F{feature}" for feature in range(features)]
        dims = random.sample([8, 16, 32] * features, features)
        rank0_split = random.randint(1, features - 1)
        splits = [rank0_split, features - rank0_split]
        if is_reversed:
            splits.reverse()
        dim_sum_per_rank = [sum(dims[: splits[0]]), sum(dims[splits[0] :])]

        if variable_batch_size:
            batch_size_per_rank = [random.randint(B, B + 4), random.randint(B, B + 4)]
        else:
            batch_size_per_rank = [B, B]

        _input, output = _generate_pooled_embedding_batch(
            keys=keys,
            dims=dims,
            splits=splits,
            batch_size_per_rank=batch_size_per_rank,
        )

        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "_input": _input[rank],
                    "output": output[rank],
                    "backend": backend,
                    "dim_sum_per_rank": dim_sum_per_rank,
                    "batch_size_per_rank": batch_size_per_rank,
                    "qcomms_config": qcomms_config,
                }
            )
        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


class PooledEmbeddingsReduceScatterTest(MultiProcessTestBase):
    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        input: torch.Tensor,
        expected_output: torch.Tensor,
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=2, backend="nccl")
        pg = dist.group.WORLD
        input = input.cuda(rank)
        input.requires_grad = True

        codecs = get_qcomm_codecs(qcomms_config)

        rs = PooledEmbeddingsReduceScatter(
            #  `Optional[_distributed_c10d.ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg,
            codecs=codecs,
        ).cuda(rank)
        actual_output = rs(input).wait()
        s = torch.sum(actual_output)
        s.backward()

        atol, rtol = None, None
        if qcomms_config is not None:
            atol, rtol = 0.003, 0.004
        torch.testing.assert_close(
            actual_output.cpu().detach(),
            expected_output.cpu().detach(),
            rtol=rtol,
            atol=atol,
        )
        if qcomms_config is None:
            torch.testing.assert_close(
                # pyrefly: ignore[missing-attribute]
                input.grad.cpu().detach(),
                torch.ones(input.size()).div_(world_size),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                    backward_loss_scale=128,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.FP8,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP8,
                ),
                # FP8 is not numerically stable for reduce_scatter
                # Not supported for now for forward case
                # QCommsConfig(
                #     forward_precision=CommType.FP8,
                #     backward_precision=CommType.FP8,
                # ),
                # QCommsConfig(
                #     forward_precision=CommType.FP8,
                #     backward_precision=CommType.BF16,
                # ),
            ]
        ),
    )
    @settings(max_examples=3, deadline=45000)
    def test_pooled_embedding_reduce_scatter(
        self, qcomms_config: Optional[QCommsConfig]
    ) -> None:
        world_size = 2
        embeddding_dim = 10
        batch_size = 4
        embeddings = torch.rand((batch_size * world_size, embeddding_dim))
        embeddings_by_rank = list(torch.chunk(embeddings, world_size, dim=0))
        expect_results = torch.chunk(
            torch.stack(embeddings_by_rank, dim=0).sum(dim=0),
            world_size,
            dim=0,
        )
        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "input": embeddings_by_rank[rank],
                    "expected_output": expect_results[rank],
                    "qcomms_config": qcomms_config,
                }
            )

        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


class PooledEmbeddingsReduceScatterVTest(MultiProcessTestBase):
    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        input: torch.Tensor,
        input_splits: List[int],
        expected_output: torch.Tensor,
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=2, backend="nccl")
        pg = dist.group.WORLD
        input = input.cuda(rank)
        input.requires_grad = True

        codecs = get_qcomm_codecs(qcomms_config)

        rs = PooledEmbeddingsReduceScatter(
            #  `Optional[_distributed_c10d.ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg,
            codecs=codecs,
        ).cuda(rank)
        actual_output = rs(input, input_splits=input_splits).wait()
        s = torch.sum(actual_output)
        s.backward()

        atol, rtol = None, None
        if qcomms_config is not None:
            atol, rtol = 0.003, 0.004
        torch.testing.assert_close(
            actual_output.cpu().detach(),
            expected_output.cpu().detach(),
            rtol=rtol,
            atol=atol,
        )
        if qcomms_config is None:
            torch.testing.assert_close(
                # pyrefly: ignore[missing-attribute]
                input.grad.cpu().detach(),
                torch.ones(input.size()).div_(world_size),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                    backward_loss_scale=128,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.FP8,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP8,
                ),
                # FP8 is not numerically stable for reduce_scatter_v
                # Not supported for now for forward case
                # QCommsConfig(
                #     forward_precision=CommType.FP8,
                #     backward_precision=CommType.FP8,
                # ),
                # QCommsConfig(
                #     forward_precision=CommType.FP8,
                #     backward_precision=CommType.BF16,
                # ),
            ]
        ),
    )
    @settings(max_examples=3, deadline=45000)
    def test_pooled_embedding_reduce_scatter_v(
        self, qcomms_config: Optional[QCommsConfig]
    ) -> None:
        world_size = 2
        embeddding_dim = 10
        batch_size = 2
        embeddings = torch.rand((batch_size * world_size, embeddding_dim))
        embeddings_by_rank = list(torch.chunk(embeddings, batch_size, dim=0))
        expect_results = torch.chunk(
            torch.stack(embeddings_by_rank, dim=0).sum(dim=0),
            2,
            dim=0,
        )
        input_splits = [er.size(0) for er in expect_results]
        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "input": embeddings_by_rank[rank],
                    "input_splits": input_splits,
                    "expected_output": expect_results[rank],
                    "qcomms_config": qcomms_config,
                }
            )

        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


class PooledEmbeddingsAllGatherTest(MultiProcessTestBase):
    @classmethod
    def _validate(
        cls,
        actual_output: torch.Tensor,
        expected_output: torch.Tensor,
        input: torch.Tensor,
        world_size: int,
    ) -> None:
        torch.testing.assert_close(
            actual_output.cpu().detach(), expected_output.cpu().detach()
        )
        torch.testing.assert_close(
            # pyrefly: ignore[missing-attribute]
            input.grad.cpu().detach(),
            torch.ones(input.size()),
        )

    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        input: torch.Tensor,
        expected_output: torch.Tensor,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=2, backend="nccl")
        pg = dist.group.WORLD
        input = input.cuda(rank)
        input.requires_grad = True
        #  `Optional[_distributed_c10d.ProcessGroup]`.
        # pyrefly: ignore[bad-argument-type]
        ag = PooledEmbeddingsAllGather(pg).cuda(rank)
        actual_output = ag(input).wait()
        s = torch.sum(actual_output)
        s.backward()
        cls._validate(actual_output, expected_output, input, world_size)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_pooled_embedding_all_gather(self) -> None:
        world_size = 2
        embeddding_dim = 10
        batch_size = 2
        embeddings = torch.rand((batch_size * world_size, embeddding_dim))
        embeddings_by_rank = list(torch.chunk(embeddings, batch_size, dim=0))
        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "input": embeddings_by_rank[rank],
                    "expected_output": embeddings,
                }
            )

        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


# For sequence embedding we do not support different dim for different tables
def _generate_sequence_embedding_batch(
    keys: List[str],
    dim: int,
    splits: List[int],
    batch_size_per_rank: List[int],
    lengths_before_a2a_per_rank: Dict[int, List],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    world_size = len(splits)

    tensor_by_feature: Dict[str, List[torch.Tensor]] = (
        {}
    )  # Model parallel, key as feature
    tensor_by_rank: Dict[str, List[torch.Tensor]] = {}  # Data parallel, key as rank

    emb_by_rank_feature = {}
    for rank in range(world_size):
        offset = 0
        current_rank_lengths = lengths_before_a2a_per_rank[rank]
        current_rank_batch_size = batch_size_per_rank[rank]

        for feature in keys:
            current_stride_lengths = current_rank_lengths[
                offset : offset + current_rank_batch_size
            ]
            offset += current_rank_batch_size
            emb_by_rank_feature[f"{feature}_{str(rank)}"] = torch.rand(
                (sum(current_stride_lengths), dim)
            ).tolist()
            tensor_by_feature[f"{feature}"] = []
            tensor_by_rank[f"{str(rank)}"] = []

    for k, v in emb_by_rank_feature.items():
        feature, rank = k.split("_")
        tensor_by_feature[feature].extend(v)
        tensor_by_rank[rank].extend(v)

    in_tensor: List[torch.Tensor] = []
    out_tensor: List[torch.Tensor] = []

    for _, v in tensor_by_feature.items():
        in_tensor.append(torch.Tensor(v))

    for _, v in tensor_by_rank.items():
        out_tensor.append(torch.Tensor(v))

    input_offsets = [0] + list(itertools.accumulate(splits))
    output_offsets = torch.arange(0, world_size + 1, dtype=torch.int).tolist()

    regroup_in_tensor: List[torch.Tensor] = []
    regroup_out_tensor: List[torch.Tensor] = []

    for i in range(world_size):
        regroup_in_tensor.append(
            torch.cat(in_tensor[input_offsets[i] : input_offsets[i + 1]])
        )
        regroup_out_tensor.append(
            torch.cat(out_tensor[output_offsets[i] : output_offsets[i + 1]])
        )

    return regroup_in_tensor, regroup_out_tensor


class SeqEmbeddingsAllToAllTest(MultiProcessTestBase):
    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        _input: torch.Tensor,
        output: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        lengths_after_sdd_a2a: torch.Tensor,
        features_per_rank: List[int],
        batch_size_per_rank: List[int],
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend="nccl")
        pg = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        _input = _input.to(device=device)
        output = output.to(device=device)
        lengths_after_sdd_a2a = lengths_after_sdd_a2a.to(device=device)

        a2a = SequenceEmbeddingsAllToAll(
            #  `Optional[_distributed_c10d.ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg=pg,
            features_per_rank=features_per_rank,
            device=device,
        )
        _input.requires_grad = True

        sparse_features_recat = (
            _get_recat(
                local_split=features_per_rank[rank],
                num_splits=world_size,
                device=device,
                stagger=1,
                batch_size_per_rank=batch_size_per_rank,
            )
            if len(set(batch_size_per_rank)) > 1
            else None
        )

        res = a2a(
            local_embs=_input,
            lengths=lengths_after_sdd_a2a,
            input_splits=input_splits,
            output_splits=output_splits,
            batch_size_per_rank=batch_size_per_rank,
            sparse_features_recat=sparse_features_recat,
        ).wait()

        atol, rtol = None, None
        if qcomms_config is not None:
            atol, rtol = 0.01, 0.01
            if (
                qcomms_config.forward_precision == CommType.FP8
                or qcomms_config.backward_precision == CommType.FP8
            ):
                atol, rtol = 0.05, 0.05
        torch.testing.assert_close(res, output, rtol=rtol, atol=atol)
        res.backward(res)
        grad = _input.grad
        torch.testing.assert_close(
            _input.cpu().detach(),
            # pyrefly: ignore[missing-attribute]
            grad.cpu().detach() * world_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        variable_batch_size=st.booleans(),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                    backward_loss_scale=128.0,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP8,
                    backward_precision=CommType.FP8,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP8,
                    backward_precision=CommType.BF16,
                ),
            ]
        ),
    )
    @settings(max_examples=4, deadline=None)
    def test_sequence_embeddings(
        self,
        variable_batch_size: bool,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:

        world_size = 2
        seq_emb_dim = 3
        features = 3
        keys = [f"F{feature}" for feature in range(features)]

        if variable_batch_size:
            variable_batch_size = True
            batch_size_per_rank = [3, 2]

            feature_num_per_rank = [1, 2]

            lengths_before_a2a_per_rank = {
                0: [3, 0, 2, 4, 1, 2, 1, 2, 0],
                1: [4, 3, 1, 0, 5, 0],
            }

            lengths_after_a2a_per_rank = [
                # pyrefly: ignore[bad-argument-type]
                torch.tensor([3, 0, 2, 4, 3], dtype=int),
                # pyrefly: ignore[bad-argument-type]
                torch.tensor([4, 1, 2, 1, 0, 1, 2, 0, 5, 0], dtype=int),
            ]

            input_splits_per_rank = {}
            output_splits_per_rank = {}

            input_splits_per_rank[0] = [5, 10]  # sum (3,0,2), sum(4, 1, 2, 1, 2, 0)
            input_splits_per_rank[1] = [7, 6]  # sum (4, 3), sum(1, 0, 5, 0)
            output_splits_per_rank[0] = [5, 7]  # emb input splits
            output_splits_per_rank[1] = [10, 6]  #
        else:
            variable_batch_size = False
            batch_size_per_rank = [2, 2]

            feature_num_per_rank = [1, 2]

            lengths_before_a2a_per_rank = {0: [3, 4, 1, 2, 6, 0], 1: [4, 0, 2, 3, 1, 2]}
            lengths_after_a2a_per_rank = [
                # pyrefly: ignore[bad-argument-type]
                torch.tensor([[3, 4, 4, 0]], dtype=int),
                # pyrefly: ignore[bad-argument-type]
                torch.tensor([[1, 2, 2, 3], [6, 0, 1, 2]], dtype=int),
            ]

            input_splits_per_rank = {}
            output_splits_per_rank = {}

            input_splits_per_rank[0] = [
                7,
                9,
            ]  # sum (3,4) rank0, sum(2, 6, 5, 0) for rank 1
            input_splits_per_rank[1] = [
                4,
                8,
            ]  # sum (9,0) rank0, sum(7, 8, 1, 5) for rank 1
            output_splits_per_rank[0] = [7, 4]  # emb input splits
            output_splits_per_rank[1] = [9, 8]  #

        _input, output = _generate_sequence_embedding_batch(
            keys=keys,
            dim=seq_emb_dim,
            splits=feature_num_per_rank,
            batch_size_per_rank=batch_size_per_rank,
            lengths_before_a2a_per_rank=lengths_before_a2a_per_rank,
        )

        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "_input": _input[rank],
                    "output": output[rank],
                    "input_splits": input_splits_per_rank[rank],
                    "output_splits": output_splits_per_rank[rank],
                    "lengths_after_sdd_a2a": lengths_after_a2a_per_rank[rank],
                    "features_per_rank": feature_num_per_rank,
                    "batch_size_per_rank": batch_size_per_rank,
                    "qcomms_config": qcomms_config,
                }
            )
        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


class VariableBatchPooledEmbeddingsAllToAllTest(MultiProcessTestBase):
    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        _input: torch.Tensor,
        output: torch.Tensor,
        backend: str,
        emb_dim_per_rank_per_feature: List[List[int]],
        batch_size_per_rank_per_feature: List[List[int]],
        batch_size_per_feature_pre_a2a: List[int],
        qcomms_config: Optional[QCommsConfig] = None,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.group.WORLD
        if backend == "gloo":
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{rank}")
        _input = _input.to(device=device)
        output = output.to(device=device)

        codecs = get_qcomm_codecs(qcomms_config)

        a2a = VariableBatchPooledEmbeddingsAllToAll(
            #  `Optional[_distributed_c10d.ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg=pg,
            emb_dim_per_rank_per_feature=emb_dim_per_rank_per_feature,
            device=device,
            codecs=codecs,
        )
        _input.requires_grad = True
        res = a2a(
            local_embs=_input,
            batch_size_per_rank_per_feature=batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a=batch_size_per_feature_pre_a2a,
        ).wait()
        res.backward(res)

        atol, rtol = None, None
        if qcomms_config is not None:
            atol, rtol = 0.01, 0.01
            if (
                qcomms_config.forward_precision == CommType.FP8
                or qcomms_config.backward_precision == CommType.FP8
            ):
                atol, rtol = 0.05, 0.05

        torch.testing.assert_close(res, output, rtol=rtol, atol=atol)

        torch.testing.assert_close(
            _input.cpu().detach().div_(world_size),
            # pyrefly: ignore[missing-attribute]
            _input.grad.cpu().detach(),
            atol=atol,
            rtol=rtol,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        backend=st.sampled_from(["nccl"]),
        features=st.integers(min_value=3, max_value=4),
        B=st.integers(min_value=2, max_value=3),
        is_reversed=st.booleans(),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
                    backward_loss_scale=128.0,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP32,
                    backward_precision=CommType.BF16,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP8,
                    backward_precision=CommType.FP8,
                ),
                QCommsConfig(
                    forward_precision=CommType.FP8,
                    backward_precision=CommType.BF16,
                ),
            ]
        ),
    )
    @settings(max_examples=4, deadline=None)
    def test_variable_batch_pooled_embeddings(
        self,
        backend: str,
        B: int,
        features: int,
        is_reversed: bool,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:
        world_size = 2
        keys = [f"F{feature}" for feature in range(features)]
        dims = random.sample([8, 16, 32] * features, features)
        rank0_split = random.randint(1, features - 1)
        splits = [rank0_split, features - rank0_split]
        if is_reversed:
            splits.reverse()
        emb_dim_per_rank_per_feature = []
        f_id = 0
        for split in splits:
            emb_dim_per_feature = []
            for _ in range(split):
                emb_dim_per_feature.append(dims[f_id])
                f_id += 1
            emb_dim_per_rank_per_feature.append(emb_dim_per_feature)

        batch_size_per_rank_per_feature_pre_a2a = []
        for _ in range(world_size):
            batch_size_per_feature = [random.randint(B, B + 4) for _ in keys]
            batch_size_per_rank_per_feature_pre_a2a.append(batch_size_per_feature)

        batch_size_per_rank_per_feature_post_a2a_per_rank = []
        fid = 0
        for i in range(world_size):
            batch_size_per_rank_per_feature_post_a2a = [[] for _ in range(world_size)]
            split = splits[i]
            for _ in range(split):
                for j in range(world_size):
                    batch_size_per_rank_per_feature_post_a2a[j].append(
                        batch_size_per_rank_per_feature_pre_a2a[j][fid]
                    )
                fid += 1
            batch_size_per_rank_per_feature_post_a2a_per_rank.append(
                batch_size_per_rank_per_feature_post_a2a
            )

        """
        before input dist:
        r_0
        f_0: [1, 2], [3, 4]
        f_1: [5, 6]
        f_2: [1],    [2],   [3]

        r_1
        f_0: [1, 2]
        f_1: [5, 6], [3, 4]
        f_2: [1],    [2]

        after input dist (splits: [1, 2]):
        r_0
        f_0: [1, 2], [3, 4], [1, 2]

        r_1
        f_1: [5, 6], [5, 6], [3, 4]
        f_2: [1], [2], [3], [1], [2]

        output layout
        r_0:
        [r_0_f_0_s_0, r_0_f_0_s_1, r_1_f_0_s_0]

        r_1:
        [r_0_f_1_s_0, r_0_f_2_s_0, r_0_f_2_s_1, r_0_f_2_s_2,
         r_1_f_1_s_0, r_1_f_1_s_1, r_1_f_2_s_0, r_1_f_2_s_1]

        after output dist
        r_0:
        [r_0_f_0_s_0, r_0_f_0_s_1, r_0_f_1_s_0, r_0_f_2_s_0, r_0_f_2_s_1, r_0_f_2_s_2]

        r_1:
        [r_1_f_0_s_0, r_1_f_1_s_0, r_1_f_1_s_1, r_1_f_2_s_0, r_1_f_2_s_1]
        """

        def _generate_variable_batch_pooled_embedding_batch(
            keys: List[str],
            dims: List[int],
            splits: List[int],
            batch_size_per_rank_per_feature: List[List[int]],
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            world_size = len(splits)
            offsets = [0] + list(itertools.accumulate(splits))
            local_embs = {}

            feature_ind = 0
            for key, dim in zip(keys, dims):
                for rank in range(world_size):
                    local_batch_size = batch_size_per_rank_per_feature[rank][
                        feature_ind
                    ]
                    if rank not in local_embs:
                        local_embs[rank] = {}
                    local_embs[rank][key] = torch.rand(
                        dim * local_batch_size, dtype=torch.float
                    )
                feature_ind += 1

            in_tensor: List[torch.Tensor] = []
            out_tensor: List[torch.Tensor] = []
            for i in range(world_size):
                in_keys = keys[offsets[i] : offsets[i + 1]]
                input_tensor_list = []
                for rank in range(world_size):
                    input_tensor_list += [local_embs[rank][key] for key in in_keys]
                input_tensor = torch.cat(input_tensor_list)
                in_tensor.append(input_tensor)

                output_tensor = torch.cat([local_embs[i][key] for key in keys])
                out_tensor.append(output_tensor)

            return in_tensor, out_tensor

        _input, output = _generate_variable_batch_pooled_embedding_batch(
            keys=keys,
            dims=dims,
            splits=splits,
            batch_size_per_rank_per_feature=batch_size_per_rank_per_feature_pre_a2a,
        )

        kwargs_per_rank = []
        for rank in range(world_size):
            kwargs_per_rank.append(
                {
                    "_input": _input[rank],
                    "output": output[rank],
                    "backend": backend,
                    "emb_dim_per_rank_per_feature": emb_dim_per_rank_per_feature,
                    "batch_size_per_rank_per_feature": batch_size_per_rank_per_feature_post_a2a_per_rank[
                        rank
                    ],
                    "batch_size_per_feature_pre_a2a": batch_size_per_rank_per_feature_pre_a2a[
                        rank
                    ],
                    "qcomms_config": qcomms_config,
                }
            )

        self._run_multi_process_test_per_rank(
            callable=self._run_test_dist,
            world_size=world_size,
            kwargs_per_rank=kwargs_per_rank,
        )


class TestJaggedTensorAllToAll(MultiProcessTestBase):
    @staticmethod
    def _test_jt_all_to_all(
        rank: int,
        world_size: int,
    ) -> None:
        backend = "nccl"
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            device = ctx.device
            if ctx.rank == 0:
                # [
                #   [1], [2,2], [3,3,3], [4,4,4,4]
                # ]
                jt = JaggedTensor(
                    values=torch.tensor(
                        [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int, device=device
                    ),
                    lengths=torch.tensor(
                        [1, 2, 3, 4], dtype=torch.int32, device=device
                    ),
                )
                input_splits = torch.tensor([3, 1], dtype=torch.int32, device=device)
                output_splits = torch.tensor([3, 2], dtype=torch.int32, device=device)
            else:
                # [
                #   [5,5,5,5,5], [6,6,6,6,6,6], [7,7,7,7,7,7,7]
                # ]
                jt = JaggedTensor(
                    values=torch.tensor(
                        [5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7],
                        device=device,
                        dtype=torch.int,
                    ),
                    lengths=torch.tensor([5, 6, 7], dtype=torch.int, device=device),
                )
                input_splits = torch.tensor([2, 1], dtype=torch.int32, device=device)
                output_splits = torch.tensor([1, 1], dtype=torch.int32, device=device)

        jt_all_to_all = JaggedTensorAllToAll(
            jt,
            num_items_to_send=input_splits,
            num_items_to_receive=output_splits,
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            pg=ctx.pg,
        )

        jt_out = jt_all_to_all.wait()

        torch.testing.assert_close(
            jt_out.values(),
            torch.tensor(
                (
                    [1, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6]
                    if ctx.rank == 0
                    else [4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7]
                ),
                dtype=torch.int,
                device=device,
            ),
        )

        torch.testing.assert_close(
            jt_out.lengths(),
            torch.tensor(
                [1, 2, 3, 5, 6] if ctx.rank == 0 else [4, 7],
                dtype=torch.int,
                device=device,
            ),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_jt_all_to_all(
        self,
    ) -> None:
        world_size = 2
        self._run_multi_process_test(
            callable=self._test_jt_all_to_all, world_size=world_size
        )


class GetRecatOverflowTest(unittest.TestCase):
    def test_get_recat_logs_and_raises_on_int32_overflow(self) -> None:
        """Verify that _get_recat raises RuntimeError and logs context when
        batch_size_per_rank contains values that overflow int32."""
        overflow_value = 2_147_483_648  # INT32_MAX + 1
        with self.assertRaises(RuntimeError) as ctx, self.assertLogs(
            level="ERROR"
        ) as log:
            _get_recat(
                local_split=2,
                num_splits=4,
                stagger=1,
                device=torch.device("cpu"),
                batch_size_per_rank=[32, overflow_value, 32, 32],
            )

        self.assertIn("overflow", str(ctx.exception).lower())

        logged = "\n".join(log.output)
        self.assertIn("_get_recat", logged)
        self.assertIn("batch_size_per_rank", logged)
        self.assertIn(str(overflow_value), logged)
        self.assertIn("input_offset", logged)
        self.assertIn("output_offset", logged)

    def test_get_recat_no_overflow_unchanged_behavior(self) -> None:
        """Verify that _get_recat still returns a valid recat tensor when
        batch_size_per_rank values are within int32 range."""
        result = _get_recat(
            local_split=2,
            num_splits=4,
            stagger=1,
            device=torch.device("cpu"),
            batch_size_per_rank=[32, 64, 32, 64],
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, torch.Tensor)

    def test_get_recat_no_overflow_uniform_batch(self) -> None:
        """Verify that _get_recat returns a valid recat tensor for the
        non-variable-batch (uniform) path."""
        result = _get_recat(
            local_split=2,
            num_splits=4,
            stagger=1,
            device=torch.device("cpu"),
            batch_size_per_rank=[32, 32, 32, 32],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.int32)


class CollectiveTagFromTest(unittest.TestCase):
    # Single-process unit tests for the _collective_tag_from helper.
    _INT32_MAX = 0x7FFFFFFF

    def test_empty_parts_fits_signed_int32(self) -> None:
        # Regression: the FNV-1a seed (0x811C9DC5) exceeds signed-int32 max,
        # so a previous implementation that only masked inside the loop body
        # returned the raw seed when parts was empty, violating the docstring's
        # int32-fit contract and risking overflow in callers that assign the
        # tag to an int32 splits tensor.
        tag = _collective_tag_from()
        self.assertGreaterEqual(tag, 0)
        self.assertLessEqual(tag, self._INT32_MAX)

    def test_various_parts_all_fit_signed_int32(self) -> None:
        # Sanity: a range of realistic call shapes all stay within int32.
        # Mirrors the actual tag shapes used by the two production call sites
        # (KJTAllToAllSplitsAwaitable, FusedKJTListSplitsAwaitable). If FNV-1a's
        # mixing changes or one of the call sites grows a new identity field,
        # this catches a regression across both.
        cases = [
            # KJTAllToAllSplits production shape: (name, keys, tuple(splits))
            ("KJTAllToAllSplits", ["f0", "f1"], (1, 1)),
            # FusedKJTListSplits production shape: (name, tuple of per-request
            # entries — meta entries are (keys, splits, count); non-meta is None).
            (
                "FusedKJTListSplits",
                (
                    (("f0", "f1"), (1, 1), 2),
                    None,
                    (("f2",), (1,), 1),
                ),
            ),
            # Boundary: empty keys / empty splits
            ("KJTAllToAllSplits", [], ()),
            # Boundary: empty fused awaitables list
            ("FusedKJTListSplits", ()),
            # Long single string — guards against FNV-1a length-related issues
            ("x" * 1024,),
        ]
        for parts in cases:
            with self.subTest(parts=parts):
                tag = _collective_tag_from(*parts)
                self.assertGreaterEqual(tag, 0)
                self.assertLessEqual(tag, self._INT32_MAX)

    def test_deterministic(self) -> None:
        # Determinism is the main reason this exists instead of hash().
        # If the implementation accidentally introduces nondeterminism
        # (e.g., switching to hash() under PYTHONHASHSEED randomization),
        # different ranks would compute different tags and the validation
        # would false-positive.
        self.assertEqual(
            _collective_tag_from("KJTAllToAllSplits", ["f0", "f1"], 3),
            _collective_tag_from("KJTAllToAllSplits", ["f0", "f1"], 3),
        )

    def test_separator_is_unambiguous(self) -> None:
        # Regression: with a "," separator, _collective_tag_from("a", "b") and
        # _collective_tag_from("a,b") both serialize to the bytes b"a,b" and
        # collide, silently disabling validation for any future call site that
        # passes raw strings containing commas. The NUL separator avoids the
        # collision because collective identifier parts cannot contain \x00.
        self.assertNotEqual(
            _collective_tag_from("a", "b"),
            _collective_tag_from("a,b"),
        )


class SplitsAllToAllCollectiveTagTest(MultiProcessTestBase):
    def __init__(self, methodName: str = "runTest") -> None:
        # Force spawn: forkserver caches the env from when it was first
        # started, so TORCHREC_VALIDATE_COLLECTIVES=1 set in setUp does not
        # propagate to workers that fork from a forkserver booted by an
        # earlier test in the file. Mirrors the py3.14 remediation in
        # MultiProcessTestBase.__init__.
        super().__init__(methodName, mp_init_mode="spawn")

    def setUp(self) -> None:
        super().setUp()
        os.environ["TORCHREC_VALIDATE_COLLECTIVES"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("TORCHREC_VALIDATE_COLLECTIVES", None)
        super().tearDown()

    @classmethod
    def _run_test_matching_tags(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([rank] * world_size, dtype=torch.int64),
            torch.tensor([rank + 1] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            collective_tag=42,
            collective_tag_parts=("test_matching",),
        )
        result = awaitable.wait()
        # Tag row should be stripped — only 2 original rows returned
        assert len(result) == 2
        # Row 0: each rank sent its rank number
        assert result[0] == list(range(world_size))
        # Row 1: each rank sent rank+1
        assert result[1] == list(range(1, world_size + 1))
        dist.destroy_process_group()

    @classmethod
    def _run_test_mismatched_tags(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            collective_tag=rank,  # rank 0 -> tag 0, rank 1 -> tag 1
            collective_tag_parts=("test_mismatched",),
        )
        try:
            awaitable.wait()
            raise AssertionError("Expected RuntimeError for mismatched tags")
        except RuntimeError as e:
            msg = str(e).lower()
            assert "collective mismatch" in msg
            assert "test_mismatched" in msg
        dist.destroy_process_group()

    @classmethod
    def _run_test_no_tag(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([rank] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            # No collective_tag — backward compatible
        )
        result = awaitable.wait()
        assert len(result) == 1
        assert result[0] == list(range(world_size))
        dist.destroy_process_group()

    def test_matching_tags_succeed(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_matching_tags,
            world_size=2,
            backend="gloo",
        )

    def test_mismatched_tags_raise(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_mismatched_tags,
            world_size=2,
            backend="gloo",
        )

    def test_no_tag_passthrough(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_no_tag,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_tag_does_not_corrupt_overflow_check(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        import torchrec.distributed.dist_data as dd

        dd._TORCHREC_OVERFLOW_DEBUG = True

        overflow_value = 2**31 + 1  # exceeds int32 max
        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
            # Last row has overflow values — overflow check must catch this
            torch.tensor([overflow_value] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            collective_tag=42,  # small tag — would NOT trigger overflow
            collective_tag_parts=("test_overflow",),
        )
        try:
            awaitable.wait()
            raise AssertionError(
                "Expected RuntimeError for int32 overflow in real data"
            )
        except RuntimeError as e:
            # Should detect overflow in the real data, not skip it
            # because the tag row (value=42) was checked instead
            assert "corrupted" in str(e).lower()
        dist.destroy_process_group()

    def test_tag_does_not_corrupt_overflow_check(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_tag_does_not_corrupt_overflow_check,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_kjt_all_to_all_emits_tag(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        kjt = KeyedJaggedTensor(
            keys=["f0", "f1"],
            values=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            lengths=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        )
        kjt_a2a = KJTAllToAll(
            pg=none_throws(dist.group.WORLD),
            splits=[1, 1],
        )
        splits_aw = kjt_a2a(kjt)
        assert splits_aw._splits_awaitable._tag_appended is True
        result = splits_aw.wait().wait()
        assert isinstance(result, KeyedJaggedTensor)
        dist.destroy_process_group()

    def test_kjt_all_to_all_emits_tag(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_kjt_all_to_all_emits_tag,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_fused_kjt_list_splits_emits_tag(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        kjt = KeyedJaggedTensor(
            keys=["f0"],
            values=torch.tensor([1, 2], dtype=torch.int64),
            lengths=torch.tensor([1, 1], dtype=torch.int64),
        )
        meta = KJTSplitsAllToAllMeta(
            pg=none_throws(dist.group.WORLD),
            _input=kjt,
            splits=[1],
            splits_tensors=[
                torch.tensor([1] * world_size, dtype=torch.int64),
            ],
            input_splits=[[1] * world_size],
            input_tensors=[kjt.lengths()],
            labels=["lengths"],
            keys=["f0"],
            device=torch.device("cpu"),
            stagger=1,
        )
        request = KJTListSplitsAwaitable(
            awaitables=cast(List[Awaitable[Awaitable[KeyedJaggedTensor]]], [meta]),
            ctx=NullShardingContext(),
        )
        fused = FusedKJTListSplitsAwaitable(
            requests=[request],
            contexts=[NullShardingContext()],
            pg=none_throws(dist.group.WORLD),
        )
        assert fused._splits_awaitable is not None
        assert fused._splits_awaitable._tag_appended is True
        fused._splits_awaitable.wait()
        dist.destroy_process_group()

    def test_fused_kjt_list_splits_emits_tag(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_fused_kjt_list_splits_emits_tag,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_tag_not_appended_when_validation_disabled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([rank] * world_size, dtype=torch.int64),
        ]
        with unittest.mock.patch.dict(
            os.environ, {"TORCHREC_VALIDATE_COLLECTIVES": "0"}
        ):
            awaitable = SplitsAllToAllAwaitable(
                input_tensors=input_tensors,
                pg=none_throws(dist.group.WORLD),
                collective_tag=42,
            )
            assert awaitable._tag_appended is False
            result = awaitable.wait()
        assert len(result) == 1
        assert result[0] == list(range(world_size))
        dist.destroy_process_group()

    def test_tag_not_appended_when_validation_disabled(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_tag_not_appended_when_validation_disabled,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_realistic_mismatched_tags(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        tag = (
            _collective_tag_from("KJTAllToAllSplits", ["f0", "f1"], 3)
            if rank == 0
            else _collective_tag_from("FusedKJTListSplits", 2, [3, 3])
        )
        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            collective_tag=tag,
            collective_tag_parts=(f"rank{rank}_collective",),
        )
        try:
            awaitable.wait()
            raise AssertionError("Expected RuntimeError for mismatched tags")
        except RuntimeError as e:
            msg = str(e).lower()
            assert "collective mismatch" in msg
            assert f"rank{rank}_collective" in str(e)
        dist.destroy_process_group()

    def test_realistic_mismatched_tags_raise(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_realistic_mismatched_tags,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_feature_keys_divergence_raises(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        # Simulates the realistic bug class: both ranks reach the same call
        # site (KJTAllToAllSplits) with the same len(input_splits), but their
        # input.keys() diverge. The tag must distinguish on input.keys() so the
        # mismatch is caught instead of corrupting the all2all silently.
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        keys = ["f0", "f1"] if rank == 0 else ["f0", "f2"]
        input_splits_len = 3
        tag = _collective_tag_from("KJTAllToAllSplits", keys, input_splits_len)
        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            collective_tag=tag,
            collective_tag_parts=("KJTAllToAllSplits", keys, input_splits_len),
        )
        try:
            awaitable.wait()
            raise AssertionError(
                "Expected RuntimeError for keys divergence at same call site"
            )
        except RuntimeError as e:
            assert "collective mismatch" in str(e).lower()
            assert "KJTAllToAllSplits" in str(e)
        dist.destroy_process_group()

    def test_feature_keys_divergence_raises(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_feature_keys_divergence_raises,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_splits_divergence_raises(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        # Companion to test_feature_keys_divergence_raises: same call site,
        # same keys, but the per-feature sharding plan (`splits`) diverges
        # across ranks. This is a config-bug class — the planner is supposed
        # to produce the same plan on every rank — but it's exactly the
        # silent-corruption mode the validation must catch. The tag must
        # therefore include the splits identity, not just len(splits).
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        keys = ["f0", "f1"]  # rank-invariant
        splits = (2, 0) if rank == 0 else (1, 1)  # divergent plan
        tag = _collective_tag_from("KJTAllToAllSplits", keys, splits)
        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        awaitable = SplitsAllToAllAwaitable(
            input_tensors=input_tensors,
            pg=none_throws(dist.group.WORLD),
            collective_tag=tag,
            collective_tag_parts=("KJTAllToAllSplits", keys, splits),
        )
        try:
            awaitable.wait()
            raise AssertionError(
                "Expected RuntimeError for splits divergence at same call site"
            )
        except RuntimeError as e:
            assert "collective mismatch" in str(e).lower()
            assert "KJTAllToAllSplits" in str(e)
        dist.destroy_process_group()

    def test_splits_divergence_raises(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_splits_divergence_raises,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_fused_splits_divergence_raises(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        # FusedKJTListSplits-layer counterpart to test_splits_divergence_raises.
        # Both ranks reach FusedKJTListSplitsAwaitable with the same structural
        # arity (1 request, 1 splits tensor each) but their per-request
        # sharding plan diverges. With the old structural-only tag
        # ("FusedKJTListSplits", len(splits_tensors), self._lengths) these
        # would collide; with the tightened per-request identity
        # (keys, splits, len(splits_tensors)) the tag detects the divergence.
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        # Construct a KJTSplitsAllToAllMeta whose `splits` differs per rank.
        # Everything else is rank-invariant: same keys, same shape.
        kjt = KeyedJaggedTensor(
            keys=["f0", "f1"],
            values=torch.tensor([1, 2], dtype=torch.int64),
            lengths=torch.tensor([1, 1], dtype=torch.int64),
        )
        splits = [2, 0] if rank == 0 else [1, 1]  # divergent sharding plan
        meta = KJTSplitsAllToAllMeta(
            pg=none_throws(dist.group.WORLD),
            _input=kjt,
            splits=splits,
            splits_tensors=[
                torch.tensor([1] * world_size, dtype=torch.int64),
            ],
            input_splits=[[1] * world_size],
            input_tensors=[kjt.lengths()],
            labels=["lengths"],
            keys=["f0", "f1"],
            device=torch.device("cpu"),
            stagger=1,
        )
        request = KJTListSplitsAwaitable(
            awaitables=cast(List[Awaitable[Awaitable[KeyedJaggedTensor]]], [meta]),
            ctx=NullShardingContext(),
        )
        fused = FusedKJTListSplitsAwaitable(
            requests=[request],
            contexts=[NullShardingContext()],
            pg=none_throws(dist.group.WORLD),
        )
        try:
            none_throws(fused._splits_awaitable).wait()
            raise AssertionError(
                "Expected RuntimeError for fused splits divergence at same call site"
            )
        except RuntimeError as e:
            assert "collective mismatch" in str(e).lower()
            assert "FusedKJTListSplits" in str(e)
        dist.destroy_process_group()

    def test_fused_splits_divergence_raises(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_fused_splits_divergence_raises,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_fused_cross_type_divergence_raises(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        # Cross-type variant: both ranks reach FusedKJTListSplitsAwaitable with
        # the same total number of splits tensors (1 — only the meta entry
        # contributes; the non-meta entry contributes 0), so the underlying
        # SplitsAllToAllAwaitable input shape matches and the all2all itself
        # can run. But the per-position TYPE in `_awaitables` differs across
        # ranks (meta vs non-meta swap). The position-preserving `None`
        # placeholder in the tag must distinguish these so the divergence is
        # caught — without it (e.g., if the production code filtered non-metas
        # out before tagging), the two ranks would compute the same tag and
        # silently corrupt the fused splits-all2all.
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        kjt = KeyedJaggedTensor(
            keys=["f0"],
            values=torch.tensor([1, 2], dtype=torch.int64),
            lengths=torch.tensor([1, 1], dtype=torch.int64),
        )
        meta = KJTSplitsAllToAllMeta(
            pg=none_throws(dist.group.WORLD),
            _input=kjt,
            splits=[1],
            splits_tensors=[
                torch.tensor([1] * world_size, dtype=torch.int64),
            ],
            input_splits=[[1] * world_size],
            input_tensors=[kjt.lengths()],
            labels=["lengths"],
            keys=["f0"],
            device=torch.device("cpu"),
            stagger=1,
        )
        # Stub Awaitable that's intentionally NOT a KJTSplitsAllToAllMeta.
        # spec=Awaitable makes isinstance(non_meta, Awaitable) succeed and
        # isinstance(non_meta, KJTSplitsAllToAllMeta) fail — the only two
        # checks production code performs on entries of `_awaitables` during
        # tag construction. We never reach FusedKJTListSplitsAwaitable._wait_impl
        # (which would call non_meta.wait()) because the test triggers only
        # the splits stage via _splits_awaitable.wait().
        non_meta = unittest.mock.MagicMock(spec=Awaitable)

        # Rank 0 has [meta, non_meta]; rank 1 has [non_meta, meta]. Both
        # contribute the same single splits tensor to the all2all (only the
        # meta does), so the collective input shape matches across ranks.
        if rank == 0:
            mixed = [meta, non_meta]
        else:
            mixed = [non_meta, meta]

        request = KJTListSplitsAwaitable(
            awaitables=cast(List[Awaitable[Awaitable[KeyedJaggedTensor]]], mixed),
            ctx=NullShardingContext(),
        )
        fused = FusedKJTListSplitsAwaitable(
            requests=[request],
            contexts=[NullShardingContext()],
            pg=none_throws(dist.group.WORLD),
        )
        try:
            none_throws(fused._splits_awaitable).wait()
            raise AssertionError(
                "Expected RuntimeError for cross-type _awaitables divergence"
            )
        except RuntimeError as e:
            assert "collective mismatch" in str(e).lower()
            assert "FusedKJTListSplits" in str(e)
        dist.destroy_process_group()

    def test_fused_cross_type_divergence_raises(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_fused_cross_type_divergence_raises,
            world_size=2,
            backend="gloo",
        )

    def test_tag_raises_on_small_dtype(self) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"TORCHREC_VALIDATE_COLLECTIVES": "1"}
        ):
            mock_pg = unittest.mock.MagicMock()
            mock_pg.size.return_value = 2

            large_tag = 0x7FFF_FFFF  # max signed int32, exceeds int16 range
            input_tensors = [torch.tensor([1, 1], dtype=torch.int16)]
            with self.assertRaises(ValueError) as ctx:
                SplitsAllToAllAwaitable(
                    input_tensors=input_tensors,
                    pg=mock_pg,
                    collective_tag=large_tag,
                )
            self.assertIn("exceeds", str(ctx.exception).lower())

    @classmethod
    def _run_test_mismatch_emits_scuba_event(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        with unittest.mock.patch(
            "torchrec.distributed.dist_data.EventLoggingHandler.log_event"
        ) as mock_log_event:
            awaitable = SplitsAllToAllAwaitable(
                input_tensors=input_tensors,
                pg=none_throws(dist.group.WORLD),
                collective_tag=rank,  # divergent tag per rank → forces mismatch
                collective_tag_parts=("scuba_test", "rank_specific"),
            )
            try:
                awaitable.wait()
                raise AssertionError("Expected RuntimeError for mismatched tags")
            except RuntimeError:
                pass
            # log_event must fire exactly once on the failing rank.
            assert (
                mock_log_event.call_count == 1
            ), f"expected 1 log_event call, got {mock_log_event.call_count}"
            kwargs = mock_log_event.call_args.kwargs
            assert kwargs["component"] == "input_dist"
            assert (
                kwargs["event_name"]
                == "SplitsAllToAllAwaitable.collective_tag_mismatch"
            )
            # Only check the enum's string value to keep the test independent
            # of how EventType is imported in the test module.
            assert kwargs["event_type"].value == "FAILURE"
            metadata = kwargs["metadata"]
            assert metadata["expected_tag"] == str(rank)
            assert metadata["pg_rank"] == str(rank)
            assert metadata["world_size"] == str(world_size)
            assert metadata["collective_kind"] == "scuba_test"
            assert "scuba_test" in metadata["collective_label"]
            # On WORLD, global rank and PG-local rank coincide.
            assert metadata["global_rank"] == str(rank)
            # Both lists are stringified; on a 2-rank job with divergent
            # tags, every peer with a different tag appears.
            assert metadata["mismatched_peer_global_ranks"].startswith("[")
            assert metadata["mismatched_peer_pg_ranks"].startswith("[")
        dist.destroy_process_group()

    def test_mismatch_emits_scuba_event(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_mismatch_emits_scuba_event,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_mismatch_event_uses_global_ranks_on_subgroup(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        """Run on a 4-rank job; create a sub-PG of [1, 3]. On that sub-PG
        the local ranks are 0 and 1 but global ranks are 1 and 3. Verify
        that the Scuba event reports global ranks (actionable for triage),
        not PG-local indices."""
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        # new_group is collective on WORLD, so every rank must call it,
        # even ranks not in the resulting subgroup.
        subgroup = dist.new_group(ranks=[1, 3], backend=backend)

        if rank not in (1, 3):
            # Non-members do not run the awaitable; just synchronize and exit.
            dist.barrier()
            dist.destroy_process_group()
            return

        # Ranks 1 and 3 run a collective on the subgroup with divergent
        # tags so the mismatch path fires. PG-local ranks here are 0 and 1.
        input_tensors = [torch.tensor([1, 1], dtype=torch.int64)]
        with unittest.mock.patch(
            "torchrec.distributed.dist_data.EventLoggingHandler.log_event"
        ) as mock_log_event:
            awaitable = SplitsAllToAllAwaitable(
                input_tensors=input_tensors,
                pg=subgroup,
                collective_tag=rank,  # divergent tag per rank → mismatch
                collective_tag_parts=("subgroup_test",),
            )
            try:
                awaitable.wait()
                raise AssertionError("Expected RuntimeError for mismatched tags")
            except RuntimeError:
                pass

            assert (
                mock_log_event.call_count == 1
            ), f"expected 1 log_event call, got {mock_log_event.call_count}"
            metadata = mock_log_event.call_args.kwargs["metadata"]

            # PG-local rank for rank 1 is 0; for rank 3 is 1.
            expected_pg_rank = 0 if rank == 1 else 1
            assert metadata["pg_rank"] == str(expected_pg_rank), (
                f"expected pg_rank={expected_pg_rank}, " f"got {metadata['pg_rank']}"
            )
            # Global rank should be the actual global rank, not the PG-local.
            assert metadata["global_rank"] == str(
                rank
            ), f"expected global_rank={rank}, got {metadata['global_rank']}"
            # The peer is the OTHER member of [1, 3].
            other_global = 3 if rank == 1 else 1
            other_pg_local = 1 if rank == 1 else 0
            assert metadata["mismatched_peer_global_ranks"] == str([other_global]), (
                f"expected mismatched_peer_global_ranks=[{other_global}], "
                f"got {metadata['mismatched_peer_global_ranks']}"
            )
            assert metadata["mismatched_peer_pg_ranks"] == str([other_pg_local]), (
                f"expected mismatched_peer_pg_ranks=[{other_pg_local}], "
                f"got {metadata['mismatched_peer_pg_ranks']}"
            )

        dist.barrier()  # keep WORLD alive while non-member ranks finish.
        dist.destroy_process_group()

    def test_mismatch_event_uses_global_ranks_on_subgroup(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_mismatch_event_uses_global_ranks_on_subgroup,
            world_size=4,
            backend="gloo",
        )

    @classmethod
    def _run_test_mismatch_raises_when_logger_unavailable(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        # Simulate OSS / inference bundle where logging_handlers is absent.
        # The off-path must still raise the same RuntimeError unchanged.
        with unittest.mock.patch(
            "torchrec.distributed.dist_data._HAS_EVENT_LOGGER", False
        ):
            awaitable = SplitsAllToAllAwaitable(
                input_tensors=input_tensors,
                pg=none_throws(dist.group.WORLD),
                collective_tag=rank,
                collective_tag_parts=("logger_off",),
            )
            try:
                awaitable.wait()
                raise AssertionError("Expected RuntimeError for mismatched tags")
            except RuntimeError as e:
                assert "collective mismatch" in str(e).lower()
                assert "logger_off" in str(e)
        dist.destroy_process_group()

    def test_mismatch_raises_when_logger_unavailable(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_mismatch_raises_when_logger_unavailable,
            world_size=2,
            backend="gloo",
        )

    @classmethod
    def _run_test_mismatch_raises_when_logger_throws(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)

        input_tensors = [
            torch.tensor([1] * world_size, dtype=torch.int64),
        ]
        # If log_event itself throws (e.g. Scuba handler bug, metadata
        # serialization failure), the diagnostic RuntimeError must still
        # fire — telemetry must never suppress the actual error.
        with unittest.mock.patch(
            "torchrec.distributed.dist_data.EventLoggingHandler.log_event",
            side_effect=RuntimeError("simulated scuba handler failure"),
        ):
            awaitable = SplitsAllToAllAwaitable(
                input_tensors=input_tensors,
                pg=none_throws(dist.group.WORLD),
                collective_tag=rank,
                collective_tag_parts=("logger_throws",),
            )
            try:
                awaitable.wait()
                raise AssertionError("Expected RuntimeError for mismatched tags")
            except RuntimeError as e:
                # Must surface the mismatch diagnostic, not the scuba failure.
                assert (
                    "collective mismatch" in str(e).lower()
                ), f"expected mismatch diagnostic, got: {e}"
                assert "logger_throws" in str(e)
                assert "simulated scuba handler failure" not in str(e)
        dist.destroy_process_group()

    def test_mismatch_raises_when_logger_throws(self) -> None:
        self._run_multi_process_test(
            callable=self._run_test_mismatch_raises_when_logger_throws,
            world_size=2,
            backend="gloo",
        )
