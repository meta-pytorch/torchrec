#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import random
import unittest
from typing import Dict, Generator, Iterable, List, Optional, Tuple, TypeVar, Union

import hypothesis.strategies as st
import torch
import torch.distributed as dist
from hypothesis import given, settings
from torchrec.distributed.dist_data import (
    _get_recat,
    JaggedTensorAllToAll,
    KJTAllToAll,
    KJTAllToAllSplitsAwaitable,
    PooledEmbeddingsAllGather,
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsReduceScatter,
    SequenceEmbeddingsAllToAll,
    VariableBatchPooledEmbeddingsAllToAll,
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
