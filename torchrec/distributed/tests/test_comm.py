#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import itertools
import multiprocessing
import os
import unittest
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import hypothesis.strategies as st
import torch
import torch.distributed as dist
import torchrec
import torchrec.distributed.comm_ops as comm_ops
from hypothesis import given, settings
from torch.distributed.distributed_c10d import GroupMember
from torchrec.test_utils import get_free_port, seed_and_log

torch.ops.import_module("fbgemm_gpu.sparse_ops")


@dataclass
class _CompileConfig:
    # backend is None means no compilation
    backend: Optional[str] = "inductor"
    fullgraph: bool = True
    skip_sync_backward: bool = False
    skip_compile_backward: bool = False
    test_compiled_with_noncompiled_ranks: bool = False


def compile_config_to_fn_transform(
    compile_config: Optional[_CompileConfig],
) -> Callable:
    if compile_config is None:
        return lambda x: x

    return functools.partial(
        torch.compile,
        backend=compile_config.backend,
        fullgraph=compile_config.fullgraph,
        dynamic=True,
    )


def _copy_input_tensors(t, device):
    if isinstance(t, torch.Tensor):
        ret = t.detach().clone().to(device)
        ret.requires_grad = True
        ret.retain_grad()
        return ret
    elif isinstance(t, list):
        return [_copy_input_tensors(_t, device) for _t in t]
    else:
        raise ValueError(f"Unsupported type {type(t)}")


def _grad_detach_clone(t):
    if isinstance(t, torch.Tensor):
        # pyrefly: ignore[missing-attribute]
        return t.grad.detach().clone()
    elif isinstance(t, list):
        return [_grad_detach_clone(_t) for _t in t]
    else:
        raise ValueError(f"Unsupported type {type(t)}")


def _assert_close(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        for _a, _e in zip(actual, expected):
            _assert_close(_a, _e)
    else:
        raise ValueError(f"Unsupported type {type(expected)}")


def _test_async_sync_compile(
    fn,
    input_tensor: Union[torch.Tensor, List[torch.Tensor]],
    device: torch.device,
    compile_config: _CompileConfig,
    rank: int,
    *args,
    **kwargs,
) -> None:
    input_tensor_async = _copy_input_tensors(input_tensor, device)
    input_tensor_sync = _copy_input_tensors(input_tensor, device)
    input_tensor_compile = _copy_input_tensors(input_tensor, device)

    # Async
    # pyrefly: ignore[implicit-import]
    torchrec.distributed.comm_ops.set_use_sync_collectives(False)
    out = fn(input_tensor_async, *args, **kwargs)
    out.retain_grad()
    out.backward(out)
    async_fwd_out = out.clone()
    async_bwd_out = _grad_detach_clone(input_tensor_async)

    # Sync
    # pyrefly: ignore[implicit-import]
    torchrec.distributed.comm_ops.set_use_sync_collectives(True)
    out = fn(input_tensor_sync, *args, **kwargs)
    sync_fwd_out = out.clone()
    _assert_close(sync_fwd_out, async_fwd_out)

    if not compile_config.skip_sync_backward:
        out.retain_grad()
        out.backward(out)
        sync_bwd_out = _grad_detach_clone(input_tensor_sync)
        _assert_close(sync_bwd_out, async_bwd_out)

    if compile_config.backend is not None:
        fn_transform = compile_config_to_fn_transform(compile_config)

        # pyrefly: ignore[implicit-import]
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            if compile_config.test_compiled_with_noncompiled_ranks and rank == 1:
                # Turn off compilation for rank==1 to test compatibility of compiled rank and non-compiled
                fn_transform = lambda x: x

            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            out = fn_transform(fn)(
                input_tensor_compile,
                *args,
                **kwargs,
            )
            compile_fwd_out = out.clone()
            _assert_close(compile_fwd_out, sync_fwd_out)

            if (
                not compile_config.skip_sync_backward
                and not compile_config.skip_compile_backward
            ):
                out.retain_grad()
                out.backward(out)
                compile_bwd_out = _grad_detach_clone(input_tensor_compile)

                _assert_close(compile_bwd_out, sync_bwd_out)


class TestAllToAll(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.WORLD_SIZE = 2

    def tearDown(self) -> None:
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _run_multi_process_test(
        self,
        world_size: int,
        backend: str,
        callable: Callable[[], None],
        *args,
        **kwargs,
    ) -> None:
        processes = []
        ctx = multiprocessing.get_context("spawn")
        for rank in range(world_size):
            p = ctx.Process(
                target=callable,
                args=(
                    rank,
                    world_size,
                    backend,
                    *args,
                ),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    @classmethod
    def _test_alltoall_sequence(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        compile_config: _CompileConfig,
        specify_pg: bool,
        gradient_division: bool,
        skip_dynamo_backwards: bool = False,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        ranks = 2
        tables_mp = [[0], [1, 2]]
        lengths_dp = [
            torch.tensor([[1, 2], [1, 1], [2, 1]], dtype=torch.int),
            torch.tensor([[1, 2], [2, 1], [3, 1]], dtype=torch.int),
        ]  # W, T_g, B_l
        lengths_a2a = [
            torch.tensor([[[1, 2]], [[1, 2]]], dtype=torch.int),  # Rank 0
            torch.tensor(
                [
                    [[1, 1], [2, 1]],  # from Rank 0
                    [[2, 1], [3, 1]],  # from rank 1
                ],
                dtype=torch.int,
            ),  # Rank 1
        ]  # W, W, T_l, B_l
        lengths_mp = [
            torch.tensor(
                [
                    [1, 2, 1, 2],
                ],
                dtype=torch.int,
            ),
            torch.tensor([[1, 1, 2, 1], [2, 1, 3, 1]], dtype=torch.int),
        ]  # w, t_l, b_g
        input_seg = list(itertools.accumulate([0] + [len(i) for i in tables_mp]))
        input_splits = [
            [
                int(lengths_dp[i][input_seg[j] : input_seg[j + 1], :].sum())
                for j in range(ranks)
            ]
            for i in range(ranks)
        ]
        output_splits = [lengths_a2a[i].sum(dim=(1, 2)).tolist() for i in range(ranks)]
        table_dim = 3
        num_features_per_rank = [len(features) for features in tables_mp]
        seq_all2all_forward_recat = []
        for j in range(ranks):
            for i in range(num_features_per_rank[rank]):
                seq_all2all_forward_recat.append(j + i * ranks)
        seq_all2all_forward_recat_tensor = torch.IntTensor(seq_all2all_forward_recat)
        seq_all2all_backward_recat = []
        for i in range(num_features_per_rank[rank]):
            for j in range(ranks):
                seq_all2all_backward_recat.append(i + j * num_features_per_rank[rank])

        seq_all2all_backward_recat_tensor = torch.IntTensor(seq_all2all_backward_recat)
        input_embeddings = torch.rand(
            int(lengths_mp[rank].sum()),
            table_dim,
            device=device,
            requires_grad=True,
        )
        lengths_after_sparse_data_all2all = torch.IntTensor(lengths_mp[rank])

        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.alltoall_sequence(*args, **kwargs).wait()

        comm_ops.set_gradient_division(gradient_division)
        _test_async_sync_compile(
            fn,
            input_embeddings,
            device,
            compile_config,
            rank,
            forward_recat_tensor=seq_all2all_forward_recat_tensor.cuda(),
            backward_recat_tensor=seq_all2all_backward_recat_tensor.cuda(),
            lengths_after_sparse_data_all2all=lengths_after_sparse_data_all2all.cuda(),
            input_splits=input_splits[rank],
            output_splits=output_splits[rank],
            group=pg if specify_pg else None,
        )
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        specify_pg=st.sampled_from([True]),
        gradient_division=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_alltoall_sequence(
        self,
        specify_pg: bool,
        gradient_division: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence,
            compile_config=_CompileConfig(),
            specify_pg=specify_pg,
            gradient_division=gradient_division,
        )

    @classmethod
    def _test_alltoall_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        compile_config: _CompileConfig,
        specify_pg: bool,
        gradient_division: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        # Each rank's local batch size

        batch_size_per_rank = [4] * world_size

        # Global batch size is the sum of all rank's local batch size
        B_global = sum(batch_size_per_rank)
        # sum of dimensions of the embedding tables hosted on each rank
        dim_sum_per_rank = [8] * world_size

        D_local_sum = dim_sum_per_rank[rank]

        # Construct pooled embeddings
        pooled_embs = torch.randn([B_global, D_local_sum], requires_grad=True).to(
            device
        )

        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.alltoall_pooled(*args, **kwargs).wait()

        comm_ops.set_gradient_division(gradient_division)
        _test_async_sync_compile(
            fn,
            pooled_embs,
            device,
            compile_config,
            rank,
            batch_size_per_rank,
            dim_sum_per_rank,
            pg,
        )

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        specify_pg=st.sampled_from([True]),
        test_compiled_with_noncompiled_ranks=st.sampled_from([False, True]),
        gradient_division=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_alltoall_pooled(
        self,
        specify_pg: bool,
        test_compiled_with_noncompiled_ranks: bool,
        gradient_division: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_pooled,
            compile_config=_CompileConfig(
                test_compiled_with_noncompiled_ranks=test_compiled_with_noncompiled_ranks
            ),
            specify_pg=specify_pg,
            gradient_division=gradient_division,
        )

    @classmethod
    def _test_reduce_scatter_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        compile_config: _CompileConfig,
        specify_pg: bool,
        gradient_division: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        batch_size_per_rank = [4] * world_size
        B_global = sum(batch_size_per_rank)
        dim_sum_per_rank = [8] * world_size

        D_local_sum = dim_sum_per_rank[rank]

        inputs: List[torch.Tensor] = []
        for _ in range(world_size):
            input = torch.randn([B_global, D_local_sum], requires_grad=True).to(device)
            input.retain_grad()
            inputs.append(input)

        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.reduce_scatter_pooled(*args, **kwargs).wait()

        comm_ops.set_gradient_division(gradient_division)
        _test_async_sync_compile(
            fn,
            inputs,
            device,
            compile_config,
            rank,
            pg if specify_pg else None,
        )

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        specify_pg=st.sampled_from([True]),
        test_compiled_with_noncompiled_ranks=st.sampled_from([False, True]),
        gradient_division=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_reduce_scatter_pooled(
        self,
        specify_pg: bool,
        test_compiled_with_noncompiled_ranks: bool,
        gradient_division: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_reduce_scatter_pooled,
            compile_config=_CompileConfig(
                test_compiled_with_noncompiled_ranks=test_compiled_with_noncompiled_ranks
            ),
            specify_pg=specify_pg,
            gradient_division=gradient_division,
        )

    @classmethod
    def _test_reduce_scatter_v_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        compile_config: _CompileConfig,
        specify_pg: bool,
        gradient_division: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        src: List[int] = [1, 2, 3] * world_size
        input_splits: List[int] = src[:world_size]
        inputs_dim: int = sum(input_splits)

        input: torch.Tensor = torch.randn(inputs_dim, 2, requires_grad=True).to(device)

        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.reduce_scatter_v_pooled(*args, **kwargs).wait()

        comm_ops.set_gradient_division(gradient_division)
        _test_async_sync_compile(
            fn,
            input,
            device,
            compile_config,
            rank,
            input_splits,
            pg if specify_pg else None,
        )

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        specify_pg=st.sampled_from([True]),
        test_compiled_with_noncompiled_ranks=st.sampled_from([False, True]),
        gradient_division=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_reduce_scatter_v_pooled(
        self,
        specify_pg: bool,
        test_compiled_with_noncompiled_ranks: bool,
        gradient_division: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_reduce_scatter_v_pooled,
            compile_config=_CompileConfig(
                test_compiled_with_noncompiled_ranks=test_compiled_with_noncompiled_ranks
            ),
            specify_pg=specify_pg,
            gradient_division=gradient_division,
        )

    @classmethod
    def _test_reduce_scatter_v_per_feature_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        compile_config: _CompileConfig,
        specify_pg: bool,
        gradient_division: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        batch_size_per_feature: List[int] = [2, 4, 4, 7, 2]
        batch_size_per_rank_per_feature: List[List[int]] = []
        for _ in range(world_size):
            batch_size_per_rank_per_feature.append(batch_size_per_feature)

        embedding_dims: List[int] = [12] * len(batch_size_per_feature)

        n = world_size * sum(
            [b * emb_dim for b, emb_dim in zip(batch_size_per_feature, embedding_dims)]
        )
        input: torch.Tensor = torch.randn(n, requires_grad=True).to(device)

        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.reduce_scatter_v_per_feature_pooled(*args, **kwargs).wait()

        comm_ops.set_gradient_division(gradient_division)
        _test_async_sync_compile(
            fn,
            input,
            device,
            compile_config,
            rank,
            batch_size_per_rank_per_feature,
            embedding_dims,
            pg if specify_pg else None,
        )
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        specify_pg=st.sampled_from([True]),
        test_compiled_with_noncompiled_ranks=st.sampled_from([False, True]),
        gradient_division=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_reduce_scatter_v_per_feature_pooled(
        self,
        specify_pg: bool,
        test_compiled_with_noncompiled_ranks: bool,
        gradient_division: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_reduce_scatter_v_per_feature_pooled,
            compile_config=_CompileConfig(
                test_compiled_with_noncompiled_ranks=test_compiled_with_noncompiled_ranks
            ),
            specify_pg=specify_pg,
            gradient_division=gradient_division,
        )

    @classmethod
    def _test_all_gather_base_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        compile_config: _CompileConfig,
        specify_pg: bool,
        gradient_division: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        input = torch.randn([4, 4], requires_grad=True).to(device)

        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.all_gather_base_pooled(*args, **kwargs).wait()

        comm_ops.set_gradient_division(gradient_division)
        _test_async_sync_compile(
            fn, input, device, compile_config, rank, pg if specify_pg else None
        )

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        specify_pg=st.sampled_from([True]),
        test_compiled_with_noncompiled_ranks=st.sampled_from([False, True]),
        gradient_division=st.sampled_from([True, False]),
    )
    @settings(deadline=None)
    def test_all_gather_base_pooled(
        self,
        specify_pg: bool,
        test_compiled_with_noncompiled_ranks: bool,
        gradient_division: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_all_gather_base_pooled,
            compile_config=_CompileConfig(
                test_compiled_with_noncompiled_ranks=test_compiled_with_noncompiled_ranks
            ),
            specify_pg=specify_pg,
            gradient_division=gradient_division,
        )

    @classmethod
    def _test_all_gather_base_pooled_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cpu")
        input_tensor = torch.randn([4, 4], requires_grad=True).to(device)
        comm_ops.all_gather_base_pooled(input_tensor, pg).wait()
        dist.destroy_process_group()

    def test_all_gather_base_pooled_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_all_gather_base_pooled_cpu,
        )
