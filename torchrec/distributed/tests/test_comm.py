#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import importlib.util
import itertools
import math
import multiprocessing
import os
import unittest
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import mock
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import torch
import torch.distributed as dist
import torchrec
import torchrec.distributed.comm_ops as comm_ops
from fbgemm_gpu.split_embedding_configs import SparseType
from hypothesis import given, settings
from torch import Tensor
from torch.distributed.distributed_c10d import GroupMember
from torch.testing._internal.distributed.fake_pg import _create_fake_pg, FakeStore
from torchrec.distributed import comm
from torchrec.distributed.comm import get_2d_pod_size, get_resolved_pod_size
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)
from torchrec.distributed.types import QuantizedCommCodecs
from torchrec.distributed.utils import none_throws
from torchrec.test_utils import get_free_port, seed_and_log

torch.ops.import_module("fbgemm_gpu.sparse_ops")

# Exit code of the Triton interpreter check when the interpreter is missing.
_TRITON_INTERPRETER_UNAVAILABLE = 77

# Default-on sequence all2all paths. Each test pins them off and turns on only
# the paths it exercises, so its reference run is the original path.
_SEQ_A2A_DEFAULT_ON_FLAGS: Tuple[str, ...] = (
    "TORCHREC_SEQ_A2A_MX4_V2",
    "TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_FWD",
    "TORCHREC_SEQ_A2A_FUSED_ROWMAP_FWD",
    "TORCHREC_SEQ_A2A_FUSED_ROWMAP_BWD",
    "TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE",
    "TORCHREC_SEQ_A2A_UNBUCKETIZE_ROWMAP",
    "TORCHREC_SEQ_UNBUCKETIZE_COMPOSE",
    "TORCHREC_UNBUCKETIZE_GATHER_BWD",
    "TORCHREC_PIPELINED_FWD_EVENT_REUSE",
    "TORCHREC_PIPELINED_FWD_EVENT_REUSE_CHECK",
)


_HAS_TRITON: bool = importlib.util.find_spec("triton") is not None


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

    sync_bwd_out: Optional[torch.Tensor] = None
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

                assert sync_bwd_out is not None
                _assert_close(compile_bwd_out, sync_bwd_out)


class TestAllToAll(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        flags_off = mock.patch.dict(
            os.environ, dict.fromkeys(_SEQ_A2A_DEFAULT_ON_FLAGS, "0")
        )
        flags_off.start()
        self.addCleanup(flags_off.stop)
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
    def _test_alltoall_pooled_fp8_bwd_padding(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        batch_size_per_rank: List[int],
        dim_sum_per_rank: List[int],
        variable_batch: bool,
        gradient_division: bool,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        comm_ops.set_gradient_division(gradient_division)

        torch.manual_seed(rank)
        pooled_embs = torch.randn(
            sum(batch_size_per_rank), dim_sum_per_rank[rank], device=device
        )
        grad_out = torch.randn(
            batch_size_per_rank[rank], sum(dim_sum_per_rank), device=device
        )

        def bwd_grad(codecs: Optional[QuantizedCommCodecs], padding: bool) -> Tensor:
            os.environ["TORCHREC_ENABLE_FP8_ROWWISE_PADDING"] = str(int(padding))
            embs = pooled_embs.clone().requires_grad_()
            if variable_batch:
                # One feature per rank, each with every rank's batch size.
                out = comm_ops.variable_batch_alltoall_pooled(
                    embs.view(-1),
                    batch_size_per_rank_per_feature=[[b] for b in batch_size_per_rank],
                    batch_size_per_feature_pre_a2a=[batch_size_per_rank[rank]]
                    * world_size,
                    emb_dim_per_rank_per_feature=[[d] for d in dim_sum_per_rank],
                    group=pg,
                    codecs=codecs,
                ).wait()
                out.backward(grad_out.view(-1))
            else:
                comm_ops.alltoall_pooled(
                    embs, batch_size_per_rank, dim_sum_per_rank, group=pg, codecs=codecs
                ).wait().backward(grad_out)
            return none_throws(embs.grad)

        fp8 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP8,
                backward_precision=CommType.FP8,
                fp8_quantize_dim=256,
                fp8_quantize_dim_bwd=256,
            )
        )
        padded = bwd_grad(fp8, padding=True)
        torch.testing.assert_close(
            padded, bwd_grad(None, padding=False), rtol=0.1, atol=0.1
        )
        if all(b * d % 256 == 0 for b in batch_size_per_rank for d in dim_sum_per_rank):
            # Backward grads don't depend on forward values, so forward padding
            # can't affect this comparison.
            torch.testing.assert_close(
                padded, bwd_grad(fp8, padding=False), rtol=0, atol=0
            )
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    @given(
        variable_batch=st.sampled_from([False, True]),
        gradient_division=st.sampled_from([False, True]),
    )
    @settings(deadline=None)
    def test_alltoall_pooled_fp8_bwd_padding(
        self, variable_batch: bool, gradient_division: bool
    ) -> None:
        for batch_size_per_rank, dim_sum_per_rank in [
            # Every B_r * D_s split unaligned to 256.
            ([3, 5], [36, 72]),
            # Rank 0 sends aligned splits but receives a padded one.
            ([4, 3], [64, 64]),
            # Aligned splits: the comm is unchanged with padding on.
            ([4, 4], [128, 256]),
        ]:
            self._run_multi_process_test(
                world_size=self.WORLD_SIZE,
                backend="nccl",
                # pyrefly: ignore[bad-argument-type]
                callable=self._test_alltoall_pooled_fp8_bwd_padding,
                batch_size_per_rank=batch_size_per_rank,
                dim_sum_per_rank=dim_sum_per_rank,
                variable_batch=variable_batch,
                gradient_division=gradient_division,
            )

    @classmethod
    def _test_variable_batch_alltoall_pooled_fp8_fwd_padding(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        batch_size_per_rank: List[int],
        dim_sum_per_rank: List[int],
        use_sync: bool,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.distributed_c10d._get_default_group()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        comm_ops.set_use_sync_collectives(use_sync)

        torch.manual_seed(rank)
        pooled_embs = torch.randn(
            sum(batch_size_per_rank), dim_sum_per_rank[rank], device=device
        )

        def fwd(codecs: Optional[QuantizedCommCodecs], padding: bool) -> Tensor:
            os.environ["TORCHREC_ENABLE_FP8_ROWWISE_PADDING"] = str(int(padding))
            # One feature per rank, each with every rank's batch size.
            return comm_ops.variable_batch_alltoall_pooled(
                pooled_embs.view(-1),
                batch_size_per_rank_per_feature=[[b] for b in batch_size_per_rank],
                batch_size_per_feature_pre_a2a=[batch_size_per_rank[rank]] * world_size,
                emb_dim_per_rank_per_feature=[[d] for d in dim_sum_per_rank],
                group=pg,
                codecs=codecs,
            ).wait()

        fp8 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP8,
                backward_precision=CommType.FP8,
                fp8_quantize_dim=256,
                fp8_quantize_dim_bwd=256,
            )
        )
        padded = fwd(fp8, padding=True)
        torch.testing.assert_close(padded, fwd(None, padding=False), rtol=0.1, atol=0.1)
        if all(b * d % 256 == 0 for b in batch_size_per_rank for d in dim_sum_per_rank):
            torch.testing.assert_close(padded, fwd(fp8, padding=False), rtol=0, atol=0)
        comm_ops.set_use_sync_collectives(False)
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    def test_variable_batch_alltoall_pooled_fp8_fwd_padding(self) -> None:
        for use_sync, (batch_size_per_rank, dim_sum_per_rank) in itertools.product(
            [False, True],
            [
                # Every B_r * D_s split unaligned to 256.
                ([3, 5], [36, 72]),
                # Rank 0 sends aligned splits but receives a padded one.
                ([4, 3], [64, 64]),
                # Aligned splits: the comm is unchanged with padding on.
                ([4, 4], [128, 256]),
            ],
        ):
            self._run_multi_process_test(
                world_size=self.WORLD_SIZE,
                backend="nccl",
                # pyrefly: ignore[bad-argument-type]
                callable=self._test_variable_batch_alltoall_pooled_fp8_fwd_padding,
                batch_size_per_rank=batch_size_per_rank,
                dim_sum_per_rank=dim_sum_per_rank,
                use_sync=use_sync,
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

    @classmethod
    def _test_alltoall_sequence_permute_quantized_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        # The shared-SID CA job's sequence qcomms: MX4 with 8-value groups
        # forward (320-byte rows at D=512) and 256-value groups backward.
        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        D, B_l = 512, 3
        tables_mp = [[0], [1, 2]]
        local_T = len(tables_mp[rank])
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(0, 6, (3, world_size * B_l), generator=gen).int()
        lengths_mp = lengths[tables_mp[rank]]
        input_splits = [
            int(lengths[tables_mp[j]][:, rank * B_l : (rank + 1) * B_l].sum())
            for j in range(world_size)
        ]
        output_splits = [
            int(lengths_mp[:, j * B_l : (j + 1) * B_l].sum()) for j in range(world_size)
        ]
        forward_recat = torch.IntTensor(
            [j + i * world_size for j in range(world_size) for i in range(local_T)]
        )
        backward_recat = torch.IntTensor(
            [i + j * local_T for i in range(local_T) for j in range(world_size)]
        )
        gen.manual_seed(1 + rank)
        rows = int(lengths_mp.sum())
        embs = torch.randn(rows, D, generator=gen) * torch.pow(
            10.0, torch.randint(-8, 8, (rows, 1), generator=gen).float()
        )
        grad = torch.randn(sum(input_splits), D, generator=gen)

        def run(env: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
            for flag in ("FWD", "CHECK"):
                name = f"TORCHREC_SEQ_A2A_PERMUTE_QUANTIZED_{flag}"
                os.environ[name] = env.get(flag, "0")
            x = embs.clone().requires_grad_(True)
            out = comm_ops.alltoall_sequence(
                a2a_sequence_embs_tensor=x,
                forward_recat_tensor=forward_recat,
                backward_recat_tensor=backward_recat,
                lengths_after_sparse_data_all2all=lengths_mp,
                input_splits=input_splits,
                output_splits=output_splits,
                group=pg,
                codecs=codecs,
            ).wait()
            out.backward(grad)
            grad_in = x.grad
            assert grad_in is not None
            return out.detach(), grad_in

        # The payload permute must reproduce the row permute bit for bit.
        out_ref, grad_ref = run({})
        for env in ({"FWD": "1", "CHECK": "1"}, {"FWD": "1"}):
            out, grad_in = run(env)
            assert torch.equal(out, out_ref) and torch.equal(grad_in, grad_ref)
        assert (D, 8, 40) in comm_ops._seq_a2a_permute_q_logged
        assert comm_ops._seq_a2a_permute_q_checks == 1
        assert not comm_ops._seq_a2a_permute_q_disabled

        # A position-dependent encode does not commute with the permute; the
        # self-check must catch it and fall back to the old path's result.
        encode = codecs.forward.encode

        def bad_encode(t: torch.Tensor, ctx: Optional[object] = None) -> torch.Tensor:
            q = encode(t, ctx).clone()
            q[::7] ^= 1
            return q

        comm_ops._seq_a2a_permute_q_checks = 0
        with mock.patch.object(codecs.forward, "encode", bad_encode):
            out_ref, grad_ref = run({})
            out, grad_in = run({"FWD": "1", "CHECK": "1"})
        assert torch.equal(out, out_ref) and torch.equal(grad_in, grad_ref)
        # Rank 0's forward recat is the identity, so only rank 1 can mismatch.
        assert comm_ops._seq_a2a_permute_q_disabled == (rank == 1)
        dist.destroy_process_group()

    def test_alltoall_sequence_permute_quantized_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence_permute_quantized_cpu,
        )

    @classmethod
    def _test_alltoall_sequence_rowmap_bwd_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        D, B_l = 512, 3
        tables_mp = [[0], [1, 2]]
        local_T = len(tables_mp[rank])
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(0, 6, (3, world_size * B_l), generator=gen).int()
        lengths_mp = lengths[tables_mp[rank]]
        input_splits = [
            int(lengths[tables_mp[j]][:, rank * B_l : (rank + 1) * B_l].sum())
            for j in range(world_size)
        ]
        output_splits = [
            int(lengths_mp[:, j * B_l : (j + 1) * B_l].sum()) for j in range(world_size)
        ]
        forward_recat = torch.IntTensor(
            [j + i * world_size for j in range(world_size) for i in range(local_T)]
        )
        backward_recat = torch.IntTensor(
            [i + j * local_T for i in range(local_T) for j in range(world_size)]
        )
        gen.manual_seed(1 + rank)
        embs = torch.randn(int(lengths_mp.sum()), D, generator=gen)
        grad_rows = sum(input_splits)
        grad = torch.randn(grad_rows, D, generator=gen) * torch.pow(
            10.0, torch.randint(-8, 8, (grad_rows, 1), generator=gen).float()
        )
        rowmap = comm_ops._seq_a2a_bwd_dequant_rowmap
        taken: List[bool] = []

        def spy(*args: object) -> Optional[torch.Tensor]:
            # pyrefly: ignore[bad-argument-type]
            ret = rowmap(*args)
            taken.append(ret is not None)
            return ret

        def run(env: Dict[str, str]) -> torch.Tensor:
            for flag in (
                "PERMUTE_QUANTIZED_FWD",
                "FUSED_ROWMAP_BWD",
                "FUSED_ROWMAP_CHECK",
            ):
                os.environ[f"TORCHREC_SEQ_A2A_{flag}"] = env.get(flag, "0")
            x = embs.clone().requires_grad_(True)
            out = comm_ops.alltoall_sequence(
                a2a_sequence_embs_tensor=x,
                forward_recat_tensor=forward_recat,
                backward_recat_tensor=backward_recat,
                lengths_after_sparse_data_all2all=lengths_mp,
                input_splits=input_splits,
                output_splits=output_splits,
                group=pg,
                codecs=codecs,
            ).wait()
            out.backward(grad)
            grad_in = x.grad
            assert grad_in is not None
            return grad_in

        # Dequantizing through the row map must reproduce decode + permute bit
        # for bit, alone and behind the quantized forward permute.
        with mock.patch.object(comm_ops, "_seq_a2a_bwd_dequant_rowmap", spy):
            grad_ref = run({})
            for env in (
                {"FUSED_ROWMAP_BWD": "1", "FUSED_ROWMAP_CHECK": "1"},
                {"FUSED_ROWMAP_BWD": "1"},
                {"PERMUTE_QUANTIZED_FWD": "1", "FUSED_ROWMAP_BWD": "1"},
            ):
                assert torch.equal(run(env), grad_ref)
        assert taken == [True, True, True]
        assert (D, 256, 258) in comm_ops._seq_a2a_rowmap_logged
        assert comm_ops._seq_a2a_rowmap_checks == 1
        assert not comm_ops._seq_a2a_rowmap_disabled

        # A wrong row map must fail the self-check, which then returns the old
        # path's result and disables the row map for the rest of the process.
        permute = comm_ops._permute_2D_sparse_data

        def bad_permute(
            *args: object,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            # pyrefly: ignore[bad-argument-type]
            out_lengths, values, weights = permute(*args)
            if values.dtype == torch.int64:
                values = values.clone()
                values[[0, 1]] = values[[1, 0]]
            return out_lengths, values, weights

        comm_ops._seq_a2a_rowmap_checks = 0
        with mock.patch.object(comm_ops, "_permute_2D_sparse_data", bad_permute):
            grad_in = run({"FUSED_ROWMAP_BWD": "1", "FUSED_ROWMAP_CHECK": "1"})
        assert torch.equal(grad_in, grad_ref)
        assert comm_ops._seq_a2a_rowmap_disabled
        grad_in = run({"FUSED_ROWMAP_BWD": "1"})
        assert torch.equal(grad_in, grad_ref)
        dist.destroy_process_group()

    def test_alltoall_sequence_rowmap_bwd_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence_rowmap_bwd_cpu,
        )

    def test_seq_a2a_bwd_dequant_rowmap_cpu(self) -> None:
        D, T, W, B = 512, 2, 2, 3
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(0, 4, (T * W, B), generator=gen).int()
        recat = torch.IntTensor([i + j * T for i in range(T) for j in range(W)])
        rows = int(lengths.sum())
        rowmap = comm_ops._seq_a2a_bwd_dequant_rowmap

        def info(codecs: QuantizedCommCodecs) -> comm_ops.All2AllSequenceInfo:
            return comm_ops.All2AllSequenceInfo(
                embedding_dim=D,
                lengths_after_sparse_data_all2all=lengths,
                forward_recat_tensor=None,
                backward_recat_tensor=recat,
                input_splits=[],
                output_splits=[],
                codecs=codecs,
                permuted_lengths_after_sparse_data_all2all=lengths * D,
            )

        for output_dtype in (None, SparseType.BF16):
            codecs = get_qcomm_codecs(
                QCommsConfig(
                    forward_precision=CommType.MX4,
                    backward_precision=CommType.MX4,
                    mx4_quantize_dim=8,
                    mx4_quantize_dim_bwd=256,
                    output_dtype=output_dtype,
                )
            )
            ctx = codecs.backward.create_context()
            payload = codecs.backward.encode(
                torch.randn(rows, D, generator=gen).view(-1), ctx
            )
            decoded = codecs.backward.decode(payload, ctx)
            _, ref, _ = comm_ops._permute_2D_sparse_data(
                recat, lengths * D, decoded, None, decoded.numel()
            )
            out = rowmap(info(codecs), payload, ctx)
            assert out is not None and out.dtype == ref.dtype
            assert torch.equal(out, ref)
        assert (D, 256, 258) in comm_ops._seq_a2a_rowmap_logged

        # Everything outside the exact, supported case takes the old path.
        a2ai = info(codecs)
        assert rowmap(a2ai, payload, ctx) is not None
        a2ai.variable_batch_size = True
        assert rowmap(a2ai, payload, ctx) is None
        a2ai = info(codecs)
        a2ai.permuted_lengths_after_sparse_data_all2all = None
        assert rowmap(a2ai, payload, ctx) is None
        assert rowmap(info(codecs), payload[:-1], ctx) is None
        assert rowmap(info(codecs), payload.view(torch.int8), ctx) is None
        with mock.patch.object(codecs.backward, "_loss_scale", 2.0):
            assert rowmap(info(codecs), payload, ctx) is None
        with mock.patch.object(comm_ops, "_seq_a2a_rowmap_disabled", True):
            assert rowmap(info(codecs), payload, ctx) is None
        fp16 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP16, backward_precision=CommType.FP16
            )
        )
        assert rowmap(info(fp16), payload, ctx) is None
        for dim, group_size in ((384, 96), (512, 1024)):
            a2ai = info(codecs)
            a2ai.embedding_dim = dim
            with mock.patch.object(ctx, "mx_group_size", group_size):
                assert rowmap(a2ai, payload, ctx) is None
            assert (dim, group_size, 0) in comm_ops._seq_a2a_rowmap_logged

    @classmethod
    def _check_mx4_rowmap_dequant(cls, device: torch.device) -> None:
        from fbgemm_gpu.triton.quantize import triton_dequantize_mx4
        from torchrec.distributed.mx4_rowmap_dequant import dequantize_mx4_rowmap

        gen = torch.Generator().manual_seed(0)
        for (D, g), out_dtype in itertools.product(
            ((512, 256), (512, 8), (128, 32)), (torch.float32, torch.bfloat16)
        ):
            T, W, B = 3, 2, 5
            lengths = torch.randint(0, 4, (T * W, B), generator=gen).int()
            lengths[0] = 0
            recat = torch.randperm(T * W, generator=gen).int()
            rows = int(lengths.sum())
            # Arbitrary packed bytes, edge exponent bytes (subnormal scale, inf
            # scale and NaN products) and an all-zero group.
            groups = torch.randint(
                0, 256, (rows * (D // g), g // 2 + 1), generator=gen
            ).to(torch.uint8)
            groups[:6, -1] = torch.tensor([0, 1, 127, 128, 254, 255])
            groups[6] = 0
            payload = groups.view(-1).to(device)
            lengths, recat = lengths.to(device), recat.to(device)
            decoded = triton_dequantize_mx4(payload, g, output_dtype=out_dtype)
            _, ref, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                recat, lengths * D, decoded, None, decoded.numel()
            )
            _, row_idx, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                recat, lengths, torch.arange(rows, device=device), None, rows
            )
            out = dequantize_mx4_rowmap(payload, row_idx, D, g, out_dtype)
            bits = torch.int16 if out_dtype == torch.bfloat16 else torch.int32
            assert torch.equal(out.view(bits), ref.view(bits)), (D, g, out_dtype)

    @unittest.skipIf(not torch.cuda.is_available(), "Needs a GPU for Triton")
    def test_mx4_rowmap_dequant_gpu(self) -> None:
        self._check_mx4_rowmap_dequant(torch.device("cuda"))

    @classmethod
    def _test_alltoall_sequence_rowmap_fwd_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        D, B_l = 512, 3
        tables_mp = [[0], [1, 2]]
        local_T = len(tables_mp[rank])
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(1, 6, (3, world_size * B_l), generator=gen).int()
        lengths_mp = lengths[tables_mp[rank]]
        input_splits = [
            int(lengths[tables_mp[j]][:, rank * B_l : (rank + 1) * B_l].sum())
            for j in range(world_size)
        ]
        output_splits = [
            int(lengths_mp[:, j * B_l : (j + 1) * B_l].sum()) for j in range(world_size)
        ]
        forward_recat = torch.IntTensor(
            [j + i * world_size for j in range(world_size) for i in range(local_T)]
        )
        backward_recat = torch.IntTensor(
            [i + j * local_T for i in range(local_T) for j in range(world_size)]
        )
        gen.manual_seed(1 + rank)
        rows = int(lengths_mp.sum())
        embs = torch.randn(rows, D, generator=gen) * torch.pow(
            10.0, torch.randint(-8, 8, (rows, 1), generator=gen).float()
        )
        grad = torch.randn(sum(input_splits), D, generator=gen)
        rowmap = comm_ops._seq_a2a_fwd_quantize_rowmap
        permute_q = comm_ops._seq_a2a_fwd_permute_quantized
        taken: List[bool] = []
        permute_q_calls: List[bool] = []

        def spy(*args: object) -> Optional[Tuple[torch.Tensor, torch.Tensor, object]]:
            # pyrefly: ignore[bad-argument-type]
            ret = rowmap(*args)
            taken.append(ret is not None)
            return ret

        def spy_permute_q(
            *args: object,
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor, object]]:
            permute_q_calls.append(True)
            # pyrefly: ignore[bad-argument-type]
            return permute_q(*args)

        def run(env: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
            for flag in (
                "PERMUTE_QUANTIZED_FWD",
                "FUSED_ROWMAP_FWD",
                "FUSED_ROWMAP_FWD_CHECK",
                "FUSED_ROWMAP_BWD",
            ):
                os.environ[f"TORCHREC_SEQ_A2A_{flag}"] = env.get(flag, "0")
            x = embs.clone().requires_grad_(True)
            out = comm_ops.alltoall_sequence(
                a2a_sequence_embs_tensor=x,
                forward_recat_tensor=forward_recat,
                backward_recat_tensor=backward_recat,
                lengths_after_sparse_data_all2all=lengths_mp,
                input_splits=input_splits,
                output_splits=output_splits,
                group=pg,
                codecs=codecs,
            ).wait()
            out.backward(grad)
            grad_in = x.grad
            assert grad_in is not None
            return out.detach(), grad_in

        # Quantizing through the row map must reproduce permute + quantize bit
        # for bit, alone, ahead of the quantized-payload permute flag (which
        # then does not run) and with the backward row map.
        with mock.patch.object(
            comm_ops, "_seq_a2a_fwd_quantize_rowmap", spy
        ), mock.patch.object(comm_ops, "_seq_a2a_fwd_permute_quantized", spy_permute_q):
            out_ref, grad_ref = run({})
            for env in (
                {"FUSED_ROWMAP_FWD": "1", "FUSED_ROWMAP_FWD_CHECK": "1"},
                {"FUSED_ROWMAP_FWD": "1"},
                {
                    "FUSED_ROWMAP_FWD": "1",
                    "PERMUTE_QUANTIZED_FWD": "1",
                    "FUSED_ROWMAP_BWD": "1",
                },
            ):
                out, grad_in = run(env)
                assert torch.equal(out, out_ref) and torch.equal(grad_in, grad_ref)
        assert taken == [True, True, True]
        assert permute_q_calls == []
        assert (D, 8, 320) in comm_ops._seq_a2a_rowmap_fwd_logged
        assert comm_ops._seq_a2a_rowmap_fwd_checks == 1
        assert not comm_ops._seq_a2a_rowmap_fwd_disabled

        # A wrong row map must fail the self-check, which then returns the old
        # path's payload and disables the row map for the rest of the process.
        permute = comm_ops._permute_2D_sparse_data

        def bad_permute(
            *args: object,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            # pyrefly: ignore[bad-argument-type]
            out_lengths, values, weights = permute(*args)
            if values.dtype == torch.int64:
                values = values.clone()
                values[[0, 1]] = values[[1, 0]]
            return out_lengths, values, weights

        comm_ops._seq_a2a_rowmap_fwd_checks = 0
        with mock.patch.object(comm_ops, "_permute_2D_sparse_data", bad_permute):
            out, grad_in = run({"FUSED_ROWMAP_FWD": "1", "FUSED_ROWMAP_FWD_CHECK": "1"})
        assert torch.equal(out, out_ref) and torch.equal(grad_in, grad_ref)
        assert comm_ops._seq_a2a_rowmap_fwd_disabled
        out, grad_in = run({"FUSED_ROWMAP_FWD": "1"})
        assert torch.equal(out, out_ref) and torch.equal(grad_in, grad_ref)
        dist.destroy_process_group()

    def test_alltoall_sequence_rowmap_fwd_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence_rowmap_fwd_cpu,
        )

    def test_seq_a2a_fwd_quantize_rowmap_cpu(self) -> None:
        D, T, W, B = 512, 2, 2, 3
        gen = torch.Generator().manual_seed(0)
        # [local_T, W * B] as All2All_Seq_Req.forward gets them.
        lengths = torch.randint(0, 4, (T, W * B), generator=gen).int()
        recat = torch.IntTensor([j + i * W for j in range(W) for i in range(T)])
        rows = int(lengths.sum())
        rowmap = comm_ops._seq_a2a_fwd_quantize_rowmap

        def info(
            codecs: Optional[QuantizedCommCodecs], dim: int = D
        ) -> comm_ops.All2AllSequenceInfo:
            return comm_ops.All2AllSequenceInfo(
                embedding_dim=dim,
                lengths_after_sparse_data_all2all=lengths,
                forward_recat_tensor=recat,
                # pyrefly: ignore [bad-argument-type]
                backward_recat_tensor=None,
                input_splits=[],
                output_splits=[],
                codecs=codecs,
            )

        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        # fbgemm's CPU MX4 encode takes fp32 only; bf16 is covered on GPU.
        x = torch.randn(rows, D, generator=gen)
        ref_lengths, ref_rows, _ = comm_ops._permute_2D_sparse_data(
            recat, (lengths * D).view(T * W, B), x.view(-1), None, x.numel()
        )
        ref = codecs.forward.encode(ref_rows, codecs.forward.create_context())
        ret = rowmap(info(codecs), x, lengths * D, W)
        assert ret is not None
        out_lengths, payload, ctx = ret
        assert torch.equal(out_lengths, ref_lengths)
        assert payload.dtype == torch.uint8 and torch.equal(payload, ref)
        assert ctx.mx_group_size == 8
        assert (D, 8, 320) in comm_ops._seq_a2a_rowmap_fwd_logged

        # Everything outside the exact, supported case takes the old path.
        assert rowmap(info(codecs), x, lengths * D, W) is not None
        a2ai = info(codecs)
        a2ai.variable_batch_size = True
        assert rowmap(a2ai, x, lengths * D, W) is None
        a2ai = info(codecs)
        a2ai.forward_recat_tensor = None
        assert rowmap(a2ai, x, lengths * D, W) is None
        assert rowmap(info(None), x, lengths * D, W) is None
        assert rowmap(info(codecs), x[:0], lengths * 0, W) is None
        assert rowmap(info(codecs), x.half(), lengths * D, W) is None
        x_t = torch.randn(D, rows, generator=gen).t()
        assert rowmap(info(codecs), x_t, lengths * D, W) is None
        with mock.patch.object(codecs.forward, "_loss_scale", 2.0):
            assert rowmap(info(codecs), x, lengths * D, W) is None
        with mock.patch.object(comm_ops, "_seq_a2a_rowmap_fwd_disabled", True):
            assert rowmap(info(codecs), x, lengths * D, W) is None
        fp16 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP16, backward_precision=CommType.FP16
            )
        )
        assert rowmap(info(fp16), x, lengths * D, W) is None
        # 3 is RoundingMode.stochastic.
        with mock.patch.object(codecs.forward, "_rounding_mode", 3):
            assert rowmap(info(codecs), x, lengths * D, W) is None
        for dim, group_size in ((768, 12), (384, 96), (512, 1024), (256, 8)):
            with mock.patch.object(codecs.forward, "_row_dim", group_size):
                assert rowmap(info(codecs, dim), x, lengths * dim, W) is None
            assert (dim, group_size, 0) in comm_ops._seq_a2a_rowmap_fwd_logged

    @classmethod
    def _check_mx4_rowmap_quantize(cls, device: torch.device) -> None:
        from fbgemm_gpu.triton.quantize import triton_quantize_mx4
        from torchrec.distributed.mx4_rowmap_quantize import quantize_mx4_rowmap

        gen = torch.Generator().manual_seed(0)
        for (D, g), dtype, mode in itertools.product(
            ((512, 8), (1024, 8), (256, 4), (2048, 32)),
            (torch.bfloat16, torch.float32),
            (2, 0, 1, 4),
        ):
            T, W, B = 3, 2, 5
            lengths = torch.randint(1, 4, (T * W, B), generator=gen).int()
            lengths[0] = 0
            recat = torch.randperm(T * W, generator=gen).int()
            rows = int(lengths.sum())
            x = torch.randn(rows, D, generator=gen) * torch.pow(
                2.0, torch.randint(-60, 60, (rows, 1), generator=gen).float()
            )
            # Edge groups: all zero, infinite, near the fp32 max, subnormal,
            # with a NaN, and tiny values next to a large one.
            groups = x.view(-1, g)
            groups[0] = 0.0
            groups[1, 0] = float("inf")
            groups[2] = 3.0e38
            groups[3] = 1e-40
            groups[4, 1] = float("nan")
            groups[5] = 1e-6
            groups[5, 0] = 3.0
            x = x.to(dtype).to(device)
            lengths, recat = lengths.to(device), recat.to(device)
            _, permuted, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                recat, lengths * D, x.view(-1), None, x.numel()
            )
            ref = triton_quantize_mx4(permuted, g, rounding_mode=mode)
            _, row_idx, _ = torch.ops.fbgemm.permute_2D_sparse_data(
                recat, lengths, torch.arange(rows, device=device), None, rows
            )
            out = quantize_mx4_rowmap(x, row_idx, D, g, mode)
            assert torch.equal(out, ref.view(-1)), (D, g, dtype, mode)

    @unittest.skipIf(not torch.cuda.is_available(), "Needs a GPU for Triton")
    def test_mx4_rowmap_quantize_gpu(self) -> None:
        self._check_mx4_rowmap_quantize(torch.device("cuda"))

    @classmethod
    def _test_alltoall_sequence_bwd_chunked_encode_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        D, B_l = 512, 4
        tables_mp = [[0], [1, 2]]
        local_T = len(tables_mp[rank])
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(3, 8, (3, world_size * B_l), generator=gen).int()
        lengths_mp = lengths[tables_mp[rank]]
        input_splits = [
            int(lengths[tables_mp[j]][:, rank * B_l : (rank + 1) * B_l].sum())
            for j in range(world_size)
        ]
        output_splits = [
            int(lengths_mp[:, j * B_l : (j + 1) * B_l].sum()) for j in range(world_size)
        ]
        forward_recat = torch.IntTensor(
            [j + i * world_size for j in range(world_size) for i in range(local_T)]
        )
        backward_recat = torch.IntTensor(
            [i + j * local_T for i in range(local_T) for j in range(world_size)]
        )
        gen.manual_seed(1 + rank)
        embs = torch.randn(int(lengths_mp.sum()), D, generator=gen)
        grad_rows = sum(input_splits)
        grad = torch.randn(grad_rows, D, generator=gen) * torch.pow(
            10.0, torch.randint(-8, 8, (grad_rows, 1), generator=gen).float()
        )
        chunked = comm_ops._seq_a2a_bwd_chunked_encode
        taken: List[bool] = []

        def spy(*args: object) -> Optional[torch.Tensor]:
            # pyrefly: ignore[bad-argument-type]
            ret = chunked(*args)
            taken.append(ret is not None)
            return ret

        def run(env: Dict[str, str]) -> torch.Tensor:
            for flag in ("BWD_CHUNKED_ENCODE", "BWD_CHUNKED_ENCODE_CHECK"):
                os.environ[f"TORCHREC_SEQ_A2A_{flag}"] = env.get(flag, "0")
            x = embs.clone().requires_grad_(True)
            out = comm_ops.alltoall_sequence(
                a2a_sequence_embs_tensor=x,
                forward_recat_tensor=forward_recat,
                backward_recat_tensor=backward_recat,
                lengths_after_sparse_data_all2all=lengths_mp,
                input_splits=input_splits,
                output_splits=output_splits,
                group=pg,
                codecs=codecs,
            ).wait()
            out.backward(grad)
            grad_in = x.grad
            assert grad_in is not None
            return grad_in

        # 16-row chunks of the fp32 gradient: each rank's 36+ rows take three
        # or more launches. The chunked encode must reproduce the codec's
        # encode bit for bit.
        num_chunks = math.ceil(grad_rows / 16)
        assert num_chunks >= 3
        with mock.patch.object(
            mx4q, "MX4_BUFFER_OPS_MAX_BYTES", 16 * D * 4
        ), mock.patch.object(comm_ops, "_seq_a2a_bwd_chunked_encode", spy):
            grad_ref = run({})
            for env in (
                {"BWD_CHUNKED_ENCODE": "1", "BWD_CHUNKED_ENCODE_CHECK": "1"},
                {"BWD_CHUNKED_ENCODE": "1"},
            ):
                assert torch.equal(run(env), grad_ref)
            assert taken == [True, True]
            assert (D, 256, num_chunks) in comm_ops._seq_a2a_bwd_chunked_logged
            assert comm_ops._seq_a2a_bwd_chunked_checks == {num_chunks: 1}
            assert not comm_ops._seq_a2a_bwd_chunked_disabled

        # The all-to-all sends the path's payload: an all-zero one changes
        # every rank's gradient.
        def zero_payload(*args: object) -> Optional[torch.Tensor]:
            # pyrefly: ignore[bad-argument-type]
            ret = chunked(*args)
            assert ret is not None
            return torch.zeros_like(ret)

        with mock.patch.object(comm_ops, "_seq_a2a_bwd_chunked_encode", zero_payload):
            assert not torch.equal(run({"BWD_CHUNKED_ENCODE": "1"}), grad_ref)

        with mock.patch.object(mx4q, "MX4_BUFFER_OPS_MAX_BYTES", 16 * D * 4):
            # An encode whose output depends on where its input starts does not
            # commute with the row chunking; the self-check must catch it and
            # fall back to the old path's payload for good.
            encode = codecs.backward.encode

            def bad_encode(
                t: torch.Tensor, ctx: Optional[object] = None
            ) -> torch.Tensor:
                q = encode(t, ctx).clone()
                q.view(-1)[0] ^= 1
                return q

            comm_ops._seq_a2a_bwd_chunked_checks.clear()
            with mock.patch.object(codecs.backward, "encode", bad_encode):
                bad_ref = run({})
                env = {"BWD_CHUNKED_ENCODE": "1", "BWD_CHUNKED_ENCODE_CHECK": "1"}
                assert torch.equal(run(env), bad_ref)
                assert comm_ops._seq_a2a_bwd_chunked_disabled
                assert torch.equal(run({"BWD_CHUNKED_ENCODE": "1"}), bad_ref)
        dist.destroy_process_group()

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_alltoall_sequence_bwd_chunked_encode_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence_bwd_chunked_encode_cpu,
        )

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_seq_a2a_bwd_chunked_encode_cpu(self) -> None:
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        D = 512
        chunked = comm_ops._seq_a2a_bwd_chunked_encode

        def info(
            codecs: Optional[QuantizedCommCodecs], dim: int = D
        ) -> comm_ops.All2AllSequenceInfo:
            return comm_ops.All2AllSequenceInfo(
                embedding_dim=dim,
                lengths_after_sparse_data_all2all=torch.zeros(1, 1, dtype=torch.int),
                forward_recat_tensor=None,
                # pyrefly: ignore [bad-argument-type]
                backward_recat_tensor=None,
                input_splits=[],
                output_splits=[],
                codecs=codecs,
            )

        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        gen = torch.Generator().manual_seed(0)
        # fbgemm's CPU MX4 encode takes fp32 only; bf16 is covered on GPU.
        rows = 37
        grad = torch.randn(rows, D, generator=gen) * torch.pow(
            10.0, torch.randint(-8, 8, (rows, 1), generator=gen).float()
        )
        grad[3, :256] = 0.0
        grad[4, 0] = float("inf")
        grad[20, 7] = float("nan")
        # One launch, and 16-row launches over 16 + 16 + 5 rows.
        for group_size, max_bytes, num_chunks in (
            (256, mx4q.MX4_BUFFER_OPS_MAX_BYTES, 1),
            (256, 16 * D * 4, 3),
            (8, 16 * D * 4, 3),
            (32, 32 * D * 4, 2),
        ):
            with mock.patch.object(
                codecs.backward, "_row_dim", group_size
            ), mock.patch.object(mx4q, "MX4_BUFFER_OPS_MAX_BYTES", max_bytes):
                ctx = codecs.backward.create_context()
                ref = codecs.backward.encode(grad, ctx)
                out = chunked(info(codecs), grad, ctx)
            assert out is not None and out.dtype == torch.uint8
            assert torch.equal(out, ref.view(-1)), (group_size, num_chunks)
            assert (D, group_size, num_chunks) in comm_ops._seq_a2a_bwd_chunked_logged

        # The self-check runs on the first calls for each number of chunks.
        comm_ops._seq_a2a_bwd_chunked_checks.clear()
        env = {"TORCHREC_SEQ_A2A_BWD_CHUNKED_ENCODE_CHECK": "1"}
        ctx = codecs.backward.create_context()
        with mock.patch.dict(os.environ, env):
            for _ in range(4):
                assert chunked(info(codecs), grad, ctx) is not None
            with mock.patch.object(mx4q, "MX4_BUFFER_OPS_MAX_BYTES", 16 * D * 4):
                assert chunked(info(codecs), grad, ctx) is not None
        assert comm_ops._seq_a2a_bwd_chunked_checks == {1: 3, 3: 1}
        assert not comm_ops._seq_a2a_bwd_chunked_disabled

        # Everything outside the exact, supported case takes the old path.
        assert chunked(info(codecs), grad, ctx) is not None
        assert chunked(info(codecs), grad[:0], ctx) is None
        assert chunked(info(codecs), grad.half(), ctx) is None
        grad_t = torch.randn(D, rows, generator=gen).t()
        assert chunked(info(codecs), grad_t, ctx) is None
        with mock.patch.object(codecs.backward, "_loss_scale", 2.0):
            assert chunked(info(codecs), grad, ctx) is None
        with mock.patch.object(comm_ops, "_seq_a2a_bwd_chunked_disabled", True):
            assert chunked(info(codecs), grad, ctx) is None
        fp16 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP16, backward_precision=CommType.FP16
            )
        )
        assert chunked(info(fp16), grad, fp16.backward.create_context()) is None
        # 3 is RoundingMode.stochastic.
        with mock.patch.object(ctx, "rounding_mode", 3):
            assert chunked(info(codecs), grad, ctx) is None
        for dim, group_size in ((384, 96), (512, 1024)):
            with mock.patch.object(ctx, "mx_group_size", group_size):
                g_dim = grad[:, :dim].contiguous()
                assert chunked(info(codecs, dim), g_dim, ctx) is None
            assert (dim, group_size, 0) in comm_ops._seq_a2a_bwd_chunked_logged
        # So does a byte bound below 16 rows, without disabling the path.
        with mock.patch.object(mx4q, "MX4_BUFFER_OPS_MAX_BYTES", 16 * D * 4 - 1):
            assert chunked(info(codecs), grad, ctx) is None
        assert not comm_ops._seq_a2a_bwd_chunked_disabled
        assert chunked(info(codecs), grad, ctx) is not None

        # A failure in the chunked encode, such as a kernel that does not
        # compile, takes the old path for good.
        def failing_encode(
            t: torch.Tensor, ctx: Optional[object] = None
        ) -> torch.Tensor:
            raise RuntimeError("compile failure")

        with mock.patch.object(comm_ops, "_seq_a2a_bwd_chunked_disabled", False):
            with mock.patch.object(codecs.backward, "encode", failing_encode):
                assert chunked(info(codecs), grad, ctx) is None
            assert comm_ops._seq_a2a_bwd_chunked_disabled
            assert chunked(info(codecs), grad, ctx) is None

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_mx4_rows_chunk_plan(self) -> None:
        from torchrec.distributed.mx4_rowmap_quantize import (
            MX4_BUFFER_OPS_MAX_BYTES,
            mx4_rows_chunk_plan,
        )

        # The job's backward gradients, bf16 [rows, 512] with 256-value groups:
        # one launch up to 2,097,136 rows, then two, with 128 groups (two full
        # 64-group steps) per program either way.
        assert mx4_rows_chunk_plan(2_000_000, 512, 2, 256) == (2_000_000, 1, 128)
        assert mx4_rows_chunk_plan(2_097_136, 512, 2, 256) == (2_097_136, 1, 128)
        assert mx4_rows_chunk_plan(2_200_000, 512, 2, 256) == (1_100_000, 2, 128)
        # No plan when 16 rows exceed the byte bound.
        assert mx4_rows_chunk_plan(100, 512, 4, 256, 16 * 512 * 4) is not None
        assert mx4_rows_chunk_plan(100, 512, 4, 256, 16 * 512 * 4 - 1) is None
        for rows, (D, g), esz, max_bytes in itertools.product(
            (1, 15, 16, 17, 1000, 2**21, 2**21 + 1, 4_500_000),
            ((512, 256), (512, 8), (128, 32)),
            (2, 4),
            (MX4_BUFFER_OPS_MAX_BYTES, 64 * 1024),
        ):
            plan = mx4_rows_chunk_plan(rows, D, esz, g, max_bytes)
            assert plan is not None
            rows_per_chunk, num_chunks, gpt = plan
            assert rows_per_chunk % 16 == 0
            assert rows_per_chunk * max(D * esz, D // 2 + D // g) <= max_bytes
            assert (num_chunks - 1) * rows_per_chunk < rows
            assert num_chunks * rows_per_chunk >= rows
            num_threads = math.ceil(math.sqrt(rows * D))
            default_gpt = max(math.ceil(rows * (D // g) / num_threads), 64)
            assert gpt % 64 == 0 and abs(gpt - default_gpt) <= 32, (rows, D, g)

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_mx4_ptr_range_view(self) -> None:
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        x = torch.zeros(64, 512, dtype=torch.bfloat16)
        chunk = x[16:48]
        view = mx4q.PtrRangeView(chunk)
        assert view.data_ptr() == chunk.data_ptr()
        assert view.dtype == torch.bfloat16 and view.device == x.device
        assert view.ptr_range() == 32 * 512 * 2
        assert chunk.untyped_storage().nbytes() == 64 * 512 * 2
        assert mx4q._ptr_arg(chunk) is chunk
        # GPU chunks go to the kernel as views, except under the interpreter.
        with mock.patch.object(
            torch.Tensor, "is_cuda", new_callable=mock.PropertyMock
        ) as is_cuda:
            is_cuda.return_value = True
            with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "0"}):
                arg = mx4q._ptr_arg(chunk)
            with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
                assert mx4q._ptr_arg(chunk) is chunk
        assert isinstance(arg, mx4q.PtrRangeView) and arg.tensor is chunk
        big = torch.empty(2**31, dtype=torch.uint8, device="meta")
        assert mx4q.PtrRangeView(big[: 2**31 - 1]).ptr_range() == 2**31 - 1
        try:
            from triton.backends.amd.compiler import HIPBackend
        except Exception:
            return
        assert HIPBackend.is_within_2gb(mx4q.PtrRangeView(big[: 2**31 - 1]))
        assert not HIPBackend.is_within_2gb(big[: 2**31 - 1])
        assert not HIPBackend.is_within_2gb(mx4q.PtrRangeView(big))

    @classmethod
    def _quantize_mx4_rows_chunked_checked(
        cls,
        x: torch.Tensor,
        group_size: int,
        rounding_mode: Optional[int],
        rows_per_chunk: int,
        groups_per_program: int,
    ) -> torch.Tensor:
        """quantize_mx4_rows_chunked, with each kernel launch's arguments
        checked against its row chunk before the launch runs."""
        from fbgemm_gpu.triton.common import get_mx4_exp_bias
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        kernel = mx4q._kernel_quantize_mx4
        rows, D = x.shape
        groups_per_row = D // group_size
        row_bytes = D // 2 + groups_per_row
        interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"
        out_starts: List[int] = []

        def span(t: object) -> Tuple[int, int]:
            # Off the interpreter, GPU chunks carry their own byte range for
            # the buffer-ops specialization.
            if isinstance(t, mx4q.PtrRangeView):
                assert x.is_cuda and not interpret
                return t.data_ptr(), t.ptr_range()
            assert isinstance(t, torch.Tensor) and not (x.is_cuda and not interpret)
            return t.data_ptr(), t.numel() * t.element_size()

        class Launcher:
            def __getitem__(self, grid: Tuple[int]) -> Callable[..., None]:
                def launch(a: object, out: object, **kwargs: object) -> None:
                    r0 = len(out_starts) * rows_per_chunk
                    r1 = min(r0 + rows_per_chunk, rows)
                    assert r0 < rows
                    a_ptr, a_bytes = span(a)
                    assert a_ptr == x[r0].data_ptr()
                    assert a_bytes == (r1 - r0) * D * x.element_size()
                    out_ptr, out_bytes = span(out)
                    out_starts.append(out_ptr - r0 * row_bytes)
                    assert out_starts[-1] == out_starts[0]
                    assert out_bytes == (r1 - r0) * row_bytes
                    n = (r1 - r0) * groups_per_row
                    assert tuple(grid) == (
                        (n + groups_per_program - 1) // groups_per_program,
                    )
                    assert kwargs == {
                        "rand_bits": None,
                        "M": r1 - r0,
                        "K": D,
                        "GROUPS_PER_ROW": groups_per_row,
                        "GROUPS_PER_THREAD": groups_per_program,
                        "ROW_PADDING": 0,
                        "GROUP_SIZE": group_size,
                        "EBITS": 2,
                        "MBITS": 1,
                        # RoundingMode.even for None, as in fp32_to_mx4.
                        "ROUNDING_MODE": 2 if rounding_mode is None else rounding_mode,
                        "STOCHASTIC_CASTING": False,
                        "FP4_EXP_BIAS": get_mx4_exp_bias(2),
                        "GROUP_LOAD": 64,
                        "USE_INT64": False,
                    }, kwargs
                    kernel[grid](a, out, **kwargs)

                return launch

        with mock.patch.object(mx4q, "_kernel_quantize_mx4", Launcher()):
            out = mx4q.quantize_mx4_rows_chunked(
                x, group_size, rounding_mode, rows_per_chunk, groups_per_program
            )
        assert len(out_starts) == (rows + rows_per_chunk - 1) // rows_per_chunk
        assert out_starts[0] == out.data_ptr() and out.numel() == rows * row_bytes
        return out

    @classmethod
    def _check_mx4_rows_chunked(
        cls,
        device: torch.device,
        shapes: Tuple[Tuple[int, int], ...] = (
            (512, 256),
            (512, 32),
            (512, 8),
            (256, 64),
        ),
    ) -> None:
        from fbgemm_gpu.triton.quantize import triton_quantize_mx4
        from torchrec.distributed.mx4_rowmap_quantize import (
            MX4_BUFFER_OPS_MAX_BYTES,
            mx4_rows_chunk_plan,
        )

        gen = torch.Generator().manual_seed(0)
        for (D, g), dtype, mode in itertools.product(
            shapes,
            (torch.bfloat16, torch.float32),
            (2, 0, 1, 4),
        ):
            rows = 53
            x = torch.randn(rows, D, generator=gen) * torch.pow(
                2.0, torch.randint(-60, 60, (rows, 1), generator=gen).float()
            )
            # Edge groups, at the start of the first and of the last chunk:
            # all zero, infinite, near the fp32 max, subnormal, with a NaN, and
            # tiny values next to a large one.
            groups = x.view(-1, g)
            for first in (0, 48 * (D // g)):
                groups[first] = 0.0
                groups[first + 1, 0] = float("inf")
                groups[first + 2] = 3.0e38
                groups[first + 3] = 1e-40
                groups[first + 4, 1] = float("nan")
                groups[first + 5] = 1e-6
                groups[first + 5, 0] = 3.0
            x = x.to(dtype).to(device)
            ref = triton_quantize_mx4(x, g, rounding_mode=mode).view(-1)
            # 16-row launches (16 + 16 + 16 + 5 rows) and one launch, with the
            # planned and other numbers of groups per program.
            for max_bytes in (16 * D * x.element_size(), MX4_BUFFER_OPS_MAX_BYTES):
                plan = mx4_rows_chunk_plan(rows, D, x.element_size(), g, max_bytes)
                assert plan is not None
                rows_per_chunk, _, gpt = plan
                for groups_per_program in (gpt, 64, 100, 192):
                    out = cls._quantize_mx4_rows_chunked_checked(
                        x, g, mode, rows_per_chunk, groups_per_program
                    )
                    assert torch.equal(out, ref), (D, g, dtype, mode, max_bytes)
                if mode == 2:
                    out = cls._quantize_mx4_rows_chunked_checked(
                        x, g, None, rows_per_chunk, gpt
                    )
                    assert torch.equal(out, ref), (D, g, dtype, max_bytes)

    @classmethod
    def _check_seq_a2a_bwd_chunked_encode_gpu_branch(cls) -> None:
        """The GPU branch of _seq_a2a_bwd_chunked_encode on CPU tensors under
        the Triton interpreter, against the old path's GPU encode,
        fp32_to_mx4 -> triton_quantize_mx4."""
        from fbgemm_gpu.triton.common import RoundingMode
        from fbgemm_gpu.triton.quantize import triton_quantize_mx4
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        D, rows = 512, 53
        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        a2ai = comm_ops.All2AllSequenceInfo(
            embedding_dim=D,
            lengths_after_sparse_data_all2all=torch.zeros(1, 1, dtype=torch.int),
            forward_recat_tensor=None,
            # pyrefly: ignore [bad-argument-type]
            backward_recat_tensor=None,
            input_splits=[],
            output_splits=[],
            codecs=codecs,
        )
        gen = torch.Generator().manual_seed(2)
        # 16-row chunks (16 + 16 + 16 + 5 rows) or a single launch.
        for dtype, mode, chunk_rows in (
            (torch.bfloat16, None, 16),
            (torch.bfloat16, RoundingMode.nearest, 0),
            (torch.float32, RoundingMode.floor, 16),
            (torch.float32, RoundingMode.ceil, 0),
        ):
            grad = torch.randn(rows, D, generator=gen) * torch.pow(
                2.0, torch.randint(-60, 60, (rows, 1), generator=gen).float()
            )
            # A flat gradient goes through the same [rows, D] encode.
            grad = grad.to(dtype).view(-1) if chunk_rows else grad.to(dtype)
            ctx = codecs.backward.create_context()
            assert ctx is not None
            ctx.rounding_mode = mode
            ref = triton_quantize_mx4(
                grad, 256, rounding_mode=RoundingMode.even if mode is None else mode
            ).view(-1)
            max_bytes = chunk_rows * D * grad.element_size()
            max_bytes = max_bytes or mx4q.MX4_BUFFER_OPS_MAX_BYTES
            plan = mx4q.mx4_rows_chunk_plan(
                rows, D, grad.element_size(), 256, max_bytes
            )
            assert plan is not None
            # Every tensor reports is_cuda, so the encode takes its GPU branch.
            with mock.patch.object(
                torch.Tensor, "is_cuda", new_callable=mock.PropertyMock
            ) as is_cuda, mock.patch.object(
                mx4q, "MX4_BUFFER_OPS_MAX_BYTES", max_bytes
            ), mock.patch.object(
                mx4q,
                "quantize_mx4_rows_chunked",
                wraps=mx4q.quantize_mx4_rows_chunked,
            ) as launcher:
                is_cuda.return_value = True
                out = comm_ops._seq_a2a_bwd_chunked_encode(a2ai, grad, ctx)
            assert out is not None and torch.equal(out, ref), (dtype, mode)
            (x, *args), _ = launcher.call_args
            assert launcher.call_count == 1 and x.shape == (rows, D)
            assert x.data_ptr() == grad.data_ptr()
            assert args == [256, mode, plan[0], plan[2]], args
            num_chunks = 4 if chunk_rows else 1
            assert plan[1] == num_chunks
            assert (D, 256, num_chunks) in comm_ops._seq_a2a_bwd_chunked_logged
        assert not comm_ops._seq_a2a_bwd_chunked_disabled

    @classmethod
    def _check_mx4_rows_chunked_interpret(cls) -> None:
        try:
            import numpy
            from triton.runtime.interpreter import InterpretedFunction

            # The interpreter takes the kernel's runtime loop bound with int()
            # of a one-element array, which newer numpy rejects.
            int(numpy.ones(1))
        except (ImportError, TypeError, DeprecationWarning):
            raise SystemExit(_TRITON_INTERPRETER_UNAVAILABLE)
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        assert isinstance(mx4q._kernel_quantize_mx4, InterpretedFunction)
        # The interpreter runs every program in turn: two shapes keep it short.
        cls._check_mx4_rows_chunked(torch.device("cpu"), ((512, 256), (256, 64)))
        cls._check_seq_a2a_bwd_chunked_encode_gpu_branch()

    def test_mx4_rows_chunked_interpret_cpu(self) -> None:
        # TRITON_INTERPRET must be set before the kernels are defined, so the
        # check runs in a fresh process.
        ctx = multiprocessing.get_context("spawn")
        with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
            p = ctx.Process(target=self._check_mx4_rows_chunked_interpret)
            p.start()
        p.join()
        if p.exitcode == _TRITON_INTERPRETER_UNAVAILABLE:
            self.skipTest("Needs the Triton interpreter (triton and numpy)")
        self.assertEqual(0, p.exitcode)

    @unittest.skipIf(not torch.cuda.is_available(), "Needs a GPU for Triton")
    def test_mx4_rows_chunked_gpu(self) -> None:
        self._check_mx4_rows_chunked(torch.device("cuda"))

    _UNBUCKETIZE_ROWMAP_FLAGS: Tuple[str, ...] = (
        "SEQ_A2A_UNBUCKETIZE_ROWMAP",
        "SEQ_A2A_UNBUCKETIZE_ROWMAP_CHECK",
        "SEQ_A2A_PERMUTE_QUANTIZED_FWD",
        "SEQ_A2A_FUSED_ROWMAP_FWD",
        "SEQ_A2A_FUSED_ROWMAP_BWD",
        "SEQ_A2A_BWD_CHUNKED_ENCODE",
        "UNBUCKETIZE_GATHER_BWD",
    )

    @classmethod
    def _set_unbucketize_rowmap_env(cls, env: Dict[str, str]) -> None:
        for flag in cls._UNBUCKETIZE_ROWMAP_FLAGS:
            os.environ[f"TORCHREC_{flag}"] = env.get(flag, "0")

    @classmethod
    def _run_unbucketize_rowmap_a2a(
        cls,
        a2a: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        splits: Tuple[List[int], List[int]],
        env: Dict[str, str],
        backward: bool = True,
        bwd_env: Optional[Dict[str, str]] = None,
        grad_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], str, bool]:
        cls._set_unbucketize_rowmap_env(env)
        x = inputs["embs"].clone().requires_grad_(True)
        with torch.set_grad_enabled(grad_enabled):
            awaitable = a2a(
                local_embs=x,
                lengths=inputs["lengths"],
                input_splits=splits[0],
                output_splits=splits[1],
                unbucketize_permute_tensor=inputs["permute"],
            )
            out = awaitable.wait()
            # The Request does not keep the permute past its wait.
            request = awaitable._tensor_awaitable
            assert getattr(request, "output_row_map", None) is None
            node = type(out.grad_fn).__name__ if out.grad_fn is not None else ""
            is_view = out._is_view()
            # The old path returns a fresh index_select output, which allows
            # in-place ops.
            out.mul_(2.0)
        if grad_enabled and backward:
            cls._set_unbucketize_rowmap_env(env if bwd_env is None else bwd_env)
            out.backward(inputs["grad"])
        return out.detach(), x.grad, node, is_view

    @staticmethod
    def _bits_equal(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
        # Raw bits, so that NaNs compare equal to themselves.
        if a is None or b is None:
            return a is None and b is None
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[a.element_size()]
        return torch.equal(a.view(bits), b.view(bits))

    @classmethod
    def _test_alltoall_sequence_unbucketize_rowmap_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        from torchrec.distributed.dist_data import SequenceEmbeddingsAllToAll

        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        D, B_l = 512, 3
        tables_mp = [[0], [1, 2]]
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(1, 6, (3, world_size * B_l), generator=gen).int()
        lengths_mp = lengths[tables_mp[rank]]
        splits = (
            [
                int(lengths[tables_mp[j]][:, rank * B_l : (rank + 1) * B_l].sum())
                for j in range(world_size)
            ],
            [
                int(lengths_mp[:, j * B_l : (j + 1) * B_l].sum())
                for j in range(world_size)
            ],
        )
        gen.manual_seed(1 + rank)
        rows_out = int(lengths_mp.sum())
        rows_in = sum(splits[0])
        permute = torch.randperm(rows_in, generator=gen)
        assert not torch.equal(permute, torch.arange(rows_in))
        inputs = {
            "lengths": lengths_mp,
            "embs": torch.randn(rows_out, D, generator=gen)
            * torch.pow(10.0, torch.randint(-8, 8, (rows_out, 1), generator=gen)),
            "grad": torch.randn(rows_in, D, generator=gen)
            * torch.pow(10.0, torch.randint(-8, 8, (rows_in, 1), generator=gen)),
            "permute": permute,
        }
        fwd = comm_ops._seq_a2a_fwd_decode_unbucketize
        bwd = comm_ops._seq_a2a_bwd_unbucketize_payload
        taken: List[str] = []

        def spy_fwd(*args: object) -> Optional[torch.Tensor]:
            # pyrefly: ignore[bad-argument-type]
            ret = fwd(*args)
            taken.append("fwd" if ret is not None else "fwd declined")
            return ret

        def spy_bwd(*args: object) -> torch.Tensor:
            taken.append("bwd")
            # pyrefly: ignore[bad-argument-type]
            return bwd(*args)

        def mx4_codecs(
            output_dtype: Optional[SparseType] = None,
        ) -> QuantizedCommCodecs:
            # The shared-SID CA job's sequence qcomms: MX4 with 8-value groups
            # forward (320-byte rows at D=512) and 256-value groups backward
            # (258-byte rows).
            return get_qcomm_codecs(
                QCommsConfig(
                    forward_precision=CommType.MX4,
                    backward_precision=CommType.MX4,
                    mx4_quantize_dim=8,
                    mx4_quantize_dim_bwd=256,
                    output_dtype=output_dtype,
                )
            )

        on = {"SEQ_A2A_UNBUCKETIZE_ROWMAP": "1"}
        on_check = dict(on, SEQ_A2A_UNBUCKETIZE_ROWMAP_CHECK="1")
        on_envs = (
            on,
            on_check,
            # Behind the job's other sequence all-to-all paths.
            dict(
                on_check,
                SEQ_A2A_PERMUTE_QUANTIZED_FWD="1",
                SEQ_A2A_FUSED_ROWMAP_FWD="1",
                SEQ_A2A_FUSED_ROWMAP_BWD="1",
                SEQ_A2A_BWD_CHUNKED_ENCODE="1",
            ),
        )
        # Applying the permute to the packed rows must reproduce decode +
        # unbucketize gather forward and gather + encode backward bit for bit,
        # drop the unbucketize node, and allow in-place ops on the result.
        # fbgemm's CPU MX4 encode takes fp32 only, so the BF16-output case
        # checks the forward.
        with mock.patch.object(
            comm_ops, "_seq_a2a_fwd_decode_unbucketize", spy_fwd
        ), mock.patch.object(comm_ops, "_seq_a2a_bwd_unbucketize_payload", spy_bwd):
            for output_dtype, gather in itertools.product(
                (SparseType.BF16, None), ("0", "1")
            ):
                a2a = SequenceEmbeddingsAllToAll(
                    pg, [1, 2], codecs=mx4_codecs(output_dtype)
                )
                backward = output_dtype is None
                old = {"UNBUCKETIZE_GATHER_BWD": gather}
                taken.clear()
                out_ref, grad_ref, node, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, old, backward
                )
                assert node == "_UnbucketizePermuteBackward", node
                assert taken == [], taken
                for env in on_envs:
                    taken.clear()
                    out, grad_in, node, is_view = cls._run_unbucketize_rowmap_a2a(
                        a2a, inputs, splits, dict(env, **old), backward
                    )
                    assert taken == (["fwd", "bwd"] if backward else ["fwd"]), taken
                    assert node == "All2All_Seq_Req_WaitBackward", node
                    assert not is_view
                    assert cls._bits_equal(out, out_ref)
                    assert cls._bits_equal(grad_in, grad_ref)
                if not backward:
                    continue
                # The backward follows its forward, not the flag.
                taken.clear()
                out, grad_in, _, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, dict(on, **old), True, old
                )
                assert taken == ["fwd", "bwd"], taken
                assert cls._bits_equal(grad_in, grad_ref)
                taken.clear()
                out, grad_in, node, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, old, True, dict(on, **old)
                )
                assert taken == [] and node == "_UnbucketizePermuteBackward"
                assert cls._bits_equal(grad_in, grad_ref)
                # Without grad the Request never gets the permute.
                out, _, node, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, on, grad_enabled=False
                )
                assert taken == [] and node == ""
                assert cls._bits_equal(out, out_ref)
                # Nor when only the wait runs without grad.
                awaitable = a2a(
                    local_embs=inputs["embs"].clone().requires_grad_(True),
                    lengths=inputs["lengths"],
                    input_splits=splits[0],
                    output_splits=splits[1],
                    unbucketize_permute_tensor=inputs["permute"],
                )
                with torch.no_grad():
                    out = awaitable.wait()
                assert taken == [] and cls._bits_equal(out * 2.0, out_ref)
                # A backward that may not encode as it is gathers the gradient
                # rows first, as the unbucketize node does.
                with mock.patch.object(
                    comm_ops, "_seq_a2a_bwd_unbucketize_packed", return_value=False
                ):
                    out, grad_in, _, _ = cls._run_unbucketize_rowmap_a2a(
                        a2a, inputs, splits, dict(on, **old)
                    )
                assert taken == ["fwd"], taken
                assert cls._bits_equal(out, out_ref)
                assert cls._bits_equal(grad_in, grad_ref)
                # A forward that declines leaves the unbucketize to the callback.
                taken.clear()
                with mock.patch.object(
                    comm_ops, "_seq_a2a_unbucketize_fwd_disabled", True
                ):
                    out, grad_in, node, _ = cls._run_unbucketize_rowmap_a2a(
                        a2a, inputs, splits, dict(on, **old)
                    )
                assert taken == ["fwd declined"], taken
                assert node == "_UnbucketizePermuteBackward", node
                assert cls._bits_equal(out, out_ref)
                assert cls._bits_equal(grad_in, grad_ref)
                taken.clear()
        assert ("forward", D, 8, 1) in comm_ops._seq_a2a_unbucketize_logged
        assert ("backward", D, 256, 1) in comm_ops._seq_a2a_unbucketize_logged
        assert comm_ops._seq_a2a_unbucketize_fwd_checks == 3
        assert comm_ops._seq_a2a_unbucketize_bwd_checks == 3
        assert not comm_ops._seq_a2a_unbucketize_fwd_disabled
        assert not comm_ops._seq_a2a_unbucketize_bwd_disabled

        # Without a bijection verdict, or without MX4 forward qcomms, the
        # Request never gets the permute either.
        codecs = mx4_codecs()
        fp16 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP16, backward_precision=CommType.FP16
            )
        )
        for a2a_codecs, verdict, ref_node in (
            (codecs, "0", "IndexSelectBackward0"),
            (fp16, "1", "_UnbucketizePermuteBackward"),
        ):
            with mock.patch.dict(
                os.environ, {"TORCHREC_UNBUCKETIZE_INDEX_COPY_BWD": verdict}
            ), mock.patch.object(comm_ops, "_seq_a2a_fwd_decode_unbucketize", spy_fwd):
                a2a = SequenceEmbeddingsAllToAll(pg, [1, 2], codecs=a2a_codecs)
                out_ref, grad_ref, _, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, {}
                )
                out, grad_in, node, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, on
                )
            assert taken == [] and node == ref_node, (taken, node)
            assert cls._bits_equal(out, out_ref)
            assert cls._bits_equal(grad_in, grad_ref)

        # A position-dependent decode does not commute with the row gather; the
        # self-check must catch it, return the old path's rows and disable the
        # forward path for the rest of the process.
        a2a = SequenceEmbeddingsAllToAll(pg, [1, 2], codecs=codecs)
        out_ref, grad_ref, _, _ = cls._run_unbucketize_rowmap_a2a(
            a2a, inputs, splits, {}
        )
        decode = codecs.forward.decode

        def bad_decode(t: torch.Tensor, ctx: Optional[object] = None) -> torch.Tensor:
            out = decode(t, ctx)
            return out + torch.arange(out.numel()).remainder(7).to(out.dtype)

        comm_ops._seq_a2a_unbucketize_fwd_checks = 0
        with mock.patch.object(codecs.forward, "decode", bad_decode):
            out_bad, grad_bad, _, _ = cls._run_unbucketize_rowmap_a2a(
                a2a, inputs, splits, {}
            )
            out, grad_in, node, _ = cls._run_unbucketize_rowmap_a2a(
                a2a, inputs, splits, on_check
            )
        assert node == "All2All_Seq_Req_WaitBackward", node
        assert cls._bits_equal(out, out_bad) and not cls._bits_equal(out, out_ref)
        assert cls._bits_equal(grad_in, grad_bad)
        assert comm_ops._seq_a2a_unbucketize_fwd_disabled
        with mock.patch.object(comm_ops, "_seq_a2a_fwd_decode_unbucketize", spy_fwd):
            out, grad_in, node, _ = cls._run_unbucketize_rowmap_a2a(
                a2a, inputs, splits, on
            )
        assert taken == ["fwd declined"] and node == "_UnbucketizePermuteBackward"
        assert cls._bits_equal(out, out_ref)
        assert cls._bits_equal(grad_in, grad_ref)
        taken.clear()
        comm_ops._seq_a2a_unbucketize_fwd_disabled = False

        # Likewise a position-dependent encode in the backward: the self-check
        # must return the old path's payload and, from then on, the backward
        # gathers the gradient rows before encoding them.
        encode = codecs.backward.encode

        def bad_encode(t: torch.Tensor, ctx: Optional[object] = None) -> torch.Tensor:
            q = encode(t, ctx).clone()
            q.view(-1)[::7] ^= 1
            return q

        comm_ops._seq_a2a_unbucketize_bwd_checks = 0
        with mock.patch.object(codecs.backward, "encode", bad_encode):
            _, grad_bad, _, _ = cls._run_unbucketize_rowmap_a2a(a2a, inputs, splits, {})
            _, grad_in, _, _ = cls._run_unbucketize_rowmap_a2a(
                a2a, inputs, splits, on_check
            )
        assert cls._bits_equal(grad_in, grad_bad)
        assert not cls._bits_equal(grad_in, grad_ref)
        assert comm_ops._seq_a2a_unbucketize_bwd_disabled
        with mock.patch.object(
            comm_ops, "_seq_a2a_fwd_decode_unbucketize", spy_fwd
        ), mock.patch.object(comm_ops, "_seq_a2a_bwd_unbucketize_payload", spy_bwd):
            out, grad_in, _, _ = cls._run_unbucketize_rowmap_a2a(
                a2a, inputs, splits, on
            )
        assert taken == ["fwd"], taken
        assert cls._bits_equal(out, out_ref)
        assert cls._bits_equal(grad_in, grad_ref)
        cls._set_unbucketize_rowmap_env({})
        dist.destroy_process_group()

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_alltoall_sequence_unbucketize_rowmap_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence_unbucketize_rowmap_cpu,
        )

    def test_seq_a2a_unbucketize_rowmap_cpu(self) -> None:
        D, rows = 512, 23
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(rows, generator=gen)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(rows)

        def info(codecs: Optional[QuantizedCommCodecs]) -> comm_ops.All2AllSequenceInfo:
            return comm_ops.All2AllSequenceInfo(
                embedding_dim=D,
                lengths_after_sparse_data_all2all=torch.zeros(1, 1, dtype=torch.int),
                forward_recat_tensor=None,
                # pyrefly: ignore [bad-argument-type]
                backward_recat_tensor=None,
                input_splits=[],
                output_splits=[],
                codecs=codecs,
            )

        def mx4_codecs(
            output_dtype: Optional[SparseType] = None,
        ) -> QuantizedCommCodecs:
            return get_qcomm_codecs(
                QCommsConfig(
                    forward_precision=CommType.MX4,
                    backward_precision=CommType.MX4,
                    mx4_quantize_dim=8,
                    mx4_quantize_dim_bwd=256,
                    output_dtype=output_dtype,
                )
            )

        values = torch.randn(rows, D, generator=gen) * torch.pow(
            2.0, torch.randint(-60, 60, (rows, 1), generator=gen).float()
        )
        values[3] = 0.0
        values[4, 0] = float("inf")
        values[5, 9] = 3.0e38
        values[6] = 1e-40
        fwd = comm_ops._seq_a2a_fwd_decode_unbucketize
        for output_dtype in (None, SparseType.BF16):
            codecs = mx4_codecs(output_dtype)
            ctx = codecs.forward.create_context()
            payload = codecs.forward.encode(values.view(-1), ctx)
            ref = codecs.forward.decode(payload, ctx).view(-1, D).index_select(0, perm)
            for p in (perm, perm.int()):
                out = fwd(info(codecs), payload, ctx, p)
                assert out is not None and not out._is_view()
                assert self._bits_equal(out, ref), (output_dtype, p.dtype)
        assert ("forward", D, 8, 1) in comm_ops._seq_a2a_unbucketize_logged

        # Everything outside the exact, supported case takes the old path.
        assert fwd(info(codecs), payload, ctx, perm) is not None
        assert fwd(info(None), payload, ctx, perm) is None
        assert fwd(info(codecs), payload[:-1], ctx, perm) is None
        assert fwd(info(codecs), payload.view(torch.int8), ctx, perm) is None
        assert fwd(info(codecs), payload.view(2, -1), ctx, perm) is None
        assert fwd(info(codecs), payload, ctx, perm[:-1]) is None
        assert fwd(info(codecs), payload, ctx, perm.view(1, -1)) is None
        assert fwd(info(codecs), payload, ctx, perm.to(torch.int16)) is None
        assert fwd(info(codecs), payload[:0], ctx, perm[:0]) is None
        with mock.patch.object(comm_ops, "_seq_a2a_unbucketize_fwd_disabled", True):
            assert fwd(info(codecs), payload, ctx, perm) is None
        with mock.patch.object(codecs.forward, "_loss_scale", 2.0):
            assert fwd(info(codecs), payload, ctx, perm) is None
        fp16 = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP16, backward_precision=CommType.FP16
            )
        )
        assert fwd(info(fp16), payload, ctx, perm) is None
        # 24-value groups do not tile a row; 256-value groups give 258-byte
        # rows, off aten's vectorized gather.
        for group_size, row_bytes in ((24, 0), (256, 258)):
            with mock.patch.object(ctx, "mx_group_size", group_size):
                p = torch.zeros(rows * max(row_bytes, 1), dtype=torch.uint8)
                assert fwd(info(codecs), p, ctx, perm) is None
            assert ("forward", D, group_size, 0) in comm_ops._seq_a2a_unbucketize_logged

        # The backward: encoding the gradient as it arrives and gathering the
        # packed rows by the inverse permute is encoding the gathered rows.
        packed = comm_ops._seq_a2a_bwd_unbucketize_packed
        bwd = comm_ops._seq_a2a_bwd_unbucketize_payload
        codecs = mx4_codecs()
        # fbgemm's CPU MX4 encode is not position independent for groups with
        # an infinite or near-maximal value: such a row encodes differently on
        # its own and inside a batch. The Triton encode is, see
        # _check_seq_a2a_unbucketize_rowmap_kernels.
        grad = values.clone()
        grad[4:6] = torch.randn(2, D, generator=gen)
        for group_size in (256, 8, 32):
            with mock.patch.object(codecs.backward, "_row_dim", group_size):
                ctx = codecs.backward.create_context()
            ref = codecs.backward.encode(grad.index_select(0, inv), ctx).view(-1)
            payload = codecs.backward.encode(grad, ctx).view(-1)
            for p in (perm, perm.int()):
                assert packed(info(codecs), grad, p, ctx)
                out = bwd(info(codecs), grad, payload, p, ctx)
                assert out.dtype == torch.uint8 and torch.equal(out, ref), group_size
            assert (
                "backward",
                D,
                group_size,
                1,
            ) in comm_ops._seq_a2a_unbucketize_logged
            # A payload that is not whole MX4 rows is encoded in the old order.
            for bad in (payload[:-1], payload.view(torch.int8), payload[1:]):
                assert torch.equal(bwd(info(codecs), grad, bad, perm, ctx), ref)
        assert torch.equal(
            comm_ops._seq_a2a_unbucketize_grad(grad, perm), grad.index_select(0, inv)
        )

        # Everything outside the exact, supported case gathers the gradient
        # rows before encoding them.
        ctx = codecs.backward.create_context()
        assert packed(info(codecs), grad, perm, ctx)
        assert not packed(info(None), grad, perm, ctx)
        assert not packed(info(codecs), grad[:-1], perm, ctx)
        assert not packed(info(codecs), grad.half(), perm, ctx)
        grad_t = torch.randn(D, rows, generator=gen).t()
        assert not packed(info(codecs), grad_t, perm, ctx)
        assert packed(info(codecs), grad_t.contiguous(), perm, ctx)
        with mock.patch.object(comm_ops, "_seq_a2a_unbucketize_bwd_disabled", True):
            assert not packed(info(codecs), grad, perm, ctx)
        with mock.patch.object(codecs.backward, "_loss_scale", 2.0):
            assert not packed(info(codecs), grad, perm, ctx)
        assert not packed(info(fp16), grad, perm, fp16.backward.create_context())
        # 3 is RoundingMode.stochastic.
        with mock.patch.object(ctx, "rounding_mode", 3):
            assert not packed(info(codecs), grad, perm, ctx)
        with mock.patch.object(ctx, "rounding_mode", None):
            assert packed(info(codecs), grad, perm, ctx)
        with mock.patch.object(ctx, "mx_group_size", 24):
            assert not packed(info(codecs), grad, perm, ctx)
        assert ("backward", D, 24, 0) in comm_ops._seq_a2a_unbucketize_logged

        # The packed rows are copied as the widest integers that tile them.
        index_select = torch.Tensor.index_select
        dtypes: List[torch.dtype] = []

        def recording_index_select(
            t: torch.Tensor, dim: int, index: torch.Tensor
        ) -> torch.Tensor:
            dtypes.append(t.dtype)
            return index_select(t, dim, index)

        flat = torch.randint(0, 256, (rows * 320 + 8,), generator=gen).to(torch.uint8)
        with mock.patch.object(torch.Tensor, "index_select", recording_index_select):
            for row_bytes, offset, dtype in (
                (320, 0, torch.int64),
                (258, 0, torch.int16),
                (12, 0, torch.int32),
                (5, 0, torch.uint8),
                (320, 4, torch.int32),
                (320, 1, torch.uint8),
            ):
                dtypes.clear()
                p = flat[offset : offset + rows * row_bytes]
                out = comm_ops._gather_packed_rows(p, row_bytes, perm)
                assert dtypes == [dtype], (row_bytes, offset, dtypes)
                assert torch.equal(out, p.view(rows, row_bytes)[perm].view(-1))

    @classmethod
    def _check_seq_a2a_unbucketize_rowmap_kernels(cls, device: torch.device) -> None:
        """Both directions with the GPU codec kernels (fbgemm's Triton MX4
        decode, its encode and the chunked encode) at the job's shapes, against
        the old path's order of operations. On CPU tensors, under the Triton
        interpreter, every tensor reports is_cuda."""
        from fbgemm_gpu import quantize_utils
        from fbgemm_gpu.triton.quantize import (
            triton_dequantize_mx4,
            triton_quantize_mx4,
        )
        from torchrec.distributed import mx4_rowmap_quantize as mx4q

        D, rows = 512, 37
        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
                output_dtype=SparseType.BF16,
            )
        )
        a2ai = comm_ops.All2AllSequenceInfo(
            embedding_dim=D,
            lengths_after_sparse_data_all2all=torch.zeros(1, 1, dtype=torch.int),
            forward_recat_tensor=None,
            # pyrefly: ignore [bad-argument-type]
            backward_recat_tensor=None,
            input_splits=[],
            output_splits=[],
            codecs=codecs,
        )
        gen = torch.Generator().manual_seed(3)
        perm = torch.randperm(rows, generator=gen).to(device)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(rows, device=device)
        x = torch.randn(rows, D, generator=gen) * torch.pow(
            2.0, torch.randint(-60, 60, (rows, 1), generator=gen).float()
        )
        x[2] = 0.0
        x[3, 0] = float("inf")
        x[4, 300] = 3.0e38
        x[5, 1] = float("nan")
        x[6] = 1e-40
        x = x.to(torch.bfloat16).to(device)
        fwd_ctx = codecs.forward.create_context()
        bwd_ctx = codecs.backward.create_context()
        payload = triton_quantize_mx4(x, 8).view(-1)
        ref_rows = triton_dequantize_mx4(payload, 8, output_dtype=torch.bfloat16)
        ref_rows = ref_rows.view(-1, D).index_select(0, perm)
        assert bwd_ctx is not None
        ref_payload = triton_quantize_mx4(
            x.index_select(0, inv), 256, rounding_mode=bwd_ctx.rounding_mode
        ).view(-1)
        # The codec's GPU kernels, which quantize_utils imports on GPU hosts.
        with mock.patch.object(
            quantize_utils, "quantize_mx4", triton_quantize_mx4, create=True
        ), mock.patch.object(
            quantize_utils, "triton_dequantize_mx4", triton_dequantize_mx4, create=True
        ), mock.patch.object(
            torch.Tensor, "is_cuda", new_callable=mock.PropertyMock
        ) as is_cuda, mock.patch.object(
            mx4q, "MX4_BUFFER_OPS_MAX_BYTES", 16 * D * x.element_size()
        ):
            is_cuda.return_value = True
            out = comm_ops._seq_a2a_fwd_decode_unbucketize(a2ai, payload, fwd_ctx, perm)
            assert comm_ops._seq_a2a_bwd_unbucketize_packed(a2ai, x, perm, bwd_ctx)
            # The chunked encode (16-row chunks) and the codec's encode.
            chunked = comm_ops._seq_a2a_bwd_chunked_encode(a2ai, x, bwd_ctx)
            assert chunked is not None
            encoded = codecs.backward.encode(x, bwd_ctx).view(-1)
            bwd_payloads = [
                comm_ops._seq_a2a_bwd_unbucketize_payload(a2ai, x, p, perm, bwd_ctx)
                for p in (chunked, encoded)
            ]
        assert out is not None and cls._bits_equal(out, ref_rows)
        for bwd_payload in bwd_payloads:
            assert torch.equal(bwd_payload, ref_payload)

    @classmethod
    def _check_seq_a2a_unbucketize_rowmap_interpret(cls) -> None:
        try:
            import numpy
            from triton.runtime.interpreter import InterpretedFunction

            # The interpreter takes the kernel's runtime loop bound with int()
            # of a one-element array, which newer numpy rejects.
            int(numpy.ones(1))
        except (ImportError, TypeError, DeprecationWarning):
            raise SystemExit(_TRITON_INTERPRETER_UNAVAILABLE)
        from fbgemm_gpu.triton import quantize

        assert isinstance(quantize._kernel_dequantize_mx4, InterpretedFunction)
        cls._check_seq_a2a_unbucketize_rowmap_kernels(torch.device("cpu"))

    def test_seq_a2a_unbucketize_rowmap_interpret_cpu(self) -> None:
        # TRITON_INTERPRET must be set before the kernels are defined, so the
        # check runs in a fresh process.
        ctx = multiprocessing.get_context("spawn")
        with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
            p = ctx.Process(target=self._check_seq_a2a_unbucketize_rowmap_interpret)
            p.start()
        p.join()
        if p.exitcode == _TRITON_INTERPRETER_UNAVAILABLE:
            self.skipTest("Needs the Triton interpreter (triton and numpy)")
        self.assertEqual(0, p.exitcode)

    @unittest.skipIf(not torch.cuda.is_available(), "Needs a GPU for Triton")
    def test_seq_a2a_unbucketize_rowmap_gpu(self) -> None:
        self._check_seq_a2a_unbucketize_rowmap_kernels(torch.device("cuda"))

    @classmethod
    def _run_unbucketize_compose_a2a(
        cls,
        a2a: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        splits: Tuple[List[int], List[int]],
        env: Dict[str, str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The all-to-all of _run_unbucketize_rowmap_a2a as
        TORCHREC_SEQ_UNBUCKETIZE_COMPOSE runs it: the caller takes the permute
        and skips the unbucketize. Returns the rows in all-to-all order and the
        input gradient for the same gradient of the unbucketized rows."""
        cls._set_unbucketize_rowmap_env(env)
        x = inputs["embs"].clone().requires_grad_(True)
        awaitable = a2a(
            local_embs=x,
            lengths=inputs["lengths"],
            input_splits=splits[0],
            output_splits=splits[1],
            unbucketize_permute_tensor=inputs["permute"],
        )
        permute = awaitable.unbucketize_permute_for_compose()
        assert permute is inputs["permute"]
        awaitable.skip_unbucketize()
        raw = awaitable.wait()
        request = awaitable._tensor_awaitable
        assert getattr(request, "output_row_map", None) is None
        assert not request.row_map_applied
        # Unbucketized row j is all-to-all row permute[j].
        grad = torch.empty_like(inputs["grad"]).index_copy_(0, permute, inputs["grad"])
        (raw * 2.0).backward(grad)
        return raw.detach(), x.grad

    @classmethod
    def _test_alltoall_sequence_unbucketize_compose_cpu(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        from torchrec.distributed.dist_data import SequenceEmbeddingsAllToAll

        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        D, B_l = 512, 3
        tables_mp = [[0], [1, 2]]
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(1, 6, (3, world_size * B_l), generator=gen).int()
        lengths_mp = lengths[tables_mp[rank]]
        splits = (
            [
                int(lengths[tables_mp[j]][:, rank * B_l : (rank + 1) * B_l].sum())
                for j in range(world_size)
            ],
            [
                int(lengths_mp[:, j * B_l : (j + 1) * B_l].sum())
                for j in range(world_size)
            ],
        )
        gen.manual_seed(1 + rank)
        rows_out = int(lengths_mp.sum())
        rows_in = sum(splits[0])
        permute = torch.randperm(rows_in, generator=gen)
        assert not torch.equal(permute, torch.arange(rows_in))
        inputs = {
            "lengths": lengths_mp,
            "embs": torch.randn(rows_out, D, generator=gen)
            * torch.pow(10.0, torch.randint(-8, 8, (rows_out, 1), generator=gen)),
            "grad": torch.randn(rows_in, D, generator=gen)
            * torch.pow(10.0, torch.randint(-8, 8, (rows_in, 1), generator=gen)),
            "permute": permute,
        }
        fwd = comm_ops._seq_a2a_fwd_decode_unbucketize
        bwd = comm_ops._seq_a2a_bwd_unbucketize_payload
        taken: List[str] = []

        def spy_fwd(*args: object) -> Optional[torch.Tensor]:
            taken.append("fwd")
            # pyrefly: ignore[bad-argument-type]
            return fwd(*args)

        def spy_bwd(*args: object) -> torch.Tensor:
            taken.append("bwd")
            # pyrefly: ignore[bad-argument-type]
            return bwd(*args)

        # The shared-SID CA job's sequence qcomms (fbgemm's CPU MX4 encode
        # takes fp32 only).
        codecs = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
                backward_precision=CommType.MX4,
                mx4_quantize_dim=8,
                mx4_quantize_dim_bwd=256,
            )
        )
        u1 = {
            "SEQ_A2A_UNBUCKETIZE_ROWMAP": "1",
            "SEQ_A2A_UNBUCKETIZE_ROWMAP_CHECK": "1",
        }
        envs = (
            {},
            u1,
            # Behind the job's other sequence all-to-all paths.
            dict(
                u1,
                SEQ_A2A_PERMUTE_QUANTIZED_FWD="1",
                SEQ_A2A_FUSED_ROWMAP_FWD="1",
                SEQ_A2A_FUSED_ROWMAP_BWD="1",
                SEQ_A2A_BWD_CHUNKED_ENCODE="1",
            ),
        )
        # Skipping the unbucketize must leave the all-to-all rows in their
        # order, never apply the permute in the Request (U1), and give the
        # old path's rows and input gradient bit for bit once the caller
        # applies the permute.
        with mock.patch.object(
            comm_ops, "_seq_a2a_fwd_decode_unbucketize", spy_fwd
        ), mock.patch.object(comm_ops, "_seq_a2a_bwd_unbucketize_payload", spy_bwd):
            for gather, env in itertools.product(("0", "1"), envs):
                env = dict(env, UNBUCKETIZE_GATHER_BWD=gather)
                a2a = SequenceEmbeddingsAllToAll(pg, [1, 2], codecs=codecs)
                taken.clear()
                out_ref, grad_ref, _, _ = cls._run_unbucketize_rowmap_a2a(
                    a2a, inputs, splits, env
                )
                assert taken == (
                    ["fwd", "bwd"] if env.get("SEQ_A2A_UNBUCKETIZE_ROWMAP") else []
                ), taken
                taken.clear()
                raw, grad_in = cls._run_unbucketize_compose_a2a(
                    a2a, inputs, splits, env
                )
                assert taken == [], taken
                assert cls._bits_equal(raw.index_select(0, permute) * 2.0, out_ref)
                assert cls._bits_equal(grad_in, grad_ref)
        assert not comm_ops._seq_a2a_unbucketize_fwd_disabled
        assert not comm_ops._seq_a2a_unbucketize_bwd_disabled

        # Without a bijection verdict there is no permute to compose.
        with mock.patch.dict(os.environ, {"TORCHREC_UNBUCKETIZE_INDEX_COPY_BWD": "0"}):
            a2a = SequenceEmbeddingsAllToAll(pg, [1, 2], codecs=codecs)
            awaitable = a2a(
                local_embs=inputs["embs"],
                lengths=inputs["lengths"],
                input_splits=splits[0],
                output_splits=splits[1],
                unbucketize_permute_tensor=permute,
            )
            assert awaitable.unbucketize_permute_for_compose() is None
            awaitable.wait()
        cls._set_unbucketize_rowmap_env({})
        dist.destroy_process_group()

    @unittest.skipUnless(_HAS_TRITON, "needs triton")
    def test_alltoall_sequence_unbucketize_compose_cpu(
        self,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_alltoall_sequence_unbucketize_compose_cpu,
        )

    @classmethod
    def _test_custom_backend_is_not_fake(
        cls,
        rank: int,
        world_size: int,
        backend: str,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = none_throws(GroupMember.WORLD)
        assert pg._get_backend_name() == "custom"
        assert not comm_ops.is_fake_pg(pg)

        comm_ops.set_use_sync_collectives(True)
        # One feature per rank, batch size 1, dim 1: rank r sends
        # [10 * r, 10 * r + 1] and receives [r, 10 + r].
        out = comm_ops.variable_batch_alltoall_pooled(
            torch.tensor([10.0 * rank, 10.0 * rank + 1]),
            batch_size_per_rank_per_feature=[[1], [1]],
            batch_size_per_feature_pre_a2a=[1, 1],
            emb_dim_per_rank_per_feature=[[1], [1]],
            group=pg,
        ).wait()
        torch.testing.assert_close(out, torch.tensor([float(rank), 10.0 + rank]))
        dist.destroy_process_group()

    def test_custom_backend_is_not_fake(self) -> None:
        # A real backend registered via register_backend (as nccl2 is) reports
        # backend name "custom", like FakeProcessGroup, and must still run the
        # real all_to_all.
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="wrapped_gloo",
            # pyrefly: ignore[bad-argument-type]
            callable=self._test_custom_backend_is_not_fake,
        )


class TestSeqA2AFlagDefaults(unittest.TestCase):
    def test_paths_default_on(self) -> None:
        from torchrec.distributed import dist_data

        env = {
            k: v for k, v in os.environ.items() if k not in _SEQ_A2A_DEFAULT_ON_FLAGS
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(comm_ops._mx4v2_requested("fwd_enc"))
            self.assertTrue(dist_data._unbucketize_gather_bwd())
            self.assertTrue(dist_data._seq_a2a_unbucketize_rowmap())

    def test_explicit_zero_turns_paths_off(self) -> None:
        from torchrec.distributed import dist_data

        with mock.patch.dict(os.environ, dict.fromkeys(_SEQ_A2A_DEFAULT_ON_FLAGS, "0")):
            self.assertFalse(comm_ops._mx4v2_requested("fwd_enc"))
            self.assertFalse(dist_data._unbucketize_gather_bwd())
            self.assertFalse(dist_data._seq_a2a_unbucketize_rowmap())


def _create_wrapped_gloo(
    store: dist.Store, rank: int, size: int, timeout: timedelta
) -> dist.ProcessGroupGloo:
    return dist.ProcessGroupGloo(store, rank, size, timeout)


# A real backend registered via register_backend, like nccl2.
dist.Backend.register_backend("wrapped_gloo", _create_wrapped_gloo, devices=["cpu"])
# A non-"fake" name for a fake implementation, to build mixed device maps.
dist.Backend.register_backend(
    "fake_alias", _create_fake_pg, extended_api=True, devices=["cpu", "cuda"]
)


class TestIsFakePg(unittest.TestCase):
    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        # Drop compiled entries guarded on (and keeping alive) destroyed PGs.
        torch._dynamo.reset()

    def _check_fake(
        self,
        backend: str,
        expected: bool,
        store: Optional[dist.Store] = None,
        world_size: int = 2,
    ) -> None:
        dist.init_process_group(
            backend=backend,
            store=store if store is not None else FakeStore(),
            rank=0,
            world_size=world_size,
        )
        pg = none_throws(GroupMember.WORLD)
        self.assertEqual(comm_ops.is_fake_pg(pg), expected)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x: Tensor) -> Tensor:
            return x + 1 if comm_ops.is_fake_pg(pg) else x - 1

        # Dynamo skips torchrec frames (including this test) by default.
        with torch._dynamo.config.patch(skip_torchrec=False):
            out = f(torch.zeros(1))
        self.assertEqual(out.item(), 1.0 if expected else -1.0)

    def test_fake_pg(self) -> None:
        self._check_fake("fake", True)

    def test_multi_device_fake_pg(self) -> None:
        self._check_fake("cpu:fake,cuda:fake", True)

    def test_mixed_device_backends_is_not_fake(self) -> None:
        self._check_fake("cpu:fake,cuda:fake_alias", False)

    def test_real_pg_is_not_fake(self) -> None:
        self._check_fake("gloo", False, store=dist.HashStore(), world_size=1)

    def test_unregistered_pg_raises(self) -> None:
        pg = dist.ProcessGroupGloo(dist.HashStore(), 0, 1, timedelta(seconds=30))
        with self.assertRaisesRegex(ValueError, "not initialized"):
            comm_ops.is_fake_pg(pg)


class TestResolvedPodSize(unittest.TestCase):
    """
    pod_size is the multiplier that makes the runtime's TwRw/Grid node width match
    the planner's Topology.intra_group_size. It is absent on every SKU whose NVLink
    domain is a single host, so these cases pin the unset default as tightly as the
    NVL72 values -- a regression there silently narrows the runtime group and drops
    embedding shards rather than failing.
    """

    def test_unset_is_none_so_non_nvl72_skus_are_unaffected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_resolved_pod_size())

    def test_reads_planner_resolved_value(self) -> None:
        for raw, expected in (("1", 1), ("2", 2), ("4", 4), ("18", 18), ("36", 36)):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"TORCHREC_RESOLVED_POD_SIZE": raw}, clear=True
            ):
                self.assertEqual(get_resolved_pod_size(), expected)

    def test_rejects_non_positive_instead_of_zeroing_the_node_width(self) -> None:
        # A 0 would zero devices_per_node and turn the callers' divisibility checks
        # into a ZeroDivisionError, hiding the misconfiguration.
        for raw in ("0", "-1"):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"TORCHREC_RESOLVED_POD_SIZE": raw}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    get_resolved_pod_size()

    def test_rejects_non_numeric_with_the_same_descriptive_error(self) -> None:
        # Otherwise a typo'd launcher value surfaces as a bare
        # "invalid literal for int()" that names neither the variable nor the value.
        for raw in ("abc", "", "2.5"):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"TORCHREC_RESOLVED_POD_SIZE": raw}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                    get_resolved_pod_size()

    def test_topology_domain_multiple_alone_does_not_set_pod_size(self) -> None:
        # Parity with the 1D path (get_topology_group_world_size): only the
        # planner-exported TORCHREC_RESOLVED_POD_SIZE is authoritative, because
        # TOPOLOGY_DOMAIN_MULTIPLE is the configured *minimum* and can disagree
        # with the placement MAST actually gave the job.
        with patch.dict(os.environ, {"TOPOLOGY_DOMAIN_MULTIPLE": "36"}, clear=True):
            self.assertIsNone(get_resolved_pod_size())


class TestTwoDPodSizeKillswitch(unittest.TestCase):
    """
    get_2d_pod_size() is the single place the 2D killswitch is read, so these cases
    pin both of its inputs: the knob and the planner-exported env var. Collapsing to
    1 when the knob is off is what makes the knob a revert rather than a hard failure
    -- the TwRw width check is gated on the same knob.
    """

    def test_knob_on_applies_planner_pod_size(self) -> None:
        with patch.dict(
            os.environ, {"TORCHREC_RESOLVED_POD_SIZE": "4"}, clear=True
        ), patch(
            "torch._utils_internal.justknobs_check", return_value=True
        ) as jk_check:
            self.assertEqual(get_2d_pod_size(), 4)
        jk_check.assert_called_with("pytorch/torchrec:enable_2D_support_for_pod_size")

    def test_knob_off_collapses_to_one_even_when_planner_exported_a_value(self) -> None:
        with patch.dict(
            os.environ, {"TORCHREC_RESOLVED_POD_SIZE": "36"}, clear=True
        ), patch("torch._utils_internal.justknobs_check", return_value=False):
            self.assertEqual(get_2d_pod_size(), 1)

    def test_knob_on_with_no_planner_value_is_a_no_op_multiplier(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "torch._utils_internal.justknobs_check", return_value=True
        ):
            self.assertEqual(get_2d_pod_size(), 1)


@dataclass
class _FakeShardingEnv2D:
    """
    Stands in for ShardingEnv2D, which cannot be constructed without live process
    groups. intra_and_cross_node_pg_2D reads exactly these three attributes, so a
    narrower double would not change the code path under test.
    """

    sharding_pg: object
    node_group_size: Optional[int]
    use_inter_host_allreduce: bool = False


class TestIntraAndCrossNodePg2DNodeWidth(unittest.TestCase):
    """
    The node width intra_and_cross_node_pg_2D builds must equal the planner's
    Topology.intra_group_size (pod_size * local_world_size). When it is narrower,
    TwRw places only the first `local_size` row shards of a longer plan and drops
    the rest, while the checkpoint metadata still advertises the whole tensor -- the
    model then trains on a partial embedding table and the gap only surfaces much
    later as a DCP "Global Plan Validation Failed" tensor-volume vs chunks-volume
    mismatch. These cases pin the width arithmetic directly, since reproducing it
    end to end needs an NVL72 job and a checkpoint save.
    """

    def setUp(self) -> None:
        # The builder memoizes its groups in module globals and skips rebuilding when
        # they are already set, so a value leaked from one case would satisfy the next
        # case's `if _INTRA_PG_2D is None` guard and silently skip the assertions.
        self._reset_module_globals()
        self.addCleanup(self._reset_module_globals)

    @staticmethod
    def _reset_module_globals() -> None:
        comm._INTRA_PG_2D = None
        comm._CROSS_PG_2D = None
        comm._NODE_GROUP_SIZE_2D = None

    @staticmethod
    def _fake_dist(world_size: int, sharding_group_size: int) -> MagicMock:
        """A dist double whose get_world_size distinguishes global from sharding-group."""
        fake = MagicMock()
        fake.get_backend.return_value = "fake_backend"
        fake.get_rank.return_value = 0
        # The builder calls get_world_size(env.sharding_pg) for the sharding group and
        # get_world_size() for the global world; the argument is what separates them.
        fake.get_world_size.side_effect = lambda group=None: (
            sharding_group_size if group is not None else world_size
        )
        fake.new_group.side_effect = lambda **kwargs: ("pg", tuple(kwargs["ranks"]))
        return fake

    def _build(
        self,
        world_size: int,
        sharding_group_size: int,
        node_group_size: Optional[int],
        pod_size: Optional[str],
        local_world_size: Optional[str] = None,
    ) -> MagicMock:
        env_vars: dict[str, str] = {}
        if pod_size is not None:
            env_vars["TORCHREC_RESOLVED_POD_SIZE"] = pod_size
        if local_world_size is not None:
            env_vars["LOCAL_WORLD_SIZE"] = local_world_size

        fake_dist = self._fake_dist(world_size, sharding_group_size)
        with patch.dict(os.environ, env_vars, clear=True), patch.object(
            comm, "dist", fake_dist
        ):
            comm.intra_and_cross_node_pg_2D(
                # pyrefly: ignore[bad-argument-type]
                _FakeShardingEnv2D(
                    sharding_pg=object(), node_group_size=node_group_size
                ),
            )
        return fake_dist

    def test_pod_size_widens_the_node_group(self) -> None:
        # The regression this fix exists for. On an NVL72 SKU one NVLink domain spans
        # pod_size hosts, so the planner cuts pod_size * node_group_size row shards.
        # Before the fix this width was node_group_size alone -- exactly pod_size too
        # narrow, which is the 2x volume ratio seen in the field.
        self._build(
            world_size=32,
            sharding_group_size=16,
            node_group_size=2,
            pod_size="2",
        )
        self.assertEqual(comm.get_node_group_size(), 4)

    def test_node_group_scales_with_each_pod_size(self) -> None:
        for pod_size, expected in (("1", 2), ("2", 4), ("4", 8)):
            with self.subTest(pod_size=pod_size):
                self._reset_module_globals()
                self._build(
                    world_size=64,
                    sharding_group_size=32,
                    node_group_size=2,
                    pod_size=pod_size,
                )
                self.assertEqual(comm.get_node_group_size(), expected)

    def test_unset_pod_size_leaves_the_width_unchanged(self) -> None:
        # Every SKU whose NVLink domain is a single host leaves this unset, so the
        # multiplier must be a no-op there rather than defaulting to something wider.
        self._build(
            world_size=32,
            sharding_group_size=16,
            node_group_size=2,
            pod_size=None,
        )
        self.assertEqual(comm.get_node_group_size(), 2)

    def test_falls_back_to_local_world_size_when_node_group_size_is_unset(self) -> None:
        # node_group_size=None routes through get_local_size(world_size), and pod_size
        # has to multiply that fallback too -- not just the explicit setting.
        self._build(
            world_size=32,
            sharding_group_size=16,
            node_group_size=None,
            pod_size="2",
            local_world_size="4",
        )
        self.assertEqual(comm.get_node_group_size(), 8)

    def test_intra_groups_are_exactly_one_node_wide(self) -> None:
        # The width is what TwRw._shard() iterates when placing row shards, so assert
        # on the rank lists actually handed to new_group, not only the cached int.
        fake_dist = self._build(
            world_size=32,
            sharding_group_size=16,
            node_group_size=2,
            pod_size="2",
        )
        intra_calls = [
            call.kwargs["ranks"]
            for call in fake_dist.new_group.call_args_list
            if call.kwargs["group_desc"] == "sharding_intra_pg"
        ]
        self.assertTrue(intra_calls)
        for ranks in intra_calls:
            self.assertEqual(len(ranks), 4)

    def test_non_tiling_topology_raises_before_building_any_group(self) -> None:
        # A width that does not divide the sharding group cannot place the plan. It has
        # to fail here, while both numbers are still in scope, rather than let the
        # truncating loop below build a malformed group.
        fake_dist = self._fake_dist(world_size=24, sharding_group_size=6)
        with patch.dict(
            os.environ, {"TORCHREC_RESOLVED_POD_SIZE": "2"}, clear=True
        ), patch.object(comm, "dist", fake_dist):
            with self.assertRaisesRegex(ValueError, "does not tile the sharding group"):
                comm.intra_and_cross_node_pg_2D(
                    # pyrefly: ignore[bad-argument-type]
                    _FakeShardingEnv2D(sharding_pg=object(), node_group_size=2),
                )
        fake_dist.new_group.assert_not_called()

    def test_non_tiling_failure_is_a_valueerror_not_an_assert(self) -> None:
        # Deliberately not an assert: under `python -O` a stripped assert would let a
        # non-tiling topology through and reintroduce the silent shard drop.
        fake_dist = self._fake_dist(world_size=24, sharding_group_size=6)
        with patch.dict(
            os.environ, {"TORCHREC_RESOLVED_POD_SIZE": "2"}, clear=True
        ), patch.object(comm, "dist", fake_dist):
            with self.assertRaises(ValueError) as ctx:
                comm.intra_and_cross_node_pg_2D(
                    # pyrefly: ignore[bad-argument-type]
                    _FakeShardingEnv2D(sharding_pg=object(), node_group_size=2),
                )
        self.assertNotIsInstance(ctx.exception, AssertionError)
        # The message has to name the operands, otherwise the failure gives no hint
        # about which of pod_size / node_group_size produced the bad width.
        message = str(ctx.exception)
        for expected in ("devices_per_node=4", "sharding_group_size=6", "pod_size=2"):
            self.assertIn(expected, message)

    def test_meta_device_short_circuits_before_touching_dist(self) -> None:
        # Meta-device construction has no process groups to build; TwRw mirrors the
        # width itself in that case, so this must stay a no-op.
        fake_dist = self._fake_dist(world_size=32, sharding_group_size=16)
        with patch.object(comm, "dist", fake_dist):
            intra, cross = comm.intra_and_cross_node_pg_2D(
                # pyrefly: ignore[bad-argument-type]
                _FakeShardingEnv2D(sharding_pg=object(), node_group_size=2),
                device=torch.device("meta"),
            )
        self.assertIsNone(intra)
        self.assertIsNone(cross)
        fake_dist.new_group.assert_not_called()


def _reference_recat(
    grad_output: torch.Tensor, dim_sum_per_rank: List[int]
) -> torch.Tensor:
    grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
    return torch.cat(
        [g.contiguous().view(-1) for g in grad_outputs_by_rank],
        dim=0,
    )


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestRecatPooledEmbeddingGradOut(unittest.TestCase):
    """
    `_recat_pooled_embedding_grad_out` has a fbgemm fast path behind
    `USE_FBGEMM_RECAT`. It is a pure permutation of the input values, so every
    path must agree bit for bit with the split/contiguous/cat reference.
    """

    def setUp(self) -> None:
        self._original: bool = comm_ops.get_use_fbgemm_recat()

    def tearDown(self) -> None:
        comm_ops.set_use_fbgemm_recat(self._original)

    def _make_case(
        self, b_local: int, dim_sum_per_rank: List[int], dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.randn(b_local, sum(dim_sum_per_rank), device="cuda", dtype=dtype)

    def _recat_tensors(
        self, dim_sum_per_rank: List[int], dtype: torch.dtype = torch.int64
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        cumsum = list(itertools.accumulate(dim_sum_per_rank))
        return (
            torch.tensor(dim_sum_per_rank, device="cuda", dtype=dtype),
            torch.tensor([0] + cumsum[:-1], device="cuda", dtype=dtype),
        )

    @given(
        b_local=st.sampled_from([1, 4, 128]),
        dtype=st.sampled_from([torch.float32, torch.float16]),
    )
    @settings(deadline=None, max_examples=6)
    def test_batch_kernel_matches_reference(
        self, b_local: int, dtype: torch.dtype
    ) -> None:
        dim_sum_per_rank = [8, 16, 4, 32, 12, 4, 20, 8]
        grad_output = self._make_case(b_local, dim_sum_per_rank, dtype)
        expected = _reference_recat(grad_output, dim_sum_per_rank)

        dims, cumsum = self._recat_tensors(dim_sum_per_rank)
        comm_ops.set_use_fbgemm_recat(True)
        actual = comm_ops._recat_pooled_embedding_grad_out(
            grad_output, dim_sum_per_rank, dims, cumsum
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mixed_d_kernel_matches_reference(self) -> None:
        # No recat tensors supplied -> the cudaMemcpy2D-per-rank fbgemm path.
        dim_sum_per_rank = [8, 16, 4, 32]
        grad_output = self._make_case(16, dim_sum_per_rank, torch.float32)
        expected = _reference_recat(grad_output, dim_sum_per_rank)

        comm_ops.set_use_fbgemm_recat(True)
        actual = comm_ops._recat_pooled_embedding_grad_out(
            grad_output, dim_sum_per_rank
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_flag_off_uses_reference(self) -> None:
        dim_sum_per_rank = [8, 16, 4, 32]
        grad_output = self._make_case(16, dim_sum_per_rank, torch.float32)
        dims, cumsum = self._recat_tensors(dim_sum_per_rank)

        comm_ops.set_use_fbgemm_recat(False)
        actual = comm_ops._recat_pooled_embedding_grad_out(
            grad_output, dim_sum_per_rank, dims, cumsum
        )
        torch.testing.assert_close(
            actual, _reference_recat(grad_output, dim_sum_per_rank), rtol=0, atol=0
        )

    def test_int32_tensors_fall_back(self) -> None:
        # The legacy All2AllPooledInfo tensors are int32 with an *inclusive*
        # cumsum. Feeding those to the batch kernel would silently shift every
        # rank's stripe, so the dtype guard must route them to the safe path.
        dim_sum_per_rank = [8, 16, 4, 32]
        grad_output = self._make_case(16, dim_sum_per_rank, torch.float32)
        dims, cumsum = self._recat_tensors(dim_sum_per_rank, dtype=torch.int32)

        comm_ops.set_use_fbgemm_recat(True)
        actual = comm_ops._recat_pooled_embedding_grad_out(
            grad_output, dim_sum_per_rank, dims, cumsum
        )
        torch.testing.assert_close(
            actual, _reference_recat(grad_output, dim_sum_per_rank), rtol=0, atol=0
        )

    def test_unsupported_dtype_falls_back(self) -> None:
        # fbgemm dispatches float and half only.
        dim_sum_per_rank = [8, 16, 4, 32]
        grad_output = self._make_case(16, dim_sum_per_rank, torch.bfloat16)
        dims, cumsum = self._recat_tensors(dim_sum_per_rank)

        comm_ops.set_use_fbgemm_recat(True)
        actual = comm_ops._recat_pooled_embedding_grad_out(
            grad_output, dim_sum_per_rank, dims, cumsum
        )
        torch.testing.assert_close(
            actual, _reference_recat(grad_output, dim_sum_per_rank), rtol=0, atol=0
        )

    def test_fbgemm_path_reports_api_usage(self) -> None:
        # torch dedupes the event itself; patching it out here only checks that
        # the fbgemm path is the one attributed.
        dim_sum_per_rank = [8, 16, 4, 32]
        grad_output = self._make_case(16, dim_sum_per_rank, torch.float32)
        dims, cumsum = self._recat_tensors(dim_sum_per_rank)

        comm_ops.set_use_fbgemm_recat(True)
        with patch.object(torch._C, "_log_api_usage_once") as log_api_usage_once:
            comm_ops._recat_pooled_embedding_grad_out(
                grad_output, dim_sum_per_rank, dims, cumsum
            )
        log_api_usage_once.assert_called_once_with(
            "torchrec.distributed.comm_ops.recat_pooled_embedding_grad_out."
            "recat_embedding_grad_output_mixed_D_batch"
        )

    def test_reference_path_reports_no_api_usage(self) -> None:
        dim_sum_per_rank = [8, 16, 4, 32]
        grad_output = self._make_case(16, dim_sum_per_rank, torch.float32)
        dims, cumsum = self._recat_tensors(dim_sum_per_rank)

        comm_ops.set_use_fbgemm_recat(False)
        with patch.object(torch._C, "_log_api_usage_once") as log_api_usage_once:
            comm_ops._recat_pooled_embedding_grad_out(
                grad_output, dim_sum_per_rank, dims, cumsum
            )
        log_api_usage_once.assert_not_called()
