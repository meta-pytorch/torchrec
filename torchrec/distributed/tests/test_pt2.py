#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import copy
import itertools
import sys
import unittest
from enum import auto, Enum
from typing import Any, Dict, List, Tuple

import torch
from fbgemm_gpu.permute_pooled_embedding_modules import PermutePooledEmbeddings
from fbgemm_gpu.permute_pooled_embedding_modules_split import (
    PermutePooledEmbeddingsSplit,
)
from fbgemm_gpu.split_embedding_utils import get_table_batched_offsets_from_dense
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    EmbeddingLocation,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from hypothesis import given, settings, strategies as st
from torch._dynamo.testing import reduce_to_scalar_loss
from torchrec.distributed.test_utils.infer_utils import (
    KJTInputExportDynamicShapeWrapper,
    KJTInputExportWrapperWithStrides,
    TestQuantFPEBCSharder,
)
from torchrec.pt2.utils import (
    deregister_fake_classes,
    kjt_for_pt2_tracing,
    register_fake_classes,
)
from torchrec.sparse.jagged_tensor import _kt_regroup_arguments

try:
    # pyre-ignore
    from caffe2.test.inductor.test_aot_inductor import AOTIRunnerUtil
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)


from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fused_params import FUSED_PARAM_BOUNDS_CHECK_MODE
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    create_test_model_ebc_only,
    KJTInputExportWrapper,
    prep_inputs,
    replace_registered_tbes_with_mock_tbes,
    replace_sharded_quant_modules_tbes_with_mock_tbes,
    TestQuantEBCSharder,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingEnv, ShardingType
from torchrec.sparse.jagged_tensor import (
    ComputeKJTToJTDict,
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)


def make_kjt(
    values: List[int], lengths: List[int], device: str = "cpu"
) -> KeyedJaggedTensor:
    values_tensor = torch.tensor(values, dtype=torch.int32, device=device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    weights_tensor = torch.randn(len(values), dtype=torch.float32, device=device)
    torch._check(torch.sum(lengths_tensor).item() == values_tensor.size(0))
    kjt = KeyedJaggedTensor(
        keys=[f"key{i}" for i in range(len(lengths))],
        values=values_tensor,
        lengths=lengths_tensor,
        weights=weights_tensor,
    )
    return kjt


def kjt_module_kjt_inputs_with_strides(kjt: KeyedJaggedTensor) -> Tuple:
    return (
        kjt._values,
        kjt._lengths,
        kjt._stride_per_key_per_rank,
    )


def _sharded_quant_ebc_model(
    local_device: str = "cuda",
    compute_device: str = "cuda",
    feature_processor: bool = False,
) -> Tuple[torch.nn.Module, List[KeyedJaggedTensor]]:
    num_embeddings = 256
    emb_dim = 12
    world_size = 2
    batch_size = 4

    local_device = torch.device(local_device)
    mi = create_test_model_ebc_only(
        num_embeddings,
        emb_dim,
        world_size,
        batch_size,
        num_features=2,
        num_weighted_features=1,
        dense_device=local_device,
        sparse_device=local_device,
        quant_state_dict_split_scale_bias=True,
        compute_device=compute_device,
        feature_processor=feature_processor,
    )
    input_kjts = [
        inp.to(local_device).idlist_features
        for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
    ]

    sharding_type: ShardingType = ShardingType.TABLE_WISE

    fused_params = {
        FUSED_PARAM_BOUNDS_CHECK_MODE: BoundsCheckMode.NONE,
    }
    if feature_processor:
        sharder = TestQuantFPEBCSharder(
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
            fused_params=fused_params,
        )
    else:
        sharder = TestQuantEBCSharder(
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
            fused_params=fused_params,
        )
    # pyre-ignore
    plan = mi.planner.plan(
        mi.quant_model,
        [sharder],
    )

    sharded_model = _shard_modules(
        module=mi.quant_model,
        # pyre-ignore
        sharders=[sharder],
        # Always shard on meta
        device=torch.device("meta"),
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )

    model: torch.nn.Module = KJTInputExportWrapper(sharded_model, input_kjts[0].keys())
    return model, input_kjts


class _TestType(Enum):
    EXPORT = auto()
    DYNAMO_COMPILE = auto()


# pyre-ignore
def _copy_input_tensors(t, device):
    if isinstance(t, torch.Tensor):
        ret = t.detach().clone().to(device)
        if ret.dtype in [torch.float, torch.double]:
            ret.requires_grad = True
            ret.retain_grad()
        return ret
    elif isinstance(t, (list, tuple)):
        return [_copy_input_tensors(_t, device) for _t in t]
    elif isinstance(t, int):
        return t
    else:
        raise ValueError(f"Unsupported type {type(t)}")


# pyre-ignore
def _grad_detach_clone(t):
    if isinstance(t, torch.Tensor):
        # pyre-ignore
        if t.grad is None:
            return None
        return t.grad.detach().clone()
    elif isinstance(t, (list, tuple)):
        return [_grad_detach_clone(_t) for _t in t]
    elif isinstance(t, int):
        return t
    else:
        raise ValueError(f"Unsupported type {type(t)}")


# pyre-ignore
def _assert_close(actual, expected) -> None:
    if actual is None and expected is None:
        return

    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        for _a, _e in zip(actual, expected):
            _assert_close(_a, _e)
    elif isinstance(expected, int):
        assert type(expected) is type(actual)
        assert expected == actual
    else:
        raise ValueError(f"Unsupported type {type(expected)}")


def _test_compile_fwd_bwd(
    fn,
    inp,
    device: torch.device,
    unpack_inp: bool = False,
    backend: str = "inductor",
    fullgraph: bool = True,
    skip_backward: bool = False,
    *args,
    **kwargs,
):
    eager_input = _copy_input_tensors(inp, device)
    compile_input = _copy_input_tensors(inp, device)

    if unpack_inp:
        eager_out = fn(*eager_input, *args, **kwargs)
    else:
        eager_out = fn(eager_input, *args, **kwargs)

    if not skip_backward:
        eager_loss = reduce_to_scalar_loss(eager_out)
        eager_loss.backward()
        eager_bwd_out = _grad_detach_clone(eager_input)

    with unittest.mock.patch(
        "torch._dynamo.config.skip_torchrec",
        False,
    ):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        if unpack_inp:
            compile_out = torch.compile(fn, backend=backend, fullgraph=fullgraph)(
                *compile_input
            )
        else:
            compile_out = torch.compile(fn, backend=backend, fullgraph=fullgraph)(
                compile_input
            )

        if not skip_backward:
            reduce_to_scalar_loss(compile_out).backward()
            compile_bwd_out = _grad_detach_clone(compile_input)

        _assert_close(compile_out, eager_out)
        if not skip_backward:
            _assert_close(compile_bwd_out, eager_bwd_out)


class TestPt2(unittest.TestCase):
    def setUp(self):
        super().setUp()
        register_fake_classes()

    def tearDown(self):
        deregister_fake_classes()
        super().tearDown()

    def _test_kjt_input_module(
        self,
        kjt_input_module: torch.nn.Module,
        kjt: KeyedJaggedTensor,
        inputs: Tuple[Any],
        test_dynamo: bool = True,
        test_aot_inductor: bool = True,
        test_pt2_ir_export: bool = False,
    ) -> None:
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            EM: torch.nn.Module = KJTInputExportWrapper(kjt_input_module, kjt.keys())
            em_inputs = (kjt.values(), kjt.lengths(), kjt.weights_or_none(), *inputs)
            eager_output = EM(*em_inputs)
            if test_dynamo:
                x = torch._dynamo.export(EM, same_signature=True)(*em_inputs)

                export_gm = x.graph_module
                export_gm_output = export_gm(*em_inputs)

                assert_close(eager_output, export_gm_output)

            if test_aot_inductor:
                # pyre-ignore
                so_path: str = AOTIRunnerUtil.compile(
                    EM,
                    inputs,
                )
                device = "cuda"
                # pyre-ignore
                aot_inductor_module = AOTIRunnerUtil.load(device, so_path)
                aot_actual_output = aot_inductor_module(*em_inputs)
                assert_close(eager_output, aot_actual_output)

            if test_pt2_ir_export:
                symint_wrapper = KJTInputExportDynamicShapeWrapper(EM)

                # KJTInputExportDynamicShapeWrapper represents sizes of values/weights
                # from first element of values/weights respectively (simulate symint)
                # Need to set as size in order to run a proper forward
                em_inputs[0][0] = kjt.values().size(0)
                em_inputs[2][0] = kjt.weights().size(0)

                if not kjt.values().is_meta:
                    eager_output = symint_wrapper(*em_inputs)

                pt2_ir = torch.export.export(
                    symint_wrapper, em_inputs, {}, strict=False
                )

                pt2_ir_output = pt2_ir.module()(*em_inputs)
                assert_close(eager_output, pt2_ir_output)

    # Separate test for Dynamo, as it fallbacks on VB path.
    # Torchrec has lazy init modules, depending on the first input => we need to run eager with tracing inputs.
    # But other test cases do not need to go VB.
    def _test_kjt_input_module_dynamo_compile(
        self,
        kjt_input_module: torch.nn.Module,
        kjt_keys: List[str],
        # pyre-ignore
        inputs,
        backend: str = "eager",
    ) -> None:
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            EM: torch.nn.Module = KJTInputExportWrapperWithStrides(
                kjt_input_module, kjt_keys
            )
            eager_output = EM(*inputs)
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            dynamo_eager_out = torch.compile(EM, backend=backend, fullgraph=True)(
                *inputs
            )
            assert_close(eager_output, dynamo_eager_out)

    @given(
        test_type_backend=st.sampled_from(
            [(_TestType.EXPORT, ""), (_TestType.DYNAMO_COMPILE, "aot_eager")]
        )
    )
    @settings(deadline=None)
    def test_kjt_split(self, test_type_backend: Tuple[_TestType, str]) -> None:
        test_type, backend = test_type_backend

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.split([1, 2, 1])

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        if test_type == _TestType.EXPORT:
            self._test_kjt_input_module(
                M(),
                kjt,
                (),
                test_aot_inductor=False,
                test_dynamo=False,
                test_pt2_ir_export=True,
            )
        elif test_type == _TestType.DYNAMO_COMPILE:
            self._test_kjt_input_module_dynamo_compile(
                M(),
                kjt.keys(),
                kjt_module_kjt_inputs_with_strides(kjt_for_pt2_tracing(kjt)),
                backend=backend,
            )

    @given(
        test_type_backend=st.sampled_from(
            [(_TestType.EXPORT, ""), (_TestType.DYNAMO_COMPILE, "aot_eager")]
        )
    )
    @settings(deadline=None)
    def test_kjt_permute(self, test_type_backend: Tuple[_TestType, str]) -> None:
        test_type, backend = test_type_backend

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, indices: List[int]):
                return kjt.permute(indices)

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        indices: List[int] = [1, 0, 3, 2]

        if test_type == _TestType.EXPORT:
            self._test_kjt_input_module(
                M(),
                kjt,
                (indices,),
                test_aot_inductor=False,
                test_pt2_ir_export=True,
            )
        elif test_type == _TestType.DYNAMO_COMPILE:

            def inputs_fn(kjt):
                return *kjt_module_kjt_inputs_with_strides(kjt), indices

            self._test_kjt_input_module_dynamo_compile(
                M(),
                kjt.keys(),
                inputs_fn(kjt_for_pt2_tracing(kjt)),
                backend=backend,
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_kt_regroup_as_dict(
        self,
    ) -> None:

        class M(torch.nn.Module):
            def forward(self, inputs: List[KeyedTensor]) -> Dict[str, torch.Tensor]:
                groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
                keys = ["group_0", "group_1"]
                return KeyedTensor.regroup_as_dict(inputs, groups, keys)

        m = M()

        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        inputs = [kt_1, kt_2]

        for t in itertools.chain(tensor_list_1, tensor_list_2):
            torch._dynamo.decorators.mark_dynamic(t, 0)
            torch._dynamo.decorators.mark_dynamic(t, 1)

        eager_output = m(inputs)
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            torch_compile_backend = "eager"

            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            opt_fn = torch.compile(
                m, backend=torch_compile_backend, fullgraph=True, dynamic=True
            )
            compile_output = opt_fn(inputs)
            torch.testing.assert_close(eager_output, compile_output)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_kjt_permute_dynamo_compile(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, indices: List[int]):
                return kjt.permute(indices)

        device = "cuda"
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1], device=device)
        indices: List[int] = [1, 0, 3, 2]
        # pyre-ignore
        inputs_fn = lambda kjt: (
            *kjt_module_kjt_inputs_with_strides(kjt),
            indices,
        )
        self._test_kjt_input_module_dynamo_compile(
            M(),
            kjt.keys(),
            inputs_fn(kjt_for_pt2_tracing(kjt)),
            backend="inductor",
        )

    def test_kjt_length_per_key(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.length_per_key()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module(
            M(),
            kjt,
            (),
            test_aot_inductor=False,
            test_pt2_ir_export=True,
        )

    def test_kjt_length_per_key_meta(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.length_per_key()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        kjt = kjt.to("meta")

        # calling forward on meta inputs once traced should error out
        # as calculating length_per_key requires a .tolist() call of lengths
        self.assertRaisesRegex(
            RuntimeError,
            r".*Tensor\.item\(\) cannot be called on meta tensors.*",
            lambda: self._test_kjt_input_module(
                M(),
                kjt,
                (),
                test_aot_inductor=False,
                test_pt2_ir_export=True,
            ),
        )

    def test_kjt_offset_per_key(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.offset_per_key()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module(
            M(),
            kjt,
            (),
            test_aot_inductor=False,
            test_pt2_ir_export=True,
        )

    @given(
        test_type_backend=st.sampled_from(
            [(_TestType.EXPORT, ""), (_TestType.DYNAMO_COMPILE, "aot_eager")]
        )
    )
    @settings(deadline=None)
    def test_kjt__getitem__(self, test_type_backend: Tuple[_TestType, str]) -> None:
        test_type, backend = test_type_backend

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                out0 = kjt["key0"]
                out1 = kjt["key1"]

                return out0, out1

        # First element represents symint for values and weights shape
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        if test_type == _TestType.EXPORT:
            self._test_kjt_input_module(
                M(),
                kjt,
                (),
                test_dynamo=False,
                test_aot_inductor=False,
                test_pt2_ir_export=True,
            )
        elif test_type == _TestType.DYNAMO_COMPILE:
            self._test_kjt_input_module_dynamo_compile(
                M(),
                kjt.keys(),
                kjt_module_kjt_inputs_with_strides(kjt_for_pt2_tracing(kjt)),
                backend=backend,
            )

    def test_kjt_to_dict_with_strides_dynamo(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
                return kjt.to_dict()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module_dynamo_compile(
            M(),
            kjt.keys(),
            kjt_module_kjt_inputs_with_strides(kjt_for_pt2_tracing(kjt)),
        )

    def test_sharded_quant_ebc_non_strict_export(self) -> None:
        sharded_model, input_kjts = _sharded_quant_ebc_model(
            local_device="cpu", compute_device="cpu"
        )
        kjt = input_kjts[0]
        sharded_model(kjt.values(), kjt.lengths())

        from torch.export import _trace

        ep = _trace._export(
            sharded_model,
            (
                kjt.values(),
                kjt.lengths(),
            ),
            {},
            strict=False,
            pre_dispatch=True,
        )

        ep.module()(kjt.values(), kjt.lengths())

        # PT2 IR autofunctionalizes mutation funcs (bounds_check_indices)
        # ensure such node isn't present, as it causes issues with IR
        for n in ep.graph_module.graph.nodes:
            self.assertFalse("auto_functionalized" in str(n.name))

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_sharded_quant_fpebc_non_strict_export(self) -> None:
        sharded_model, input_kjts = _sharded_quant_ebc_model(
            local_device="cpu", compute_device="cpu", feature_processor=True
        )
        kjt = input_kjts[0]

        sharded_model(kjt.values(), kjt.lengths())

        from torch.export import _trace

        ep = _trace._export(
            sharded_model,
            (
                kjt.values(),
                kjt.lengths(),
            ),
            {},
            strict=False,
            pre_dispatch=True,
        )
        ep.module()(kjt.values(), kjt.lengths())

        # PT2 IR autofunctionalizes mutation funcs (bounds_check_indices)
        # ensure such node isn't present, as it causes issues with IR
        for n in ep.graph_module.graph.nodes:
            self.assertFalse("auto_functionalized" in str(n.name))

    def test_maybe_compute_kjt_to_jt_dict(self) -> None:
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        self._test_kjt_input_module(
            ComputeKJTToJTDict(),
            kjt,
            (),
            # TODO: turn on AOT Inductor test once the support is ready
            test_aot_inductor=False,
        )

    def test_kjt_values_specialization(self):
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            from torch._dynamo.testing import CompileCounter

            kjt0 = KeyedJaggedTensor(
                values=torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.int64),
                keys=["f0", "f1", "f2"],
                lengths=torch.tensor([0, 0, 1, 1, 2, 2]),
                stride=2,
            )
            torch._dynamo.decorators.mark_unbacked(kjt0._values, 0)

            counter = CompileCounter()

            @torch._dynamo.optimize(counter, nopython=True)
            def f(kjt):
                l: List[KeyedJaggedTensor] = kjt.split([1, 1, 1])
                return l[0].values().sum() + l[1].values().sum() + l[2].values().sum()

            f(kjt0)
            self.assertEqual(counter.frame_count, 1)

            kjt1 = KeyedJaggedTensor(
                values=torch.tensor([], dtype=torch.int64),
                keys=["f0", "f1", "f2"],
                lengths=torch.tensor([0, 0, 0, 0, 0, 0]),
                stride=2,
            )
            torch._dynamo.decorators.mark_unbacked(kjt1._values, 0)
            f(kjt1)
            self.assertEqual(counter.frame_count, 1)

    def test_kjt_values_specialization_utils(self):
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            from torch._dynamo.testing import CompileCounter

            kjt0 = KeyedJaggedTensor(
                values=torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.int64),
                keys=["f0", "f1", "f2"],
                lengths=torch.tensor([0, 0, 1, 1, 2, 2]),
                stride=2,
            ).sync()

            counter = CompileCounter()

            @torch._dynamo.optimize(counter, nopython=True)
            def f(kjt):
                l: List[KeyedJaggedTensor] = kjt.split([1, 1, 1])
                return l[0].values().sum() + l[1].values().sum() + l[2].values().sum()

            f(kjt_for_pt2_tracing(kjt0))
            self.assertEqual(counter.frame_count, 1)

            kjt1 = KeyedJaggedTensor(
                values=torch.tensor([], dtype=torch.int64),
                keys=["f0", "f1", "f2"],
                lengths=torch.tensor([0, 0, 0, 0, 0, 0]),
                stride=2,
            ).sync()
            f(kjt_for_pt2_tracing(kjt1))
            self.assertEqual(counter.frame_count, 1)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_ebc_vb_reindex(self) -> None:
        device = "cuda"

        def fn(
            embs: torch.Tensor,
            indices: torch.Tensor,
            input_num_indices: List[int],
            input_rows: List[int],
            input_columns: List[int],
        ):
            reindex_output = torch.ops.fbgemm.batch_index_select_dim0_tensor(
                inputs=embs,
                indices=indices.view(-1),
                input_num_indices=torch.tensor(input_num_indices, dtype=torch.int64),
                input_rows=torch.tensor(input_rows, dtype=torch.int64),
                input_columns=torch.tensor(input_columns, dtype=torch.int64),
                permute_output_dim_0_1=True,
            )
            return reindex_output

        N = 5
        batch_size = 10
        emb_dim = 12
        embs: torch.Tensor = torch.randn(
            [N * batch_size * emb_dim], device=device, requires_grad=True
        )
        torch._dynamo.mark_dynamic(embs, 0)
        input_num_indices = [batch_size] * N
        input_rows = [batch_size] * N
        input_columns = [emb_dim] * N
        indices: torch.Tensor = (
            torch.arange(batch_size)
            .expand(N, batch_size)
            .contiguous()
            .to(device=device)
        )

        ins = (embs, indices, input_num_indices, input_rows, input_columns)
        _test_compile_fwd_bwd(fn, ins, device, unpack_inp=True)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_permute_pooled_embs(self) -> None:
        device = "cuda"
        m = PermutePooledEmbeddings(
            embs_dims=[12, 12, 12],
            permute=[2, 1, 0],
        )
        inp = torch.randn(12, 3)
        _test_compile_fwd_bwd(m, inp, device, backend="aot_eager")

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_permute_pooled_embs_split(self) -> None:
        device = "cuda"
        m = PermutePooledEmbeddingsSplit(
            embs_dims=[12, 12, 12],
            permute=[2, 1, 0],
        )
        inp = torch.randn(12, 3)
        _test_compile_fwd_bwd(m, inp, device)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_permute_multi_embedding(self) -> None:
        device = "cuda"
        batch_size = 16

        def func(values, permutes, in_shapes, out_shapes, out_lengths):
            return torch.ops.fbgemm.permute_multi_embedding(
                values, permutes, in_shapes, out_shapes, out_lengths.tolist()
            )

        keys = [["f1", "f2"], ["f3", "f4", "f5"], ["f6"]]
        lengths = [[3, 4], [5, 6, 7], [8]]
        groups = [["f1", "f3"], ["f2"], ["f4", "f1", "f6"], ["f1", "f5"]]
        values = [torch.randn(batch_size, sum(L), device=device) for L in lengths]
        for embs in values:
            torch._dynamo.mark_dynamic(embs, 0)
            torch._dynamo.mark_dynamic(embs, 1)
        permutes, in_shapes, out_shapes, out_lengths = _kt_regroup_arguments(
            values[0], keys, lengths, groups
        )
        out_lengths = torch.tensor(out_lengths, device=device, dtype=torch.int32)
        inp = (values, permutes, in_shapes, out_shapes, out_lengths)
        _test_compile_fwd_bwd(func, inp, device, unpack_inp=True)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_tbe_compile(self) -> None:
        D = 4
        T = 2
        E = 10
        Ds = [D] * T
        Es = [E] * T

        device = "cuda"
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    (
                        EmbeddingLocation.MANAGED
                        if device == "cuda"
                        else EmbeddingLocation.HOST
                    ),
                    ComputeDevice.CUDA if device == "cuda" else ComputeDevice.CPU,
                )
                for (E, D) in zip(Es, Ds)
            ],
        )
        tbe.init_embedding_weights_uniform(0, 1)

        class M(torch.nn.Module):
            def __init__(self, tbe) -> None:
                super().__init__()
                self.tbe = tbe

            def forward(self, indices, offsets, f) -> torch.Tensor:
                e = self.tbe(indices, offsets)
                return torch.mul(torch.mean(e, dim=1), f)

        m = M(tbe)
        m.train(True)
        m_compile = copy.deepcopy(m)
        m_compile.train(True)

        def get_weights(m):
            return m.tbe.weights_uvm.clone().detach()

        original_weights = get_weights(m)

        x = torch.Tensor(
            [
                [
                    [1],
                    [1],
                ],
                [[3], [4]],
            ]
        ).to(dtype=torch.int64, device=device)
        (indices, offsets) = get_table_batched_offsets_from_dense(
            x, use_cpu=device == "cpu"
        )
        inp_f = torch.randn(T, requires_grad=True, device=device)

        # EAGER
        out = m(indices, offsets, inp_f.clone())
        reduce_to_scalar_loss(out).backward()
        eager_weights_diff = get_weights(m) - original_weights

        # COMPILE
        orig_compile_weights = get_weights(m_compile)
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            compile_out = torch.compile(m_compile, backend="aot_eager", fullgraph=True)(
                indices, offsets, inp_f.clone()
            )
            reduce_to_scalar_loss(compile_out).backward()
            compile_weights_diff = get_weights(m_compile) - orig_compile_weights

            assert_close(eager_weights_diff, compile_weights_diff)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_tbe_compile_vb(self) -> None:
        D = 4
        T = 2
        E = 10
        Ds = [D] * T
        Es = [E] * T

        device = "cuda"
        tbe = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    (
                        EmbeddingLocation.MANAGED
                        if device == "cuda"
                        else EmbeddingLocation.HOST
                    ),
                    ComputeDevice.CUDA if device == "cuda" else ComputeDevice.CPU,
                )
                for (E, D) in zip(Es, Ds)
            ],
        )
        tbe.init_embedding_weights_uniform(0, 1)

        class M(torch.nn.Module):
            def __init__(self, tbe) -> None:
                super().__init__()
                self.tbe = tbe

            def forward(
                self, indices, offsets, batch_size_per_feature_per_rank, f
            ) -> torch.Tensor:
                e = self.tbe(
                    indices,
                    offsets,
                    batch_size_per_feature_per_rank=batch_size_per_feature_per_rank,
                )
                return torch.mul(torch.mean(e, dim=0), f)

        m = M(tbe)
        m.train(True)
        m_compile = copy.deepcopy(m)
        m_compile.train(True)

        def get_weights(m):
            return m.tbe.weights_uvm.clone().detach()

        original_weights = get_weights(m)

        indices = torch.Tensor([1, 2, 0, 1, 2]).to(dtype=torch.int64, device=device)
        lengths = torch.Tensor([2, 3]).to(dtype=torch.int64, device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        batch_size_per_feature_per_rank = [[1], [2]]
        inp_f = torch.randn(1, requires_grad=True, device=device)

        # EAGER
        out = m(indices, offsets, batch_size_per_feature_per_rank, inp_f.clone())
        reduce_to_scalar_loss(out).backward()
        eager_weights_diff = get_weights(m) - original_weights

        # COMPILE
        orig_compile_weights = get_weights(m_compile)
        with unittest.mock.patch(
            "torch._dynamo.config.skip_torchrec",
            False,
        ):
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            compile_out = torch.compile(m_compile, backend="aot_eager", fullgraph=True)(
                indices, offsets, batch_size_per_feature_per_rank, inp_f.clone()
            )
            reduce_to_scalar_loss(compile_out).backward()
            compile_weights_diff = get_weights(m_compile) - orig_compile_weights

            assert_close(eager_weights_diff, compile_weights_diff)
