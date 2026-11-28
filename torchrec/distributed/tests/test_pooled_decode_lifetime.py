#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torchrec.distributed.comm_ops import (
    All2All_Pooled_Wait,
    All2AllPooledInfo,
    Request,
    Variable_Batch_All2All_Pooled_Wait,
    VariableBatchAll2AllPooledInfo,
)
from torchrec.distributed.dist_data import PooledEmbeddingsAwaitable
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionAwaitable,
    VariableBatchEmbeddingBagCollectionAwaitable,
)
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)
from torchrec.distributed.types import LazyAwaitable, NoWait
from torchrec.sparse.jagged_tensor import KeyedTensor
from torchrec.streamable import Multistreamable


class PooledDecodeLifetimeTest(unittest.TestCase):
    """Exercise pooled consumers after completed transport releases its receive."""

    def _request(
        self,
        receive: torch.Tensor,
        rows: int,
        variable: bool,
        decode: bool,
        precision: CommType = CommType.FP16,
    ) -> Request[torch.Tensor]:
        pg = cast(dist.ProcessGroup, MagicMock(spec=dist.ProcessGroup))
        request = Request[torch.Tensor](pg, receive.device)
        request.req = cast(dist.Work, MagicMock(spec=dist.Work))
        request.tensor = receive
        codecs = (
            get_qcomm_codecs(QCommsConfig(forward_precision=precision))
            if decode
            else None
        )
        if variable:
            request.a2ai = VariableBatchAll2AllPooledInfo(
                [[rows, rows]],
                [rows, rows],
                [[4, 8]],
                codecs=codecs,
                output_splits=[rows * 4, rows * 8],
            )
            request.wait_function = Variable_Batch_All2All_Pooled_Wait
        else:
            request.a2ai = All2AllPooledInfo([rows], [12], None, None, codecs=codecs)
            request.wait_function = All2All_Pooled_Wait
        return request

    def _output_awaitable(
        self, request: Request[torch.Tensor], rows: int, variable: bool
    ) -> EmbeddingBagCollectionAwaitable | VariableBatchEmbeddingBagCollectionAwaitable:
        pooled = PooledEmbeddingsAwaitable(request)
        if variable:
            return VariableBatchEmbeddingBagCollectionAwaitable(
                awaitables=[pooled],
                inverse_indices=(
                    ["first", "second"],
                    torch.arange(rows, device=request.dummy_tensor.device).repeat(2, 1),
                ),
                inverse_indices_permute_indices=None,
                batch_size_per_feature_pre_a2a=[rows, rows],
                uncombined_embedding_dims=[4, 8],
                embedding_names=["first", "second"],
                embedding_dims=[4, 8],
                permute_op=lambda tensor: tensor,
            )
        return EmbeddingBagCollectionAwaitable([pooled], [12], ["embedding"])

    def test_record_stream_does_not_wait(self) -> None:
        for variable in (False, True):
            with self.subTest(variable=variable):
                request = self._request(torch.ones(12), 1, variable, False)
                receive = MagicMock(spec=torch.Tensor)
                receive.device = torch.device("cuda")
                request.tensor = receive
                stream = MagicMock(spec=torch.Stream)
                awaitable = self._output_awaitable(request, 1, variable)
                self.assertIsInstance(awaitable, Multistreamable)
                with patch.object(request, "wait") as wait:
                    awaitable.record_stream(stream)
                receive.record_stream.assert_called_once_with(stream)
                wait.assert_not_called()
                self.assertIs(request.tensor, receive)
                self.assertIsNone(awaitable._result)
                request.tensor = None
                request.record_stream(stream)
                receive.record_stream.assert_called_once_with(stream)

    def test_record_stream_after_wait_records_result(self) -> None:
        for variable in (False, True):
            for lazy in (False, True):
                with self.subTest(variable=variable, lazy=lazy):
                    request = self._request(torch.ones(12), 1, variable, False)
                    awaitable = self._output_awaitable(request, 1, variable)
                    result = MagicMock(spec=KeyedTensor)
                    result.values.return_value.device = torch.device("cuda")
                    callback_result = MagicMock(spec=KeyedTensor)
                    callback_result.values.return_value.device = torch.device("cuda")
                    awaitable.callbacks.append(MagicMock(return_value=callback_result))
                    with patch.object(awaitable, "_wait_impl", return_value=result):
                        if lazy:
                            self.assertIs(
                                LazyAwaitable._wait_async(awaitable), callback_result
                            )
                        else:
                            self.assertIs(awaitable.wait(), callback_result)
                    stream = MagicMock(spec=torch.Stream)
                    with patch.object(request, "record_stream") as record:
                        awaitable.record_stream(stream)
                    callback_result.values.return_value.record_stream.assert_called_once_with(
                        stream
                    )
                    result.values.return_value.record_stream.assert_not_called()
                    record.assert_not_called()

    def test_cpu_and_non_streamable_children(self) -> None:
        request = self._request(torch.ones(12), 1, False, False)
        awaitable = EmbeddingBagCollectionAwaitable(
            [
                PooledEmbeddingsAwaitable(request),
                PooledEmbeddingsAwaitable(NoWait(torch.ones(12))),
            ],
            [12, 12],
            ["first", "second"],
        )
        with patch.object(torch.Tensor, "record_stream") as record:
            awaitable.record_stream(MagicMock(spec=torch.Stream))
        record.assert_not_called()

    def test_cpu_numeric_outputs(self) -> None:
        for variable in (False, True):
            for precision in (None, CommType.FP16, CommType.BF16, CommType.FP32):
                with self.subTest(variable=variable, precision=precision):
                    dtype = {
                        None: torch.float32,
                        CommType.FP16: torch.float16,
                        CommType.BF16: torch.bfloat16,
                        CommType.FP32: torch.float32,
                    }[precision]
                    request = self._request(
                        torch.full((320 * 12,), 0.25, dtype=dtype),
                        320,
                        variable,
                        precision is not None,
                        precision or CommType.FP16,
                    )
                    with patch("torch.distributed.get_rank", return_value=0):
                        awaitable = self._output_awaitable(request, 320, variable)
                        awaitable.record_stream(MagicMock(spec=torch.Stream))
                        output = awaitable.wait().values()
                    shape = (320, 12)
                    self.assertEqual(output.dtype, torch.float32)
                    torch.testing.assert_close(output, torch.full(shape, 0.25))
                    self.assertIsNone(request.tensor)

    def _warm_consumer(
        self,
        stream: torch.cuda.Stream,
        rows: int,
        variable: bool,
        decode: bool,
        precision: CommType = CommType.FP16,
    ) -> None:
        dtype = (
            {
                CommType.FP16: torch.float16,
                CommType.BF16: torch.bfloat16,
                CommType.FP32: torch.float32,
            }[precision]
            if decode
            else torch.float32
        )
        with torch.cuda.stream(stream):
            request = self._request(
                torch.full((rows * 12,), 0.25, dtype=dtype, device="cuda"),
                rows,
                variable,
                decode,
                precision,
            )
            with patch("torch.distributed.get_rank", return_value=0):
                output = self._output_awaitable(request, rows, variable).wait().values()
        stream.synchronize()
        torch.testing.assert_close(output.cpu(), torch.full(output.shape, 0.25))

    def _reuse_trial(
        self,
        rows: int,
        variable: bool,
        decode: bool,
        enabled: bool,
        precision: CommType = CommType.FP16,
    ) -> tuple[bool, bool]:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        self._warm_consumer(consumer, rows, variable, decode, precision)
        dtype = (
            {
                CommType.FP16: torch.float16,
                CommType.BF16: torch.bfloat16,
                CommType.FP32: torch.float32,
            }[precision]
            if decode
            else torch.float32
        )
        with torch.cuda.stream(producer):
            request = self._request(
                torch.full((rows * 12,), 0.25, dtype=dtype, device="cuda"),
                rows,
                variable,
                decode,
                precision,
            )
        producer.synchronize()
        receive = request.tensor
        assert receive is not None
        pointer = receive.data_ptr()
        consumed = torch.cuda.Event()
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000_000)
            awaitable = self._output_awaitable(request, rows, variable)
            if enabled:
                awaitable.record_stream(consumer)
            with patch("torch.distributed.get_rank", return_value=0):
                output = awaitable.wait().values()
            consumed.record()
        del receive
        pending = not consumed.query()
        with torch.cuda.stream(producer):
            replacement = torch.full(
                (rows * 12,), float("nan"), dtype=dtype, device="cuda"
            )
        reused = replacement.data_ptr() == pointer
        torch.cuda.synchronize()
        observed = output.cpu()
        clean = bool(torch.isfinite(observed).all()) and bool((observed == 0.25).all())
        self.assertTrue(pending, "The fixture did not establish an unfinished consumer")
        self.assertIsNone(request.tensor)
        return reused, clean

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
    def test_decode_with_and_without_recording(self) -> None:
        self.assertEqual(torch.cuda.memory.get_allocator_backend(), "native")
        for precision in (CommType.FP16, CommType.BF16):
            for rows in (320, 640, 960):
                for variable in (False, True):
                    with self.subTest(
                        precision=precision, rows=rows, variable=variable
                    ):
                        reused, clean = self._reuse_trial(
                            rows, variable, True, False, precision=precision
                        )
                        self.assertTrue(
                            reused, "Negative control did not establish reuse"
                        )
                        self.assertFalse(
                            clean, "Negative control did not expose corruption"
                        )
                        reused, clean = self._reuse_trial(
                            rows, variable, True, True, precision=precision
                        )
                        self.assertFalse(reused)
                        self.assertTrue(clean)

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
    def test_dtype_preserving_decode_outputs(self) -> None:
        for variable in (False, True):
            with self.subTest(variable=variable):
                request = self._request(
                    torch.full((320 * 12,), 0.25, device="cuda"),
                    320,
                    variable,
                    True,
                    CommType.FP32,
                )
                with patch("torch.distributed.get_rank", return_value=0):
                    output = request.wait()
                shape = (320 * 12,) if variable else (320, 12)
                self.assertEqual(output.dtype, torch.float32)
                torch.testing.assert_close(output.cpu(), torch.full(shape, 0.25))
                self.assertIsNone(request.tensor)

    @unittest.skipIf(not torch.cuda.is_available(), "requires a CUDA device")
    def test_no_decode_fixed_pooling_with_and_without_recording(self) -> None:
        reused, clean = self._reuse_trial(960, False, False, False)
        self.assertTrue(reused)
        self.assertFalse(clean)
        reused, clean = self._reuse_trial(960, False, False, True)
        self.assertFalse(reused)
        self.assertTrue(clean)
