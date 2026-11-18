#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import sys
import types
import unittest
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, List

import torch
from torchrec.distributed.train_pipeline import batch_attribution
from torchrec.distributed.train_pipeline.batch_attribution import (
    Attributed,
    batch_operation,
    get_batch_attribution,
    is_attributed,
    to_attributed,
    unwrap_attributed,
)


@dataclass
class _Envelope:
    batch: object
    batch_metadata: object


def _load_second_copy() -> types.ModuleType:
    """A separate copy of the module, the way torch.package loads its own."""
    name = "batch_attribution_second_copy"
    module = types.ModuleType(name)
    sys.modules[name] = module  # dataclasses resolves ClassVar through sys.modules
    exec(compile(inspect.getsource(batch_attribution), name, "exec"), module.__dict__)
    return module


class BatchOperationTest(unittest.TestCase):
    def test_failure_keeps_the_original_exception(self) -> None:
        metadata = [b"partition-a"]
        error = ValueError("bad batch")

        @batch_operation("forward")
        def forward(batch: object) -> None:
            raise error

        with self.assertRaises(ValueError) as raised:
            forward(Attributed(object(), metadata))

        self.assertIs(raised.exception, error)
        self.assertEqual("bad batch", str(raised.exception))
        self.assertEqual(
            {"operation": "forward", "metadata": metadata},
            get_batch_attribution(raised.exception),
        )

    def test_raw_owner_is_unchanged(self) -> None:
        raw = object()
        received: List[object] = []

        @batch_operation("forward", propagate=True)
        def forward(batch: object) -> object:
            received.append(batch)
            raise RuntimeError("raw failure")

        with self.assertRaises(RuntimeError) as raised:
            forward(raw)

        self.assertEqual([raw], received)
        self.assertIsNone(get_batch_attribution(raised.exception))

    def test_unwraps_the_owner_parameter(self) -> None:
        raw = object()
        received: List[object] = []

        class Pipeline:
            @batch_operation("input_dist", owner="batch")
            def start(self, batch: object, context: object) -> None:
                received.append(batch)

        pipeline = Pipeline()
        pipeline.start(Attributed(raw, "m"), object())
        pipeline.start(batch=Attributed(raw, "m"), context=object())

        self.assertEqual([raw, raw], received)

    def test_propagate_pairs_the_result_with_the_metadata(self) -> None:
        metadata = object()

        @batch_operation("copy", propagate=True)
        def copy(batch: Any) -> Any:
            return None if batch is None else batch + "-copied"

        result = copy(Attributed("b", metadata))
        self.assertTrue(is_attributed(result))
        self.assertEqual("b-copied", unwrap_attributed(result))
        self.assertIs(metadata, result.metadata)
        self.assertIsNone(copy(Attributed(None, metadata)))
        self.assertEqual("b-copied", copy("b"))

    def test_without_propagate_the_result_stays_raw(self) -> None:
        @batch_operation("forward")
        def forward(batch: Any) -> Any:
            return batch + "-out"

        self.assertEqual("b-out", forward(Attributed("b", "m")))

    def test_innermost_operation_wins(self) -> None:
        @batch_operation("input_dist")
        def inner(batch: object) -> None:
            raise RuntimeError("inner")

        @batch_operation("forward", owner="current")
        def outer(current: object, next_batch: object) -> None:
            inner(next_batch)

        with self.assertRaises(RuntimeError) as raised:
            outer(Attributed(1, "current-batch"), Attributed(2, "next-batch"))

        self.assertEqual(
            {"operation": "input_dist", "metadata": "next-batch"},
            get_batch_attribution(raised.exception),
        )

    def test_stop_iteration_passes_through_unannotated(self) -> None:
        @batch_operation("load")
        def load(batch: object) -> None:
            raise StopIteration

        with self.assertRaises(StopIteration) as raised:
            load(Attributed(object(), "m"))

        self.assertIsNone(get_batch_attribution(raised.exception))

    def test_callable_owner_is_resolved_before_the_call(self) -> None:
        class Pipeline:
            def __init__(self) -> None:
                self.batches = [Attributed(1, "retiring"), Attributed(2, "next")]

            @batch_operation("dequeue", owner=lambda self: self.batches[0])
            def dequeue(self) -> None:
                self.batches.pop(0)
                raise RuntimeError("after the pop")

        with self.assertRaises(RuntimeError) as raised:
            Pipeline().dequeue()

        attribution = get_batch_attribution(raised.exception)
        self.assertIsNotNone(attribution)
        self.assertEqual("retiring", attribution["metadata"])

    def test_keyword_only_owner(self) -> None:
        received: List[object] = []

        @batch_operation("forward")
        def forward(*, batch: object) -> None:
            received.append(batch)

        forward(batch=Attributed("raw", "m"))
        self.assertEqual(["raw"], received)

    def test_exception_that_rejects_attributes_still_propagates(self) -> None:
        class Frozen(Exception):
            def __setattr__(self, name: str, value: Any) -> None:
                raise AttributeError("frozen")

        error = Frozen("x")

        @batch_operation("forward")
        def forward(batch: object) -> None:
            raise error

        with self.assertRaises(Frozen) as raised:
            forward(Attributed(object(), "m"))

        self.assertIs(raised.exception, error)
        self.assertIsNone(get_batch_attribution(raised.exception))

    def test_reader_walks_the_cause_chain(self) -> None:
        @batch_operation("copy")
        def copy(batch: object) -> None:
            raise RuntimeError("inner")

        attribution = None
        try:
            try:
                copy(Attributed(object(), "m"))
            except RuntimeError as inner:
                raise KeyError("outer") from inner
        except KeyError as outer:
            attribution = get_batch_attribution(outer)

        self.assertEqual({"operation": "copy", "metadata": "m"}, attribution)

    def test_rejects_unknown_owner_and_async_functions(self) -> None:
        def no_batch(x: object) -> None:
            pass

        async def coroutine(batch: object) -> None:
            pass

        def generator(batch: object) -> Iterator[object]:
            yield batch

        async def async_generator(batch: object) -> AsyncIterator[object]:
            yield batch

        with self.assertRaisesRegex(TypeError, "no parameter named 'batch'"):
            batch_operation("forward")(no_batch)
        for func in (coroutine, generator, async_generator):
            with self.subTest(func=func.__name__):
                with self.assertRaisesRegex(TypeError, "synchronous"):
                    batch_operation("forward")(func)


class CarrierTest(unittest.TestCase):
    def test_to_attributed_converts_envelopes_only(self) -> None:
        batch, metadata = object(), [b"p"]
        converted = to_attributed(_Envelope(batch, metadata))
        carrier = Attributed(object(), "m")
        raw = object()

        self.assertTrue(is_attributed(converted))
        self.assertIs(batch, converted.value)
        self.assertIs(metadata, converted.metadata)
        self.assertIs(carrier, to_attributed(carrier))
        self.assertIs(raw, to_attributed(raw))

    def test_carriers_compare_by_identity(self) -> None:
        tensor = torch.ones(2)
        self.assertNotEqual(Attributed(tensor, "m"), Attributed(tensor, "m"))

    def test_real_tensor_copy_keeps_the_metadata(self) -> None:
        metadata = [b"p"]

        @batch_operation("copy", propagate=True)
        def copy(batch: Any, device: torch.device) -> Any:
            return batch.to(device)

        result = copy(Attributed(torch.arange(3), metadata), torch.device("cpu"))

        torch.testing.assert_close(unwrap_attributed(result), torch.arange(3))
        self.assertIs(metadata, result.metadata)


class ModuleCopyTest(unittest.TestCase):
    """torch.package can load a second copy of this module in the same process."""

    def test_attribution_and_carriers_work_across_copies(self) -> None:
        other = _load_second_copy()
        self.assertIsNot(other.Attributed, Attributed)
        received: List[object] = []

        @other.batch_operation("forward")
        def forward_in_other_copy(batch: object) -> None:
            received.append(batch)
            raise RuntimeError("raised in the other copy")

        with self.assertRaises(RuntimeError) as raised:
            # A carrier from this copy, consumed by the other copy's decorator.
            forward_in_other_copy(Attributed("raw", "m"))

        self.assertEqual(["raw"], received)
        # Read back with this copy's reader.
        self.assertEqual(
            {"operation": "forward", "metadata": "m"},
            get_batch_attribution(raised.exception),
        )
        self.assertTrue(is_attributed(other.Attributed("x", "m")))
