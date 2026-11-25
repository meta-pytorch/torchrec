#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator, Iterator

from torchrec.distributed.attributed_logging import attributed_logger
from torchrec.distributed.batch_attribution import (
    AttributedBatch,
    get_exception_attribution,
)


class _Owner:
    __slots__ = ("batch_metadata", "index")

    def __init__(
        self,
        batch_metadata: object | None,
        index: int | None = None,
    ) -> None:
        self.batch_metadata = batch_metadata
        self.index = index


class _Value:
    pass


class AttributedLoggingTest(unittest.TestCase):
    def test_success_preserves_arguments_and_result(self) -> None:
        context = _Owner({"batch": "A"})
        batch = AttributedBatch(object(), context.batch_metadata)
        loss = _Value()
        optimizer = _Value()
        expected = object()
        received: list[tuple[object, object, object]] = []

        class Pipeline:
            def __init__(self) -> None:
                self.contexts = [context]

            @attributed_logger("forward", owner=lambda self: self.contexts[0])
            def forward(
                self,
                payload: object,
                objective: object,
                optimizer: object,
            ) -> object:
                received.append((payload, objective, optimizer))
                return expected

        result = Pipeline().forward(batch, loss, optimizer)

        self.assertIs(result, expected)
        self.assertEqual([(batch, loss, optimizer)], received)
        self.assertEqual({}, vars(loss))
        self.assertEqual({}, vars(optimizer))

    def test_owner_is_resolved_before_each_operation(self) -> None:
        first_metadata = object()
        second_metadata = object()

        class Pipeline:
            def __init__(self) -> None:
                self.contexts = [_Owner(first_metadata, index=7)]

            @attributed_logger("backward", owner=lambda self: self.contexts[0])
            def fail_after_advancing_context(self, error: ValueError) -> None:
                self.contexts.pop(0)
                raise error

        pipeline = Pipeline()
        first_error = ValueError("first")
        with self.assertRaises(ValueError) as raised:
            pipeline.fail_after_advancing_context(first_error)

        self.assertIs(raised.exception, first_error)
        first_attribution = get_exception_attribution(first_error)
        self.assertIsNotNone(first_attribution)
        assert first_attribution is not None
        self.assertIs(first_attribution.metadata, first_metadata)
        self.assertEqual(first_attribution.batch_index, 7)

        pipeline.contexts.append(_Owner(second_metadata, index=8))
        second_error = ValueError("second")
        with self.assertRaises(ValueError):
            pipeline.fail_after_advancing_context(second_error)

        second_attribution = get_exception_attribution(second_error)
        self.assertIsNotNone(second_attribution)
        assert second_attribution is not None
        self.assertIs(second_attribution.metadata, second_metadata)
        self.assertEqual(second_attribution.batch_index, 8)

    def test_none_metadata_disables_attribution(self) -> None:
        error = ValueError("failed")

        class Pipeline:
            context = _Owner(None)

            @attributed_logger("forward", owner=lambda self: self.context)
            def forward(self) -> None:
                raise error

        with self.assertRaises(ValueError) as raised:
            Pipeline().forward()

        self.assertIs(raised.exception, error)
        self.assertIsNone(get_exception_attribution(error))

    def test_falsey_metadata_remains_attributable(self) -> None:
        metadata: list[object] = []
        error = ValueError("failed")

        class Pipeline:
            context = _Owner(metadata)

            @attributed_logger("forward", owner=lambda self: self.context)
            def forward(self) -> None:
                raise error

        with self.assertRaises(ValueError):
            Pipeline().forward()

        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertIs(attribution.metadata, metadata)

    def test_stop_iteration_is_not_attributed(self) -> None:
        error = StopIteration()

        class Pipeline:
            context = _Owner(object())

            @attributed_logger("read", owner=lambda self: self.context)
            def read(self) -> None:
                raise error

        with self.assertRaises(StopIteration) as raised:
            Pipeline().read()

        self.assertIs(raised.exception, error)
        self.assertIsNone(get_exception_attribution(error))

    def test_existing_inner_attribution_is_preserved(self) -> None:
        inner = ValueError("inner")
        translated = RuntimeError("translated")

        class Pipeline:
            inner_context = _Owner({"batch": "B"})
            outer_context = _Owner({"batch": "A"})

            @attributed_logger(
                "inner_operation",
                owner=lambda self: self.inner_context,
            )
            def inner_operation(self) -> None:
                raise inner

            @attributed_logger("forward", owner=lambda self: self.outer_context)
            def forward(self) -> None:
                try:
                    self.inner_operation()
                except ValueError as error:
                    raise translated from error

        with self.assertRaises(RuntimeError) as raised:
            Pipeline().forward()

        self.assertIs(raised.exception, translated)
        self.assertIs(translated.__cause__, inner)
        attribution = get_exception_attribution(translated)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual(attribution.operation, "inner_operation")
        self.assertEqual(attribution.metadata, {"batch": "B"})

    def test_keyword_owner_attributes_failure_without_changing_call(self) -> None:
        metadata = object()
        context = _Owner(metadata, index=11)
        payload = object()
        error = ValueError("copy failed")
        received: list[tuple[object, object]] = []

        @attributed_logger("copy_to_device", owner_kwarg="context")
        def copy(
            value: object,
            *,
            context: object | None = None,
        ) -> None:
            received.append((value, context))
            raise error

        with self.assertRaises(ValueError) as raised:
            copy(payload, context=context)

        self.assertIs(raised.exception, error)
        self.assertEqual([(payload, context)], received)
        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual("copy_to_device", attribution.operation)
        self.assertIs(metadata, attribution.metadata)
        self.assertEqual(11, attribution.batch_index)

    def test_owner_kwarg_resolves_positional_argument(self) -> None:
        metadata = object()
        context = _Owner(metadata, index=11)
        payload = object()
        error = ValueError("copy failed")
        received: list[tuple[object, object]] = []

        @attributed_logger("copy_to_device", owner_kwarg="context")
        def copy(
            value: object,
            context: object | None = None,
        ) -> None:
            received.append((value, context))
            raise error

        with self.assertRaises(ValueError) as raised:
            copy(payload, context)

        self.assertIs(raised.exception, error)
        self.assertEqual([(payload, context)], received)
        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual("copy_to_device", attribution.operation)
        self.assertIs(metadata, attribution.metadata)
        self.assertEqual(11, attribution.batch_index)

    def test_owner_kwarg_resolves_keyword_for_positional_parameter(self) -> None:
        metadata = object()
        context = _Owner(metadata, index=3)
        payload = object()
        error = ValueError("copy failed")

        @attributed_logger("copy_to_device", owner_kwarg="context")
        def copy(
            value: object,
            context: object | None = None,
        ) -> None:
            raise error

        with self.assertRaises(ValueError) as raised:
            copy(payload, context=context)

        self.assertIs(raised.exception, error)
        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertIs(metadata, attribution.metadata)
        self.assertEqual(3, attribution.batch_index)

    def test_explicit_none_owner_kwarg_preserves_failure_without_attribution(
        self,
    ) -> None:
        error = ValueError("copy failed")

        @attributed_logger("copy_to_device", owner_kwarg="context")
        def copy(
            value: object,
            context: object | None = None,
        ) -> None:
            raise error

        with self.assertRaises(ValueError) as raised:
            copy(object(), context=None)

        self.assertIs(raised.exception, error)
        self.assertIsNone(get_exception_attribution(error))

    def test_omitted_owner_kwarg_preserves_failure_without_attribution(self) -> None:
        payload = object()
        error = ValueError("copy failed")
        received: list[tuple[object, object | None]] = []

        @attributed_logger("copy_to_device", owner_kwarg="context")
        def copy(
            value: object,
            context: object | None = None,
        ) -> None:
            received.append((value, context))
            raise error

        with self.assertRaises(ValueError) as raised:
            copy(payload)

        self.assertIs(raised.exception, error)
        self.assertEqual([(payload, None)], received)
        self.assertIsNone(get_exception_attribution(error))

    def test_owner_kwarg_typo_is_rejected(self) -> None:
        def copy(
            value: object,
            *,
            context: object | None = None,
        ) -> object:
            return value

        with self.assertRaisesRegex(TypeError, "owner_kwarg 'contex'"):
            attributed_logger("copy_to_device", owner_kwarg="contex")(copy)

    def test_owner_kwarg_is_rejected_for_var_keyword_parameter(self) -> None:
        def copy(**kwargs: object) -> None:
            pass

        with self.assertRaisesRegex(
            TypeError,
            "owner_kwarg 'kwargs' must name a positional-or-keyword",
        ):
            attributed_logger("copy_to_device", owner_kwarg="kwargs")(copy)

    def test_owner_kwarg_is_rejected_for_positional_only_parameter(self) -> None:
        def copy(context: object, /) -> None:
            pass

        with self.assertRaisesRegex(
            TypeError,
            "owner_kwarg 'context' must name a positional-or-keyword",
        ):
            attributed_logger("copy_to_device", owner_kwarg="context")(copy)

    def test_owner_resolution_failure_warns_once_without_changing_call(self) -> None:
        expected = object()
        errors = [ValueError("first"), ValueError("second")]
        received: list[ValueError] = []

        def broken_owner(_pipeline: object) -> object:
            raise RuntimeError("owner failed")

        class Pipeline:
            @attributed_logger("forward", owner=broken_owner)
            def forward(self, error: ValueError | None = None) -> object:
                if error is None:
                    return expected
                received.append(error)
                raise error

        pipeline = Pipeline()
        with self.assertLogs(
            "torchrec.distributed.attributed_logging",
            level="WARNING",
        ) as logs:
            self.assertIs(pipeline.forward(), expected)
            for error in errors:
                with self.assertRaises(ValueError) as raised:
                    pipeline.forward(error)
                self.assertIs(raised.exception, error)

        self.assertEqual(errors, received)
        self.assertEqual(1, len(logs.records))
        self.assertIn("forward", logs.records[0].getMessage())
        for error in errors:
            self.assertIsNone(get_exception_attribution(error))

    def test_metadata_only_owner_remains_supported(self) -> None:
        metadata = object()
        error = ValueError("failed")

        class MetadataOnlyOwner:
            batch_metadata = metadata

        class Pipeline:
            context = MetadataOnlyOwner()

            @attributed_logger("forward", owner=lambda self: self.context)
            def forward(self) -> None:
                raise error

        with self.assertRaises(ValueError):
            Pipeline().forward()

        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertIs(metadata, attribution.metadata)
        self.assertIsNone(attribution.batch_index)

    def test_decorator_requires_one_owner_source(self) -> None:
        with self.assertRaisesRegex(TypeError, "exactly one"):
            attributed_logger("copy_to_device")

        with self.assertRaisesRegex(TypeError, "exactly one"):
            attributed_logger(
                "copy_to_device",
                owner=lambda value: value,
                owner_kwarg="context",
            )

    def test_deferred_callables_are_rejected(self) -> None:
        async def coroutine(self: object) -> None:
            pass

        def generator(self: object) -> Iterator[object]:
            yield self

        async def async_generator(self: object) -> AsyncIterator[object]:
            yield self

        for func in (coroutine, generator, async_generator):
            with self.subTest(func=func.__name__):
                with self.assertRaisesRegex(TypeError, "synchronous callables"):
                    attributed_logger("forward", owner=lambda self: None)(func)
