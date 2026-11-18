#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
import unittest
from io import BytesIO

import torch
from torch.package import PackageExporter, PackageImporter
from torchrec.distributed.train_pipeline.batch_attribution import (
    attach_exception_attribution,
    AttributedBatch,
    ExceptionAttribution,
    get_exception_attribution,
    unwrap_attributed_batch,
)


class _RejectsMetadataError(Exception):
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("exception attributes are read-only")


class _RejectsAttributionLookupError(Exception):
    def __getattribute__(self, name: str) -> object:
        if name == "_torchrec_exception_attribution":
            raise RuntimeError("attribution lookup failed")
        return super().__getattribute__(name)


class _FalseyError(Exception):
    def __bool__(self) -> bool:
        return False


class BatchAttributionTest(unittest.TestCase):
    def test_carrier_fields_are_read_only(self) -> None:
        batch = object()
        metadata = object()
        attributed = AttributedBatch(batch, metadata)

        self.assertIs(attributed.value, batch)
        self.assertIs(attributed.metadata, metadata)
        self.assertIsNot(attributed, AttributedBatch(batch, metadata))
        self.assertFalse(hasattr(attributed, "to"))
        self.assertFalse(hasattr(attributed, "record_stream"))
        for field in ("value", "metadata"):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(attributed, field, object())

    def test_unwrap_returns_batch_and_metadata(self) -> None:
        batch = object()
        metadata = object()

        self.assertEqual(
            (batch, metadata),
            unwrap_attributed_batch(AttributedBatch(batch, metadata)),
        )
        self.assertEqual((batch, None), unwrap_attributed_batch(batch))
        self.assertEqual((None, None), unwrap_attributed_batch(None))

    def test_attach_returns_and_preserves_existing_attribution(self) -> None:
        error = RuntimeError("failed")
        metadata = object()

        attribution = attach_exception_attribution(
            error,
            "forward",
            metadata,
            batch_index=7,
        )
        existing = attach_exception_attribution(
            error,
            "backward",
            object(),
            batch_index=8,
        )

        self.assertIs(get_exception_attribution(error), attribution)
        self.assertIs(existing, attribution)
        self.assertEqual(existing.operation, "forward")
        self.assertIs(existing.metadata, metadata)
        self.assertEqual(existing.batch_index, 7)

    def test_exception_override_cannot_reject_attribution(self) -> None:
        error = _RejectsMetadataError("failed")
        metadata = object()

        attribution = attach_exception_attribution(error, "forward", metadata)

        self.assertIs(get_exception_attribution(error), attribution)
        self.assertEqual(attribution.operation, "forward")
        self.assertIs(attribution.metadata, metadata)
        self.assertIsNone(attribution.batch_index)

    def test_falsey_exception_keeps_its_attribution(self) -> None:
        metadata = object()
        error = _FalseyError("inner")

        attribution = attach_exception_attribution(error, "forward", metadata)

        self.assertIs(get_exception_attribution(error), attribution)
        self.assertEqual(attribution.operation, "forward")
        self.assertIs(attribution.metadata, metadata)

    def test_get_exception_attribution_follows_explicit_cause(self) -> None:
        metadata = object()
        inner = ValueError("inner")
        attach_exception_attribution(inner, "copy_to_device", metadata)
        outer = RuntimeError("outer")
        outer.__cause__ = inner

        attribution = get_exception_attribution(outer)

        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual(attribution.operation, "copy_to_device")
        self.assertIs(attribution.metadata, metadata)

    def test_implicit_context_does_not_block_new_attribution(self) -> None:
        inner_metadata = object()
        outer_metadata = object()
        inner = RuntimeError("inner")
        inner_attribution = attach_exception_attribution(
            inner,
            "forward",
            inner_metadata,
            batch_index=3,
        )

        try:
            raise inner
        except RuntimeError:
            with self.assertRaises(ValueError) as raised:
                raise ValueError("outer")

        outer = raised.exception
        self.assertIs(outer.__context__, inner)
        self.assertFalse(outer.__suppress_context__)
        self.assertIsNone(get_exception_attribution(outer))

        outer_attribution = attach_exception_attribution(
            outer,
            "copy_to_device",
            outer_metadata,
            batch_index=4,
        )

        self.assertIs(get_exception_attribution(outer), outer_attribution)
        self.assertEqual(outer_attribution.operation, "copy_to_device")
        self.assertIs(outer_attribution.metadata, outer_metadata)
        self.assertEqual(outer_attribution.batch_index, 4)
        self.assertIs(get_exception_attribution(inner), inner_attribution)

    def test_exception_attribution_handles_cyclic_cause(self) -> None:
        error = RuntimeError("failed")
        with self.assertRaises(RuntimeError) as raised:
            raise error from error

        self.assertIs(raised.exception, error)
        self.assertIs(error.__cause__, error)
        self.assertIsNone(get_exception_attribution(error))

        attribution = attach_exception_attribution(
            error,
            "forward",
            [b"metadata"],
        )

        self.assertIs(get_exception_attribution(error), attribution)

    def test_get_exception_attribution_swallows_lookup_failure(self) -> None:
        error = _RejectsAttributionLookupError("failed")
        outer = RuntimeError("outer")
        outer.__cause__ = error

        self.assertIsNone(get_exception_attribution(error))
        self.assertIsNone(get_exception_attribution(outer))

    def test_carrier_and_exception_attribution_are_pickleable(self) -> None:
        metadata = {"source": [b"table"]}
        attributed = pickle.loads(
            pickle.dumps(AttributedBatch(torch.tensor([7]), metadata))
        )
        error = ValueError("failed")
        attach_exception_attribution(
            error,
            "forward",
            metadata,
            batch_index=7,
        )
        restored_error = pickle.loads(pickle.dumps(error))
        restored_attribution = get_exception_attribution(restored_error)

        self.assertTrue(torch.equal(attributed.value, torch.tensor([7])))
        self.assertEqual(attributed.metadata, metadata)
        self.assertIs(type(restored_error), ValueError)
        self.assertEqual(restored_error.args, ("failed",))
        self.assertIsNotNone(restored_attribution)
        assert restored_attribution is not None
        self.assertIs(type(restored_attribution), ExceptionAttribution)
        self.assertEqual(restored_attribution.operation, "forward")
        self.assertEqual(restored_attribution.metadata, metadata)
        self.assertEqual(restored_attribution.batch_index, 7)

    def test_torch_package_preserves_and_recognizes_carrier(self) -> None:
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("torchrec.distributed.train_pipeline.batch_attribution")
            exporter.extern("*")
            exporter.save_pickle(
                "work",
                "batch.pkl",
                AttributedBatch(torch.tensor([7]), {"batch": "A"}),
            )
        buffer.seek(0)
        packaged = PackageImporter(buffer).load_pickle("work", "batch.pkl")

        batch, metadata = unwrap_attributed_batch(packaged)
        self.assertIsNot(type(packaged), AttributedBatch)
        self.assertTrue(torch.equal(batch, torch.tensor([7])))
        self.assertEqual(metadata, {"batch": "A"})

    def test_torch_package_preserves_exception_attribution(self) -> None:
        error = ValueError("failed")
        attach_exception_attribution(
            error,
            "forward",
            {"batch": "A"},
            batch_index=7,
        )
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("torchrec.distributed.train_pipeline.batch_attribution")
            exporter.extern("*")
            exporter.save_pickle("work", "error.pkl", error)
        buffer.seek(0)
        restored_error = PackageImporter(buffer).load_pickle("work", "error.pkl")

        attribution = get_exception_attribution(restored_error)
        self.assertIs(type(restored_error), ValueError)
        self.assertEqual(restored_error.args, ("failed",))
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertIsNot(type(attribution), ExceptionAttribution)
        self.assertEqual(attribution.operation, "forward")
        self.assertEqual(attribution.metadata, {"batch": "A"})
        self.assertEqual(attribution.batch_index, 7)
