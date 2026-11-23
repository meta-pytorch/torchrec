#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torchrec.experimental.torch_tpu.pallas.ops import generate_vbe_row_output_offsets


def _metadata(counts, dim, input_dtype=torch.int32, output_dtype=torch.int64):
    counts = torch.tensor(counts, dtype=input_dtype)
    B_offsets = torch.cat(
        (torch.zeros(1, dtype=input_dtype), counts.sum(1).cumsum(0))
    ).to(input_dtype)
    rank_offsets = torch.cat(
        (torch.zeros((counts.shape[0], 1), dtype=input_dtype), counts.cumsum(1)), dim=1
    ).to(input_dtype)
    output_offsets = torch.cat(
        (
            torch.zeros(1, dtype=output_dtype),
            (counts.T.flatten().to(output_dtype) * dim).cumsum(0),
        )
    ).to(output_dtype)
    return B_offsets, rank_offsets, output_offsets


class VbeMetadataCpuTest(unittest.TestCase):
    def test_cpu_ops_import_and_package_without_tpu_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import ast
import importlib.abc
import io
import sys

sys.path[:] = ast.literal_eval(sys.argv[1])

import torch
from torch.package import PackageExporter

class NoTpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"jax", "jaxlib", "torch_tpu"}:
            raise ImportError(f"Unexpected TPU dependency: {fullname}")

sys.meta_path.insert(0, NoTpuImports())
from torchrec.experimental.torch_tpu.pallas import ops

with PackageExporter(io.BytesIO()) as exporter:
    exporter.extern("torch.**")
    exporter.intern("torchrec.experimental.torch_tpu.pallas.**")
    exporter.save_module(ops.__name__)
""",
                repr(sys.path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_known_rank_major_offsets(self):
        metadata = _metadata([[2, 3], [1, 2]], 16)
        expected = torch.tensor([0, 16, 48, 64, 80, 32, 96, 112])
        actual = generate_vbe_row_output_offsets(*metadata, 8, 16, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_none_and_partial_metadata(self):
        self.assertIsNone(generate_vbe_row_output_offsets(None, None, None, 0, 16))
        with self.assertRaisesRegex(AssertionError, "all None or all provided"):
            generate_vbe_row_output_offsets(None, torch.zeros((1, 2)), None, 0, 16)
        with self.assertRaisesRegex(AssertionError, "required for VBE"):
            generate_vbe_row_output_offsets(torch.zeros(2), None, None, 0, 16)

    def test_fake_output_shape_and_dtype(self):
        actual = torch.ops.torchrec.generate_vbe_row_output_offsets(
            torch.empty(3, dtype=torch.int32, device="meta"),
            torch.empty((2, 4), dtype=torch.int32, device="meta"),
            torch.empty(7, dtype=torch.int64, device="meta"),
            17,
            16,
            9,
        )
        self.assertEqual(actual.shape, (17,))
        self.assertEqual(actual.dtype, torch.int64)

    def test_cpu_reference_keeps_large_int64_offsets(self):
        metadata = _metadata([[2, 1], [1, 2]], 1 << 30)
        expected = torch.tensor([0, 1, 3, 2, 4, 5], dtype=torch.int64) * (1 << 30)
        actual = generate_vbe_row_output_offsets(*metadata, 6, 1 << 30, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(actual.max().item(), torch.iinfo(torch.int32).max)


class VbeMetadataTpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_tpu  # pyrefly: ignore[missing-import]  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("requires an attached TPU")
        if not torch.accelerator.is_available():
            raise unittest.SkipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import impl

        cls.impl = impl
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")
            cls.addClassCleanup(dist.destroy_process_group)

    def test_matches_torch_reference(self):
        cases = [
            [[1]],
            [[0, 0], [0, 0]],
            [[0, 2, 0], [3, 0, 1]],
            [[255, 256, 257], [0, 513, 1]],
            [[1023, 1024, 1025], [0, 2049, 1]],
            [[1, 1024, 1024], [1025, 0, 2048]],
            [[1, 0, 0, 0, 0, 0, 0, 0], [0] * 8],
            [
                [(feature * 13 + rank * 7) % 31 for rank in range(8)]
                for feature in range(19)
            ],
        ]
        for counts in cases:
            for dim, input_dtype, output_dtype in (
                (16, torch.int32, torch.int32),
                (128, torch.int64, torch.int64),
            ):
                with self.subTest(counts=counts, dim=dim, output_dtype=output_dtype):
                    metadata = _metadata(counts, dim, input_dtype, output_dtype)
                    num_bags = sum(map(sum, counts))
                    bound = max(map(max, counts))
                    expected = generate_vbe_row_output_offsets(*metadata, num_bags, dim)
                    with patch.object(
                        self.impl,
                        "_generate_vbe_metadata",
                        wraps=self.impl._generate_vbe_metadata,
                    ) as kernel:
                        actual = generate_vbe_row_output_offsets(
                            *(t.to("tpu") for t in metadata),
                            num_bags,
                            dim,
                            bound,
                        )
                        kernel.assert_called_once()
                    self.assertEqual(actual.device.type, "tpu")
                    self.assertEqual(actual.shape, (num_bags,))
                    self.assertEqual(actual.dtype, output_dtype)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_rejects_output_address_overflow(self):
        metadata = _metadata([[1]], 16)
        with self.assertRaisesRegex(ValueError, "must fit in int32"):
            generate_vbe_row_output_offsets(
                *(t.to("tpu") for t in metadata), 3, 1 << 30, 1
            )


if __name__ == "__main__":
    unittest.main()
