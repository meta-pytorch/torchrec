# TorchRec Test Patterns Reference

Complete templates for each test type. Use these as scaffolding when generating tests.

## File Header (All Test Files)

```python
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
```

## Unit Test Template

For non-distributed code in `torchrec/modules/`, `torchrec/sparse/`, `torchrec/optim/`, `torchrec/metrics/`.

```python
import unittest

import torch
# Import the module under test
from torchrec.modules.my_module import MyModule, MyConfig


class TestMyModule(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.config = MyConfig(
            # ... representative config ...
        )

    def test_forward_basic(self) -> None:
        """Test forward pass with standard inputs."""
        module = MyModule(config=self.config)
        input_tensor = torch.randn(4, 64)
        output = module(input_tensor)
        self.assertEqual(output.shape, (4, 128))

    def test_forward_empty_input(self) -> None:
        """Test forward pass with empty input."""
        module = MyModule(config=self.config)
        input_tensor = torch.randn(0, 64)
        output = module(input_tensor)
        self.assertEqual(output.shape, (0, 128))

    def test_forward_single_element(self) -> None:
        """Test forward pass with single-element batch."""
        module = MyModule(config=self.config)
        input_tensor = torch.randn(1, 64)
        output = module(input_tensor)
        self.assertEqual(output.shape, (1, 128))

    def test_invalid_config_raises(self) -> None:
        """Test that invalid config raises ValueError."""
        with self.assertRaises(ValueError):
            MyModule(config=MyConfig(invalid_param=-1))
```

## Unit Test with KJT Inputs

For modules that consume `KeyedJaggedTensor`:

```python
import unittest

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestMyEmbeddingModule(unittest.TestCase):
    def _create_kjt(self, batch_size: int = 2) -> KeyedJaggedTensor:
        """Helper to create a representative KJT for testing."""
        return KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 1, 2, 3, 4, 5]),
            lengths=torch.tensor([2, 1, 1, 2]),  # batch_size * num_features
        )

    def _create_ebc(self) -> EmbeddingBagCollection:
        """Helper to create an EBC for testing."""
        return EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    embedding_dim=8,
                    num_embeddings=100,
                    feature_names=["feature_0"],
                ),
                EmbeddingBagConfig(
                    name="table_1",
                    embedding_dim=8,
                    num_embeddings=100,
                    feature_names=["feature_1"],
                ),
            ],
        )

    def test_forward(self) -> None:
        ebc = self._create_ebc()
        kjt = self._create_kjt()
        output = ebc(kjt)
        # KeyedTensor output: batch_size x total_embedding_dim
        self.assertEqual(output.values().shape, (2, 16))

    def test_empty_kjt(self) -> None:
        ebc = self._create_ebc()
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([], dtype=torch.long),
            lengths=torch.tensor([0, 0, 0, 0]),
        )
        output = ebc(kjt)
        self.assertEqual(output.values().shape, (2, 16))
```

## Distributed Test Template

For code in `torchrec/distributed/` using collectives, sharding, or process groups.

```python
import unittest

import torch
import torch.distributed as dist
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.test_utils import skip_if_asan_class

# Module-level test function: runs once per rank
def _test_my_sharding(
    rank: int,
    world_size: int,
    backend: str = "gloo",
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        device = ctx.device
        pg = ctx.pg

        # Setup: create module, inputs, etc.
        # ...

        # Run the operation under test
        # ...

        # Assert results are correct on this rank
        # torch.testing.assert_close(actual, expected)


@skip_if_asan_class
class TestMySharding(MultiProcessTestBase):
    def test_sharding_world_size_2(self) -> None:
        self._run_multi_process_test(
            callable=_test_my_sharding,
            world_size=2,
            backend="gloo",
        )

    def test_sharding_world_size_4(self) -> None:
        self._run_multi_process_test(
            callable=_test_my_sharding,
            world_size=4,
            backend="gloo",
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_sharding_nccl(self) -> None:
        self._run_multi_process_test(
            callable=_test_my_sharding,
            world_size=2,
            backend="nccl",
        )
```

## Distributed Test with Static Method Pattern

Alternative pattern using `@staticmethod` on the test class:

```python
@skip_if_asan_class
class TestMyDistributedModule(MultiProcessTestBase):
    @staticmethod
    def _test_forward(rank: int, world_size: int) -> None:
        with MultiProcessContext(rank, world_size, "gloo") as ctx:
            # ... test logic ...
            pass

    def test_forward(self) -> None:
        self._run_multi_process_test(
            callable=self._test_forward,
            world_size=2,
        )
```

## Hypothesis-Parameterized Test Template

For tests that need to cover multiple configurations (sharding types, kernels, pooling modes).

```python
import unittest

import torch
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingType
from torchrec.test_utils import skip_if_asan_class


def _test_sharding_type(
    rank: int,
    world_size: int,
    sharding_type: str,
    kernel_type: str,
) -> None:
    with MultiProcessContext(rank, world_size, "gloo") as ctx:
        # ... test with the given sharding_type and kernel_type ...
        pass


@skip_if_asan_class
class TestShardingVariants(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=6, deadline=None)
    def test_sharding(self, sharding_type: str, kernel_type: str) -> None:
        self._run_multi_process_test(
            callable=_test_sharding_type,
            world_size=2,
            sharding_type=sharding_type,
            kernel_type=kernel_type,
        )
```

## Hypothesis Unit Test Template

For parameterized unit tests (non-distributed):

```python
import unittest

import torch
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st
from torchrec.modules.my_module import MyModule


class TestMyModuleParameterized(unittest.TestCase):
    @given(
        batch_size=st.sampled_from([1, 4, 32]),
        use_weights=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_forward_variants(self, batch_size: int, use_weights: bool) -> None:
        module = MyModule(use_weights=use_weights)
        input_tensor = torch.randn(batch_size, 64)
        output = module(input_tensor)
        self.assertEqual(output.shape[0], batch_size)
```

## BUCK Target Templates

### CPU-Only Unit Test
```python
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

python_unittest(
    name = "test_my_module",
    srcs = ["test_my_module.py"],
    deps = [
        "//caffe2:_torch",
        "//torchrec/modules:my_module",
    ],
)
```

### GPU Distributed Test
```python
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")
load("@fbcode_macros//build_defs/lib:re_test_utils.bzl", "re_test_utils")

python_unittest(
    name = "test_my_sharding",
    srcs = ["test_my_sharding.py"],
    remote_execution = re_test_utils.remote_execution(
        mig = "false",
        platform = "gpu-remote-execution",
        resource_units = 2,
    ),
    deps = [
        "//caffe2:_torch",
        "//torchrec/distributed/test_utils:multi_process",
        "//torchrec/test_utils:test_utils",
        "//torchrec/distributed:types",
    ],
)
```

### Test with Hypothesis
```python
python_unittest(
    name = "test_my_sharding",
    srcs = ["test_my_sharding.py"],
    remote_execution = re_test_utils.remote_execution(
        mig = "false",
        platform = "gpu-remote-execution",
        resource_units = 2,
    ),
    supports_static_listing = False,
    deps = [
        "//caffe2:_torch",
        "//torchrec/distributed/test_utils:multi_process",
        "//torchrec/test_utils:test_utils",
        "fbsource//third-party/pypi/hypothesis:hypothesis",
    ],
)
```

## Common Assertions

| Assertion | When to Use |
|-----------|-------------|
| `self.assertEqual(a, b)` | Exact equality (shapes, counts, string values) |
| `self.assertTrue(condition)` | Boolean conditions |
| `self.assertIsInstance(obj, cls)` | Type checking |
| `torch.testing.assert_close(a, b)` | Tensor value comparison with tolerance |
| `torch.equal(a, b)` | Exact tensor equality (use inside `assertTrue`) |
| `self.assertRaises(ErrorType)` | Expected exceptions |
| `self.assertRaisesRegex(ErrorType, "msg")` | Expected exceptions with message matching |

## Edge Cases to Always Cover

For **KJT-based code:**
- Empty KJT (no values, all-zero lengths)
- Single feature, single batch element
- Variable-length sequences (some lengths = 0)

For **distributed code:**
- `world_size=2` (minimum)
- `world_size=4` for sharding strategies that partition differently
- Single rank (`world_size=1`) if the code should support it

For **embedding modules:**
- Single table, single feature
- Multiple tables with different embedding dimensions
- Zero `num_embeddings` if applicable

For **metrics:**
- Empty predictions/labels
- All-zero weights
- Single task vs multi-task
