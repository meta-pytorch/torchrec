#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pickle
import unittest
from typing import Dict

import torch
import torch.fx
from torchrec.distributed.types import LazyAwaitable, LazyGetItemMixin, LazyNoWait


class NeedWait(LazyAwaitable[torch.Tensor]):
    def __init__(self, actual_value: torch.Tensor) -> None:
        super().__init__()
        self.actual_value = actual_value

    def _wait_impl(self) -> torch.Tensor:
        self.actual_value += 8
        return self.actual_value


class NeedWaitNoInit(LazyAwaitable[torch.Tensor]):
    def __init__(self, actual_value: torch.Tensor) -> None:
        # ill-formed type, no super.__init__() here
        # should error out when using it
        self.actual_value = actual_value

    def _wait_impl(self) -> torch.Tensor:
        self.actual_value += 8
        return self.actual_value


class NeedWaitDict(LazyAwaitable[Dict[str, torch.Tensor]]):
    def __init__(self, actual_value: Dict[str, torch.Tensor], key: str) -> None:
        super().__init__()
        self.actual_value = actual_value
        self.key = key

    def _wait_impl(self) -> Dict[str, torch.Tensor]:
        self.actual_value[self.key] *= 3
        return self.actual_value


class AsyncModule(torch.nn.Module):
    """
    Dummy async module

    Args:


    Call Args:
        x: torch.Tensor

    Returns:
        LazyAwaitable[torch.Tensor]

    Example::

        AsyncModule()
    """

    def forward(self, x: torch.Tensor) -> LazyAwaitable[torch.Tensor]:
        return NeedWait(x)


class TestLazyAwaitable(unittest.TestCase):
    def test_lazy_torch_function(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute = AsyncModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                async_fut = self.async_compute(x)
                y = x * 3 + 5
                return torch.add(async_fut, y)

        # ensure computation of y happens earlier than wait()
        m = Model()
        ref_res = m(torch.ones(3, 4))
        self.assertTrue(torch.equal(ref_res, 17 * torch.ones(3, 4)))

        # ensure fx tracing works
        gm = torch.fx.symbolic_trace(m)
        traced_res = gm(torch.ones(3, 4))
        self.assertTrue(torch.equal(traced_res, ref_res))

    def test_lazy_getattr(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute = AsyncModule()

            def forward(self, x: torch.Tensor) -> int:
                async_fut = self.async_compute(x)
                y = x * 3 + 5
                return async_fut.numel() + y.numel()

        m = Model()
        ref_res = m(torch.ones(3, 4))
        self.assertEqual(ref_res, 24)

        # ensure fx tracing works
        gm = torch.fx.symbolic_trace(m)
        traced_res = gm(torch.ones(3, 4))
        self.assertEqual(traced_res, ref_res)

    def test_lazy_getattr_explicit_wait(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute = AsyncModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                async_fut = self.async_compute(x)
                y = x * 3 + 5
                return async_fut.wait() + y

        m = Model()
        ref_res = m(torch.ones(3, 4))
        self.assertTrue(torch.equal(ref_res, 17 * torch.ones(3, 4)))

        # ensure fx tracing works
        gm = torch.fx.symbolic_trace(m)
        traced_res = gm(torch.ones(3, 4))
        self.assertTrue(torch.equal(traced_res, ref_res))

    def test_lazy_awaitable_init_error(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lazy_awaitable = NeedWaitNoInit(torch.ones(2, 3))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # pyrefly: ignore [unsupported-operation]
                return x + self.lazy_awaitable

        m = Model()

        with self.assertRaisesRegex(RuntimeError, "has not been initialized properly"):
            m(torch.ones(2, 3))

    def test_lazy_wait_and_result(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute = AsyncModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                async_fut = self.async_compute(x)
                y = x * 3 + 5
                numel = async_fut.numel()
                return numel + async_fut._result + y

        m = Model()
        ref_res = m(torch.ones(3, 4))
        self.assertTrue(torch.equal(ref_res, 29 * torch.ones(3, 4)))

        # ensure fx tracing works
        gm = torch.fx.symbolic_trace(m)
        traced_res = gm(torch.ones(3, 4))
        self.assertTrue(torch.equal(traced_res, ref_res))

    def test_lazy_get_item(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute = AsyncModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                async_fut = self.async_compute(x)
                return async_fut[1:3]

        m = Model()
        ref_res = m(torch.ones(3, 4))
        self.assertTrue(torch.equal(ref_res, 9 * torch.ones(2, 4)))

        # ensure fx tracing works
        gm = torch.fx.symbolic_trace(m)
        traced_res = gm(torch.ones(3, 4))
        self.assertTrue(torch.equal(traced_res, ref_res))

    def test_lazy_magic_methods(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute1 = AsyncModule()
                self.async_compute2 = AsyncModule()

            def forward(self, x: torch.Tensor) -> int:
                async_fut1 = self.async_compute1(x)
                async_fut2 = self.async_compute2(x)
                y = x * 3 + 5
                return 2 * async_fut1 + y - async_fut2

        m = Model()
        ref_res = m(torch.ones(3, 4))
        self.assertTrue(torch.equal(ref_res, 9 * torch.ones(3, 4)))

        gm = torch.fx.symbolic_trace(m)
        traced_res = gm(torch.ones(3, 4))
        self.assertTrue(torch.equal(traced_res, ref_res))

    def test_awatiable_pickle(self) -> None:
        awaitable = LazyNoWait(torch.randn(2, 3))
        self.assertTrue(
            torch.allclose(pickle.loads(pickle.dumps(awaitable))._obj, awaitable._obj)
        )

    def test_lazy_wait_dict(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dict = {"t1": torch.ones(2, 3)}
                self.wait_dict = NeedWaitDict(self.dict, "t1")

            def forward(self) -> torch.Tensor:
                return self.wait_dict["t1"] + 2

        m = Model()
        ref_res = m()
        self.assertTrue(torch.equal(ref_res, 5 * torch.ones(2, 3)))

        # ensure fx tracing works
        gm = torch.fx.symbolic_trace(m)
        traced_res = gm()
        self.assertTrue(torch.equal(traced_res, ref_res))

    def test_lazy_awaitable_serde(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.async_compute = AsyncModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                async_fut = self.async_compute(x)
                y = x * 3 + 5
                return torch.add(async_fut, y)

        m = Model()
        gm = torch.fx.symbolic_trace(m)

        import pickle
        import tempfile

        tempFile = None
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(gm, f)
            tempFile = f

        with open(tempFile.name, "rb") as f:
            loaded = pickle.load(f)

            ref_res = loaded(torch.ones(3, 4))
            self.assertTrue(torch.equal(ref_res, 17 * torch.ones(3, 4)))

        tempFile.close()

    def test_lazy_getitem_mixin(self) -> None:
        class LazyGetItemAwaitable(
            LazyGetItemMixin[str, torch.Tensor], LazyAwaitable[Dict[str, torch.Tensor]]
        ):
            def __init__(self, actual_value: Dict[str, torch.Tensor]):
                super().__init__()
                self.actual_value = actual_value

            def _wait_impl(self) -> Dict[str, torch.Tensor]:
                for v in self.actual_value.values():
                    v *= 3
                return self.actual_value

        actual_value = {"foo": torch.tensor(1), "bar": torch.tensor(2)}
        a = LazyGetItemAwaitable(actual_value)
        lazy_foo = a["foo"]
        lazy_bar = a["bar"]
        # The returned value should be lazy
        self.assertIsInstance(lazy_foo, LazyAwaitable)
        self.assertIsInstance(lazy_bar, LazyAwaitable)

        # Our lazy values should not have been waited yet
        self.assertIsNone(lazy_foo._result)
        self.assertIsNone(lazy_bar._result)
        self.assertIsNone(a._result)

        # The use of a torch op should trigger exactly one wait on the parent object.
        # pyrefly: ignore[no-matching-overload]
        result = torch.add(lazy_foo, lazy_bar)
        self.assertEqual(result, torch.tensor(1 * 3 + 2 * 3))
