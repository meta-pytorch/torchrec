#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import ast
import inspect
import re
import textwrap
import unittest
from typing import Tuple

import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torchrec.modules.lazy_extension import lazy_apply, LazyModuleExtensionMixin


def remove_comment(source_code: str) -> str:
    result = re.sub(r"\s*#.*", "", str(source_code))
    return result


class TestLazyModuleExtensionMixin(unittest.TestCase):
    @unittest.skip("Not forward compatiable, Fix with D36428528 after 5/20/22")
    def test_source_code_parity_on_call_impl(self) -> None:
        original_call_impl_src = inspect.getsource(torch.nn.Module._call_impl)
        lazy_ext_call_impl_src = inspect.getsource(LazyModuleExtensionMixin._call_impl)

        # normalize the source code
        original_call_impl_src = ast.unparse(
            textwrap.dedent(ast.parse(original_call_impl_src))
        )
        lazy_ext_call_impl_src = ast.unparse(
            textwrap.dedent(ast.parse(lazy_ext_call_impl_src))
        )

        # reproduce the only change:
        old_code = """
                result = hook(self, input)
        """
        new_code = """
                if len(inspect.signature(hook).parameters) == 3:
                    result = hook(self, input, kwargs)
                else:
                    result = hook(self, input)
        """
        expected_lazy_ext_call_impl_src = original_call_impl_src.replace(
            old_code, new_code
        )

        self.assertEqual(
            lazy_ext_call_impl_src,
            expected_lazy_ext_call_impl_src,
            "Please make sure `LazyModuleExtensionMixin._call_impl` has the same source code "
            "as `torch.nn.Module._call_impl` except the expected difference that is checked "
            "in this unit test.",
        )

    def test_source_code_parity_on_infer_parameters(self) -> None:
        original_infer_parameters_src = inspect.getsource(
            LazyModuleMixin._infer_parameters
        )
        lazy_ext_infer_parameters_src = inspect.getsource(
            LazyModuleExtensionMixin._infer_parameters
        )

        # normalize the source code
        original_infer_parameters_src = ast.unparse(
            textwrap.dedent(ast.parse(original_infer_parameters_src))
        )
        lazy_ext_infer_parameters_src = ast.unparse(
            textwrap.dedent(ast.parse(lazy_ext_infer_parameters_src))
        )

        # reproduce the only changes:
        expected_lazy_ext_infer_parameters_src = original_infer_parameters_src.replace(
            "def _infer_parameters(self: _LazyProtocol, module, args, kwargs=None):",
            "def _infer_parameters(self: _LazyExtensionProtocol, module, args, kwargs) -> None:",
        )
        self.assertEqual(
            lazy_ext_infer_parameters_src,
            expected_lazy_ext_infer_parameters_src,
            "Please make sure `LazyModuleExtensionMixin._infer_parameters` has the same source "
            "code as `LazyModuleMixin._infer_parameters` except the expected difference that "
            "is checked in this unit test.",
        )

    def test_forward_pre_hook_self_function_with_input_only(self) -> None:
        class TestModule(LazyModuleExtensionMixin, torch.nn.Module):
            """
            Create this unit test to make sure the old way of initialize self hook function
            is enabled, with the hook definition as:
                valid_input_only_hook(self, module, input)

            if we run the TestModule as::

                m = TestModule()
                output = m()
                expected_output = torch.zeros(2, 2)
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_forward_pre_hook(self.valid_input_only_hook)

            def valid_input_only_hook(self, module, input):
                self.output = torch.zeros(2, 2)

            def initialize_parameters(self) -> None:
                return None

            def forward(self) -> torch.Tensor:
                return self.output

        m = TestModule()

        # test for self function registeration with register_forward_pre_hook
        output_forward = m()
        self.assertTrue(
            torch.allclose(output_forward, torch.zeros(2, 2)),
            "Please make sure forward function is executed as expected.",
        )

    def test_forward_pre_hook_global_function_with_input_only(self) -> None:
        class TestModule(LazyModuleExtensionMixin, torch.nn.Module):
            """
            Create this unit test to make sure the old way of insert hook function is enabled,
            with the hook definition as:
                valid_input_only_hook(self, module, input)

            if we run the TestModule as::

                def input_only_hook(module, input_tuple):
                    return input_tuple[0] + 1
                m = TestModule()
                m.register_forward_pre_hook(input_only_hook)
                output = m(torch.zeros(2, 2))
                expected_output = torch.ones(2, 2)
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def initialize_parameters(self, input) -> None:
                return None

            def forward(self, input) -> torch.Tensor:
                return input

        def input_only_hook(
            module: torch.nn.Module, input: Tuple[torch.Tensor, ...]
        ) -> torch.Tensor:
            # input is tuple
            return input[0] + 1

        m = TestModule()
        m.register_forward_pre_hook(input_only_hook)
        output = m(torch.zeros(2, 2))
        self.assertTrue(torch.allclose(output, torch.ones(2, 2)))

    def test_lazy_apply(self) -> None:
        count_original: int = 0
        count_increment: int = 1

        class TestModule(LazyModuleExtensionMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.count = torch.tensor(count_original)

            def initialize_parameters(self, input) -> None:
                pass

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.count.clone()

        def increment_count(module: torch.nn.Module) -> None:
            if isinstance(module, TestModule):
                module.count += torch.tensor(count_increment)

        def check_result(m: torch.nn.Module, count_after_first_forward: int) -> None:
            # This check ensures that `lazy_apply()` is a delayed operation (i.e. the function is not applied immediately).
            for count in m.parameters():
                self.assertTrue(torch.allclose(count, torch.tensor(0)))

            input = torch.tensor(321)
            out = m(input)

            # This check ensures that the lazy-applied function is not called before forward function is called.
            self.assertTrue(torch.allclose(out, torch.tensor(count_original)))

            # This check ensures that the lazy-applied function is called after forward function is called.
            for count in m.parameters():
                self.assertTrue(
                    torch.allclose(count, torch.tensor(count_after_first_forward))
                )

            # This check ensures that the lazy-applied function is removed after first forward pass is run.
            out = m(input)
            self.assertTrue(
                torch.allclose(out, torch.tensor(count_after_first_forward)), str(out)
            )
            # Since `increment_count` is not run the second time, value of `count` parameter is not changed.
            for count in m.parameters():
                self.assertTrue(
                    torch.allclose(count, torch.tensor(count_after_first_forward))
                )

        check_result(
            lazy_apply(
                TestModule(),
                increment_count,
            ),
            count_after_first_forward=1,
        )
        check_result(
            lazy_apply(
                torch.nn.Sequential(
                    TestModule(),
                    TestModule(),
                ),
                increment_count,
            ),
            count_after_first_forward=1,
        )
        check_result(
            lazy_apply(
                lazy_apply(
                    TestModule(),
                    increment_count,
                ),
                increment_count,
            ),
            count_after_first_forward=2,
        )
        check_result(
            lazy_apply(
                lazy_apply(
                    torch.nn.Sequential(
                        TestModule(),
                        TestModule(),
                    ),
                    increment_count,
                ),
                increment_count,
            ),
            count_after_first_forward=2,
        )
        check_result(
            lazy_apply(
                torch.nn.Sequential(
                    lazy_apply(
                        TestModule(),
                        increment_count,
                    ),
                    torch.nn.Identity(),
                ),
                increment_count,
            ),
            count_after_first_forward=2,
        )

    def test_apply(self) -> None:
        class TestModule(LazyModuleExtensionMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.tensor(1.0)

            def initialize_parameters(self, input) -> None:
                return None

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return input

        @torch.no_grad()
        def init_weights(m: torch.nn.Module) -> None:
            if isinstance(m, TestModule):
                m.param.fill_(7.0)

        # Case 1: Running `.apply()` without running first forward pass to
        # initialize the module will result in error.
        net = torch.nn.Sequential(TestModule(), TestModule())
        with self.assertRaisesRegex(RuntimeError, "has not been initialized"):
            net.apply(init_weights)

        # Case 2: Running `.apply()` after running first forward pass will succeed.
        net(torch.tensor(2.0))
        net.apply(init_weights)
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Union[Tensor,
        #  Module]`.
        self.assertTrue(torch.allclose(net[0].param, torch.tensor(7.0)))

        # Case 3: Running `.lazy_apply()` without running first forward pass will succeed,
        # and the function will be applied right after first forward pass.
        net = torch.nn.Sequential(TestModule(), TestModule())
        net = lazy_apply(net, init_weights)
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        self.assertTrue(torch.allclose(net[0].param, torch.tensor(1.0)))
        net(torch.tensor(2.0))
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        self.assertTrue(torch.allclose(net[0].param, torch.tensor(7.0)))
