#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import inspect
from typing import Any, Callable

import torch
from torch.nn.modules.lazy import _LazyProtocol, LazyModuleMixin
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_forward_hooks,
    _global_forward_pre_hooks,
)
from torch.utils.hooks import BackwardHook


def _apply_functions_after_first_forward(
    module: torch.nn.Module,
    # pyre-ignore[2]
    input: Any,
    # pyre-ignore[2]
    output: Any,
) -> None:
    _functions_to_lazy_apply = getattr(module, "_functions_to_lazy_apply", None)
    if _functions_to_lazy_apply is not None:
        # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Tensor, Module]` is not a
        #  function.
        for fn in _functions_to_lazy_apply:
            module.apply(fn)
        delattr(module, "_functions_to_lazy_apply")
    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `remove`.
    module._lazy_apply_hook.remove()
    delattr(module, "_lazy_apply_hook")


def lazy_apply(
    module: torch.nn.Module, fn: Callable[[torch.nn.Module], None]
) -> torch.nn.Module:
    """Attaches a function to a module, which will be applied recursively to every
    submodule (as returned by `.children()`) of the module as well as the module itself
    right after the first forward pass (i.e. after all submodules and parameters have
    been initialized).

    Typical use includes initializing the numerical value of the parameters of a lazy
    module (i.e. modules inherited from `LazyModuleMixin`).

    NOTE:
        `lazy_apply()` can be used on both lazy and non-lazy modules.

    Args:
        module (torch.nn.Module): module to recursively apply `fn` on.
        fn (Callable[[torch.nn.Module], None]): function to be attached to `module` and
            later be applied to each submodule of `module` and the `module` itself.

    Returns:
        torch.nn.Module: `module` with `fn` attached.

    Example::

        @torch.no_grad()
        def init_weights(m):
            print(m)
            if type(m) == torch.nn.LazyLinear:
                m.weight.fill_(1.0)
                print(m.weight)

        linear = torch.nn.LazyLinear(2)
        lazy_apply(linear, init_weights)  # doesn't run `init_weights` immediately
        input = torch.randn(2, 10)
        linear(input)  # runs `init_weights` only once, right after first forward pass

        seq = torch.nn.Sequential(torch.nn.LazyLinear(2), torch.nn.LazyLinear(2))
        lazy_apply(seq, init_weights)  # doesn't run `init_weights` immediately
        input = torch.randn(2, 10)
        seq(input)  # runs `init_weights` only once, right after first forward pass
    """

    if not hasattr(module, "_functions_to_lazy_apply"):
        # pyre-fixme[16]: `Module` has no attribute `_functions_to_lazy_apply`.
        module._functions_to_lazy_apply = []
    if not hasattr(module, "_lazy_apply_hook"):
        # pyre-fixme[16]: `Module` has no attribute `_lazy_apply_hook`.
        module._lazy_apply_hook = module.register_forward_hook(
            _apply_functions_after_first_forward
        )
    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `append`.
    module._functions_to_lazy_apply.append(fn)
    return module


class _LazyExtensionProtocol(_LazyProtocol):
    # pyre-ignore[2,3]
    def _call_impl(self, *input, **kwargs): ...


class LazyModuleExtensionMixin(LazyModuleMixin):
    """
    This is a temporary extension of `LazyModuleMixin` to support passing keyword
    arguments to lazy module's `forward` method.

    The long-term plan is to upstream this feature to `LazyModuleMixin`. Please see
    https://github.com/pytorch/pytorch/issues/59923 for details.

    Please see `TestLazyModuleExtensionMixin`, which contains unit tests that ensure:
      * `LazyModuleExtensionMixin._infer_parameters` has source code parity with
        torch.nn.modules.lazy.LazyModuleMixin._infer_parameters, except that the former
        can accept keyword arguments.
      * `LazyModuleExtensionMixin._call_impl` has source code parity with
        `torch.nn.Module._call_impl`, except that the former can pass keyword arguments
        to forward pre hooks."
    """

    def apply(self, fn: Callable[[torch.nn.Module], None]) -> torch.nn.Module:
        """Applies `fn` recursively to every submodule (as returned by `.children()`)
        as well as self. Typical use includes initializing the parameters of a model.

        NOTE:
            Calling `apply()` on an uninitialized lazy-module will result in an error.
            User is required to initialize a lazy-module (by doing a dummy forward pass)
            before calling `apply()` on the lazy-module.

        Args:
            fn (torch.nn.Module -> None): function to be applied to each submodule.

        Returns:
            torch.nn.Module: self

        Example::

            @torch.no_grad()
            def init_weights(m):
                print(m)
                if type(m) == torch.nn.LazyLinear:
                    m.weight.fill_(1.0)
                    print(m.weight)

            linear = torch.nn.LazyLinear(2)
            linear.apply(init_weights)  # this fails, because `linear` (a lazy-module) hasn't been initialized yet

            input = torch.randn(2, 10)
            linear(input)  # run a dummy forward pass to initialize the lazy-module

            linear.apply(init_weights)  # this works now
        """

        if hasattr(self, "_initialize_hook"):
            raise RuntimeError(
                "Module {} has not been initialized. ".format(self)
                + "Please run a dummy forward pass on the model to initialize all modules, "
                + "or use torchrec.modules.lazy_extension.lazy_apply to attach a function "
                + "to this module which would be applied after this module is initialized."
            )
        # If the module is already initialized, call `super().apply(fn)` to
        # run the usual apply logic.
        # pyre-ignore[16]
        return super().apply(fn)

    # pyre-ignore[2, 47]
    #  `LazyModuleMixin` inconsistently.
    def _infer_parameters(self: _LazyExtensionProtocol, module, args, kwargs) -> None:
        r"""Infers the size and initializes the parameters according to the provided input batch.

        Given a module that contains parameters that were declared inferable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
        """
        kwargs = kwargs if kwargs else {}
        module.initialize_parameters(*args, **kwargs)
        if module.has_uninitialized_params():
            raise RuntimeError(
                f"module {self._get_name()} has not been fully initialized"
            )
        module._initialize_hook.remove()
        module._load_hook.remove()
        delattr(module, "_initialize_hook")
        delattr(module, "_load_hook")
        if module.cls_to_become is not None:
            module.__class__ = module.cls_to_become

    # pyre-ignore[2,3]
    def _call_impl(self, *input, **kwargs):  # noqa: C901
        # pyre-ignore[16]
        forward_call = (
            self._slow_forward if torch._C._get_tracing_state() else self.forward
        )
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        # pyre-ignore[16]
        if not (
            self._backward_hooks
            or self._forward_hooks
            or self._forward_pre_hooks
            or _global_backward_hooks
            or _global_forward_hooks
            or _global_forward_pre_hooks
        ):
            return forward_call(*input, **kwargs)
        # Do not call functions when jit is used
        full_backward_hooks, non_full_backward_hooks = [], []
        if self._backward_hooks or _global_backward_hooks:
            # pyre-ignore[16]
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        if _global_forward_pre_hooks or self._forward_pre_hooks:
            # pyre-ignore[60]: Concatenation not yet support for multiple variadic
            #  tuples: `*torch.nn.modules.module._global_forward_pre_hooks.values(),
            #  *self._forward_pre_hooks.values()`.
            for hook in (
                *_global_forward_pre_hooks.values(),
                *self._forward_pre_hooks.values(),
            ):
                if len(inspect.signature(hook).parameters) == 3:
                    result = hook(self, input, kwargs)
                else:
                    result = hook(self, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result

        bw_hook = None
        if full_backward_hooks:
            # pyre-fixme[20]: Argument `user_pre_hooks` expected.
            bw_hook = BackwardHook(self, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        result = forward_call(*input, **kwargs)
        if _global_forward_hooks or self._forward_hooks:
            # pyre-ignore[60]: Concatenation not yet support for multiple variadic
            #  tuples: `*torch.nn.modules.module._global_forward_hooks.values(),
            #  *self._forward_hooks.values()`.
            for hook in (
                *_global_forward_hooks.values(),
                *self._forward_hooks.values(),
            ):
                hook_result = hook(self, input, result)
                if hook_result is not None:
                    result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)

        # Handle the non-full backward hooks
        if non_full_backward_hooks:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                # pyre-ignore[16]
                self._maybe_warn_non_full_backward_hook(input, result, grad_fn)

        return result

    # pyre-ignore[4]
    # pyre-fixme[15]: `__call__` overrides attribute defined in `type` inconsistently.
    __call__: Callable[..., Any] = _call_impl
