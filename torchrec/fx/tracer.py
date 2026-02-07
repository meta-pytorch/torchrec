#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Argument
from torchrec.distributed.types import LazyAwaitable, NoWait
from torchrec.fx.utils import dmp_fx_trace_forward

_is_fx_tracing_flag = False


def is_fx_tracing() -> bool:
    return _is_fx_tracing_flag


class Tracer(torch.fx.Tracer):
    """
    Custom FX tracer for torchrec

    See `Torch.FX documentation <https://pytorch.org/docs/stable/fx.html>`_

    We create a custom FX tracer to trace torchrec based models. The custom tracer
    handles python generic types (i.e. NoWait[T], Awaitable[T]) and lower it to
    TorchScript if needed
    """

    def __init__(self, leaf_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        Override FX definition to include quantized embedding bags
        """
        if type(m).__name__ in self._leaf_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)

    @compatibility(is_backward_compatible=True)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True

        try:
            # TODO(ivankobzarev): support DMP not only on the root level
            from torchrec.distributed.model_parallel import DistributedModelParallel

            if isinstance(root, DistributedModelParallel):
                # In the case where the module is wrapped in DMP, you need to replace DMP's forward
                # call with a new signature, one with explicit args, because fx can't handle variable args.
                # Furthermore, we need to provide the `fn_root` argument because when tracing a function,
                # fx uses an empty module as the root (unless one is explicitly provided), which leads to
                # issues with path_of_module and named_buffers.

                # TODO(shababayub): This can be removed if we either stop supporting dmp wrapping
                # for fx trace or strip dmp name in named_modules path (much like named_buffers).
                if isinstance(root, torch.nn.Module):
                    for prefix, module in root.named_modules():
                        # TODO(T140754678): Remove this workaround to _fx_path
                        module._fx_path = prefix

                dmp = root
                graph = super().trace(
                    root=dmp_fx_trace_forward(dmp, self),
                    concrete_args=concrete_args,
                )
                self.root._dmp_wrapped_module = dmp._dmp_wrapped_module
            else:
                # Unwrapped dmp modules and composibility api will enter here.
                graph = super().trace(
                    root=root,
                    concrete_args=concrete_args,
                )
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return graph

    def create_arg(self, a: Any) -> Argument:
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        Adds support for the NoWait type in addition to the default tracer

        Args:
            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.

        Returns:
            Argument: The value ``a`` converted into the appropriate ``Argument``
        """
        if isinstance(a, NoWait):
            return self.create_node(
                "call_function",
                target=NoWait,
                # Ugh. This line seems to be triggering some bug in pyre - so
                # cast instead of fixme.
                args=typing.cast(
                    Tuple[torch.fx.node.Argument, ...], self.create_arg((a._obj,))
                ),
                kwargs={},
                type_expr=NoWait,
            )

        # Not equivalent to when LazyAwaitable.wait() is called in eager. Here can be called earlier, as attr was not requested and this is not guranteed to be torch function
        # TODO(ivankobzarev): support equivalent timing of LazyAwaitable
        if isinstance(a, LazyAwaitable):
            if a._result is None:
                a._result = a.wait()
            return super().create_arg(a._result)

        return super().create_arg(a)

    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Allows trace-ability of non registered modules. This is typically used for Table Batched Embeddings
        made to look like nn.EmbeddingBags
        """

        if hasattr(mod, "_fx_path"):
            return mod._fx_path
        else:
            return super().path_of_module(mod)


def symbolic_trace(
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Dict[str, Any]] = None,
    leaf_modules: Optional[List[str]] = None,
) -> torch.fx.GraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    tracer = Tracer(leaf_modules)
    graph = tracer.trace(root, concrete_args)
    # Ugh. This line seems to be triggering some bug in pyre - so cast instead
    # of fixme.
    return torch.fx.GraphModule(typing.cast(torch.nn.Module, root), graph)
