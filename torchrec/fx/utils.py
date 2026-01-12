#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
from typing import Any, Dict, List, Set

import torch
from torch.fx._symbolic_trace import is_fx_tracing

# Not importing DistributedModelParallel here to avoid circular dependencies as DMP depends on torchrec.fx.tracer
# def dmp_fx_trace_forward(dmp: DistributedModelParallel)


# pyre-ignore
def fake_range():
    # pyre-fixme[16]: Module `_C` has no attribute `_jit_tree_views`.
    return torch._C._jit_tree_views.SourceRangeFactory("", None, 0, 0).make_raw_range(
        0, 1
    )


# pyre-ignore
def dmp_fx_trace_forward(  # noqa: C901
    # pyre-ignore
    dmp,
    tracer: torch.fx.Tracer,
):
    func = dmp._dmp_wrapped_module.forward
    sign: inspect.Signature = inspect.signature(func)

    module_to_type_str: Dict[str, Set[str]] = {}

    def add_if_missing(module: str, type_str: str) -> None:
        if module not in module_to_type_str:
            _set = set()
            _set.add(type_str)
            module_to_type_str[module] = _set
        else:
            s = module_to_type_str[module]
            if type_str not in s:
                s.add(type_str)

    def torch_no_import(t: torch.Type) -> bool:
        return isinstance(
            t, (torch.FloatType, torch.IntType, torch.ComplexType, torch.StringType)
        )

    def torch_typing(t: torch.Type) -> bool:
        return isinstance(
            t,
            (
                torch.TupleType,
                torch.ListType,
                torch.DictType,
                torch.OptionalType,
                torch.AnyType,
            ),
        )

    exec_imports = []
    args_call = ", ".join([f"{p.name}" for p in sign.parameters.values()])

    types = []
    try:
        args_decls: List[str] = []
        for p in sign.parameters.values():
            pann = p.annotation

            ptype = torch.jit.annotations.try_ann_to_type(pann, fake_range())
            types.append(ptype)
            args_decls.append(f"{p.name}: {ptype}")

        while len(types) > 0:
            t = types.pop()
            if torch_no_import(t):
                continue

            t_base_name = f"{t}".split("[")[0]
            if torch_typing(t):
                add_if_missing("typing", t_base_name)
            else:
                if hasattr(t, "__module__") and not torch_no_import(t):
                    m = t.__module__
                    add_if_missing(f"{m}", f"{t}".split("[")[0])

            if hasattr(t, "containedTypes"):
                contained_types = getattr(t, "containedTypes", None)()
                for ctype in contained_types:
                    types.append(ctype)

            if hasattr(t, "getElementType"):
                el_type = getattr(t, "getElementType", None)()

        args_decl = ", ".join(args_decls)

        for m, s in module_to_type_str.items():
            ts = ", ".join(s)
            exec_imports.append(f"from {m} import {ts}")
    except Exception as e:
        print(f"Exception:{e}")
        # Catching here if source is not available to proceed hoping that jit will infer correct types without annotations.
        # Often it fails here when can not access to dataclass generated __init__
        args_decl = args_call

    exec_def_fn_name = "__fx_forward"
    exec_dmp_wrapper_local_name = "_dmp_wrapped_module_local"
    _dmp_wrapped_module_local = dmp
    locals_dict = locals()
    exec_def = f"def {exec_def_fn_name}({args_decl}):\n    return {exec_dmp_wrapper_local_name}({args_call})"

    exec_imports_str = "\n".join(exec_imports)
    pycode = f"{exec_imports_str}\n{exec_def}"

    exec(pycode, locals_dict)  # noqa: P204  Allow use of exec

    wrapper = locals_dict[exec_def_fn_name]
    wrapper.__signature__ = sign

    return wrapper


@torch.fx.wrap
# pyre-ignore
def _fx_marker(s: str, any_proxy_unused: Any) -> None:
    pass


# pyre-ignore
def fx_marker(s: str, any_proxy_unused: Any) -> None:
    if is_fx_tracing():
        _fx_marker(s, any_proxy_unused)


def is_marker_node(node: torch.fx.Node, marker_name: str) -> bool:
    # bool() syntax for pyre
    return bool(
        node.op == "call_function"
        and node.target == _fx_marker
        and isinstance(node.args[0], str)
        and node.args[0] == marker_name
    )


@torch.jit.ignore
def assert_fx_safe(condition: bool, message: str) -> None:
    if not is_fx_tracing():
        assert condition, message
