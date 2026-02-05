#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import ast
import json
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple


MAX_NUM_ARGS_IN_MODULE_CTOR = 5


def print_error_message(
    python_path: str, node: ast.AST, name: str, message: str, severity: str = "warning"
) -> None:
    """
    This function will print linter error in a format that is compatible with
    our internal tools.

    Args:
        python_path: Path to the file with the error
        node: AST node describing snippet of code
        name: Name of the linter error
        message: Error message to show to user

    Optional Args:
        severity: How severe should be considered the error. Default level: 'error'

    Returns:
        None
    """
    lint_item = {
        "path": python_path,
        # pyre-fixme[16]: `AST` has no attribute `lineno`.
        "line": node.lineno,
        # pyre-fixme[16]: `AST` has no attribute `col_offset`.
        "char": node.col_offset + 1,
        "severity": severity,
        "name": name,
        "description": message,
    }
    print(json.dumps(lint_item))


def get_function_args(node: ast.FunctionDef) -> Tuple[List[Any], List[Any]]:
    """
    This functon will process function definition and will extract all
    arguments used by a given function and return all optional and non-optional
    args used by the function.

    Args:
        node: Function node containing function that needs to be analyzed

    Returns:
        (non_optional_args, optional_args): named function args
    """
    assert isinstance(
        node, ast.FunctionDef
    ), "Incorrect node type. Expected ast.FunctionDef, got {}".format(type(node))
    total_args = len(node.args.args)
    default_args = len(node.args.defaults)

    optional_args = []
    non_optional_args = []
    # Handle positional args
    for i in range(total_args):
        if i + default_args < total_args:
            non_optional_args.append(node.args.args[i].arg)
        else:
            optional_args.append(node.args.args[i].arg)

    # Handle named args
    for arg in node.args.kwonlyargs:
        optional_args.append(arg.arg)

    return non_optional_args, optional_args


def check_class_definition(python_path: str, node: ast.ClassDef) -> None:
    """
    This function will run set of sanity checks against class definitions
    and their docstrings.

    Args:
        python_path: Path to the file that is getting checked
        node: AST node with the ClassDef that needs to be checked

    Returns:
        None
    """
    assert isinstance(
        node, ast.ClassDef
    ), "Received invalid node type. Expected ClassDef, got: {}".format(type(node))

    is_TorchRec_module = False
    is_test_file = "tests" in python_path
    for base in node.bases:
        # For now only names and attributes are supported
        if not isinstance(base, ast.Name) and not isinstance(
            base, ast.Attribute
        ):  # pragma: nocover
            continue

        # We assume that TorchRec module has one of the following inheritance patterns:
        # 1. `class SomeTorchRecModule(LazyModuleExtensionMixin, torch.nn.Module)`
        # 2. `class SomeTorchRecModule(torch.nn.Module)`
        # pyre-ignore[16]: `_ast.expr` has no attribute `id`.
        if hasattr(base, "id") and base.id == "LazyModuleExtensionMixin":
            is_TorchRec_module = True
            break
        # pyre-ignore[16]: `_ast.expr` has no attribute `id`.
        elif hasattr(base, "attr") and base.attr == "Module":
            is_TorchRec_module = True
            break

    if not is_TorchRec_module or is_test_file:
        return

    docstring: Optional[str] = ast.get_docstring(node)
    if docstring is None:
        print_error_message(
            python_path,
            node,
            "No docstring found in a TorchRec module",
            "TorchRec modules are required to have a docstring describing how "
            "to use them. Given Module don't have a docstring, please fix this.",
        )
        return

    # Check presence of the example:
    if "Example:" not in docstring:
        print_error_message(
            python_path,
            node,
            "No runnable example in a TorchRec module",
            "TorchRec modules are required to have runnable examples in "
            '"Example:" section. Please fix the docstring',
        )

    # Check correctness of the Args for a class definition:
    required_keywords = ["Args:"]
    missing_keywords = []
    for keyword in required_keywords:
        if keyword not in docstring:
            missing_keywords.append(keyword)

    if len(missing_keywords) > 0:
        print_error_message(
            python_path,
            node,
            "Missing required keywords from TorchRec module",
            "TorchRec modules are required to description of their args and "
            'results in "Args:". '
            "Missing keywords: {}.".format(missing_keywords),
        )

    # Check actual args from the functions
    functions: Dict[str, Tuple[List[Any], List[Any]]] = {}
    function_sub_nodes = {}
    for sub_node in node.body:
        if isinstance(sub_node, ast.FunctionDef):
            assert isinstance(sub_node, ast.FunctionDef)
            functions[sub_node.name] = get_function_args(sub_node)
            function_sub_nodes[sub_node.name] = sub_node

    def check_function(function_name: str) -> None:
        if function_name not in functions:
            return

        if function_name == "__init__":
            # NOTE: -1 to not count the `self` argument.
            num_args = sum([len(args) for args in functions[function_name]]) - 1
            if num_args > MAX_NUM_ARGS_IN_MODULE_CTOR:
                print_error_message(
                    python_path,
                    node,
                    "TorchRec module has too many constructor arguments",
                    "TorchRec module can have at most {} constructor arguments, but this module has {}.".format(
                        MAX_NUM_ARGS_IN_MODULE_CTOR,
                        num_args,
                    ),
                )
        if function_name in functions:
            missing_required_args = []
            missing_optional_args = []
            for arg in functions[function_name][0]:
                # Ignore checks for required self and net args
                if arg == "self" or arg == "net":
                    continue
                assert docstring is not None
                if arg not in docstring:
                    missing_required_args.append(arg)
            for arg in functions[function_name][1]:
                assert docstring is not None
                if arg not in docstring:
                    missing_optional_args.append(arg)
            if len(missing_required_args) > 0 or len(missing_optional_args) > 0:
                print_error_message(
                    python_path,
                    node,
                    "Missing docstring descriptions for {} function arguments.".format(
                        function_name
                    ),
                    (
                        "Missing descriptions for {} function arguments. "
                        "Missing required args: {}, missing optional args: {}"
                    ).format(
                        function_name,
                        missing_required_args,
                        missing_optional_args,
                    ),
                )

    # pyre-ignore[53]
    def check_function_docstring(function_name: str) -> None:
        if function_name not in functions:
            return

        function_docstring: Optional[str] = None
        function_docstring = ast.get_docstring(function_sub_nodes[function_name])
        if function_docstring is None:
            print_error_message(
                python_path,
                node,
                "Missing docstring for {} function".format(function_name),
                "Missing docstring for {} function".format(function_name),
            )
            return

        missing_required_args = []
        missing_optional_args = []
        for arg in functions[function_name][0]:
            # Ignore checks for required self and net args
            if arg == "self" or arg == "net":
                continue
            assert function_docstring is not None
            if arg not in function_docstring:
                missing_required_args.append(arg)
        for arg in functions[function_name][1]:
            assert function_docstring is not None
            if arg not in function_docstring:
                missing_optional_args.append(arg)
        if len(missing_required_args) > 0 or len(missing_optional_args) > 0:
            print_error_message(
                python_path,
                node,
                "Missing docstring descriptions for {} function arguments.".format(
                    function_name
                ),
                (
                    "Missing descriptions for {} function arguments. "
                    "Missing required args: {}, missing optional args: {}"
                ).format(
                    function_name,
                    missing_required_args,
                    missing_optional_args,
                ),
            )
        assert function_docstring is not None
        if "Returns:" not in function_docstring:
            print_error_message(
                python_path,
                node,
                "Missing docstring descriptions for {} function arguments.".format(
                    function_name
                ),
                (
                    "Missing descriptions for {} function arguments. "
                    "Missing Returns section"
                ).format(
                    function_name,
                ),
            )

    check_function("__init__")
    check_function_docstring("forward")


def read_file(path: str) -> str:  # pragma: nocover
    """
    This function simply reads contents of the file. It's moved out to a function
    purely to simplify testing process.

    Args:
        path: File to read.

    Returns:
        content(str): Content of given file.
    """
    return open(path).read()


def linter_one_file(python_path: str) -> None:
    """
    This function will check all Modules defined in the given file for a valid
    documentation based on the AST.

    Input args:
        python_path: Path to the file that need to be verified with the linter.

    Returns:
        None
    """
    python_path = python_path.strip()
    try:
        for node in ast.parse(read_file(python_path)).body:
            if isinstance(node, ast.ClassDef):
                assert isinstance(node, ast.ClassDef)
                check_class_definition(python_path, node)
    except SyntaxError as e:  # pragma: nocover
        # possible failing due to file parsing error
        lint_item = {
            "path": python_path,
            "line": e.lineno,
            "char": e.offset,
            "severity": "warning",
            "name": "syntax-error",
            "description": (
                f"There is a linter parser error with message: {e.msg}. "
                "Please report the diff to torchrec oncall"
            ),
            "bypassChangedLineFiltering": True,
        }
        print(json.dumps(lint_item))


def _make_argparse() -> ArgumentParser:  # pragma: nocover
    parser = ArgumentParser(
        description="TorchRec docstring linter", fromfile_prefix_chars="@"
    )
    parser.add_argument("source_files", nargs="+", help="Path to python source files")

    return parser


def _parse_args() -> Namespace:  # pragma: nocover
    ap = _make_argparse()
    return ap.parse_args()


if __name__ == "__main__":  # pragma: nocover
    args: Namespace = _parse_args()
    for filename in args.source_files:
        linter_one_file(filename)
