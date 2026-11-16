#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import typing


def _is_annot_compatible(prev: object, curr: object) -> bool:
    if prev == curr:
        return True

    if not (prev_origin := typing.get_origin(prev)):
        return False
    if not (curr_origin := typing.get_origin(curr)):
        return False

    if prev_origin != curr_origin:
        return False

    prev_args = typing.get_args(prev)
    curr_args = typing.get_args(curr)
    if len(prev_args) != len(curr_args):
        return False

    for prev_arg, curr_arg in zip(prev_args, curr_args):
        if not _is_annot_compatible(prev_arg, curr_arg):
            return False

    return True


def is_signature_compatible(
    previous_signature: inspect.Signature,
    current_signature: inspect.Signature,
) -> bool:
    """Check if two signatures are compatible.

    Args:
        sig1: The first signature.
        sig2: The second signature.

    Returns:
        True if the signatures are compatible, False otherwise.

    """

    # If current signature has less parameters than expected signature
    # BC is automatically broken, no need to check further
    if len(previous_signature.parameters) > len(current_signature.parameters):
        return False

    # Check order of positional arguments
    expected_args = list(previous_signature.parameters.values())
    current_args = list(current_signature.parameters.values())

    # Store the names of all keyword only arguments
    # to check if all expected keyword only arguments
    # are present in current signature
    expected_keyword_only_args = set()
    current_keyword_only_args = set()

    expected_args_len = len(expected_args)

    for i in range(len(current_args)):
        current_arg = current_args[i]
        if current_arg.kind == current_arg.KEYWORD_ONLY:
            current_keyword_only_args.add(current_arg.name)

        if i >= expected_args_len:
            continue

        expected_arg = expected_args[i]

        # If the kinds of arguments are different, BC is broken
        # unless current arg is a keyword argument
        if expected_arg.kind != current_arg.kind:
            if expected_arg.kind == expected_arg.VAR_KEYWORD:
                # Any arg can be inserted before **kwargs and still maintain BC
                continue
            else:
                return False

        # Potential positional arguments need to have the same name
        # keyword only arguments can be mixed up
        if expected_arg.kind == expected_arg.POSITIONAL_OR_KEYWORD:
            if expected_arg.name != current_arg.name:
                return False

            # Positional arguments need to have the same type annotation
            # TODO: Account for Union Types?
            if expected_arg.annotation != current_arg.annotation:
                return False

            # Positional arguments need to have the same default value
            if expected_arg.default != current_arg.default:
                return False
        elif expected_arg.kind == expected_arg.KEYWORD_ONLY:
            expected_keyword_only_args.add(expected_arg.name)

    # All kwargs in expected signature must be present in current signature
    for kwarg in expected_keyword_only_args:
        if kwarg not in current_keyword_only_args:
            return False

    # TODO: Account for Union Types?
    if not _is_annot_compatible(
        previous_signature.return_annotation, current_signature.return_annotation
    ):
        return False
    return True
