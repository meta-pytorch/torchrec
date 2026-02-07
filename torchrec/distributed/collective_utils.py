#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This file contains utilities for constructing collective based control flows.
"""

from functools import wraps
from typing import Any, Callable, cast, Optional, TypeVar

import torch.distributed as dist


def is_leader(pg: Optional[dist.ProcessGroup], leader_rank: int = 0) -> bool:
    """
    Checks if the current processs is the leader.

    Args:
        pg (Optional[dist.ProcessGroup]): the process's rank within the pg is used to
            determine if the process is the leader. pg being None implies that the
            process is the only member in the group (e.g. a single process program).
        leader_rank (int): the definition of leader (defaults to 0). The caller can
            override it with a context-specific definition.
    """
    if pg is None:
        return leader_rank == 0
    return pg.rank() == leader_rank


T = TypeVar("T")


def invoke_on_rank_and_broadcast_result(
    pg: dist.ProcessGroup,
    rank: int,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Invokes a function on the designated rank and broadcasts the result to all
    members within the group.

    Example::

        id = invoke_on_rank_and_broadcast_result(pg, 0, allocate_id)
    """
    if pg.rank() == rank:
        res = func(*args, **kwargs)
        object_list = [res]
    else:
        object_list = [None]
    if pg.size() > 1:
        dist.broadcast_object_list(object_list, rank, group=pg)
    return cast(T, object_list[0])


def run_on_leader(pg: dist.ProcessGroup, rank: int):
    def callable(func: Callable[..., T]) -> T:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return invoke_on_rank_and_broadcast_result(pg, rank, func, *args, **kwargs)

        return wrapped

    return callable
