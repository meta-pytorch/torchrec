#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Chained-autograd demo for the preallocated-output use case: two autograd
Functions both write disjoint slots of the same preallocated tensor, chained so
that op2's forward consumes op1's output. This keeps both backward passes in the
graph -- the SSD+VBE fix analyzed in
md-docs/tech-docs/ssd_vbe_bwd_issue_analysis.md.

    pre = torch.empty(4)                 # preallocated output
    pre = op1(pre, x1)                   # pre[0]=x1[0], pre[1]=x1[1]**2
    pre = op2(pre, x2)                   # pre[2]=x2[0]**3, pre[3]=x2[1]**4
    loss = pre.sum()

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt \
        fbcode//torchrec/distributed/benchmark:benchmark_chained_autograd -- \
        preallocated_output

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_chained_autograd \
        preallocated_output
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

import torch
from torch.autograd import Function
from torchrec.distributed.benchmark.base import BenchFuncConfig, cmd_conf

logger: logging.Logger = logging.getLogger(__name__)

# pyrefly: ignore[missing-argument]
_cc = cmd_conf()


def register_benchmark(
    config: type[BenchFuncConfig],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator factory: register a function as a CLI subcommand bound to the given
    config class. The function name becomes the subcommand. Define the config
    first, then:

    @register_benchmark(MyConfig)
    def my_bench(arg: MyConfig) -> None: ...
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        func.__annotations__ = {"arg": config, "return": None}
        # pyrefly: ignore[missing-attribute]
        _cc.register(func)
        return func

    return decorator


#################################### autograd ops ####################################
class Op1(Function):
    """Writes the first two slots of ``pre``: pre[0]=x[0], pre[1]=x[1]**2."""

    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(ctx: Any, pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        print(f"run op1.forward with pre={pre} x={x}")
        pre[0] = x[0]
        pre[1] = x[1] ** 2
        ctx.save_for_backward(x)
        # `pre` is an input mutated in place and returned; autograd needs to know
        ctx.mark_dirty(pre)
        return pre

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x,) = ctx.saved_tensors
        print(f"run op1.backward with grad_output={grad_output} x={x}")
        grad_x = torch.stack([grad_output[0], grad_output[1] * 2 * x[1]])
        # op1 overwrote pre[0:2], so those slots do not depend on the incoming
        # `pre`; zero them before threading the grad back to the previous op
        grad_pre = grad_output.clone()
        grad_pre[0] = 0
        grad_pre[1] = 0
        return grad_pre, grad_x


class Op2(Function):
    """Writes the last two slots of ``pre``: pre[2]=x[0]**3, pre[3]=x[1]**4."""

    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(ctx: Any, pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        print(f"run op2.forward with pre={pre} x={x}")
        pre[2] = x[0] ** 3
        pre[3] = x[1] ** 4
        ctx.save_for_backward(x)
        ctx.mark_dirty(pre)
        return pre

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x,) = ctx.saved_tensors
        print(f"run op2.backward with grad_output={grad_output} x={x}")
        grad_x = torch.stack(
            [grad_output[2] * 3 * x[0] ** 2, grad_output[3] * 4 * x[1] ** 3]
        )
        grad_pre = grad_output.clone()
        grad_pre[2] = 0
        grad_pre[3] = 0
        return grad_pre, grad_x


###################################### benchmark ######################################
@dataclass
class PreallocatedOutputConfig(BenchFuncConfig):
    name: str = ""
    world_size: int = 1
    num_profiles: int = 0
    num_benchmarks: int = 0
    x1: List[float] = field(default_factory=lambda: [1.1, 2.1])
    x2: List[float] = field(default_factory=lambda: [3.1, 4.1])


@register_benchmark(PreallocatedOutputConfig)
def preallocated_output(arg: PreallocatedOutputConfig) -> None:
    """
    op1 and op2 write disjoint slots of one preallocated tensor ``pre``.
    Chaining ``pre = op2(op1(pre, x1), x2)`` -- instead of returning op1's output
    directly -- keeps both op1.backward and op2.backward in the graph.
    """
    arg.set_log_level()

    pre = torch.empty(4)
    x1 = torch.tensor(arg.x1, requires_grad=True)
    x2 = torch.tensor(arg.x2, requires_grad=True)

    print(f"initial: pre={pre}")
    pre = Op1.apply(pre, x1)
    print(f"after op1: pre={pre}")
    pre = Op2.apply(pre, x2)
    print(f"after op2: pre={pre}")

    loss = torch.sum(pre)
    print(f"loss={loss}")
    loss.backward()
    print(f"x1.grad={x1.grad}, x2.grad={x2.grad}")


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
