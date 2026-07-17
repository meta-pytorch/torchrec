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

A second use case, ``split_backward``, splits one op's backward across two
Functions so an output-grad-independent (x-only) step can overlap a long comm:

    y, dummy = op1(x)                    # y=x**2; dummy is a grad-carrying token
    empty    = op2(x, dummy)             # trivial fwd, x-only bwd
    z        = all2all(y)                # op3: long-running comm (real collective)
    loss     = sum(cat([z, empty]))      # op4

op1's x-only gradient factor is computed in op2.backward and handed back to
op1.backward through ``dummy``'s gradient, so it can run alongside op3's comm.
This use case runs multi-process (default nccl, world_size=2) so op3 is a real
all2all; its backward is also an all2all, independent of op2's x-only compute.
Each op is a named ``record_function`` span, and a chrome trace is exported to
``--profile_dir`` (default "."). Inspect it at https://ui.perfetto.dev.

Example usage (swap ``preallocated_output`` for ``split_backward`` to run the
other use case):

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.autograd import Function
from torch.autograd.profiler import record_function
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    cmd_conf,
)
from torchrec.distributed.comm_ops import AllToAllSingle, pg_name
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)

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
        logger.info("run op1.forward with pre=%s x=%s", pre, x)
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
        logger.info("run op1.backward with grad_output=%s x=%s", grad_output, x)
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
        logger.info("run op2.forward with pre=%s x=%s", pre, x)
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
        logger.info("run op2.backward with grad_output=%s x=%s", grad_output, x)
        grad_x = torch.stack(
            [grad_output[2] * 3 * x[0] ** 2, grad_output[3] * 4 * x[1] ** 3]
        )
        grad_pre = grad_output.clone()
        grad_pre[2] = 0
        grad_pre[3] = 0
        return grad_pre, grad_x


class SplitBwdOp1(Function):
    """op1: y = x ** 2, plus a ``dummy`` token output.

    op1's true x-gradient factors as ``dy/dx = 2*x`` (x-only, expensive) times
    the incoming ``grad_y``. We move the x-only factor out of this backward and
    into op2.backward, which hands it back through ``dummy``'s gradient. So
    op1.backward only does the cheap combine ``grad_dummy * grad_y``.
    """

    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(ctx: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("run op1.forward with x.shape=%s", tuple(x.shape))
        y = x**2
        # Token output; its value is never read -- only its gradient carries the
        # x-only factor back from op2.backward. A true ``meta`` tensor can't be
        # used here: autograd requires a gradient's device to match its tensor's,
        # so a meta output would reject op2's cuda gradient. Instead build a
        # zero-storage view -- one real element ``expand``ed to x's shape -- so
        # forward allocates ~nothing while op2.backward still hands back a full
        # cuda gradient of shape ``x.shape`` in the backward pass.
        dummy = x.new_empty(1).expand(x.shape)
        return y, dummy

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(
        ctx: Any, grad_y: torch.Tensor, grad_dummy: torch.Tensor
    ) -> torch.Tensor:
        with record_function("## op1.backward ##"):
            logger.info(
                "run op1.backward with grad_y.device=%s grad_dummy.device=%s",
                grad_y.device,
                grad_dummy.device,
            )
            # grad_dummy == 2*x, precomputed by op2.backward. Cheap combine only.
            grad_x = grad_dummy * grad_y
            return grad_x


class SplitBwdOp2(Function):
    """op2: trivial forward, x-only backward.

    Forward just emits an ``empty`` output consumed by op4 -- the graph edge that
    gets op2.backward invoked. The real work is in backward: it runs the x-only
    factor that op1.backward would otherwise compute (``2*x`` for ``y=x**2``, an
    elementwise op of about the same cost as op1's own backward), and routes the
    result to op1 through ``dummy``'s gradient.
    """

    @staticmethod
    # pyrefly: ignore[bad-override]
    def forward(ctx: Any, x: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
        logger.info("run op2.forward with x.shape=%s", tuple(x.shape))
        ctx.save_for_backward(x)
        # Trivial output; value irrelevant, only used to keep op2 in the graph.
        empty = x.new_empty(0)
        return empty

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(
        ctx: Any, grad_empty: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        (x,) = ctx.saved_tensors
        with record_function("## op2.backward (x-only) ##"):
            logger.info("run op2.backward with x.shape=%s", tuple(x.shape))
            # x-only step: depends on x alone, not on any output grad, so it can
            # run as soon as backward starts and overlap op3's long-running comm.
            # Elementwise ``2*x`` -- about the same cost as op1's own backward.
            heavy = 2 * x
            # ``empty`` does not depend on x, so op2 contributes no grad to x.
            return None, heavy


###################################### benchmark ######################################
@dataclass
class PreallocatedOutputConfig(BenchFuncConfig):
    name: str = ""
    world_size: int = 1
    num_profiles: int = 0
    num_benchmarks: int = 0
    # Demo logs (op forward/backward traces) are emitted at INFO; override with
    # --loglevel to quiet them.
    loglevel: str = "INFO"
    x1: List[float] = field(default_factory=lambda: [1.1, 2.1])
    x2: List[float] = field(default_factory=lambda: [3.1, 4.1])


@register_benchmark(PreallocatedOutputConfig)
def preallocated_output(arg: PreallocatedOutputConfig) -> None:
    """
    op1 and op2 write disjoint slots of one preallocated tensor ``pre``.
    Chaining ``pre = op2(op1(pre, x1), x2)`` -- instead of returning op1's output
    directly -- keeps both op1.backward and op2.backward in the graph.
    """
    logging.basicConfig()
    arg.set_log_level()

    pre = torch.empty(4)
    x1 = torch.tensor(arg.x1, requires_grad=True)
    x2 = torch.tensor(arg.x2, requires_grad=True)

    logger.info("initial: pre=%s", pre)
    pre = Op1.apply(pre, x1)
    logger.info("after op1: pre=%s", pre)
    pre = Op2.apply(pre, x2)
    logger.info("after op2: pre=%s", pre)

    loss = torch.sum(pre)
    logger.info("loss=%s", loss)
    loss.backward()
    logger.info("x1.grad=%s, x2.grad=%s", x1.grad, x2.grad)


@dataclass
class SplitBackwardConfig(BenchFuncConfig):
    name: str = ""
    world_size: int = 2
    num_benchmarks: int = 1
    num_profiles: int = 2
    profile_dir: str = "."
    backend: str = "nccl"
    # Demo logs (op forward/backward traces) are emitted at INFO; override with
    # --loglevel to quiet them.
    loglevel: str = "INFO"
    # Number of elements of ``x`` (and thus of ``y``) that flow through op3's
    # all2all -- larger means heavier comms. Must be divisible by world_size.
    comm_numel: int = 4_194_304  # 2^22 fp32 = 16 MiB per rank


def _split_backward_iter(
    _batch_inputs: List[Dict[str, Any]],
    ctx: MultiProcessContext,
    comm_numel: int,
    **_kwargs: Any,
) -> None:
    """One measured iteration: build the graph and run fwd + bwd.

    Each op is wrapped in ``record_function`` so it is a named span in the chrome
    trace. op1/op2 also annotate their backward internally (see the Functions),
    and op3's all2all backward shows up as a collective under ``## backward ##``.
    """
    x = torch.rand(comm_numel, device=ctx.device, requires_grad=True)

    with record_function("## op1.forward (y, dummy) ##"):
        y, dummy = SplitBwdOp1.apply(x)
    with record_function("## op2.forward (trivial) ##"):
        # pyrefly: ignore[missing-attribute]
        empty = SplitBwdOp2.apply(x, dummy)
    pg = ctx.pg
    assert pg is not None
    with record_function("## op3.forward all2all (comm) ##"):
        # Real differentiable all2all across ranks -- the long-running comm. Its
        # backward is itself an all2all, independent of op2's x-only compute.
        # pyrefly: ignore[missing-attribute]
        z = AllToAllSingle.apply(y, None, None, pg_name(pg), pg.size(), False)
    with record_function("## op4.forward sum(cat) ##"):
        # empty is size 0, so it only routes grad to op2 (not into the sum).
        loss = torch.sum(torch.cat([z, empty]))

    with record_function("## backward ##"):
        loss.backward()

    assert x.grad is not None and torch.allclose(x.grad, 2 * x.detach())


def _split_backward_worker(
    rank: int, world_size: int, arg: SplitBackwardConfig
) -> None:
    # Spawned worker: ensure a stream handler exists so logger output is emitted,
    # then apply the level from the config.
    logging.basicConfig()
    arg.set_log_level()
    assert (
        arg.comm_numel % world_size == 0
    ), f"comm_numel={arg.comm_numel} must be divisible by world_size={world_size}"
    name = arg.name or "split_backward"
    with MultiProcessContext(
        rank=rank, world_size=world_size, backend=arg.backend
    ) as ctx:
        result = benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            benchmark_func_kwargs={
                "ctx": ctx,
                "comm_numel": arg.comm_numel,
            },
            func_to_benchmark=_split_backward_iter,
            rank=rank,
            sample_count=0,
            **arg.benchmark_func_kwargs(name=name),
        )
        if rank == 0:
            logger.info("%s", result)


@register_benchmark(SplitBackwardConfig)
def split_backward(arg: SplitBackwardConfig) -> None:
    """
    Split op1's backward across two autograd Functions.

        y, dummy = op1(x)          # y = x**2; dummy is a grad-carrying token
        empty    = op2(x, dummy)   # trivial fwd; x-only bwd
        z        = all2all(y)      # op3: long-running comm (real collective)
        loss     = sum(cat([z, empty]))   # op4

    op1's x-gradient is ``2*x * grad_y``. The x-only factor ``2*x`` depends on no
    output grad; the ``grad_y`` combine is cheap. We compute the x-only factor in
    op2.backward (an elementwise op of about the same cost as op1's own backward)
    and pass it to op1.backward through ``dummy``'s gradient. op1.backward needs
    both grad_y (from op3's all2all backward) and grad_dummy (from op2), so it
    waits on both -- letting op2's x-only compute overlap op3's comm.

    x feeds both op1 and op2, so x.grad accumulates op1's ``2*x`` (via dummy) and
    op2's direct ``None``, giving the expected ``2*x``.

    Runs multi-process (default nccl, world_size=2) and, with ``profile_dir`` set
    (default "."), exports a chrome trace per the ``num_profiles`` profiled
    iterations -- each op is a named ``record_function`` span in the trace.
    ``comm_numel`` sizes op3's all2all (heavier comms); the trace shows op2's
    x-only compute against op3's comm backward.
    """
    arg.set_log_level()
    run_multi_process_func(
        func=_split_backward_worker,
        world_size=arg.world_size,
        arg=arg,
    )


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
