#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from collections import OrderedDict
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple, Union

import torch
from torch.nn.modules.module import _IncompatibleKeys
from torch.profiler import record_function
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.types import CallArgs
from torchrec.streamable import Pipelineable

logger: logging.Logger = logging.getLogger(__name__)


class NoOpStream:
    """No-Op Context manager that takes in a stream"""

    def __init__(self, stream: Optional[torch.Stream]) -> None:
        self._stream = stream

    def __enter__(self) -> "NoOpStream":
        """Return `self` upon entering the runtime context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class PipelinedPostproc(torch.nn.Module):
    """
    Wrapper around postproc module found during model graph traversal for sparse data dist
    pipelining. In addition to the original module, it encapsulates information needed for
    execution such as list of ArgInfo and the current training pipeline context.

    Args:
        postproc_module (torch.nn.Module): postproc module to run
        fqn (str): fqn of the postproc module in the model being pipelined
        args (CallArgs): CallArgs for the postproc module
        context (TrainPipelineContext): Training context for the next iteration / batch

    Returns:
        Any

    Example:
        postproc = PipelinedPostproc(postproc_module, fqn, args, context)
        # module-swap with pipeliend postproc
        setattr(model, fqn, postproc)
    """

    _FORCE_STATE_DICT_LOAD = True

    def __init__(
        self,
        postproc_module: torch.nn.Module,
        fqn: str,
        args: CallArgs,
        context: TrainPipelineContext,
        # TODO: make streams non-optional - skipping now to avoid ripple effect
        default_stream: Optional[torch.Stream],
        dist_stream: Optional[torch.Stream],
    ) -> None:
        super().__init__()
        self._postproc_module = postproc_module
        self._fqn = fqn
        self._args = args
        self._context = context
        self._default_stream = default_stream
        self._dist_stream = dist_stream
        if not default_stream:
            logger.warning(
                f"Postproc module {fqn} has no default stream. "
                "This may cause race conditions and NaNs during training!"
            )
        if not dist_stream:
            logger.warning(
                f"Postproc module {fqn} has no dist stream. "
                "This may cause race conditions and NaNs during training!"
            )

        if self._dist_stream:
            device: torch.device = self._dist_stream.device
            self._stream_context = (
                torch.get_device_module(device).stream
                if device.type in ["cuda", "mtia"]
                else torch.cuda.stream
            )
        else:
            self._stream_context = NoOpStream

    @property
    def postproc_module(self) -> torch.nn.Module:
        return self._postproc_module

    @property
    def fqn(self) -> str:
        return self._fqn

    def forward(self, *input, **kwargs) -> Any:
        """
        Args:
            Any args and kwargs during model fwd
            During _start_data_dist, input[0] contains the current data
        Returns:
            Any
        """
        if self._fqn in self._context.postproc_fwd_results:
            # This should only be hit in two cases:
            # 1) During model forward
            # During model forward, avoid duplicate work
            # by returning the cached result from previous
            # iteration's _start_data_dist
            # 2) During _start_data_dist when postproc module is
            # shared by more than one args. e.g. if we have
            # postproc_out_a = postproc_a(input)
            # postproc_out_b = postproc_b(postproc_out_a) <- postproc_a shared
            # postproc_out_c = postproc_c(postproc_out_a) <-^
            # When processing postproc_b, we cache value of postproc_a(input)
            # so when processing postproc_c, we can reuse postproc_a(input)
            res = self._context.postproc_fwd_results[self._fqn]
            return res

        # Everything below should only be called during _start_data_dist stage

        # Build up arg and kwargs from recursive call to pass to postproc module
        # Arguments to postproc module can be also be a derived product
        # of another postproc module call, as long as module is pipelineable

        # Use input[0] as _start_data_dist only passes 1 arg
        args, kwargs = self._args.build_args_kwargs(input[0])

        with record_function(
            f"## input_postproc {type(self.postproc_module)} {self._context.index} ##"
        ):
            # should be no-op as we call this in dist stream
            # pyrefly: ignore[bad-argument-type]
            with self._stream_context(self._dist_stream):
                res = self._postproc_module(*args, **kwargs)

            # Ensure postproc modules output is safe to use from default stream later
            if self._default_stream and self._dist_stream:
                self._default_stream.wait_stream(self._dist_stream)

                if isinstance(res, (torch.Tensor, Pipelineable, Iterable, Dict)):
                    # Result from module forward might be a complex type such as
                    # Tuple[KeyedJaggedTensor, Dict[str, torch.Tensor]]
                    # In this case, we need to first iterate over each element of tuple
                    # and call record_stream on first item as KJT is Pipelineable
                    # for the second item (Dict), we iterate over the values and call
                    # record_stream accordingly.

                    PipelinedPostproc.recursive_record_stream(res, self._default_stream)
                elif self._context.index == 0:
                    logger.warning(
                        f"Result of postproc module {self._fqn} is of type {type(res)}. "
                        "We currently expect it to be a Tensor, Pipelineable, Iterable, "
                        "or Dict to handle memory safety. If your output is not of this "
                        "type, please add support for it above. Otherwise you might run "
                        "into NaNs or CUDA Illegal Memory issues during training!"
                    )

            # pyrefly: ignore[bad-argument-type]
            with self._stream_context(self._default_stream):
                # Cache results, only during _start_data_dist
                self._context.postproc_fwd_results[self._fqn] = res

            return res

    @property
    def args(self) -> CallArgs:
        return self._args

    def set_context(self, context: TrainPipelineContext) -> None:
        self._context = context

    def get_context(self) -> TrainPipelineContext:
        return self._context

    def named_modules(
        self,
        memo: Optional[Set[torch.nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Module]]:
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            # This is needed because otherwise the rewrite won't find the existing postproc, and will create a new one
            # Also, `named_modules` need to include self - see base implementation in the nn.modules.Module
            yield prefix, self
            # Difference from base implementation is here - the child name (_postproc_module) is not added to the prefix
            yield from self._postproc_module.named_modules(
                memo, prefix, remove_duplicate
            )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from self._postproc_module.named_parameters(
            prefix,
            recurse,
            remove_duplicate,
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self._postproc_module.named_buffers(
            prefix, recurse, remove_duplicate
        )

    # pyrefly: ignore[bad-override]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        # super().state_dict(destination, prefix, keep_vars)
        if destination is None:
            destination = OrderedDict()
            # pyrefly: ignore[missing-attribute]
            destination._metadata = OrderedDict()
        self._postproc_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    # pyrefly: ignore[bad-override]
    def load_state_dict(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        strict: bool = True,
    ) -> _IncompatibleKeys:
        return self._postproc_module.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def recursive_record_stream(
        res: Union[torch.Tensor, Pipelineable, Iterable[Any], Dict[Any, Any]],
        stream: torch.Stream,
    ) -> None:
        if isinstance(res, torch.Tensor) and res.device.type in ["cuda", "mtia"]:
            res.record_stream(stream)
        elif isinstance(res, Pipelineable):
            res.record_stream(stream)
        elif isinstance(res, (list, tuple)):
            for v in res:
                PipelinedPostproc.recursive_record_stream(v, stream)
        elif isinstance(res, dict):
            for v in res.values():
                PipelinedPostproc.recursive_record_stream(v, stream)
