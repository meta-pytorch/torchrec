#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import abc
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, List, Tuple


class BaseArgInfoStep(abc.ABC):
    @abc.abstractmethod
    # pyre-ignore
    def process(self, arg) -> Any:
        raise Exception("Not implemented in the BaseArgInfoStep")

    def __eq__(self, other: object) -> bool:
        """
        Some tests use the equality checks on the ArgInfo and/or CallArgs, so it's
        natural to use dataclasses for ArgInfoStep implementations. However
        Torchrec doesn't like dataclasses: https://github.com/pytorch/pytorch/issues/74909

        So, this class creates a makeshift generic implementation similar to dataclass, but without
        dataclass.
        """
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, field_name) == getattr(other, field_name)
            for field_name in self.__dict__.keys()
        )


@dataclass
class ArgInfo:
    """
    Representation of args from a node.

    Attributes:
        steps (List[ArgInfoStep]): sequence of transformations from input batch.
            Steps can be thought of consequtive transformations on the input, with
            output of previous step used as an input for the next. I.e. for 3 steps
            it is similar to step3(step2(step1(input)))
            See `BaseArgInfoStep` class hierearchy for supported transformations
    """

    steps: List[BaseArgInfoStep]

    def add_step(self, step: BaseArgInfoStep) -> "ArgInfo":
        self.steps.insert(0, step)
        return self

    def append_step(self, step: BaseArgInfoStep) -> "ArgInfo":
        self.steps.append(step)
        return self

    # pyre-ignore[3]
    def process_steps(
        self,
        arg: Any,  # pyre-ignore[2]
    ) -> Any:
        if not self.steps:
            return None
        for step in self.steps:
            arg = step.process(arg)

        return arg


@dataclass
class CallArgs:
    args: List[ArgInfo]
    kwargs: Dict[str, ArgInfo]

    # pyre-ignore[3]
    def build_args_kwargs(
        self, initial_input: Any  # pyre-ignore[2]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        args = [arg.process_steps(initial_input) for arg in self.args]
        kwargs = {
            key: arg.process_steps(initial_input) for key, arg in self.kwargs.items()
        }
        return args, kwargs


@unique
class PipelineState(Enum):
    """
    Pipeline state for the train pipeline.
    """

    IDLE = 0
    CALL_FWD = 1
    CALL_BWD = 2

    def __str__(self) -> str:
        return self.name


@unique
class PipelinePhase(Enum):
    """
    Pipeline phase for the train pipeline

    please:
        1. order the phases in the order of execution of base pipeline.
        2. add notes to explain the phases if needed.

    """

    def __str__(self) -> str:
        return self.value

    def __eq__(self, obj: "PipelinePhase") -> bool:
        return self.value == obj.value

    # placeholder for empty
    NULL = "null"

    # usually the data is first available on CPU when loading from dataloader
    # need to move/copy the input batch to device if using GPU training
    COPY_BATCH_TO_DEVICE = "copy_batch_to_device"

    # input post processing is needed for sparse data dist pipeline, where the sparse features
    # are traced (built) from the ModelInput via fx tracing
    INPUT_POST_PROC = "input_post_proc"

    # the sparse features (AKA, KJTs) are in a jagged format so the data size are unknown to
    # other ranks. so a comms is needed to exchange the data size info, i.e., the splits
    INPUT_SPLITS_DIST = "input_splits_dist"

    # once a rank knows the data size from other ranks (via splits dist), it can initialize
    # a all-to-all comms to exchange the actual data of the sparse features
    # NOTE: the splits have to be available on the host side
    INPUT_DATA_DIST = "input_data_dist"

    # embedding lookup is done in FBGEMM.TBE on each rank
    EMBEDDING_LOOKUP = "embedding_lookup"

    # the embedding lookup results (i.e., the embeddings) are needed in each rank, it's often done
    # with the output dist with an all_to_all comms
    EMBEDDING_OUTPUT_DIST = "embedding_output_dist"

    # A typical DLRM model arch contains sparse arch and dense arch, here we treat the model excluding
    # "sparse modules" as dense part. It actually also includes the dense-sharded embedding tables.
    DENSE_FORWARD = "dense_forward"

    # model's backward usually uses torch.autograd, the embedding modules' backward is handled by TBE
    DENSE_BACKWARD = "dense_backward"

    # on each rank, after dense arch's backward, the gradients are available for the embedding tables
    # a backward of "embedding output dist" is needed to gather the embedding gradients from all ranks
    # to the rank where the embedding table is hosted.
    EMBEDDING_GRAD_DIST = "embedding_grad_dist"

    # TBE backward usually update the embedding table weights inplace
    EMBEDDING_BACKWARD = "embedding_backward"

    # we decouple the embedding update from backward just in case the change is not coupled
    EMBEDDING_UPDATE = "embedding_update"

    # the optimizer step usually only includes the dense module weights
    DENSE_OPTIMIZER_STEP = "dense_optimizer_step"
