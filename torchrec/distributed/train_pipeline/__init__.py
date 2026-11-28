#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchrec.distributed.train_pipeline.gradient_accumulation import (
    GradientAccumulationConfig as GradientAccumulationConfig,
    GradientAccumulationWrapper as GradientAccumulationWrapper,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    In as In,
    Out as Out,
    TrainPipelineContext as TrainPipelineContext,
    TrainPipelineContextLite as TrainPipelineContextLite,
)
from torchrec.distributed.train_pipeline.pipeline_stage import (
    SparseDataDistUtil as SparseDataDistUtil,
    StageOut as StageOut,
)
from torchrec.distributed.train_pipeline.tracing import (
    ArgInfoStepFactory as ArgInfoStepFactory,
    Tracer as Tracer,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    EvalPipelineSparseDist as EvalPipelineSparseDist,
    PrefetchTrainPipelineSparseDist as PrefetchTrainPipelineSparseDist,
    StagedTrainPipeline as StagedTrainPipeline,
    TorchCompileConfig as TorchCompileConfig,
    TrainPipeline as TrainPipeline,
    TrainPipelineBase as TrainPipelineBase,
    TrainPipelineFusedSparseDist as TrainPipelineFusedSparseDist,
    TrainPipelinePT2 as TrainPipelinePT2,
    TrainPipelineSparseDist as TrainPipelineSparseDist,
    TrainPipelineSparseDistCompAutograd as TrainPipelineSparseDistCompAutograd,
)
from torchrec.distributed.train_pipeline.types import (
    ArgInfo as ArgInfo,
    CallArgs as CallArgs,
)
from torchrec.distributed.train_pipeline.utils import (
    _override_input_dist_forwards,  # noqa
    _rewrite_model,  # noqa
    _start_data_dist,  # noqa
    _to_device,  # noqa
    _wait_for_batch,  # noqa
    DataLoadingThread as DataLoadingThread,
)
