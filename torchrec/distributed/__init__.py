#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Distributed

Torchrec distributed provides the necessary modules and operations to enable model parallelism.

These include:

* model parallelism through `DistributedModelParallel`.
* collective operations for comms, including All-to-All and Reduce-Scatter.

  * collective operations wrappers for sparse features, KJT, and various embedding
    types.

* sharded implementations of various modules including `ShardedEmbeddingBag` for
  `nn.EmbeddingBag`, `ShardedEmbeddingBagCollection` for `EmbeddingBagCollection`

  * embedding sharders that define sharding for any sharded module implementation.
  * support for various compute kernels, which are optimized for compute device
    (CPU/GPU) and may include batching together embedding tables and/or optimizer
    fusion.

* pipelined training through `TrainPipelineSparseDist` that overlaps dataloading
  device transfer (copy to GPU), inter*device communications (input_dist), and
  computation (forward, backward) for increased performance.
* quantization support for reduced precision training and inference.

"""

from torchrec.distributed.comm import (
    get_local_rank as get_local_rank,
    get_local_size as get_local_size,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel as DistributedModelParallel,
)
from torchrec.distributed.train_pipeline import (
    DataLoadingThread as DataLoadingThread,
    EvalPipelineSparseDist as EvalPipelineSparseDist,
    PrefetchTrainPipelineSparseDist as PrefetchTrainPipelineSparseDist,
    TrainPipeline as TrainPipeline,
    TrainPipelineBase as TrainPipelineBase,
    TrainPipelineSparseDist as TrainPipelineSparseDist,
)
from torchrec.distributed.types import (
    Awaitable as Awaitable,
    ModuleSharder as ModuleSharder,
    NoWait as NoWait,
    ParameterSharding as ParameterSharding,
    ShardedModule as ShardedModule,
    ShardedTensor as ShardedTensor,
    ShardingEnv as ShardingEnv,
    ShardingPlanner as ShardingPlanner,
)
from torchrec.distributed.utils import (
    get_unsharded_module_names as get_unsharded_module_names,
    sharded_model_copy as sharded_model_copy,
)
