#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Any, Dict, Type, Union

import torch
from torch import nn
from torchrec.distributed.train_pipeline import (
    TrainPipelineBase,
    TrainPipelineFusedSparseDist,
    TrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.experimental_pipelines import (
    TrainEvalHybridPipelineBase,
    TrainPipelineSparseDistBwdOpt,
    TrainPipelineSparseDistEmbStash,
    TrainPipelineSparseDistOptStash,
    TrainPipelineSparseDistT,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    EvalPipelineFusedSparseDist,
    EvalPipelineSparseDist,
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSemiSync,
    TrainPipelineSparseDistLite,
)
from torchrec.distributed.types import ShardingType


@dataclass
class PipelineConfig:
    """
    Configuration for training pipelines.

    This class defines the parameters for configuring the training pipeline.

    Args:
        pipeline (str): The type of training pipeline to use. Options include:
            - "base": Basic training pipeline
            - "sparse": Pipeline optimized for sparse operations
            - "fused": Pipeline with fused sparse distribution
            - "semi": Semi-synchronous training pipeline
            - "prefetch": Pipeline with prefetching for sparse distribution
            Default is "base".
        emb_lookup_stream (str): The stream to use for embedding lookups.
            Only used by certain pipeline types (e.g., "fused").
            Default is "data_dist".
        apply_jit (bool): Whether to apply JIT (Just-In-Time) compilation to the model.
            Default is False.
    """

    pipeline: str = "base"
    enable_inplace_copy_batch: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_kwargs(self, **default_kwargs) -> Dict[str, Any]:
        kwargs = default_kwargs | self.kwargs
        if "sharding_type" in kwargs:
            kwargs["sharding_type"] = ShardingType(kwargs["sharding_type"])
        if self.pipeline in ("base", "sparse", "sparse_lite"):
            for key in ("site_fqn", "sharding_type"):
                if key in kwargs:
                    kwargs.pop(key)
        if self.pipeline in ("sparse-emb-stash",):
            kwargs.pop("sharding_type", None)
        return kwargs

    def generate_pipeline(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        device: torch.device,
    ) -> Union[TrainPipelineBase, TrainPipelineSparseDist]:
        """
        Generate a training pipeline instance based on the configuration.

        This function creates and returns the appropriate training pipeline object
        based on the pipeline type specified. Different pipeline types are optimized
        for different training scenarios.

        Args:
            pipeline_type (str): The type of training pipeline to use. Options include:
                - "base": Basic training pipeline
                - "sparse": Pipeline optimized for sparse operations
                - "fused": Pipeline with fused sparse distribution
                - "semi": Semi-synchronous training pipeline
                - "prefetch": Pipeline with prefetching for sparse distribution
            emb_lookup_stream (str): The stream to use for embedding lookups.
                Only used by certain pipeline types (e.g., "fused").
            model (nn.Module): The model to be trained.
            opt (torch.optim.Optimizer): The optimizer to use for training.
            device (torch.device): The device to run the training on.
            apply_jit (bool): Whether to apply JIT (Just-In-Time) compilation to the model.
                Default is False.

        Returns:
            Union[TrainPipelineBase, TrainPipelineSparseDist]: An instance of the
            appropriate training pipeline class based on the configuration.

        Raises:
            RuntimeError: If an unknown pipeline type is specified.
        """

        _pipeline_cls: Dict[
            str, Type[Union[TrainPipelineBase, TrainPipelineSparseDist]]
        ] = {
            "base": TrainPipelineBase,
            "sparse_lite": TrainPipelineSparseDistLite,
            "sparse": TrainPipelineSparseDist,
            "fused": TrainPipelineFusedSparseDist,
            "semi": TrainPipelineSemiSync,
            "prefetch": PrefetchTrainPipelineSparseDist,
            "hybrid_base": TrainEvalHybridPipelineBase,
            "eval-sdd": EvalPipelineSparseDist,
            "eval-fused": EvalPipelineFusedSparseDist,
            "sparse-threading": TrainPipelineSparseDistT,
            "sparse-bwd-opt": TrainPipelineSparseDistBwdOpt,
            "sparse-opt-stash": TrainPipelineSparseDistOptStash,
            "sparse-emb-stash": TrainPipelineSparseDistEmbStash,
        }

        match self.pipeline:
            case "semi":
                return TrainPipelineSemiSync(
                    model=model,
                    optimizer=opt,
                    device=device,
                    **self.get_kwargs(start_batch=0),
                )
            case "fused":
                return TrainPipelineFusedSparseDist(
                    model=model,
                    optimizer=opt,
                    device=device,
                    enable_inplace_copy_batch=self.enable_inplace_copy_batch,
                    **self.get_kwargs(emb_lookup_stream="data_dist"),
                )
            case _:
                Pipeline = _pipeline_cls[self.pipeline]
                # pyrefly: ignore[unexpected-keyword]
                return Pipeline(
                    model=model,
                    optimizer=opt,
                    device=device,
                    enable_inplace_copy_batch=self.enable_inplace_copy_batch,
                    **self.get_kwargs(),
                )
