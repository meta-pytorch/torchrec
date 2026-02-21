#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Dict, Type, Union

import torch
from torch import nn
from torchrec.distributed.train_pipeline import (
    TrainPipelineBase,
    TrainPipelineFusedSparseDist,
    TrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    EvalPipelineFusedSparseDist,
    EvalPipelineSparseDist,
    PrefetchTrainPipelineSparseDist,
    TrainEvalHybridPipelineBase,
    TrainPipelineSemiSync,
    TrainPipelineSparseDistLite,
    TrainPipelineSparseDistT,
)


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
    emb_lookup_stream: str = "data_dist"
    inplace_copy_batch_to_gpu: bool = False
    apply_jit: bool = False

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
        }

        if self.pipeline == "semi":
            return TrainPipelineSemiSync(
                model=model,
                optimizer=opt,
                device=device,
                start_batch=0,
                apply_jit=self.apply_jit,
            )
        elif self.pipeline == "fused":
            return TrainPipelineFusedSparseDist(
                model=model,
                optimizer=opt,
                device=device,
                emb_lookup_stream=self.emb_lookup_stream,
                apply_jit=self.apply_jit,
                inplace_copy_batch_to_gpu=self.inplace_copy_batch_to_gpu,
            )
        elif self.pipeline == "base":
            assert self.apply_jit is False, "JIT is not supported for base pipeline"

            return TrainPipelineBase(
                model=model,
                optimizer=opt,
                device=device,
                inplace_copy_batch_to_gpu=self.inplace_copy_batch_to_gpu,
            )
        else:
            Pipeline = _pipeline_cls[self.pipeline]
            return Pipeline(
                model=model,
                optimizer=opt,
                device=device,
                # pyrefly: ignore[unexpected-keyword]
                apply_jit=self.apply_jit,
                inplace_copy_batch_to_gpu=self.inplace_copy_batch_to_gpu,
            )
