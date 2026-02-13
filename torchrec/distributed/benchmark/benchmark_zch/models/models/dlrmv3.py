#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torchrec.distributed.comm import get_local_size
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.mc_adapter import McEmbeddingCollectionAdapter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


# Dummy batch data class as an example of features
@dataclass
class Batch(Pipelineable):
    uih_features: KeyedJaggedTensor
    candidates_features: KeyedJaggedTensor


class DLRMv3(nn.Module):
    """
    Wrapper for DLRMv3 model with HSTU
    Making the model congest the inputs in Batch format, and outputs the loss, predictions, and labels
    as specified in the training loop in benchmark_zch.py
    Args:
        hstu_configs (DlrmHSTUConfig): the HSTU configs
        embedding_tables (Dict[str, EmbeddingConfig]): the embedding tables
        is_inference (bool): whether the model is in inference mode
    """

    def __init__(
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool = False,
    ) -> None:
        super().__init__()
        self.dlrm_hstu = DlrmHSTU(
            hstu_configs=hstu_configs,
            embedding_tables=embedding_tables,
            is_inference=is_inference,
        )
        self.eval_flag = False
        self.table_configs: List[EmbeddingConfig] = list(embedding_tables.values())

    def forward(
        self,
        batch: Batch,
    ) -> Tuple[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """
        Args:
            Batch: the batch dataclass, should include the following attributes:
                uih_features (KeyedJaggedTensor): the uih features
                candidates_features (KeyedJaggedTensor): the candidates features
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
                loss (torch.Tensor): the loss
                loss_values (torch.Tensor): the loss values
                pred_logits (torch.Tensor): the predicted logits
                labels (torch.Tensor): the labels
                weights (torch.Tensor): the dummy all-ones weights which does not take effect in this training loop, just used as placeholder
        """
        (
            _,
            __,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        ) = self.dlrm_hstu(
            uih_features=batch.uih_features,
            candidates_features=batch.candidates_features,
        )
        # convert labels to int64 and squeeze to [batch_size, ]
        mt_target_labels = mt_target_labels.squeeze().to(torch.int64)
        # convert the predictions to [batch_size, ]
        mt_target_preds = mt_target_preds.squeeze()
        # return the loss and the predictions
        return sum(aux_losses.values()), (
            aux_losses,
            mt_target_preds.detach(),
            mt_target_labels.detach(),
            mt_target_weights.detach(),
        )

    # pyrefly: ignore[bad-override]
    def eval(self) -> None:
        self.dlrm_hstu.eval()


def make_model_dlrmv3(
    args: argparse.Namespace, configs: Dict[str, Any], device: torch.device
) -> nn.Module:

    hstu_config = DlrmHSTUConfig(
        hstu_num_heads=configs["hstu_num_heads"],
        hstu_attn_linear_dim=configs["hstu_attn_linear_dim"],
        hstu_attn_qk_dim=configs["hstu_attn_qk_dim"],
        hstu_attn_num_layers=configs["hstu_attn_num_layers"],
        hstu_embedding_table_dim=configs["hstu_embedding_table_dim"],
        hstu_transducer_embedding_dim=configs["hstu_transducer_embedding_dim"],
        hstu_group_norm=True,
        hstu_input_dropout_ratio=configs["hstu_input_dropout_ratio"],
        hstu_linear_dropout_rate=configs["hstu_linear_dropout_rate"],
        causal_multitask_weights=configs["causal_multitask_weights"],
    )

    hstu_config.user_embedding_feature_names = configs["user_embedding_feature_names"]
    hstu_config.item_embedding_feature_names = configs["item_embedding_feature_names"]
    hstu_config.uih_post_id_feature_name = configs["uih_post_id_feature_name"]
    hstu_config.uih_weight_feature_name = (
        # pyrefly: ignore[bad-assignment]
        configs["uih_weight_feature_name"]
        if "uih_weight_feature_name" in configs
        else None
    )
    hstu_config.uih_action_time_feature_name = configs["uih_action_time_feature_name"]
    hstu_config.candidates_weight_feature_name = configs[
        "candidates_weight_feature_name"
    ]
    hstu_config.candidates_watchtime_feature_name = configs[
        "candidates_watchtime_feature_name"
    ]
    hstu_config.candidates_querytime_feature_name = configs[
        "candidates_querytime_feature_name"
    ]
    hstu_config.contextual_feature_to_min_uih_length = (
        # pyrefly: ignore[bad-assignment]
        configs["contextual_feature_to_min_uih_length"]
        if "contextual_feature_to_min_uih_length" in configs
        else None
    )
    hstu_config.merge_uih_candidate_feature_mapping = configs[
        "merge_uih_candidate_feature_mapping"
    ]
    hstu_config.hstu_uih_feature_names = configs["hstu_uih_feature_names"]
    hstu_config.hstu_candidate_feature_names = configs["hstu_candidate_feature_names"]
    task_type_list = []
    for i in range(len(configs["multitask_configs"])):
        if configs["multitask_configs"][i]["task_type"] == "regression":
            task_type_list.append(MultitaskTaskType.REGRESSION)
        elif configs["multitask_configs"][i]["task_type"] == "classification":
            task_type_list.append(MultitaskTaskType.BINARY_CLASSIFICATION)
        else:
            raise ValueError(
                f"Invalid task type {configs['multitask_configs'][i]['task_type']}, expected regression or classification"
            )
    hstu_config.multitask_configs = [
        TaskConfig(
            task_name=configs["multitask_configs"][i]["task_name"],
            task_weight=configs["multitask_configs"][i]["task_weight"],
            task_type=task_type_list[i],
        )
        for i in range(len(configs["multitask_configs"]))
    ]
    hstu_config.action_weights = (
        configs["action_weights"] if "action_weights" in configs else None
    )

    table_config = {}
    for i in range(len(configs["user_embedding_feature_names"])):
        if configs["user_embedding_feature_names"][i] == "movie_id":
            feature_names = ["movie_id", "item_movie_id"]
        elif configs["user_embedding_feature_names"][i] == "video_id":
            feature_names = ["video_id", "item_video_id"]
        else:
            feature_names = [configs["user_embedding_feature_names"][i]]
        table_config[configs["user_embedding_feature_names"][i]] = EmbeddingConfig(
            name=configs["user_embedding_feature_names"][i],
            embedding_dim=configs["hstu_embedding_table_dim"],
            num_embeddings=(
                args.num_embeddings
                if args.num_embeddings
                else configs["num_embeddings"]
            ),
            feature_names=feature_names,
        )

    model = DLRMv3(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=False,
    )
    model.dlrm_hstu.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)

    if args.zch_method == "mpzch":
        ec_adapter = (
            McEmbeddingCollectionAdapter(  # TODO: add switch for other ZCH or no ZCH
                tables=list(table_config.values()),
                input_hash_size=args.input_hash_size,
                device=device,
                world_size=get_local_size(),
                zch_method="mpzch",
                mpzch_num_buckets=args.num_buckets,
                mpzch_max_probe=args.max_probe,
            )
        )
        # This is because we want to implement managed collision functions without changing the DLRM_HSTU class. The EmbeddingCollectionAdapter will simulate all the
        # APIs for EmbeddingCollection, and we can use it to replace the EmbeddingCollection in DLRM_HSTU for managed collision functions.
        # pyrefly: ignore[bad-assignment]
        model.dlrm_hstu._embedding_collection = ec_adapter

    return model
