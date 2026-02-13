#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @nolint


import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import fbgemm_gpu.sparse_ops  # noqa: F401

import torch
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.utils import Batch
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.fx.tracer import Tracer
from torchrec.inference.modules import (
    PredictFactory,
    PredictModule,
    quantize_inference_model,
    shard_quant_model,
)
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)


def create_training_batch(args) -> Batch:
    return RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=args.batch_size,
        hash_size=args.num_embedding_features,
        ids_per_feature=1,
        num_dense=len(DEFAULT_INT_NAMES),
    ).batch_generator._generate_batch()


# OSS Only


@dataclass
class DLRMModelConfig:
    """
    Model Config for specifying DLRM model parameters.
    """

    dense_arch_layer_sizes: List[int]
    dense_in_features: int
    embedding_dim: int
    id_list_features_keys: List[str]
    num_embeddings_per_feature: List[int]
    num_embeddings: int
    over_arch_layer_sizes: List[int]
    sample_input: Batch


class DLRMPredictModule(PredictModule):
    """
    nn.Module to wrap DLRM model to use for inference.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (List[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        id_list_features_keys (List[str]): the names of the sparse features. Used to
            construct a batch for inference.
        dense_device: (Optional[torch.device]).
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        id_list_features_keys: List[str],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        module = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        # pyrefly: ignore[bad-argument-type]
        super().__init__(module, dense_device)

        self.id_list_features_keys: List[str] = id_list_features_keys

    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch (Dict[str, torch.Tensor]): currently expects input dense features
                to be mapped to the key "float_features" and input sparse features
                to be mapped to the key "id_list_features".

        Returns:
            Dict[str, torch.Tensor]: output of inference.
        """

        try:
            logits = self.predict_module(
                batch["float_features"],
                KeyedJaggedTensor(
                    keys=self.id_list_features_keys,
                    lengths=batch["id_list_features.lengths"],
                    values=batch["id_list_features.values"],
                ),
            )
            predictions = logits.sigmoid()
        except Exception as e:
            logger.info(e)
            raise e

        # Flip predictions tensor to be 1D. TODO: Determine why prediction shape
        # can be 2D at times (likely due to input format?)
        predictions = predictions.reshape(
            [
                predictions.size()[0],
            ]
        )

        return {
            "default": predictions.to(torch.device("cpu"), non_blocking=True).float()
        }


class DLRMPredictFactory(PredictFactory):
    """
    Factory Class for generating TorchScript DLRM Model for C++ inference.

    Args:
        model_config (DLRMModelConfig): model config

    """

    def __init__(self, model_config: DLRMModelConfig) -> None:
        self.model_config = model_config

    # pyrefly: ignore[bad-override]
    def create_predict_module(self, world_size: int, device: str) -> torch.nn.Module:
        logging.basicConfig(level=logging.INFO)
        set_propogate_device(True)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.model_config.embedding_dim,
                num_embeddings=(
                    self.model_config.num_embeddings_per_feature[feature_idx]
                    if self.model_config.num_embeddings is None
                    else self.model_config.num_embeddings
                ),
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                self.model_config.id_list_features_keys
            )
        ]
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            id_list_features_keys=self.model_config.id_list_features_keys,
            # pyrefly: ignore[bad-argument-type]
            dense_device=device,
        )

        quant_model = quantize_inference_model(module)
        sharded_model, _ = shard_quant_model(
            quant_model, compute_device=device, sharding_device=device
        )

        batch = {}
        batch["float_features"] = self.model_config.sample_input.dense_features.to(
            device
        )
        batch["id_list_features.lengths"] = (
            self.model_config.sample_input.sparse_features.lengths().to(device)
        )
        batch["id_list_features.values"] = (
            self.model_config.sample_input.sparse_features.values().to(device)
        )

        sharded_model(batch)

        tracer = Tracer(leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"])

        graph = tracer.trace(sharded_model)
        gm = torch.fx.GraphModule(sharded_model, graph)

        gm(batch)
        scripted_gm = torch.jit.script(gm)
        scripted_gm(batch)
        return scripted_gm

    # pyrefly: ignore[bad-override]
    def batching_metadata(self) -> Dict[str, str]:
        return {
            "float_features": "dense",
            "id_list_features": "sparse",
        }

    def result_metadata(self) -> str:
        return "dict_of_tensor"

    def run_weights_independent_tranformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        return predict_module

    def run_weights_dependent_transformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that depends on weights of the predict module. e.g. lowering to a backend.
        """
        return predict_module
