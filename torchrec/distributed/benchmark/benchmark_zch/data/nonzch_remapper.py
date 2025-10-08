#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class Batch(Pipelineable):
    batch_example_attribute: KeyedJaggedTensor

    def get_dict(self) -> Dict[str, KeyedJaggedTensor]:
        return {
            "batch_example_attribute": self.batch_example_attribute,
        }


class NonZchModTableRemapperModule(object):
    """
    Managed Collision Module with mod remapping
    Given a list of input features, the module will return a list of remapped features
    For each input feature value x, the remapped value is x % num_embeddings
    """

    def __init__(
        self,
        zch_size: int,
        input_hash_size: int,
        device: torch.device,
    ) -> None:
        self._zch_size = zch_size
        self._input_hash_size = input_hash_size
        self._hash_zch_identities: torch.Tensor = (
            torch.zeros(self._zch_size, dtype=torch.int64).to(device) - 1
        )
        self.table_name_on_device_remapped_ids_dict: Dict[str, torch.Tensor] = (
            {}
        )  # {table_name: on_device_remapped_ids}
        ## on-device input ids
        self.table_name_on_device_input_ids_dict: Dict[str, torch.Tensor] = (
            {}
        )  # {table_name: input JT values that maps to the current rank}

    def remap(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        with torch.no_grad():
            remapped_features: Dict[str, JaggedTensor] = {}
            for name, feature in features.items():
                values = feature.values()
                self.table_name_on_device_input_ids_dict[name] = values.clone()
                remapped_ids = values % self._zch_size
                # update identity table _hash_zch_identities
                ## if the slot on _hash_zch_identities indezed by remapped_ids is -1, update it to the input value
                ## if the slot on _hash_zch_identities indezed by remapped_ids is not -1, skip the update
                ## do that in one pass with torch operations
                self._hash_zch_identities[remapped_ids] = torch.where(
                    self._hash_zch_identities[remapped_ids] == -1,
                    values,
                    self._hash_zch_identities[remapped_ids],
                )
                self.table_name_on_device_remapped_ids_dict[name] = remapped_ids.clone()
                remapped_features[name] = JaggedTensor(
                    values=remapped_ids,
                    lengths=feature.lengths(),
                    offsets=feature.offsets(),
                    weights=feature.weights_or_none(),
                )
            return remapped_features


class NonZchModRemapperModule(object):
    """
    Managed Collision Module with mod remapping
    Given a list of input features, the module will return a list of remapped features
    For each input feature value x, the remapped value is x % num_embeddings
    """

    def __init__(
        self,
        table_configs: List[Union[EmbeddingBagConfig, EmbeddingConfig]],
        input_hash_size: int,
        device: torch.device,
    ) -> None:
        self.mod_modules: Dict[str, NonZchModTableRemapperModule] = {}
        self.feature_table_name_dict: Dict[str, str] = {}  # {feature_name: table_name}
        for table_config in table_configs:
            table_name = table_config.name
            feature_names = table_config.feature_names
            for feature_name in feature_names:
                self.feature_table_name_dict[feature_name] = table_name
            self.mod_modules[table_name] = NonZchModTableRemapperModule(
                zch_size=table_config.num_embeddings,
                input_hash_size=input_hash_size,
                device=device,
            )
        self._input_hash_size = input_hash_size

    def get_batch_kjt_dict(self, batch: Batch) -> Dict[str, KeyedJaggedTensor]:
        """
        Get the KJT in each batch
        Parameters:
            batch: the batch whose KJT is ought to be fetched
        Returns:
            batch_kjt_dict: a dictionary of [batch_attribute_name: KeyedJaggedTensor]
            where only attributes whose values are KeyedJaggedTensor are fetched.
        """
        batch_kjt_dict = {}  # create a dictionary for return
        batch_attr_dict = vars(batch)  # get batch's attributes and values
        for batch_attr_name, batch_attr_value in batch_attr_dict.items():
            if isinstance(
                batch_attr_value, KeyedJaggedTensor
            ):  # only fetch attributes whose values are KeyedJaggedTensor
                batch_kjt_dict[batch_attr_name] = batch_attr_value
        return batch_kjt_dict

    def remap(self, batch: Batch) -> Batch:
        # for all the attributes under batch, like batch.uih_features, batch.candidates_features,
        # get the kjt as a dict, and remap the kjt
        # where batch is a dataclass defined like
        # @dataclass
        # class Batch(Pipelineable):
        #     uih_features: KeyedJaggedTensor
        #     candidates_features: KeyedJaggedTensor

        # for every attribute in
        # for all the attributes under batch, like batch.uih_features, batch.candidates_features,
        # get the kjt as a dict, and remap the kjt
        # where batch is a dataclass defined like
        # @dataclass
        # class Batch(Pipelineable):
        #     uih_features: KeyedJaggedTensor
        #     candidates_features: KeyedJaggedTensor

        # for every attribute in batch, remap the kjt
        for attr_name, feature_kjt_dict in self.get_batch_kjt_dict(batch).items():
            # separate feature kjt with {feature_name_1: feature_kjt_1, feature_name_2: feature_kjt_2, ...}
            # to multiple dict with {feature_name_1: jt_1}, {feature_name_2: jt_2}, ...
            attr_feature_jt_dict = {}
            for feature_name, feature_jt in feature_kjt_dict.to_dict().items():
                if feature_name not in self.feature_table_name_dict:
                    feature_remapped_jt = JaggedTensor(
                        values=feature_jt.values() % self._input_hash_size,
                        lengths=feature_jt.lengths(),
                    )
                    attr_feature_jt_dict[feature_name] = feature_remapped_jt
                else:
                    feature_name_feature_remapped_jt_dict = self.mod_modules[
                        self.feature_table_name_dict[feature_name]
                    ].remap({feature_name: feature_jt})
                    attr_feature_jt_dict.update(feature_name_feature_remapped_jt_dict)
            feature_kjt_dict = KeyedJaggedTensor.from_jt_dict(
                attr_feature_jt_dict
            )  # {feature_name: feature_kjt}
            setattr(batch, attr_name, feature_kjt_dict)
        return batch
