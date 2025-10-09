#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    pooling_type_to_str,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt


@torch.fx.wrap
def reorder_inverse_indices(
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]],
    feature_names: List[str],
) -> torch.Tensor:
    if inverse_indices is None:
        return torch.empty(0)
    index_per_name = {name: i for i, name in enumerate(inverse_indices[0])}
    index = torch.tensor(
        [index_per_name[name.split("@")[0]] for name in feature_names],
        device=inverse_indices[1].device,
    )
    return torch.index_select(inverse_indices[1], 0, index)


@torch.fx.wrap
def process_pooled_embeddings(
    pooled_embeddings: List[torch.Tensor],
    inverse_indices: torch.Tensor,
) -> torch.Tensor:
    if inverse_indices.numel() > 0:
        pooled_embeddings = torch.ops.fbgemm.group_index_select_dim0(
            pooled_embeddings, list(torch.unbind(inverse_indices))
        )
    if not pooled_embeddings:
        # Return a tensor with shape [batch_size, 0] if pooled_embeddings is empty list
        batch_size = inverse_indices.shape[0]
        return torch.zeros((batch_size, 0), device=inverse_indices.device)
    return torch.cat(pooled_embeddings, dim=1)


class EmbeddingBagCollectionInterface(abc.ABC, nn.Module):
    """
    Interface for `EmbeddingBagCollection`.
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        pass

    @abc.abstractmethod
    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        pass

    @abc.abstractmethod
    def is_weighted(self) -> bool:
        pass


def get_embedding_names_by_table(
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
) -> List[List[str]]:
    shared_feature: Dict[str, bool] = {}
    for embedding_config in tables:
        for feature_name in embedding_config.feature_names:
            if feature_name not in shared_feature:
                shared_feature[feature_name] = False
            else:
                shared_feature[feature_name] = True
    embedding_names_by_table: List[List[str]] = []
    for embedding_config in tables:
        embedding_names: List[str] = []
        for feature_name in embedding_config.feature_names:
            if shared_feature[feature_name]:
                embedding_names.append(feature_name + "@" + embedding_config.name)
            else:
                embedding_names.append(feature_name)
        embedding_names_by_table.append(embedding_names)
    return embedding_names_by_table


class EmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).

    NOTE:
        EmbeddingBagCollection is an unsharded module and is not performance optimized.
        For performance-sensitive scenarios, consider using the sharded version ShardedEmbeddingBagCollection.


    It is callable on arguments representing sparse data in the form of `KeyedJaggedTensor` with values of the shape
    `(F, B, L[f][i])` where:

    * `F`: number of features (keys)
    * `B`: batch size
    * `L[f][i]`: length of sparse features (potentially distinct for each feature `f` and batch index `i`, that is, jagged)

    and outputs a `KeyedTensor` with values with shape `(B, D)` where:

    * `B`: batch size
    * `D`: sum of embedding dimensions of all embedding tables, that is, `sum([config.embedding_dim for config in tables])`

    Assuming the argument is a `KeyedJaggedTensor` `J` with `F` features, batch size `B` and `L[f][i]` sparse lengths
    such that `J[f][i]` is the bag for feature `f` and batch index `i`, the output `KeyedTensor` `KT` is defined as follows:
    `KT[i]` = `torch.cat([emb[f](J[f][i]) for f in J.keys()])` where `emb[f]` is the `EmbeddingBag` corresponding to the feature `f`.

    Note that `J[f][i]` is a variable-length list of integer values (a bag), and `emb[f](J[f][i])` is pooled embedding
    produced by reducing the embeddings of each of the values in `J[f][i]`
    using the `EmbeddingBag` `emb[f]`'s mode (default is the mean).

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (Optional[torch.device]): default compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[table_0, table_1])

        #        i = 0     i = 1    i = 2  <-- batch indices
        # "f1"   [0,1]     None      [2]
        # "f2"   [3]       [4]     [5,6,7]
        #  ^
        # features

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1,                  2,    # feature 'f1'
                                    3,      4,    5, 6, 7]),  # feature 'f2'
                            #    i = 1    i = 2    i = 3   <--- batch indices
            offsets=torch.tensor([
                    0, 2, 2,       # 'f1' bags are values[0:2], values[2:2], and values[2:3]
                    3, 4, 5, 8]),  # 'f2' bags are values[3:4], values[4:5], and values[5:8]
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([
            #  f1 pooled embeddings              f2 pooled embeddings
            #     from bags (dim. 3)                from bags (dim. 4)
            [-0.8899, -0.1342, -1.9060,  -0.0905, -0.2814, -0.9369, -0.7783],  # i = 0
            [ 0.0000,  0.0000,  0.0000,   0.1598,  0.0695,  1.3265, -0.1011],  # i = 1
            [-0.4256, -1.1846, -2.1648,  -1.0893,  0.3590, -1.9784, -0.7681]],  # i = 2
            grad_fn=<CatBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])  # embeddings have dimensions 3 and 4, so embeddings are at [0, 3) and [3, 7).
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._is_weighted = is_weighted
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        self._embedding_bag_configs = tables
        self._lengths_per_embedding: List[int] = []

        table_names = set()
        for embedding_config in tables:
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            dtype = (
                torch.float32
                if embedding_config.data_type == DataType.FP32
                else torch.float16
            )
            self.embedding_bags[embedding_config.name] = nn.EmbeddingBag(
                num_embeddings=embedding_config.num_embeddings,
                embedding_dim=embedding_config.embedding_dim,
                mode=pooling_type_to_str(embedding_config.pooling),
                device=device,
                include_last_offset=True,
                dtype=dtype,
            )
            if device is None:
                device = self.embedding_bags[embedding_config.name].weight.device

            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            self._lengths_per_embedding.extend(
                len(embedding_config.feature_names) * [embedding_config.embedding_dim]
            )

        # pyre-fixme[8]: Attribute has type `device`; used as `Union[Module, device,
        #  Tensor]`.
        self._device: torch.device = device or torch.device("cpu")
        self._embedding_names: List[str] = [
            embedding
            for embeddings in get_embedding_names_by_table(tables)
            for embedding in embeddings
        ]
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]
        self.reset_parameters()

    def forward(
        self,
        features: KeyedJaggedTensor,  # can also take TensorDict as input
    ) -> KeyedTensor:
        """
        Run the EmbeddingBagCollection forward pass. This method takes in a `KeyedJaggedTensor`
        and returns a `KeyedTensor`, which is the result of pooling the embeddings for each feature.

        Args:
            features (KeyedJaggedTensor): Input KJT
        Returns:
            KeyedTensor
        """
        flat_feature_names: List[str] = []
        features = maybe_td_to_kjt(features, None)
        for names in self._feature_names:
            flat_feature_names.extend(names)
        inverse_indices = reorder_inverse_indices(
            inverse_indices=features.inverse_indices_or_none(),
            feature_names=flat_feature_names,
        )
        pooled_embeddings: List[torch.Tensor] = []
        feature_dict = features.to_dict()
        for i, embedding_bag in enumerate(self.embedding_bags.values()):
            for feature_name in self._feature_names[i]:
                f = feature_dict[feature_name]
                res = embedding_bag(
                    input=f.values(),
                    offsets=f.offsets(),
                    per_sample_weights=f.weights() if self._is_weighted else None,
                ).float()
                pooled_embeddings.append(res)
        return KeyedTensor(
            keys=self._embedding_names,
            values=process_pooled_embeddings(
                pooled_embeddings=pooled_embeddings,
                inverse_indices=inverse_indices,
            ),
            length_per_key=self._lengths_per_embedding,
        )

    def is_weighted(self) -> bool:
        """
        Returns:
            bool: Whether the EmbeddingBagCollection is weighted.
        """
        return self._is_weighted

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        """
        Returns:
            List[EmbeddingBagConfig]: The embedding bag configs.
        """
        return self._embedding_bag_configs

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: The compute device.
        """
        return self._device

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the EmbeddingBagCollection. Parameter values
        are intiialized based on the `init_fn` of each EmbeddingBagConfig if it exists.
        """
        if (isinstance(self.device, torch.device) and self.device.type == "meta") or (
            isinstance(self.device, str) and self.device == "meta"
        ):
            return
        # Initialize embedding bags weights with init_fn
        for table_config in self._embedding_bag_configs:
            assert table_config.init_fn is not None
            param = self.embedding_bags[f"{table_config.name}"].weight
            # pyre-ignore
            table_config.init_fn(param)


class EmbeddingCollectionInterface(abc.ABC, nn.Module):
    """
    Interface for `EmbeddingCollection`.
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        pass

    @abc.abstractmethod
    def embedding_configs(
        self,
    ) -> List[EmbeddingConfig]:
        pass

    @abc.abstractmethod
    def need_indices(self) -> bool:
        pass

    @abc.abstractmethod
    def embedding_dim(self) -> int:
        pass

    @abc.abstractmethod
    def embedding_names_by_table(self) -> List[List[str]]:
        pass


class EmbeddingCollection(EmbeddingCollectionInterface):
    """
    EmbeddingCollection represents a collection of non-pooled embeddings.

    NOTE:
        EmbeddingCollection is an unsharded module and is not performance optimized.
        For performance-sensitive scenarios, consider using the sharded version ShardedEmbeddingCollection.

    It is callable on arguments representing sparse data in the form of `KeyedJaggedTensor` with values of the shape
    `(F, B, L[f][i])` where:

    * `F`: number of features (keys)
    * `B`: batch size
    * `L[f][i]`: length of sparse features (potentially distinct for each feature `f` and batch index `i`, that is, jagged)

    and outputs a `result` of type `Dict[Feature, JaggedTensor]`,
    where `result[f]` is a `JaggedTensor` with shape `(EB[f], D[f])` where:

    * `EB[f]`: a "expanded batch size" for feature `f` equal to the sum of the lengths of its bag values,
      that is, `sum([len(J[f][i]) for i in range(B)])`.
    * `D[f]`: is the embedding dimension of feature `f`.

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables.
        device (Optional[torch.device]): default compute device.
        need_indices (bool): if we need to pass indices to the final lookup dict.

    Example::

        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1,                  2,    # feature 'f1'
                                    3,      4,    5, 6, 7]),  # feature 'f2'
                            #    i = 1    i = 2    i = 3   <--- batch indices
            offsets=torch.tensor([
                    0, 2, 2,       # 'f1' bags are values[0:2], values[2:2], and values[2:3]
                    3, 4, 5, 8]),  # 'f2' bags are values[3:4], values[4:5], and values[5:8]
        )

        feature_embeddings = ec(features)
        print(feature_embeddings['f2'].values())
        tensor([
            # embedding for value 3 in f2 bag values[3:4]:
            [-0.2050,  0.5478,  0.6054],

            # embedding for value 4 in f2 bag values[4:5]:
            [ 0.7352,  0.3210, -3.0399],

            # embedding for values 5, 6, 7 in f2 bag values[5:8]:
            [ 0.1279, -0.1756, -0.4130],
            [ 0.7519, -0.4341, -0.0499],
            [ 0.9329, -1.0697, -0.8095],

        ], grad_fn=<EmbeddingBackward>)
    """

    def __init__(  # noqa C901
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        self._embedding_configs = tables
        self._embedding_dim: int = -1
        self._need_indices: bool = need_indices
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )

        table_names = set()
        for config in tables:
            if config.name in table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            table_names.add(config.name)
            self._embedding_dim = (
                config.embedding_dim if self._embedding_dim < 0 else self._embedding_dim
            )
            if self._embedding_dim != config.embedding_dim:
                raise ValueError(
                    "All tables in a EmbeddingCollection are required to have same embedding dimension."
                    + f" Violating case: {config.name}'s embedding_dim {config.embedding_dim} !="
                    + f" {self._embedding_dim}"
                )
            dtype = (
                torch.float32 if config.data_type == DataType.FP32 else torch.float16
            )
            self.embeddings[config.name] = nn.Embedding(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
                device=device,
                dtype=dtype,
            )
            if config.init_fn is not None:
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Module, Tensor]`.
                config.init_fn(self.embeddings[config.name].weight)

            if not config.feature_names:
                config.feature_names = [config.name]

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            tables
        )
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]

    def forward(
        self,
        features: KeyedJaggedTensor,  # can also take TensorDict as input
    ) -> Dict[str, JaggedTensor]:
        """
        Run the EmbeddingBagCollection forward pass. This method takes in a `KeyedJaggedTensor`
        and returns a `Dict[str, JaggedTensor]`, which is the result of the individual embeddings for each feature.

        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """

        features = maybe_td_to_kjt(features, None)
        feature_embeddings: Dict[str, JaggedTensor] = {}
        jt_dict: Dict[str, JaggedTensor] = features.to_dict()
        for i, emb_module in enumerate(self.embeddings.values()):
            feature_names = self._feature_names[i]
            embedding_names = self._embedding_names_by_table[i]
            for j, embedding_name in enumerate(embedding_names):
                feature_name = feature_names[j]
                f = jt_dict[feature_name]
                lookup = emb_module(
                    input=f.values(),
                ).float()
                feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup,
                    lengths=f.lengths(),
                    weights=f.values() if self._need_indices else None,
                )
        return feature_embeddings

    def need_indices(self) -> bool:
        """
        Returns:
            bool: Whether the EmbeddingCollection needs indices.
        """
        return self._need_indices

    def embedding_dim(self) -> int:
        """
        Returns:
            int: The embedding dimension.
        """
        return self._embedding_dim

    def embedding_configs(self) -> List[EmbeddingConfig]:
        """
        Returns:
            List[EmbeddingConfig]: The embedding configs.
        """
        return self._embedding_configs

    def embedding_names_by_table(self) -> List[List[str]]:
        """
        Returns:
            List[List[str]]: The embedding names by table.
        """
        return self._embedding_names_by_table

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: The compute device.
        """
        return self._device

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the EmbeddingCollection. Parameter values
        are intiialized based on the `init_fn` of each EmbeddingConfig if it exists.
        """

        if (isinstance(self.device, torch.device) and self.device.type == "meta") or (
            isinstance(self.device, str) and self.device == "meta"
        ):
            return
        # Initialize embedding bags weights with init_fn
        for table_config in self._embedding_configs:
            assert table_config.init_fn is not None
            param = self.embeddings[f"{table_config.name}"].weight
            # pyre-ignore
            table_config.init_fn(param)
