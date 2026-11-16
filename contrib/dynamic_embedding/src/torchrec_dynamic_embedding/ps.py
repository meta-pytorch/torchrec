#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.types import ParameterSharding

from .tensor_list import TensorList

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = ["PS", "PSCollection"]


DEFAULT_PS_CHUNK_SIZE = 8 * 1024 * 1024


class PS:
    def __init__(
        self,
        table_name: str,
        tensors: Union[List[torch.Tensor], List[ShardedTensor]],
        url: str,
        chunk_size: str,
    ):
        """
        PS table of an embedding table.

        Args:
            table_name: name of the table.
            tensors: tensors of the table, the first one is the parameter tensor, others are
                tenors of optimizers, e.g. for Adam, it will be [weight, m, v].
            url: url of the PS.
        """
        shards = torch.classes.tde.LocalShardList()
        num_optimizer_stats = len(tensors)
        if isinstance(tensors[0], ShardedTensor):
            # Here we assume the shard metadata of optimizer state and weight are the same.
            for i, shard in enumerate(tensors[0].local_shards()):
                local_tensors = [tensor.local_shards()[i].tensor for tensor in tensors]
                shards.append(
                    shard.metadata.shard_offsets[0],
                    shard.metadata.shard_offsets[1],
                    shard.metadata.shard_sizes[0],
                    shard.metadata.shard_sizes[1],
                    TensorList(local_tensors).tensor_list,
                )
                # This assumes all shard have the same column size.
                col_size = shard.tensor.shape[1]
        elif isinstance(tensors[0], torch.Tensor):
            shards.append(
                0,
                0,
                tensors[0].shape[0],
                tensors[0].shape[1],
                TensorList(tensors).tensor_list,
            )
            col_size = tensors[0].shape[1]
        self._ps = torch.classes.tde.PS(
            table_name, shards, col_size, num_optimizer_stats, url, chunk_size
        )

    def evict(self, ids_to_evict: torch.Tensor):
        """
        Evict the `ids_to_evict` to PS.
        """
        self._ps.evict(ids_to_evict)

    def fetch(
        self,
        ids_to_fetch: torch.Tensor,
        time: int,
        reinit: bool = False,
        weight_init_max: float = 0,
        weight_init_min: float = 0,
    ):
        """
        Fetch `ids_to_fetch` from tensor. If `reinit` is set to `True`, will
        reinitialize the embedding if the global id is not in PS.
        """
        return self._ps.fetch(
            ids_to_fetch, time, reinit, weight_init_max, weight_init_min
        )


class PSCollection:
    """
    PS tables correspond to an EmbeddingCollection or EmbeddingBagCollection.
    """

    def __init__(
        self,
        path: str,
        plan: Dict[str, Tuple[ParameterSharding, Union[torch.Tensor, ShardedTensor]]],
        url: Union[str, Callable[[str], str]],
        ps_config=None,
    ):
        """
        Args:
            path: module path.
            plan: dict keyed by table name of ParameterSharding and tensor of the table.
            url: configuration for PS, e.g. redis://127.0.0.1:6379/?prefix=model.
            ps_config: config of the PS, currently only support setting chunk size.
        """
        self._path = path
        self._ps_collection = {}
        chunk_size = DEFAULT_PS_CHUNK_SIZE
        if ps_config is not None and "chunk_size" in ps_config:
            chunk_size = ps_config["chunk_size"]
        for table_name, (param_plan, tensor) in plan.items():
            if isinstance(url, str):
                table_config = url
            else:
                table_config = url(table_name)
            self._ps_collection[table_name] = PS(
                f"{path}.{table_name}", tensor, table_config, chunk_size
            )

    def table_names(self):
        return self._ps_collection.keys()

    def __getitem__(self, table_name):
        return self._ps_collection[table_name]

    @staticmethod
    def fromModule(path, sharded_module, params_plan, url, ps_config=None):
        """
        Create PSCollection for `sharded_module`, whose module path is `path`

        Args:
            path: module path of the sharded module.
            sharded_module: the sharded module.
            params_plan: the sharding plan of `sharded_module`.
            url: configuration for PS, e.g. redis://127.0.0.1:6379/?prefix=model.
            ps_config: config of the PS, currently only support setting chunk size.

        Return:
            PSCollection of the sharded module.
        """

        state_dict = sharded_module.state_dict()
        optimizer_state_dict = sharded_module.fused_optimizer.state_dict()["state"]
        tensor_infos = {}
        for key, tensor in state_dict.items():
            # Here we use the fact that state_dict will be shape of
            # `embeddings.xxx.weight` or `embeddingbags.xxx.weight`
            if len(key.split(".")) <= 1 or key.split(".")[1] not in params_plan:
                continue
            table_name = key.split(".")[1]
            param_plan = params_plan.pop(table_name)
            tensors = [tensor]
            optimizer_state = optimizer_state_dict[key]
            tensors.extend(optimizer_state.values())
            tensor_infos[table_name] = (param_plan, tensors)

        assert (
            len(params_plan) == 0
        ), f"There are sharded param not found, leaving: {params_plan}."

        if isinstance(url, str):
            collection_schema = url
        else:
            collection_schema = lambda table_name: url(path, table_name)

        return PSCollection(path, tensor_infos, collection_schema, ps_config)
