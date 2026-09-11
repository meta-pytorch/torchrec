#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import List, Tuple, Union

import math
import torch
import torch.distributed as dist
from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor

from .distributed import (
    broadcast_ids_to_evict,
    broadcast_transform_result,
    gather_global_ids,
    scatter_cache_ids,
)
from .id_transformer import IDTransformer, TensorList
from .ps import PSCollection


__all__ = []


class IDTransformerCollection:
    def __init__(
        self,
        tables: Union[List[EmbeddingConfig], List[EmbeddingBagConfig]],
        eviction_config=None,
        transform_config=None,
        ps_collection: PSCollection = None,
    ):
        """
        IDTransformerCollection could transform the input of a `Embedding(Bag)Collection`.
        It contains the `IDTransformer` of tables in the
        `Embedding(Bag)Collection`.

        Args:
            tables: list of `Embedding(Bag)Config` or `EmbeddingBagConfig` one passed to
                `Embedding(Bag)Collection`.
            eviction_config: config of the eviction strategy for IDTransformers.
            transform_config: config of the transform strategy for IDTransformers.
            ps_collection: `PSCollection` of the collection, if `None`, won't do eviction or fetch.
                By default, IDTransformerCollection will evict half the ids when full.
        """
        self._configs = tables
        self._ps_collection = ps_collection

        self._transformers = []
        self._table_names = []
        feature_names = set()
        for config in tables:
            if config.name in self._table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            if not config.feature_names:
                config.feature_names = [config.name]
            self._table_names.append(config.name)
            for feature_name in config.feature_names:
                if feature_name in feature_names:
                    raise ValueError(f"Shared feature not allowed yet.")
            # only rank 0 will have the id transformer
            # and other ranks will gather their to rank 0.
            if dist.get_rank() == 0:
                transformer = IDTransformer(
                    num_embedding=config.num_embeddings,
                    eviction_config=eviction_config,
                    transform_config=transform_config,
                )
            else:
                transformer = None
            self._transformers.append(transformer)
        self._feature_names: List[List[str]] = [
            config.feature_names for config in tables
        ]
        self._ever_evicted = False
        self._time = 0

        if dist.get_world_size() > 1:
            self._pg = dist.new_group(backend="gloo")
        self._stream = torch.cuda.Stream()

        self._evict_ratio = 0.5
        if eviction_config is not None and "ratio" in eviction_config:
            self._evict_ratio = float(eviction_config["ratio"])
        
    def _transform(
        self, transformer, global_ids: List[torch.Tensor], cache_ids: List[torch.Tensor]
    ):
        with torch.cuda.stream(self._stream):
            total_numel = sum([tensor.numel() for tensor in global_ids])
            if total_numel > 1e6:
                all_tensor = torch.cat(global_ids).to("cuda:0")
                unique_all_tensor, index = torch.unique(all_tensor, return_inverse=True)
                unique_all_tensor = unique_all_tensor.to("cpu")
                all_cache = torch.empty_like(unique_all_tensor)
                success, ids_to_fetch = transformer.transform(
                    TensorList([unique_all_tensor]),
                    TensorList([all_cache]),
                    self._time,
                )
                del all_tensor
                all_tensor = torch.take(all_cache.to("cuda:0"), index)
                offset = 0
                for tensor in cache_ids:
                    numel = tensor.numel()
                    tensor.copy_(all_tensor[offset : offset + numel])
                    offset += numel
                assert (
                    total_numel == offset
                ), f"total_numel not equal offset, {total_numel} vs {offset}"
            else:
                # broadcast result
                success, ids_to_fetch = transformer.transform(
                    TensorList(global_ids),
                    TensorList(cache_ids),
                    self._time,
                )
            return success, ids_to_fetch

    def transform(
        self, global_features: KeyedJaggedTensor
    ) -> Tuple[KeyedJaggedTensor, List[torch.classes.tde.FetchHandle]]:
        """
        Transform global kjts into local kjts.

        Return:
            KeyedJaggedTensor: the transformed kjt.
            List[torch.classes.tde.FetchHandle]: list of fetch handles to wait.
        """
        global_values = global_features.values()
        cache_values = torch.empty_like(global_values)

        global_feature_indices = {
            feature_name: i for i, feature_name in enumerate(global_features.keys())
        }
        offset_per_key = global_features.offset_per_key()

        fetch_handles = []
        for i, transformer in enumerate(self._transformers):
            feature_names = self._feature_names[i]
            feature_indices = [
                global_feature_indices[feature_name] for feature_name in feature_names
            ]
            global_ids = [
                global_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                for idx in feature_indices
            ]
            cache_ids = [
                cache_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                for idx in feature_indices
            ]

            if dist.get_world_size() > 1:
                concat_global_ids, concat_numel_list = gather_global_ids(
                    global_ids, self._pg
                )
                if dist.get_rank() == 0:
                    global_ids = global_ids + concat_global_ids[1:]
                    cache_ids = cache_ids + [
                        torch.empty_like(tensor) for tensor in concat_global_ids[1:]
                    ]

                    success, ids_to_fetch = self._transform(
                        transformer, global_ids, cache_ids
                    )
                else:
                    success, ids_to_fetch = True, None
                success, ids_to_fetch = broadcast_transform_result(
                    success, ids_to_fetch, self._pg
                )

                if self._ps_collection is not None:
                    table_name = self._table_names[i]
                    ps = self._ps_collection[table_name]
                    if ids_to_fetch.numel() > 0:
                        handle = ps.fetch(
                            ids_to_fetch,
                            self._time,
                            self._ever_evicted,
                            self._configs[i].get_weight_init_min(),
                            self._configs[i].get_weight_init_max(),
                        )
                        fetch_handles.append(handle)
                    if not success:
                        # TODO(zilinzhu): make this configurable
                        # broadcast ids_to_evict
                        if dist.get_rank() == 0:
                            ids_to_evict = transformer.evict(
                                math.floor(transformer._num_embedding * self._evict_ratio)
                            )
                        else:
                            ids_to_evict = None
                        ids_to_evict = broadcast_ids_to_evict(ids_to_evict, self._pg)

                        ps.evict(ids_to_evict)
                        self._ever_evicted = True

                        # retry after eviction.
                        # broadcast result
                        if dist.get_rank() == 0:
                            success, ids_to_fetch = transformer.transform(
                                TensorList(global_ids),
                                TensorList(cache_ids),
                                self._time,
                            )
                        else:
                            success, ids_to_fetch = True, None
                        success, ids_to_fetch = broadcast_transform_result(
                            success, ids_to_fetch, self._pg
                        )

                        if not success:
                            raise RuntimeError(
                                "Failed to transform global ids after eviction. "
                                f"Maybe the num_embedding of table {table_name} is too small?"
                            )
                        if ids_to_fetch.numel() > 0:
                            fetch_handles.append(
                                ps.fetch(
                                    ids_to_fetch,
                                    self._time,
                                    self._ever_evicted,
                                    self._configs[i].get_weight_init_min(),
                                    self._configs[i].get_weight_init_max(),
                                )
                            )

                scatter_cache_ids(cache_ids, concat_numel_list, self._pg)
            else:
                success, ids_to_fetch = self._transform(
                    transformer, global_ids, cache_ids
                )
                if self._ps_collection is not None:
                    table_name = self._table_names[i]
                    ps = self._ps_collection[table_name]
                    if ids_to_fetch.numel() > 0:
                        handle = ps.fetch(
                            ids_to_fetch,
                            self._time,
                            self._ever_evicted,
                            self._configs[i].get_weight_init_min(),
                            self._configs[i].get_weight_init_max(),
                        )
                        fetch_handles.append(handle)
                    if not success:
                        ids_to_evict = transformer.evict(
                            math.floor(transformer._num_embedding * self._evict_ratio)
                        )
                        ps.evict(ids_to_evict)
                        self._ever_evicted = True

                        # retry after eviction.
                        success, ids_to_fetch = transformer.transform(
                            TensorList(global_ids),
                            TensorList(cache_ids),
                            self._time,
                        )
                        if not success:
                            raise RuntimeError(
                                "Failed to transform global ids after eviction. "
                                f"Maybe the num_embedding of table {table_name} is too small?"
                            )
                        if ids_to_fetch is not None:
                            fetch_handles.append(
                                ps.fetch(
                                    ids_to_fetch,
                                    self._time,
                                    self._ever_evicted,
                                    self._configs[i].get_weight_init_min(),
                                    self._configs[i].get_weight_init_max(),
                                )
                            )

        cache_values = KeyedJaggedTensor(
            keys=global_features.keys(),
            values=cache_values,
            lengths=global_features.lengths(),
            weights=global_features.weights_or_none(),
        )
        self._time += 1
        return cache_values, fetch_handles

    def save(self):
        if self._ps_collection is None:
            return
        for i, transformer in enumerate(self._transformers):
            table_name = self._table_names[i]
            if dist.get_world_size() > 1:
                if dist.get_rank() == 0:
                    ids = transformer.save()
                    numel = torch.tensor(ids.numel())
                    dist.broadcast(numel, src=0, group=self._pg)
                    dist.broadcast(ids, src=0, group=self._pg)
                else:
                    numel = torch.tensor(0)
                    dist.broadcast(numel, src=0, group=self._pg)
                    ids = torch.empty((numel // 2, 2), dtype=torch.int64)
                    dist.broadcast(ids, src=0, group=self._pg)
            else:
                ids = transformer.save()

            self._ps_collection[table_name].evict(ids)
