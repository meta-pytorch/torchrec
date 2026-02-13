#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, Dict, List, Optional, Set, Tuple, Type

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollection,
    ShardedQuantEmbeddingCollection,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    ShardedQuantEmbeddingBagCollection,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)


def get_tbes_from_sharded_module(
    module: torch.nn.Module,
) -> List[IntNBitTableBatchedEmbeddingBagsCodegen]:
    assert type(module) in [
        ShardedQuantEmbeddingBagCollection,
        ShardedQuantEmbeddingCollection,
    ], "Only support ShardedQuantEmbeddingBagCollection and ShardedQuantEmbeddingCollection for get TBEs"
    tbes = []
    for lookup in module._lookups:
        for lookup_per_rank in lookup._embedding_lookups_per_rank:
            for emb_module in lookup_per_rank._emb_modules:
                tbes.append(emb_module._emb_module)
    return tbes


def get_tbe_specs_from_sharded_module(
    module: torch.nn.Module,
) -> List[
    Tuple[str, int, int, str, str]
]:  # # tuple of (feature_names, rows, dims, str(SparseType), str(EmbeddingLocation/placement))
    assert type(module) in [
        ShardedQuantEmbeddingBagCollection,
        ShardedQuantEmbeddingCollection,
    ], "Only support ShardedQuantEmbeddingBagCollection and ShardedQuantEmbeddingCollection for get TBE specs"
    tbe_specs = []
    tbes = get_tbes_from_sharded_module(module)
    for tbe in tbes:
        for spec in tbe.embedding_specs:
            tbe_specs.append(
                (
                    spec[0],
                    spec[1],
                    spec[2],
                    str(spec[3]),
                    f"{type(spec[4]).__name__}.{spec[4].name}",
                )
            )
    return tbe_specs


def get_path_device_tuples(
    module: object, ignore_list: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    path_device_tuples: List[Tuple[str, str]] = []
    visited_path: Set[str] = set()

    cur_ignore_list: List[str] = ignore_list if ignore_list else ["embedding_tables"]

    def recursive_find_device(
        module: object, cur_depth: int, path: str = "", max_depth: int = 50
    ) -> None:
        nonlocal path_device_tuples
        nonlocal visited_path

        if cur_depth > max_depth:
            return

        if path in visited_path:
            return

        visited_path.add(path)
        if (
            isinstance(module, (int, float, str, bool, torch.Tensor))
            or type(module).__name__ in ["method", "function", "Proxy"]
            or module is None
        ):
            return

        device_attrs = ("device", "_device", "_device_str", "_device_type")

        for name in dir(module):
            if name in cur_ignore_list:
                continue
            child = getattr(module, name)
            if name.startswith("__"):
                continue
            if name in device_attrs:
                device = getattr(module, name)
                path_device_tuples.append((path + "." + name, str(device)))
            elif isinstance(child, list):
                for idx, child_ in enumerate(child):
                    recursive_find_device(
                        child_,
                        cur_depth + 1,
                        f"{path}.{name}[{idx}]",
                        max_depth=max_depth,
                    )
            elif isinstance(child, dict):
                for key, child_ in child.items():
                    recursive_find_device(
                        child_,
                        cur_depth + 1,
                        f"{path}.{name}[{key}]",
                        max_depth=max_depth,
                    )
            else:
                recursive_find_device(
                    child, cur_depth + 1, f"{path}.{name}", max_depth=max_depth
                )

    recursive_find_device(module, 0, "")

    return path_device_tuples


def get_all_torchrec_modules(
    model: torch.nn.Module,
    trec_module_class_types: Optional[List[Type[torch.nn.Module]]] = None,
) -> Dict[str, torch.nn.Module]:
    """
    Get all targeted TorchRec modules in the model.
    Args:
        model (torch.nn.Module): The input module to search for TREC modules.
        trec_module_class_types (List[Type[torch.nn.Module]], optional): List of type of Trec modules
    """
    if not trec_module_class_types:
        trec_module_class_types = [
            ShardedQuantEmbeddingBagCollection,
            ShardedQuantEmbeddingCollection,
            QuantEmbeddingBagCollection,
            QuantEmbeddingCollection,
            EmbeddingBagCollection,
            EmbeddingCollection,
        ]
    trec_modules: Dict[str, torch.nn.Module] = {}

    def _recursive_get_module(
        module: torch.nn.Module,
        path: str,
        target_module_class: List[Type[torch.nn.Module]],
    ) -> None:
        if type(module) in target_module_class:
            trec_modules[path] = module
            return
        for name, c_module in module.named_children():
            child_path = f"{path}.{name}" if path else name
            if type(c_module) in target_module_class:
                trec_modules[child_path] = c_module
            else:
                _recursive_get_module(c_module, child_path, target_module_class)

    _recursive_get_module(
        model, "", cast(List[Type[torch.nn.Module]], trec_module_class_types)
    )

    return trec_modules


def get_non_scriptable_trec_module(
    model: torch.nn.Module,
) -> Dict[str, torch.nn.Module]:
    """
    Get all targeted TorchRec modules in that model that is not torchsciptable before trace.
    Args:
        model (torch.nn.Module): The input module to search for TREC modules.
    """
    return get_all_torchrec_modules(
        model,
        trec_module_class_types=[
            ShardedQuantEmbeddingBagCollection,
            ShardedQuantEmbeddingCollection,
        ],
    )
