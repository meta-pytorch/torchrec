#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Optional, Union

import torch
from torch import nn
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.quant_embeddingbag import ShardedQuantEmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
)


def populate_fx_names(
    quant_ebc: Union[QuantEmbeddingBagCollection, ShardedQuantEmbeddingBagCollection],
) -> None:
    """
    Assigns fx path to non registered lookup modules. This allows the Torchrec tracer to fallback to
    emb_module._fx_path for table batched embeddings.
    """
    if isinstance(quant_ebc, QuantEmbeddingBagCollection):
        for emb_configs, emb_module in zip(
            quant_ebc._key_to_tables, quant_ebc._emb_modules
        ):
            table_names = []
            for config in emb_configs:
                # pyrefly: ignore[missing-attribute]
                table_names.append(config.name)
            joined_table_names = ",".join(table_names)
            # pyrefly: ignore[bad-argument-type]
            emb_module._fx_path = f"emb_module.{joined_table_names}"
    elif isinstance(quant_ebc, ShardedQuantEmbeddingBagCollection):
        for i, (emb_module, emb_dist_module) in enumerate(
            zip(quant_ebc._lookups, quant_ebc._output_dists)
        ):
            embedding_fx_path = f"embedding_lookup.sharding_{i}"
            # pyrefly: ignore[bad-argument-type]
            emb_module._fx_path = embedding_fx_path
            # pyrefly: ignore[bad-argument-type]
            emb_dist_module._fx_path = f"embedding_dist.{i}"
            #  `Union[Module, Tensor]`.
            # pyrefly: ignore[bad-argument-type]
            for rank, rank_module in enumerate(emb_module._embedding_lookups_per_rank):
                rank_fx_path = f"{embedding_fx_path}.rank_{rank}"
                rank_module._fx_path = rank_fx_path
                for group, group_module in enumerate(rank_module._emb_modules):
                    group_module._fx_path = f"{rank_fx_path}.group_{group}"
                    group_module._emb_module._fx_path = (
                        f"{rank_fx_path}.group_{group}.tbe"
                    )


def recursive_populate_fx_names(module: nn.Module) -> None:
    if isinstance(module, QuantEmbeddingBagCollection) or isinstance(
        module, ShardedQuantEmbeddingBagCollection
    ):
        populate_fx_names(module)
        return
    for submodule in module.children():
        recursive_populate_fx_names(submodule)


def meta_to_cpu_placement(module: torch.nn.Module) -> None:
    if hasattr(module, "_dmp_wrapped_module"):
        # for placement update of dmp module, we need to fetch .module (read access) and write
        # to .dmp_wrapped_module (write access)
        assert type(module) == DistributedModelParallel
        _meta_to_cpu_placement(module.module, module, "_dmp_wrapped_module")
    else:
        # shard module case
        _meta_to_cpu_placement(module, module)


def _meta_to_cpu_placement(
    module: nn.Module, root_module: nn.Module, name: Optional[str] = None
) -> None:
    if (
        name is not None
        and isinstance(module, QuantEmbeddingBagCollection)
        and module.device.type == "meta"
    ):
        qebc_cpu = QuantEmbeddingBagCollection(
            tables=module.embedding_bag_configs(),
            is_weighted=module.is_weighted(),
            device=torch.device("cpu"),
            output_dtype=module.output_dtype(),
            register_tbes=module.register_tbes,
            row_alignment=module.row_alignment,
        )
        setattr(root_module, name, qebc_cpu)
    elif (
        name is not None
        and isinstance(module, QuantEmbeddingCollection)
        and module.device.type == "meta"
    ):
        qec_cpu = QuantEmbeddingCollection(
            tables=module.embedding_configs(),
            device=torch.device("cpu"),
            need_indices=module.need_indices(),
            output_dtype=module.output_dtype(),
            register_tbes=module.register_tbes,
            row_alignment=module.row_alignment,
        )
        setattr(root_module, name, qec_cpu)
    else:
        for name, submodule in module.named_children():
            _meta_to_cpu_placement(submodule, module, name)
