#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Type

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
)
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.fused_embedding_modules import (
    convert_optimizer_type_and_kwargs,
    FusedEmbeddingBagCollection,
)


class ShardedFusedEmbeddingBagCollection(
    ShardedEmbeddingBagCollection,
):
    def __init__(
        self,
        module: FusedEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        optimizer_type = module.optimizer_type()
        optimizer_kwargs = module.optimizer_kwargs()

        fused_params = {}
        emb_opt_type_and_kwargs = convert_optimizer_type_and_kwargs(
            optimizer_type, optimizer_kwargs
        )
        assert emb_opt_type_and_kwargs is not None
        (emb_optim_type, emb_opt_kwargs) = emb_opt_type_and_kwargs

        fused_params["optimizer"] = emb_optim_type
        fused_params.update(emb_opt_kwargs)

        super().__init__(
            module=module,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            fused_params=fused_params,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

        for index, (sharding, lookup) in enumerate(
            zip(self._embedding_shardings, self._lookups)
        ):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                self._lookups[index] = DistributedDataParallel(
                    module=lookup,
                    device_ids=[device] if device is not None else None,
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False,
                    static_graph=True,
                )
                # pyrefly: ignore[not-callable]
                self._lookups[index]._register_fused_optim(
                    optimizer_type, **optimizer_kwargs
                )
                # TODO - We need a way to get this optimizer back (and add to optims) so it
                # can be checkpointed.
                # We need to ensure that a checkpoint from DDP and a checkpoint from a
                # model parallel version are compatible.


class FusedEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[FusedEmbeddingBagCollection]
):
    def shard(
        self,
        module: FusedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingBagCollection:

        return ShardedFusedEmbeddingBagCollection(
            module,
            params,
            env,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )

    def shardable_parameters(
        self, module: FusedEmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:

        params = {
            name.split(".")[-2]: param
            for name, param in module.state_dict().items()
            if name.endswith(".weight")
        }
        return params

    @property
    def module_type(self) -> Type[FusedEmbeddingBagCollection]:
        return FusedEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]
        if compute_device_type in {"cuda"}:
            types += [
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]

        return types

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        ret = []
        if sharding_type != ShardingType.DATA_PARALLEL.value:
            ret += [
                EmbeddingComputeKernel.FUSED.value,
            ]
            if compute_device_type in {"cuda"}:
                ret += [
                    EmbeddingComputeKernel.FUSED_UVM.value,
                    EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                ]
        else:
            ret.append(EmbeddingComputeKernel.DENSE.value)
        return ret
