#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, Iterator, List, Optional, Type

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
)
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.types import ParameterSharding, ShardingEnv, ShardingType
from torchrec.distributed.utils import append_prefix
from torchrec.modules.fused_embedding_modules import (
    convert_optimizer_type_and_kwargs,
    FusedEmbeddingCollection,
)
from torchrec.optim.fused import FusedOptimizerModule


class ShardedFusedEmbeddingCollection(
    ShardedEmbeddingCollection,
    FusedOptimizerModule,
):
    def __init__(
        self,
        module: FusedEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
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
            # pyre-ignore
            module=module,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            fused_params=fused_params,
            env=env,
            device=device,
        )

        for index, (sharding, lookup) in enumerate(
            zip(self._sharding_type_to_sharding.values(), self._lookups)
        ):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                self._lookups[index] = DistributedDataParallel(
                    module=lookup,
                    device_ids=[device],
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False,
                    static_graph=True,
                )
                self._lookups[index]._register_fused_optim(
                    optimizer_type, **optimizer_kwargs
                )
                # TODO - We need a way to get this optimizer back (and add to optims) so it
                # can be checkpointed.
                # We need to ensure that a checkpoint from DDP and a checkpoint from a
                # model parallel version are compatible.

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        # different than ShardedEmbeddingCollection - we consider DDP to be "sharded", so that it doesn't get wrapped up in ddp again
        # semantics of this is actually "parameters that don't need to have their gradients reduced"
        for lookup, _ in zip(self._lookups, self._sharding_type_to_sharding.keys()):
            for name, _ in lookup.named_parameters(append_prefix(prefix, "embeddings")):
                yield name


class FusedEmbeddingCollectionSharder(BaseEmbeddingSharder[FusedEmbeddingCollection]):
    def shard(
        self,
        module: FusedEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedFusedEmbeddingCollection:

        return ShardedFusedEmbeddingCollection(module, params, env, device)

    def shardable_parameters(
        self, module: FusedEmbeddingCollection
    ) -> Dict[str, nn.Parameter]:

        params = {
            # state_dict value looks like model.embeddings.table_0.weights
            name.split(".")[-2]: param
            for name, param in module.state_dict().items()
            if name.endswith(".weight")
        }
        return params

    @property
    def module_type(self) -> Type[FusedEmbeddingCollection]:
        return FusedEmbeddingCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.ROW_WISE.value,
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
